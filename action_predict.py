# -*- coding: utf-8 -*-
import os
import time
import tarfile
import pickle
import yaml
import json
import wget
import cv2
import scipy.misc
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

from utils import *
from custom_callbacks import EpochSaveCallback, DetailedLoggingCallback, MetricsVisualizationCallback
from base_models import AlexNet, C3DNet, convert_to_fcn, C3DNet2
from base_models import I3DNet

from tensorflow.keras.layers import (Input, Concatenate, Dense, GRU, LSTM, GRUCell,
                                     Dropout, LSTMCell, RNN, Flatten, Average, Add,
                                     ConvLSTM2D, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Lambda, dot, concatenate, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model, Sequence, register_keras_serializable
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import gelu

try:
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
except ImportError:
    # TensorFlow 2.6及以下版本使用这个路径
    from tensorflow.keras.layers.experimental import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import BinaryCrossentropy, Loss, MeanSquaredError


class BinaryFocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, reduction=tf.keras.losses.Reduction.AUTO,
                 name='binary_focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Focal Loss 公式：FL(p_t) = - alpha_t * (1 - p_t)^{gamma} * log(p_t)
        # 其中 p_t 表示针对真实标签的预测概率，alpha_t 控制正负样本权重，gamma 调节难易样本的抑制强度。
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -(y_true * tf.math.log(y_pred) +
                          (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)
        modulating_factor = tf.math.pow(1.0 - p_t, self.gamma)
        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_mean(loss, axis=-1)


class FocalL2Loss(Loss):
    def __init__(self, gamma=1.5, epsilon=1e-6, reduction=tf.keras.losses.Reduction.AUTO,
                 name='focal_l2_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        error = y_true - y_pred
        squared = tf.square(error)
        modulating = tf.pow(tf.abs(error) + self.epsilon, self.gamma)
        loss = modulating * squared
        return tf.reduce_mean(loss, axis=-1)


class UncertaintyWeightedModel(tf.keras.Model):
    """
    Keras model wrapper that applies the uncertainty-weighted multi-task loss from PedCMT:
        L = L_cls / sigma_cls^2 + L_reg / sigma_reg^2 + log(sigma_cls) + log(sigma_reg)
    where sigma_{cls,reg} are learnable scalars.
    """

    def __init__(self, inputs, outputs, cls_loss=None, reg_loss=None, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.cls_loss_fn = cls_loss or BinaryFocalLoss()
        # Temporarily disable focal regression loss; fall back to MSE
        # self.reg_loss_fn = reg_loss or MeanSquaredError()
        self.reg_loss_fn = reg_loss or FocalL2Loss()

        # 不确定性加权多任务损失：
        # L = L_cls / sigma_cls^2 + L_reg / sigma_reg^2 + log(sigma_cls) + log(sigma_reg)
        # sigma_cls 与 sigma_reg 为可学习标量，数值越大代表对应任务的不确定性越高，模型会自动调节两个任务的相对权重。
        he_init = tf.keras.initializers.HeNormal()
        self.sigma_cls = self.add_weight(
            name='sigma_cls',
            shape=(1, 1),
            initializer=he_init,
            trainable=True)
        self.sigma_reg = self.add_weight(
            name='sigma_reg',
            shape=(1, 1),
            initializer=he_init,
            trainable=True)

        # trackers so fit() can report loss components just like standard compile
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.cls_loss_tracker = tf.keras.metrics.Mean(name='cls_loss')
        self.reg_loss_tracker = tf.keras.metrics.Mean(name='reg_loss')

    def _split_targets(self, y):
        if isinstance(y, dict):
            return y.get('intention'), y.get('etraj')
        if isinstance(y, (list, tuple)):
            if len(y) >= 2:
                return y[0], y[1]
            if len(y) == 1:
                return y[0], None
        return y, None

    def _split_sample_weights(self, sample_weight):
        if isinstance(sample_weight, dict):
            return sample_weight.get('intention'), sample_weight.get('etraj')
        if isinstance(sample_weight, (list, tuple)):
            if len(sample_weight) >= 2:
                return sample_weight[0], sample_weight[1]
            if len(sample_weight) == 1:
                return sample_weight[0], None
        return sample_weight, None

    def _sigma_value(self, var):
        sigma = tf.math.abs(var) + 1e-6
        return tf.reshape(sigma, [])

    def get_sigma_values(self):
        return self._sigma_value(self.sigma_cls), self._sigma_value(self.sigma_reg)

    def _compute_combined_loss(self, cls_loss, reg_loss, has_reg_loss):
        sigma_cls, sigma_reg = self.get_sigma_values()
        total_loss = cls_loss / tf.square(sigma_cls) + tf.math.log(sigma_cls)

        if has_reg_loss:
            total_loss += reg_loss / tf.square(sigma_reg) + tf.math.log(sigma_reg)
        return total_loss
        # return cls_loss

    @property
    def metrics(self):
        base = [self.loss_tracker, self.cls_loss_tracker, self.reg_loss_tracker]
        if hasattr(self, 'compiled_metrics'):
            return base + self.compiled_metrics.metrics
        return base

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_int_true, y_traj_true = self._split_targets(y)
        sw_int, sw_traj = self._split_sample_weights(sample_weight)

        with tf.GradientTape() as tape:
            y_int_pred, y_traj_pred = self(x, training=True)
            cls_loss = self.cls_loss_fn(y_int_true, y_int_pred, sample_weight=sw_int)

            has_reg_target = y_traj_true is not None
            if has_reg_target:
                reg_loss = self.reg_loss_fn(y_traj_true, y_traj_pred, sample_weight=sw_traj)
            else:
                reg_loss = tf.constant(0.0, dtype=cls_loss.dtype)

            combined_loss = self._compute_combined_loss(cls_loss, reg_loss, has_reg_target)
            reg_losses = tf.add_n(self.losses) if self.losses else 0.0
            total_loss = combined_loss + reg_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.reg_loss_tracker.update_state(reg_loss)

        metrics_y_true = [
            y_int_true,
            y_traj_true if y_traj_true is not None else y_traj_pred
        ]
        metrics_y_pred = [y_int_pred, y_traj_pred]
        metrics_sample_weight = None
        if sw_int is not None or sw_traj is not None:
            metrics_sample_weight = [sw_int, sw_traj]
        self.compiled_metrics.update_state(metrics_y_true, metrics_y_pred, metrics_sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_int_true, y_traj_true = self._split_targets(y)
        sw_int, sw_traj = self._split_sample_weights(sample_weight)

        y_int_pred, y_traj_pred = self(x, training=False)
        cls_loss = self.cls_loss_fn(y_int_true, y_int_pred, sample_weight=sw_int)
        has_reg_target = y_traj_true is not None
        if has_reg_target:
            reg_loss = self.reg_loss_fn(y_traj_true, y_traj_pred, sample_weight=sw_traj)
        else:
            reg_loss = tf.constant(0.0, dtype=cls_loss.dtype)

        combined_loss = self._compute_combined_loss(cls_loss, reg_loss, has_reg_target)
        reg_losses = tf.add_n(self.losses) if self.losses else 0.0
        total_loss = combined_loss + reg_losses

        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.reg_loss_tracker.update_state(reg_loss)

        metrics_y_true = [
            y_int_true,
            y_traj_true if y_traj_true is not None else y_traj_pred
        ]
        metrics_y_pred = [y_int_pred, y_traj_pred]
        metrics_sample_weight = None
        if sw_int is not None or sw_traj is not None:
            metrics_sample_weight = [sw_int, sw_traj]
        self.compiled_metrics.update_state(metrics_y_true, metrics_y_pred, metrics_sample_weight)

        return {m.name: m.result() for m in self.metrics}


tf.keras.utils.get_custom_objects()['UncertaintyWeightedModel'] = UncertaintyWeightedModel
tf.keras.utils.get_custom_objects()['BinaryFocalLoss'] = BinaryFocalLoss
tf.keras.utils.get_custom_objects()['FocalL2Loss'] = FocalL2Loss


def _has_intention_and_etraj(model):
    try:
        return {'intention', 'etraj'}.issubset(set(model.output_names))
    except AttributeError:
        return False


def maybe_apply_uncertainty_wrapper(model):
    if isinstance(model, UncertaintyWeightedModel):
        return model
    if _has_intention_and_etraj(model):
        return UncertaintyWeightedModel(inputs=model.inputs, outputs=model.outputs, name=model.name)
    return model


class SigmaLoggingCallback(tf.keras.callbacks.Callback):
    """在 Keras 默认日志输出中添加 sigma 指标"""

    def _maybe_add_metric(self, logs, name, value):
        if value is None:
            return
        logs[name] = value
        logs[f'val_{name}'] = value

    def on_train_begin(self, logs=None):
        metrics = self.params.get('metrics') if isinstance(self.params, dict) else None
        if isinstance(metrics, list):
            for name in ['sigma_cls', 'sigma_reg', 'val_sigma_cls', 'val_sigma_reg']:
                if name not in metrics:
                    metrics.append(name)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        model = self.model
        sigma_cls = None
        sigma_reg = None

        getter = getattr(model, 'get_sigma_values', None)
        if getter is not None:
            sigma_cls_tensor, sigma_reg_tensor = getter()
            sigma_cls = float(sigma_cls_tensor.numpy())
            sigma_reg = float(sigma_reg_tensor.numpy())

        self._maybe_add_metric(logs, 'sigma_cls', sigma_cls)
        self._maybe_add_metric(logs, 'sigma_reg', sigma_reg)

        if sigma_cls is not None or sigma_reg is not None:
            parts = []
            if sigma_cls is not None:
                parts.append(f"sigma_cls={sigma_cls:.4f}")
            if sigma_reg is not None:
                parts.append(f"sigma_reg={sigma_reg:.4f}")
            if parts:
                print(f"[Sigma] Epoch {epoch + 1}: {' '.join(parts)}")


# ================================
# Base Class
# ================================
class ActionPredict(object):
    """
    A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None # use data generator for train/test 


    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts, 
                       'train_opts': {'batch_size':batch_size, 'epochs': epochs, 'lr': lr}},
                       fid, default_flow_style=False)

        print('Wrote configs to {}'.format(config_path))

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def train(self, data_train,
              data_val,
              batch_size=2,
              epochs=60,
              lr=0.000005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, save_path = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size}) 

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = maybe_apply_uncertainty_wrapper(
            self.get_model(data_train['data_params'])
        )
        uses_uncertainty_loss = isinstance(train_model, UncertaintyWeightedModel)

        # 统计参数数量
        total_params = train_model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in train_model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print(f"📊 MODEL PARAMETER STATISTICS")
        print(f"{'='*60}")
        print(f"Total parameters:        {total_params:,}")
        print(f"Trainable parameters:    {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"{'='*60}\n")

        # plot_model(train_model, to_file=path_params['save_folder']+'/model_structure.png', show_shapes=True)
        # Generate detailed model architecture diagram
        plot_model(
            train_model,
            to_file=os.path.join(save_path, 'model_structure.png'),
            show_layer_names=True,
            rankdir='TB',
        )
        
        # 显示详细的模型结构
        # train_model.summary()

        # Train the model
        # class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        # base_lr = 3e-4          # 可按你现在的学习率起步
        # weight_decay = 1e-4     # 替代原先各层 L2(3e-4)

        compile_metrics = {
            'intention': ['accuracy']
        }

        if uses_uncertainty_loss:
            train_model.compile(
                optimizer=optimizer,
                metrics=compile_metrics
            )
        else:
            train_model.compile(
                loss={
                    'intention': BinaryFocalLoss(),
                    # Temporarily switch etraj loss to standard MSE
                    'etraj': FocalL2Loss()
                    # 'etraj': MeanSquaredError()
                },
                loss_weights={
                    'intention': 1,
                    'etraj': 0.0000  # 轨迹 loss 权重可调
                },
                optimizer=optimizer,
                metrics=compile_metrics
            )

        # === 新的回调设置：保存每个epoch的训练结果 ===
        callbacks = []
        
        # 1. 使用自定义回调保存每个epoch和最佳模型
        epoch_save_callback = EpochSaveCallback(
            save_dir=os.path.dirname(model_path),  # 保存在模型目录下
            save_weights_only=False,  # 保存完整模型
            save_format='h5'
        )
        callbacks.append(epoch_save_callback)
        
        # 2. 详细日志记录
        log_callback = DetailedLoggingCallback(
            log_dir=os.path.dirname(model_path),
            log_frequency=1  # 每个epoch都记录
        )
        callbacks.append(log_callback)

        if uses_uncertainty_loss:
            callbacks.append(SigmaLoggingCallback())
        
        # 5. 学习率调度器（可选）
        if model_opts.get('use_lr_scheduler', False):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.2,
                patience=5,
                cooldown=1,
                min_lr=1e-7,
                verbose=1)
            callbacks.append(reduce_lr)
        
        # data_val = data_val.batch(batch_size)
        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=None,
                                  epochs=epochs,
                                  validation_data=data_val,
                                #   class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        # print(history.history.keys())
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        if model_opts['model'] == 'Transformer_depth':
            from training_plots import save_training_plots
            save_training_plots(history, path_params, model_opts['model'])

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        
        # 转换 history.history 为可序列化的格式
        history_data = {}
        for key, values in history.history.items():
            # 将numpy数组转换为Python列表
            if hasattr(values, 'tolist'):
                history_data[key] = values.tolist()
            else:
                history_data[key] = list(values)

        # 保存为YAML格式
        with open(history_path, 'w') as fid:
            yaml.dump(history_data, fid, default_flow_style=False, 
                    allow_unicode=True, indent=2)
            
        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)

        custom_objects = {
            'UncertaintyWeightedModel': UncertaintyWeightedModel,
            'BinaryFocalLoss': BinaryFocalLoss,
            'FocalL2Loss': FocalL2Loss
        }
        test_model = load_model(os.path.join(model_path, 'model.h5'),
                                custom_objects=custom_objects)
        # test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        # 只取输入部分
        X_test, y_test = test_data['data']

        # 如果是 generator 模式
        if isinstance(X_test, DataGenerator):
            test_results = test_model.predict(X_test, verbose=1)
        else:
            test_results = test_model.predict(X_test, batch_size=1, verbose=1)

        intention_results = test_results[0]

        acc = accuracy_score(y_test, np.round(intention_results))
        f1 = f1_score(y_test, np.round(intention_results))
        auc = roc_auc_score(y_test, np.round(intention_results))
        precision = precision_score(y_test, np.round(intention_results))
        recall = recall_score(y_test, np.round(intention_results))
        
        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                        # 'pid': test_data['ped_id'],
                        'gt':test_data['data'][1],
                        'y': test_results,
                        # 'image': test_data['image']
                        }, 
                        picklefile)


        # print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))
        print('\n' + '\033[96m' + '='*70 + '\033[0m')
        print('\033[1m\033[92m🎯 MODEL TEST RESULTS 🎯\033[0m')
        print('\033[96m' + '='*70 + '\033[0m')
        print('\033[93mAccuracy:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(acc))
        print('\033[94mAUC:        \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(0 if np.isnan(auc) else auc))
        print('\033[95mF1-Score:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(f1))
        print('\033[96mPrecision:  \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(precision))
        print('\033[91mRecall:     \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(recall))
        print('\033[96m' + '='*70 + '\033[0m\n')


        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': '{:.4f}'.format(acc),
                    'auc': '{:.4f}'.format(auc),
                    'f1': '{:.4f}'.format(f1),
                    # 'roc': '{:.4f}'.format(roc),
                    'precision': '{:.4f}'.format(precision),
                    'recall': '{:.4f}'.format(recall),
                    # 'pre_recall_curve': '{:.4f}'.format(pre_recall)
                    }

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))
    

class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=False,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False,
                 class_weight=None):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list

        base_len = len(self.labels[0]) if (isinstance(self.labels, list) and len(self.labels) > 0) else len(self.labels)
        self.batch_size = 1 if base_len < batch_size else batch_size

        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()
        self.class_weight = None  # 不再使用 class weight

    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            sw = self._generate_sample_weight(y)
            # sw = {'intention': np.array([0, 0.001], dtype=np.float32), 'etraj': np.array([0, 0], dtype=np.float32)}
            return X, y, sw
        else:
            return X

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1] // len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i, ..., j * num_ch:j * num_ch + num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        if isinstance(self.labels, list) and len(self.labels) > 1:
            intention_labels = np.array(self.labels[0][indices])
            etraj_labels = np.array(self.labels[1][indices]) if self.labels[1] is not None else None
            return [intention_labels, etraj_labels]
        else:
            if isinstance(self.labels, list):
                return np.array(self.labels[0][indices])
            else:
                return np.array(self.labels[indices])

    def _generate_sample_weight(self, y):
        """
        统一规则（无配置、无模式）：
        - intention：不再使用 class weight，所有样本权重为 1
        - etraj：    与 intention 保持一致；若轨迹标签无效则置 0，避免污染
        """
        if isinstance(y, list) and len(y) > 1:
            y_int = np.asarray(y[0]).astype(np.int32).reshape(-1)
            etraj_labels = y[1]

            sw_int = np.ones_like(y_int, dtype='float32')
            sw_etraj = sw_int.copy()

            # 若轨迹标签缺失/无效（比如 NaN/Inf），则将该样本的 etraj 权重置 0，避免污染回归损失
            if etraj_labels is None:
                sw_etraj[:] = 0.0
            else:
                etraj_np = np.asarray(etraj_labels)
                invalid = ~np.isfinite(etraj_np).all(axis=1)
                if invalid.any():
                    sw_etraj[invalid] = 0.0

            return {'intention': sw_int, 'etraj': sw_etraj}

        else:
            y_int = np.asarray(y).astype(np.int32).reshape(-1)
            return np.ones_like(y_int, dtype='float32')

@tf.keras.utils.register_keras_serializable()
class CLSTokenLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token"
        )

    # @tf.autograph.experimental.do_not_convert
    def call(self, x):
        batch_size = tf.shape(x)[0]
        # 返回可广播的 token tensor
        return tf.tile(self.cls_token, [batch_size, 1, 1])

    @tf.autograph.experimental.do_not_convert
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class Transformer_depth(ActionPredict):
    """
    多模态Transformer网络结构，支持行人过街意图与轨迹联合预测。
    输入：Bounding Box, Depth, Vehicle Speed, Pedestrian Speed（均为序列）
    输出：Intention（二分类），E-Traj（下一帧xy坐标）
    """
    def __init__(self, num_heads=8, d_model=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model 
        self.dropout = dropout
        self.dataset = kwargs['dataset']
        self.sample = kwargs['sample_type']

    def embedding_norm_block(self, input_tensor, name=None):
        """Dense + LayerNorm"""
        # x = Dense(self.d_model, activation=None, kernel_regularizer=regularizers.L2(0.0003), name=f'{name}_embedding_norm')(input_tensor)
        x = Dense(self.d_model, activation=None, name=f'{name}_embedding_norm')(input_tensor)
        x = LayerNormalization(name=f'{name}_ln')(x)
        return x

    def cmim_block(self, x1, x2, dropout = 0.1, name=None):
        """Cross-Modal Interaction Module: 双向交叉注意力 + 残差"""
        attn1 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            output_shape=self.d_model,
            dropout=dropout,
            # kernel_regularizer=regularizers.L2(0.001),  # 权重正则化
            name=f'{name}_attn1'
        )
        attn2 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            output_shape=self.d_model,
            dropout=dropout,
            # kernel_regularizer=regularizers.L2(0.001),  # 权重正则化
            name=f'{name}_attn2'
        )
        y1 = attn1(query=x2, value=x1, key=x1)
        y1 = Dropout(dropout)(y1)
        y1 = Add(name=f'{name}_add1')([x1, y1])
        # y1 = LayerNormalization(name=f'{name}_ln1')(y1)

        y2 = attn2(query=x1, value=x2, key=x2)
        y2 = Dropout(dropout)(y2)
        y2 = Add(name=f'{name}_add2')([x2, y2])
        # y2 = LayerNormalization(name=f'{name}_ln2')(y2)

        return Add(name=f'{name}_fuse')([y1, y2])
    
    def fem_block(self, x, dropout = 0.1, name=None):
        """Feature Enhancement Module: PreNorm -> FFN (GELU+Linear) -> Residual Add"""
        # x_in = x
        x = LayerNormalization(name=f'{name}_fem_norm')(x)
        shortcut = x
        x = Dense(2 * self.d_model, activation=tf.nn.gelu, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn1_dense1')(x)
        x = Dense(self.d_model, activation=None, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn1_dense2')(x)
        # x = Dense(2 * self.d_model, activation=tf.nn.gelu, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn2_dense1')(x)
        # x = Dense(self.d_model, activation=None, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn2_dense2')(x)
        # x = Dense(2 * self.d_model, activation=tf.nn.gelu, name=f'{name}_fem_ffn1_dense1')(x)
        # x = Dense(self.d_model, activation=None, name=f'{name}_fem_ffn1_dense2')(x)
        x = Dropout(dropout, name=f'{name}_fem_drop')(x)
        # x = tfa.layers.StochasticDepth(
        #     survival_probability=0.9, name=f'{name}_sd'
        #     )([x, x_in])  # 随机深度
        x = Add(name=f'{name}_fem_add')([shortcut, x])
        # x = Add(name=f'{name}_fem_add')([x_in, x])
        return x

    def positional_encoding(self, x):
        """正余弦位置编码"""
        def compute_pos_encoding(inputs):
            seq_len = tf.shape(inputs)[1]
            d_model = tf.shape(inputs)[2]
            
            # 创建位置和维度索引
            pos = tf.range(tf.cast(seq_len, tf.float32))[:, tf.newaxis]
            i = tf.range(tf.cast(d_model, tf.float32))[tf.newaxis, :]
            
            # 计算角度率
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            angle_rads = pos * angle_rates
            
            # 对偶数索引应用sin，对奇数索引应用cos
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            
            # 拼接sin和cos
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            
            # 使用tf.cond处理维度匹配，避免直接使用Python if
            def pad_encoding():
                return tf.pad(pos_encoding, [[0, 0], [0, d_model - tf.shape(pos_encoding)[-1]]])
            
            def slice_encoding():
                return pos_encoding[:, :d_model]
            
            pos_encoding_adjusted = tf.cond(
                tf.shape(pos_encoding)[-1] < d_model,
                pad_encoding,
                slice_encoding
            )
            
            # 添加batch维度并与输入相加
            pos_encoding_adjusted = pos_encoding_adjusted[tf.newaxis, :, :]
            return inputs + pos_encoding_adjusted

        return Lambda(compute_pos_encoding, name="positional_encoding")(x)

    def mhsa_block(self, x, dropout = 0.1, name=None, attention_mask=None):
        """Pre-LN Multi-Head Self-Attention + 残差"""
        # x_in = x
        x_norm = LayerNormalization(name=f'{name}_mhsa_norm')(x)

        attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,     # 每头维度
            value_dim=self.d_model // self.num_heads,   # 显式给出
            output_shape=self.d_model,                  # 输出回 d_model，方便残差相加
            dropout=dropout,
            # kernel_regularizer=regularizers.L2(0.001),  # 权重正则化
            name=f'{name}_mhsa'
        )

        attn_out = attn(
            query=x_norm, value=x_norm, key=x_norm,
        )
        x = Dropout(dropout, name=f'{name}_mhsa_drop')(attn_out)
        # x = tfa.layers.StochasticDepth(
        #     survival_probability=0.9, name=f'{name}_sd'
        # )([x, x_in])
        x = Add(name=f'{name}_mhsa_res')([x, x_norm])
        # x = Add(name=f'{name}_mhsa_res')([x, x_in])
        return x

    def get_model(self, data_params):
        bbox_in = Input(shape=(None, 4), name='bbox')
        depth_in = Input(shape=(None, 1), name='depth')
        vehspd_in = Input(shape=(None, 1), name='vehspd')
        if 'watch' in self.dataset:
            pedspd_in = Input(shape=(None, 3), name='pedspd')
        else:
            pedspd_in = Input(shape=(None, 4), name='pedspd')

        bbox = self.embedding_norm_block(bbox_in, name='bbox')
        depth = self.embedding_norm_block(depth_in, name='depth')
        vehspd = self.embedding_norm_block(vehspd_in, name='vehspd')
        pedspd = self.embedding_norm_block(pedspd_in, name='pedspd')

        x = self.cmim_block(vehspd, pedspd, name='cmim_vehspd_pedspd')
        x = self.fem_block(x, name='fem_vehspd_pedspd')
        x = self.cmim_block(depth, x, name='cmim_depth_vehspd_pedspd')
        x = self.fem_block(x, name='fem_depth_vehspd_pedspd')
        x = self.cmim_block(bbox, x, name='cmim_all')
        x = self.fem_block(x, name='fem_all')

        cls_token = CLSTokenLayer(self.d_model)(x)
        x = Concatenate(axis=1, name='add_cls')([cls_token, x])

        # Add positional encoding
        x = self.positional_encoding(x)

        x = self.mhsa_block(x, dropout = 0.1, name='mhsa_1')
        x = self.fem_block(x, dropout = 0.1, name='fem_after_mhsa_1')
        # x = self.mhsa_block(x, dropout = 0.2, name='mhsa_2')
        # x = self.fem_block(x, dropout = 0.2, name='fem_after_mhsa_2')

        cls_out = Lambda(lambda t: t[:, 0, :], name='cls_slice')(x)
        
        # —— Intention head
        ci = Dropout(0.1, name='cls_dropout')(cls_out)
        h = Dense(128, activation='gelu', kernel_regularizer=regularizers.l2(5e-3), name='head_fc1')(ci)
        h = Dropout(0.1, name='head_dropout1')(h)
        intention = Dense(1, activation='sigmoid', name='intention')(h)

        # —— Etraj head（独立分支）
        # ce = Dropout(0.1, name='cls_dropout_e')(cls_out)
        e = Dense(128, activation='gelu', kernel_regularizer=regularizers.l2(5e-3), name='head_fc2')(cls_out)
        # e = Dropout(0.1, name='head_dropout2')(e)
        etraj = Dense(4, activation=None, name='etraj')(e)

        model = Model(inputs=[bbox_in, depth_in, vehspd_in, pedspd_in],
                      outputs=[intention, etraj],
                      name='Transformer_depth')

        return model

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        # data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        # model_opts_3d = model_opts.copy()
        if model_opts['dataset'] == 'jaad' or 'pie':
            data['vehicle_speed'] = data['speed']
            data['ped_speed'] = data['ped_center_diff']

        for d_type in model_opts['obs_input_type']:
            features = data[d_type]
            feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=[data['crossing'], data['trajectory']],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                # 'ped_id': data['ped_id'],
                'tte': data['tte'],
                # 'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_data_sequence(self, data_type, data_raw, opts):
        dataset = opts['dataset']  # 'jaad' 或 'pie'
        d = {
            'center': data_raw['center'].copy(),
            'box': data_raw['bbox'].copy(),
            'ped_id': data_raw['pid'].copy(),
            'crossing': data_raw['activities'].copy(),
            'image': data_raw['image'].copy()
        }

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        # 加速度信息
        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()

        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])

        d['tte'] = []

        # ========== 切片序列 ==========
        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

        # ========== 计算 ped_center_diff 与 轨迹（4维 bbox ==========
        d['ped_center_diff'] = []
        d['trajectory'] = []
        for idx, seq in enumerate(data_raw['bbox']):
            # 帧间差分
            diffs = []
            for j in range(1, len(seq)):
                diff = np.array(seq[j]) - np.array(seq[j - 1])
                diffs.append(diff)
            diffs = [diffs[0]] + diffs

            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            d['ped_center_diff'].extend([diffs[i:i + obs_length] for i in
                                            range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
            # 下一帧 4维 bbox（像素坐标）
            # d['trajectory'].extend([[seq[i + obs_length][0] / data_raw['image_dimension'][0], seq[i + obs_length][1] / data_raw['image_dimension'][1],
            #                          seq[i + obs_length][2] / data_raw['image_dimension'][0], seq[i + obs_length][3] / data_raw['image_dimension'][1]]
            #                         for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
            # 下0.5秒bbox
            d['trajectory'].extend([[(seq[i + obs_length][0] + time_to_event[0] - 1) / data_raw['image_dimension'][0], (seq[i + obs_length][1] + time_to_event[0] - 1) / data_raw['image_dimension'][1],
                                        (seq[i + obs_length][2] + time_to_event[0] - 1) / data_raw['image_dimension'][0], (seq[i + obs_length][3] + time_to_event[0] - 1) / data_raw['image_dimension'][1]]
                                    for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
            # d['trajectory'].extend([[seq[i + obs_length][0], seq[i + obs_length][1],
            #                          seq[i + obs_length][2], seq[i + obs_length][3]]
            #                         for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])

        # ========== 深度信息 ==========
        import os, pickle
        cache_dir = f'{dataset.upper()}/data_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        cache_filename = (
            f'depth_{self.dataset}_{self.sample}_{data_type}_'
            f'obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
            if dataset == 'jaad' else
            f'depth_{self.dataset}_{data_type}_obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
        )
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                d['depth'] = pickle.load(f)
        else:
            d['depth'] = []
            for idx, seq in enumerate(data_raw['bbox']):
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
                images = data_raw['image'][idx][start_idx:end_idx + obs_length + 1]
                boxes = data_raw['bbox'][idx][start_idx:end_idx + obs_length + 1]
                depth_seq = []
                for image_path, box in zip(images, boxes):
                    depth_image_path = image_path.replace('/images/', '/images_depth_gray/')
                    img = cv2.imread(depth_image_path)
                    img_height, img_width = img.shape[:2]
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(int(x1), img_width - 1))
                    y1 = max(0, min(int(y1), img_height - 1))
                    x2 = max(x1 + 1, min(int(x2), img_width))
                    y2 = max(y1 + 1, min(int(y2), img_height))
                    bbox_region = img[y1:y2, x1:x2]
                    if bbox_region.size == 0:
                        print(f"Warning: Empty bbox region for image {image_path}")
                        depth_seq.append(None)
                        continue
                    pixel_mean = np.mean(bbox_region)
                    depth_seq.append(float(pixel_mean))
                d['depth'].extend([depth_seq[i:i + obs_length] for i in
                                range(0, end_idx - start_idx + 1, olap_res)])
                if (idx + 1) % 10 == 0:
                    print(f"Processed depth for {idx + 1}/{len(data_raw['bbox'])} sequences")

            with open(cache_path, 'wb') as f:
                pickle.dump(d['depth'], f, pickle.HIGHEST_PROTOCOL)

        # ========== 归一化 ==========
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        if dataset == 'pie':
            print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count
