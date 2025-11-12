 cd /home/minshi/Pedestrian_Crossing_Intention_Prediction ; /usr/bin/env /home/minshi/miniconda3/envs/tf26/bin/python /home/minshi/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 51515 -- /home/minshi/Pedestrian_Crossing_Intention_Prediction/train_and_test_all_epoch_pipeline.py -c config_files/my/my_jaad.yaml 
================================================================================
🎯 训练和测试管道启动
================================================================================
配置文件: config_files/my/my_jaad.yaml
开始时间: 2025-11-12 02:03:16

🚀 开始训练...
2025-11-12 02:03:18.834777: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:18.838051: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:18.838141: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   config_files/my/my_jaad.yaml
model_opts {'model': 'Transformer_depth', 'obs_input_type': ['box', 'depth', 'vehicle_speed', 'ped_speed'], 'enlarge_ratio': 1.5, 'obs_length': 16, 'time_to_ev
ent': [30, 60], 'overlap': 0.8, 'balance_data': False, 'apply_class_weights': True, 'dataset': 'jaad', 'normalize_boxes': True, 'generator': True, 'fusion_point': 'early', 'fusion_method': 'sum'}                                                                                                                           data_opts {'fstride': 1, 'sample_type': 'beh', 'subset': 'default', 'data_split_type': 'default', 'seq_type': 'crossing', 'min_track_size': 76}
net_opts {'num_hidden_units': 256, 'global_pooling': 'avg', 'regularizer_val': 0.0001, 'cell_type': 'gru', 'backbone': 'vgg16', 'dropout': 0.1}
train_opts {'batch_size': 2, 'epochs': 30, 'lr': 5e-05, 'learning_scheduler': {}}
---------------------------------------------------------
Generating action sequence data
fstride: 1
sample_type: beh
subset: default
height_rng: [0, inf]
squarify_ratio: 0
data_split_type: default
seq_type: crossing
min_track_size: 76
random_params: {'ratios': None, 'val_data': True, 'regen_data': False}
kfold_params: {'num_folds': 5, 'fold': 1}
---------------------------------------------------------
Generating database for jaad
jaad database loaded from /home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD/data_cache/jaad_database.pkl
---------------------------------------------------------
Generating crossing data
Split: train
Number of pedestrians: 324 
Total number of samples: 194 
---------------------------------------------------------
Generating action sequence data
fstride: 1
sample_type: beh
subset: default
height_rng: [0, inf]
squarify_ratio: 0
data_split_type: default
seq_type: crossing
min_track_size: 76
random_params: {'ratios': None, 'val_data': True, 'regen_data': False}
kfold_params: {'num_folds': 5, 'fold': 1}
---------------------------------------------------------
Generating database for jaad
jaad database loaded from /home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD/data_cache/jaad_database.pkl
---------------------------------------------------------
Generating crossing data
Split: val
Number of pedestrians: 48 
Total number of samples: 22 
---------------------------------------------------------
Generating action sequence data
fstride: 1
sample_type: beh
subset: default
height_rng: [0, inf]
squarify_ratio: 0
data_split_type: default
seq_type: crossing
min_track_size: 76
random_params: {'ratios': None, 'val_data': True, 'regen_data': False}
kfold_params: {'num_folds': 5, 'fold': 1}
---------------------------------------------------------
Generating database for jaad
jaad database loaded from /home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD/data_cache/jaad_database.pkl
---------------------------------------------------------
Generating crossing data
Split: test
Number of pedestrians: 276 
Total number of samples: 171 
[DataGenerator] auto class_weight -> {0: 0.8247422680412371, 1: 0.17525773195876287}
[DataGenerator] auto class_weight -> {0: 0.7272727272727273, 1: 0.2727272727272727}
2025-11-12 02:03:20.693188: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (o
neDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 02:03:20.694040: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.694273: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.694341: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.960037: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.960144: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.960205: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:03:20.960264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3871 MB m
emory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                              
============================================================
📊 MODEL PARAMETER STATISTICS
============================================================
Total parameters:        2,968,717
Trainable parameters:    2,968,711.0
Non-trainable parameters: 6.0
============================================================

/home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use `learni
ng_rate` instead.                                                                                                                                                warnings.warn(

🚀 Training started!
📁 Models will be saved to: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s
📋 已复制 action_predict.py 到模型目录
2025-11-12 02:03:22.100760: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/30
WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimi
zing the loss.                                                                                                                                                 WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimi
zing the loss.                                                                                                                                                 2025-11-12 02:03:25.735385: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be l
ogged once.                                                                                                                                                    2025-11-12 02:03:26.037039: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1067/1067 [==============================] - 14s 10ms/step - loss: 9.0920 - cls_loss: 0.2098 - reg_loss: 0.3073 - intention_accuracy: 0.5450 - val_loss: 4.9273
 - val_cls_loss: 0.2694 - val_reg_loss: 0.1699 - val_intention_accuracy: 0.5207                                                                                /home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must
 override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.                                                         warnings.warn('Custom mask layers require a config and must override '
[Sigma] Epoch 1: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 2/30
1067/1067 [==============================] - 9s 9ms/step - loss: 3.1901 - cls_loss: 0.1780 - reg_loss: 0.1042 - intention_accuracy: 0.6275 - val_loss: 2.2477 -
 val_cls_loss: 0.2778 - val_reg_loss: 0.1254 - val_intention_accuracy: 0.5579                                                                                  [Sigma] Epoch 2: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 3/30
1067/1067 [==============================] - 9s 9ms/step - loss: 1.6788 - cls_loss: 0.1658 - reg_loss: 0.1059 - intention_accuracy: 0.6387 - val_loss: 1.4759 -
 val_cls_loss: 0.2999 - val_reg_loss: 0.1401 - val_intention_accuracy: 0.6488                                                                                  [Sigma] Epoch 3: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 4/30
1067/1067 [==============================] - 10s 9ms/step - loss: 1.1173 - cls_loss: 0.1567 - reg_loss: 0.1116 - intention_accuracy: 0.6715 - val_loss: 1.0471 
- val_cls_loss: 0.2697 - val_reg_loss: 0.1434 - val_intention_accuracy: 0.6281                                                                                 [Sigma] Epoch 4: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 5/30
1067/1067 [==============================] - 10s 9ms/step - loss: 0.7916 - cls_loss: 0.1500 - reg_loss: 0.1132 - intention_accuracy: 0.6931 - val_loss: 0.8136 
- val_cls_loss: 0.2898 - val_reg_loss: 0.1445 - val_intention_accuracy: 0.5702                                                                                 [Sigma] Epoch 5: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 6/30
1067/1067 [==============================] - 10s 9ms/step - loss: 0.5821 - cls_loss: 0.1458 - reg_loss: 0.1136 - intention_accuracy: 0.6945 - val_loss: 0.7534 
- val_cls_loss: 0.3933 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5620                                                                                 [Sigma] Epoch 6: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 7/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.4429 - cls_loss: 0.1394 - reg_loss: 0.1136 - intention_accuracy: 0.7067 - val_loss: 0.6185
 - val_cls_loss: 0.3643 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6033                                                                                [Sigma] Epoch 7: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 8/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.3557 - cls_loss: 0.1372 - reg_loss: 0.1136 - intention_accuracy: 0.7188 - val_loss: 0.5978
 - val_cls_loss: 0.4108 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6033                                                                                [Sigma] Epoch 8: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 9/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.2900 - cls_loss: 0.1259 - reg_loss: 0.1136 - intention_accuracy: 0.7371 - val_loss: 0.5539
 - val_cls_loss: 0.4098 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6074                                                                                [Sigma] Epoch 9: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 10/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.2499 - cls_loss: 0.1207 - reg_loss: 0.1136 - intention_accuracy: 0.7521 - val_loss: 0.5542
 - val_cls_loss: 0.4381 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                                [Sigma] Epoch 10: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 11/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.2138 - cls_loss: 0.1080 - reg_loss: 0.1136 - intention_accuracy: 0.7919 - val_loss: 0.4891
 - val_cls_loss: 0.3927 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                                [Sigma] Epoch 11: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 12/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1874 - cls_loss: 0.0985 - reg_loss: 0.1136 - intention_accuracy: 0.8210 - val_loss: 0.5408
 - val_cls_loss: 0.4586 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6653                                                                                [Sigma] Epoch 12: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 13/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1742 - cls_loss: 0.0976 - reg_loss: 0.1136 - intention_accuracy: 0.8219 - val_loss: 0.4515
 - val_cls_loss: 0.3805 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6818                                                                                [Sigma] Epoch 13: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 14/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1558 - cls_loss: 0.0892 - reg_loss: 0.1136 - intention_accuracy: 0.8515 - val_loss: 0.4768
 - val_cls_loss: 0.4136 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6281                                                                                [Sigma] Epoch 14: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 15/30
1067/1067 [==============================] - 24s 23ms/step - loss: 0.1462 - cls_loss: 0.0879 - reg_loss: 0.1136 - intention_accuracy: 0.8552 - val_loss: 0.5042
 - val_cls_loss: 0.4501 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7190                                                                                [Sigma] Epoch 15: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 16/30
1067/1067 [==============================] - 24s 23ms/step - loss: 0.1371 - cls_loss: 0.0862 - reg_loss: 0.1136 - intention_accuracy: 0.8571 - val_loss: 0.5646
 - val_cls_loss: 0.5171 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6488                                                                                [Sigma] Epoch 16: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 17/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1310 - cls_loss: 0.0865 - reg_loss: 0.1136 - intention_accuracy: 0.8463 - val_loss: 0.5135
 - val_cls_loss: 0.4716 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6240                                                                                [Sigma] Epoch 17: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 18/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1171 - cls_loss: 0.0777 - reg_loss: 0.1136 - intention_accuracy: 0.8557 - val_loss: 0.4789
 - val_cls_loss: 0.4422 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6281                                                                                [Sigma] Epoch 18: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 19/30
1067/1067 [==============================] - 18s 17ms/step - loss: 0.1122 - cls_loss: 0.0774 - reg_loss: 0.1136 - intention_accuracy: 0.8590 - val_loss: 0.4773
 - val_cls_loss: 0.4432 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                                [Sigma] Epoch 19: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 20/30
1067/1067 [==============================] - 13s 13ms/step - loss: 0.1051 - cls_loss: 0.0743 - reg_loss: 0.1136 - intention_accuracy: 0.8664 - val_loss: 0.4300
 - val_cls_loss: 0.4009 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6736                                                                                [Sigma] Epoch 20: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 21/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.1006 - cls_loss: 0.0735 - reg_loss: 0.1136 - intention_accuracy: 0.8683 - val_loss: 0.5143
 - val_cls_loss: 0.4885 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                                [Sigma] Epoch 21: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 22/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.0948 - cls_loss: 0.0707 - reg_loss: 0.1136 - intention_accuracy: 0.8716 - val_loss: 0.4701
 - val_cls_loss: 0.4476 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7107                                                                                [Sigma] Epoch 22: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 23/30
1067/1067 [==============================] - 13s 13ms/step - loss: 0.0907 - cls_loss: 0.0691 - reg_loss: 0.1136 - intention_accuracy: 0.8889 - val_loss: 0.4937
 - val_cls_loss: 0.4737 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6983                                                                                [Sigma] Epoch 23: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 24/30
1067/1067 [==============================] - 20s 18ms/step - loss: 0.0894 - cls_loss: 0.0701 - reg_loss: 0.1136 - intention_accuracy: 0.8824 - val_loss: 0.5142
 - val_cls_loss: 0.4962 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7231                                                                                [Sigma] Epoch 24: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 25/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0854 - cls_loss: 0.0676 - reg_loss: 0.1136 - intention_accuracy: 0.8843 - val_loss: 0.5986
 - val_cls_loss: 0.5818 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6694                                                                                [Sigma] Epoch 25: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 26/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0845 - cls_loss: 0.0689 - reg_loss: 0.1136 - intention_accuracy: 0.8838 - val_loss: 0.5540
 - val_cls_loss: 0.5393 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6074                                                                                [Sigma] Epoch 26: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 27/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0891 - cls_loss: 0.0739 - reg_loss: 0.1136 - intention_accuracy: 0.8688 - val_loss: 0.5550
 - val_cls_loss: 0.5408 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6488                                                                                [Sigma] Epoch 27: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 28/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0805 - cls_loss: 0.0673 - reg_loss: 0.1136 - intention_accuracy: 0.8824 - val_loss: 0.5390
 - val_cls_loss: 0.5259 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6901                                                                                [Sigma] Epoch 28: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 29/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0742 - cls_loss: 0.0618 - reg_loss: 0.1136 - intention_accuracy: 0.8964 - val_loss: 0.6846
 - val_cls_loss: 0.6727 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6983                                                                                [Sigma] Epoch 29: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 30/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0786 - cls_loss: 0.0669 - reg_loss: 0.1136 - intention_accuracy: 0.8754 - val_loss: 0.5910
 - val_cls_loss: 0.5801 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6612                                                                                [Sigma] Epoch 30: sigma_cls=0.6931 sigma_reg=0.6931

🎯 Training completed!
📁 All epoch models saved in: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/epochs
Train model is saved to data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/model.h5
Available metrics: ['loss', 'cls_loss', 'reg_loss', 'intention_accuracy', 'val_loss', 'val_cls_loss', 'val_reg_loss', 'val_intention_accuracy', 'sigma_cls', 'v
al_sigma_cls', 'sigma_reg', 'val_sigma_reg']                                                                                                                   Training plots saved to model directory
Wrote configs to data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/configs.yaml
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 14s 7ms/step

======================================================================
🎯 MODEL TEST RESULTS 🎯
======================================================================
Accuracy:   0.6050
AUC:        0.5471
F1-Score:   0.7112
Precision:  0.6554
Recall:     0.7774
======================================================================

Model saved to data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/

✅ 训练完成 (耗时: 9.8 分钟)
🔍 查找最新模型目录...
📁 找到模型目录: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s

🧪 开始测试模型...
2025-11-12 02:13:08.384255: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:08.388455: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:08.388584: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   🚀 开始测试模型目录: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s
✅ 配置文件加载成功
✅ 数据集初始化成功
🔄 生成测试数据...
---------------------------------------------------------
Generating action sequence data
fstride: 1
sample_type: beh
subset: default
height_rng: [0, inf]
squarify_ratio: 0
data_split_type: default
seq_type: crossing
min_track_size: 76
random_params: {'ratios': None, 'val_data': True, 'regen_data': False}
kfold_params: {'num_folds': 5, 'fold': 1}
---------------------------------------------------------
Generating database for jaad
jaad database loaded from /home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD/data_cache/jaad_database.pkl
---------------------------------------------------------
Generating crossing data
Split: test
Number of pedestrians: 276 
Total number of samples: 171 
✅ 测试数据生成完成
📁 找到 31 个模型文件

============================================================
进度: 1/31

🔍 测试模型: epoch_001_loss_4.9273_acc_0.5207.h5
2025-11-12 02:13:09.442888: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (o
neDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 02:13:09.444525: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.444670: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.444755: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.787631: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.787780: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.787872: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 02:13:09.787954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4026 MB m
emory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                              WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
2025-11-12 02:13:10.858168: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2025-11-12 02:13:12.102035: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be l
ogged once.                                                                                                                                                    2025-11-12 02:13:12.510975: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1881/1881 [==============================] - 13s 6ms/step
✅ 准确率: 0.5965, AUC: 0.7168, F1: 0.6016

============================================================
进度: 2/31

🔍 测试模型: epoch_002_loss_2.2477_acc_0.5579.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6443, AUC: 0.7150, F1: 0.6613

============================================================
进度: 3/31

🔍 测试模型: epoch_003_loss_1.4759_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6784, AUC: 0.7056, F1: 0.7511

============================================================
进度: 4/31

🔍 测试模型: epoch_004_loss_1.0471_acc_0.6281.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6411, AUC: 0.6757, F1: 0.6787

============================================================
进度: 5/31

🔍 测试模型: epoch_005_loss_0.8136_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6566, AUC: 0.7018, F1: 0.6849

============================================================
进度: 6/31

🔍 测试模型: epoch_006_loss_0.7534_acc_0.5620.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6757, AUC: 0.7135, F1: 0.7362

============================================================
进度: 7/31

🔍 测试模型: epoch_007_loss_0.6185_acc_0.6033.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6231, AUC: 0.6527, F1: 0.7344

============================================================
进度: 8/31

🔍 测试模型: epoch_008_loss_0.5978_acc_0.6033.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6502, AUC: 0.7078, F1: 0.6942

============================================================
进度: 9/31

🔍 测试模型: epoch_009_loss_0.5539_acc_0.6074.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6390, AUC: 0.6821, F1: 0.6976

============================================================
进度: 10/31

🔍 测试模型: epoch_010_loss_0.5542_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6087, AUC: 0.6662, F1: 0.7241

============================================================
进度: 11/31

🔍 测试模型: epoch_011_loss_0.4891_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6241, AUC: 0.6413, F1: 0.6890

============================================================
进度: 12/31

🔍 测试模型: epoch_012_loss_0.5408_acc_0.6653.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 16s 8ms/step
✅ 准确率: 0.6337, AUC: 0.6631, F1: 0.7170

============================================================
进度: 13/31

🔍 测试模型: epoch_013_loss_0.4515_acc_0.6818.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6454, AUC: 0.6687, F1: 0.7203

============================================================
进度: 14/31

🔍 测试模型: epoch_014_loss_0.4768_acc_0.6281.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6156, AUC: 0.6457, F1: 0.7121

============================================================
进度: 15/31

🔍 测试模型: epoch_015_loss_0.5042_acc_0.7190.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6183, AUC: 0.6331, F1: 0.7116

============================================================
进度: 16/31

🔍 测试模型: epoch_016_loss_0.5646_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6007, AUC: 0.6076, F1: 0.6728

============================================================
进度: 17/31

🔍 测试模型: epoch_017_loss_0.5135_acc_0.6240.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6300, AUC: 0.6414, F1: 0.6984

============================================================
进度: 18/31

🔍 测试模型: epoch_018_loss_0.4789_acc_0.6281.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6390, AUC: 0.6454, F1: 0.7085

============================================================
进度: 19/31

🔍 测试模型: epoch_019_loss_0.4773_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.5976, AUC: 0.6088, F1: 0.6980

============================================================
进度: 20/31

🔍 测试模型: epoch_020_loss_0.4300_acc_0.6736.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 15s 8ms/step
✅ 准确率: 0.6199, AUC: 0.6268, F1: 0.7039

============================================================
进度: 21/31

🔍 测试模型: epoch_021_loss_0.5143_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6002, AUC: 0.6268, F1: 0.7056

============================================================
进度: 22/31

🔍 测试模型: epoch_022_loss_0.4701_acc_0.7107.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6465, AUC: 0.6594, F1: 0.7280

============================================================
进度: 23/31

🔍 测试模型: epoch_023_loss_0.4937_acc_0.6983.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.5949, AUC: 0.6326, F1: 0.7049

============================================================
进度: 24/31

🔍 测试模型: epoch_024_loss_0.5142_acc_0.7231.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6188, AUC: 0.6501, F1: 0.7176

============================================================
进度: 25/31

🔍 测试模型: epoch_025_loss_0.5986_acc_0.6694.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6188, AUC: 0.6327, F1: 0.7233

============================================================
进度: 26/31

🔍 测试模型: epoch_026_loss_0.5540_acc_0.6074.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.5864, AUC: 0.6049, F1: 0.6521

============================================================
进度: 27/31

🔍 测试模型: epoch_027_loss_0.5550_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6194, AUC: 0.6331, F1: 0.7122

============================================================
进度: 28/31

🔍 测试模型: epoch_028_loss_0.5390_acc_0.6901.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6241, AUC: 0.6489, F1: 0.7082

============================================================
进度: 29/31

🔍 测试模型: epoch_029_loss_0.6846_acc_0.6983.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 17s 9ms/step
✅ 准确率: 0.6427, AUC: 0.6756, F1: 0.7290

============================================================
进度: 30/31

🔍 测试模型: epoch_030_loss_0.5910_acc_0.6612.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6050, AUC: 0.6199, F1: 0.7112

============================================================
进度: 31/31

🔍 测试模型: model.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6050, AUC: 0.6199, F1: 0.7112

📊 结果已保存到: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/test_results_20251112_022047.csv
📝 报告已保存到: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/test_report_20251112_022047.txt

🏆 准确率最高的模型: epoch_003_loss_1.4759_acc_0.6488.h5 (准确率: 0.6784)
🗑️  开始清理epochs目录，将删除 30 个模型文件...
🗑️  已删除: epoch_010_loss_0.5542_acc_0.6529.h5
🗑️  已删除: epoch_013_loss_0.4515_acc_0.6818.h5
🗑️  已删除: epoch_023_loss_0.4937_acc_0.6983.h5
🗑️  已删除: epoch_012_loss_0.5408_acc_0.6653.h5
🗑️  已删除: epoch_004_loss_1.0471_acc_0.6281.h5
🗑️  已删除: epoch_009_loss_0.5539_acc_0.6074.h5
🗑️  已删除: epoch_025_loss_0.5986_acc_0.6694.h5
🗑️  已删除: epoch_028_loss_0.5390_acc_0.6901.h5
📋 已将最佳模型复制到: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s/epoch_003_loss_1.4759_acc_0.6488.h5
🗑️  已删除: epoch_003_loss_1.4759_acc_0.6488.h5
🗑️  已删除: epoch_008_loss_0.5978_acc_0.6033.h5
🗑️  已删除: epoch_019_loss_0.4773_acc_0.6777.h5
🗑️  已删除: epoch_007_loss_0.6185_acc_0.6033.h5
🗑️  已删除: epoch_017_loss_0.5135_acc_0.6240.h5
🗑️  已删除: epoch_002_loss_2.2477_acc_0.5579.h5
🗑️  已删除: epoch_015_loss_0.5042_acc_0.7190.h5
🗑️  已删除: epoch_020_loss_0.4300_acc_0.6736.h5
🗑️  已删除: epoch_011_loss_0.4891_acc_0.5868.h5
🗑️  已删除: epoch_005_loss_0.8136_acc_0.5702.h5
🗑️  已删除: epoch_030_loss_0.5910_acc_0.6612.h5
🗑️  已删除: epoch_018_loss_0.4789_acc_0.6281.h5
🗑️  已删除: epoch_022_loss_0.4701_acc_0.7107.h5
🗑️  已删除: epoch_027_loss_0.5550_acc_0.6488.h5
🗑️  已删除: epoch_006_loss_0.7534_acc_0.5620.h5
🗑️  已删除: epoch_021_loss_0.5143_acc_0.6777.h5
🗑️  已删除: epoch_016_loss_0.5646_acc_0.6488.h5
🗑️  已删除: epoch_026_loss_0.5540_acc_0.6074.h5
🗑️  已删除: epoch_014_loss_0.4768_acc_0.6281.h5
🗑️  已删除: epoch_029_loss_0.6846_acc_0.6983.h5
🗑️  已删除: epoch_001_loss_4.9273_acc_0.5207.h5
🗑️  已删除: epoch_024_loss_0.5142_acc_0.7231.h5
🗑️  已删除空的epochs目录
🔄 模型目录已重命名:
   原目录: 12Nov2025-02h03m20s
   新目录: 12Nov2025-02h03m20s_acc_0.6784

================================================================================
🎯 测试结果汇总
================================================================================
总模型数量: 31
成功测试: 31
失败测试: 0

📊 性能统计:
平均准确率: 0.6258 (±0.0230)
平均AUC: 0.6556 (±0.0336)
平均F1: 0.7032 (±0.0286)

🏆 最佳模型:
最高准确率: epoch_003_loss_1.4759_acc_0.6488.h5 (Acc: 0.6784)
最高AUC: epoch_001_loss_4.9273_acc_0.5207.h5 (AUC: 0.7168)
最高F1: epoch_003_loss_1.4759_acc_0.6488.h5 (F1: 0.7511)

✅ 测试完成 (耗时: 7.7 分钟)

================================================================================
🎉 训练和测试管道完成!
================================================================================
模型目录: data/models/jaad/Transformer_depth/12Nov2025-02h03m20s
总耗时: 17.6 分钟
结束时间: 2025-11-12 02:20:49
================================================================================
