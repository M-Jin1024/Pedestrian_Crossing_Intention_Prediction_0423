cd /home/minshi/Pedestrian_Crossing_Intention_Prediction
================================================================================
🎯 训练和测试管道启动
================================================================================
配置文件: config_files/my/my_jaad.yaml
开始时间: 2025-11-12 00:57:57

🚀 开始训练...
2025-11-12 00:58:02.696978: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:02.703008: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:02.703196: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   config_files/my/my_jaad.yaml
model_opts {'model': 'Transformer_depth', 'obs_input_type': ['box', 'depth', 'vehicle_speed', 'ped_speed'], 'enlarge_ratio': 1.5, 'obs_length': 16, 'ti
me_to_event': [30, 60], 'overlap': 0.8, 'balance_data': False, 'apply_class_weights': True, 'dataset': 'jaad', 'normalize_boxes': True, 'generator': True, 'fusion_point': 'early', 'fusion_method': 'sum'}                                                                                                   data_opts {'fstride': 1, 'sample_type': 'beh', 'subset': 'default', 'data_split_type': 'default', 'seq_type': 'crossing', 'min_track_size': 76}
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
2025-11-12 00:58:06.360772: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Li
brary (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 00:58:06.363702: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.364323: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.364731: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.882044: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.882259: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.882399: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 00:58:06.882529: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4
116 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                              
============================================================
📊 MODEL PARAMETER STATISTICS
============================================================
Total parameters:        2,968,717
Trainable parameters:    2,968,711.0
Non-trainable parameters: 6.0
============================================================

/home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use
 `learning_rate` instead.                                                                                                                                warnings.warn(

🚀 Training started!
📁 Models will be saved to: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s
📋 已复制 action_predict.py 到模型目录
2025-11-12 00:58:09.374004: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registere
d 2)                                                                                                                                                   Epoch 1/30
WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] whe
n minimizing the loss.                                                                                                                                 WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] whe
n minimizing the loss.                                                                                                                                 2025-11-12 00:58:17.262697: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will o
nly be logged once.                                                                                                                                    2025-11-12 00:58:17.871382: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1067/1067 [==============================] - 32s 22ms/step - loss: 9.0686 - cls_loss: 0.2147 - reg_loss: 0.1404 - intention_accuracy: 0.5403 - val_loss
: 4.9107 - val_cls_loss: 0.2714 - val_reg_loss: 0.1236 - val_intention_accuracy: 0.5372                                                                /home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config 
and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.                                         warnings.warn('Custom mask layers require a config and must override '
[Sigma] Epoch 1: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 2/30
1067/1067 [==============================] - 18s 17ms/step - loss: 3.1804 - cls_loss: 0.1819 - reg_loss: 0.1008 - intention_accuracy: 0.6261 - val_loss
: 2.2539 - val_cls_loss: 0.2973 - val_reg_loss: 0.1326 - val_intention_accuracy: 0.6570                                                                [Sigma] Epoch 2: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 3/30
1067/1067 [==============================] - 12s 12ms/step - loss: 1.6675 - cls_loss: 0.1665 - reg_loss: 0.1026 - intention_accuracy: 0.6387 - val_loss
: 1.4249 - val_cls_loss: 0.2604 - val_reg_loss: 0.1368 - val_intention_accuracy: 0.6116                                                                [Sigma] Epoch 3: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 4/30
1067/1067 [==============================] - 12s 11ms/step - loss: 1.1128 - cls_loss: 0.1618 - reg_loss: 0.1114 - intention_accuracy: 0.6420 - val_loss
: 1.0310 - val_cls_loss: 0.2613 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5909                                                                [Sigma] Epoch 4: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 5/30
1067/1067 [==============================] - 11s 11ms/step - loss: 0.7891 - cls_loss: 0.1516 - reg_loss: 0.1136 - intention_accuracy: 0.6664 - val_loss
: 0.8633 - val_cls_loss: 0.3390 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5992                                                                [Sigma] Epoch 5: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 6/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.5854 - cls_loss: 0.1444 - reg_loss: 0.1136 - intention_accuracy: 0.6949 - val_loss
: 0.6993 - val_cls_loss: 0.3303 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                [Sigma] Epoch 6: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 7/30
1067/1067 [==============================] - 13s 12ms/step - loss: 0.4549 - cls_loss: 0.1395 - reg_loss: 0.1136 - intention_accuracy: 0.7034 - val_loss
: 0.5754 - val_cls_loss: 0.3070 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5496                                                                [Sigma] Epoch 7: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 8/30
1067/1067 [==============================] - 17s 16ms/step - loss: 0.3626 - cls_loss: 0.1294 - reg_loss: 0.1136 - intention_accuracy: 0.7207 - val_loss
: 0.6390 - val_cls_loss: 0.4367 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6446                                                                [Sigma] Epoch 8: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 9/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.3080 - cls_loss: 0.1286 - reg_loss: 0.1136 - intention_accuracy: 0.7146 - val_loss
: 0.5419 - val_cls_loss: 0.3837 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5661                                                                [Sigma] Epoch 9: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 10/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.2603 - cls_loss: 0.1182 - reg_loss: 0.1136 - intention_accuracy: 0.7291 - val_loss
: 0.4753 - val_cls_loss: 0.3472 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                [Sigma] Epoch 10: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 11/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.2307 - cls_loss: 0.1143 - reg_loss: 0.1136 - intention_accuracy: 0.7357 - val_loss
: 0.4122 - val_cls_loss: 0.3069 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5620                                                                [Sigma] Epoch 11: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 12/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1980 - cls_loss: 0.1018 - reg_loss: 0.1136 - intention_accuracy: 0.7915 - val_loss
: 0.4968 - val_cls_loss: 0.4087 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6157                                                                [Sigma] Epoch 12: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 13/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1816 - cls_loss: 0.0999 - reg_loss: 0.1136 - intention_accuracy: 0.8065 - val_loss
: 0.6473 - val_cls_loss: 0.5726 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6488                                                                [Sigma] Epoch 13: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 14/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1613 - cls_loss: 0.0920 - reg_loss: 0.1136 - intention_accuracy: 0.8243 - val_loss
: 0.6683 - val_cls_loss: 0.6040 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6901                                                                [Sigma] Epoch 14: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 15/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1483 - cls_loss: 0.0886 - reg_loss: 0.1136 - intention_accuracy: 0.8374 - val_loss
: 0.6601 - val_cls_loss: 0.6048 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                [Sigma] Epoch 15: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 16/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1377 - cls_loss: 0.0859 - reg_loss: 0.1136 - intention_accuracy: 0.8393 - val_loss
: 0.4900 - val_cls_loss: 0.4421 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6942                                                                [Sigma] Epoch 16: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 17/30
1067/1067 [==============================] - 19s 17ms/step - loss: 0.1282 - cls_loss: 0.0833 - reg_loss: 0.1136 - intention_accuracy: 0.8435 - val_loss
: 0.4862 - val_cls_loss: 0.4445 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                [Sigma] Epoch 17: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 18/30
1067/1067 [==============================] - 13s 12ms/step - loss: 0.1117 - cls_loss: 0.0728 - reg_loss: 0.1136 - intention_accuracy: 0.8627 - val_loss
: 0.6122 - val_cls_loss: 0.5756 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7025                                                                [Sigma] Epoch 18: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 19/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.1132 - cls_loss: 0.0786 - reg_loss: 0.1136 - intention_accuracy: 0.8561 - val_loss
: 0.5767 - val_cls_loss: 0.5428 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7355                                                                [Sigma] Epoch 19: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 20/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.0981 - cls_loss: 0.0681 - reg_loss: 0.1136 - intention_accuracy: 0.8768 - val_loss
: 0.7379 - val_cls_loss: 0.7102 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6240                                                                [Sigma] Epoch 20: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 21/30
1067/1067 [==============================] - 13s 12ms/step - loss: 0.0991 - cls_loss: 0.0725 - reg_loss: 0.1136 - intention_accuracy: 0.8739 - val_loss
: 0.6120 - val_cls_loss: 0.5864 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                [Sigma] Epoch 21: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 22/30
1067/1067 [==============================] - 20s 19ms/step - loss: 0.1007 - cls_loss: 0.0775 - reg_loss: 0.1136 - intention_accuracy: 0.8571 - val_loss
: 0.5459 - val_cls_loss: 0.5239 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                [Sigma] Epoch 22: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 23/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.0901 - cls_loss: 0.0700 - reg_loss: 0.1136 - intention_accuracy: 0.8791 - val_loss
: 0.6054 - val_cls_loss: 0.5865 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.7025                                                                [Sigma] Epoch 23: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 24/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0867 - cls_loss: 0.0686 - reg_loss: 0.1136 - intention_accuracy: 0.8744 - val_loss
: 0.7215 - val_cls_loss: 0.7046 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6116                                                                [Sigma] Epoch 24: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 25/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.0915 - cls_loss: 0.0741 - reg_loss: 0.1136 - intention_accuracy: 0.8622 - val_loss
: 0.5405 - val_cls_loss: 0.5241 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                [Sigma] Epoch 25: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 26/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.0828 - cls_loss: 0.0677 - reg_loss: 0.1136 - intention_accuracy: 0.8810 - val_loss
: 0.7439 - val_cls_loss: 0.7300 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6736                                                                [Sigma] Epoch 26: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 27/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0775 - cls_loss: 0.0643 - reg_loss: 0.1136 - intention_accuracy: 0.8918 - val_loss
: 0.7146 - val_cls_loss: 0.7023 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6901                                                                [Sigma] Epoch 27: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 28/30
1067/1067 [==============================] - 19s 18ms/step - loss: 0.0717 - cls_loss: 0.0599 - reg_loss: 0.1136 - intention_accuracy: 0.8903 - val_loss
: 0.6052 - val_cls_loss: 0.5938 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6983                                                                [Sigma] Epoch 28: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 29/30
1067/1067 [==============================] - 14s 13ms/step - loss: 0.0748 - cls_loss: 0.0639 - reg_loss: 0.1136 - intention_accuracy: 0.8908 - val_loss
: 0.5674 - val_cls_loss: 0.5560 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                [Sigma] Epoch 29: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 30/30
1067/1067 [==============================] - 14s 13ms/step - loss: 0.0725 - cls_loss: 0.0623 - reg_loss: 0.1136 - intention_accuracy: 0.8927 - val_loss
: 0.7210 - val_cls_loss: 0.7113 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                [Sigma] Epoch 30: sigma_cls=0.6931 sigma_reg=0.6931

🎯 Training completed!
📁 All epoch models saved in: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/epochs
Train model is saved to data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/model.h5
Available metrics: ['loss', 'cls_loss', 'reg_loss', 'intention_accuracy', 'val_loss', 'val_cls_loss', 'val_reg_loss', 'val_intention_accuracy', 'sigma_
cls', 'val_sigma_cls', 'sigma_reg', 'val_sigma_reg']                                                                                                   Training plots saved to model directory
Wrote configs to data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/configs.yaml
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 6ms/step

======================================================================
🎯 MODEL TEST RESULTS 🎯
======================================================================
Accuracy:   0.5965
AUC:        0.5546
F1-Score:   0.6911
Precision:  0.6633
Recall:     0.7213
======================================================================

Model saved to data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/

✅ 训练完成 (耗时: 9.8 分钟)
🔍 查找最新模型目录...
📁 找到模型目录: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s

🧪 开始测试模型...
2025-11-12 01:07:48.834076: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:48.838236: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:48.838361: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   🚀 开始测试模型目录: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s
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

🔍 测试模型: epoch_001_loss_4.9107_acc_0.5372.h5
2025-11-12 01:07:49.858263: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Li
brary (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 01:07:49.859328: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:49.859474: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:49.859560: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:50.218524: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:50.218671: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:50.218763: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), b
ut there must be at least one NUMA node, so returning NUMA node zero                                                                                   2025-11-12 01:07:50.218844: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4
077 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                              WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
2025-11-12 01:07:51.282863: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registere
d 2)                                                                                                                                                   2025-11-12 01:07:52.874264: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will o
nly be logged once.                                                                                                                                    2025-11-12 01:07:53.464311: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6401, AUC: 0.7146, F1: 0.6858

============================================================
进度: 2/31

🔍 测试模型: epoch_002_loss_2.2539_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6699, AUC: 0.7266, F1: 0.7204

============================================================
进度: 3/31

🔍 测试模型: epoch_003_loss_1.4249_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6539, AUC: 0.7230, F1: 0.6928

============================================================
进度: 4/31

🔍 测试模型: epoch_004_loss_1.0310_acc_0.5909.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6273, AUC: 0.6999, F1: 0.6396

============================================================
进度: 5/31

🔍 测试模型: epoch_005_loss_0.8633_acc_0.5992.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 6ms/step
✅ 准确率: 0.6247, AUC: 0.6988, F1: 0.7090

============================================================
进度: 6/31

🔍 测试模型: epoch_006_loss_0.6993_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6603, AUC: 0.7027, F1: 0.7136

============================================================
进度: 7/31

🔍 测试模型: epoch_007_loss_0.5754_acc_0.5496.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6507, AUC: 0.7172, F1: 0.6834

============================================================
进度: 8/31

🔍 测试模型: epoch_008_loss_0.6390_acc_0.6446.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6417, AUC: 0.7009, F1: 0.7082

============================================================
进度: 9/31

🔍 测试模型: epoch_009_loss_0.5419_acc_0.5661.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6374, AUC: 0.7003, F1: 0.6831

============================================================
进度: 10/31

🔍 测试模型: epoch_010_loss_0.4753_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6449, AUC: 0.7041, F1: 0.6849

============================================================
进度: 11/31

🔍 测试模型: epoch_011_loss_0.4122_acc_0.5620.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 7ms/step
✅ 准确率: 0.6475, AUC: 0.6881, F1: 0.6703

============================================================
进度: 12/31

🔍 测试模型: epoch_012_loss_0.4968_acc_0.6157.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.5848, AUC: 0.6391, F1: 0.6561

============================================================
进度: 13/31

🔍 测试模型: epoch_013_loss_0.6473_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6358, AUC: 0.6858, F1: 0.7164

============================================================
进度: 14/31

🔍 测试模型: epoch_014_loss_0.6683_acc_0.6901.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6087, AUC: 0.6646, F1: 0.7125

============================================================
进度: 15/31

🔍 测试模型: epoch_015_loss_0.6601_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6528, AUC: 0.7019, F1: 0.7323

============================================================
进度: 16/31

🔍 测试模型: epoch_016_loss_0.4900_acc_0.6942.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6188, AUC: 0.6679, F1: 0.7163

============================================================
进度: 17/31

🔍 测试模型: epoch_017_loss_0.4862_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6135, AUC: 0.6535, F1: 0.7134

============================================================
进度: 18/31

🔍 测试模型: epoch_018_loss_0.6122_acc_0.7025.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6162, AUC: 0.6817, F1: 0.7144

============================================================
进度: 19/31

🔍 测试模型: epoch_019_loss_0.5767_acc_0.7355.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6401, AUC: 0.6851, F1: 0.7437

============================================================
进度: 20/31

🔍 测试模型: epoch_020_loss_0.7379_acc_0.6240.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6342, AUC: 0.6671, F1: 0.7092

============================================================
进度: 21/31

🔍 测试模型: epoch_021_loss_0.6120_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 7ms/step
✅ 准确率: 0.6194, AUC: 0.6643, F1: 0.7078

============================================================
进度: 22/31

🔍 测试模型: epoch_022_loss_0.5459_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 17s 9ms/step
✅ 准确率: 0.6055, AUC: 0.6372, F1: 0.7088

============================================================
进度: 23/31

🔍 测试模型: epoch_023_loss_0.6054_acc_0.7025.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.5954, AUC: 0.6526, F1: 0.7007

============================================================
进度: 24/31

🔍 测试模型: epoch_024_loss_0.7215_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6172, AUC: 0.6631, F1: 0.6941

============================================================
进度: 25/31

🔍 测试模型: epoch_025_loss_0.5405_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.5827, AUC: 0.6147, F1: 0.6813

============================================================
进度: 26/31

🔍 测试模型: epoch_026_loss_0.7439_acc_0.6736.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6635, AUC: 0.6923, F1: 0.7473

============================================================
进度: 27/31

🔍 测试模型: epoch_027_loss_0.7146_acc_0.6901.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.5991, AUC: 0.6307, F1: 0.7129

============================================================
进度: 28/31

🔍 测试模型: epoch_028_loss_0.6052_acc_0.6983.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6491, AUC: 0.6897, F1: 0.7270

============================================================
进度: 29/31

🔍 测试模型: epoch_029_loss_0.5674_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6045, AUC: 0.6311, F1: 0.7024

============================================================
进度: 30/31

🔍 测试模型: epoch_030_loss_0.7210_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.5965, AUC: 0.6413, F1: 0.6911

============================================================
进度: 31/31

🔍 测试模型: model.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 15s 7ms/step
✅ 准确率: 0.5965, AUC: 0.6413, F1: 0.6911

📊 结果已保存到: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/test_results_20251112_011518.csv
📝 报告已保存到: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/test_report_20251112_011518.txt

🏆 准确率最高的模型: epoch_002_loss_2.2539_acc_0.6570.h5 (准确率: 0.6699)
🗑️  开始清理epochs目录，将删除 30 个模型文件...
🗑️  已删除: epoch_015_loss_0.6601_acc_0.6777.h5
🗑️  已删除: epoch_024_loss_0.7215_acc_0.6116.h5
🗑️  已删除: epoch_011_loss_0.4122_acc_0.5620.h5
🗑️  已删除: epoch_020_loss_0.7379_acc_0.6240.h5
🗑️  已删除: epoch_013_loss_0.6473_acc_0.6488.h5
🗑️  已删除: epoch_007_loss_0.5754_acc_0.5496.h5
🗑️  已删除: epoch_025_loss_0.5405_acc_0.6529.h5
🗑️  已删除: epoch_009_loss_0.5419_acc_0.5661.h5
🗑️  已删除: epoch_019_loss_0.5767_acc_0.7355.h5
🗑️  已删除: epoch_005_loss_0.8633_acc_0.5992.h5
🗑️  已删除: epoch_022_loss_0.5459_acc_0.6322.h5
🗑️  已删除: epoch_014_loss_0.6683_acc_0.6901.h5
🗑️  已删除: epoch_010_loss_0.4753_acc_0.5868.h5
🗑️  已删除: epoch_008_loss_0.6390_acc_0.6446.h5
🗑️  已删除: epoch_017_loss_0.4862_acc_0.6777.h5
🗑️  已删除: epoch_006_loss_0.6993_acc_0.6529.h5
🗑️  已删除: epoch_029_loss_0.5674_acc_0.6570.h5
🗑️  已删除: epoch_026_loss_0.7439_acc_0.6736.h5
🗑️  已删除: epoch_016_loss_0.4900_acc_0.6942.h5
🗑️  已删除: epoch_018_loss_0.6122_acc_0.7025.h5
🗑️  已删除: epoch_030_loss_0.7210_acc_0.6570.h5
🗑️  已删除: epoch_012_loss_0.4968_acc_0.6157.h5
🗑️  已删除: epoch_028_loss_0.6052_acc_0.6983.h5
🗑️  已删除: epoch_001_loss_4.9107_acc_0.5372.h5
🗑️  已删除: epoch_004_loss_1.0310_acc_0.5909.h5
🗑️  已删除: epoch_003_loss_1.4249_acc_0.6116.h5
🗑️  已删除: epoch_023_loss_0.6054_acc_0.7025.h5
🗑️  已删除: epoch_027_loss_0.7146_acc_0.6901.h5
🗑️  已删除: epoch_021_loss_0.6120_acc_0.6322.h5
📋 已将最佳模型复制到: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s/epoch_002_loss_2.2539_acc_0.6570.h5
🗑️  已删除: epoch_002_loss_2.2539_acc_0.6570.h5
🗑️  已删除空的epochs目录
🔄 模型目录已重命名:
   原目录: 12Nov2025-00h58m06s
   新目录: 12Nov2025-00h58m06s_acc_0.6699

================================================================================
🎯 测试结果汇总
================================================================================
总模型数量: 31
成功测试: 31
失败测试: 0

📊 性能统计:
平均准确率: 0.6269 (±0.0241)
平均AUC: 0.6768 (±0.0305)
平均F1: 0.7023 (±0.0231)

🏆 最佳模型:
最高准确率: epoch_002_loss_2.2539_acc_0.6570.h5 (Acc: 0.6699)
最高AUC: epoch_002_loss_2.2539_acc_0.6570.h5 (AUC: 0.7266)
最高F1: epoch_026_loss_0.7439_acc_0.6736.h5 (F1: 0.7473)

✅ 测试完成 (耗时: 7.6 分钟)

================================================================================
🎉 训练和测试管道完成!
================================================================================
模型目录: data/models/jaad/Transformer_depth/12Nov2025-00h58m06s
总耗时: 17.4 分钟
结束时间: 2025-11-12 01:15:20
================================================================================
