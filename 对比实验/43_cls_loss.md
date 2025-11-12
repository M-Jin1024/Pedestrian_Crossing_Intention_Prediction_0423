 cd /home/minshi/Pedestrian_Crossing_Intention_Prediction ; /usr/bin/env /home/minshi/miniconda3/envs/tf26/bin/python /home/minshi/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 53503 -- /home/minshi/Pedestrian_Crossing_Intention_Prediction/train_and_test_all_epoch_pipeline.py -c config_files/my/my_jaad.yaml 
================================================================================
🎯 训练和测试管道启动
================================================================================
配置文件: config_files/my/my_jaad.yaml
开始时间: 2025-11-12 01:39:07

🚀 开始训练...
2025-11-12 01:39:09.761791: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:09.765090: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:09.765179: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
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
2025-11-12 01:39:11.609987: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (o
neDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 01:39:11.611116: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.611225: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.611284: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.878667: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.878776: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.878843: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:39:11.878906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4028 MB m
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
📁 Models will be saved to: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s
📋 已复制 action_predict.py 到模型目录
2025-11-12 01:39:13.002182: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/30
WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimi
zing the loss.                                                                                                                                                 WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimi
zing the loss.                                                                                                                                                 2025-11-12 01:39:16.651778: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be l
ogged once.                                                                                                                                                    2025-11-12 01:39:16.949666: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1067/1067 [==============================] - 14s 10ms/step - loss: 9.0480 - cls_loss: 0.2132 - reg_loss: 0.3191 - intention_accuracy: 0.5459 - val_loss: 4.9052
 - val_cls_loss: 0.2763 - val_reg_loss: 0.2407 - val_intention_accuracy: 0.2727                                                                                /home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must
 override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.                                                         warnings.warn('Custom mask layers require a config and must override '
[Sigma] Epoch 1: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 2/30
1067/1067 [==============================] - 9s 9ms/step - loss: 3.1847 - cls_loss: 0.1897 - reg_loss: 0.1468 - intention_accuracy: 0.5886 - val_loss: 2.2509 -
 val_cls_loss: 0.2831 - val_reg_loss: 0.1491 - val_intention_accuracy: 0.5537                                                                                  [Sigma] Epoch 2: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 3/30
1067/1067 [==============================] - 9s 9ms/step - loss: 1.6895 - cls_loss: 0.1673 - reg_loss: 0.1122 - intention_accuracy: 0.6495 - val_loss: 1.4472 -
 val_cls_loss: 0.2554 - val_reg_loss: 0.1406 - val_intention_accuracy: 0.6446                                                                                  [Sigma] Epoch 3: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 4/30
1067/1067 [==============================] - 9s 9ms/step - loss: 1.1349 - cls_loss: 0.1584 - reg_loss: 0.1114 - intention_accuracy: 0.6692 - val_loss: 1.1317 -
 val_cls_loss: 0.3388 - val_reg_loss: 0.1429 - val_intention_accuracy: 0.6033                                                                                  [Sigma] Epoch 4: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 5/30
1067/1067 [==============================] - 11s 10ms/step - loss: 0.8150 - cls_loss: 0.1566 - reg_loss: 0.1132 - intention_accuracy: 0.6701 - val_loss: 0.8288
 - val_cls_loss: 0.2863 - val_reg_loss: 0.1445 - val_intention_accuracy: 0.5496                                                                                [Sigma] Epoch 5: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 6/30
1067/1067 [==============================] - 19s 18ms/step - loss: 0.6017 - cls_loss: 0.1464 - reg_loss: 0.1136 - intention_accuracy: 0.6828 - val_loss: 0.7000
 - val_cls_loss: 0.3206 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6033                                                                                [Sigma] Epoch 6: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 7/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.4641 - cls_loss: 0.1413 - reg_loss: 0.1136 - intention_accuracy: 0.6954 - val_loss: 0.6437
 - val_cls_loss: 0.3707 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5950                                                                                [Sigma] Epoch 7: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 8/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.3691 - cls_loss: 0.1336 - reg_loss: 0.1136 - intention_accuracy: 0.6992 - val_loss: 0.5347
 - val_cls_loss: 0.3322 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5455                                                                                [Sigma] Epoch 8: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 9/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.3032 - cls_loss: 0.1252 - reg_loss: 0.1136 - intention_accuracy: 0.7268 - val_loss: 0.5509
 - val_cls_loss: 0.3947 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5661                                                                                [Sigma] Epoch 9: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 10/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.2572 - cls_loss: 0.1172 - reg_loss: 0.1136 - intention_accuracy: 0.7352 - val_loss: 0.6550
 - val_cls_loss: 0.5301 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                                [Sigma] Epoch 10: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 11/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.2239 - cls_loss: 0.1104 - reg_loss: 0.1136 - intention_accuracy: 0.7784 - val_loss: 0.6230
 - val_cls_loss: 0.5200 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5083                                                                                [Sigma] Epoch 11: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 12/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1985 - cls_loss: 0.1045 - reg_loss: 0.1136 - intention_accuracy: 0.7873 - val_loss: 0.6427
 - val_cls_loss: 0.5566 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5372                                                                                [Sigma] Epoch 12: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 13/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1795 - cls_loss: 0.0993 - reg_loss: 0.1136 - intention_accuracy: 0.7980 - val_loss: 0.6381
 - val_cls_loss: 0.5646 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6198                                                                                [Sigma] Epoch 13: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 14/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1668 - cls_loss: 0.0993 - reg_loss: 0.1136 - intention_accuracy: 0.8112 - val_loss: 0.5991
 - val_cls_loss: 0.5372 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6405                                                                                [Sigma] Epoch 14: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 15/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1498 - cls_loss: 0.0915 - reg_loss: 0.1136 - intention_accuracy: 0.8238 - val_loss: 0.6837
 - val_cls_loss: 0.6302 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5992                                                                                [Sigma] Epoch 15: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 16/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.1396 - cls_loss: 0.0896 - reg_loss: 0.1136 - intention_accuracy: 0.8271 - val_loss: 0.6231
 - val_cls_loss: 0.5770 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6281                                                                                [Sigma] Epoch 16: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 17/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1332 - cls_loss: 0.0896 - reg_loss: 0.1136 - intention_accuracy: 0.8215 - val_loss: 0.8860
 - val_cls_loss: 0.8454 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6116                                                                                [Sigma] Epoch 17: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 18/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1258 - cls_loss: 0.0873 - reg_loss: 0.1136 - intention_accuracy: 0.8421 - val_loss: 0.6119
 - val_cls_loss: 0.5757 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                [Sigma] Epoch 18: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 19/30
1067/1067 [==============================] - 17s 16ms/step - loss: 0.1181 - cls_loss: 0.0842 - reg_loss: 0.1136 - intention_accuracy: 0.8421 - val_loss: 0.5542
 - val_cls_loss: 0.5232 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6157                                                                                [Sigma] Epoch 19: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 20/30
1067/1067 [==============================] - 12s 11ms/step - loss: 0.1103 - cls_loss: 0.0814 - reg_loss: 0.1136 - intention_accuracy: 0.8472 - val_loss: 0.6364
 - val_cls_loss: 0.6094 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5826                                                                                [Sigma] Epoch 20: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 21/30
1067/1067 [==============================] - 12s 12ms/step - loss: 0.1031 - cls_loss: 0.0772 - reg_loss: 0.1136 - intention_accuracy: 0.8529 - val_loss: 0.7091
 - val_cls_loss: 0.6850 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                [Sigma] Epoch 21: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 22/30
1067/1067 [==============================] - 13s 13ms/step - loss: 0.0983 - cls_loss: 0.0749 - reg_loss: 0.1136 - intention_accuracy: 0.8604 - val_loss: 0.6726
 - val_cls_loss: 0.6497 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5661                                                                                [Sigma] Epoch 22: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 23/30
1067/1067 [==============================] - 20s 19ms/step - loss: 0.0945 - cls_loss: 0.0736 - reg_loss: 0.1136 - intention_accuracy: 0.8590 - val_loss: 0.7973
 - val_cls_loss: 0.7779 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6446                                                                                [Sigma] Epoch 23: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 24/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0975 - cls_loss: 0.0790 - reg_loss: 0.1136 - intention_accuracy: 0.8430 - val_loss: 0.4410
 - val_cls_loss: 0.4218 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                [Sigma] Epoch 24: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 25/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0924 - cls_loss: 0.0753 - reg_loss: 0.1136 - intention_accuracy: 0.8561 - val_loss: 0.6770
 - val_cls_loss: 0.6615 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5537                                                                                [Sigma] Epoch 25: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 26/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0841 - cls_loss: 0.0691 - reg_loss: 0.1136 - intention_accuracy: 0.8739 - val_loss: 0.5992
 - val_cls_loss: 0.5848 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5620                                                                                [Sigma] Epoch 26: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 27/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0933 - cls_loss: 0.0778 - reg_loss: 0.1136 - intention_accuracy: 0.8515 - val_loss: 0.6924
 - val_cls_loss: 0.6778 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5579                                                                                [Sigma] Epoch 27: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 28/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.0857 - cls_loss: 0.0728 - reg_loss: 0.1136 - intention_accuracy: 0.8641 - val_loss: 0.6603
 - val_cls_loss: 0.6484 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                                [Sigma] Epoch 28: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 29/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0884 - cls_loss: 0.0770 - reg_loss: 0.1136 - intention_accuracy: 0.8594 - val_loss: 0.6196
 - val_cls_loss: 0.6076 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5992                                                                                [Sigma] Epoch 29: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 30/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0756 - cls_loss: 0.0650 - reg_loss: 0.1136 - intention_accuracy: 0.8828 - val_loss: 0.6407
 - val_cls_loss: 0.6308 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                [Sigma] Epoch 30: sigma_cls=0.6931 sigma_reg=0.6931

🎯 Training completed!
📁 All epoch models saved in: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/epochs
Train model is saved to data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/model.h5
Available metrics: ['loss', 'cls_loss', 'reg_loss', 'intention_accuracy', 'val_loss', 'val_cls_loss', 'val_reg_loss', 'val_intention_accuracy', 'sigma_cls', 'v
al_sigma_cls', 'sigma_reg', 'val_sigma_reg']                                                                                                                   Training plots saved to model directory
Wrote configs to data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/configs.yaml
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step

======================================================================
🎯 MODEL TEST RESULTS 🎯
======================================================================
Accuracy:   0.6167
AUC:        0.5927
F1-Score:   0.6920
Precision:  0.6959
Recall:     0.6882
======================================================================

Model saved to data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/

✅ 训练完成 (耗时: 10.2 分钟)
🔍 查找最新模型目录...
📁 找到模型目录: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s

🧪 开始测试模型...
2025-11-12 01:49:23.087786: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:23.092171: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:23.092299: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   🚀 开始测试模型目录: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s
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

🔍 测试模型: epoch_001_loss_4.9052_acc_0.2727.h5
2025-11-12 01:49:24.169712: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (o
neDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                     To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 01:49:24.171161: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.171312: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.171403: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.559413: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.559592: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.559716: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there
 must be at least one NUMA node, so returning NUMA node zero                                                                                                   2025-11-12 01:49:24.559829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4108 MB m
emory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                              WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
2025-11-12 01:49:25.713214: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2025-11-12 01:49:26.978052: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be l
ogged once.                                                                                                                                                    2025-11-12 01:49:27.373624: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1881/1881 [==============================] - 12s 5ms/step
✅ 准确率: 0.3775, AUC: 0.6671, F1: 0.0101

============================================================
进度: 2/31

🔍 测试模型: epoch_002_loss_2.2509_acc_0.5537.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6629, AUC: 0.7244, F1: 0.7073

============================================================
进度: 3/31

🔍 测试模型: epoch_003_loss_1.4472_acc_0.6446.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6587, AUC: 0.6877, F1: 0.7149

============================================================
进度: 4/31

🔍 测试模型: epoch_004_loss_1.1317_acc_0.6033.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6502, AUC: 0.7136, F1: 0.6940

============================================================
进度: 5/31

🔍 测试模型: epoch_005_loss_0.8288_acc_0.5496.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6135, AUC: 0.7055, F1: 0.6160

============================================================
进度: 6/31

🔍 测试模型: epoch_006_loss_0.7000_acc_0.6033.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 16s 8ms/step
✅ 准确率: 0.6263, AUC: 0.6747, F1: 0.7246

============================================================
进度: 7/31

🔍 测试模型: epoch_007_loss_0.6437_acc_0.5950.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6518, AUC: 0.6981, F1: 0.6935

============================================================
进度: 8/31

🔍 测试模型: epoch_008_loss_0.5347_acc_0.5455.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6411, AUC: 0.6968, F1: 0.6715

============================================================
进度: 9/31

🔍 测试模型: epoch_009_loss_0.5509_acc_0.5661.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6502, AUC: 0.6966, F1: 0.7114

============================================================
进度: 10/31

🔍 测试模型: epoch_010_loss_0.6550_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6241, AUC: 0.6892, F1: 0.7048

============================================================
进度: 11/31

🔍 测试模型: epoch_011_loss_0.6230_acc_0.5083.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6411, AUC: 0.6990, F1: 0.6991

============================================================
进度: 12/31

🔍 测试模型: epoch_012_loss_0.6427_acc_0.5372.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6326, AUC: 0.6683, F1: 0.6757

============================================================
进度: 13/31

🔍 测试模型: epoch_013_loss_0.6381_acc_0.6198.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6023, AUC: 0.6386, F1: 0.6725

============================================================
进度: 14/31

🔍 测试模型: epoch_014_loss_0.5991_acc_0.6405.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6151, AUC: 0.6495, F1: 0.6917

============================================================
进度: 15/31

🔍 测试模型: epoch_015_loss_0.6837_acc_0.5992.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 15s 8ms/step
✅ 准确率: 0.5949, AUC: 0.6330, F1: 0.6887

============================================================
进度: 16/31

🔍 测试模型: epoch_016_loss_0.6231_acc_0.6281.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6268, AUC: 0.6559, F1: 0.7028

============================================================
进度: 17/31

🔍 测试模型: epoch_017_loss_0.8860_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6002, AUC: 0.6586, F1: 0.7158

============================================================
进度: 18/31

🔍 测试模型: epoch_018_loss_0.6119_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6316, AUC: 0.6616, F1: 0.7032

============================================================
进度: 19/31

🔍 测试模型: epoch_019_loss_0.5542_acc_0.6157.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6188, AUC: 0.6659, F1: 0.7048

============================================================
进度: 20/31

🔍 测试模型: epoch_020_loss_0.6364_acc_0.5826.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 5ms/step
✅ 准确率: 0.6310, AUC: 0.6696, F1: 0.7024

============================================================
进度: 21/31

🔍 测试模型: epoch_021_loss_0.7091_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.5949, AUC: 0.6432, F1: 0.6937

============================================================
进度: 22/31

🔍 测试模型: epoch_022_loss_0.6726_acc_0.5661.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6385, AUC: 0.6711, F1: 0.7082

============================================================
进度: 23/31

🔍 测试模型: epoch_023_loss_0.7973_acc_0.6446.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 9s 5ms/step
✅ 准确率: 0.6061, AUC: 0.6501, F1: 0.7184

============================================================
进度: 24/31

🔍 测试模型: epoch_024_loss_0.4410_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 10s 5ms/step
✅ 准确率: 0.6151, AUC: 0.6617, F1: 0.6700

============================================================
进度: 25/31

🔍 测试模型: epoch_025_loss_0.6770_acc_0.5537.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6135, AUC: 0.6552, F1: 0.7002

============================================================
进度: 26/31

🔍 测试模型: epoch_026_loss_0.5992_acc_0.5620.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 17s 9ms/step
✅ 准确率: 0.6029, AUC: 0.6344, F1: 0.6955

============================================================
进度: 27/31

🔍 测试模型: epoch_027_loss_0.6924_acc_0.5579.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6369, AUC: 0.6880, F1: 0.7100

============================================================
进度: 28/31

🔍 测试模型: epoch_028_loss_0.6603_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6103, AUC: 0.6715, F1: 0.6985

============================================================
进度: 29/31

🔍 测试模型: epoch_029_loss_0.6196_acc_0.5992.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.5901, AUC: 0.6489, F1: 0.6975

============================================================
进度: 30/31

🔍 测试模型: epoch_030_loss_0.6407_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 9ms/step
✅ 准确率: 0.6167, AUC: 0.6583, F1: 0.6920

============================================================
进度: 31/31

🔍 测试模型: model.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 16s 8ms/step
✅ 准确率: 0.6167, AUC: 0.6583, F1: 0.6920

📊 结果已保存到: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/test_results_20251112_015713.csv
📝 报告已保存到: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/test_report_20251112_015713.txt

🏆 准确率最高的模型: epoch_002_loss_2.2509_acc_0.5537.h5 (准确率: 0.6629)
🗑️  开始清理epochs目录，将删除 30 个模型文件...
🗑️  已删除: epoch_024_loss_0.4410_acc_0.5702.h5
🗑️  已删除: epoch_027_loss_0.6924_acc_0.5579.h5
🗑️  已删除: epoch_016_loss_0.6231_acc_0.6281.h5
🗑️  已删除: epoch_026_loss_0.5992_acc_0.5620.h5
🗑️  已删除: epoch_011_loss_0.6230_acc_0.5083.h5
🗑️  已删除: epoch_010_loss_0.6550_acc_0.6529.h5
🗑️  已删除: epoch_028_loss_0.6603_acc_0.5868.h5
🗑️  已删除: epoch_013_loss_0.6381_acc_0.6198.h5
🗑️  已删除: epoch_020_loss_0.6364_acc_0.5826.h5
🗑️  已删除: epoch_005_loss_0.8288_acc_0.5496.h5
🗑️  已删除: epoch_001_loss_4.9052_acc_0.2727.h5
🗑️  已删除: epoch_009_loss_0.5509_acc_0.5661.h5
📋 已将最佳模型复制到: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s/epoch_002_loss_2.2509_acc_0.5537.h5
🗑️  已删除: epoch_002_loss_2.2509_acc_0.5537.h5
🗑️  已删除: epoch_012_loss_0.6427_acc_0.5372.h5
🗑️  已删除: epoch_007_loss_0.6437_acc_0.5950.h5
🗑️  已删除: epoch_023_loss_0.7973_acc_0.6446.h5
🗑️  已删除: epoch_019_loss_0.5542_acc_0.6157.h5
🗑️  已删除: epoch_014_loss_0.5991_acc_0.6405.h5
🗑️  已删除: epoch_030_loss_0.6407_acc_0.5702.h5
🗑️  已删除: epoch_008_loss_0.5347_acc_0.5455.h5
🗑️  已删除: epoch_025_loss_0.6770_acc_0.5537.h5
🗑️  已删除: epoch_015_loss_0.6837_acc_0.5992.h5
🗑️  已删除: epoch_021_loss_0.7091_acc_0.5702.h5
🗑️  已删除: epoch_017_loss_0.8860_acc_0.6116.h5
🗑️  已删除: epoch_029_loss_0.6196_acc_0.5992.h5
🗑️  已删除: epoch_004_loss_1.1317_acc_0.6033.h5
🗑️  已删除: epoch_018_loss_0.6119_acc_0.5702.h5
🗑️  已删除: epoch_022_loss_0.6726_acc_0.5661.h5
🗑️  已删除: epoch_006_loss_0.7000_acc_0.6033.h5
🗑️  已删除: epoch_003_loss_1.4472_acc_0.6446.h5
🗑️  已删除空的epochs目录
🔄 模型目录已重命名:
   原目录: 12Nov2025-01h39m11s
   新目录: 12Nov2025-01h39m11s_acc_0.6629

================================================================================
🎯 测试结果汇总
================================================================================
总模型数量: 31
成功测试: 31
失败测试: 0

📊 性能统计:
平均准确率: 0.6159 (±0.0483)
平均AUC: 0.6708 (±0.0237)
平均F1: 0.6736 (±0.1247)

🏆 最佳模型:
最高准确率: epoch_002_loss_2.2509_acc_0.5537.h5 (Acc: 0.6629)
最高AUC: epoch_002_loss_2.2509_acc_0.5537.h5 (AUC: 0.7244)
最高F1: epoch_006_loss_0.7000_acc_0.6033.h5 (F1: 0.7246)

✅ 测试完成 (耗时: 7.9 分钟)

================================================================================
🎉 训练和测试管道完成!
================================================================================
模型目录: data/models/jaad/Transformer_depth/12Nov2025-01h39m11s
总耗时: 18.1 分钟
结束时间: 2025-11-12 01:57:14
================================================================================
