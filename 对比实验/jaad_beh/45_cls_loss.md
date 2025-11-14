 cd /home/minshi/Pedestrian_Crossing_Intention_Prediction ; /usr/bin/env /home/minshi/miniconda3/envs/tf26/bin/python /home/minshi/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 49695 -- /home/minshi/Pedestrian_Crossing_Intention_Prediction/train_and_test_all_epoch_pipeline.py -c config_files/my/my_jaad.yaml 
================================================================================
🎯 训练和测试管道启动
================================================================================
配置文件: config_files/my/my_jaad.yaml
开始时间: 2025-11-12 03:14:09

🚀 开始训练...
2025-11-12 03:14:12.296684: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:12.300186: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:12.300296: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             config_files/my/my_jaad.yaml
model_opts {'model': 'Transformer_depth', 'obs_input_type': ['box', 'depth', 'vehicle_speed', 'ped_speed'], 'enlarge_ratio': 1.5, 'obs_length': 16, 'time_to_event':
 [30, 60], 'overlap': 0.8, 'balance_data': False, 'apply_class_weights': True, 'dataset': 'jaad', 'normalize_boxes': True, 'generator': True, 'fusion_point': 'early', 'fusion_method': 'sum'}                                                                                                                                          data_opts {'fstride': 1, 'sample_type': 'beh', 'subset': 'default', 'data_split_type': 'default', 'seq_type': 'crossing', 'min_track_size': 76}
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
2025-11-12 03:14:14.170293: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN
) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                               To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 03:14:14.171143: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.171264: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.171324: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.462879: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.462994: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.463068: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:14:14.463135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3786 MB memory
:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                                        
============================================================
📊 MODEL PARAMETER STATISTICS
============================================================
Total parameters:        2,968,717
Trainable parameters:    2,968,711.0
Non-trainable parameters: 6.0
============================================================

/home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use `learning_ra
te` instead.                                                                                                                                                          warnings.warn(

🚀 Training started!
📁 Models will be saved to: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s
📋 已复制 action_predict.py 到模型目录
2025-11-12 03:14:15.645817: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/30
WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimizing 
the loss.                                                                                                                                                           WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimizing 
the loss.                                                                                                                                                           2025-11-12 03:14:20.807537: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be logged
 once.                                                                                                                                                              2025-11-12 03:14:21.200926: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1067/1067 [==============================] - 21s 14ms/step - loss: 9.0579 - cls_loss: 0.2122 - reg_loss: 0.3200 - intention_accuracy: 0.5347 - val_loss: 4.8977 - va
l_cls_loss: 0.2868 - val_reg_loss: 0.2218 - val_intention_accuracy: 0.5331                                                                                          /home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must over
ride get_config. When loading, the custom mask layer must be passed to the custom_objects argument.                                                                   warnings.warn('Custom mask layers require a config and must override '
[Sigma] Epoch 1: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 2/30
1067/1067 [==============================] - 16s 15ms/step - loss: 3.1386 - cls_loss: 0.1812 - reg_loss: 0.1367 - intention_accuracy: 0.5951 - val_loss: 2.2767 - va
l_cls_loss: 0.3604 - val_reg_loss: 0.1311 - val_intention_accuracy: 0.7107                                                                                          [Sigma] Epoch 2: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 3/30
1067/1067 [==============================] - 22s 20ms/step - loss: 1.6391 - cls_loss: 0.1705 - reg_loss: 0.0991 - intention_accuracy: 0.6331 - val_loss: 1.4489 - va
l_cls_loss: 0.3090 - val_reg_loss: 0.1305 - val_intention_accuracy: 0.6694                                                                                          [Sigma] Epoch 3: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 4/30
1067/1067 [==============================] - 23s 21ms/step - loss: 1.0949 - cls_loss: 0.1631 - reg_loss: 0.1084 - intention_accuracy: 0.6471 - val_loss: 1.0644 - va
l_cls_loss: 0.3080 - val_reg_loss: 0.1429 - val_intention_accuracy: 0.5620                                                                                          [Sigma] Epoch 4: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 5/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.7815 - cls_loss: 0.1533 - reg_loss: 0.1132 - intention_accuracy: 0.6659 - val_loss: 0.8502 - va
l_cls_loss: 0.3327 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                                          [Sigma] Epoch 5: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 6/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.5855 - cls_loss: 0.1497 - reg_loss: 0.1136 - intention_accuracy: 0.6982 - val_loss: 0.7675 - va
l_cls_loss: 0.4019 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6157                                                                                          [Sigma] Epoch 6: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 7/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.4564 - cls_loss: 0.1428 - reg_loss: 0.1136 - intention_accuracy: 0.6884 - val_loss: 0.6486 - va
l_cls_loss: 0.3805 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5496                                                                                          [Sigma] Epoch 7: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 8/30
1067/1067 [==============================] - 24s 23ms/step - loss: 0.3712 - cls_loss: 0.1369 - reg_loss: 0.1136 - intention_accuracy: 0.7085 - val_loss: 0.6156 - va
l_cls_loss: 0.4106 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5992                                                                                          [Sigma] Epoch 8: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 9/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.3158 - cls_loss: 0.1328 - reg_loss: 0.1136 - intention_accuracy: 0.7085 - val_loss: 0.5979 - va
l_cls_loss: 0.4351 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                                          [Sigma] Epoch 9: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 10/30
1067/1067 [==============================] - 24s 23ms/step - loss: 0.2752 - cls_loss: 0.1280 - reg_loss: 0.1136 - intention_accuracy: 0.7216 - val_loss: 0.5311 - va
l_cls_loss: 0.3982 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5372                                                                                          [Sigma] Epoch 10: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 11/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.2462 - cls_loss: 0.1251 - reg_loss: 0.1136 - intention_accuracy: 0.7170 - val_loss: 0.5995 - va
l_cls_loss: 0.4880 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6446                                                                                          [Sigma] Epoch 11: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 12/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.2168 - cls_loss: 0.1158 - reg_loss: 0.1136 - intention_accuracy: 0.7409 - val_loss: 0.5671 - va
l_cls_loss: 0.4744 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5537                                                                                          [Sigma] Epoch 12: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 13/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.1987 - cls_loss: 0.1138 - reg_loss: 0.1136 - intention_accuracy: 0.7366 - val_loss: 0.6946 - va
l_cls_loss: 0.6173 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5785                                                                                          [Sigma] Epoch 13: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 14/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.1820 - cls_loss: 0.1100 - reg_loss: 0.1136 - intention_accuracy: 0.7455 - val_loss: 0.5581 - va
l_cls_loss: 0.4924 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5537                                                                                          [Sigma] Epoch 14: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 15/30
1067/1067 [==============================] - 25s 23ms/step - loss: 0.1696 - cls_loss: 0.1083 - reg_loss: 0.1136 - intention_accuracy: 0.7662 - val_loss: 0.5712 - va
l_cls_loss: 0.5150 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5785                                                                                          [Sigma] Epoch 15: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 16/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.1502 - cls_loss: 0.0978 - reg_loss: 0.1136 - intention_accuracy: 0.8055 - val_loss: 0.6452 - va
l_cls_loss: 0.5967 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                                          [Sigma] Epoch 16: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 17/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1437 - cls_loss: 0.0979 - reg_loss: 0.1136 - intention_accuracy: 0.8229 - val_loss: 0.5990 - va
l_cls_loss: 0.5565 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                                          [Sigma] Epoch 17: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 18/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1346 - cls_loss: 0.0942 - reg_loss: 0.1136 - intention_accuracy: 0.8294 - val_loss: 0.5552 - va
l_cls_loss: 0.5167 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                          [Sigma] Epoch 18: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 19/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1242 - cls_loss: 0.0885 - reg_loss: 0.1136 - intention_accuracy: 0.8430 - val_loss: 0.6726 - va
l_cls_loss: 0.6395 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5579                                                                                          [Sigma] Epoch 19: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 20/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1133 - cls_loss: 0.0819 - reg_loss: 0.1136 - intention_accuracy: 0.8435 - val_loss: 0.6861 - va
l_cls_loss: 0.6552 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                                          [Sigma] Epoch 20: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 21/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.1128 - cls_loss: 0.0846 - reg_loss: 0.1136 - intention_accuracy: 0.8440 - val_loss: 0.6055 - va
l_cls_loss: 0.5790 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6983                                                                                          [Sigma] Epoch 21: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 22/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1071 - cls_loss: 0.0818 - reg_loss: 0.1136 - intention_accuracy: 0.8486 - val_loss: 0.4743 - va
l_cls_loss: 0.4503 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6405                                                                                          [Sigma] Epoch 22: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 23/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.1018 - cls_loss: 0.0789 - reg_loss: 0.1136 - intention_accuracy: 0.8613 - val_loss: 0.5285 - va
l_cls_loss: 0.5072 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6942                                                                                          [Sigma] Epoch 23: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 24/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.0973 - cls_loss: 0.0770 - reg_loss: 0.1136 - intention_accuracy: 0.8547 - val_loss: 0.5723 - va
l_cls_loss: 0.5531 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6116                                                                                          [Sigma] Epoch 24: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 25/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.0948 - cls_loss: 0.0763 - reg_loss: 0.1136 - intention_accuracy: 0.8561 - val_loss: 0.6002 - va
l_cls_loss: 0.5827 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6446                                                                                          [Sigma] Epoch 25: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 26/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0915 - cls_loss: 0.0741 - reg_loss: 0.1136 - intention_accuracy: 0.8561 - val_loss: 0.6616 - va
l_cls_loss: 0.6452 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5702                                                                                          [Sigma] Epoch 26: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 27/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0925 - cls_loss: 0.0772 - reg_loss: 0.1136 - intention_accuracy: 0.8552 - val_loss: 0.6174 - va
l_cls_loss: 0.6029 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                                          [Sigma] Epoch 27: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 28/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0865 - cls_loss: 0.0725 - reg_loss: 0.1136 - intention_accuracy: 0.8664 - val_loss: 0.5427 - va
l_cls_loss: 0.5290 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6694                                                                                          [Sigma] Epoch 28: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 29/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0924 - cls_loss: 0.0792 - reg_loss: 0.1136 - intention_accuracy: 0.8505 - val_loss: 0.6122 - va
l_cls_loss: 0.5991 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6488                                                                                          [Sigma] Epoch 29: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 30/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.0779 - cls_loss: 0.0660 - reg_loss: 0.1136 - intention_accuracy: 0.8805 - val_loss: 0.8285 - va
l_cls_loss: 0.8170 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                                          [Sigma] Epoch 30: sigma_cls=0.6931 sigma_reg=0.6931

🎯 Training completed!
📁 All epoch models saved in: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/epochs
Train model is saved to data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/model.h5
Available metrics: ['loss', 'cls_loss', 'reg_loss', 'intention_accuracy', 'val_loss', 'val_cls_loss', 'val_reg_loss', 'val_intention_accuracy', 'sigma_cls', 'val_si
gma_cls', 'sigma_reg', 'val_sigma_reg']                                                                                                                             Training plots saved to model directory
Wrote configs to data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/configs.yaml
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step

======================================================================
🎯 MODEL TEST RESULTS 🎯
======================================================================
Accuracy:   0.6279
AUC:        0.5782
F1-Score:   0.7229
Precision:  0.6768
Recall:     0.7757
======================================================================

Model saved to data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/

✅ 训练完成 (耗时: 12.2 分钟)
🔍 查找最新模型目录...
📁 找到模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s

🧪 开始测试模型...
2025-11-12 03:26:27.231765: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:27.238323: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:27.238529: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             🚀 开始测试模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s
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

🔍 测试模型: epoch_001_loss_4.8977_acc_0.5331.h5
2025-11-12 03:26:28.766340: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN
) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                               To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 03:26:28.769155: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:28.769379: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:28.769514: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:29.329173: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:29.329387: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:29.329531: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:26:29.329655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3862 MB memory
:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                                        WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
2025-11-12 03:26:31.025010: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2025-11-12 03:26:32.933109: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be logged
 once.                                                                                                                                                              2025-11-12 03:26:33.543331: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1881/1881 [==============================] - 22s 10ms/step
✅ 准确率: 0.6401, AUC: 0.7119, F1: 0.7000

============================================================
进度: 2/31

🔍 测试模型: epoch_002_loss_2.2767_acc_0.7107.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6624, AUC: 0.7237, F1: 0.7216

============================================================
进度: 3/31

🔍 测试模型: epoch_003_loss_1.4489_acc_0.6694.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6619, AUC: 0.7060, F1: 0.7064

============================================================
进度: 4/31

🔍 测试模型: epoch_004_loss_1.0644_acc_0.5620.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6417, AUC: 0.7080, F1: 0.6630

============================================================
进度: 5/31

🔍 测试模型: epoch_005_loss_0.8502_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6704, AUC: 0.7079, F1: 0.7316

============================================================
进度: 6/31

🔍 测试模型: epoch_006_loss_0.7675_acc_0.6157.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6486, AUC: 0.6987, F1: 0.6983

============================================================
进度: 7/31

🔍 测试模型: epoch_007_loss_0.6486_acc_0.5496.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6422, AUC: 0.7072, F1: 0.6660

============================================================
进度: 8/31

🔍 测试模型: epoch_008_loss_0.6156_acc_0.5992.h5                                                                                                        -1.4830 - cl
s_loss: 0.0761 - reg_loss: 0.0013 - inten1011/1067 [===========================>..] - ETA: 1s - loss: -1.4832 - cls_loss: 0.0760 - reg_loss: 0.0013 - inten1013/1067 [===========================>..] - ETA: 1s - loss: -1.4838 - cls_loss: 0.0759 - reg_loss: 0.0013 - inten1016/1067 [===========================>..] - ETA: 1s - loss: -1.4847 - cls_loss: 0.0757 - reg_loss: 0.0013 - inten1020/1067 [===========================>..] - ETA: 0s - loss: -1.4840 - cls_loss: 0.0759 - reg_loss: 0.0013 - inten1024/1067 [===========================>..] - ETA: 0s - loss: -1.4840 - cls_loss: 0.0759 - reg_loss: 0.0013 - inten1026/1067 [===========================>..] - ETA: 0s - loss: -1.4834 - cls_loss: 0.0761 - reg_loss: 0.0013 - inten1029/1067 [===========================>..] - ETA: 0s - loss: -1.4837 - cls_loss: 0.0760 - reg_loss: 0.0013 - inten1033/1067 [============================>.] - ETA: 0s - loss: -1.4845 - cls_loss: 0.0759 - reg_loss: 0.0013 - inten1036/1067 [============================>.] - ETA: 0s - loss: -1.4855 - cls_loss: 0.0757 - reg_loss: 0.0013 - inten1039/1067 [============================>.] - ETA: 0s - loss: -1.4858 - cls_loss: 0.0756 - reg_loss: 0.0013 - inten1042/1067 [============================>.] - ETA: 0s - loss: -1.4861 - cls_loss: 0.0756 - reg_loss: 0.0013 - inten1045/1067 [============================>.] - ETA: 0s - loss: -1.4871 - cls_loss: 0.0753 - reg_loss: 0.0013 - inten1048/1067 [============================>.] - ETA: 0s - loss: -1.4877 - cls_loss: 0.0752 - reg_loss: 0.0013 - inten1051/1067 [============================>.] - ETA: 0s - loss: -1.4885 - cls_loss: 0.0750 - reg_loss: 0.0012 - inten1053/1067 [1881/1881 [==============================] - 21s 10ms/step- cls_loss: 0.0749 - reg_loss: 0.0012 - inten1056/1067 [============================>.] - ETA     ✅ 准确率: 0.6374, AUC: 0.6921, F1: 0.7024

============================================================
进度: 10/31

🔍 测试模型: epoch_010_loss_0.5311_acc_0.5372.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step
✅ 准确率: 0.6326, AUC: 0.6975, F1: 0.6536

============================================================
进度: 11/31

🔍 测试模型: epoch_011_loss_0.5995_acc_0.6446.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6762, AUC: 0.6998, F1: 0.7596

============================================================
进度: 12/31

🔍 测试模型: epoch_012_loss_0.5671_acc_0.5537.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6353, AUC: 0.6842, F1: 0.6650

============================================================
进度: 13/31

🔍 测试模型: epoch_013_loss_0.6946_acc_0.5785.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6279, AUC: 0.6724, F1: 0.6673

============================================================
进度: 14/31

🔍 测试模型: epoch_014_loss_0.5581_acc_0.5537.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step
✅ 准确率: 0.6374, AUC: 0.6757, F1: 0.6780

============================================================
进度: 15/31

🔍 测试模型: epoch_015_loss_0.5712_acc_0.5785.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6486, AUC: 0.6805, F1: 0.7219

============================================================
进度: 16/31

🔍 测试模型: epoch_016_loss_0.6452_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6550, AUC: 0.6755, F1: 0.7434

============================================================
进度: 17/31

🔍 测试模型: epoch_017_loss_0.5990_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6427, AUC: 0.6647, F1: 0.7155

============================================================
进度: 18/31

🔍 测试模型: epoch_018_loss_0.5552_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6497, AUC: 0.6624, F1: 0.7108

============================================================
进度: 19/31

🔍 测试模型: epoch_019_loss_0.6726_acc_0.5579.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6257, AUC: 0.6460, F1: 0.6994

============================================================
进度: 20/31

🔍 测试模型: epoch_020_loss_0.6861_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.5805, AUC: 0.6135, F1: 0.7039

============================================================
进度: 21/31

🔍 测试模型: epoch_021_loss_0.6055_acc_0.6983.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6279, AUC: 0.6507, F1: 0.7173

============================================================
进度: 22/31

🔍 测试模型: epoch_022_loss_0.4743_acc_0.6405.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6252, AUC: 0.6407, F1: 0.7027

============================================================
进度: 23/31

🔍 测试模型: epoch_023_loss_0.5285_acc_0.6942.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6300, AUC: 0.6440, F1: 0.7180

============================================================
进度: 24/31

🔍 测试模型: epoch_024_loss_0.5723_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6396, AUC: 0.6575, F1: 0.6924

============================================================
进度: 25/31

🔍 测试模型: epoch_025_loss_0.6002_acc_0.6446.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6241, AUC: 0.6375, F1: 0.7068

============================================================
进度: 26/31

🔍 测试模型: epoch_026_loss_0.6616_acc_0.5702.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.5991, AUC: 0.6285, F1: 0.6998

============================================================
进度: 27/31

🔍 测试模型: epoch_027_loss_0.6174_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6390, AUC: 0.6488, F1: 0.7317

============================================================
进度: 28/31

🔍 测试模型: epoch_028_loss_0.5427_acc_0.6694.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6619, AUC: 0.6706, F1: 0.7393

============================================================
进度: 29/31

🔍 测试模型: epoch_029_loss_0.6122_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6215, AUC: 0.6461, F1: 0.7099

============================================================
进度: 30/31

🔍 测试模型: epoch_030_loss_0.8285_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6279, AUC: 0.6443, F1: 0.7229

============================================================
进度: 31/31

🔍 测试模型: model.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6279, AUC: 0.6443, F1: 0.7229

📊 结果已保存到: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/test_results_20251112_033800.csv
📝 报告已保存到: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/test_report_20251112_033800.txt

🏆 准确率最高的模型: epoch_008_loss_0.6156_acc_0.5992.h5 (准确率: 0.6778)
🗑️  开始清理epochs目录，将删除 30 个模型文件...
🗑️  已删除: epoch_025_loss_0.6002_acc_0.6446.h5
📋 已将最佳模型复制到: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s/epoch_008_loss_0.6156_acc_0.5992.h5
🗑️  已删除: epoch_008_loss_0.6156_acc_0.5992.h5
🗑️  已删除: epoch_009_loss_0.5979_acc_0.6570.h5
🗑️  已删除: epoch_023_loss_0.5285_acc_0.6942.h5
🗑️  已删除: epoch_001_loss_4.8977_acc_0.5331.h5
🗑️  已删除: epoch_005_loss_0.8502_acc_0.5868.h5
🗑️  已删除: epoch_029_loss_0.6122_acc_0.6488.h5
🗑️  已删除: epoch_021_loss_0.6055_acc_0.6983.h5
🗑️  已删除: epoch_010_loss_0.5311_acc_0.5372.h5
🗑️  已删除: epoch_004_loss_1.0644_acc_0.5620.h5
🗑️  已删除: epoch_030_loss_0.8285_acc_0.6322.h5
🗑️  已删除: epoch_002_loss_2.2767_acc_0.7107.h5
🗑️  已删除: epoch_020_loss_0.6861_acc_0.6529.h5
🗑️  已删除: epoch_028_loss_0.5427_acc_0.6694.h5
🗑️  已删除: epoch_017_loss_0.5990_acc_0.6777.h5
🗑️  已删除: epoch_026_loss_0.6616_acc_0.5702.h5
🗑️  已删除: epoch_007_loss_0.6486_acc_0.5496.h5
🗑️  已删除: epoch_018_loss_0.5552_acc_0.5702.h5
🗑️  已删除: epoch_016_loss_0.6452_acc_0.6322.h5
🗑️  已删除: epoch_015_loss_0.5712_acc_0.5785.h5
🗑️  已删除: epoch_006_loss_0.7675_acc_0.6157.h5
🗑️  已删除: epoch_003_loss_1.4489_acc_0.6694.h5
🗑️  已删除: epoch_014_loss_0.5581_acc_0.5537.h5
🗑️  已删除: epoch_022_loss_0.4743_acc_0.6405.h5
🗑️  已删除: epoch_012_loss_0.5671_acc_0.5537.h5
🗑️  已删除: epoch_019_loss_0.6726_acc_0.5579.h5
🗑️  已删除: epoch_024_loss_0.5723_acc_0.6116.h5
🗑️  已删除: epoch_011_loss_0.5995_acc_0.6446.h5
🗑️  已删除: epoch_027_loss_0.6174_acc_0.6570.h5
🗑️  已删除: epoch_013_loss_0.6946_acc_0.5785.h5
🗑️  已删除空的epochs目录
🔄 模型目录已重命名:
   原目录: 12Nov2025-03h14m14s
   新目录: 12Nov2025-03h14m14s_acc_0.6778

================================================================================
🎯 测试结果汇总
================================================================================
总模型数量: 31
成功测试: 31
失败测试: 0

📊 性能统计:
平均准确率: 0.6393 (±0.0205)
平均AUC: 0.6723 (±0.0291)
平均F1: 0.7072 (±0.0262)

🏆 最佳模型:
最高准确率: epoch_008_loss_0.6156_acc_0.5992.h5 (Acc: 0.6778)
最高AUC: epoch_002_loss_2.2767_acc_0.7107.h5 (AUC: 0.7237)
最高F1: epoch_011_loss_0.5995_acc_0.6446.h5 (F1: 0.7596)

✅ 测试完成 (耗时: 11.7 分钟)

================================================================================
🎉 训练和测试管道完成!
================================================================================
模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h14m14s
总耗时: 23.9 分钟
结束时间: 2025-11-12 03:38:01
================================================================================
