 cd /home/minshi/Pedestrian_Crossing_Intention_Prediction ; /usr/bin/env /home/minshi/miniconda3/envs/tf26/bin/python /home/minshi/.vscode/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 34257 -- /home/minshi/Pedestrian_Crossing_Intention_Prediction/train_and_test_all_epoch_pipeline.py -c config_files/my/my_jaad.yaml 
================================================================================
🎯 训练和测试管道启动
================================================================================
配置文件: config_files/my/my_jaad.yaml
开始时间: 2025-11-12 03:40:16

🚀 开始训练...
2025-11-12 03:40:19.398537: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:19.401785: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:19.401872: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
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
2025-11-12 03:40:21.359499: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN
) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                               To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 03:40:21.360558: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.360669: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.360731: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.658545: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.658663: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.658770: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:40:21.658842: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3675 MB memory
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
📁 Models will be saved to: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s
📋 已复制 action_predict.py 到模型目录
2025-11-12 03:40:22.923139: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/30
WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimizing 
the loss.                                                                                                                                                           WARNING:tensorflow:Gradients do not exist for variables ['head_fc2/bias:0', 'etraj/kernel:0', 'etraj/bias:0', 'log_sigma_cls:0', 'log_sigma_reg:0'] when minimizing 
the loss.                                                                                                                                                           2025-11-12 03:40:27.671021: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be logged
 once.                                                                                                                                                              2025-11-12 03:40:28.027790: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1067/1067 [==============================] - 17s 12ms/step - loss: 9.0290 - cls_loss: 0.2151 - reg_loss: 0.5167 - intention_accuracy: 0.5187 - val_loss: 4.8827 - va
l_cls_loss: 0.2817 - val_reg_loss: 0.3163 - val_intention_accuracy: 0.2851                                                                                          /home/minshi/miniconda3/envs/tf26/lib/python3.8/site-packages/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must over
ride get_config. When loading, the custom mask layer must be passed to the custom_objects argument.                                                                   warnings.warn('Custom mask layers require a config and must override '
[Sigma] Epoch 1: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 2/30
1067/1067 [==============================] - 12s 11ms/step - loss: 3.1494 - cls_loss: 0.1836 - reg_loss: 0.1776 - intention_accuracy: 0.6162 - val_loss: 2.2296 - va
l_cls_loss: 0.2928 - val_reg_loss: 0.1673 - val_intention_accuracy: 0.6116                                                                                          [Sigma] Epoch 2: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 3/30
1067/1067 [==============================] - 17s 15ms/step - loss: 1.6590 - cls_loss: 0.1683 - reg_loss: 0.1211 - intention_accuracy: 0.6396 - val_loss: 1.4368 - va
l_cls_loss: 0.2743 - val_reg_loss: 0.1432 - val_intention_accuracy: 0.6612                                                                                          [Sigma] Epoch 3: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 4/30
1067/1067 [==============================] - 23s 21ms/step - loss: 1.1117 - cls_loss: 0.1583 - reg_loss: 0.1131 - intention_accuracy: 0.6696 - val_loss: 1.0514 - va
l_cls_loss: 0.2749 - val_reg_loss: 0.1438 - val_intention_accuracy: 0.6033                                                                                          [Sigma] Epoch 4: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 5/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.7942 - cls_loss: 0.1480 - reg_loss: 0.1134 - intention_accuracy: 0.6968 - val_loss: 0.8668 - va
l_cls_loss: 0.3337 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6074                                                                                          [Sigma] Epoch 5: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 6/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.5902 - cls_loss: 0.1421 - reg_loss: 0.1136 - intention_accuracy: 0.7231 - val_loss: 0.6855 - va
l_cls_loss: 0.3114 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6116                                                                                          [Sigma] Epoch 6: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 7/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.4616 - cls_loss: 0.1433 - reg_loss: 0.1136 - intention_accuracy: 0.7029 - val_loss: 0.5617 - va
l_cls_loss: 0.2921 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5620                                                                                          [Sigma] Epoch 7: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 8/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.3689 - cls_loss: 0.1358 - reg_loss: 0.1136 - intention_accuracy: 0.7071 - val_loss: 0.5618 - va
l_cls_loss: 0.3607 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                                          [Sigma] Epoch 8: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 9/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.3017 - cls_loss: 0.1248 - reg_loss: 0.1136 - intention_accuracy: 0.7484 - val_loss: 0.5804 - va
l_cls_loss: 0.4233 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5950                                                                                          [Sigma] Epoch 9: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 10/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.2631 - cls_loss: 0.1224 - reg_loss: 0.1136 - intention_accuracy: 0.7573 - val_loss: 0.5096 - va
l_cls_loss: 0.3832 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6653                                                                                          [Sigma] Epoch 10: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 11/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.2325 - cls_loss: 0.1167 - reg_loss: 0.1136 - intention_accuracy: 0.7737 - val_loss: 0.5717 - va
l_cls_loss: 0.4660 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5372                                                                                          [Sigma] Epoch 11: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 12/30
1067/1067 [==============================] - 23s 21ms/step - loss: 0.2068 - cls_loss: 0.1088 - reg_loss: 0.1136 - intention_accuracy: 0.7990 - val_loss: 0.4473 - va
l_cls_loss: 0.3568 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5909                                                                                          [Sigma] Epoch 12: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 13/30
1067/1067 [==============================] - 25s 23ms/step - loss: 0.1861 - cls_loss: 0.1021 - reg_loss: 0.1136 - intention_accuracy: 0.8074 - val_loss: 0.6069 - va
l_cls_loss: 0.5290 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6240                                                                                          [Sigma] Epoch 13: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 14/30
1067/1067 [==============================] - 25s 23ms/step - loss: 0.1711 - cls_loss: 0.0982 - reg_loss: 0.1136 - intention_accuracy: 0.8172 - val_loss: 0.4651 - va
l_cls_loss: 0.3954 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                                          [Sigma] Epoch 14: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 15/30
1067/1067 [==============================] - 24s 23ms/step - loss: 0.1583 - cls_loss: 0.0945 - reg_loss: 0.1136 - intention_accuracy: 0.8191 - val_loss: 0.5931 - va
l_cls_loss: 0.5339 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6488                                                                                          [Sigma] Epoch 15: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 16/30
1067/1067 [==============================] - 22s 20ms/step - loss: 0.1433 - cls_loss: 0.0875 - reg_loss: 0.1136 - intention_accuracy: 0.8229 - val_loss: 0.5600 - va
l_cls_loss: 0.5082 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6653                                                                                          [Sigma] Epoch 16: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 17/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1361 - cls_loss: 0.0873 - reg_loss: 0.1136 - intention_accuracy: 0.8374 - val_loss: 0.6115 - va
l_cls_loss: 0.5659 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                                          [Sigma] Epoch 17: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 18/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.1292 - cls_loss: 0.0860 - reg_loss: 0.1136 - intention_accuracy: 0.8313 - val_loss: 0.5582 - va
l_cls_loss: 0.5178 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6694                                                                                          [Sigma] Epoch 18: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 19/30
1067/1067 [==============================] - 20s 19ms/step - loss: 0.1203 - cls_loss: 0.0819 - reg_loss: 0.1136 - intention_accuracy: 0.8458 - val_loss: 0.5824 - va
l_cls_loss: 0.5465 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6570                                                                                          [Sigma] Epoch 19: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 20/30
1067/1067 [==============================] - 13s 12ms/step - loss: 0.1168 - cls_loss: 0.0829 - reg_loss: 0.1136 - intention_accuracy: 0.8411 - val_loss: 0.7647 - va
l_cls_loss: 0.7312 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6240                                                                                          [Sigma] Epoch 20: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 21/30
1067/1067 [==============================] - 13s 12ms/step - loss: 0.1052 - cls_loss: 0.0751 - reg_loss: 0.1136 - intention_accuracy: 0.8650 - val_loss: 0.4812 - va
l_cls_loss: 0.4530 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6777                                                                                          [Sigma] Epoch 21: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 22/30
1067/1067 [==============================] - 13s 13ms/step - loss: 0.1062 - cls_loss: 0.0789 - reg_loss: 0.1136 - intention_accuracy: 0.8510 - val_loss: 0.6515 - va
l_cls_loss: 0.6251 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5950                                                                                          [Sigma] Epoch 22: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 23/30
1067/1067 [==============================] - 17s 16ms/step - loss: 0.0979 - cls_loss: 0.0738 - reg_loss: 0.1136 - intention_accuracy: 0.8618 - val_loss: 0.6960 - va
l_cls_loss: 0.6732 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6281                                                                                          [Sigma] Epoch 23: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 24/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0953 - cls_loss: 0.0737 - reg_loss: 0.1136 - intention_accuracy: 0.8622 - val_loss: 0.6349 - va
l_cls_loss: 0.6142 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6198                                                                                          [Sigma] Epoch 24: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 25/30
1067/1067 [==============================] - 22s 21ms/step - loss: 0.0920 - cls_loss: 0.0726 - reg_loss: 0.1136 - intention_accuracy: 0.8646 - val_loss: 0.5385 - va
l_cls_loss: 0.5205 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6116                                                                                          [Sigma] Epoch 25: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 26/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0874 - cls_loss: 0.0702 - reg_loss: 0.1136 - intention_accuracy: 0.8693 - val_loss: 0.5004 - va
l_cls_loss: 0.4839 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5868                                                                                          [Sigma] Epoch 26: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 27/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0874 - cls_loss: 0.0719 - reg_loss: 0.1136 - intention_accuracy: 0.8650 - val_loss: 0.5705 - va
l_cls_loss: 0.5556 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6322                                                                                          [Sigma] Epoch 27: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 28/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0803 - cls_loss: 0.0660 - reg_loss: 0.1136 - intention_accuracy: 0.8754 - val_loss: 0.6524 - va
l_cls_loss: 0.6391 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6364                                                                                          [Sigma] Epoch 28: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 29/30
1067/1067 [==============================] - 23s 22ms/step - loss: 0.0787 - cls_loss: 0.0655 - reg_loss: 0.1136 - intention_accuracy: 0.8754 - val_loss: 0.8004 - va
l_cls_loss: 0.7880 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.5744                                                                                          [Sigma] Epoch 29: sigma_cls=0.6931 sigma_reg=0.6931
Epoch 30/30
1067/1067 [==============================] - 24s 22ms/step - loss: 0.0828 - cls_loss: 0.0706 - reg_loss: 0.1136 - intention_accuracy: 0.8730 - val_loss: 0.6572 - va
l_cls_loss: 0.6447 - val_reg_loss: 0.1446 - val_intention_accuracy: 0.6529                                                                                          [Sigma] Epoch 30: sigma_cls=0.6931 sigma_reg=0.6931

🎯 Training completed!
📁 All epoch models saved in: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/epochs
Train model is saved to data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/model.h5
Available metrics: ['loss', 'cls_loss', 'reg_loss', 'intention_accuracy', 'val_loss', 'val_cls_loss', 'val_reg_loss', 'val_intention_accuracy', 'sigma_cls', 'val_si
gma_cls', 'sigma_reg', 'val_sigma_reg']                                                                                                                             Training plots saved to model directory
Wrote configs to data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/configs.yaml
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step

======================================================================
🎯 MODEL TEST RESULTS 🎯
======================================================================
Accuracy:   0.6411
AUC:        0.5977
F1-Score:   0.7288
Precision:  0.6913
Recall:     0.7706
======================================================================

Model saved to data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/

✅ 训练完成 (耗时: 11.3 分钟)
🔍 查找最新模型目录...
📁 找到模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s

🧪 开始测试模型...
2025-11-12 03:51:38.805559: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:38.812281: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:38.812487: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             🚀 开始测试模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s
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

🔍 测试模型: epoch_001_loss_4.8827_acc_0.2851.h5
2025-11-12 03:51:40.332275: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN
) to use the following CPU instructions in performance-critical operations:  AVX2 FMA                                                                               To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-11-12 03:51:40.334529: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.334747: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.334960: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.866762: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.866991: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.867167: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must
 be at least one NUMA node, so returning NUMA node zero                                                                                                             2025-11-12 03:51:40.867299: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3429 MB memory
:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9                                                        WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
2025-11-12 03:51:42.592432: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2025-11-12 03:51:44.525627: I tensorflow/stream_executor/cuda/cuda_blas.cc:1760] TensorFloat-32 will be used for the matrix multiplication. This will only be logged
 once.                                                                                                                                                              2025-11-12 03:51:45.157631: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.3950, AUC: 0.6953, F1: 0.0641

============================================================
进度: 2/31

🔍 测试模型: epoch_002_loss_2.2296_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6555, AUC: 0.7350, F1: 0.6873

============================================================
进度: 3/31

🔍 测试模型: epoch_003_loss_1.4368_acc_0.6612.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6725, AUC: 0.7200, F1: 0.7286

============================================================
进度: 4/31

🔍 测试模型: epoch_004_loss_1.0514_acc_0.6033.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step
✅ 准确率: 0.6603, AUC: 0.7115, F1: 0.6909

============================================================
进度: 5/31

🔍 测试模型: epoch_005_loss_0.8668_acc_0.6074.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6683, AUC: 0.7259, F1: 0.7059

============================================================
进度: 6/31

🔍 测试模型: epoch_006_loss_0.6855_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step
✅ 准确率: 0.6592, AUC: 0.7077, F1: 0.6995

============================================================
进度: 7/31

🔍 测试模型: epoch_007_loss_0.5617_acc_0.5620.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6406, AUC: 0.6948, F1: 0.6673

============================================================
进度: 8/31

🔍 测试模型: epoch_008_loss_0.5618_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6534, AUC: 0.7006, F1: 0.6959

============================================================
进度: 9/31

🔍 测试模型: epoch_009_loss_0.5804_acc_0.5950.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6629, AUC: 0.7124, F1: 0.7121

============================================================
进度: 10/31

🔍 测试模型: epoch_010_loss_0.5096_acc_0.6653.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6481, AUC: 0.6895, F1: 0.7169

============================================================
进度: 11/31

🔍 测试模型: epoch_011_loss_0.5717_acc_0.5372.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6497, AUC: 0.6918, F1: 0.6953

============================================================
进度: 12/31

🔍 测试模型: epoch_012_loss_0.4473_acc_0.5909.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 17s 9ms/step
✅ 准确率: 0.6342, AUC: 0.6671, F1: 0.6800

============================================================
进度: 13/31

🔍 测试模型: epoch_013_loss_0.6069_acc_0.6240.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6396, AUC: 0.6802, F1: 0.7088

============================================================
进度: 14/31

🔍 测试模型: epoch_014_loss_0.4651_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6273, AUC: 0.6637, F1: 0.6951

============================================================
进度: 15/31

🔍 测试模型: epoch_015_loss_0.5931_acc_0.6488.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6236, AUC: 0.6749, F1: 0.7000

============================================================
进度: 16/31

🔍 测试模型: epoch_016_loss_0.5600_acc_0.6653.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 19s 10ms/step
✅ 准确率: 0.6300, AUC: 0.6691, F1: 0.7063

============================================================
进度: 17/31

🔍 测试模型: epoch_017_loss_0.6115_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6172, AUC: 0.6476, F1: 0.6845

============================================================
进度: 18/31

🔍 测试模型: epoch_018_loss_0.5582_acc_0.6694.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6050, AUC: 0.6393, F1: 0.6956

============================================================
进度: 19/31

🔍 测试模型: epoch_019_loss_0.5824_acc_0.6570.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 6ms/step
✅ 准确率: 0.6252, AUC: 0.6575, F1: 0.7121

============================================================
进度: 20/31

🔍 测试模型: epoch_020_loss_0.7647_acc_0.6240.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 14s 7ms/step
✅ 准确率: 0.6523, AUC: 0.6938, F1: 0.7363

============================================================
进度: 21/31

🔍 测试模型: epoch_021_loss_0.4812_acc_0.6777.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6257, AUC: 0.6564, F1: 0.7100

============================================================
进度: 22/31

🔍 测试模型: epoch_022_loss_0.6515_acc_0.5950.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6263, AUC: 0.6594, F1: 0.7115

============================================================
进度: 23/31

🔍 测试模型: epoch_023_loss_0.6960_acc_0.6281.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 20s 10ms/step
✅ 准确率: 0.6316, AUC: 0.6616, F1: 0.7255

============================================================
进度: 24/31

🔍 测试模型: epoch_024_loss_0.6349_acc_0.6198.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 11ms/step
✅ 准确率: 0.6295, AUC: 0.6672, F1: 0.7133

============================================================
进度: 25/31

🔍 测试模型: epoch_025_loss_0.5385_acc_0.6116.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.6108, AUC: 0.6315, F1: 0.7063

============================================================
进度: 26/31

🔍 测试模型: epoch_026_loss_0.5004_acc_0.5868.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 11s 6ms/step
✅ 准确率: 0.6130, AUC: 0.6448, F1: 0.6913

============================================================
进度: 27/31

🔍 测试模型: epoch_027_loss_0.5705_acc_0.6322.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 12s 6ms/step
✅ 准确率: 0.5981, AUC: 0.6229, F1: 0.7103

============================================================
进度: 28/31

🔍 测试模型: epoch_028_loss_0.6524_acc_0.6364.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 13s 7ms/step
✅ 准确率: 0.6199, AUC: 0.6583, F1: 0.7249

============================================================
进度: 29/31

🔍 测试模型: epoch_029_loss_0.8004_acc_0.5744.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 14s 7ms/step
✅ 准确率: 0.6087, AUC: 0.6244, F1: 0.6944

============================================================
进度: 30/31

🔍 测试模型: epoch_030_loss_0.6572_acc_0.6529.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 18s 9ms/step
✅ 准确率: 0.6411, AUC: 0.6543, F1: 0.7288

============================================================
进度: 31/31

🔍 测试模型: model.h5
WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
[DataGenerator] auto class_weight -> {0: 0.6257309941520468, 1: 0.3742690058479532}
1881/1881 [==============================] - 21s 10ms/step
✅ 准确率: 0.6411, AUC: 0.6543, F1: 0.7288

📊 结果已保存到: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/test_results_20251112_040109.csv
📝 报告已保存到: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/test_report_20251112_040109.txt

🏆 准确率最高的模型: epoch_003_loss_1.4368_acc_0.6612.h5 (准确率: 0.6725)
🗑️  开始清理epochs目录，将删除 30 个模型文件...
🗑️  已删除: epoch_007_loss_0.5617_acc_0.5620.h5
🗑️  已删除: epoch_023_loss_0.6960_acc_0.6281.h5
🗑️  已删除: epoch_002_loss_2.2296_acc_0.6116.h5
🗑️  已删除: epoch_020_loss_0.7647_acc_0.6240.h5
🗑️  已删除: epoch_018_loss_0.5582_acc_0.6694.h5
🗑️  已删除: epoch_025_loss_0.5385_acc_0.6116.h5
🗑️  已删除: epoch_005_loss_0.8668_acc_0.6074.h5
🗑️  已删除: epoch_017_loss_0.6115_acc_0.6322.h5
🗑️  已删除: epoch_027_loss_0.5705_acc_0.6322.h5
📋 已将最佳模型复制到: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s/epoch_003_loss_1.4368_acc_0.6612.h5
🗑️  已删除: epoch_003_loss_1.4368_acc_0.6612.h5
🗑️  已删除: epoch_008_loss_0.5618_acc_0.5868.h5
🗑️  已删除: epoch_010_loss_0.5096_acc_0.6653.h5
🗑️  已删除: epoch_026_loss_0.5004_acc_0.5868.h5
🗑️  已删除: epoch_004_loss_1.0514_acc_0.6033.h5
🗑️  已删除: epoch_011_loss_0.5717_acc_0.5372.h5
🗑️  已删除: epoch_022_loss_0.6515_acc_0.5950.h5
🗑️  已删除: epoch_028_loss_0.6524_acc_0.6364.h5
🗑️  已删除: epoch_014_loss_0.4651_acc_0.6570.h5
🗑️  已删除: epoch_001_loss_4.8827_acc_0.2851.h5
🗑️  已删除: epoch_030_loss_0.6572_acc_0.6529.h5
🗑️  已删除: epoch_015_loss_0.5931_acc_0.6488.h5
🗑️  已删除: epoch_019_loss_0.5824_acc_0.6570.h5
🗑️  已删除: epoch_006_loss_0.6855_acc_0.6116.h5
🗑️  已删除: epoch_012_loss_0.4473_acc_0.5909.h5
🗑️  已删除: epoch_016_loss_0.5600_acc_0.6653.h5
🗑️  已删除: epoch_013_loss_0.6069_acc_0.6240.h5
🗑️  已删除: epoch_024_loss_0.6349_acc_0.6198.h5
🗑️  已删除: epoch_021_loss_0.4812_acc_0.6777.h5
🗑️  已删除: epoch_009_loss_0.5804_acc_0.5950.h5
🗑️  已删除: epoch_029_loss_0.8004_acc_0.5744.h5
🗑️  已删除空的epochs目录
🔄 模型目录已重命名:
   原目录: 12Nov2025-03h40m21s
   新目录: 12Nov2025-03h40m21s_acc_0.6725

================================================================================
🎯 测试结果汇总
================================================================================
总模型数量: 31
成功测试: 31
失败测试: 0

📊 性能统计:
平均准确率: 0.6279 (±0.0473)
平均AUC: 0.6746 (±0.0300)
平均F1: 0.6848 (±0.1163)

🏆 最佳模型:
最高准确率: epoch_003_loss_1.4368_acc_0.6612.h5 (Acc: 0.6725)
最高AUC: epoch_002_loss_2.2296_acc_0.6116.h5 (AUC: 0.7350)
最高F1: epoch_020_loss_0.7647_acc_0.6240.h5 (F1: 0.7363)

✅ 测试完成 (耗时: 9.6 分钟)

================================================================================
🎉 训练和测试管道完成!
================================================================================
模型目录: data/models/jaad/Transformer_depth/12Nov2025-03h40m21s
总耗时: 20.9 分钟
结束时间: 2025-11-12 04:01:11
================================================================================
