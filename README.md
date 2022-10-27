## 文件结构
- data_preprocess: （不重要，不用管）数据预处理相关，将SVHN用protocolbuffer转化为lmdb格式
- modules: nn.Module
  - MobileNet: MobileNetV2，主体
  - SPP: Spatial Pyramid Pooling，在MobileNet骨干网最后一层，用于适应不同尺寸的输入
  - STN: Spatial Transformer Network，用于对输入进行仿射变换
- SVHN_lmdb: SVHN数据集，lmdb格式
- SVHN_lmdb: 另一批SVHN数据集，lmdb格式
- testimg: 自己随便做的测试图片
- trained_models: 训练好的模型
- Dataset.py: 数据集类，用于读取lmdb格式的数据
- display_dataset.py: 用于显示数据集中的图片
- eval.py: 用于测试模型，自己测试着玩
- example3_pb2.py: （不重要，不用管）protocolbuffer parser，用于Dataset类读取lmdb格式的数据
- example_pb2.py: （不重要，不用管）protocolbuffer parser
- train.py: 训练模型脚本


## 数据集
- SVHN_lmdb  # SVHN数据集， format 1过滤掉尺寸过大或过小的图片后，裁剪成54*54，转换为lmdb格式，数字在图片中位置基本处于中间
- SVHN_lmdb2 # SVHN数据集， format 1没有过滤，图片大小为128*128，转换为lmdb格式，数字在图片中的位置不固定（方便训练STN）
