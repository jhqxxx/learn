### 常规网络
* VGG
* ResNet及变体
    核心思想：残差连接
* Inception系列
    多分支结构：使用不同尺寸卷积核，增加网络广度

### 轻量级网络
* MobileNet系列
    深度可分离卷积
* ShuffleNet系列
    分组卷积，通道shuffle
* RepVGG
    结构重参数化思想：训练时尽量用分支结构来提升网络性能，推理时，采用结构重参数化思想，将其变为单路结构
    将训练时的多路结构转换成推理时的单路结构

### attention