<!--
 * @Author: jhq
 * @Date: 2022-11-13 15:19:39
 * @LastEditTime: 2023-03-20 17:46:40
 * @Description: 
-->
### 常规网络
* VGG
* ResNet及变体
    核心思想：残差连接
* Inception系列
    多分支结构：使用不同尺寸卷积核，增加网络广度

### 轻量级网络
* MobileNet系列
    v1:深度可分离卷积
    v2:倒残差结构：两边窄中间宽，去掉倒残差结构最后的ReLU，ReLU会使一些一些神经元失活
        作者发现很多卷积核参数为0，卷积核没有发挥提取特征的作用，作者通过1*1卷积将维度上升，再使用深度卷积，深度卷积的输入输出通道数更高，能够提取更多信息
    v3:引入SE结构,使用h-swish激活函数
        SE：
            假设每个通道的重要程度不同，有的通道更有用，有的通道则不太有用
            对每一个输出通道，先global average pool，每个通道得到1个标量，C个通道得到C个数，
            然后经过FC-ReLU-FC-Sigmoid得到C个0-1之间的标量，作为通道的权重
            然后原来的输出通道每个通道用对应得权重进行加权（对应通道的每个元素与权重分别相乘
            得到新的加权后的特征

DW卷积或分组卷积虽然能够有效降低计算量，但缺少通道间的信息交互与整合，MobileNet使用PW卷积来解决这个问题，但PW卷积计算量比较大(相对dw卷积)
shufflenet使用一种更加经济的方式，channel shuffe
* ShuffleNet系列
    v1:分组卷积，通道shuffle
    v2:提出4条轻量级网络设计准则
        在flops相同的情况下，输入通道=输出通道时，MAC(访问代价)最小
        在flops相同且输入固定的情况下，卷积的分组越大，MAC越大
        网络的分支越多，效率越低
        Element wise操作虽然flops不大，但是MAC代价不容忽视
        残差连接通过channel split分两个支路，使用concat, 减低MAC
        channel shuffle放在代码块最后

精度：mobilenetv2 > shufflenetv2
flops: shufflenet>mobilenet
参数规模：shufflenet > mobilenet
内存消耗：shufflenet > mobilenet
模型文件：shufflenet > mobilenet
推理延时：shufflenet > mobilenet

* RepVGG
    结构重参数化思想：训练时尽量用分支结构来提升网络性能，推理时，采用结构重参数化思想，将其变为单路结构
    将训练时的多路结构转换成推理时的单路结构

torchstate工具 ？？

### attention

#### convNeXt

#### MAE