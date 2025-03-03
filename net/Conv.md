<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-20 23:32:49
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2025-03-01 17:01:53
-->
### 卷积
* 普通卷积：
    - 特征图Hin*Win*Din * 滤波器h*w*Din*Dout -> 输出Hout*Wout*Dout
    - 特征图大小i, kernei size=k, padding=p, stride=s,输出为：lower_bound((i+2p-k)/s)+1

* 1x1卷积：跨通道交互与整合，升维/降维

* 分组卷积：
    - 将filter分成n个组，每个组负责特征图的部分深度，再将各组结果concat。
    - 当filter分组数量等于输入层通道数量相同时，即为DW卷积
    - 卷积被划分为多个路径，每个路径可以由不同的GPU分别处理
    - 参数量减少，h*w*Din*Dout -> (h*w*Din/n*Dout/n) * n

* DW卷积/Depthwise Convolution:
    - filter的每个kernel只负责特征图的单个个通道，一个通道只和一个kernel进行计算    

* PW卷积/Pointwise Convolution/1*1卷积：H*W*D  *  1*1*D  输出为H*W*1， 如果执行N次1*1卷积，并将结果连接在一起，得到 H*W*N的输出  
    * 降低维度以实现高效计算
    * 高效的低维嵌入，或特征池
    * 卷积后再次应用非线性


* 随机分组卷积：
    - shufflenet混合来自不同组filters的信息,将每组中的通道划分为几个子组，我们将这些子组混合在一起

* 逐点分组卷积：将组卷积应用于1*1卷积

在ShuffleNet论文中，作者使用了三种类型的卷积：

（1）随机分组卷积（Shuffled Grouped Convolution）

（2） 逐点分组卷积（Pointwise Grouped Convolution）

（3）深度可分离卷积

* 可分离卷积/Separable Convolution:
    * 空间可分离卷积/Spatially Separable Convolution:
        - 3*3的kernel可分为 3*1 和 1*3 两个kernel，输入特征图首先和3*1的kernel卷积，然后再和1*3的kernel卷积
        - 减少参数
        - 减少矩阵乘法
        - 很少使用，因为不是所有的kernel都可以被分为两个更小的kernel，而且其可能会限制在训练过程中找到所有可能的kernel，找到的结果也许不是最优的。
    * 深度可分离卷积/Depthwise Separable Convolution:
        - 该卷积分为两步：DW卷积和PW卷积
        - 首先使用DW卷积压缩空间维度，但是深度不变
        - 然后使用PW卷积扩展深度
        - 计算量显著减少

* 空洞卷积/扩张卷积/dilated convolutions/离散卷积：
    - 通过在卷积核元素之间插入空格来“扩张”卷积核，扩充参数取决于我们想如何扩大卷积核
    - 可扩大输入的感受野，而不增加kernel的尺寸

* 转置卷积/逆卷积：  
    逆向的卷积，要进行上采样
    初始化输入的stride代表做正向卷积时的步长
    输入n*n, 如果stride > 1,正向卷积时，卷积步长>1，生成的特征图(n+2*p-k)/s +1,变很小，所以反卷积时需要把正向跳过的步长补充上,则n=n+(stride-1)*(n-1)
    逆卷积时，步长始终stride=1，kernel_size=输入数据，
    padding=dilation*(kernel_size-1)-padding

* 3D卷积定义为filter的深度小于输入层的深度（即卷积核的个数小于输入层通道数），3Dfilter需要在h,w,c三个维度上滑动

Conv1D 2D 3D:这里的维度按照卷积核可移动的维度进行定义的
    * Conv1D: 只沿着一个轴，一维CNN的输入和输出数据是二维的，主要用于时间序列数据。input: sequence_len*feature_dimension, kernel: 1维tensor，len=kernrl_size, 
    * Conv2D: 在平面上沿两个轴滑动， 2D CNN的输入输出是3维的，主要用于图像数据
    * Conv3D: 可以沿着3个方向移动(高，宽，及图像通道)， 3D CNN的输入输出数据是4维的，通常用于3D图像数据（MRI，CT）扫描

###### 池化
* 平均池化
* 最大池化
* k-max池化