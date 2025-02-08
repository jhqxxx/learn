<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-20 22:22:06
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2023-03-23 09:57:16
-->
## 正则化
    防止模型在训练过程中出现过拟合的现象
* 数据正则化：专注于对输入数据的更改
    - 数据增强策略  
        * Cutout 
        * RandomErasing
        * AutoAugment/Fast AutoAugment
        * PBA:Population Based Augmentation
        * RandAugment
        * Mixup
        * CutMix
        * CutBlur
        * BatchAugment
        * FixRes
        * Bag-of-Tricks
* 结构正则化：主要修改神经网络生成特征映射的过程
    - dropout  
        * MaxDropout
        * DropBlock
        * TargetDrop
        * AutoDrop
        * LocalDrop
        * Shake-Shake
        * ShakeDrop
        * Manifold Mixup
    - 残差连接：创建了一种名为'identity mapping'的结构，其是对原始输入的重建
    
    - L0-norm:非零参数的个数，用于产生稀疏性，实际很少用
    - L1-norm:绝对值之和，用以产生稀疏性，是L0范式的一个最优凸近似，容易优化求解
        L1的norm ball，以二维平面为例，在坐标轴间形状是矩形，权重等高线与norm ball容易在坐标轴上面相交，
    - L2-norm:平方和开方，更多是防止过拟合，让优化求解变得稳定，因为加了L2满足了强凸
        L2的norm ball，以二维平面为例，在坐标轴间形状是圆，权重等高线与norm ball在坐标轴上相遇的概率较小
    - L∞-norm:计算向量中的最大值
    总结：
        L1会趋向于产生少量的特征，而其他特征都是0， 而L2会选择更多的特征，这些特征都会接近于0，L1在特征选择时非常有用，而L2只是一种规则化而已， 在所有特征中只有少数特征起重要作用的情况下，L1比较合适，因为它能自动选择特征，大部分特征都能起作用，而且起的作用很平均，那么使用L2可能更合适
        L1优点是能够获得sparse(稀疏)模型，对于large-scale的问题来说这一点很重要，因为可以减少存储空间，缺点是加入L1后目标函数在原点不可导，需要做特殊处理。
        L2优点是实现简单，能起到正则化的作用，缺点是无法获得稀疏模型

* 标签正则化：主要对给定输入的标签进行转化和修正  
    - Label Smoothing
    - TSLA: 两阶段标签平滑
    - SLS: 结构标签平滑通过
    - JoCor


