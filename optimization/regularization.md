<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-20 22:22:06
 * @LastEditors: jhq
 * @LastEditTime: 2022-09-20 22:55:41
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
        * 残差连接：创建了一种名为'identity mapping'的结构，其是对原始输入的重建
    - L1-norm
    - L2-norm
* 标签正则化：主要对给定输入的标签进行转化和修正  
    - Label Smoothing
    - TSLA: 两阶段标签平滑
    - SLS: 结构标签平滑通过
    - JoCor


