<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:11:35
 * @LastEditTime: 2025-03-02 19:24:39
 * @Description: ###
-->

###### 学习率

- 分段常数衰减-piecewise constant decay
  - 在不同学习阶段-epoch 指定不同的学习率
- 指数衰减-exponential decay
  - new_learning_rate = last_learning_rate \* gamma
  - gamma: 衰减系数
- 自然指数衰减-natural exponential decay
  - new_learning_rate = last_learning_rate _ e^(-gamma _ epoch)
  - gamma: 衰减率
- 多项式衰减-polynomial decay
  - cycle:学习率下降后是否会重新上升
  - cycle=True:学习率下降后重新上升,decay_steps = decay_steps\*math.ceil(epoch/decay_steps)
  - cycle=False:学习率单调递减到最低值,epoch = mint(epoch,decay_steps)
  - new_learning_rate = (least_learning_rate - end_lr)\*(1 - epoch/decay_steps)^power + end_lr
  - learning_rate 为初始学习率，decay_step 为进行衰减的步长，end_lr 为最低学习率，power 为多项式的幂
- 间隔衰减
- 多间隔衰减
- 逆时间衰减
- Lambda 衰减
  - lr_lambda = lambda epoch: 0.95 \*\* epoch
  - learning_rate= lr_lambda(epoch) \* initial_lr
- 余弦衰减
- 诺姆衰减
  - new_learning_rate=learning_rate∗dmode^−0.5∗min(epoch^−0.5,epoch∗warmup_steps^−1.5)
  - dmodel 代表模型的输入、输出向量特征维度，warmup_steps 为预热步数，learning_rate 为初始学习率
- loss 自适应衰减：当 loss 停止下降时，降低学习率
- 线性学习率 WarmUp
  - epoch < warmup_steps: learning_rate =start_lr+(end_lr−start_lr)∗epoch/warmup_steps
  - epoch >= warmup_steps: learning_rate = end_lr

###### 注意力机制

###### 正则化 regularization
* 减少过拟合
* 数据增强
* 正则化代价函数 θ=argminθ*(1/N)∑i=1N(L(y^i,y)+λR(w))
* L1正则化:RL2(w)=||w||2^2
* L2正则化:RL2(w)=||w||1
* dropout:随机丢弃一些神经元，防止过拟合
* 早停法

###### batch size

###### 归一化 normalization
* 将数据处理为某个分布之间的数，如[0,1],[-1,1]
* 标准化 standardization：z-score 归一化 (x-mean)/std
* 输入归一化
* 层归一化：对网络中间层的整个输入数据进行归一化
* 批量归一化：在小批量中独立的归一化每个特征





#### 行为识别

https://zhuanlan.zhihu.com/p/103566134
https://zhuanlan.zhihu.com/p/107983551

#### 行人重识别

#### 图像超分辨

#### 图像压缩

#### 图像生成

房屋打分