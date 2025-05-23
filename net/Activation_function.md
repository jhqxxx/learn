<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-20 23:36:07
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2025-05-05 15:50:12
-->
### 激活函数
* identity: f(x) = x,线性任务
* step:f(x)=0/1(x<0/x>=0)，全有或全无，导数为0，无法运算
* sigmoid
    - f(z)=1/(1+e^-z)
    - 输出范围0-1， 因此它对每个神经元的输出进行了归一化
    - 梯度平滑
    - 缺点：
        * 容易梯度消失
        * 函数输出不是以0为中心，会降低权重更新的效率
        * 函数执行指数运算，计算较慢
* Tanh/双曲正切函数
    - f(x)=tanh(x)=2/(1+e^(-2*x)) - 1
    - 输出范围-1～1，以0为中心
    - 负输入被强映射为负，零输入被映射为接近0
    - 梯度消失
    - 计算慢
* relu
    - f(x)=max(0, x), x>=0
    - f(x)=0,  x<0
    - 优点：
        * 当输入为正时，不存在梯度饱和问题
        * 计算速度快，因为ReLU函数中只存在线性关系
    - 缺点：
        * Dead ReLU问题，当输入为负时，ReLU完全失效，在反向传播过程中，则梯度为0
        * ReLU输出为0或正数，这意味着ReLU函数不以0为中心

* leakyrelu：
    - f(x)=x,  x>0
    - f(x)=a*x, x<=0
    - a: 人为给定的常数
    - 为解决Dead ReLU问题而设计
    - 注意：从理论上讲，Leaky ReLU具有ReLU的优点，但在实际操作中，并未完全证明leaky relu总是比relu好

* ELU:
    - f(x)=x,  x>0
    - f(x)=alpha*(e^x-1), x<=0
    - 具有relu的优点
    - 输出平均值接近0，以0为中心
    - 梯度更接近于单位自然梯度，从而使均值向零加速学习
    - 在较小输入下会饱和至负值，从而减少前向传播的变异和信息
    - 注意，实践中没有充分证据表明总是比ReLU好
* SELU：
    - f(x)=λ*x,  x>0
    - f(x)=λ*a*(e^x-1), x<=0
    - λ和a是固定数值

* PReLU/parametric ReLU:
    - f(x)=x,  x>0
    - f(x)=alpha*x,  x<=0
    - alpha: 通过梯度下降学习
    - alpha: 通常为0-1之间的数字，并且通常相对较小
    - 与ELU相比，PReLU在负值域是线性运算

* RReLU
    - f(x)=x,  x>=0
    - f(x)=a*x,  x<0, a是一个分布（lower，upper）里随机取样得到的值

* softsign

* softmax：
    - f(X)=e^xi/∑j=1...n(e^xj)
    - 用于多分类问题的激活函数，对于长度为k的任意实向量，Softmax可以将其压缩为长度为k,值在0-1范围内，并且向量中元素的总和为1的实向量
    - softmax与max不同，max函数:仅输出最大值，但softmax使较小的值具有较小的概率，并且不会直接丢弃
    - 函数分母结合了原始输出值的所有因子，意味着Softmax函数获得各种概率彼此相关
    - 缺点：
        * 在0点不可微
        * 负输入的梯度为0，反向传播不会更新，会产生永不激活的死亡神经元

* Swish:
    - f(x)=x*sigmoid(βx)
    - 优点：
        * 无界性有助于防止慢速训练期间，梯度逐渐接近0并导致饱和
        * 有界激活函数可以具有很强的正则化，并且较大的负输入问题也能解决
        * 导数恒大于0
        * 平滑度在优化和泛化中起了重要作用

* Maxout:
    - 单个Maxout节点：hi=max(z)
    - g(x) = h1(x) - h2(x)
    - 由两个Maxout节点组成的Maxout层可以很好地近似任何连续函数

* Softplus:
    - f(x)=ln(1+e^x)
    - 导数为f'(x)=e^x/(1+e^x)=1/(1+e^(-x))->即sigmoid函数
    - 类似ReLU函数，但是相对平滑，像ReLU一样是单侧抑制

* SiLU:
    - f(x)=x*sigmoid(x)

* GELU:
    - f(x) = x * p(X<=x)
    -      0.5*x*(1+Tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    - p(X<=x): 伯努利分布

* H-swish：
    f(x) = x*(ReLu6(x+3)/6)
注意：在一般的二元分类问题中，tanh函数用于隐藏层，而sigmoid函数用于输出层，但不固定，可具体问题调整

* AGLU: