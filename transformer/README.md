* transformer
    - point-wise
    - self-attentin
    - add & norm
    - feed-forward
    - linear
    - softmax
        - scale: 使softmax的结果更加平滑，避免梯度消失，有实验证明如果不scale，模型预训练很难收敛
        - sqrt(d_k): 线性变换后的Q和K经过norm, 均值接近0，方差接近1， q*k^T点积后，均值为0，方差为d_k, 除以标准差sqrt(d_k)达到归一化效果

* batch norm
    - 对于每个特征维度，计算它在整个批次中的均值和标准差，然后对该特征进行归一化
   
* layer norm
    - 对每个样本单独计算所有特征的均值和标准差，在该样本内进行归一化
    - nlp中，对每个token的特征向量进行归一化
    - 某个token的特征向量 LN(x) = alpha * (x - mean(x)) / std(x) + beta

* qkv
    - xi = q*K^T:q与每一个k值的点积，反应了Query和每一个Key的相关性
    - softmax(xi) = e^xi/sum(e^xi):将其转化为和为1的分数
    - attention(Q,K,V) = softmax(Q*K^T)*V:将注意力分数和值向量做乘积
    - 因Q，K维度大，softmax后的结果差异大，影响梯度稳定性，需做放缩
    - attention(Q,K,V) = softmax(Q*K^T/sqrt(d_k))*V:d_k为K的维度

* self-attention: Q,K,V都由同一个输入通过不同参数矩阵计算得到

* casual self attention:因果自注意力，也叫掩码自注意力-mask self attention，
    - 使用注意力掩码，遮蔽一些特定位置的token
    - 让模型只能使用历史信息进行预测而不能看到未来信息
    - 模型预测根据之前的token生成下一个token，这个过程是一个串行的过程
    - 为了能够进行并行计算，引入掩码自注意力的方法：
        <BOS> 【MASK】【MASK】【MASK】【MASK】
        <BOS>    I   【MASK】 【MASK】【MASK】
        <BOS>    I     like  【MASK】【MASK】
        <BOS>    I     like    you  【MASK】
        <BoS>    I     like    you   </EOS>
    - 由于引入掩码，上面的五个序列可以同时进行并行计算

* Multi-Head Attention:
    - 一次注意力计算只能拟合一种相关关系，单一的attention机制很难全面拟合语句序列里的相关关系
    - 多头注意力机制，同时对一个预料进行多次注意力计算，每次注意力计算拟合不同的关系，将最后的多次结果拼接起来,再通过一个线性层，得到最后的输出
    - MultiHead(Q,K,V) = Concat(head_1,head_2,...,head_h)*W^O where head_i = Attention(QW^Q_i,KW^K_i,VW^V_i)

* decoder：q来自于前一层的decoder输出，k,v来自于encoder最后一层的输出

* position embedding:
    - 给embedding vector加上位置信息
    - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

* self-attention:
    - 能够在一层内直接捕获全局依赖关系，每个token都能与序列中任意位置的token进行信息交互，不受固定窗口大小限制
* cnn
    - 典型的卷积操作受限与卷积核的大小，捕获的是局部信息，虽然可以通过堆叠多层卷积或使用扩张卷积来扩大感受野，但这种扩展时逐层进行的，依赖网络深度

