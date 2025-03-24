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