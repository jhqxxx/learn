####  特点
* 无局部假设
    *  可并行计算
    * 对相对位置不敏感

* 无有序假设
    * 需要位置编码来反映位置变化对于特征的影响
    * 对绝对位置不敏感

* 任意两字符都可以建模：
    * 擅长长短程建模
    * 自注意力机制需要序列长度的平方级别复杂度

* encoder:
    * input word embedding: 
    * position encoding:
        * 通过sin/cos来固定表征：
         * 每个位置确定性
         * 对于不同的句子，相同位置的距离一致
         * 可以推广到更长的测试句子
        * pe(pos+k)可以写成pe(pos)的线性组合
        * 通过残差连接来使得位置信息流入深层
    * multi-head self-attention:
        * 使得建模能力更强，表征空间更丰富
        * 由多组QK，V构成， 每组单独计算一个attention向量
        * 把每组的attention向量拼起来，并进入一个FFN得到最终的向量
    * feed-forward network 
        * 只考虑每个单独位置进行建模
        * 不同位置参数共享
        * 类似于1*1卷积
    字符输入
    状态输出
* decoder：上一时刻的字符和encoder状态作为输入，返回字符预测概率
    * output word embedding
    * masked multi-head self-attention
    * multi-head cross-attention
    * feed-forward network
    * softmax


ViT(Vision Transformer):
    