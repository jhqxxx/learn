- tokenizer
  - 分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index
  - 可以切分成词、子词、字符
  - tokenizer定义了词表大小后，每个单词就转换成立为一个索引
  - 再使用Embedding将索引转换为对应的词向量

- sentencepiece:
    - 提供多种关于词的切分方法
    - 用来训练分词器
* 输入：[batch_size, seq_len]
    - 通过embedding：[batch_size, seq_len, embedding_dim]
* 输出：[bach_size, seq_len, vocab_size]
    - 单个单词-由整个词表的概率分布，选择概率最大或者temperature进行采样得到的单词
* Llama:
    - 模型结构：
        - 只有transformer的decoder部分
        - RMSNorm：
            - 认为LayerNorm的中心偏移(减去均值等操作)没什么用
        - Q、K进行RoPE旋转式位置编码：位置编码用于捕捉序列中的位置信息，RoPE能有效的处理长序列，提高模型性能
        - Causal mask: 该机制保证每个位置只能看到前面的tokens
        - 使用Group Query Attention-分组查询注意力:保持性能的同时，降低模型计算复杂度
            - 多头注意力Multi-head Attention：每个头的Q/K/V独立，需要的缓存量很大
            - Multi-Query Attention: 多个头之间共享K/V，
            - Group Query Attention: 将多个头分组，组内共享K/V, 当group=1时，与multiti-query一致
        - MLP：down(up(x)*SiLU(gate(x))), down/up/gate: 线性层
        - SiLU
        - RoPE: 旋转式位置编码???

* Dolphin:
    - 语音大模型
    - CTC-Attention
        - CTC：序列建模
        - Attention：上下文捕捉
    - E-Branchformer编码器：
        - 采用并行分支结构
    - Transformer解码器
    - 引入4倍下采样
        - 减少输入特征的序列长度，加速计算，同时保留关键信息

* LLM评测：
    - 根据任务类型指定评测metric
        - 长文本问答：rouge
        - 短文本问答：F1
        - 生成式选择：accuracy
        - blue
    - 根据目标数据总结模型引导prompt
    - 确定预测结构的抽取方式
    - 计算pred和anwser的得分

* RAG
    - 基本结构
        - 向量化模块，将文档片段向量化
        - 文档加载和切分的模块，加载文档并切分成文档片段
        - 数据库模块，存放文档片段和对应的向量表示
        - 检索模块，根据query检索相关的文档片段
        - LLM模块，根据检索出来的文档回答用户问题
    - 流程
        - 索引
            - 文档加载和切分
            - 向量化：Embedding
            - 数据库
                - 数据库持久化，本地保存
                - 数据库加载，加载本地数据库
                - 获得文档的向量表示
                - 检索：根据query检索相关的文档片段
        - 检索
            - 余弦相似度：关注向量之间的角度，适合文本处理和信息检索
            - 欧几里得距离
            - 曼哈顿距离
        - 生成

* Agent
    - React??
    - SOP??
    - 工具
        - google search
        - 

* 架构
    - 应用层
    - 模型层
    - 框架层
    - 硬件层


* FlashMLA

* MCoT