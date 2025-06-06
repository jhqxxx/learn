- tokenizer
  - 分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index
  - 可以切分成词、子词、字符
  - tokenizer定义了词表大小后，每个单词就转换成立为一个索引
  - 再使用Embedding将索引转换为对应的词向量

- sentencepiece:
    - 提供多种关于词的切分方法
    - 用来训练分词器

- embedding：
    - 将索引映射到向量空间中，每个词对应一个向量

* LLM：
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
    - llama1:
        - RMSNorm:
            - RMSNorm(x) = gamma*x/RMS(x)
            - RMS(x) = sqrt(sum(x^2)/len(x) + epsilon)
            - gamma
        - FFN_SwiGLU
            - FFN：Position-wise Feed-Forward Network,前馈神经网络，两个线性层，中间有激活函数
            - linear->activation->linear
            - GELU-Gaussian Error Linear Unit-高斯误差线性单元
            - Swish(x) = x * Sigmoid(beta*x) = x/(1+exp(-beta*x))
                - 当beta=1时，与Sigmoid一致
            - GLU：定义为输入的两个线性变换的逐元素乘积，其中一个线性层经过sigmoid激活
                - GLU(x, W, V, b, c) = sigmoid(xW+b) * (xV+c)
                - 省略激活函数，称之为双线性层
                - bilinear(x, W, V, b, c) = (xW+b) * (xV+c)
            - FFN_SwiGLU(x, W1, W2 W3) = (SiLU(xW1)*xW3)W2 
                - 逐元素相乘
                - 有三个线性层权重矩阵
        - RoPE
            - 将位置编码与词向量通过旋转矩阵相乘
            - 对输入token计算query和key
            - 对token位置计算旋转位置编码
            - 对query和key应用旋转位置编码
    - llama2:
        - kv cache优化-GQA-Group Query Attention
            - kv cache内存计算公式：memory_kv_cache = 2*2*nh*b(s+o)
            - MHA-Multi-Head Attention, QKV三部分有相同数量的头，且一一对应，每个头的QKV各算各的，最后将各个头的结果concate
            - MQA-Multi-Query Attention，让Q保持原来的头数，K和V只有一个头，多个头的Q共享一个K，V头，减少KVcache的内存消耗，如Q64个head，KV一个head,64个Q共享一个KV
            - GQA-Group Query Attention，是MHA和MQA的折中，将Q分为几组，如共有64个head， 分成8组，则kv头数：64/8=8，组内的Q共享一个kv头
    - llama3:
        - 采用新的Tokenizer,词汇表大小扩展至128k
        - 由sentencepiece换成tiktoken

* ViT
    - patch Embeddings, 将图像切分成patch，每个patch被展平成为一个向量，并通过一个线性层，相当于将patch转换为token embedding
    - position embeddings

* GPT：
    - GPT1:
        - 提出一个半监督学习方法，后续论文叫自监督self supervised learning：在没有标记的文本上训练一个大语言模型，然后在子任务上进行有监督微调
        - 无监督预训练：损失函数使用最大化似然估计：L(theta) = sum(logp(ui|ui-1,...ui-k;theta))
        - ui-1,...ui-k是上下文词，k是上下文窗口大小，每次拿连续k个词预测下一个词
        - 实际训练，最小化负对数似然损失-Negative Log-Likelihood：NLL(theta) = -sum(logp(ui|ui-1,...ui-k;theta))
    - GPT2:
        - 核心创新点：使用zero-shot,任务通过特定的提示进行，而不是微调
        - 构建下游任务的输入要跟之前预训练的输入文本一样，即输入的形式更像自然语言文本表示，是llm提示词prompt工程的开端
        - 修改初始化、预归一化和可以反转的词元
    - GPT3：
        - few-shot:通过少量的示例来完成特定任务，不微调模型
        - 采用Sparse Transformer中的attention结构，即稀疏注意力机制
    - 零样本、单样本和少样本

* 位置编码：
    - 为啥不直接用索引？？？ 长序列的索引值会非常大，如果将索引归一化到0-1之间，对于不同长度的序列，会导致问题。

* MOE-Mixed Export Models-混合专家模型
    - 存在一个router network会挑选两组"exports"-参数量更小的FFN，来分别处理该token，并通过add融合两组输出的结果
    - 门控或Router网络：模块负责根据输入token的特征动态选择激活哪些专家，router是由可学习的参数组成的网络
    - exports网络：每层MOE都包含 若干个专家网络，其通常是小型的FFN，在实际推理中只有部分专家会被激活参与计算

* deepseek:    
    - v2
        - MLA-Multi head Latent Attention
            - Q向量也采用了低秩压缩的方式，将输入向量投影到q_lora_rank维的低维空间
        - deepseekMOE:
            - 更精细的划分专家网络
            - 引入部分共享专家
            - 共享专家：1个共享专家，用于捕捉通用、全局的特征信息
            - 路由专家：每个MOE层有256个路由专家，负责精细化处理输入tokens的专业特征
            - Gate: 
                - 计算token与各个路由专家之间的匹配得分
                - 选择top-k个专家
                - 被选中的专家各自对token进行独立处理，并产生输出，然后根据gate权重进行加权求和得到最终输出，再和共享专家的输出进行融合
    - v3
        - 无辅助损失的负载平衡策略
        - MTP-Multi-Token Prediction

* 生成式模型的推理过程是迭代的
    - 模型输出回答长度为N
    - 实际上是执行了N次模型前向传播过程
    - 模型一次推理只输出一个token
    - 当前轮输出的token与之前输入tokens拼接，并作为下一轮的输入tokens，
    - 循环执行上述过程，直到遇到终止符EOS或生成的token数目达到设置的max_num_tokens才会停止

* KV cache:
    - 推理时，第i+1轮计算包含了第i轮的部分计算
    - 缓存当前轮可重复利用的计算结果，下一轮计算时直接读取缓存结果
    - FlashDecoding

* smoothquant
* AWQ

* Llava

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
        - 文档加载和切分的模块，加载文档并切分成文档片段
        - 向量化模块，将文档片段向量化
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
    - 优化方法
        - 根据语义分块文档
            - 先用普通方法分块文档-固定大小或特定符号分块
            - 计算文档块embedding
            - 计算相邻文档块embedding之间的相似度
            - 如果相似度满足条件
            - 将满足条件的文档合并为一个块
        - 测试不同大小的文档块效果，选择效果最好的文档块size
        - 上下文增强：查找到最匹配的文档块后，返回该块及其邻居块的内容
        - contextual chunk headers: 给文档块添加标题内容
            - 使用llm给文档块生成标题
            - prompt: Generate a concise and informative title for the given text
            - {"header": "...", "content": "..."}
            - {"header_embedding": "...", "content_embedding": "..."}
            - 如何表示header和content的对应关系
            - 查询时，query与[header_embedding, content_embedding]都计算相似性
            - 生成时将header和content拼接放到prompt中
        - 文档增强-给文档块生成问题
            - 使用llm给文档块生成问题
            - prompt: Based on the following text, generate {num_questions} different questions that can be answered using only this text
            - 如何表示questions和content的对应关系
            - 查询时，query与question_embedding, content_embedding都计算相似性，
            - 生成时，(referenced by question 'question'：'content')放到prompt中
        
        - query transformation-查询变换:
            - query rewriting:让查询更具体更详细
                - 使用llm生成更详细的query
                - system prompt: "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
                - user prompt: user_prompt = f"""
                    Rewrite the following query to make it more specific and detailed. Include relevant terms and concepts that might help in retrieving accurate information.
                    
                    Original query: {original_query}
                    
                    Rewritten query:
                    """
            - step-back prompting：生成更广泛的查询
                - 使用llm生成更广泛的query
                - system prompt: "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."
                - user prompt: user_prompt = f"""
                    Generate a broader, more general version of the following query that could help retrieve useful background information.
                    
                    Original query: {original_query}
                    
                    Step-back query:
                    """
            - sub-query-decomposition：将查询分解为多个子查询
                - system prompt: "You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query."
                - user prompt: user_prompt = f"""
                    Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question.
                    
                    Original query: {original_query}
                    
                    Generate {num_subqueries} sub-queries, one per line, in this format:
                    1. [First sub-query]
                    2. [Second sub-query]
                    And so on...
                    """
        - reranking-重排序
            - 使用llm给查询结果打分
            - system_prompt = """You are an expert at evaluating document relevance for search queries.
                Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

                Guidelines:
                - Score 0-2: Document is completely irrelevant
                - Score 3-5: Document has some relevant information but doesn't directly answer the query
                - Score 6-8: Document is relevant and partially answers the query
                - Score 9-10: Document is highly relevant and directly answers the query

                You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text."""
            - 将llm的打分和查询相似性分数结合起来，重新排序
        - rse-Relevant Segment Extraction相关片段提取
            - top-k查询得到相关文档块
            - 遍历所有文档块，给每个文档块打分，如果在top-kk列表中，改分为查询时的similarity分数，不在的话置为0或者负数
            - best_score = 0
            - best_segments = []
            - for start in range(len(scores)):
                for len in range(1, min(max_len, len(scores) - start) + 1):
                    end = start + len
                    segment_scores = sum(scores[start:end])
                    if segment_scores > best_score:
                        best_score = segment_scores
                        best_segment = [start, end]
            - 根据连续多个文档块的得分，选择得分最高的一个或几个文档块

        - 内容压缩，减少噪声
            - 使用llm对文档进行总结或信息提取
            - system_prompt = """You are an expert at information extraction.
                Your task is to extract ONLY the exact sentences from the document chunk that contain information relevant 
                to answering the user's query.

                Your output should:
                1. Include ONLY direct quotes of relevant sentences from the original text
                2. Preserve the original wording (do not modify the text)
                3. Include ONLY sentences that directly relate to the query
                4. Separate extracted sentences with newlines
                5. Do not add any commentary or additional text

                Format your response as plain text with no additional comments."""
        - 反馈循环？？？没看懂
            - 记录之前的查询和检索结果-feedback
            - 执行检索后，将feedback与问题和检索结果给到llm判断是否相关
            - 如果相关，将相关的feedback也放到prompt中
        - adaptive rag-自适应
            - 使用llm,根据query类型，选择合适的检索策略
            - 根据不同的检索策略，给llm不同prompt，得到检索增强方式
            - 处理不同增强方式，再做检索
        - self-rag:
            - 使用llm判断是否需要检索
            - 使用llm判断检索到的文档与query是否相关
            - 使用llm判断检索到的文档与llm的response是否相关
            - 使用llm判断response回答query的好坏
            - 选择回答最好的那个，作为response
        - proposition chunking
            - 先用字符长度分块文档
            - 使用llm将每个文档块重新分块，得到 proposition chunk
            - 使用llm给proposition chunk打分,高于阈值的采用
        - multi-model RAG with Image Caption
            - 提取文件中的图片和文字
            - 使用llm对图片进行描述
            - 将图片描述信息embedding之后保存
        - Fusion Retrieval:combining vector and keyword search
            - 使用bm25做关键词查询
        - graph RAG:
            - 使用llm对文本生成关键词
            - 创建知识图谱
            - 使用vector数据库对关键词进行检索
            - 找到知识图谱中检索结果的的邻居节点也作为加入检索结果
        - Hierarchical Indices for RAG-层次结构索引
            - 首先通过摘要识别相关文档章节，然后从这些章节中检索具体细节
            - 使用llm对文档进行摘要生成
        - Hypothetical Document Embedding for RAG
            - 使用llm对问题生成假想的文档
            - 使用假想文档进行搜索
        - corrective RAG-纠正
            - 使用llm评估检索到的文档
            - 使用网络搜索纠正文档
        - RAG with RL:

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