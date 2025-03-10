<!--
 * @Author: jhq
 * @Date: 2025-03-09 12:18:27
 * @LastEditTime: 2025-03-10 17:17:38
 * @Description: 
-->


* tokenizer:分词器，处理文本，将文本转换为用于输入模型的数字数组
  - 有多个用来管理分词过程的规则，如何拆分单词和句子
  - 分词器返回字典：
    - input_ids: 用数字表示的token
    - attention_mask: 应该关注哪些token
    - token_type_ids: 对于多句子输入，表示一个token属于哪个序列
  - 分词器也可以接收列表作为输入，并填充和截断文本，返回具有统一长度的批次

* AutoModel: 加载预训练的实例，为不同任务选择正确的AutoModel
    - AutoTokenizer: 加载预训练的分词器，为不同任务选择正确的AutoTokenizer
    - AutoImageProcessor: 对于视觉任务，image processor将图像处理成正确的输入格式
    - AutoFeatureExtractor: 对于音频任务，feature extractor将音频处理成正确的输入格式
    - AutoProcessor: 多模态任务需要一种processor，将文本和视觉两种类型的预处理工具结合起来
    - AutoModelForXXX: 加载预训练的模型，为不同任务选择正确的AutoModelForXXX
    
* pipelines

* agent
    - 智能体是一个系统，它使用LLM作为引擎，并且具有访问称为工具的功能
    - 这些工具是执行任务的函数，包含所有必要的描述信息，帮助智能体正确使用他们
    - 一次性设计一系列工具并同时执行它们，像CodeAgent
    - 一次执行一个工具，并等待每个工具的结构后再启动下一个，像ReactJsonAgent
    - 代码智能体
    - 推理智能体
    - 初始化智能体：
        - 一个LLM，使用LLM作为引擎
        - 一个系统提示，告诉LLM引擎应该如何生成输出
        - 一个工具箱，智能体可以从中选择工具执行
        - 一个解析器，从LLM输出中提取出哪些工具需要调用，以及使用哪些参数
        