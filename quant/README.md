<!--
 * @Author: jhq
 * @Date: 2025-04-09 11:03:01
 * @LastEditTime: 2025-04-09 11:03:15
 * @Description: 
-->

###### 量化
- 激活值：模型中需要进行量化的计算密集层的输入，典型的：线性层的输入，attention的输入
- SmoothQuant:PTQ-训练后量化，量化位宽W8A8-即权重和激活都是8bit量化
  - 基于权重易量化而激活难量化的观察提出了一个解决方法：
    - 引入平滑因子s来平滑激活中的异常值-离群值，
    - 并通过数学上的等效转换将量化的难度从激活迁移至权重上
- 量化粒度：基于不同的粒度去计算量化缩放稀疏
  - 逐张量量化是整个矩阵共用一个缩放系数
  - 逐token量化是为每个token设定不同的缩放系数
  - 逐通道量化是为每个通道设定不同的缩放系数
    - 分组量化，给通道分组，不同组使用不同缩放系数
- 量化难点：
  - 激活比权重更难量化
    - 权重分布比较均匀，易于量化
  - 激活值中的离群值是导致大模型难以量化的重要因素
  - 离群值通常出现于特定通道
  - 离群值于通道相关与token无关，应该对激活采用逐通道量化
  - 逐通道量化并不适合硬件加速GEMM内核
- r = Round(S*(q-Z))
  - q: 原始值
  - Z: 偏移量，又叫零点
  - S: 缩放因子
  - r: 量化后的值
- int8量化通过映射关系将输入数据映射到[-128,127]
- 参考<https://huggingface.co/blog/zh/hf-bitsandbytes-integration>
- 参考<https://www.cnblogs.com/huggingface/p/17816374.html>
- 模型大小由其参数量及其精度决定，精度通常为 float32、float16、bfloat16 之一
- 理想情况下训练和推理都应该在 FP32 中完成，但 FP32 比 FP16/BF16 慢两倍,因此实践中常常使用混合精度方法，其中使用 FP32 权重作为精确的主权重，
  而使用 FP16/BF16 权重进行前向和后向传播计算以提高训练速度，最后在梯度更新阶段再使用 FP16/BF16 梯度更新 FP32 主权重。
- 使用低精度，推理结果质量也下降了
- 引入 8 位量化/1 字节，量化过程是从一种数据类型舍入到另一种数据类型，量化是一个有噪过程，会导致信息丢失，是一种有损压缩
  - 零点量化
  - 最大绝对值量化
  - 他们都将浮点值映射为更紧凑的 Int8 值
- LLM.int8(): 性能下降是由离群特征引起的，使用自定义阈值提取离群值，并将矩阵分解为两部分，离群部分用 FP16 表示
- NF4-4bit-NormalFloat
- FP8
- FP4
- QLoRA: 使用 4bit 量化来压缩预训练模型，冻结基础模型参数，并将相对少量的可训练参数以低秩适配器的形式添加到模型中
- 加载模型时使用量化配置在每次加载模型时进行量化，加载时非常耗时
- 预量化：直接保存量化后的模型，使用时直接加载
  - GPTQ：Post-Training Quantization for GPT models,训练后量化，关注GPU推理和性能
    - pip install optimum
    - pip install auto-gptq
  - AWQ：Activation-aware Quantization,激活感知权重量化
    - 假设并非所有权重对LLM的性能都同等重要
    - 量化过程中会跳过一小部分权重，减轻量化损失
    - pip install autoawq
  - GGUF/GGML：GPT-Generated Unified Format,允许用户CPU来运行LLM，但也可以将其某些层加载到GPU以提高速度
    - pip install ctransformers