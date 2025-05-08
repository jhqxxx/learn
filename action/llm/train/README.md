<!--
 * @Author: jhq
 * @Date: 2025-04-28 20:58:41
 * @LastEditTime: 2025-04-28 21:02:00
 * @Description: 
-->
* qwen 1.5b lora微调
    1. 准备alpace格式的微调数据
    2. 使用transformers和peft库lora微调
    3. 合并base模型和lora模型
    4. 测试合并后的模型效果
    5. 使用llama.cpp转换gguf
    6. 使用llama.cpp进行量化
    7. llama.cpp推理，测试模型效果