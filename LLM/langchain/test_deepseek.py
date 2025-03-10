'''
Author: jhq
Date: 2025-03-04 18:27:40
LastEditTime: 2025-03-05 19:11:12
Description: 
'''
from LangChainLLM import DeepSeek_R1_Distill_Qwen_LLM


llm = DeepSeek_R1_Distill_Qwen_LLM(mode_name_or_path = "C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

response = llm.invoke("我如何为学习大模型制定目标？")
print(f"{"-"*20}回答{"-"*20}")
print(response) # 输出回答