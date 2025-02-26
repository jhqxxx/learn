'''
Author: jhq
Date: 2025-02-26 11:55:43
LastEditTime: 2025-02-26 12:39:37
Description: 
参考
https://github.com/datawhalechina/self-llm/blob/master/models/DeepSeek-R1-Distill-Qwen/01-DeepSeek-R1-Distill-Qwen-7B%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re

mode_path = 'C:/jhq/huggingface_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, quantization_config=nf4_config)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, 
    quantization_config=nf4_config,
    device_map={'': 0}  # 設定使用的設備，此處指定為 GPU 0
    ).eval()

DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息
# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片
# 文本分割函数
def split_text(text):
    pattern = re.compile(r'(.*?)</think>(.*)', re.DOTALL) # 定义正则表达式模式
    match = pattern.search(text) # 匹配 <think>思考过程</think>回答
  
    if match: # 如果匹配到思考过程
        think_content = match.group(1).strip() # 获取思考过程
        answer_content = match.group(2).strip() # 获取回答
    else:
        think_content = "" # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip() # 直接返回回答
  
    return think_content, answer_content

if __name__ == '__main__':
    prompt = "春节到了，帮我写一下祝福语给王老师，不少于50字，不要说其他话"
    messages = [
            {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=8192) # 思考需要输出更多的Token数，设为8K
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    think_content, answer_content = split_text(response) # 调用split_text函数，分割思考过程和回答
    print(f"思考过程：{think_content}")
    print(f"回答：{answer_content}")