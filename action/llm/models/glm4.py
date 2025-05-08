'''
Author: jhq
Date: 2025-04-29 11:16:23
LastEditTime: 2025-04-29 12:27:47
Description: 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch

model_name = "/mnt/c/jhq/huggingface_model/ZhipuAI/GLM-4-9B-0414"

# 以4bit加载模型，以 nf4 为量化类型、使用嵌套量化并使用 bfloat16 作为计算数据类型的模型
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,     # 使用 4-bit 量化
    bnb_4bit_quant_type="nf4",     # 使用 NF4量化类型
    bnb_4bit_use_double_quant=True,  # 使用嵌套量化：将会在第一轮量化之后启用第二轮量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    quantization_config=nf4_config
)

prompts = ["草莓种植有哪些要点", "八月份猕猴桃园怎么管理", "避免萝卜糠心有什么办法", "冬春育雏鸡如何防缺氧", "瓜实蝇咋消灭", "花卉如何安全度夏"]
for p in prompts:
    messages = [
        {"role": "system", "content": "你是一个厉害的农业助手"},
        {"role": "user", "content": p},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        
        outputs = generated_ids[:, model_inputs['input_ids'].shape[1]:]

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(p)
        print(response)
        print("-" * 50)