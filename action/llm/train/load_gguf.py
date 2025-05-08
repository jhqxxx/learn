'''
Author: jhq
Date: 2025-04-28 17:15:05
LastEditTime: 2025-04-28 17:24:53
Description: 
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 有问题
model_id = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_1_5B_nongye-GGUF"
filename = "Qwen2.5-1.5b-nongye-f16.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, device_map="auto", torch_dtype=torch.bfloat16).eval()

prompts = ["草莓种植有哪些要点", "八月份猕猴桃园怎么管理", "避免萝卜糠心有什么办法", "冬春育雏鸡如何防缺氧", "瓜实蝇咋消灭", "花卉如何安全度夏"]
for p in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant and good at farming problems. 你是一个乐于助人的助手且擅长处理农业问题。"},
        {"role": "user", "content": p},
    ]
    inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')
    gen_kwargs = {"max_length": 2048, "do_sample": True, "temperature": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(p)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-" * 50)