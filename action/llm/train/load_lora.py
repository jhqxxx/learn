from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B-Instruct"
lora_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_lora_nongye/checkpoint-3126"

model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(mode_path, use_fast=False, trust_remote_code=True)

model = PeftModel.from_pretrained(model, model_id=lora_path, torch_dtype=torch.float16)

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
    