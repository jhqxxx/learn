'''
Author: jhq
Date: 2025-04-15 17:25:46
LastEditTime: 2025-04-28 16:51:53
Description: 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# model_name = "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B-Instruct"
model_name = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_1_5B_nongye"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(p)
        print(response)
        print("-" * 50)

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512,
#     streamer=streamer,
# )
