'''
Author: jhq
Date: 2025-04-28 16:25:54
LastEditTime: 2025-04-28 16:52:55
Description: 
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B-Instruct"
lora_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_lora_nongye/checkpoint-3126"
merge_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_1_5B_nongye"

model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(mode_path, use_fast=False, trust_remote_code=True)

lora_model = PeftModel.from_pretrained(model, model_id=lora_path, torch_dtype=torch.bfloat16)
print("Applying the LoRA")
model = lora_model.merge_and_unload()

print(f"Saving the target model to {merge_path}")
model.save_pretrained(merge_path)
tokenizer.save_pretrained(merge_path)