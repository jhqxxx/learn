'''
Author: jhq
Date: 2025-03-10 13:12:26
LastEditTime: 2025-03-10 15:53:03
Description: 参数高效微调-peft，在微调过程中冻结预训练模型的参数，并在其顶部添加少量可训练参数-adapters
'''
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM
from peft import PeftConfig, LoraConfig

# 加载预训练模型
model_path = r"C:\jhq\huggingface_model\facebook\opt-350m"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True
    )
    )
# 加载tokennizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # 加载指令微调模型
# peft_model_path = r"C:\jhq\huggingface_model\ybelkada\opt-350m-lora"
# # model.load_adapter(peft_model_path)
# peft_config = PeftConfig.from_pretrained(peft_model_path)
# peft_config.init_lora_weights = False
# model.add_adapter(peft_config)
# # 启用adapter
# model.enable_adapter()
# inputs = tokenizer("hello", return_tensors="pt")
# output = model.generate(**inputs)
# # 禁用adapter
# model.disable_adapter()
# output = model.generate(**inputs)

# lora_config = LoraConfig(
#     target_modules=["q_proj", "k_proj"],
#     init_lora_weights=False
# )

# # 添加新的adapter
# model.add_adapter(lora_config, adapter_name="adapter_1")
# model.add_adapter(lora_config, adapter_name="adapter_2")


# # 使用 adapter_1
# model.set_adapter("adapter_1")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# output = model.generate(**inputs)
# print(tokenizer.decode(output[0], skip_special_tokens=True))

# # 使用 adapter_2
# model.set_adapter("adapter_2")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# output = model.generate(**inputs)

# print(tokenizer.decode(output[0], skip_special_tokens=True))


# 训练peft
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # 将adapter添加到模型中
# model.add_adapter(peft_config)
# # 将模型传递给Trainer
# trainer = Trainer(model=model)
# trainer.train()
# #保存训练好的adapter并重新加载它
# save_dir = ""
# model.save_pretrained(save_dir)
# model = AutoModelForCausalLM.from_pretrained(save_dir)