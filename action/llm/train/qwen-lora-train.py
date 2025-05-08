'''
Author: jhq
Date: 2025-04-28 12:18:20
LastEditTime: 2025-04-28 15:55:57
Description: 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

def generate_training_data(data_point):
    max_length = 512    
    prompt = f"""\
        <|im_start|>system
        You are a helpful assistant and good at farming problems. 你是一个乐于助人的助手且擅长处理农业问题。
        <|im_end|>
        <|im_start|>user
        {data_point["instruction"]}
        {data_point["input"]}
        <|im_end|>
        <|im_start|>assistant
        """
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(data_point["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    

model_name = "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

ckpt_dir = "/mnt/c/jhq/huggingface_model/Qwen/Qwen_lora_nongye"
num_epoch = 3
LEARNING_RATE = 3e-4
dataset_dir = "/mnt/c/jhq/rag_file/farming/sft_data/nongye_sft_data.json"
logging_steps = 100
save_steps = 1000
save_total_limit = 3
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
CUTOFF_LEN = 512 
LORA_R = 8  # 設定LORA（Layer-wise Random Attention）的R值
LORA_ALPHA = 16  # 設定LORA的Alpha值
LORA_DROPOUT = 0.05  # 設定LORA的Dropout率
VAL_SET_SIZE = 0  # 設定驗證集的大小，預設為無
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj",
                  "down_proj", "gate_proj", "v_proj"]
device_map = "auto"
world_size = 1

os.makedirs(ckpt_dir, exist_ok=True)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    inference_mode=False, # 训练模式
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, config)


data = load_dataset('json', data_files=dataset_dir,
                    download_mode="force_redownload")

train_data = data['train'].shuffle().map(generate_training_data)
val_data = None

args = TrainingArguments(
    output_dir=ckpt_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100, 
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    warmup_steps=50,
    fp16=True,  # 使用混合精度训练
    save_total_limit=save_total_limit,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, padding=True
    ),
)
trainer.train()