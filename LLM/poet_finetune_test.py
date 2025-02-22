import datasets
import transformers
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from colorama import *
from peft import PeftModel
from datasets import load_dataset, load_from_disk
import bitsandbytes as bnb
import torch.nn as nn
import torch
import os
import sys
import argparse
import json
import warnings
import logging
warnings.filterwarnings("ignore")

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    

def evaluate(instruction, generation_config, max_len, input="", verbose=True):
    """
    (1) Goal:
        - This function is used to get the model's output given input strings

    (2) Arguments:
        - instruction: str, description of what you want model to do
        - generation_config: transformers.GenerationConfig object, to specify decoding parameters relating to model inference
        - max_len: int, max length of model's output
        - input: str, input string the model needs to solve the instruction, default is "" (no input)
        - verbose: bool, whether to print the mode's output, default is True

    (3) Returns:
        - output: str, the mode's response according to the instruction and the input

    (3) Example:
        - If you the instruction is "ABC" and the input is "DEF" and you want model to give an answer under 128 tokens, you can use the function like this:
            evaluate(instruction="ABC", generation_config=generation_config, max_len=128, input="DEF")

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{instruction}
{input}
[/INST]"""
    # 將提示文本轉換為模型所需的數字表示形式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    # 使用模型進行生成回覆
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    # 將生成的回覆解碼並印出
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>", "").replace(
            "<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if (verbose):
            print(output)

    return output


ckpt_dir = "./exp1"


# find all available checkpoints
ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if (ckpt.startswith("checkpoint-")):
        ckpts.append(ckpt)

# list all the checkpoints
ckpts = sorted(ckpts, key = lambda ckpt: int(ckpt.split("-")[-1]))
print("all available checkpoints:")
print(" id: checkpoint name")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

id_of_ckpt_to_use = -1  # 要用來進行推理的checkpoint的id(對應上一個cell的輸出結果)
                        # 預設值-1指的是上列checkpoints中的"倒數"第一個，也就是最後一個checkpoint
                        # 如果想要選擇其他checkpoint，可以把-1改成有列出的checkpoint id中的其中一個

ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])

max_len = 128   # 生成回復的最大長度
temperature = 0.1  # 設定生成回覆的隨機度，值越小生成的回覆越穩定
top_p = 0.3  # Top-p (nucleus) 抽樣的機率閾值，用於控制生成回覆的多樣性
# top_k = 5 # 調整Top-k值，以增加生成回覆的多樣性和避免生成重複的詞彙
output_dir = "./run"  # 設定作業結果輸出目錄
os.makedirs(output_dir, exist_ok=True)

test_data_path = "D:/code/GenAI-Hw5/Tang_testing_data.json"
output_path = os.path.join(output_dir, "results.txt")

cache_dir = "./cache"  # 設定快取目錄路徑
seed = 42  # 設定隨機種子，用於重現結果
no_repeat_ngram_size = 3  # 設定禁止重複 Ngram 的大小，用於避免生成重複片段

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "./cache/models--MediaTek-Research--Breeze-7B-Instruct-v0_1"
# 使用 tokenizer 將模型名稱轉換成模型可讀的數字表示形式
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# 從預訓練模型載入模型並設定為 8 位整數 (INT8) 模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},  # 設定使用的設備，此處指定為 GPU 0
    cache_dir=cache_dir
)

# 從指定的 checkpoint 載入模型權重
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

results = []

# 設定生成配置，包括隨機度、束搜索等相關參數
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2
)

# 讀取測試資料
with open(test_data_path, "r", encoding = "utf-8") as f:
    test_datas = json.load(f)

# 對於每個測試資料進行預測，並存下結果
with open(output_path, "w", encoding = "utf-8") as f:
  for (i, test_data) in enumerate(test_datas):
      predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose = False)
      f.write(f"{i+1}. "+test_data["input"]+predict+"\n")
      print(f"{i+1}. "+test_data["input"]+predict)


# using the same demo examples as before
test_tang_list = ['相見時難別亦難，東風無力百花殘。', '重帷深下莫愁堂，臥後清宵細細長。', '芳辰追逸趣，禁苑信多奇。']

# inference our fine-tuned model
demo_after_finetune = []
for tang in test_tang_list:
  demo_after_finetune.append(f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n'+evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose = False))

# print and store the output to text file
for idx in range(len(demo_after_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_after_finetune[idx])
  print("-" * 80)