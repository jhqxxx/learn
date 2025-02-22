'''
Author: jhq
Date: 2025-02-20 13:19:47
LastEditTime: 2025-02-22 16:06:40
Description: 
'''
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


def generate_training_data(data_point):
    """
    (1) Goal:
        - This function is used to transform a data point (input and output texts) to tokens that our model can read

    (2) Arguments:
        - data_point: dict, with field "instruction", "input", and "output" which are all str

    (3) Returns:
        - a dict with model's input tokens, attention mask that make our model causal, and corresponding output targets

    (3) Example:
        - If you construct a dict, data_point_1, with field "instruction", "input", and "output" which are all str, you can use the function like this:
            formulate_article(data_point_1)

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""
    # count the number of input tokens
    len_user_prompt_tokens = (
        len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        ) - 1
    )
    # transform input prompt into tokens
    full_tokens = tokenizer(
        prompt + " " + data_point["output"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

# 進行生成回覆的評估


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


model_name = "./cache/models--MediaTek-Research--Breeze-7B-Instruct-v0_1"

cache_dir = "./cache"

# 以4bit加载模型，以 nf4 为量化类型、使用嵌套量化并使用 bfloat16 作为计算数据类型的模型
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,     # 使用 4-bit 量化
    bnb_4bit_quant_type="nf4",     # 使用 NF4量化类型
    bnb_4bit_use_double_quant=True,  # 使用嵌套量化：将会在第一轮量化之后启用第二轮量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型
)

# 從指定的模型名稱或路徑載入預訓練的語言模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)

# 創建 tokenizer 並設定結束符號 (eos_token)
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

# 設定模型推理時需要用到的decoding parameters
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)

# demo examples
test_tang_list = ['相見時難別亦難，東風無力百花殘。', '重帷深下莫愁堂，臥後清宵細細長。', '芳辰追逸趣，禁苑信多奇。']

# get the model output for each examples
demo_before_finetune = []
for tang in test_tang_list:
    demo_before_finetune.append(f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n'+evaluate(
        '以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose=False))

# print and store the output to text file
for idx in range(len(demo_before_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_before_finetune[idx])
    print("-" * 80)

num_train_data = 1040


# 設定model checkpoint儲存目錄 (如果想要將model checkpoints存在其他目錄下可以修改這裡)
ckpt_dir = "./exp1"
num_epoch = 1  # 設定訓練的總Epoch數 (數字越高，訓練越久，若使用免費版的colab需要注意訓練太久可能會斷線)
LEARNING_RATE = 3e-4

cache_dir = "./cache"  # 設定快取目錄路徑
from_ckpt = False  # 是否從checkpoint載入模型的權重，預設為否
ckpt_name = None  # 從特定checkpoint載入權重時使用的檔案名稱，預設為無
dataset_dir = "D:/code/GenAI-Hw5/Tang_training_data.json"  # 設定資料集的目錄或檔案路徑
logging_steps = 20  # 定義訓練過程中每隔多少步驟輸出一次訓練誌
save_steps = 65  # 定義訓練過程中每隔多少步驟保存一次模型
save_total_limit = 3  # 控制最多保留幾個模型checkpoint
report_to = None  # 設定上報實驗指標的目標，預設為無
MICRO_BATCH_SIZE = 4  # 定義微批次的大小
BATCH_SIZE = 16  # 定義一個批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 計算每個微批次累積的梯度步數
CUTOFF_LEN = 256  # 設定文本截斷的最大長度
LORA_R = 8  # 設定LORA（Layer-wise Random Attention）的R值
LORA_ALPHA = 16  # 設定LORA的Alpha值
LORA_DROPOUT = 0.05  # 設定LORA的Dropout率
VAL_SET_SIZE = 0  # 設定驗證集的大小，預設為無
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj",
                  "down_proj", "gate_proj", "v_proj"]  # 設定目標模組，這些模組的權重將被保存為checkpoint
device_map = "auto"  # 設定設備映射，預設為"auto"
# 獲取環境變數"WORLD_SIZE"的值，若未設定則預設為1
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1  # 根據world_size判斷是否使用分散式數據處理(DDP)，若world_size為1則不使用DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


# create the output directory you specify

os.makedirs(ckpt_dir, exist_ok=True)

# 根據 from_ckpt 標誌，從 checkpoint 載入模型權重
if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

# 將模型準備好以使用 INT8 訓練
model = prepare_model_for_int8_training(model)

# 使用 LoraConfig 配置 LORA 模型
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# 將 tokenizer 的 padding token 設定為 0
tokenizer.pad_token_id = 0

# 載入並處理訓練數據
with open(dataset_dir, "r", encoding="utf-8") as f:
    data_json = json.load(f)
with open("tmp_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)

data = load_dataset('json', data_files="tmp_dataset.json",
                    download_mode="force_redownload")

# 將訓練數據分為訓練集和驗證集（若 VAL_SET_SIZE 大於 0）
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None

# 使用 Transformers Trainer 進行模型訓練
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,  # 使用混合精度訓練
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,  # 是否使用 DDP，控制梯度更新策略
        report_to=report_to,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)

# 禁用模型的 cache 功能
model.config.use_cache = False

# 若使用 PyTorch 2.0 版本以上且非 Windows 系統，進行模型編譯
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

# 開始模型訓練
trainer.train()

# 將訓練完的模型保存到指定的目錄中
model.save_pretrained(ckpt_dir)

# 印出訓練過程中可能的缺失權重的警告信息
print("\n If there's a warning about missing keys above, please disregard :)")
