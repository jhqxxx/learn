'''
Author: jhq
Date: 2025-03-10 12:36:43
LastEditTime: 2025-03-10 13:10:36
Description: 
'''
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AdamW, get_scheduler, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from tqdm import tqdm


# 有bug，待解决
accelerator = Accelerator()

# 加载数据集
# dataset = load_dataset(r"C:\jhq\huggingface_dataset\Yelp\yelp_review_full")
train_dataset = load_dataset(r"C:\jhq\huggingface_dataset\Yelp\yelp_review_full", split="train[:1000]")
val_dataset = load_dataset(r"C:\jhq\huggingface_dataset\Yelp\yelp_review_full", split="test[:1000]")
# print(dataset["train"][100])

# 处理数据集
tokenizer = AutoTokenizer.from_pretrained(r"C:\jhq\huggingface_model\google-bert\bert-base-cased")
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_val_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

small_train_dataset = train_dataset.map(tokenize_function, batched=True)
small_val_dataset = val_dataset.map(tokenize_function, batched=True)
# 加载模型并指定期望的标签数量，由数据集确定
model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\jhq\huggingface_model\google-bert\bert-base-cased", num_labels=5)

optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    small_train_dataset, small_val_dataset, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)