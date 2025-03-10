'''
Author: jhq
Date: 2025-03-09 17:41:19
LastEditTime: 2025-03-09 18:09:15
Description: 
'''
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 加载数据集
dataset = load_dataset(r"C:\jhq\huggingface_dataset\Yelp\yelp_review_full")
print(dataset["train"][100])

# 处理数据集
tokenizer = AutoTokenizer.from_pretrained(r"C:\jhq\huggingface_model\google-bert\bert-base-cased")
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_val_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 加载模型并指定期望的标签数量，由数据集确定
model = AutoModelForSequenceClassification.from_pretrained(
    r"C:\jhq\huggingface_model\google-bert\bert-base-cased", num_labels=5)

# 训练超参数
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# 评估指标
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()