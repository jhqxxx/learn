'''
Author: jhq
Date: 2025-03-09 10:46:24
LastEditTime: 2025-03-09 13:17:28
Description: 
'''
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer

# 文本分类
model = DistilBertForSequenceClassification.from_pretrained(r"C:\jhq\huggingface_model\distilbert\distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained(r"C:\jhq\huggingface_model\distilbert\distilbert-base-uncased-finetuned-sst-2-english")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# classifier = pipeline("sentiment-analysis")

print(classifier("We are very happy to show you the 🤗 Transformers library."))

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")