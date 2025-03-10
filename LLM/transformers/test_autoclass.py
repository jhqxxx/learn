'''
Author: jhq
Date: 2025-03-09 13:19:56
LastEditTime: 2025-03-09 17:38:33
Description: 
'''
from transformers import (
    AutoTokenizer, 
    AutoImageProcessor, 
    ViTForImageClassification, 
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    )
from datasets import load_dataset, Audio

from PIL import Image
import requests

# 文本预处理
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=r"C:\jhq\huggingface_model\google-bert\bert-base-cased")
# sequence = "In a hole in the ground there lived a hobbit."
# encoded_input = tokenizer(sequence)
# print(encoded_input)
# # tokenizer解码成文本
# # [CLS] In a hole in the ground there lived a hobbit. [SEP]
# # CLS：分类器
# # SEP：分隔符
# print(tokenizer.decode(encoded_input["input_ids"]))

# # 多序列
# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_inputs = tokenizer(batch_sentences)
# print(encoded_inputs)

# # 填充padding，当句子的长度不同时，在较短的句子中添加特殊的padding token，以确保张量是矩形的、
# # 截断truncation，当某些序列对模型来说太长时，需要将序列截断为更短的长度
# encoded_inputs_padding = tokenizer(batch_sentences, padding=True, truncation=True)
# print(encoded_inputs_padding)
# # 构建张量，返回实际输入到模型的张量
# encoded_inputs_tensor = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_inputs_tensor)


# # 图像预处理，图像分类
# image_processor = AutoImageProcessor.from_pretrained(r"C:\jhq\huggingface_model\google\vit-base-patch16-224", use_fast=True)
# model = ViTForImageClassification.from_pretrained(r"C:\jhq\huggingface_model\google\vit-base-patch16-224")
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# input = image_processor(image, return_tensors="pt")
# output = model(**input)
# logits = output.logits
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

# # 语音预处理
# feature_extractor = AutoFeatureExtractor.from_pretrained(r"C:\jhq\huggingface_model\facebook\wav2vec2-base")
# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train", cache_dir="C:/jhq/huggingface_dataset", trust_remote_code=True)
# dataset = load_dataset(r"C:\jhq\huggingface_dataset\PolyAI\minds14", name="en-US", split="train", trust_remote_code=True)
# # 返回三个对象
# #   # array: 加载的语音信号
# #   # path: 指向音频文件的位置
# #   # sampling_rate: 音频的采样率
# print(dataset[0]["audio"])
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# print(dataset[0]["audio"])
# # audio_input = [dataset[0]["audio"]["array"]]
# # feature_extractor(audio_input, sampling_rate=16000)
# # print(dataset[0]["audio"]["array"].shape)
# # print(dataset[1]["audio"]["array"].shape)
# def preprocess_function(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = feature_extractor(audio_arrays, sampling_rate=16000, padding=True, max_length=100000, truncation=True)
#     return inputs
# print(type(dataset))
# processed_dataset = preprocess_function(dataset[:5])
# print(processed_dataset["input_values"][0].shape)
# print(processed_dataset["input_values"][1].shape)

# 多模态任务预处理， 待测
# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

# 加载预训练模型
# 相同模型可能适用多种任务， 待测
# model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
# model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")

