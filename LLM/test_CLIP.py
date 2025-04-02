'''
Author: jhq
Date: 2025-03-21 14:32:03
LastEditTime: 2025-03-26 13:53:35
Description: 
'''
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 加载预训练的CLIP模型和处理器
model = CLIPModel.from_pretrained(
    r"C:\jhq\huggingface_model\openai\clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained(
    r"C:\jhq\huggingface_model\openai\clip-vit-base-patch32")

# 加载图像并进行预处理
image1 = Image.open(r"D:\messy\img\20250322155726.png")  # 假设图片在当前目录下
inputs1 = processor(images=image1, return_tensors="pt")
# image2 = Image.open(
#     r"D:\messy\img\20250321111449_no_bg_image.png")  # 假设图片在当前目录下
# inputs2 = processor(images=image2, return_tensors="pt")

# image3 = Image.open(
#     r"D:\messy\img\0b90f72daab4c7081cce15fee985e9109e6fdbcd70a27d6da3af090d74db10e6.jpg")  # 假设图片在当前目录下
# inputs3 = processor(images=image3, return_tensors="pt")

# 提取图像特征
with torch.no_grad():
    image_features1 = model.get_image_features(**inputs1)
    # image_features2 = model.get_image_features(**inputs2)
    # image_features3 = model.get_image_features(**inputs3)

# 打印图像特征
print(image_features1)
# np.savetxt("data.txt", image_features1.numpy())
