'''
Author: jhq
Date: 2022-11-23 12:28:03
LastEditTime: 2025-03-16 20:26:21
Description: 
'''
import os
import glob
from PIL import Image
import numpy as np
import random

data_path = r"D:\dataset\diffusion-anime-face\anime-faces64x64\val"
img_list = os.listdir(data_path)
random.shuffle(img_list)
result = []
for file in img_list[:5000]:
    img = Image.open(os.path.join(data_path, file)).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = img/255.
    result.append(img)

print(np.shape(result))
mean = np.mean(result, axis=(0, 1, 2))   # 对RGB三通道分别求均值
std = np.std(result, axis=(0, 1, 2))
print(f'mean:{mean}')
print(f'std:{std}')
