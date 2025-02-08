'''
Author: jhq
Date: 2022-11-23 12:28:03
LastEditTime: 2022-11-23 13:14:03
Description: 
'''
import os
import glob
from PIL import Image
import numpy as np

data_path = r"D:\data\garden_staff"
train_files = glob.glob(os.path.join(data_path, 'train', '*', '*.jpg'))
result = []
for file in train_files:
    img = Image.open(file).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = img/255.
    result.append(img)

print(np.shape(result))
mean = np.mean(result, axis=(0, 1, 2))   # 对RGB三通道分别求均值
std = np.std(result, axis=(0, 1, 2))
print(f'mean:{mean}')
print(f'std:{std}')
