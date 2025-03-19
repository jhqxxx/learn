'''
Author: jhq
Date: 2025-03-15 14:02:44
LastEditTime: 2025-03-16 19:55:32
Description: 
'''
import os
import shutil
import random
from tqdm import tqdm

image_dir = r"D:\dataset\diffusion-anime-face\anime-faces64x64"
train_path = os.path.join(image_dir, "train")
val_path = os.path.join(image_dir, "val")

img_list = os.listdir(image_dir)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

val_ratio = 0.3
random.shuffle(img_list)
val_num = int(len(img_list) * val_ratio)
train_imgs = img_list[val_num:]
val_imgs = img_list[:val_num]

for img in tqdm(train_imgs):
    if os.path.isfile(os.path.join(image_dir, img)):
        shutil.move(os.path.join(image_dir, img), os.path.join(train_path, img))

for img in tqdm(val_imgs):
    if os.path.isfile(os.path.join(image_dir, img)):
        shutil.move(os.path.join(image_dir, img), os.path.join(val_path, img))