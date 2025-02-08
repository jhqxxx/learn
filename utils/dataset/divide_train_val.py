'''
Author: jhq
Date: 2025-01-06 20:02:32
LastEditTime: 2025-01-08 15:52:46
Description: datasets divide train and val
'''
import os
import shutil
import random
from tqdm import tqdm

Dataset_root = r"D:\data\card-corner\detect"
images_path = os.path.join(Dataset_root, "images")
json_path = os.path.join(Dataset_root, "labelme_detect")
test_ratio = 0.2
random.seed(0)

img_path = os.listdir(images_path)
random.shuffle(img_path)
val_number = int(len(img_path) * test_ratio)
train_files = img_path[val_number:]
val_files = img_path[:val_number]

print("all dataset number:", len(img_path))
print("train dataset number:", len(train_files))
print("val dataset number:", len(val_files))

image_train_path = os.path.join(images_path, "train")
if not os.path.exists(image_train_path):
    os.mkdir(image_train_path) # train
for file in tqdm(train_files):
    shutil.move(os.path.join(images_path, file), image_train_path)

image_val_path = os.path.join(images_path, "val")
if not os.path.exists(image_val_path): # val
    os.mkdir(image_val_path)
for file in tqdm(val_files):
    shutil.copy(os.path.join(images_path, file), image_val_path)

json_train_path = os.path.join(json_path, "train")
if not os.path.exists(json_train_path):
    os.mkdir(json_train_path)
for file in tqdm(train_files):
    json_file = file.replace("jpg", "json")
    shutil.move(os.path.join(json_path, json_file), json_train_path)

json_val_path = os.path.join(json_path, "val")
if not os.path.exists(json_val_path):
    os.mkdir(json_val_path)
for file in tqdm(val_files):
    json_file = file.replace("jpg", "json")
    shutil.move(os.path.join(json_path, json_file), json_val_path)