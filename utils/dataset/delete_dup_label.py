'''
Author: jhq
Date: 2025-02-15 12:34:04
LastEditTime: 2025-02-15 12:54:20
Description: 
'''
import os
import shutil

label_path = r"D:\data\detection\fire\all_label\label_original"
image_path = r"D:\data\detection\fire\all_image\train_original"

image_list = os.listdir(image_path)
label_list = os.listdir(label_path)

surfix = ['.jpg', '.png', '.jpeg']

for label_name in label_list:
    name = label_name.split('.')[0]
    exist = False
    for sur in surfix:
        if name + sur in image_list:
            exist = True
            break
    if not exist:
        label_dup = os.path.join(label_path, label_name)
        print(f'删除label: {label_dup}')
        os.remove(label_dup)
    
    