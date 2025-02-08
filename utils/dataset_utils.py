import os
from os.path import join
import os.path as osp
import numpy as np
import shutil

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

data_path = r"D:\data\garden_staff"

train_path = join(data_path, 'train')
test_path = join(data_path, 'test')
val_path = join(data_path, 'val')
os.makedirs(test_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

data_clas = sorted(os.listdir(train_path))
num_test = 0
num_val = 0
for clas in data_clas:
    test_clas_path = join(test_path, clas)
    val_clas_path = join(val_path, clas)
    os.makedirs(test_clas_path, exist_ok=True)
    os.makedirs(val_clas_path, exist_ok=True)
    train_clas_path = join(train_path, clas)
    clas_img_list = os.listdir(train_clas_path)
    clas_num = len(clas_img_list)
    train_array = np.arange(0, clas_num)
    np.random.shuffle(train_array)   # np.random.shuffle()作用于原函数，返回 None
    test_array = train_array[0:int(clas_num*test_ratio)]    
    val_array = train_array[-int(clas_num*val_ratio):-1]
    for i in test_array:
        src_file = join(train_clas_path, clas_img_list[i])
        shutil.move(src_file, join(test_clas_path, clas_img_list[i]))
        num_test += 1
    for i in val_array:
        src_file = join(train_clas_path, clas_img_list[i])
        shutil.move(src_file, join(val_clas_path, clas_img_list[i]))
        num_val += 1

print(f'test:{num_test}\nval:{num_val}')

