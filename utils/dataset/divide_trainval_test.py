import os 
import shutil 
import random
from tqdm import tqdm

'''
train val need label
test not need label
'''
image_path = r"D:\data\detection\fire\all_image\train"   
file_list = os.listdir(image_path) 

test_ratio = 0.2
random.seed(3)

img_path = os.listdir(image_path) 
random.shuffle(img_path)

test_number = int(len(img_path) * test_ratio) 
trainval_files = img_path[test_number:] 
test_files = img_path[:test_number]

# image_train_path = os.path.join(image_path, "train") 
# if not os.path.exists(image_train_path):     
#     os.mkdir(image_train_path) # train 
# for file in tqdm(trainval_files):     
#     shutil.move(os.path.join(image_path, file), image_train_path)  

image_test_path = os.path.join(r"D:\data\detection\fire\all_image", "train02") 
# if not os.path.exists(image_test_path): # test     
#     os.mkdir(image_test_path) 
for file in tqdm(trainval_files):     
    shutil.move(os.path.join(image_path, file), image_test_path)