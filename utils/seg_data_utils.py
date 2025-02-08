'''
Author: jhq
Date: 2022-11-27 16:35:48
LastEditTime: 2022-11-27 17:39:18
Description: 
'''
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
import cv2

image_size = (128, 128)
image_path = r'D:\data\segmentation\pet\images'
label_path = r'D:\data\segmentation\pet\annotations\trimaps'
num_class = 3

png_path = r"D:\data\segmentation\pet\annotations\trimaps\shiba_inu_135.png"
label = np.array(Image.open(png_path))
plt.imshow(label.astype('uint8'), cmap='gray')
plt.show()
# def data_generator(batch_size, is_train=True, image_path=image_path, label_path=label_path, image_size=image_size, num_class=num_class):
#     img_dirs = glob(join(image_path, '*jpg'))
#     np.random.shuffle(img_dirs)
#     images = np.zeros(batch_size, 3, image_size[0], image_size[1])
#     labels = np.zeros(batch_size, 3, image_size[0], image_size[1])
#     num = 0
#     for filename in img_dirs:
#         if filename.endswith('.jpg'):
#             img = Image.open(filename).resize(image_size).convert('RGB')
#             row_min = np.random.randint

# data_generator(8)