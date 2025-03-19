'''
Author: jhq
Date: 2025-03-14 22:40:19
LastEditTime: 2025-03-14 23:45:43
Description: 线性插值放大
'''
import cv2
import os

def scale_image(image_path, new_height, new_width, new_path):
    image = cv2.imread(image_path)
    enlarged_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(new_path, enlarged_image)

original_image_dir = r"D:\dataset\diffusion-anime-face\anime-faces64x64"
new_image_dir = r"D:\dataset\diffusion-anime-face\anime-faces64-to-96"

original_image_list = os.listdir(original_image_dir)
for image_name in original_image_list:
    image_path = os.path.join(original_image_dir, image_name)
    new_image_path = os.path.join(new_image_dir, image_name)
    scale_image(image_path, 96, 96, new_image_path)

# # 读取原始图像
# image = cv2.imread(r"D:\dataset\diffusion-anime-face\anime-faces64x64\1.png")

# # 获取原始图像尺寸
# height, width = image.shape[:2]

# # 设置放大倍数
# scale_factor = 2

# # 计算新尺寸
# new_height = height * scale_factor
# new_width = width * scale_factor

# # 使用双三次插值进行放大
# enlarged_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)

# # 保存放大后的图像
# cv2.imwrite('enlarged.jpg', enlarged_image)
# cv2.imshow('enlarged.jpg', enlarged_image)
# cv2.waitKey(0)