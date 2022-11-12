'''
Descripttion: 
version: 
Author: jhq
Date: 2022-10-22 17:26:00
LastEditors: jhq
LastEditTime: 2022-10-22 17:34:32
'''

import torchvision.transforms as transforms

images_size = (224, 224)
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(images_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])