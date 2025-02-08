'''
Author: jhq
Date: 2022-11-23 11:50:00
LastEditTime: 2022-11-23 13:18:07
Description: 
'''
import glob
import os
import os.path as osp
from os.path import join
from PIL import Image


desired_size = 224  # 图片缩放后的统一大小
data_path = r"D:\data\garden_staff"


def img_resize(data_path, desired_size=224):    
    data_dir = glob.glob(join(data_path, '*'))
    data_dir = [d for d in data_dir if osp.isdir(d)]

    dirs = []
    for i in data_dir:
        clas_dir = glob.glob(join(i, '*'))
        clas_dir = [d for d in clas_dir if osp.isdir(d)]
        dirs += clas_dir

    for path in dirs:
        files = glob.glob(join(path, '*jpg'))
        files += glob.glob(join(path, '*JPG'))
        files += glob.glob(join(path, '*png'))
        files += glob.glob(join(path, '*PNG'))
        files += glob.glob(join(path, '*jpeg'))

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            old_size = img.size
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new('RGB', (desired_size, desired_size)) #默认黑图，不满224*224尺寸的，用0像素填充
            new_im.paste(im, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))
            new_im.save(file.split('.')[0] + '.jpg')
    

