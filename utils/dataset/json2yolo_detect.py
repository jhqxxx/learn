'''
Author: jhq
Date: 2025-01-08 15:51:22
LastEditTime: 2025-01-08 16:24:06
Description: 
'''
import os 
import json 
import numpy as np 
from tqdm import tqdm  
bbox_class = {     
    "card": 0 
}

Dataset_root = r"D:\data\card-corner\detect" 
labelme_path = os.path.join(Dataset_root, "labelme_detect") 
txt_path = os.path.join(Dataset_root, "labels") 
txt_train_path = os.path.join(txt_path, "train")
txt_val_path = os.path.join(txt_path, "val") 
os.mkdir(txt_path) 
os.mkdir(txt_train_path) 
os.mkdir(txt_val_path)

with open(os.path.join(txt_path, "classes.txt"), "w", encoding="utf-8") as f:
    for each in list(bbox_class.keys()):         
        f.write(each + "\n")

def process_single_json(labelme_file, save_foler):    
    with open(labelme_file, "r", encoding="utf-8") as f:         
        labelme = json.load(f)     
    img_width = labelme["imageWidth"]     
    img_height = labelme["imageHeight"]     
    suffix = os.path.basename(labelme_file).split(".")[-2]     
    yolo_txt_path = os.path.join(save_foler, suffix + ".txt")

    with open(yolo_txt_path, "w", encoding="utf-8") as f:         
        for shape in labelme["shapes"]:             
            if shape["shape_type"] == "rectangle":                 
                yolo_str = ""                 
                bbox_class_id = bbox_class[shape["label"]]                 
                yolo_str += "{}".format(bbox_class_id)

                bbox_top_left_x = int(min(shape["points"][0][0], shape["points"][1][0]))                 
                bbox_bottom_right_x = int(max(shape["points"][0][0], shape["points"][1][0]))                 
                bbox_top_left_y = int(min(shape["points"][0][1], shape["points"][1][1]))                 
                bbox_bottom_right_y = int(max(shape["points"][0][1], shape["points"][1][1]))                  
                
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)                 
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)                  
                
                bbox_width = int(bbox_bottom_right_x - bbox_top_left_x)                 
                bbox_height = int(bbox_bottom_right_y - bbox_top_left_y)                  
                
                bbox_center_x_norm = bbox_center_x / img_width                 
                bbox_center_y_norm = bbox_center_y / img_height                  
                
                bbox_width_norm = bbox_width / img_width                 
                bbox_height_norm = bbox_height / img_height
                yolo_str += " {:.5f} {:.5f} {:.5f} {:.5f}".format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                f.write(yolo_str + "\n")

    print('{} --> {} 转换完成'.format(labelme_file, yolo_txt_path))

for aa in ["train", "val"]:
    for labelme_file in tqdm(os.listdir(os.path.join(labelme_path, aa))):
        try:
            process_single_json(os.path.join(labelme_path, aa, labelme_file), os.path.join(txt_path, aa)) 
        except Exception as e:
            print("{} 转换失败 {}".format(labelme_file, e))