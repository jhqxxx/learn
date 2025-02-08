import os 
import json 
import numpy as np 
from tqdm import tqdm


polygon_class = {     
    "card": 0 
}

Dataset_root = r"D:\data\card-corner\segmentation" 
labelme_path = os.path.join(Dataset_root, "labelme-segmentation") 
txt_path = os.path.join(Dataset_root, "labels") 
txt_train_path = os.path.join(txt_path, "train") 
txt_val_path = os.path.join(txt_path, "val") 
os.mkdir(txt_path) 
os.mkdir(txt_train_path) 
os.mkdir(txt_val_path)

def process_single_json(labelme_file, save_foler):
    with open(labelme_file, "r", encoding="utf-8") as f:
        labelme = json.load(f)
    img_width = labelme["imageWidth"]
    img_height = labelme["imageHeight"]
    suffix = os.path.basename(labelme_file).split(".")[-2]
    yolo_txt_path= os.path.join(save_foler, suffix + ".txt")

    with open(yolo_txt_path, "w", encoding="utf-8") as f:
        for shape in labelme["shapes"]:
            if shape["shape_type"] == "polygon":
                yolo_str = ""
                polygon_class_id = polygon_class[shape["label"]]
                yolo_str += "{}".format(polygon_class_id)
                for point in shape["points"]:
                    x = int(point[0])
                    y = int(point[1])
                    x_norm = x / img_width
                    y_norm = y / img_height
                    yolo_str += " {:.5f} {:.5f}".format(x_norm, y_norm)
                yolo_str += "\n"
                f.write(yolo_str)
    print('{} --> {} 转换完成'.format(labelme_file, yolo_txt_path))

for labelme_file in tqdm(os.listdir(os.path.join(labelme_path, "train"))):
    try:         
        process_single_json(os.path.join(labelme_path, "train", labelme_file), txt_train_path)     
    except:         
        print("{} 转换失败".format(labelme_file)) 
print("train 转换 yolo txt 完成")

for labelme_file in tqdm(os.listdir(os.path.join(labelme_path, "val"))):     
    try:                  
        process_single_json(os.path.join(labelme_path, "val", labelme_file), txt_val_path)          
    except:                  
        print("{} 转换失败".format(labelme_file))
print("val 转换 yolo txt 完成")