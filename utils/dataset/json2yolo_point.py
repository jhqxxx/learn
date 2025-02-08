'''
Author: jhq
Date: 2025-01-06 20:33:15
LastEditTime: 2025-01-08 14:24:45
Description:  point json2yolo 
'''
import os
import json
import numpy as np
from tqdm import tqdm

bbox_class = {
    "card": 0
}

keypoint_class = [ "left_up", "left_bottom", "right_up", "right_bottom"]

Dataset_root = r"D:\data\card-corner"
labelme_path = os.path.join(Dataset_root, "labelme_labels")
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
    yolo_txt_path = os.path.join(save_foler, suffix + ".txt")

    with open(yolo_txt_path, "w", encoding="utf-8") as f:
        for shape in labelme["shapes"]:
            if shape["shape_type"] == "rectangle":
                yolo_str = ""
                bbox_class_id = bbox_class[shape["label"]]
                yolo_str += "{}".format(bbox_class_id)

                # 
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

                bbox_keypoints_dict = {}

                for shape2 in labelme["shapes"]:
                    if shape2["shape_type"] == "point":
                        x = int(shape2["points"][0][0])
                        y = int(shape2["points"][0][1])
                        label = shape2["label"]
                        # 判断是否在bbox内
                        # 标注时尽量确保一个框里只有一个目标物体，目标物体的关键点，每个类别的点框里只有一个或者没有
                        if (x >= bbox_top_left_x and x <= bbox_bottom_right_x and y >= bbox_top_left_y and y <= bbox_bottom_right_y):
                            bbox_keypoints_dict[label] = [x, y]

                for each_class in keypoint_class:
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += " {:.5f} {:.5f} {}".format(keypoint_x_norm, keypoint_y_norm, 2)
                    else:
                        yolo_str += " 0 0 0"
                f.write(yolo_str + "\n")
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