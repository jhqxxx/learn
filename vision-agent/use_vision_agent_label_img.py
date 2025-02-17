'''
Author: jhq
Date: 2025-02-11 17:05:34
LastEditTime: 2025-02-12 21:12:37
Description: 
'''
import vision_agent.tools as T
import os
import cv2
import json
import time

image_path = r"D:\data\detection\fire\all_image\train"
json_path = r"D:\data\detection\fire\all_label\train"
file_list = os.listdir(image_path)


for file in file_list:
    image = T.load_image(os.path.join(image_path, file))
    dets = T.countgd_object_detection("fire, smoke", image)
    labelme_json = {
        'version': '5.5.0',
        'flags': {},
        'shapes': [],
        'imagePath': None,
        'imageData': None,
        'imageHeight': None,
        'imageWidth': None,
    }
    labelme_json['imagePath'] = os.path.join(image_path, file)
    image_cv = cv2.imread(os.path.join(image_path, file))
    h, w = image.shape[:2]
    labelme_json['imageHeight'] = h
    labelme_json['imageWidth'] = w

    for det in dets:
        shape = {
            "label": det["label"],
            "points": [
                [float(det["bbox"][0]*w), float(det["bbox"][1]*h)],
                [float(det["bbox"][2]*w), float(det["bbox"][3]*h)]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        labelme_json["shapes"].append(shape)
    json_name = file.split(".")[0] + ".json"
    json_name_path = os.path.join(json_path, json_name)
    fd = open(json_name_path, "w")
    json.dump(labelme_json, fd, indent=4)
    fd.close()
    print(f"{file} done, save json = {json_name_path}")

    # visualize the countgd bounding boxes on the image
    # viz = T.overlay_bounding_boxes(image, dets)
