'''
Author: jhq
Date: 2025-02-12 21:09:22
LastEditTime: 2025-02-14 13:09:51
Description: 
'''
from ultralytics import YOLO
import os
import cv2
import json

model = YOLO(r"C:\jhq\learn\yolo-series\runs\detect\train10\weights\best.pt")

image_path = r"D:\data\detection\fire\all_image\yololabeled"
json_path = r"D:\data\detection\fire\all_label\yololabeled"
file_list = os.listdir(image_path)

CLS_MAPPING = {
    0: {
        "en": 'fire',
        "cn": '火'
    },
    1: {
        "en": "smoke",
        "cn": "烟雾"
    }
}

for file in file_list:
    # image = T.load_image(os.path.join(image_path, file))
    # dets = T.countgd_object_detection("fire, smoke", image)
    results = model(os.path.join(image_path, file), device='0', save=False, conf=0.5)
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
    h, w = image_cv.shape[:2]
    labelme_json['imageHeight'] = h
    labelme_json['imageWidth'] = w

    result = results[0]
    for det in result.boxes:
        cls = det.cls.int().item()
        shape = {
            "label": CLS_MAPPING[cls]["en"],
            "points": [
                [det.xyxy[0, 0].int().item(), det.xyxy[0, 1].int().item()],
                [det.xyxy[0, 2].int().item(), det.xyxy[0, 3].int().item()]
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