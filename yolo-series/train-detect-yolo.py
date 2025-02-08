'''
Author: jhq
Date: 2025-01-03 16:49:44
LastEditTime: 2025-01-08 16:27:37
Description: use yolo11 train dog cat detect model
'''

from ultralytics import YOLO
# train5-v11 dogcat
model = YOLO(r"D:\code\ultralytics\ultralytics\cfg\models\11\yolo11n.yaml") # 11 
# model = YOLO(r"D:\code\ultralytics\ultralytics\cfg\models\v10\yolov10n.yaml") # 10 train6
# model = YOLO(r"D:\code\ultralytics\ultralytics\cfg\models\v9\yolov9s.yaml") # 9 train8

# results = model.train(data="detect-dog-cat.yaml", epochs=100, imgsz=640, device=0, batch=16, workers=0)

results = model.train(data="card_detect.yaml", epochs=100, imgsz=640, device=0, batch=16, workers=0)