'''
Author: jhq
Date: 2025-02-16 16:04:42
LastEditTime: 2025-02-16 16:49:03
Description: 
'''
from ultralytics import YOLO

# model = YOLO(r"C:\jhq\learn\yolo-series\runs\detect\train12\weights\best.pt")
# fire_smoke_yolo11n_best.pt
model = YOLO("fire_smoke_yolo11n_best.pt")
model.export(format="onnx")