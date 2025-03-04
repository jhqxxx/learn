'''
Author: jhq
Date: 2025-02-27 12:40:11
LastEditTime: 2025-03-03 17:54:45
Description: 
'''
from ultralytics import YOLO

model = YOLO("yolo11n.onnx")
results = model.track(source=r"D:\messy\7df65dc3f1643786ada6b17c01ec04fd.mp4", conf=0.3, iou=0.5, show=True)