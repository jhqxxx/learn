'''
Author: jhq
Date: 2025-01-07 15:13:05
LastEditTime: 2025-01-07 15:21:03
Description: infer card point model
'''
from ultralytics import YOLO
model = YOLO(r"D:\code\learn\yolo-series\runs\pose\train\weights\best.pt")
results = model.predict(source=r"D:\messy\img\gold-dog.jpg")

for result in results:     
    boxes = result.boxes
    print(boxes)     
    result.show()     
    result.save(filename="yolo11n-pose-card-result.jpg")