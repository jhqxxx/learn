'''
Author: jhq
Date: 2025-01-04 14:22:11
LastEditTime: 2025-01-04 14:50:23
Description: 
'''
from ultralytics import YOLO

model = YOLO(r"C:\jhq\learn\yolo-series\runs\detect\train5\weights\best.pt")
# results = model([r"D:\messy\img\gold-dog.jpg", r"D:\messy\img\xishi-dog.jpg"])
# results = model([r"D:\messy\img\gold-dog.jpg", r"D:\messy\img\douniu-dog.jpg", r"D:\messy\img\xishi-dog.jpg"])
results = model(r"D:\messy\img\gold-dog.jpg")
                 
for result in results:
    boxes = result.boxes
    print(boxes)
    result.show()
    result.save(filename="yolo11n-dog-cat-result.jpg")