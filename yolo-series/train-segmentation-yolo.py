from ultralytics import YOLO

model = YOLO("yolo11n-seg.yaml")

results = model.train(data="card_segmentation.yaml", epochs=100, imgsz=640, device=0, batch=16, workers=0)