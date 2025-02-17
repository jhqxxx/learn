import os 
import shutil

image_path = r"D:\data\detection\fire\all_image"

file_list = os.listdir(image_path)
start = 0
for file in file_list:
    prefix = file.split(".")[-1]
    shutil.move(os.path.join(image_path, file), os.path.join(image_path, f"fire_smoke_{start:07}.{prefix}"))
    start += 1
