'''
Author: jhq
Date: 2025-02-11 13:28:12
LastEditTime: 2025-02-12 13:09:47
Description: 
'''
import vision_agent.tools as T
import matplotlib.pyplot as plt
import time
import cv2
import os
import json

start_time = time.time()
image = T.load_image(
    r"D:\data\detection\fire\all_image\not_label\fire_smoke_0000001.jpg")
dets = T.countgd_object_detection("fire, smoke", image)
print(dets)
# visualize the countgd bounding boxes on the image
viz = T.overlay_bounding_boxes(image, dets)

labelme_json = {
    'version': '5.5.0',
    'flags': {},
    'shapes': [],
    'imagePath': None,
    'imageData': None,
    'imageHeight': None,
    'imageWidth': None,
}
labelme_json['imagePath'] = r"D:\data\detection\fire\all_image\not_label\fire_smoke_0000001.jpg"
image_cv = cv2.imread(
    r"D:\data\detection\fire\all_image\not_label\fire_smoke_0000001.jpg")
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
json_name = "fire_smoke_0000001" + ".json"
json_path = r"D:\data\detection\fire\all_label\train"
json_name_path = os.path.join(json_path, json_name)
fd = open(json_name_path, "w")
json.dump(labelme_json, fd, indent=4)
fd.close()
print(f"fire_smoke_0000001 done, save json = {json_name_path}")

print("Detection took {:.3f}s".format(time.time() - start_time))
# save the visualization to a file
# T.save_image(viz, "pyramid_detected.png")
# display the visualization
plt.imshow(viz)
plt.show()
