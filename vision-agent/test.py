'''
Author: jhq
Date: 2025-02-11 13:28:12
LastEditTime: 2025-02-11 13:45:19
Description: 
'''
import vision_agent.tools as T 
import matplotlib.pyplot as plt 
import time

start_time = time.time()
image = T.load_image(r"D:\data\detection\fire\fire_Image\Normal_Images_2\_91175853_a53933fd-a2e4-48aa-bbc5-735e4110cb0b.jpg") 
dets = T.countgd_object_detection("smoke", image) 
# visualize the countgd bounding boxes on the image 
viz = T.overlay_bounding_boxes(image, dets) 
print("Detection took {:.3f}s".format(time.time() - start_time))
# save the visualization to a file 
# T.save_image(viz, "pyramid_detected.png")  
# display the visualization 
plt.imshow(viz) 
plt.show()