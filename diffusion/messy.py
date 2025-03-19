'''
Author: jhq
Date: 2025-03-19 10:18:13
LastEditTime: 2025-03-19 10:18:18
Description: 
'''
from datasets import load_dataset
import matplotlib.pyplot as plt


dataset = load_dataset(
    "imagefolder", data_dir=r"D:\dataset\diffusion-anime-face\lhy-faces-96")
fig, axs = plt.subplots(1, 4, figsize=(16, 8))
print(dataset.keys())
for i, image in enumerate(dataset['train'][:4]['image']):
    axs[i].imshow(image)
    axs[i].set_axis_off()
plt.show()