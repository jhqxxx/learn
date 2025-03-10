from transformers import AutoImageProcessor
from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose
import matplotlib.pyplot as plt

# 加载数据集
dataset = load_dataset(r"C:\jhq\huggingface_dataset\ethz\food101", split="train[:100]")
dataset[0]["image"].show()

# 创建图像处理器
image_processor = AutoImageProcessor.from_pretrained(r"C:\jhq\huggingface_model\google\vit-base-patch16-224")
size = (
    image_processor.size["shortest_edge"] 
    if "shortest_edge" in image_processor.size 
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose(
    [
        RandomResizedCrop(size=size),
        ColorJitter(brightness=0.5, hue=0.5)
    ])

def transforms(examples):
    images = [_transforms(image.convert("RGB")) for image in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples

dataset.set_transform(transforms)
print(dataset[0].keys())


img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0).numpy())
plt.show()