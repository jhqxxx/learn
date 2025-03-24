from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from modelscope import AutoModelForImageSegmentation
import time

model = AutoModelForImageSegmentation.from_pretrained(r'C:\jhq\huggingface_model\maple775885\RMBG-2___0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cpu')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(r"D:\messy\img\20250321111449.jpg")
input_images = transform_image(image).unsqueeze(0).to('cpu')

# Prediction
start_time = time.time()
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
print(time.time() - start_time)
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save(r"D:\messy\img\20250321111449_no_bg_image.png")