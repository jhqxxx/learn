from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
from PIL import Image
import numpy as np

model = UNet2DModel.from_pretrained(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\unet", use_safetensors=True)

sample_size = model.config.sample_size
example_input = torch.randn(1, 3, sample_size, sample_size)
t = torch.tensor(1000)
print(t)
save_onnx_path = r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\unet\unet.onnx"
with torch.no_grad():
    torch.onnx.export(model, (example_input, t), save_onnx_path)
