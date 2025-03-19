from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
from PIL import Image
import numpy as np

model = UNet2DModel.from_pretrained(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\unet", use_safetensors=True).to("cuda")
noise_scheduler = DDPMScheduler.from_pretrained(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\scheduler")

noise_scheduler.set_timesteps(50)
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise
for t in noise_scheduler.timesteps:
    print(t)
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = noise_scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.show()

# model.save_pretrained(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\unet", safe_serialization=False)