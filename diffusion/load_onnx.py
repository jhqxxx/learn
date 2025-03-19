import numpy as np
import onnxruntime as ort
from PIL import Image
from diffusers import DDPMScheduler
import torch

ort_session = ort.InferenceSession(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\unet\unet.onnx")

noise_scheduler = DDPMScheduler.from_pretrained(r"C:\jhq\learn\diffusion\ddpm-anime-faces-64\scheduler")

input_name_1 = ort_session.get_inputs()[0].name
input_name_2 = ort_session.get_inputs()[1].name
output_name = ort_session.get_outputs()[0].name

print(f"input_name_1:{input_name_1}, input_name_2:{input_name_2}, output_name:{output_name}")
noise = np.random.randn(1, 3, 64, 64)
noise = noise.astype(np.float32)
print(noise.shape)
# step生成是线性递减的，不是随机的
noise_scheduler.set_timesteps(50)
t_step = np.linspace(980, 0, 50).astype(np.int32)
print(t_step)
print(noise_scheduler.timesteps)
for t in t_step:
    # print(t)
    # t也需要传入tensor，如果直接传int会报错
    outputs = ort_session.run(None, 
                              {input_name_1: noise, input_name_2: np.array(t).astype(np.int64)})
    # outputs维度为[1,1,3,64,64],多了一个维度，需要处理一下
    noisy_residual = torch.from_numpy(np.array(outputs[0]))
    input = torch.from_numpy(noise)
    previous_noisy_sample = noise_scheduler.step(noisy_residual, t, input).prev_sample
    noise = previous_noisy_sample.numpy()

noise = torch.from_numpy(noise)
image = (noise / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.show()
    
    
    