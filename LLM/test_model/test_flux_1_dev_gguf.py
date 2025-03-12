'''
Author: jhq
Date: 2025-03-11 12:28:47
LastEditTime: 2025-03-12 19:07:56
Description: 
'''
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import login


ckpt_path = (
    r"C:\jhq\huggingface_model\city96\FLUX.1-dev-gguf\flux1-dev-Q2_K.gguf"
)
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    r"C:\jhq\huggingface_model\black-forest-labs\FLUX___1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
image.save("flux-gguf.png")