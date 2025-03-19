'''
Author: jhq
Date: 2025-03-17 21:40:05
LastEditTime: 2025-03-19 12:09:54
Description: 
'''
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from PIL import Image
import torch.nn.functional as F
import os
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
from pathlib import Path
import glob


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-anime-faces-96"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    # hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    # hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
dataset = load_dataset(
    "imagefolder", data_dir=r"D:\dataset\diffusion-anime-face\lhy-faces-96")
# fig, axs = plt.subplots(1, 4, figsize=(16, 8))
# for i, image in enumerate(dataset['train'][:4]['image']):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# plt.show()


# dataset['train'] = dataset['train'].shuffle(seed=config.seed).select(range(0, 10000))
dataset['train'] = dataset['train'].select(range(0, 10000))

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(example):
    images = [preprocess(image.convert("RGB")) for image in example["image"]]
    return {"images": images}


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(
    dataset['train'], batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
sample_image = dataset["train"][0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
)
# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
# show_img = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
# show_img.show()

# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)
# print("Loss:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=config.num_epochs * len(train_dataloader),
)


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    ).images
    grid = make_image_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps,
                                   return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
                0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(
                model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


args = (config, model, noise_scheduler,
        optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)

# sample_image = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
# Image.open(sample_image[-1])
