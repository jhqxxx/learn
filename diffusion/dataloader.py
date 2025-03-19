'''
Author: jhq
Date: 2025-03-17 13:33:43
LastEditTime: 2025-03-17 13:34:28
Description: 
'''
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms
        img_list = os.listdir(image_paths)
        # mini_size = int(len(img_list)*0.01)
        # img_list = img_list[0:mini_size]
        self.imgs_path_list = [os.path.join(
            image_paths, img) for img in img_list]

    def __len__(self):
        return len(self.imgs_path_list)

    def __getitem__(self, index):
        img = read_image(self.imgs_path_list[index])
        img = img.float() / 255
        # print(f"type img:{type(img)}, shape: {img.shape}, dtype: {img.dtype}")
        if self.transforms:
            img = self.transforms(img)

        return img


def load_transformed_dataset(train_path, val_path, img_size, batch_size):
    train_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageDataset(train_path, train_data_transform)
    val_dataset = ImageDataset(val_path, val_data_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
