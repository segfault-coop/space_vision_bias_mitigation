import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt

def create_patches(img: Image, patch_size: int) -> torch.Tensor:
    img = ToTensor()(img)
    imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(3).permute((3, 0, 1, 2))
    return imgp

def show_patches(img:torch.Tensor) -> None:
    fix, ax = plt.subplots(figsize=(4, 4), nrows=8, ncols=8)
    for n, i in enumerate(img):
        ax.flat[n].imshow(ToPILImage()(i))
        ax.flat[n].axis("off")
    plt.show()

patch_size = 32

imgs = os.listdir('data/MOD_LSTD_E_downscaled')

# Get the first img
img_file_batch = imgs[0:3]
img_batch = [Image.open(f'data/MOD_LSTD_E_downscaled/{img}') for img in img_file_batch]

patches_imgs = [create_patches(img, patch_size) for img in img_batch]

idx = 8

mask = torch.ones_like(patches_imgs[0])
mask[:idx] = 0

mask_2 = torch.ones_like(patches_imgs[1])
mask_2[idx:idx*2] = 0
print(mask.shape)


mask_3 = torch.ones_like(patches_imgs[2])
mask_3[idx*2:idx*3] = 0

masked_img1 = patches_imgs[0] * mask
masked_img2 = patches_imgs[1] * mask_2
masked_img3 = patches_imgs[2] * mask_3

show_patches(masked_img1)
show_patches(masked_img2)
show_patches(masked_img3)
