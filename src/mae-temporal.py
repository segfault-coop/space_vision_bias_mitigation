import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage
import os


img_size = 224
patch_size = 32

imgs = os.listdir('data/MOD_LSTD_E_downscaled')

# Get the first img
img_file = imgs[0]
img = Image.open(f'data/MOD_LSTD_E_downscaled/{img_file}')
img = img.resize((img_size, img_size))

imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(3).permute((3, 0, 1, 2))
