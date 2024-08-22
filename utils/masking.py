import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
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

def mask_block(imgs, idx):
    for i, img in enumerate(imgs):
        mask = torch.ones_like(img)
        mask[i::idx] = 0
        imgs[i] = img * mask
    return imgs