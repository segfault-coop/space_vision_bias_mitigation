import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage
import matplotlib.pyplot as plt
import wandb
import datasets
import pandas as pd
from tqdm import tqdm
def wandb_init(config):
    project_name = "temporal_mae"
    wandb.init(
        project=project_name,
    )

def create_patches(img: Image, patch_size: int) -> torch.Tensor:
    img = ToTensor()(img)
    imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(
        3).permute((3, 0, 1, 2))
    return imgp


def show_patches(img: torch.Tensor) -> None:
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

def transforms(img_size:int):
    return Compose([RandomResizedCrop(size=img_size, scale=[0.4, 1], ratio=[0.75, 1.33], interpolation=2), 
                    RandomHorizontalFlip(p=0.5), 
                    ToTensor()])

def load_ds(imgs, patch_size, transforms):
    img = transforms(imgs)
    imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(3).permute((3, 0, 1, 2))
    return imgp

def create_tokens_and_mask(imgp, batch_idx):
    tokens = imgp.shape[0]
    mask = torch.zeros(tokens).long()
    if batch_idx == 1:
        mask_idx = torch.arange(0,7)
    elif batch_idx == 2:
        mask_idx = torch.arange(7,14)
    elif batch_idx == 3:
        mask_idx = torch.arange(14,21)
    else:
        raise ValueError("Invalid batch_idx, 1,2,3 supported")
    
    mask[mask_idx] = 1
    print(mask)
    return imgp, mask 

def main():
    data_imgs = datasets.load_dataset("spacevision-upb/MOD_LSTD_E_downscaled")
    idx_df = pd.read_csv('local_ds.csv')['hf_idx']
    imgs = data_imgs['train']['image']
    imgs = [imgs[i] for i in idx_df]
    imgs = [load_ds(img,patch_size = 32, transforms=transforms(224)) for img in imgs]

    from Hydra_MAE import SimpleMAE, HydraMAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HydraMAE(train_mode=True).to(device)
    print(model)
    if not list(model.parameters()):
        raise ValueError("The model has no parameters to optimize.")
    # out, o_mask = model(test_imgs)
    # print(out.shape)
    # print(o_mask.shape)
    # ll = HydraMAE.hydra_loss(test_imgs,out,o_mask,pwr_weight)
    # print(ll)
    pwr_weight = [1,1,1]
    total_epochs = 20
    loss_fn = HydraMAE.hydra_loss
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    optim.zero_grad()
    avg_losses = []
    for e in range(total_epochs):
        model.train()
        losses = []
        progress_bar = tqdm(range(0, len(imgs), 3), desc=f"Epoch {e+1}/{total_epochs}", unit="batch")
        
        for i in progress_bar:
            img_chunk = imgs[i:i+3]
            if len(img_chunk) < 3:
                raise Exception("Smth went v wrong")
            img_chunk_cuda = [img.to(device) for img in img_chunk]
            # img_chunk = [img.to(device) for img in img_chunk]
            out, o_mask = model(img_chunk_cuda)
            out = out.to("cpu")
            o_mask = o_mask.to("cpu")
            ll = loss_fn(img_chunk, out, o_mask, pwr_weight)
            # print(f"Loss {ll}")
            losses.append(ll.item())
            optim.zero_grad()
            ll.backward()
            optim.step()
            progress_bar.set_postfix(loss=ll.item())
        
        avg_loss = np.mean(losses)
        avg_losses.append(avg_loss)
        print(f"Epoch {e+1}/{total_epochs}, Loss: {avg_loss}")
    
    torch.save(model.state_dict(), 'model_v2.pth')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs + 1), avg_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Time')
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()