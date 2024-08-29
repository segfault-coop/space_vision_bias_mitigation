import torch
import torch.nn as nn
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage
import numpy as np
from timm.models.vision_transformer import PatchEmbed,Block

class SimpleMAE(nn.Module):
    def __init__(self,img_size = 224, img_size_p = (32,32), embed_dim = 768, patch_size = (32,32), num_heads = 12, encoder_num_layers = 24, decoder_num_heads = 8, decoder_emb_size = 512, batch_idx = 1):
        super().__init__()
        self.batch_idx = batch_idx
        self.patch_size = patch_size[0]
        self.decoder_emb_size = decoder_emb_size
        
        self.pe = PatchEmbed(img_size=img_size_p, patch_size=patch_size, embed_dim=embed_dim, in_chans=3)
        self.num_patches = (img_size // patch_size[0]) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(self.num_patches, embed_dim), requires_grad = False)
        self.cls_token = nn.Parameter(torch.zeros(1,embed_dim))
        self.encoder = Block(dim=embed_dim,num_heads=12)
        self.decoder_embed = nn.Linear(embed_dim, decoder_emb_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1,decoder_emb_size))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(self.num_patches, decoder_emb_size))
        
        self.decoder_block = Block(decoder_emb_size, decoder_num_heads, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
        
        self.decoder_norm = nn.LayerNorm(decoder_emb_size)
        self.decoder_pred = nn.Linear(decoder_emb_size, self.patch_size ** 2 * 3, bias = True)
        
        
    def forward(self, img):
        ## ENCODER
        device = img.device
        self.tokens = img.shape[0]
        mask = torch.zeros(self.tokens, device=device).long()
        if self.batch_idx == 1:
            mask_idx = torch.arange(0,7)
        elif self.batch_idx == 2:
            mask_idx = torch.arange(7,14)
        elif self.batch_idx == 3:
            mask_idx = torch.arange(14,21)
        else:
            raise ValueError("Invalid batch_idx, 1,2,3 supported")
        
        mask[mask_idx] = 1
        input_tokens = self.pe(img[~mask.bool(),...]).squeeze(1)
        pos_input_tokens = input_tokens + self.pos_embed[~mask.bool(),...]
        self.tokens = torch.cat([self.cls_token, pos_input_tokens])
        encodings = self.encoder(self.tokens.unsqueeze(0)).squeeze(0)
        
        ## Decoder
        decoder_tokens = self.decoder_embed(encodings)
        tokens = torch.zeros((self.num_patches, self.decoder_emb_size),device=device)
        tokens[~mask.bool()] = decoder_tokens[1:,:]
        tokens[mask.bool()] = torch.cat([self.mask_token] * (self.num_patches - (~mask.bool()).sum()))
        
        tokens = tokens + self.decoder_pos_embed
        tokens = torch.cat([decoder_tokens[:1,:], tokens])
        
        out_tokens = self.decoder_block(tokens.unsqueeze(0)).squeeze(0)
        
        out = self.decoder_norm(out_tokens.unsqueeze(0)).squeeze(0)
        out = self.decoder_pred(out)
        
        final_pred = out[1:,:]
        return final_pred, mask
    
    
class HydraMAE(nn.Module):
    def __init__(self, num_heads = 3, train_mode = False):
        super().__init__()
        self.num_heads = num_heads
        self.mae_blocks = nn.ModuleList()
        for i in range(self.num_heads):
            head = SimpleMAE(batch_idx=(i+1))
            self.mae_blocks.append(head)
        
        if train_mode == True:
            for h in self.mae_blocks:
                h.train()
        
        
    def forward(self, x):
        i1,i2,i3 = x[0],x[1],x[2]
        o1,m1 = self.mae_blocks[0](i1)
        o2,m2 = self.mae_blocks[1](i2)
        o3,m3 = self.mae_blocks[2](i3)
        outputs = torch.stack([o1,o2,o3])
        masks = torch.stack([m1,m2,m3])
        return outputs, masks
        
    @staticmethod
    def hydra_loss(inputs, outputs, masks, pwr_weight):
        inputs = [inp.flatten(1) for inp in inputs]
        inputs = torch.stack(inputs)
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1,keepdim=True)
        inputs = ((inputs - mean) / (var + 1.e-6) ** .5)
        token_lvl_loss = ((outputs - inputs) ** 2).mean(dim=-1)
        loss_01 = (token_lvl_loss * masks).sum() / masks.sum()
        return loss_01
        