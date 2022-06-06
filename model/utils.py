import torch
import torch.nn as nn

def make_trg_mask(trg):
    mask_size = trg.size(1)
    trg_mask = torch.tril(torch.ones(mask_size,mask_size))
    return trg_mask

def make_pad_mask(attention_mask): #b,512
    attn_maskSize = attention_mask.size(1)
    attention_mask = attention_mask.eq(0).unsqueeze(1) # b 1 512
    pad_mask  = attention_mask.expand(attn_maskSize,attention_mask)
    return pad_mask
