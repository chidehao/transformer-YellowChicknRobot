import torch
import torch.nn as nn
from model.modelling import *
from model.dataset import *
from tqdm import tqdm
Params = {
    'EPOCH':1,
    'lr' :1e-4,
    'train_path':'src/xiaohuangji50w_nofenci.conv',
    'batch_size':32
}
device = torch.device('cuda')
transformer = Transformer(Config).to(device)
optimizer = torch.optim.Adam(transformer.parameters(),lr=Params['lr'])
loss_func = nn.CrossEntropyLoss()
xiaoHJDataset = XiaoHJDataset(Params['train_path'])
train_loader = DataLoader(xiaoHJDataset,batch_size=Params['batch_size'],shuffle=True,pin_memory=True)
bar = tqdm(range(1,Params['EPOCH'] + 1))
for epoch in bar:
    for i,batch in enumerate(train_loader):
        transformer.train()
        optimizer.zero_grad()
        srcs,trgs,src_masks,trg_masks = batch
        srcs = srcs.to('cuda');trgs = trgs.to('cuda');src_masks = src_masks.to('cuda');trg_masks = trg_masks.to('cuda')
        dec_logit = transformer(srcs,trgs,src_masks,trg_masks)
        loss = loss_func(dec_logit,trgs)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            bar.set_postfix(Epoch = epoch,batch = i,Loss = loss.data.item())
