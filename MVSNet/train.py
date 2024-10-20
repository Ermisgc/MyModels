import time

import torch
import numpy as np
from mvsnet import MVSNet
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from dataset import MVSNetDTUDataset
from torch.optim import Adam
import loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# imgs_ori = torch.rand(1, 10, 3, 512, 512).to(device)
# homo_matrices = torch.rand(1, 10, 4, 4).to(device)

learning_rate = 0.0002
model = MVSNet()
state_dict = torch.load('Trained_models/mvsnet_0_e80.pth')  # 继续训练
model.load_state_dict(state_dict)
model.to(device)
train_dataset = MVSNetDTUDataset(1, 2)
train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

#  优化器选用Adam
optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
loss_fn = loss.mvsnet_loss


def train_batch(_imgs, _homos, _depth, gt, _model, _opt, _loss_fn):
    _model.train()
    initial_depth, refined_depth = _model(_imgs, _homos, _depth)
    batch_loss = _loss_fn(gt, initial_depth, refined_depth)
    batch_loss.backward()
    _opt.step()
    _opt.zero_grad()
    return batch_loss.item()  # 此处返回的是损失值


for epoch in range(20):
    train_epoch_losses = []
    start = time.time()
    for ix, batch in enumerate(iter(train_dl)):
        imgs, homos, vis_img, gt_img, depth_value, scale = batch
        loss = train_batch(imgs.to(device), homos.to(device), depth_value.to(device), gt_img.to(device), model, optimizer, loss_fn)
        train_epoch_losses.append(loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    end = time.time()
    print(f'epoch: {epoch}, time used: {end - start}, loss: {train_epoch_loss}')

torch.save(model.to('cpu').state_dict(), 'Trained_models/mvsnet_0_e100.pth')