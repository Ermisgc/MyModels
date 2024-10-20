import time

import numpy as np
import torch
import RD_Dataset
import RD_Model
from torch.utils.data import DataLoader
import RD_Test


# 系列主要参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
learning_rate = 0.005
max_epochs = 100
train_folder = "E:\\learning\\Datasets\\Box\\Train\\Camera1-0809-Generate"  # 烟厂数据集
# train_folder = "E:\\learning\\Datasets\\mvtec ad\\toothbrush\\train\\good"  # 牙刷数据集

test_folder = "E:\\learning\\Datasets\\Box\\Test\\Camera1_Label_balanced"  # 烟厂数据集
# test_folder = "E:\\learning\\Datasets\\mvtec ad\\toothbrush\\test"  # 牙刷数据集
test_dataset = RD_Test.TestDatasetWithLabel(test_folder)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_path = "MyModels/rd_model_0809_100_8.pth"


# 官方定义的损失函数
def loss_fn(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1)))
    return loss


train_dl = RD_Dataset.get_data(train_folder, batch_size)
encoder, bn, decoder = RD_Model.get_model()
optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))

for epoch in range(max_epochs):
    epoch_losses = []
    start = time.time()
    learning_rate = (0.8 ** (epoch//20)) * 0.005
    # 动态调整学习率
    bn.train()
    decoder.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    for ix, data in enumerate(iter(train_dl)):
        x = data
        inputs = encoder(x)
        mid = bn(inputs)
        outputs = decoder(mid)
        loss = loss_fn(inputs, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
    end = time.time()
    epoch_loss = np.array(epoch_losses).mean()
    print(f'epoch[{epoch+1}/{max_epochs}]，loss:{epoch_loss}, speed:{end - start}s')
    if (epoch + 1) % 10 == 0:
        aucroc, _, _ = RD_Test.test_aucroc(encoder, bn, decoder, test_dl, "aucroc")
        print(f"aucroc = {aucroc}")

torch.save({'bn': bn.to("cpu").state_dict(),
            'decoder': decoder.to("cpu").state_dict()}, save_path)
