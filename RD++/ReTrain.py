import torch.nn as nn
import torch
import RD_Model
import ResNetRD
import RDpp_Dataset
import utils_train
import RDpp_Test
from glob import glob
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
learning_rate = 0.005
proj_lr = 0.001
weight_proj = 0.2
update_epoch = 2  # 多少epoch更新一次参数
max_epochs = 100
train_folder = "E:\\learning\\Datasets\\Box\\Train\\Camera1-0812"  # 烟厂数据集
# train_folder = "E:\\learning\\Datasets\\mvtec ad\\toothbrush\\train\\good"  # 牙刷数据集

test_folder = "E:\\learning\\Datasets\\Box\\Test\\Camera1_Label_balanced"  # 烟厂数据集
# test_folder = "E:\\learning\\Datasets\\mvtec ad\\toothbrush\\test"  # 牙刷数据集
test_dataset = RDpp_Test.TestDatasetWithLabel(test_folder)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
save_path = "Trained_Models/rd_model_0812_200_16.pth"


encoder, bn = ResNetRD.wide_resnet50_2(pretrained=True)
decoder = RD_Model.de_wide_resnet50_2(pretrained=True)
proj_layer = utils_train.MultiProjectionLayer(base=64).to(device)
state_dict = torch.load("Trained_Models/rd_model_0812_100_16.pth")
bn.load_state_dict(state_dict["bn"])
proj_layer.load_state_dict(state_dict['proj'])
encoder.to(device)
decoder.to(device)
bn.to(device)
proj_layer.to(device)
encoder.eval()
# torch.save(bn.state_dict(), "lookout.pth")

train_dl = RDpp_Dataset.get_data(train_folder, batch_size)
# encoder, bn, decoder = RD_Model.get_model()
proj_layer = utils_train.MultiProjectionLayer(base=64).to(device)
proj_loss = utils_train.Revisit_RDLoss()
optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=proj_lr, betas=(0.5, 0.999))
optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                     betas=(0.5, 0.999))
# optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))

for epoch in range(max_epochs):
    epoch_losses = []
    start = time.time()
    # 动态调整学习率
    learning_rate = (0.8 ** (epoch//20)) * 0.005
    proj_lr = (0.8 ** (epoch // 20)) * 0.001
    for param_group in optimizer_distill.param_groups:
        param_group['lr'] = learning_rate
    for param_group in optimizer_proj.param_groups:
        param_group['lr'] = proj_lr

    bn.train()
    decoder.train()
    proj_layer.train()

    for ix, data in enumerate(iter(train_dl)):  # 训练过程
        img, img_noise, _ = data
        inputs = encoder(img)
        inputs_noise = encoder(img_noise)
        (feature_space_noise, feature_space) = proj_layer(inputs, features_noise=inputs_noise)

        mid = bn(feature_space)
        outputs = decoder(mid)

        # 计算各种误差
        L_distill = utils_train.loss_fucntion(inputs, outputs)
        L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
        loss = L_distill + weight_proj * L_proj
        loss.backward()

        # if (epoch + 1) % update_epoch == 0:
        optimizer_proj.step()
        optimizer_distill.step()
        # Clear gradients
        optimizer_proj.zero_grad()
        optimizer_distill.zero_grad()

        epoch_losses.append(loss.item())

    end = time.time()
    epoch_loss = np.array(epoch_losses).mean()
    print(f'epoch[{epoch+101}/{max_epochs}]，loss:{epoch_loss}, speed:{end - start}s')
    # if (epoch + 1) % 2 == 0:
    aucroc, _, _ = RDpp_Test.test_aucroc(encoder, bn, decoder, proj_layer, test_dl, "aucroc")
    print(f"aucroc = {aucroc}")

torch.save({'bn': bn.to("cpu").state_dict(),
            'decoder': decoder.to("cpu").state_dict(),
            'proj': proj_layer.to("cpu").state_dict()}, save_path)