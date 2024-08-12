# AutoEncoder_Train.py
# 实现并训练一个基于卷积自编码器的神经网络
import time
import cv2
import numpy as np
import torch
import numpy
import torch.nn as nn
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from glob import glob
from torch.optim import SGD, lr_scheduler


# 全局变量设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_1_folder = 'E:\\learning\\Datasets\\Box\\Train\\Camera1_Normal_Generate\\'
test_1_folder = 'E:\\learning\\Datasets\\Box\\Test\\Camera1\\'
batch_size = 101


# 定义数据集对象WuYanDataset
class WuYanDataset(Dataset):
    def __init__(self, folder, aug=None):
        self.files = glob(folder + '*.jpg')
        ori_file = self.files[0]
        img = cv2.imread(ori_file)
        self.wid = img.shape[1]
        self.hei = img.shape[0]
        self.aug = aug  # 图像增强函数序贯

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        file = self.files[ix]
        img = cv2.imread(file)  # imread读入的是numpy，格式为Numpy(H,W,C)
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.aug is not None:
        #     aug.augment_image(img)
        # transf = transforms.ToTensor()
        # img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
        return img, file

    def collate_fn(self, batch):  # 图像批处理，加快速度
        ims, file_batch = list(zip(*batch))
        if self.aug:
            ims = self.aug.augment_images(images=ims)
        # transf = transforms.ToTensor()
        # img_tensor = transf(ims)  # tensor数据格式是torch(C,H,W)
        ims = np.array(ims).transpose(0, 3, 1, 2)
        ims = torch.tensor(ims).to(device) / 255
        # [:, None, :, :]
        return ims, file_batch


# 定义卷积自编码器
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),
            nn.Conv2d(16, 32, 3, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
        )  # 瓶颈层大小暂定为5 × 5

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=3, padding=1),
            nn.Tanh(),
            # nn.Sigmoid()
        )

    def forward(self, _x):
        _x = self.encoder(_x)
        _x = self.decoder(_x)
        return _x


# 定义训练函数
def train_batch(_input, _model, _opt, _loss_fn):
    _model.train()
    prediction = _model(_input)
    _batch_loss = _loss_fn(prediction, _input)
    _batch_loss.backward()
    _opt.step()
    _opt.zero_grad()
    return _batch_loss.item()


# 在测试集上试验精度
@torch.no_grad()
def test_loss(_x, _model, _loss_fn):
    _model.eval()
    prediction = _model(_x)
    return _loss_fn(prediction, _x).item()


# 创建
aug = iaa.Sequential([
    iaa.Resize({'height': 1024, 'width': 1024}),
    # iaa.Affine(translate_px={'x': (-100, 200), 'y': (-100, 200)}, cval=(240, 255)),
    # iaa.Affine(rotate=(-10, 10), fit_output=True, cval=(240, 255))
], random_order=True)  # 数据量够多其实不太需要数据增强

# 设置数据集
train_1_dataset = WuYanDataset(train_1_folder, aug)
train_1_dl = DataLoader(train_1_dataset, batch_size=batch_size, collate_fn=train_1_dataset.collate_fn, shuffle=True)
test_1_dataset = WuYanDataset(test_1_folder, aug)
test_1_dl = DataLoader(test_1_dataset, batch_size=batch_size, collate_fn=test_1_dataset.collate_fn, shuffle=False)

# 建立模型，设置损失函数与优化器
model = ConvAutoEncoder().to(device)
loss_fn = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    train_epoch_losses = []
    test_epoch_losses = []
    lr = 1e-2 * (0.5 ** (epoch // 20))  # 动态调整学习率
    start_time = time.time()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for ix, batch in enumerate(iter(train_1_dl)):
        x, y = batch
        batch_loss = train_batch(x, model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
    # for ix, batch in enumerate(iter(test_1_dl)):
    #     x, y = batch
    #     batch_loss = test_loss(x, model, loss_fn)
    #     test_epoch_losses.append(batch_loss)
        # scheduler.step(batch_loss)
    test_epoch_loss = np.mean(test_epoch_losses)
    train_epoch_loss = np.mean(train_epoch_losses)
    end_time = time.time()
    used_time = end_time - start_time
    print(f"epoch:{epoch + 1}/100, train_epoch_loss:{train_epoch_loss}, test_epoch_loss:{test_epoch_loss}, lr:{optimizer.state_dict()['param_groups'][0]['lr']}, consume:{used_time}")


torch.save(model.to('cpu').state_dict(), 'Camera1_ConvAutoEncoder_GenerateNormal4.pth')
