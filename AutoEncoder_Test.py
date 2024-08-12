import cv2
import numpy as np
import torch
import numpy
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


model = ConvAutoEncoder()
state_dict = torch.load('Camera1_ConvAutoEncoder_GenerateNormal2.pth')
model.load_state_dict(state_dict)
model.to(device)

aug = iaa.Sequential([
    iaa.Resize({'height': 1024, 'width': 1024}),
    # iaa.Affine(translate_px={'x': (-200, 200), 'y': (-200, 200)}, cval=(240, 255)),
    # iaa.Affine(rotate=(-10, 10), cval=(240, 255))
], random_order=True)  # 数据量够多其实不太需要数据增强

test_1_folder = 'E:\\learning\\Datasets\\Box\\Train\\Camera1_Normal_Generate\\'
test_1_dataset = WuYanDataset(test_1_folder, aug)
test_1_dl = DataLoader(test_1_dataset, batch_size=101, collate_fn=test_1_dataset.collate_fn, shuffle=False)
loss_fn = nn.BCELoss()


@torch.no_grad()
def eval_loss(_model, _file_name, _loss_fn):
    _model.eval()
    img = cv2.imread(_file_name)
    img = cv2.resize(img, dsize=(1024, 1024))
    img = np.array(img).transpose(2, 0, 1)
    img = torch.tensor(img) / 255
    prediction = _model(img)
    single_loss = _loss_fn(prediction, img).item()
    print(f'{_file_name}, loss:{single_loss}')
    return single_loss


@ torch.no_grad()
def batch_eval_loss(_x, _model, _loss_fn):
    model.eval()
    prediction = _model(_x)
    _loss = _loss_fn(prediction, _x)
    # _loss.backward()
    return _loss.item()


# for file in files:
#     losses.append(eval_loss(model, file, loss_fn))
losses = []
for ix, batch in enumerate(iter(test_1_dl)):
    x, y = batch
    losses.append(batch_eval_loss(x, model, loss_fn))
print(losses)
print(np.mean(losses))

# df1 = pd.DataFrame(np.array(losses), columns=['loss'])
# df1.insert(df1.shape[1], 'file', 0)
# for index in range(len(files)):
#     df1.iloc[index, 1] = files[index]
# outputpath = 'loss_test_normalgenerate.csv'
# df1.to_csv(outputpath, sep=',', index=False, header=True)


# latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()
# tsne = TSNE(2)
# clustered = tsne.fit_transform(latent_vectors)
#
# fig = plt.figure(figsize=(12, 10))
# plt.scatter(*zip(*clustered))
# plt.title('Model5')
# plt.show()
#
#
# data = pd.DataFrame(np.array(clustered), columns=['dim1', 'dim2'])
# data.insert(data.shape[1], 'files', 0)
# for index in range(len(files)):
#     data.iloc[index, 2] = files[index]
#
# outputpath = 'result_test6.csv'
# data.to_csv(outputpath, sep=',', index=False, header=True)
