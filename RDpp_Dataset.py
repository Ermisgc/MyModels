import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from random import shuffle, seed
import noise
import numpy as np


# 本文件对开始对不同的图像处理进行比较和研究，查看是否有比较好的处理方法

class WuYanDataset(Dataset):
    def __init__(self, folder):
        self.files = glob(folder + '\\*.jpg') + glob(folder + '\\*.png')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.simplexNoise = noise.Simplex_CLASS()
        seed(10)
        shuffle(self.files)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        file = self.files[ix]
        img = cv2.imread(file)[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[100:835, 200:1780]
        B, G, R = cv2.split(img)  # get single 8-bits channel
        EB = cv2.equalizeHist(B)
        EG = cv2.equalizeHist(G)
        ER = cv2.equalizeHist(R)
        img = cv2.merge((EB, EG, ER))  # merge it back
        img = cv2.resize(img, (256, 256))
        ## Normal
        img_normal = img.copy()
        img_normal = torch.tensor(img_normal / 255).permute(2, 0, 1)
        img_normal = self.normalize(img_normal)
        ## simplex_noise
        size = 256
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
        img_noise = img + init_zero
        img_noise = torch.tensor(img_noise / 255).permute(2, 0, 1)
        img_noise = self.normalize(img_noise)
        return img_normal.to(self.device).float(), img_noise.to(self.device).float(), file.split('\\')[-1]

        # 裁剪
        # img = img[400:800, 350:700]
        # img = img[100:835, 200:1780]

        # 直方图均衡化
        # B, G, R = cv2.split(img)  # get single 8-bits channel
        # EB = cv2.equalizeHist(B)
        # EG = cv2.equalizeHist(G)
        # ER = cv2.equalizeHist(R)
        # img = cv2.merge((EB, EG, ER))  # merge it back
        #
        # img = cv2.resize(img, dsize=(256, 256))
        # img = torch.tensor(img/255).permute(2, 0, 1)
        # img = self.normalize(img)
        # return img.to(self.device).float()


def get_data(train_folder, batch_size):
    train_dataset = WuYanDataset(train_folder)
    # test_dataset = WuYanDataset(test_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader # test_dataloader
