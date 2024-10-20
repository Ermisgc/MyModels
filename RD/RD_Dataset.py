import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from random import shuffle, seed


# 本文件对开始对不同的图像处理进行比较和研究，查看是否有比较好的处理方法

class WuYanDataset(Dataset):
    def __init__(self, folder):
        self.files = glob(folder + '\\*.jpg') + glob(folder + '\\*.png')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        seed(10)
        shuffle(self.files)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        file = self.files[ix]
        img = cv2.imread(file)[:, :, ::-1]
        # img = img[400:800, 350:700]
        img = img[100:835, 200:1780]

        # 直方图均衡化
        B, G, R = cv2.split(img)  # get single 8-bits channel
        EB = cv2.equalizeHist(B)
        EG = cv2.equalizeHist(G)
        ER = cv2.equalizeHist(R)
        img = cv2.merge((EB, EG, ER))  # merge it back

        img = cv2.resize(img, dsize=(256, 256))
        img = torch.tensor(img/255).permute(2, 0, 1)
        img = self.normalize(img)
        return img.to(self.device).float()


def get_data(train_folder, batch_size):
    train_dataset = WuYanDataset(train_folder)
    # test_dataset = WuYanDataset(test_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader # test_dataloader
