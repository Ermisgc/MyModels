import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import functional as F
import ResNetRD
import RD_Model
from torch.utils.data import Dataset, DataLoader
from glob import glob
from random import shuffle, seed
from torchvision import transforms
import cv2
from scipy.ndimage import gaussian_filter
from sklearn import metrics
import pandas as pd


# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
test_folder = "E:\\learning\\Datasets\\Box\\Test\\Camera1_Label_balanced"  # 烟厂数据集
# test_folder = "E:\\learning\\Datasets\\mvtec ad\\toothbrush\\test" # mvtec数据集


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':  # mul是值相乘，a是值相加
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


class TestDatasetWithLabel(Dataset):
    def __init__(self, folder):
        files_ori = glob(folder + "\\*\\*.jpg")  # 烟厂数据集
        # files_ori = glob(folder + "\\*\\*.png")  # 牙刷数据集

        self.files = files_ori
        seed(10)
        shuffle(self.files)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        img = cv2.imread(self.files[ix])[:, :, ::-1]
        # 裁剪
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
        # img = self.normalize(img)
        label = self.files[ix].split("\\")[-2].startswith('g')  # 烟厂数据集
        # label = self.files[ix].split("\\")[-2].startswith('g')  # mvtec牙刷数据集

        return img.to(device).float(), label

    def getlabel(self):
        return self.files


def test_aucroc(encoder, bn, decoder, test_dl, mode="aucroc"):
    scores = []
    labels = []
    encoder.eval()
    bn.eval()
    decoder.eval()
    for ix, data in enumerate(iter(test_dl)):
        img, label = data
        ft = encoder(img)
        fs = decoder(bn(ft))
        # input = [ft[-1]]
        # output = [fs[-1]]
        # anomaly_map, amap_list = cal_anomaly_map(input, output, img.shape[-1], amap_mode='a')
        anomaly_map, amap_list = cal_anomaly_map(ft, fs, img.shape[-1], amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        score = np.max(anomaly_map)
        label = label[0]
        label = True if label == torch.tensor(True) else False
        scores.append(score)
        labels.append(label)
        # if score > 0.75 and label == torch.tensor([True]):
        #     goodones += 1
        # if score <= 0.75 and label == torch.tensor([False]):
        #     goodones += 1

    scores_reserved = scores.copy()
    # print(scores_reserved)
    scores.sort()

    if mode == "precise":
        goodones = 0
        accuracies = []
        for ix in range(len(scores)):
            thres = scores[ix]
            goodones = 0
            for iy in range(len(scores)):
                score = scores_reserved[iy]
                if score > thres and labels[iy] == torch.tensor([True]):
                    goodones += 1
                if score <= thres and labels[iy] == torch.tensor([False]):
                    goodones += 1
                accuracies.append(goodones / len(scores))

        return np.max(accuracies)

    else:  # 计算aucroc
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores_reserved, pos_label=False)
        # plt.plot(fpr, tpr)
        # plt.show()
        return metrics.auc(fpr, tpr), scores_reserved, labels
        # auroc_px = round(metrics.roc_auc_score(labels, scores_reserved), 3)
        # return auroc_px


def visualization(encoder, bn, decoder, test_dl, mode="aucroc"):
    scores = []
    labels = []
    encoder.eval()
    bn.eval()
    decoder.eval()
    for ix, data in enumerate(iter(test_dl)):
        img, label = data
        ft = encoder(img)
        fs = decoder(bn(ft))
        # anomaly_map, amap_list = cal_anomaly_map([ft[-1]], [fs[-1]], img.shape[-1], amap_mode='a')
        anomaly_map, amap_list = cal_anomaly_map(ft, fs, img.shape[-1], amap_mode='a')
        ano_map = min_max_norm(anomaly_map)
        ano_map = cvt2heatmap(ano_map * 255)
        img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        # img = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
        img = np.uint8(min_max_norm(img) * 255)
        ano_map = show_cam_on_image(img, ano_map)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        print(np.max(anomaly_map))
        plt.imshow(ano_map)
        plt.axis('off')
        plt.show()


def save_csv(_scores, _labels, _files, _save_path):
    # df1 = pd.DataFrame(None, columns=["scores", "labels", "files"])
    # for ix in range(len(files)):
    #     df1.iloc[ix, 0] = _scores[ix]
    #     df1.iloc[ix, 1] = _labels[ix]
    #     df1.iloc[ix, 2] = _files[ix]
    data = {"scores": _scores, "labels": _labels, "files": _files}
    df1 = pd.DataFrame(data)
    df1.to_csv(_save_path, sep=",", index=False, header=True)


# 导入训练好的模型
encoder, bn = ResNetRD.wide_resnet50_2(pretrained=True)
decoder = RD_Model.de_wide_resnet50_2(pretrained=True)
state_dict = torch.load("MyModels/rd_model_0809_100_8.pth")
bn.load_state_dict(state_dict["bn"])
# torch.save(bn.state_dict(), "lookout.pth")
encoder.to(device)
decoder.to(device)
bn.to(device)
encoder.eval()
decoder.eval()
bn.eval()

# 加载数据
test_dataset = TestDatasetWithLabel(test_folder)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
aucroc, scores, labels = test_aucroc(encoder, bn, decoder, test_dl, "aucroc")
print(aucroc)
files = test_dataset.files
save_csv(scores, labels, files, "MyModels/model_8_analyse.csv")  # 存Excel进行分析
# visualization(encoder, bn, decoder, test_dl, "aucroc")  # 可视化
