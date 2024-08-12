# CameraClassification.py
# 用于将数据集中不同机位的文件分开


import cv2
import shutil
from glob import glob


PathDir = ['E:\\learning\\Project\\WuYan\\BoxImg\\1643\\', 'E:\\learning\\Project\\WuYan\\BoxImg\\240520Early\\',
           'E:\\learning\\Project\\WuYan\\BoxImg\\240520Noon\\', 'E:\\learning\\Project\\WuYan\\BoxImg\\240521Early\\'
           ]
SaveDir = ['E:\\learning\\Datasets\\Box\\test\\Camera1\\', 'E:\\learning\\Datasets\\Box\\test\\Camera2\\',
           'E:\\learning\\Datasets\\Box\\test\\Camera3\\', 'E:\\learning\\Datasets\\Box\\test\\Camera4\\']


def cameraClassify(folder):
        files = glob(folder + '\\*\\*.jpg')
        for img_path in files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            save_path = img_path.split('\\')[-1]
            if img.shape[0] == 1100:
                save_path = SaveDir[0] + save_path
            elif img.shape[0] == 950:
                save_path = SaveDir[1] + save_path
            elif img.shape[0] == 900:
                save_path = SaveDir[2] + save_path
            elif img.shape[0] == 550:
                save_path = SaveDir[3] + save_path
            else:
                continue
            shutil.copy(img_path, save_path)
            print(img_path + ' is Done')


# cameraClassify(PathDir)
folder = 'E:\\learning\\Project\\WuYan\\BoxImg'
cameraClassify(folder)
