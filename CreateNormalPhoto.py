# CreateNormalPhoto.py
# 针对正常样本偏少的问题，采用数据增强的方法制作更多正样本，并保存于Normal文件夹
import cv2
from imgaug import augmenters as iaa
from glob import glob


folder = 'E:\\learning\\Datasets\\Box\\Train\\Camera1-0809\\'
save_folder = 'E:\\learning\\Datasets\\Box\\Train\\Camera1-0809-Generate\\'
aug = iaa.Sequential([
    iaa.Affine(translate_px={'x': (-20, 20), 'y': (-2, 2)}, mode='edge'),
    iaa.Affine(rotate=(-2, 2), fit_output=True, mode='edge'),
], random_order=True)
files = glob(folder + '*.jpg')
for file in files:
    img = cv2.imread(file)
    for index in range(8):
        img_after = aug.augment_image(img)
        save_file = save_folder + file.split('\\')[-1].split('.')[0] + str(index) + 'aug.jpg'
        cv2.imwrite(save_file, img_after)

