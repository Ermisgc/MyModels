import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvsnet import MVSNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from glob import glob
from PIL import Image
import re


# 从原本的data_io.py文件中得来，然后进行了部分改编
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':  # 对应灰度图
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class MVSNetDTUDataset(Dataset):
    def __init__(self, scan_idx=1, light_idx=0, ):
        scan_idx = scan_idx if 128 >= scan_idx > 0 else 1
        self.imgs_files = []  # 图片所在路径
        self.vis_files = []  # 可视区域所在路径
        self.depth_files = []  # 深度图所在路径
        self.extrinsic_homos = []  # 相机外参矩阵
        self.intrinsic_homos = []  # 相机内参矩阵
        self.img_pairs = []  # 图像配对关系所在路径

        pair_lines = []
        # 解析视角的配对关系
        try:
            pair_file = open("E:\\learning\\Datasets\\mvs_training\\dtu\\Cameras\\pair.txt", 'r')
            content = pair_file.read()
        except IOError:
            print("打开pair.txt文件失败")
        else:
            pair_lines = content.split('\n')

        for idx in range(49):
            self.imgs_files.append("E:\\learning\\Datasets\\mvs_training\\dtu\\Rectified\\scan{}_train\\rect_{:03}_{}_r5000.png".format(scan_idx, 1 + idx, light_idx))
            self.vis_files.append("E:\\learning\\Datasets\\mvs_training\\dtu\\Depths\\scan{}_train\\depth_visual_{:04}.png".format(scan_idx, idx))
            self.depth_files.append("E:\\learning\\Datasets\\mvs_training\\dtu\\Depths\\scan{}_train\\depth_map_{:04}.pfm".format(scan_idx, idx))

            # 解析配对关系
            pairs_idx_line = pair_lines[(idx + 1) * 2].split(' ')
            pairs_of_idx = []
            for jdx in range(10):
                pairs_of_idx.append(int(pairs_idx_line[2 * jdx + 1]))
            self.img_pairs.append(pairs_of_idx)

            # 解析相机内外参数矩阵
            try:
                homo_file = open("E:\\learning\\Datasets\\mvs_training\\dtu\\Cameras\\train\\{:08}_cam.txt".format(idx), 'r')
                content = homo_file.read()
            except IOError:
                print("打开相机参数文件失败")
            else:
                lines = content.split('\n')
                extr = torch.from_numpy(np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4)))
                intr = torch.from_numpy(np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3)))
                self.extrinsic_homos.append(extr)
                self.intrinsic_homos.append(intr)

    def __getitem__(self, ix):
        to_tensor_tran = transforms.ToTensor()
        ref_img = to_tensor_tran(Image.open(self.imgs_files[ix]))  # [C, H, W]

        # 参见https://www.yuque.com/ermis/qblp21/anso8s3oxtgsxy4l 单应部分的公式，求一个虚拟的齐次变换矩阵
        ref_homo = self.extrinsic_homos[ix]
        ref_in = self.intrinsic_homos[ix]
        ref_homo[:3, :3] = ref_in.matmul(ref_homo[:3, :3])
        ref_homo[:3, 3] = ref_in.matmul(ref_homo[:3, 3])

        src_imgs = []
        src_homos = []
        for idx in range(10):
            temp_idx = self.img_pairs[ix][idx]
            src_imgs.append(to_tensor_tran(Image.open(self.imgs_files[temp_idx])))
            src_homo = self.extrinsic_homos[temp_idx]
            src_in = self.intrinsic_homos[temp_idx]
            src_homo[:3, :3] = ref_in.matmul(src_homo[:3, :3])
            src_homo[:3, 3] = ref_in.matmul(src_homo[:3, 3])
            src_homos.append(src_homo)

        src_imgs.insert(0, ref_img)
        src_homos.insert(0, ref_homo)
        imgs = torch.stack(src_imgs)
        homos = torch.stack(src_homos)

        vis_img = to_tensor_tran(Image.open(self.vis_files[ix]))
        # gt_img = to_tensor_tran(Image.open(self.depth_files[ix]))
        gt_path = self.depth_files[ix]
        scale: float
        gt_img, scale = read_pfm(gt_path)
        gt_img = torch.from_numpy(gt_img.copy())

        depth_value = (torch.arange(0, 192) * 2.5 + 425)
        return imgs, homos, vis_img, gt_img, depth_value, scale

    def __len__(self):
        return len(self.imgs_files)


