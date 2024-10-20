import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchsummary import summary


# 中间用的连续的ConvBn块
class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.model = nn.Sequential(
            # 这里一般 k = 3用来大小让channels大小不变做中间层
            # k = 1用来做直接的channels升降维
            # s = 1时，图像尺寸不变
            # s = 2是，图像尺寸减半
            # p 通常而言按需设置，一般取0或者1
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


# 特征提取层
class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBn(3, 8, 3, 1, 1),  # 图像尺寸不变，只升维，k的大小其实可试试1，3，7
            ConvBn(8, 8, 3, 1, 1),  # 中间层不变
            ConvBn(8, 16, 5, 2, 2),  # 图像尺寸减半，小升维，k 可以其他选择
            ConvBn(16, 16, 3, 1, 1),  # 中间层，k = 3比较好
            ConvBn(16, 16, 3, 1, 1),  # 图像尺寸减半，小升维，k 可以其他选择
            ConvBn(16, 32, 5, 2, 2),  # 做了一个同上的升维
            ConvBn(32, 32, 3, 1, 1),  # 中间层仍然是老特色
            nn.Conv2d(32, 32, 3, 1, 1)  # 意义不明的卷积
        )

    def forward(self, x):
        return self.model(x)


# 将src图像的特征投影到参考图像上
def homo_warping(src_fea, src_homo, ref_homo, depth_values):
    # src_fea, src图像经过特征提取后的特征，由上面应该是[B, C, H, W]：
    # 这里的B是batchsize的大小, C = 32, H = H_ori/4, W = W_ori/4
    # src_Homo，src图像相对于世界坐标系的齐次变换矩阵T，大小是[B, 4, 4]
    # ref_Homo，ref图像相对于世界坐标系的齐次变换矩阵T，大小是[B, 4, 4]
    # depth_values，即假设的深度范围，[B, Ndepth]，这个很好理解，因为MVSNet实际上假设了不同深度下的若干个平面
    # 返回值：[B, C, Ndepth, H, W]，即：有Ndepth个深度，然后每个深度都有一个对应的[B, C, H, W]的特征图

    batch, channels = src_fea.shape[0], src_fea.shape[1]  # src_fea的shape是[B, C, H, W]
    num_depth = depth_values.shape[1]  # depth_values的shape是[B, Ndepth]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():  # 这一步只做聚合，通过相机内外参计算，因此不用计算梯度
        # 从src图像到ref图像的齐次变换矩阵：
        homo = torch.matmul(src_homo, torch.inverse(ref_homo))
        rot = homo[:, :3, :3]  # [B, 3, 3]
        trans = homo[:, :3, 3:4]  # [B, 3, 1]

        # <---从这一步开始就是为了得到整个像素面上的所有二维像素的齐次坐标(x, y, 1)--->
        # torch.arange(a, b)生成[a,b)默认间隔维1的均布一维张量
        # torch.meshgrid由两个一维张量生成一个坐标为(x, y)两个二维张量，其中前者是行轴，后者是列轴（行列式读法）
        # x, y获取所有二维张量的(x)坐标与(y)坐标，x是行轴坐标，y的列轴坐标
        # 假如height为2592， width为2048，即行轴长度为2592， 列轴长度为2048，那么结果为：
        # x:每一行均为同样的数，每行的数从0递增为2591，共有2592行
        # y:每一行均为从0到2047的1*2048向量，共有2592行
        x, y = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])

        # 这一步其实可以省去，它是保证数据的存储顺序与访问顺序相同的一步，一般用于转置、permute后的Tensor
        # 因此转置与permute不会改变数据的存储结构，而是重新定义了数据的访问顺序
        # 如果没有转置与permute则可以省去
        y, x = y.contiguous(), x.contiguous()

        # 从(1, height * width)的角度处理和看数据
        x, y = x.view(height * width), y.view(height * width)  # [1, H * W]

        # 堆叠起来，此时每一个列向量可看作一个像素点的三维齐次坐标
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H * W]

        # 在第0维的位置增加一个维度
        xyz = torch.unsqueeze(xyz, 0)  # [1, 3, H * W]

        # 重复Tensor的内容，在batch值所在的维重复batch个Tensor
        # 得到的最终值就可以看成：有B个特征图片，每个特征图都有H * W个齐次坐标
        xyz = xyz.repeat(batch, 1, 1)  # [B, 3, H * W]

        # <---先旋转后平移得到不同深度条件下的齐次坐标--->
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

        # 除以深度可以得一点在成像平面对应的像素点
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea  # [B, C, Ndepth, H, W]


def cost_volume(img_nums, sum_of_square, sum_of_add):
    # img_nums为标量
    # sum_of_square的大小为[B, C, Ndepth, H, W]
    # sum_of_add的大小为[B, C, Ndepth, H, W]
    # 返回值仍为[B, C, Ndepth, H, W]
    return sum_of_square / img_nums - (sum_of_add / img_nums).pow(2)


class ConvBn3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.model = nn.Sequential(
            # 这里一般 k = 3用来大小让channels大小不变做中间层
            # k = 1用来做直接的channels升降维
            # s = 1时，图像尺寸不变
            # s = 2是，图像尺寸减半
            # p 通常而言按需设置，一般取0或者1
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBn3d(32, 8)

        self.conv1 = ConvBn3d(8, 16, stride=2)
        self.conv2 = ConvBn3d(16, 16)

        self.conv3 = ConvBn3d(16, 32, stride=2)
        self.conv4 = ConvBn3d(32, 32)

        self.conv5 = ConvBn3d(32, 64, stride=2)
        self.conv6 = ConvBn3d(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        # conv7 = self.conv7(x)
        x = conv4 + self.conv7(x)
        # x = conv4 + conv7
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


def depth_regression(cost_regularized, depth_values):
    # Cost_Regularized是Cost_Volume经过CostRegNet后得到的结果，大小为[B, 1, Ndepth, H, W]
    # depth_values是假定的不同深度值，前面已经提到了它的大小为[B, Ndepth]
    # 返回值大小为[B, H, W]，每个[H, W]的坐标对应一个深度值，即initial depth
    cost_reg = cost_regularized.squeeze(1)  # 挤掉已经没用的第1维，得到[B, Ndepth, H, W]
    prob_volume = F.softmax(cost_reg, dim=1)  # 对第1维进行softmax操作，即[H, W]特征图下每个像素对应的不同深度的概率，大小仍为[B, Ndepth, H, W]
    depth_values = depth_values.view(*depth_values.shape, 1, 1)  # 将depth_values升维为[B, Ndepth, 1, 1]

    # “*” 操作矩阵大小不同，Tensor会将depth_values扩展为[B, Ndepth, H, W]， 每个depth对应一个[H, W]大小数据全为depth的图
    # 这两个矩阵每个元素相乘，在[H, W]的每一个像素点看来，都是p * depth
    # [H, W]的每一个像素点求sum，即是实现了一个soft argmax
    depth = torch.sum(prob_volume * depth_values, 1)
    return depth


def resize_for_refine(ref_img, initial_depth):
    # ref_img：[B, 3, H * 4, W * 4]，就是原始参考图像
    # initial_depth：[B, H, W]，通过深度回归获得的初始深度估计值
    h, w = initial_depth.shape[1], initial_depth.shape[2]

    resize_transform = Resize([h, w])
    ref_img_resize = resize_transform(ref_img)  # 将原图resize为[B, 3, H, W]
    initial_depth_resized = initial_depth.unsqueeze(1)  # 将initial_depth变形为[B, 1, H, W]
    return ref_img_resize, initial_depth_resized


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBn(4, 32)
        self.conv2 = ConvBn(32, 32)
        self.conv3 = ConvBn(32, 32)
        self.res = ConvBn(32, 1)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureExtraction = FeatureExtraction()
        self.homo = homo_warping
        self.cost_volume = cost_volume
        self.cost_regular = CostRegNet()
        self.depth_regression = depth_regression
        self.resize_for_refine = resize_for_refine
        self.refine = RefineNet()

    def forward(self, imgs, homography_matrices, depth_values, is_refind=True):
        # imgs: [B, n, 3, H * 4, W * 4]
        # homography_matrices: [B, n, 4, 4]
        # depth_values: [B, Ndepth]

        # Step0.信息处理
        imgs = imgs.unbind(1)  # imgs变成[B, 3, H * 4, W * 4]的列表
        homography_matrices = homography_matrices.unbind(1)  # homography_matrices也变成了[B, 1, 4, 4]的列表
        assert len(imgs) == len(homography_matrices), "Different number of images and projection matrices"  # 检查数据
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # Step1. 特征提取：
        features = [self.featureExtraction(img) for img in imgs]  # [B, 32, H, W]的列表

        # Step2. 微分单应
        ref_feature = features[0]  # [B, C, H, W]
        src_features = features[1:]
        ref_homo = homography_matrices[0]  # [B, 4, 4]
        src_homos = homography_matrices[1:]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)  # 对齐为[B, C, Ndepth, H, W]
        sum_of_add = ref_volume
        sum_of_square = ref_volume ** 2  # 像素的平方
        del ref_volume  # 删去避免占用空间

        for src_feature, src_homo in zip(src_features, src_homos):
            ret_volume = self.homo(src_feature, src_homo, ref_homo, depth_values)  # [B, C, Ndepth, H, W]
            sum_of_add = sum_of_add + ret_volume
            sum_of_square = sum_of_square + ret_volume ** 2
            del ret_volume

        # Step3.Cost Volume Regularization
        cost_volume_not_reg = self.cost_volume(num_views, sum_of_square, sum_of_add)
        cost_volume_regularized = self.cost_regular(cost_volume_not_reg)

        # Step4.Initial Depth
        initial_depth = depth_regression(cost_volume_regularized, depth_values)
        if not is_refind:
            return initial_depth

        # Step5.Refine
        ref_img_resize, initial_depth_resized = self.resize_for_refine(imgs[0], initial_depth)
        refined_depth = self.refine(ref_img_resize, initial_depth_resized)
        return initial_depth, refined_depth.squeeze(1)
