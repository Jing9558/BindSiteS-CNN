"""
TOUGHC1 classification model
"""
from deepsphere.layers.samplings.healpix_pool_unpool import Healpix
from deepsphere.utils.laplacian_funcs import get_healpix_laplacians
from deepsphere.layers.chebyshev import SphericalChebConv
from torch import nn
import torch.nn.functional as F


class SphericalConvBNPool(nn.Module):
    def __init__(self, in_channels, out_channels, lap, kernel_size, pooling):
        super().__init__()
        self.sphconv = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.pooling = pooling

    def forward(self, x):
        x = self.sphconv(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = F.relu(x.permute(0, 2, 1))
        x = self.pooling(x)
        return x


class DeepSphereCls(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.pooling = Healpix(mode='max').pooling
        self.laps = get_healpix_laplacians(N, 4, 'combinatorial')
        self.conv_2 = SphericalConvBNPool(9, 32, self.laps[3], 4, self.pooling)
        self.conv_3 = SphericalConvBNPool(32, 64, self.laps[2], 4, self.pooling)
        self.conv_4 = SphericalConvBNPool(64, 128, self.laps[1], 4, self.pooling)
        self.conv_5 = SphericalConvBNPool(128, 256, self.laps[0], 4, self.pooling)
        self.fc1 = nn.Sequential(*[nn.Linear(256, 3), nn.LogSoftmax(dim=1)])

    def forward(self, x):
        x = self.conv_2(x)  # 32 -> 64
        x = self.conv_3(x)  # 64 -> 128
        x = self.conv_4(x)  # 128 -> 256
        x = self.conv_5(x)  # 256 -> 512
        x = self.fc1(x.mean(axis=1))
        return x