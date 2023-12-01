""" 原论文是网络结果以后进行resize操作，修改网络使得网络的输出正好等于输入尺寸 """

from turtle import forward
from numpy import diff
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ 在UNET中不管是下采样还是上采样，每一层都会进行两次连续的卷积操作 """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # sequential是一个时序容器，卷积-BN-ReLU-卷积-BN-ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """ UNet一共有四次下采样，每一次是一个maxpool池化层，然后接一个DoubleConv模块 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """ 右半部分的上采样，还有特征融合操作"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        """ 定义了两种上采样方法，一个是双线性插值，一个是反卷积 """
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """前向传播，x1是上采样的数据，x2接受的是特征融合的数据"""
        """特征融合的方法:先对小的特征图进行padding，然后concat"""
        # 先进行上采样
        x1 = self.up(x1)
        # 输入是CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
    
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

       #特征融合以后送入到两次卷积
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出模块，两次卷积，一次1*1卷积"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1*1的卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)