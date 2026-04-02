import torch
import torch.nn as nn
from .conv import Conv, Concat  # 导入必要的模块，例如Conv模块


class LGFFM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1: 输入通道 (自动传入)
        # c2: 输出通道 (来自YAML args)
        c_mid = c2 // 2

        self.conv_p0_s2 = Conv(c1, c_mid, k=3, s=2)
        self.conv_p0_s2_2 = Conv(c_mid, c2, k=3, s=2)

        # 局部特征分支
        self.conv_left_1 = Conv(c2, c_mid, k=1, s=1)
        self.conv_left_2 = Conv(c_mid, c_mid, k=3, s=1)
        self.conv_left_3 = Conv(c_mid, c_mid, k=1, s=1)

        # 全局特征分支 (修正后的 MaxPool，保持尺寸不变)
        self.maxpool_right = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_right = Conv(c2, c_mid, k=1, s=1)

        # 融合
        self.conv_final = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        x_p0 = self.conv_p0_s2(x)
        x_p0 = self.conv_p0_s2_2(x_p0)

        p_left = self.conv_left_1(x_p0)
        p_left = self.conv_left_2(p_left)
        p_left = self.conv_left_3(p_left)

        p_right = self.maxpool_right(x_p0)
        p_right = self.conv_right(p_right)

        p_concat = torch.cat([p_left, p_right], dim=1)
        return self.conv_final(p_concat)