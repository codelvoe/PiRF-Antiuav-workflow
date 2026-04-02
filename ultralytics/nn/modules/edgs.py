import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from torchvision.ops import deform_conv2d

# --- 1. EMA (Efficient Multi-Scale Attention) ---
# 论文 Section 3.4 [cite: 1017-1071]
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h = hw[:, :, :h, :]
        x_w = hw[:, :, h:, :].permute(0, 1, 3, 2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# --- 2. Ghost Modules ---
# 论文 Section 3.3 [cite: 986-1016]
# GhostConv 和 GhostBottleneck 实际上在 Ultralytics 源码中已存在
# 但为了确保结构一致，我们显式定义论文中描述的 C3Ghost

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1): 
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False)  # pw-linear
        )
        self.shortcut = nn.Sequential(
            Conv(c1, c1, 3, s=2, p=1, g=c1, act=False), Conv(c1, c2, 1, 1, act=False)
        ) if s == 2 else nn.Identity()

    def forward(self, x):
        return x + self.conv(x)

class C3Ghost(nn.Module):
    # CSP Ghost module
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# --- 3. DCNv2 (DDetect Components) ---
# 论文 Section 3.5 [cite: 1135-1148]
# DDetect 替换了 Detect 头中的卷积。为了方便在 YAML 中使用，
# 我们定义一个 DCNv2 模块，可以在 Head 之前使用，模拟 DDetect 的特征提取能力。

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv_offset = nn.Conv2d(c1, 3 * k * k, kernel_size=k, stride=s, padding=k//2)
        self.weight = nn.Parameter(torch.Tensor(c2, c1 // g, k, k))
        self.bias = nn.Parameter(torch.Tensor(c2))
        self.stride = s
        self.padding = k // 2
        self.dilation = 1
        self.groups = g
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        # DEBUG: Replaced deform_conv2d with standard conv to isolate crash
        # out = self.conv_offset(x)
        # o1, o2, mask = torch.chunk(out, 3, dim=1)
        # offset = torch.cat((o1, o2), dim=1)
        # mask = torch.sigmoid(mask)
        # x = deform_conv2d(x, offset, self.weight, self.bias, 
        #                   stride=self.stride, padding=self.padding, 
        #                   dilation=self.dilation, mask=mask)
        # Mimic standard convolution using the weights (ignoring offset/mask)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.act(self.bn(x))
