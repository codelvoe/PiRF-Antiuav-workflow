import torch
import torch.nn as nn
from .conv import Conv, Concat

# ==============================================================================
# 1. EdgeFlow (formerly LGFFM) 
# Captures edge flow of moving drones (P2 Backbone)
# ==============================================================================
class EdgeFlow(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_mid = c2 // 2
        self.conv_p0_s2 = Conv(c1, c_mid, k=3, s=2)
        self.conv_p0_s2_2 = Conv(c_mid, c2, k=3, s=2)
        # Local Branch
        self.conv_left_1 = Conv(c2, c_mid, k=1, s=1)
        self.conv_left_2 = Conv(c_mid, c_mid, k=3, s=1)
        self.conv_left_3 = Conv(c_mid, c_mid, k=1, s=1)
        # Global Branch
        self.maxpool_right = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_right = Conv(c2, c_mid, k=1, s=1)
        # Fusion
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


# ==============================================================================
# 2. SignalFocus (formerly EMA)
# Focuses on target signal amidst noise
# ==============================================================================
class SignalFocus(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
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
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h = hw[:, :, :h, :]
        x_w = hw[:, :, h:, :].permute(0, 1, 3, 2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# ==============================================================================
# 3. SkyVisionBlock (formerly LERFBlock)
# Optimized for large-scale sky context
# ==============================================================================
class ChannelAttention_SVB(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.pool(x)))

class SpatialAttention_SVB(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # P_avg + P_max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class SkyAttention(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.ca = ChannelAttention_SVB(c1)
        self.sa = SpatialAttention_SVB()
    def forward(self, x):
        return self.sa(self.ca(x))

class SkyVisionBlock(nn.Module):
    def __init__(self, c1, c2, k=31):
        super().__init__()
        c_mid = c2 // 2
        
        # Path A (Large Kernel)
        self.path_lk = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, k, padding=k//2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU()
        )
        
        # Path B (Dilated)
        d = max(1, k // 2)
        p = d 
        self.path_dc = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, 3, padding=p, dilation=d, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU()
        )
        
        self.attn = SkyAttention(c2)
        self.transition = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2))

    def forward(self, x):
        x1 = self.path_lk(x)
        x2 = self.path_dc(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.attn(out)
        return self.transition(out)


# ==============================================================================
# 4. ElasticHead (formerly DCNv2)
# Adapts to elastic/rigid body changes
# ==============================================================================
from torchvision.ops import deform_conv2d
class ElasticHead(nn.Module):
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
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = deform_conv2d(x, offset, self.weight, self.bias, 
                          stride=self.stride, padding=self.padding, 
                          dilation=self.dilation, mask=mask)
        return self.act(self.bn(x))


# ==============================================================================
# 5. CSDI (Cross-Scale Detail Injector) - INNOVATION
# Direct high-pass highway from P2 EdgeFlow to Deep Layers
# ==============================================================================
class CSDI(nn.Module):
    def __init__(self, c1, c2, scale=4):
        super().__init__()
        # Use simple strided Conv for efficient downsampling
        self.down = Conv(c1, c2, k=3, s=scale)
    
    def forward(self, x):
        return self.down(x)
