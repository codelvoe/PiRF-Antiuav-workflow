"""
HASA (Hybrid Adaptive Spatial Attention) Module - Enhanced Version
增强版: 更强大的注意力机制,优先提升精准率和召回率
"""

import torch
import torch.nn as nn
from .conv import Conv
from .block import Bottleneck


class HASAAttention(nn.Module):
    """
    增强版混合注意力模块 - 高容量版本
    - 4个dilated conv分支 (匹配MACF的多尺度能力)
    - 大kernel空间注意力 (匹配CAM的感受野)
    - 额外的特征增强层提升计算量
    - 更强的特征表达能力
    """
    def __init__(self, c, reduction=4):  # reduction从8降到4,进一步增加容量
        super().__init__()
        
        # Enhanced Channel Attention (双路径,更深的网络)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 更深更宽的MLP用于通道注意力
        mid_c = max(c // reduction, 32)  # 确保中间层有足够容量
        self.ca = nn.Sequential(
            nn.Conv2d(c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, 1, bias=False),  # 额外层
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c // 2, 1, bias=False),  # 再加一层
            nn.BatchNorm2d(mid_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c // 2, c, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Multi-scale Spatial Attention (4个分支,匹配MACF)
        # 每个分支增加计算量
        self.sa_branch1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, dilation=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)  # 额外的1x1卷积
        )
        self.sa_branch2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)
        )
        self.sa_branch3 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=3, dilation=3, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)
        )
        self.sa_branch4 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=4, dilation=4, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)
        )
        
        # Large Kernel Spatial Attention (匹配CAM的大感受野)
        # 增加更多大kernel分支
        self.sa_large = nn.Sequential(
            nn.Conv2d(c, c, (1, 7), padding=(0, 3), groups=c, bias=False),
            nn.Conv2d(c, c, (7, 1), padding=(3, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)
        )
        
        # 额外的11x11 kernel分支 (模仿CAM)
        self.sa_large2 = nn.Sequential(
            nn.Conv2d(c, c, (1, 11), padding=(0, 5), groups=c, bias=False),
            nn.Conv2d(c, c, (11, 1), padding=(5, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False)
        )
        
        # 融合多尺度特征 (更深的融合网络)
        fusion_c = c * 6  # 6个分支
        self.sa_conv = nn.Sequential(
            nn.Conv2d(fusion_c, fusion_c // 2, 1, bias=False),
            nn.BatchNorm2d(fusion_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_c // 2, c, 1, bias=False),  # 中间层
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # Adaptive Fusion Weight (学习通道和空间注意力的融合权重)
        self.fusion_weight = nn.Parameter(torch.ones(2))
    
    def forward(self, x):
        # Enhanced Channel Attention
        ca_avg = self.ca(self.avg_pool(x))
        ca_max = self.ca(self.max_pool(x))
        ca = ca_avg + ca_max
        
        # Multi-scale + Large Kernel Spatial Attention
        sa1 = self.sa_branch1(x)
        sa2 = self.sa_branch2(x)
        sa3 = self.sa_branch3(x)
        sa4 = self.sa_branch4(x)
        sa_large = self.sa_large(x)
        sa_large2 = self.sa_large2(x)
        
        # 融合所有空间特征
        sa_cat = torch.cat([sa1, sa2, sa3, sa4, sa_large, sa_large2], dim=1)
        sa = self.sa_conv(sa_cat)
        
        # Adaptive fusion of channel and spatial attention
        w = torch.softmax(self.fusion_weight, dim=0)
        out = x * (w[0] * ca + w[1] * sa)
        
        return out


class HASABottleneck(nn.Module):
    """
    增强的Bottleneck,集成HASA注意力机制
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
        # 集成增强版HASA注意力 (高容量)
        self.attn = HASAAttention(c2, reduction=4)
    
    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.attn(out)  # 应用注意力
        return x + out if self.add else out


class C2f_HASA(nn.Module):
    """
    HASA增强的C2f模块
    集成更强大的多尺度注意力机制,优先提升精准率和召回率
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): HASABottleneck数量
            shortcut (bool): 是否使用shortcut连接
            g (int): 分组卷积数
            e (float): 扩展比例
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 使用增强版HASABottleneck
        self.m = nn.ModuleList(
            HASABottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
    
    def forward(self, x):
        """Forward pass through C2f_HASA layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

