import torch
import torch.nn as nn

# --- 1. LAM: 线性空间-通道注意力模块 ---
# 论文引用: [cite: 383-442]

class ChannelAttention(nn.Module):
    """对应论文 Eq(2) 和 Eq(3)"""
    def __init__(self, c1):
        super().__init__()
        # 论文提到使用 Max Pooling 和 卷积分支
        self.pool = nn.AdaptiveMaxPool2d(1)
        # 模拟论文中的通道交互：Conv3x3 -> Conv1x1
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False), # 简化为1x1以适配YOLO速度
            nn.ReLU(),
            nn.Conv2d(c1, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # F_med = P_max(F_in) ... (Eq 2)
        y = self.pool(x)
        y = self.fc(y)
        # ATT_ch = sigmoid(...) (Eq 3)
        return x * self.sigmoid(y)

class SpatialAttention(nn.Module):
    """对应论文 Eq(4)"""
    def __init__(self, kernel_size=5):
        super().__init__()
        # 论文明确指出使用 5x5 卷积 [cite: 405]
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # P_avg 和 P_max 拼接 [cite: 403]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # ATT_spt = sigmoid(Conv5x5(...)) (Eq 4)
        return x * self.sigmoid(self.conv(x_cat))

class LAM(nn.Module):
    """Linear Spatial-Channel Attention Module"""
    def __init__(self, c1):
        super().__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 顺序：先通道，后空间 [cite: 439-440]
        x = self.ca(x)
        x = self.sa(x)
        return x


# --- 2. LERFBlock: 大有效感受野模块 ---
# LERFBlock uses a dual-path design to balance speed and accuracy. 
# Path A: Large Kernel (RepLK) for shape preference.
# Path B: Dilated Convolution for large receptive field with low computation.

class LERFBlock(nn.Module):
    def __init__(self, c1, c2, k=31): # k 为大核尺寸
        super().__init__()
        # 论文中提到将输入通道分为两条路径 [cite: 343]
        # 这里我们简化为：两个分支都接受输入，最后拼接并降维
        c_mid = c2 // 2 

        # --- 路径 A: RepLK 分支 (Large Kernel) ---
        # 论文: 3x3 Conv -> RepLK Block (DW Conv) -> 3x3 Conv [cite: 346, 348, 365]
        self.path_lk = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False), # 降维适配
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            # 大核深度可分离卷积 (Depth-wise)
            nn.Conv2d(c_mid, c_mid, k, padding=k//2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            # 融合
            nn.Conv2d(c_mid, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU()
        )

        # --- 路径 B: 空洞卷积分支 (Dilated Conv) ---
        # 论文: DW Dilated Convolution [cite: 366]
        # Dilation factor (DF) 通常设为 k // 2
        d = max(1, k // 2)
        # Padding 计算: (d * (kernel_size - 1)) / 2. 这里用 3x3 核
        p = d 
        
        self.path_dc = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            # 空洞卷积
            nn.Conv2d(c_mid, c_mid, 3, padding=p, dilation=d, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU()
        )

        # --- LAM 注意力 ---
        self.lam = LAM(c2) # c_mid + c_mid = c2

        # --- Transition Block ---
        self.transition = nn.Sequential(
             nn.Conv2d(c2, c2, 1, bias=False),
             nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        # 双路处理
        x1 = self.path_lk(x)
        x2 = self.path_dc(x)
        
        # 拼接 [cite: 376]
        out = torch.cat([x1, x2], dim=1)
        
        # 注意力加权 [cite: 444]
        out = self.lam(out)
        
        # 输出转换
        return self.transition(out)
