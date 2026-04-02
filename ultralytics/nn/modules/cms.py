import torch
import torch.nn as nn
from .conv import Conv

class MACF(nn.Module):
    """Multi-scale Adaptive Context Fusion (MACF) module to replace Concat."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # c1 is a list of input channels [ch_local, ch_global] from parser
        # c2 is output channels from parser
        if isinstance(c1, int):
            c1 = [c1, c1] # Fallback if not list
        
        self.c1 = c1
        self.c2 = c2
        
        # Project inputs to output channels (c2)
        self.cv1 = Conv(c1[0], c2, 1, 1)
        self.cv2 = Conv(c1[1], c2, 1, 1) 
        
        # MSFF: Parallel dilated convs (d=1,2,3,4)
        mid_c = c2 // 4
        self.msff_convs = nn.ModuleList([
            Conv(c2, mid_c, 3, 1, p=1, d=1),
            Conv(c2, mid_c, 3, 1, p=2, d=2),
            Conv(c2, mid_c, 3, 1, p=3, d=3),
            Conv(c2, mid_c, 3, 1, p=4, d=4)
        ])
        
        # Adaptive Gating
        self.gate_conv = nn.Sequential(
            Conv(c2 * 2, c2, 1),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is [x_local, x_global]
        if not isinstance(x, list):
            # If accidentally single input, duplicate (should not happen in correct config)
            x_local = x_global = x
        else:
            x_local, x_global = x[0], x[1]
        
        # Project
        xl = self.cv1(x_local)
        xg = self.cv2(x_global)
        
        # MSFF (produce I)
        msff_out = [conv(xl) for conv in self.msff_convs]
        I = torch.cat(msff_out, 1)
        
        # Adaptive Gating
        cat_feat = torch.cat([xl, xg], 1)
        alpha = self.gate_conv(cat_feat)
        
        # Output
        return I + alpha * xl + (1 - alpha) * xg

class CAM(nn.Module):
    """Cooperative Attention Module (CAM)."""
    def __init__(self, c1, c2):
        super().__init__()
        # c1 is input channels. c2 is output channels (usually same)
        c_out = c2
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(c1, c1 // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // 16, c1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention: 3 parallel branches with different scales
        self.sa_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, c1, (1, 7), padding=(0, 3), groups=c1, bias=False),
                nn.Conv2d(c1, c1, (7, 1), padding=(3, 0), groups=c1, bias=False)
            ),
            nn.Sequential(
                nn.Conv2d(c1, c1, (1, 11), padding=(0, 5), groups=c1, bias=False),
                nn.Conv2d(c1, c1, (11, 1), padding=(5, 0), groups=c1, bias=False)
            ),
            nn.Sequential(
                nn.Conv2d(c1, c1, (1, 21), padding=(0, 10), groups=c1, bias=False),
                nn.Conv2d(c1, c1, (21, 1), padding=(10, 0), groups=c1, bias=False)
            )
        ])
        
        # Pixel Attention Refinement
        self.sa_scale = nn.Conv2d(c1 * 3, 1, 7, padding=3, bias=False)
        self.sa_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        ca = self.ca_mlp(self.avg_pool(x))
        x_ca = x * ca
        
        # Spatial Attention
        sa_outs = [branch(x_ca) for branch in self.sa_branches]
        sa_cat = torch.cat(sa_outs, 1)
        
        # Pixel Attention
        sa_map = self.sa_scale(sa_cat)
        sa_map = self.sa_sigmoid(sa_map)
        
        return x_ca * sa_map
