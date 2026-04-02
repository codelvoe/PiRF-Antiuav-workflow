import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    """
    DySample: Lightweight Dynamic Upsampler for Semantic Segmentation (ICCV 2023)
    Adapted for YOLO Object Detection (Ultralytics)
    Stability Fix: Tanh + Pixel Normalization
    """
    def __init__(self, c1, c2, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        in_channels = c1
        assert c1 == c2, f"DySample expects in/out channels to be same, but got {c1} and {c2}"
        if in_channels % groups != 0:
            self.groups = 1
            
        assert style in ['lp', 'pl']
        if style == 'pl':
            in_channels = c1 // self.groups
            out_channels = 2 * self.groups * scale ** 2
        else:
            out_channels = 2 * self.groups * scale ** 2

        self.offset = nn.Conv2d(c1, out_channels, 1)
        normal_init(self.offset, std=0.001)

        self.register_buffer('base_grid', None, persistent=False)

    def _init_pos(self, x):
        h, w = x.shape[2:]
        rng_h = torch.linspace(-1 + 1/(h*self.scale), 1 - 1/(h*self.scale), h*self.scale, device=x.device)
        rng_w = torch.linspace(-1 + 1/(w*self.scale), 1 - 1/(w*self.scale), w*self.scale, device=x.device)
        grid_y, grid_x = torch.meshgrid(rng_h, rng_w, indexing='ij')
        
        # [1, H*s, W*s, 2]
        self.base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. Generate Offsets
        offset = self.offset(x) 
        
        # 2. Pixel Shuffle
        offset = F.pixel_shuffle(offset, self.scale)
        
        # 3. Stability Fix: Tanh to bound offset to (-1, 1)
        # This prevents the "exploding offset" issue
        offset = torch.tanh(offset)
        
        # 4. Reshape for Group-wise Sampling
        x_reshaped = x.reshape(B * self.groups, C // self.groups, H, W)
        
        offset_reshaped = offset.view(B, self.groups, 2, H*self.scale, W*self.scale)
        offset_reshaped = offset_reshaped.reshape(B * self.groups, 2, H*self.scale, W*self.scale)
        offset_field = offset_reshaped.permute(0, 2, 3, 1) 

        # 5. Init Grid
        if self.base_grid is None or self.base_grid.shape[1] != H*self.scale or self.base_grid.shape[2] != W*self.scale:
            self._init_pos(x)
        
        sample_grid = self.base_grid.expand(B * self.groups, -1, -1, -1)
        
        # 6. Apply Offset with Normalization
        # Previous bug: offset was applied directly to normalized grid [-1, 1] without scaling
        # Now we interpret offset as "percentage of image size".
        # 0.25 means max shift is 1/4 of the image.
        # With Tanh, max value is 1.0. So max shift is 0.125 * ImageSize.
        # This is strictly bounded.
        sample_grid = sample_grid + offset_field * 0.125 
        
        # 7. Sampling
        x_reshaped = x_reshaped.float()
        sample_grid = sample_grid.float()
        output = F.grid_sample(x_reshaped, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # 8. Reshape back
        output = output.view(B, C, H*self.scale, W*self.scale).to(x.dtype)
        
        return output
