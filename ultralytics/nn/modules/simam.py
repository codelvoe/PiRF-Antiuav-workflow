import torch
import torch.nn as nn

class SimAM(nn.Module):
    """
    Parameter-free attention (SimAM-style) for Ultralytics parse_model.

    Compatible init signatures:
      - SimAM(c1, c2, e)
      - SimAM(c1, c2)
      - SimAM(c1, e)   (rare, but keep compatible)
    """
    def __init__(self, c1: int, c2: int = None, e: float = 1e-4):
        super().__init__()
        self.c1 = int(c1)

        # 兼容两种传参：
        # 1) (c1, c2, e) -> c2是int, e是float
        # 2) (c1, e)     -> c2被传成float，此时把它当作e
        if c2 is not None and isinstance(c2, float):
            e = c2
            c2 = None

        self.c2 = int(c2) if c2 is not None else self.c1
        self.e = float(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"SimAM expects BCHW, got {tuple(x.shape)}")

        b, c, h, w = x.shape
        n = h * w - 1
        if n <= 0:
            n = 1

        mu = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mu) ** 2).sum(dim=(2, 3), keepdim=True) / n

        energy = (x - mu) ** 2 / (4.0 * (var + self.e)) + 0.5
        return x * torch.sigmoid(energy)
