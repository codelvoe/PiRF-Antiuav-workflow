import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResTR(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 8, d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.seq_len = seq_len
        self.proj = nn.Linear(512, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        b, _, _, w = x.shape
        chunk = max(1, w // self.seq_len)
        feats = []
        for i in range(self.seq_len):
            patch = x[:, :, :, i * chunk : (i + 1) * chunk]
            feats.append(self.cnn(patch).view(b, 512))
        seq = torch.stack(feats, dim=1)
        seq = self.proj(seq)
        seq = torch.cat([self.cls_token.expand(b, -1, -1), seq], dim=1) + self.pos
        return self.head(self.encoder(seq)[:, 0, :])
