import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcra.config import MCRAConfig
from src.mcra.models import ResTR


class RFDataset(Dataset):
    def __init__(self, csv_path: Path, classes: tuple):
        self.df = pd.read_csv(csv_path)
        self.classes = classes
        self.cls_to_id = {c: i for i, c in enumerate(classes)}
        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.tf(Image.open(row["rf_png_path"]).convert("RGB"))
        y = self.cls_to_id.get(row["uav_type"], self.cls_to_id.get("Background", 0))
        return x, y


def train_rf():
    cfg = MCRAConfig()
    if not cfg.pairs_csv.exists():
        raise FileNotFoundError(cfg.pairs_csv)
    ds = RFDataset(cfg.pairs_csv, cfg.classes)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResTR(num_classes=len(cfg.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    cfg.rf_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.rf_model_path)


if __name__ == "__main__":
    train_rf()
