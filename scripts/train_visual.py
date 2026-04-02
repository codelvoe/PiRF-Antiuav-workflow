import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def train_visual():
    model = YOLO("ultralytics/cfg/models/11/Hasa-Yolo.yaml")
    model.train(
        data="configs/datasets/example.yaml",
        epochs=200,
        batch=16,
        imgsz=1280,
        dws=True,# other models is False
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        warmup_epochs=5.0,
        close_mosaic=10,
        weight_decay=0.0005,
        workers=4,
        project="artifacts/runs",
        name="hasa_dws",
    )


if __name__ == "__main__":
    train_visual()
