import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def compute_iou(box1, box2):
    xx1, yy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xx2, yy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / (union + 1e-6)


def compute_nwd(box1, box2, c=12.8):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    cx1, cy1 = box1[0] + w1 / 2, box1[1] + h1 / 2
    cx2, cy2 = box2[0] + w2 / 2, box2[1] + h2 / 2
    w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    return np.exp(-np.sqrt(w2_dist) / c)


def evaluate_visual_only():
    model = YOLO("artifacts/checkpoints/hasa_dws_best.pt")
    img_dir = Path("data-sample/visual/rgb")
    imgs = sorted(list(img_dir.glob("*.jpg")))
    t0 = time.time()
    for img in imgs:
        model.predict(str(img), imgsz=1280, conf=0.25, verbose=False)
    elapsed = time.time() - t0
    fps = len(imgs) / max(elapsed, 1e-6)
    print({"num_images": len(imgs), "fps": fps})


if __name__ == "__main__":
    evaluate_visual_only()
