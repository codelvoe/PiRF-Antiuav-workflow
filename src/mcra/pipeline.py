from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from .config import MCRAConfig
from .fusion.state_machine import ThreeStateMachine
from .models.restr import ResTR


class MCRAFramework:
    def __init__(self, cfg: MCRAConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual = YOLO(str(cfg.visual_model_path))
        self.rf = self._load_rf_model(cfg.rf_model_path)
        self.rf_tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.machine = ThreeStateMachine(
            conf_thres=cfg.conf_thres,
            fusion_thres=cfg.fusion_thres,
            lambda_rf=cfg.lambda_rf,
            alpha_noise=cfg.alpha_noise,
        )

    def _load_rf_model(self, weight_path: Path):
        model = ResTR(num_classes=len(self.cfg.classes)).to(self.device)
        if weight_path.exists():
            state = torch.load(weight_path, map_location=self.device)
            model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def rf_predict(self, rf_png_path: str):
        img = Image.open(rf_png_path).convert("RGB")
        x = self.rf_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.rf(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        cls = self.cfg.classes[idx]
        if "Background" in self.cfg.classes:
            bg_idx = self.cfg.classes.index("Background")
            p_noise = float(probs[bg_idx])
            p_target = float(1.0 - p_noise)
        else:
            p_target = float(np.max(probs))
            p_noise = float(1.0 - p_target)
        return p_target, p_noise, cls

    def visual_predict(self, img_path: str):
        pred = self.visual.predict(img_path, imgsz=1280, conf=0.01, verbose=False)[0]
        if len(pred.boxes) == 0:
            return 0.0, None
        score = float(pred.boxes.conf[0])
        box = pred.boxes.xyxy[0].cpu().numpy().tolist()
        return score, box

    def infer_pairs(self, pairs_csv: Path):
        df = pd.read_csv(pairs_csv)
        rows = []
        for _, row in df.iterrows():
            img_path = row["img_path"]
            rf_path = row["rf_png_path"]
            gt_cls = row.get("uav_type", "Background")
            is_target = gt_cls != "Background"
            s_vis, vis_box = self.visual_predict(img_path)
            p_target, p_noise, rf_cls = self.rf_predict(rf_path)
            out = self.machine.decide(s_vis, p_target, p_noise, is_target=is_target)
            rows.append(
                {
                    "img_path": img_path,
                    "rf_png_path": rf_path,
                    "uav_type": gt_cls,
                    "rf_pred": rf_cls,
                    "score_visual": out.score_visual,
                    "prob_target": out.prob_target,
                    "prob_noise": out.prob_noise,
                    "score_final": out.score_final,
                    "state": out.state,
                    "vis_box": vis_box,
                }
            )
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = self.cfg.output_dir / "mcra_infer.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return out_csv
