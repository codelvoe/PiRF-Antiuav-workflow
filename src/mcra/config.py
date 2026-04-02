from dataclasses import dataclass
from pathlib import Path


@dataclass
class MCRAConfig:
    data_root: Path = Path("data-sample")
    pairs_csv: Path = Path("artifacts/pairs_index.csv")
    rf_model_path: Path = Path("artifacts/checkpoints/restr_best.pth")
    visual_model_path: Path = Path("artifacts/checkpoints/hasa_dws_best.pt")
    output_dir: Path = Path("artifacts/infer")
    sample_window_ms: float = 30000.0
    center_offset: float = 0.5
    conf_thres: float = 0.25
    fusion_thres: float = 0.40
    lambda_rf: float = 0.6
    alpha_noise: float = 2.0
    classes: tuple = ("AIR2s", "AIR3", "Background", "jinlin4p", "mavic3 pro", "mini2SE")
