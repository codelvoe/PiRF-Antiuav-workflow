import re
from pathlib import Path

import numpy as np
import pandas as pd


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", name)]


def ms_of_day_from_hhmmss(hhmmss: str):
    hh, mm, ss = map(int, hhmmss.split("."))
    return (hh * 3600 + mm * 60 + ss) * 1000


def find_time_dirs(cat_root: Path):
    out = []
    for path in cat_root.rglob("*"):
        if path.is_dir() and re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", path.name):
            out.append((path.parent, path, path.name))
    return sorted(out, key=lambda x: (str(x[0]), natural_key(x[2])))


def list_rf_files(time_dir: Path):
    local_dir = time_dir / "sumCorr"
    if local_dir.is_dir():
        files = [p for p in local_dir.iterdir() if p.is_file() and p.name.lower().startswith("sumcorr")]
    else:
        files = [p for p in time_dir.iterdir() if p.is_file() and p.name.lower().startswith("sumcorr")]
    return sorted(files, key=lambda p: natural_key(p.name))


def find_video_path(action_dir: Path, hhmmss: str):
    pattern = f"*-{hhmmss.replace('.', '-')}.avi"
    candidates = list(action_dir.glob(pattern))
    return candidates[0] if candidates else None


def build_timeline_mapping(cat_dir: Path, total_ms: float, center_offset: float):
    mapping = {}
    for action_dir, time_dir, hhmmss in find_time_dirs(cat_dir):
        base_ms = ms_of_day_from_hhmmss(hhmmss)
        video_path = find_video_path(action_dir, hhmmss)
        if video_path is None:
            continue
        rf_files = list_rf_files(time_dir)
        if not rf_files:
            continue
        dt = total_ms / len(rf_files)
        t_abs_list = np.round(base_ms + (np.arange(len(rf_files)) + center_offset) * dt).astype(np.int64)
        for idx, (rf_file, t_abs) in enumerate(zip(rf_files, t_abs_list)):
            item = {
                "uav_type": cat_dir.name,
                "video_path": str(video_path),
                "sumcorr_idx": int(idx),
                "sumcorr_path": str(rf_file),
                "t_abs_ms": int(t_abs),
                "t_rel_ms": float(t_abs - base_ms),
            }
            mapping.setdefault(int(t_abs), []).append(item)
    return mapping


def build_pairs_index(data_root: Path, output_csv: Path, total_ms: float, center_offset: float):
    rows = []
    cats = [p for p in data_root.iterdir() if p.is_dir()]
    for cat_dir in cats:
        mapping = build_timeline_mapping(cat_dir, total_ms=total_ms, center_offset=center_offset)
        for t_abs, entries in mapping.items():
            chosen = entries[0]
            rgb_path = data_root / cat_dir.name / "rgb" / f"{t_abs}.jpg"
            rf_path = data_root / cat_dir.name / "rf" / f"{t_abs}.png"
            if not rgb_path.exists() or not rf_path.exists():
                continue
            rows.append(
                {
                    "uav_type": cat_dir.name,
                    "video_path": chosen["video_path"],
                    "sumcorr_idx": chosen["sumcorr_idx"],
                    "sumcorr_path": chosen["sumcorr_path"],
                    "img_path": str(rgb_path),
                    "rf_png_path": str(rf_path),
                    "t_abs_ms": int(t_abs),
                    "t_rel_ms": chosen["t_rel_ms"],
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(by=["uav_type", "t_abs_ms"]).to_csv(output_csv, index=False)
    return output_csv
