import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcra.config import MCRAConfig
from src.mcra.data import build_pairs_index


if __name__ == "__main__":
    cfg = MCRAConfig()
    out = build_pairs_index(
        data_root=cfg.data_root,
        output_csv=cfg.pairs_csv,
        total_ms=cfg.sample_window_ms,
        center_offset=cfg.center_offset,
    )
    print(out)
