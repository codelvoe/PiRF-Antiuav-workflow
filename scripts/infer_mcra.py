import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcra import MCRAFramework
from src.mcra.config import MCRAConfig


if __name__ == "__main__":
    cfg = MCRAConfig()
    framework = MCRAFramework(cfg)
    out = framework.infer_pairs(cfg.pairs_csv)
    print(out)
