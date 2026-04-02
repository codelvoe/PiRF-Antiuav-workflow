# M-CRA Example Framework

This repository is organized as a compact example implementation of the M-CRA paper framework.

## Core Modules

- `src/mcra/data`: temporal anchoring and visual-RF pair indexing
- `src/mcra/rf`: IQ parsing and RF spectrogram construction
- `src/mcra/models`: ResTR RF parser model
- `src/mcra/fusion`: three-state probabilistic collaboration logic
- `src/mcra/pipeline.py`: end-to-end collaborative inference pipeline

## Script Entry Points

- `scripts/build_pairs_index.py`
- `scripts/train_visual.py`
- `scripts/train_rf.py`
- `scripts/infer_mcra.py`
- `scripts/evaluate_visual.py`

## Workflow

1. Build paired temporal index
2. Train HASA-DWS visual detector and ResTR RF parser
3. Run three-state probabilistic collaborative inference
