# Agent notes

Companion code for the paper in `paper/` (gated ResNet50 + AlexNet fusion for stroke on non-contrast CT).
See README.md for setup, commands and results.

## Rules
- `model_weights/*.weights.h5` are the weights behind every number in the paper. Never overwrite them;
  `scripts/train.py` writes to `model_weights_retrained/`.
- `stroke/data.py:split()` must stay byte-for-byte equivalent: the weights were trained on that exact split.
- `stroke/models.py`: don't reorder or rename layers; weights are loaded by structure.
- Acceptance check after any change: `python scripts/evaluate.py` must print hybrid AUC 0.9672, accuracy 0.8817.
- External datasets live in `data/external/` (gitignored). Respect their non-commercial licenses.

## Layout
- `stroke/` library: data split + loading, model builders, metrics/statistics, Grad-CAM++ and brain masking
- `scripts/` one entry point per task: evaluate, explain, train, prepare_cq500, external
- `results/` generated tables and figures (`paper_*.csv` are paper results whose weights are not in the repo)
- `data/teknofest/` TEKNOFEST-2021 PNGs (tracked), `data/kaggle/` second dataset zip (LFS)
