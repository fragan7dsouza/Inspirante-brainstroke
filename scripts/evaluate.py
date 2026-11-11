"""Evaluate the six saved models on the TEKNOFEST test split and write every result table to results/.

    python scripts/evaluate.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import pandas as pd

from stroke import data, models, stats

out = data.RESULTS
out.mkdir(exist_ok=True)
paths, y = data.subset("test")
X = data.load_images(paths)
print(f"test slices: {len(y)} ({int(y.sum())} stroke)", flush=True)

P = {}
for name in models.NAMES:
    P[name] = models.load(name).predict(X, batch_size=32, verbose=0).reshape(-1)
    keras.backend.clear_session()
    print(f"{name:26s} AUC {stats.metrics(y, P[name])['AUC']:.4f}", flush=True)

pred = pd.DataFrame({"path": [str(p.relative_to(data.TEKNOFEST)) for p in paths], "y": y,
                     "subtype": [data.subtype(p) for p in paths], **P})
pred.to_csv(out / "test_predictions.csv", index=False)

rows = []
for n in models.NAMES:
    m, ci = stats.metrics(y, P[n]), stats.boot_ci(y, P[n])
    rows.append({"Model": n, **m, **{f"{k}_CI": f"{lo:.4f}-{hi:.4f}" for k, (lo, hi) in ci.items()}})
pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
pd.DataFrame(stats.compare(y, P, models.NAMES[0])).to_csv(out / "test_comparisons.csv", index=False)
(pred[pred.y == 1].groupby("subtype")[models.NAMES].agg(lambda p: (p >= 0.5).mean())
 .to_csv(out / "test_subtype_sensitivity.csv"))

pd.set_option("display.width", 200)
print(pd.DataFrame(rows)[["Model", "Accuracy", "Recall", "Specificity", "F1", "AUC", "AUC_CI"]].round(4).to_string(index=False))
print(f"\nwritten to {out}")
