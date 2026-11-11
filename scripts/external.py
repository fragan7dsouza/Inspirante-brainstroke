"""External validation: models trained on TEKNOFEST, tested unchanged on AISD (ischemic) and CQ500 (hemorrhage).

    python scripts/external.py aisd     # needs data/external/aisd/{image,mask}  (see README)
    python scripts/external.py cq500    # needs scripts/prepare_cq500.py to have run

AISD   slice level. Positive: >= MIN_LESION_PX pixels with label 1, 2, 3 or 5 (authors' infarct definition).
       Slices whose only lesion is label 4 (invisible acute infarct) are excluded. Negatives: lesion-free slices of
       the same patients, so this is harder than TEKNOFEST, where normals come from normal scans.
CQ500  scan level. Positive: intracranial hemorrhage by majority of 3 radiologists. Scan score = max slice probability.
       Only scans with a 3-5 mm series are used.
No retraining, no recalibration, threshold 0.5 as on TEKNOFEST.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np
import pandas as pd
from PIL import Image

from stroke import data, models, stats

MIN_LESION_PX = 50
which = sys.argv[1] if len(sys.argv) > 1 else "aisd"
root = data.DATA / "external" / which


def aisd_slices():
    rows = []
    for pdir in sorted((root / "mask").iterdir()):
        for m in sorted(pdir.glob("*.png")):
            mask = np.array(Image.open(m))
            infarct = np.isin(mask, [1, 2, 3, 5]).sum()
            if infarct < MIN_LESION_PX and (mask == 4).any():
                continue
            rows.append(dict(group=pdir.name, path=root / "image" / pdir.name / m.name, y=int(infarct >= MIN_LESION_PX)))
    return pd.DataFrame(rows)


def cq500_slices():
    labels = pd.read_csv(root / "labels.csv").set_index("scan").ich
    # Thin-slice-only scans (< 3 mm) look unlike the 5 mm training slices and give hundreds of slices per scan,
    # which inflates the max-slice score, so they are excluded.
    scans = [d for d in sorted((root / "png").iterdir())
             if d.is_dir() and 3 <= float((d / "thickness_mm.txt").read_text()) <= 5]
    rows = [dict(group=d.name, path=p, y=int(labels[d.name])) for d in scans for p in sorted(d.glob("*.png"))]
    return pd.DataFrame(rows)


df = aisd_slices() if which == "aisd" else cq500_slices()
print(f"{which}: {df.group.nunique()} scans, {len(df)} slices, {int(df.y.sum())} positive slices", flush=True)

for name in models.NAMES:
    model = models.load(name)
    probs = []
    for _, g in df.groupby("group", sort=False):
        probs.append(model.predict(data.load_images(g.path.tolist()), batch_size=32, verbose=0).reshape(-1))
    df[name] = np.concatenate(probs)
    keras.backend.clear_session()
    print(f"{name} done", flush=True)

if which == "cq500":
    df = df.groupby("group").agg({"y": "first", **{n: "max" for n in models.NAMES}}).reset_index()
df.assign(path=df.get("path", pd.Series(dtype=str)).astype(str)).to_csv(data.RESULTS / f"external_{which}_predictions.csv",
                                                                       index=False)

y = df.y.to_numpy()
P = {n: df[n].to_numpy() for n in models.NAMES}
rows = []
for n in models.NAMES:
    m, ci = stats.metrics(y, P[n]), stats.boot_ci(y, P[n], B=1000, groups=df.group.to_numpy())
    rows.append({"Model": n, **m, **{f"{k}_CI": f"{lo:.4f}-{hi:.4f}" for k, (lo, hi) in ci.items()}})
pd.DataFrame(rows).to_csv(data.RESULTS / f"external_{which}_metrics.csv", index=False)
pd.DataFrame(stats.compare(y, P, models.NAMES[0])).to_csv(data.RESULTS / f"external_{which}_comparisons.csv", index=False)
print(f"\n{which}: n={len(y)} ({int(y.sum())} positive)")
print(pd.DataFrame(rows)[["Model", "AUC", "AUC_CI", "Accuracy", "Recall", "Specificity"]].round(4).to_string(index=False))
