"""Grad-CAM++ saliency analysis of the hybrid model over the whole test split (paper Table 9, Fig. 4).

    python scripts/explain.py            # all 668 test slices
    python scripts/explain.py --figure   # also save a pipeline figure for the most confident stroke slice
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stroke import data, models, xai

out = data.RESULTS
out.mkdir(exist_ok=True)
paths, y = data.subset("test")
X = data.load_images(paths)
model = models.load("HybridFusion_R50_AlexNet")
cam = xai.HybridCAM(model)
p = model.predict(X, batch_size=32, verbose=0).reshape(-1)

rows = []
for i in range(len(X)):
    r = xai.analyse(X[i], cam(X[i:i + 1]))
    rows.append(dict(path=str(paths[i].relative_to(data.TEKNOFEST)), y=int(y[i]), p=float(p[i]),
                     **{f"{b}_{k}": (len(r[b][k]) > 0 if k == "roi" else r[b][k])
                        for b in "AB" for k in ["ohr", "ac_raw", "ac_masked", "roi"]}))
    if i % 100 == 0:
        print(f"{i}/{len(X)}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(out / "saliency_per_slice.csv", index=False)

q = lambda v: f"{np.median(v):.2f} ({np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f})"
summary = []
for b, layer in [("A", "ResNet50 conv5"), ("B", "conv3")]:
    for lbl, cls in [(1, "Stroke"), (0, "Normal")]:
        d = df[df.y == lbl]
        summary.append({"Map": f"Branch {b} ({layer})", "Class": cls, "OHR (unmasked)": q(d[f"{b}_ohr"]),
                        "AC (unmasked)": q(d[f"{b}_ac_raw"]), "AC (masked)": q(d[f"{b}_ac_masked"]),
                        "Slices with ROI (%)": round(100 * d[f"{b}_roi"].mean(), 1),
                        "Wilcoxon p (AC masked vs raw)": wilcoxon(d[f"{b}_ac_masked"], d[f"{b}_ac_raw"]).pvalue})
summary = pd.DataFrame(summary)
summary.to_csv(out / "saliency_summary.csv", index=False)
print(summary.to_string(index=False))

if "--figure" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    i = int(np.argmax(np.where(y == 1, p, -1)))
    r = xai.analyse(X[i], cam(X[i:i + 1]))
    img = X[i].astype(np.uint8)
    overlay = np.zeros(img.shape[:2] + (3,))
    overlay[r["skull"]] = (1, 0, 0)
    overlay[r["brain"]] = (0, 1, 0)
    panels = [("Input", img, None), ("Grad-CAM++ (Branch A)", img, r["A"]["smooth"]), ("Head mask", r["head"], None),
              ("Skull (red), brain (green)", overlay, None), ("Masked map + ROI", img, r["A"]["masked"])]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (title, base, heat) in zip(axes, panels):
        ax.imshow(base, cmap="gray")
        if heat is not None:
            ax.imshow(heat, cmap="jet", alpha=0.5)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for x1, y1, x2, y2 in r["A"]["roi"]:
        axes[-1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.5))
    fig.suptitle(f"{paths[i].name}  p(stroke) = {p[i]:.3f}", fontsize=10)
    fig.savefig(out / "explainability_pipeline.png", dpi=200, bbox_inches="tight")
    print(f"figure: {out / 'explainability_pipeline.png'}")
