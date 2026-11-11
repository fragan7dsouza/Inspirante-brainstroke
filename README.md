# Brain stroke classification on non-contrast CT

Companion code for *Gated Fusion of ImageNet-Pretrained and CT-Specific Convolutional Features with Brain-Masked
Grad-CAM++ for Brain Stroke Classification on Non-Contrast CT* (D'Souza, MJAIH 2026; `paper/`).

**HybridFusion_R50_AlexNet** joins an ImageNet-pretrained ResNet50 branch and a shallow AlexNet-style branch trained
from scratch on CT through a learned sigmoid gate. It is compared with five single-network baselines, explained with
brain-masked Grad-CAM++, and tested on two external datasets it never saw during training.

## Results

### TEKNOFEST-2021 held-out test set (668 slices, 334 stroke)

Reproduced exactly by `python scripts/evaluate.py` from the weights in `model_weights/`.

| Model | AUC (95% CI) | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| **HybridFusion_R50_AlexNet** | **0.967** (0.954–0.979) | **0.882** | **0.799** | **0.964** | **0.871** |
| AlexNet (from scratch) | 0.955 (0.939–0.969) | 0.879 | 0.796 | 0.961 | 0.868 |
| ResNet50 | 0.915 (0.893–0.936) | 0.811 | 0.677 | 0.946 | 0.782 |
| DenseNet121 | 0.896 (0.872–0.919) | 0.796 | 0.665 | 0.928 | 0.766 |
| MobileNetV2 | 0.896 (0.873–0.918) | 0.778 | 0.611 | 0.946 | 0.734 |
| InceptionV3 | 0.892 (0.867–0.914) | 0.810 | 0.728 | 0.892 | 0.793 |

The hybrid beats the four ImageNet backbones (DeLong, Holm-adjusted p < 0.001) but not the from-scratch AlexNet
(p = 0.131). Sensitivity is 87.1% for hemorrhagic and 73.7% for ischemic slices.
Full tables: `results/test_metrics.csv`, `test_comparisons.csv`, `test_subtype_sensitivity.csv`.

### External validation (trained on TEKNOFEST, tested unchanged on other hospitals' data)

`python scripts/external.py aisd` / `cq500`. No retraining, no recalibration, threshold 0.5.
CIs resample patients (AISD) or scans (CQ500).

| Model | AISD ischemic, AUC (95% CI) | CQ500 hemorrhage, AUC (95% CI) |
|---|---|---|
| HybridFusion_R50_AlexNet | 0.625 (0.603–0.647) | 0.750 (0.706–0.794) |
| AlexNet | 0.663 (0.639–0.687) | 0.707 (0.659–0.755) |
| ResNet50 | 0.700 (0.679–0.719) | 0.712 (0.662–0.758) |
| DenseNet121 | 0.628 (0.605–0.650) | 0.705 (0.658–0.752) |
| MobileNetV2 | 0.587 (0.566–0.611) | 0.753 (0.707–0.796) |
| InceptionV3 | 0.716 (0.700–0.733) | 0.702 (0.652–0.747) |

- **AISD** (397 patients with acute ischemic stroke, 10,677 slices, 4,269 with infarct). Slice level: infarct slices
  versus lesion-free slices of the same patients. The hybrid detects only 28% of infarct slices (specificity 88%).
- **CQ500** (464 scans with a 3–5 mm series, 199 with hemorrhage by majority of 3 radiologists). Scan level, scan
  score = highest slice probability. The hybrid flags 95% of hemorrhage scans but also 81% of normal ones at the
  0.5 threshold; it ranks scans better than four of five baselines, but no difference survives Holm correction.

**What this means.** Performance drops sharply outside TEKNOFEST (0.967 → 0.63–0.75 AUC), for every model.
The hybrid's advantage on TEKNOFEST does not carry over: on ischemic stroke it is worse than ResNet50 and InceptionV3.
The likely causes are the slice-level TEKNOFEST split (patient leakage inflates in-domain results), differences in
scanners and image export, and, for CQ500, a 0.5 threshold that was never calibrated for scan-level max scores.
Any clinical claim needs training on multi-site data with patient-level splits.
Per-slice/per-scan predictions and paired tests: `results/external_*`. The AISD paired tests treat slices as
independent, so their p-values are optimistic.

## Setup (Windows, macOS or Linux; CPU is enough for everything except training)

```
python -m venv .venv            # Python 3.12
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install -r requirements.txt
```

The TEKNOFEST images are in `data/teknofest/` and the weights in `model_weights/` (Git LFS: run `git lfs pull`).

## Commands

| Command | What it does | Time (CPU) |
|---|---|---|
| `python scripts/evaluate.py` | All 6 models on the test split: metrics, bootstrap CIs, DeLong/McNemar/Brier tests, subtype sensitivity | ~4 min |
| `python scripts/explain.py --figure` | Grad-CAM++ of both hybrid branches on all test slices: saliency table + pipeline figure | ~15 min |
| `python scripts/prepare_cq500.py` | Download CQ500 and convert to brain-window PNGs | ~1 h |
| `python scripts/external.py aisd` / `cq500` | External validation (see below) | ~30–60 min |
| `python scripts/train.py` | Retrain from ImageNet weights into `model_weights_retrained/` | GPU, hours |

Retraining will not reproduce the paper's numbers exactly (GPU training is not deterministic) and selects
fine-tuning phases on the validation set, whereas the original notebook used the test set for that one decision.

## Layout

```
stroke/        data.py (split + loading)  models.py  stats.py  xai.py (Grad-CAM++, brain mask, OHR/AC, ROI)
scripts/       evaluate.py  explain.py  train.py  prepare_cq500.py  external.py
results/       generated tables and figures; paper_*.csv are paper results whose weights are not in the repo
data/          teknofest/ (tracked)  kaggle/ (LFS zip)  external/ (downloaded, gitignored)
model_weights/ the six trained models (Git LFS)
paper/         manuscript and the original reproducibility scripts
```

## Known limitations

- **Slice-level split.** TEKNOFEST has no patient IDs, so slices of one patient can sit in both train and test.
  The file contents contain 18 duplicate image pairs, 3 of which cross train/test. Test performance is therefore
  likely optimistic, which is why the external results below matter.
- **Saliency table.** `scripts/explain.py` reimplements the paper's Grad-CAM++ pipeline (its `xai.py` helper was not
  preserved). Branch B matches Table 9 closely. Branch A is close but not identical (stroke OHR 0.41 vs 0.45, ROI rate
  67% vs 59%), because the paper's Branch A maps came from a cached earlier version. The conclusions are unchanged:
  masking removes the off-brain saliency and makes maps more focal, and the ROI box does not separate the classes.
- **Heatmaps are not lesion maps.** The ROI box marks the saliency peak, not the lesion.
- Research code; not for clinical use.

## Data and licenses

- TEKNOFEST-2021 stroke dataset (Turkish Ministry of Health), as used in the paper.
- AISD: Liang et al., MICCAI 2021, github.com/GriffinLiang/AISD. Non-commercial use only.
- CQ500: Chilamkurthy et al., Lancet 2018, CC BY-NC-SA 4.0. Non-commercial use only.
