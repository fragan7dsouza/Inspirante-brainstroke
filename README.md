# Inspirante-brainstroke

Benchmarking lightweight CNN architectures for brain stroke classification from cranial CT, and a custom gated fusion model that outperforms all of them, paired with a tissue-aware explainability pipeline so the model's attention stays inside the brain instead of leaking into the skull or image padding.

## What's in here

- `brainstroke.ipynb` - the full pipeline: data loading/augmentation, model definitions, training, benchmarking, and Grad-CAM++ visualization.
- `benchmark.csv` - raw benchmark results for every architecture evaluated.
- `optimize_blend_report.py` - stacked-ensemble / decision-threshold tuning experiments on top of the individual models.
- `implementation_plan.md` - the working plan for turning this into a research paper (architecture math, dataset details, writing structure).
- `Draft.pdf` - the in-progress paper draft.
- `agents.md` - notes on running the project locally.
- `normal_qualitative_evaluation.png`, `stroke_qualitative_evaluation.png` - qualitative Grad-CAM++ output on normal vs. stroke-positive scans.

## The task

Binary classification (stroke vs. normal) on a balanced set of 4,100 cranial CT slices (2,050 normal, 2,050 stroke), 224x224x3, with standard augmentation (flips, small rotations/zoom/contrast jitter).

## Models benchmarked

Five standard transfer-learning backbones (ResNet50, DenseNet121, MobileNetV2, InceptionV3, a from-scratch AlexNet baseline), plus a custom hybrid: HybridFusion_R50_AlexNet, which merges frozen ResNet50 features with a from-scratch AlexNet-style branch through a learned sigmoid gate, rather than simple concatenation.

Results (from `benchmark.csv`):

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|
| HybridFusion_R50_AlexNet | 0.836 | 0.952 | 0.944 | 0.714 | 0.813 | 0.958 |
| AlexNet (from scratch) | 0.805 | 0.901 | 0.859 | 0.731 | 0.789 | 0.880 |
| ResNet50 | 0.638 | 0.722 | 0.668 | 0.549 | 0.602 | 0.727 |
| InceptionV3 | 0.627 | 0.678 | 0.681 | 0.477 | 0.561 | 0.776 |
| MobileNetV2 | 0.599 | 0.664 | 0.620 | 0.513 | 0.561 | 0.685 |
| DenseNet121 | 0.545 | 0.582 | 0.590 | 0.299 | 0.397 | 0.792 |

The hybrid model beats every single-backbone baseline by a wide margin, and the pure transfer-learning models (ResNet50, InceptionV3, MobileNetV2, DenseNet121) underperform even the from-scratch AlexNet - the ImageNet-pretrained layers don't transfer well to the low-contrast density differences that matter in CT.

## Explainability

Standard Grad-CAM++ tends to highlight the skull and black image padding along with the actual lesion. This project adds a tissue-masking step (grayscale thresholding, morphological opening/closing, largest-connected-component extraction) that restricts the saliency map to intracranial brain tissue before overlaying it, so the heatmap is judged against the anatomy rather than the whole image.

## Stack

TensorFlow/Keras, scikit-learn, scipy (for the morphological masking), pandas/numpy, matplotlib, run from a Jupyter notebook.
