# BrainStroke Native Windows Setup Guide

## Project Overview
This repository implements a CT-slice stroke detection and explainability pipeline.

Key contents:
- `brainstroke.ipynb` — main notebook for model loading, inference, evaluation, GradCAM++ visualization, lesion ROI extraction, and benchmark generation.
- `data_split/` — image dataset split into `train/`, `val/`, and `test/`, each containing `Normal/` and `Stroke/` subfolders.
- `model_weights/` — saved weights for pre-trained and custom models, including:
  - `AlexNet.weights.h5`
  - `ResNet50.weights.h5`
  - `DenseNet121.weights.h5`
  - `MobileNetV2.weights.h5`
  - `InceptionV3.weights.h5`
  - `HybridFusion_R50_AlexNet.weights.h5`
- `benchmark.csv` — consolidated benchmark results produced by the notebook.
- `requirements.txt` — native Windows Python package list for inference and visualization.
- `.vscode/settings.json` — workspace interpreter settings for `.venv`.

This guide is focused on native Windows execution only, using inference-ready weights and avoiding WSL-specific GPU/CUDA setup.

## Detailed Project Context

### What the project does
- Loads a medical image dataset from `data_split/`.
- Builds and evaluates binary classification models for stroke versus normal CT slices.
- Uses both custom and transfer-learning model architectures.
- Generates GradCAM++ saliency maps and applies brain tissue masks to derive attention-based ROIs.
- Produces a final benchmark CSV sorted by key metrics.

### Model architectures
- `AlexNet` — custom convolutional network built from scratch for binary stroke detection.
- `ResNet50`, `DenseNet121`, `MobileNetV2`, `InceptionV3` — transfer learning backbones with ImageNet weights, frozen feature extractors, and custom classifier heads.
- `HybridFusion_R50_AlexNet` — custom fusion model combining a frozen ResNet50 branch with a modified AlexNet branch, using gated feature fusion and dense layers for final stroke probability.

### Evaluation and explainability
- The notebook evaluates saved weights on the held-out `test/` split.
- Metrics include accuracy, precision, recall, F1, AUC, loss, specificity, sensitivity, and confusion matrix counts.
- GradCAM++ heatmaps are computed for a selected convolutional layer to visualize stroke attention.
- Tissue-aware ROI extraction uses binary morphology and connected component analysis to identify high-activation regions inside the brain.
- A final presentation section selects one normal image and one stroke image for qualitative visualization.

### Supporting script
- `optimize_blend_report.py` performs model blending and meta-model optimization for candidate models, including logistic regression on val/test predictions.
- It also depends on the same `data_split/` structure and `model_weights/` weights.

### Current native Windows state
- A native Windows Python environment was created as `.venv` using Python 3.12.
- The notebook and script paths were updated to use workspace-relative paths via `Path(".").resolve()`.
- `.vscode/settings.json` now points at `${workspaceFolder}/.venv/Scripts/python.exe`.
- `requirements.txt` has been trimmed for native Windows inference usage and no longer includes CUDA-specific `nvidia-*` packages.
- The old WSL environment `venv_gpu/` is not required and may be deleted once Windows setup is confirmed.

## File Inventory and Important Paths

- `brainstroke.ipynb`
  - Uses `PROJECT_DIR = Path(".").resolve()`.
  - Loads dataset from `data_split/`.
  - Loads weights from `model_weights/{model_name}.weights.h5`.
  - Writes benchmark output to `benchmark.csv`.
- `optimize_blend_report.py`
  - Uses the same relative workspace path model and dataset layout.
  - Contains logistic regression stacking and threshold tuning.
- `.vscode/settings.json`
  - Configures VS Code to use the `.venv` interpreter and local Jupyter server.
- `requirements.txt`
  - Includes `tensorflow==2.17.0`, `keras==3.13.2`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `matplotlib==3.10.8`, `pandas==3.0.2`, `numpy==1.26.4`, and related dependencies.

## Data and supported formats

- `data_split/` contains three splits: `train/`, `val/`, `test/`.
- Each split contains two classes: `Normal/` and `Stroke/`.
- Supported image file extensions in the notebook include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`.
- The notebook resizes images to `224x224` and uses `tf.keras.utils.image_dataset_from_directory`.

## Native Windows Setup Instructions

### 1. Install Python

Install Python 3.12.x for Windows and add it to `PATH`.

### 2. Create the virtual environment

From the repository root:

```powershell
cd C:\Work\Inspirante_Brainstroke
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Python and TensorFlow

```powershell
python -c "import sys; import tensorflow as tf; print(sys.version); print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

### 5. Confirm VS Code configuration

Open `brainstroke.ipynb` and ensure the selected kernel uses `.venv\Scripts\python.exe`.

### 6. Run the notebook

Run notebook cells in order:
- imports and environment checks
- dataset loading and `train/val/test` validation
- model build/load and weight evaluation
- GradCAM++ visualization and ROI extraction
- benchmark generation via `run_full_benchmark_from_weights()` if desired.

## Research-Paper Context

This document should be considered the complete operational summary for the native Windows workflow.
For a paper, the important points are:

- Problem statement: stroke classification from CT-like images using binary classification.
- Dataset: `Normal` vs `Stroke`, split into train/validation/test.
- Models: custom AlexNet, transfer learning (ResNet50, DenseNet121, MobileNetV2, InceptionV3), and hybrid fusion model (`HybridFusion_R50_AlexNet`).
- Training status: models are already trained; the repository is now focused on inference, evaluation, and explainability.
- Metrics: accuracy, precision, recall, F1, AUC, loss, specificity, sensitivity, and confusion metrics.
- Explainability: GradCAM++ heatmaps, tissue masking, connected components, and ROI extraction inside brain tissue.
- Deployment target: native Windows notebook environment, with `.venv` and `requirements.txt` validated.
- Benchmark artifact: `benchmark.csv` is the final consolidated results file.

If the paper needs additional detail, reference the following implementation notes:
- `brainstroke.ipynb` defines a fixed random seed (`SEED = 42`) for reproducibility.
- Image size is fixed at `224x224` with batch size `32`.
- The `HybridFusion_R50_AlexNet` model uses gated concatenation followed by dense layers and dropout.
- GradCAM++ heatmap extraction is implemented manually with nested `tf.GradientTape` calls.

## Troubleshooting and validation

### Common issues

- Missing weights: verify all `model_weights/*.weights.h5` files exist.
- Missing test images: verify `data_split/test/Stroke` and `data_split/test/Normal` contain valid image files.
- Notebook retraining warning: if `HybridFusion_R50_AlexNet.weights.h5` is missing, the notebook may attempt to retrain that model.
- Interpreter mismatch: make sure `.vscode/settings.json` points to `.venv/Scripts/python.exe`.

### Validation commands

```powershell
python -c "import tensorflow as tf; import numpy as np; import pandas as pd; import matplotlib; print('ok')"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

### Cleanup

- Remove `venv_gpu/` after verifying `.venv` works.
- Remove `__pycache__/` if present.
- Keep `benchmark.csv` as the published result file.

## Notes

- This repository is now configured for native Windows inference and visualization.
- The original WSL-specific environment is no longer required.
- The notebook is intended for evaluation and explainability; retraining is not necessary unless a required weight file is absent.
