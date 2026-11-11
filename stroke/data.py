"""TEKNOFEST-2021 split and image loading.

The split reproduces the one the saved weights were trained on: seed 42, sorted file lists,
Normal undersampled to the Stroke count with default_rng(42), then 70/15/15 per class."""
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEKNOFEST = DATA / "teknofest"
WEIGHTS = ROOT / "model_weights"
RESULTS = ROOT / "results"
SEED = 42
IMAGE_SIZE = (224, 224)
CLASSES = ["Normal", "Stroke"]


def _pngs(folder):
    return sorted(folder.glob("*.png"))


def split():
    """Return {"Normal"|"Stroke": {"train"|"val"|"test": [Path]}}."""
    stroke = sorted(_pngs(TEKNOFEST / "Bleeding" / "PNG") + _pngs(TEKNOFEST / "Ischemia" / "PNG"))
    normal_all = _pngs(TEKNOFEST / "Normal" / "PNG")
    if not stroke or not normal_all:
        raise FileNotFoundError(f"TEKNOFEST images not found under {TEKNOFEST}")
    normal = np.random.default_rng(SEED).choice(np.array(normal_all, dtype=object), size=min(len(stroke), len(normal_all)),
                                                replace=False).tolist()
    out = {}
    for cls, paths in [("Stroke", stroke), ("Normal", normal)]:
        arr = list(paths)
        np.random.default_rng(SEED).shuffle(arr)
        n_tr, n_va = int(len(arr) * 0.70), int(len(arr) * 0.15)
        out[cls] = {"train": arr[:n_tr], "val": arr[n_tr:n_tr + n_va], "test": arr[n_tr + n_va:]}
    return out


def subset(name):
    """Paths and labels (0 Normal, 1 Stroke) for one split, in image_dataset_from_directory order."""
    s = split()
    paths, y = [], []
    for lbl, cls in enumerate(CLASSES):
        ps = sorted(s[cls][name], key=lambda p: p.name)
        paths += ps
        y += [lbl] * len(ps)
    return paths, np.array(y, np.int32)


def load_images(paths):
    """Decode -> RGB -> bilinear resize (no antialias) -> float32 in 0..255, as image_dataset_from_directory does."""
    arr = np.zeros((len(paths),) + IMAGE_SIZE + (3,), np.float32)
    for i, p in enumerate(paths):
        img = tf.io.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False)
        arr[i] = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=False).numpy()
    return arr


def subtype(path):
    return "Bleeding" if "Bleeding" in Path(path).parts else "Ischemia" if "Ischemia" in Path(path).parts else "Normal"
