"""Retrain the models on the TEKNOFEST train/val split (GPU strongly recommended).

    python scripts/train.py                      # all six models
    python scripts/train.py HybridFusion_R50_AlexNet --out model_weights_new

Writes to --out (default model_weights_retrained/) so the weights behind the paper are never overwritten.
Retraining will not reproduce the paper's numbers exactly: GPU training is not bit-for-bit deterministic.

Schedule (from the original notebook):
  phase 1  all models: frozen backbone, Adam 1e-4, early stop on val AUC (patience 5)
  phase 2  hybrid: unfreeze ResNet50 conv5, Adam 1e-5
  phase 3  hybrid: unfreeze conv4 + conv5, Adam 5e-6
Phases 2 and 3 are kept only if they improve validation AUC (the original notebook used test AUC for this).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from stroke import data, models

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*", default=models.NAMES)
ap.add_argument("--out", type=Path, default=data.ROOT / "model_weights_retrained")
ap.add_argument("--epochs", type=int, default=100)
args = ap.parse_args()
args.out.mkdir(exist_ok=True)
keras.utils.set_random_seed(data.SEED)


def dataset(name, shuffle):
    paths, y = data.subset(name)
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=data.SEED, reshuffle_each_iteration=True)
    else:
        ds = ds.cache()
    load = lambda p, l: (tf.image.resize(tf.io.decode_png(tf.io.read_file(p), channels=3), data.IMAGE_SIZE), l)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)


train_ds = dataset("train", True)
val_ds = dataset("val", False)


def fit(model, lr, patience, factor, rlr_patience, min_lr, epochs):
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=factor, patience=rlr_patience, min_lr=min_lr)])
    y = np.concatenate([l.numpy() for _, l in val_ds])
    return roc_auc_score(y, model.predict(val_ds, verbose=0).reshape(-1))


def unfreeze(model, blocks):
    resnet = next(l for l in model.layers if isinstance(l, keras.Model) and l.name.startswith("resnet"))
    resnet.trainable = True
    for l in resnet.layers:
        l.trainable = any(b in l.name for b in blocks)


for name in args.names:
    print(f"\n=== {name}", flush=True)
    model = models.build(name, imagenet=True)
    best = fit(model, 1e-4, 5, 0.3, 2, 1e-6, args.epochs)
    best_weights = model.get_weights()
    print(f"phase 1 val AUC {best:.4f}")
    if name == "HybridFusion_R50_AlexNet":
        for phase, blocks, lr, patience, factor, rlr_p, min_lr, epochs in [
                (2, ["conv5_block"], 1e-5, 5, 0.3, 2, 1e-7, args.epochs),
                (3, ["conv4_block", "conv5_block"], 5e-6, 8, 0.5, 3, 1e-8, 50)]:
            unfreeze(model, blocks)
            auc = fit(model, lr, patience, factor, rlr_p, min_lr, epochs)
            print(f"phase {phase} val AUC {auc:.4f} ({'kept' if auc >= best else 'discarded'})")
            if auc >= best:
                best, best_weights = auc, model.get_weights()
            else:
                model.set_weights(best_weights)
    model.set_weights(best_weights)
    model.save_weights(str(args.out / f"{name}.weights.h5"))
    keras.backend.clear_session()
