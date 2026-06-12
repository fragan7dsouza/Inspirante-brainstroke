import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_DIR = Path(".").resolve()
SPLIT_DIR = PROJECT_DIR / "data_split"
BENCHMARK_CSV = PROJECT_DIR / "benchmark.csv"
MODEL_WEIGHTS_DIR = PROJECT_DIR / "model_weights"

CLASS_NAMES = ["Normal", "Stroke"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def make_data_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_alexnet():
    model = keras.Sequential(
        [
            layers.Input(shape=IMAGE_SIZE + (3,)),
            layers.Rescaling(1.0 / 255.0),
            make_data_augmentation(),
            layers.Conv2D(96, 11, strides=4, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(3, strides=2),
            layers.Conv2D(256, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(3, strides=2),
            layers.Conv2D(384, 3, padding="same", activation="relu"),
            layers.Conv2D(384, 3, padding="same", activation="relu"),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(3, strides=2),
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid", dtype="float32"),
        ],
        name="AlexNet",
    )
    return model


def build_resnet50():
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE + (3,),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,), name="ResNet50_input")
    x = make_data_augmentation()(inputs)
    x = layers.Lambda(resnet_preprocess, name="ResNet50_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="ResNet50_gap")(x)
    x = layers.Dropout(0.3, name="ResNet50_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="ResNet50_output")(x)
    return keras.Model(inputs, outputs, name="ResNet50")


def build_hybrid_fusion_r50_alexnet():
    resnet_backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE + (3,),
    )
    resnet_backbone.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,), name="HybridFusion_input")
    augmented = make_data_augmentation()(inputs)

    a = layers.Lambda(resnet_preprocess, name="hybrid_resnet_preprocess")(augmented)
    a = resnet_backbone(a, training=False)
    a = layers.GlobalAveragePooling2D(name="hybrid_resnet_gap")(a)

    b = layers.Rescaling(1.0 / 255.0, name="hybrid_alex_rescale")(augmented)
    b = layers.Conv2D(96, 11, strides=4, padding="same", activation="relu", name="hybrid_alex_conv1")(b)
    b = layers.BatchNormalization(name="hybrid_alex_bn1")(b)
    b = layers.MaxPooling2D(3, strides=2, name="hybrid_alex_pool1")(b)
    b = layers.Conv2D(256, 5, padding="same", activation="relu", name="hybrid_alex_conv2")(b)
    b = layers.BatchNormalization(name="hybrid_alex_bn2")(b)
    b = layers.MaxPooling2D(3, strides=2, name="hybrid_alex_pool2")(b)
    b = layers.Conv2D(256, 3, padding="same", activation="relu", name="hybrid_alex_conv3")(b)
    b = layers.GlobalAveragePooling2D(name="hybrid_alex_gap")(b)

    merged = layers.Concatenate(name="hybrid_concat")([a, b])
    gate = layers.Dense(merged.shape[-1], activation="sigmoid", name="hybrid_gate")(merged)
    fused = layers.Multiply(name="hybrid_gated_fusion")([merged, gate])

    x = layers.Dense(512, activation="relu", name="hybrid_fc1")(fused)
    x = layers.Dropout(0.35, name="hybrid_dropout1")(x)
    x = layers.Dense(128, activation="relu", name="hybrid_fc2")(x)
    x = layers.Dropout(0.25, name="hybrid_dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="hybrid_output")(x)

    return keras.Model(inputs, outputs, name="HybridFusion_R50_AlexNet")


def load_split_dataset(split_name: str, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / split_name,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=shuffle,
    ).prefetch(tf.data.AUTOTUNE)


def collect_labels(dataset):
    labels = []
    for _, y in dataset.as_numpy_iterator():
        labels.append(np.asarray(y).reshape(-1))
    return np.concatenate(labels).astype(np.int32)


def predict_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1).astype(np.float32)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.clip(np.asarray(y_prob).astype(np.float32), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "AUC": roc_auc_score(y_true, y_prob),
        "Loss": log_loss(y_true, y_prob, labels=[0, 1]),
    }


def load_model_by_name(name):
    if name == "AlexNet":
        return build_alexnet()
    if name == "ResNet50":
        return build_resnet50()
    if name == "HybridFusion_R50_AlexNet":
        return build_hybrid_fusion_r50_alexnet()
    raise ValueError(f"Unsupported model for blend optimization: {name}")


def tune_threshold(y_true, y_prob, start=0.30, stop=0.71, step=0.01):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(start, stop, step):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def fit_stacked_meta_model(x_train, y_train):
    meta = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=SEED)
    meta.fit(x_train, y_train)
    return meta


def main():
    val_ds = load_split_dataset("val", shuffle=False)
    test_ds = load_split_dataset("test", shuffle=False)
    y_val = collect_labels(val_ds)
    y_test = collect_labels(test_ds)

    candidates = ["AlexNet", "HybridFusion_R50_AlexNet", "ResNet50"]
    missing = [m for m in candidates if not (MODEL_WEIGHTS_DIR / f"{m}.weights.h5").exists()]
    if missing:
        raise FileNotFoundError(f"Missing required weights: {missing}")

    val_probs = {}
    test_probs = {}
    base_rows = []

    for name in candidates:
        model = load_model_by_name(name)
        total_params = int(model.count_params())
        trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
        model.load_weights(str(MODEL_WEIGHTS_DIR / f"{name}.weights.h5"))
        val_probs[name] = predict_probabilities(model, val_ds)
        test_probs[name] = predict_probabilities(model, test_ds)

        base_test_metrics = compute_binary_metrics(y_test, test_probs[name], threshold=0.5)
        base_rows.append(
            {
                "Model": name,
                "Family": "Hybrid" if "Hybrid" in name else "Baseline",
                "TotalParams": total_params,
                "TrainableParams": trainable_params,
                "ValAUC": float(roc_auc_score(y_val, val_probs[name])),
                "ValAccuracy": float(accuracy_score(y_val, (val_probs[name] >= 0.5).astype(np.int32))),
                "ValLoss": float(log_loss(y_val, np.clip(val_probs[name], 1e-7, 1 - 1e-7), labels=[0, 1])),
                "InferenceTimePerImageSec": np.nan,
                "WeightsPath": str(MODEL_WEIGHTS_DIR / f"{name}.weights.h5"),
                **base_test_metrics,
            }
        )

        keras.backend.clear_session()
        gc.collect()

    val_matrix = np.column_stack([val_probs[name] for name in candidates])
    test_matrix = np.column_stack([test_probs[name] for name in candidates])

    best_cfg = None
    grid = np.arange(0.0, 1.01, 0.05)
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < 0:
                continue
            weights = np.array([w0, w1, w2], dtype=np.float32)
            if np.isclose(weights.sum(), 0):
                continue
            weights = weights / weights.sum()

            mixed_val = (
                weights[0] * val_probs[candidates[0]]
                + weights[1] * val_probs[candidates[1]]
                + weights[2] * val_probs[candidates[2]]
            )

            val_auc = roc_auc_score(y_val, mixed_val)
            val_f1 = f1_score(y_val, (mixed_val >= 0.5).astype(int), zero_division=0)
            score = (val_auc, val_f1)
            if best_cfg is None or score > best_cfg["score"]:
                best_cfg = {"weights": weights, "score": score, "val_auc": val_auc}

    best_val_mix = (
        best_cfg["weights"][0] * val_probs[candidates[0]]
        + best_cfg["weights"][1] * val_probs[candidates[1]]
        + best_cfg["weights"][2] * val_probs[candidates[2]]
    )

    best_thr, best_thr_f1 = tune_threshold(y_val, best_val_mix)

    stacked_meta = fit_stacked_meta_model(val_matrix, y_val)
    stacked_val_prob = stacked_meta.predict_proba(val_matrix)[:, 1]
    stacked_val_auc = roc_auc_score(y_val, stacked_val_prob)
    stacked_thr, stacked_thr_f1 = tune_threshold(y_val, stacked_val_prob)
    stacked_test_prob = stacked_meta.predict_proba(test_matrix)[:, 1]
    stacked_metrics = compute_binary_metrics(y_test, stacked_test_prob, threshold=stacked_thr)

    mixed_test = (
        best_cfg["weights"][0] * test_probs[candidates[0]]
        + best_cfg["weights"][1] * test_probs[candidates[1]]
        + best_cfg["weights"][2] * test_probs[candidates[2]]
    )

    metrics = compute_binary_metrics(y_test, mixed_test, threshold=best_thr)

    if BENCHMARK_CSV.exists():
        df = pd.read_csv(BENCHMARK_CSV)
    else:
        df = pd.DataFrame(base_rows)
    blend_name = f"OptimizedBlend({'+'.join(candidates)})"
    total_params = int(df.set_index("Model")["TotalParams"].reindex(candidates).fillna(0).sum())

    row = {
        "Model": blend_name,
        "Family": "Ensemble",
        "TotalParams": total_params,
        "TrainableParams": 0,
        "ValAUC": float(best_cfg["val_auc"]),
        "ValAccuracy": np.nan,
        "ValLoss": np.nan,
        "InferenceTimePerImageSec": np.nan,
        "WeightsPath": f"weights={best_cfg['weights'].round(3).tolist()}|thr={best_thr}",
        **metrics,
    }

    stacked_name = f"StackedMeta({'+'.join(candidates)})"
    stacked_row = {
        "Model": stacked_name,
        "Family": "Ensemble",
        "TotalParams": total_params,
        "TrainableParams": 0,
        "ValAUC": float(stacked_val_auc),
        "ValAccuracy": np.nan,
        "ValLoss": np.nan,
        "InferenceTimePerImageSec": np.nan,
        "WeightsPath": f"meta=logreg|thr={stacked_thr}",
        **stacked_metrics,
    }

    df = df[df["Model"] != blend_name]
    df = df[df["Model"] != stacked_name]
    df = pd.concat([df, pd.DataFrame([row, stacked_row])], ignore_index=True)
    df = df.sort_values(["AUC", "ValAUC", "Accuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(BENCHMARK_CSV, index=False)

    print("Best weights:", {candidates[i]: float(best_cfg["weights"][i]) for i in range(3)})
    print("Best threshold:", best_thr)
    print("Stacked val AUC:", stacked_val_auc)
    print("Stacked threshold:", stacked_thr)
    print("Updated:", BENCHMARK_CSV)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
