"""The six models. Layer order must stay as is: the saved .weights.h5 files are matched by structure."""
import keras
from keras import layers
from keras.applications import DenseNet121, InceptionV3, MobileNetV2, ResNet50
from keras.applications.densenet import preprocess_input as densenet_pre
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre

from .data import IMAGE_SIZE, WEIGHTS

NAMES = ["HybridFusion_R50_AlexNet", "AlexNet", "ResNet50", "DenseNet121", "MobileNetV2", "InceptionV3"]
SHAPE = IMAGE_SIZE + (3,)
BACKBONES = {"ResNet50": (ResNet50, resnet_pre), "DenseNet121": (DenseNet121, densenet_pre),
             "MobileNetV2": (MobileNetV2, mobilenet_pre), "InceptionV3": (InceptionV3, inception_pre)}


def augmentation():
    # Active only in training; identity at inference.
    return keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.03),
                             layers.RandomZoom(0.08), layers.RandomContrast(0.08)], name="augmentation")


def transfer(name, imagenet=False):
    ctor, pre = BACKBONES[name]
    base = ctor(include_top=False, weights="imagenet" if imagenet else None, input_shape=SHAPE)
    base.trainable = False
    inp = keras.Input(shape=SHAPE)
    x = layers.Lambda(pre)(augmentation()(inp))
    x = layers.GlobalAveragePooling2D()(base(x, training=False))
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid", dtype="float32")(x), name=name)


def alexnet():
    return keras.Sequential([
        layers.Input(shape=SHAPE), layers.Rescaling(1.0 / 255.0), augmentation(),
        layers.Conv2D(96, 11, strides=4, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        layers.Conv2D(256, 5, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        layers.Conv2D(384, 3, padding="same", activation="relu"), layers.Conv2D(384, 3, padding="same", activation="relu"),
        layers.Conv2D(256, 3, padding="same", activation="relu"), layers.MaxPooling2D(3, strides=2), layers.Flatten(),
        layers.Dense(2048, activation="relu"), layers.Dropout(0.5),
        layers.Dense(512, activation="relu"), layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid", dtype="float32")], name="AlexNet")


def hybrid(imagenet=False):
    """Branch A: ImageNet ResNet50. Branch B: shallow AlexNet-style CNN from scratch. Sigmoid gate over the concatenation."""
    rb = ResNet50(include_top=False, weights="imagenet" if imagenet else None, input_shape=SHAPE)
    rb.trainable = False
    inp = keras.Input(shape=SHAPE)
    aug = augmentation()(inp)
    a = layers.GlobalAveragePooling2D()(rb(layers.Lambda(resnet_pre)(aug), training=False))
    b = layers.Rescaling(1.0 / 255.0)(aug)
    b = layers.Conv2D(96, 11, strides=4, padding="same", activation="relu")(b)
    b = layers.MaxPooling2D(3, strides=2)(layers.BatchNormalization()(b))
    b = layers.Conv2D(256, 5, padding="same", activation="relu")(b)
    b = layers.MaxPooling2D(3, strides=2)(layers.BatchNormalization()(b))
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(b)
    b = layers.GlobalAveragePooling2D()(b)
    merged = layers.Concatenate()([a, b])
    fused = layers.Multiply()([merged, layers.Dense(merged.shape[-1], activation="sigmoid")(merged)])
    x = layers.Dropout(0.35)(layers.Dense(512, activation="relu")(fused))
    x = layers.Dropout(0.25)(layers.Dense(128, activation="relu")(x))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid", dtype="float32")(x), name="HybridFusion_R50_AlexNet")


def build(name, imagenet=False):
    """imagenet=True only for training from scratch; saved weights already contain the backbone."""
    if name == "AlexNet":
        return alexnet()
    if name == "HybridFusion_R50_AlexNet":
        return hybrid(imagenet)
    return transfer(name, imagenet)


def load(name):
    m = build(name)
    m.load_weights(str(WEIGHTS / f"{name}.weights.h5"))
    return m
