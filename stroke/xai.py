"""Grad-CAM++ for both hybrid branches, brain masking, saliency measures (OHR, AC) and ROI boxes."""
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening,
                           gaussian_filter, label)

BONE = 240
SIGMA = 1.2


class HybridCAM:
    """Grad-CAM++ on the pre-sigmoid logit (closed form, Chattopadhay et al.) for Branch A (ResNet50 conv5) and B (conv3)."""

    def __init__(self, model):
        # Keras 3 can't expose a nested model's activations in the outer graph, so the forward pass is
        # replayed with the loaded layers (augmentation and dropout are identity at inference).
        of = lambda t: [l for l in model.layers if isinstance(l, t)]
        self.resnet = next(l for l in model.layers if isinstance(l, keras.Model) and l.name.startswith("resnet"))
        (self.pre,), (self.rescale,) = of(layers.Lambda), of(layers.Rescaling)
        self.convs, self.bns, self.pools, self.gaps = of(layers.Conv2D), of(layers.BatchNormalization), \
            of(layers.MaxPooling2D), of(layers.GlobalAveragePooling2D)
        self.gate, self.fc1, self.fc2, self.out = of(layers.Dense)
        x = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
        assert np.allclose(tf.sigmoid(self._forward(x)[2]), model.predict(x, verbose=0)[:, 0], atol=1e-4), \
            "replayed forward pass does not match the model"

    def _forward(self, x):
        fa = self.resnet(self.pre(x), training=False)
        b = self.rescale(x)
        for conv, bn, pool in zip(self.convs[:2], self.bns, self.pools):
            b = pool(bn(conv(b), training=False))
        fb = self.convs[2](b)
        m = tf.concat([self.gaps[0](fa), self.gaps[1](fb)], -1)
        h = self.fc2(self.fc1(m * self.gate(m)))
        return fa, fb, (h @ self.out.kernel + self.out.bias)[:, 0]

    def __call__(self, x):
        """x: (1, 224, 224, 3) float in 0..255. Returns {"A": map, "B": map}, each 224x224 in [0, 1]."""
        x = tf.constant(x)
        with tf.GradientTape() as tape:
            fa, fb, logit = self._forward(x)
        ga, gb = tape.gradient(logit, [fa, fb])
        return {k: _resize(_gcpp(A[0], g[0]), x.shape[1:3]) for k, A, g in [("A", fa, ga), ("B", fb, gb)]}


def _gcpp(A, g):
    g2, g3 = g ** 2, g ** 3
    alpha = g2 / (2 * g2 + tf.reduce_sum(A, axis=(0, 1), keepdims=True) * g3 + 1e-7)
    alpha = tf.where(g != 0, alpha, 0.0)
    h = tf.nn.relu(tf.reduce_sum(A * tf.reduce_sum(alpha * tf.nn.relu(g), axis=(0, 1)), -1))
    m = tf.reduce_max(h)
    return tf.where(m > 0, h / m, h)


def _resize(h, size):
    return np.clip(tf.image.resize(h[..., None], size).numpy().squeeze(), 0, 1)


def largest_component(mask):
    lab, n = label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def head_mask(gray):
    nz = gray[gray > 0]
    if nz.size < 20:
        return np.ones_like(gray, bool)
    m = binary_opening(gray > np.percentile(nz, 25), structure=np.ones((3, 3), bool))
    m = binary_fill_holes(largest_component(m))
    return binary_closing(m, structure=np.ones((5, 5), bool))


def brain_mask(img):
    """Head mask minus the skull: bright components (>= 240) that reach the 8-px rim of the head.
    Bright regions that don't reach the rim, such as hemorrhages, are kept."""
    gray = img.astype(np.float32).mean(-1)
    head = head_mask(gray)
    bright = head & (gray >= BONE)
    rim = head & ~binary_erosion(head, iterations=8)
    lab, _ = label(bright, structure=np.ones((3, 3), int))
    keep = np.unique(lab[rim & bright])
    skull = np.isin(lab, keep[keep > 0])
    brain = binary_opening(head & ~binary_dilation(skull, iterations=2), structure=np.ones((3, 3), bool))
    return head, skull, binary_fill_holes(largest_component(brain))


def ohr(heat, brain):
    """Outside heat ratio: share of the map's energy outside the brain."""
    return float((heat * ~brain).sum() / (heat.sum() + 1e-8))


def ac(heat):
    """Activation concentration: Gini index of the map's pixel values (higher = more focal)."""
    v = np.sort(heat.ravel().astype(np.float32))
    return float(np.clip(1 - 2 * np.cumsum(v).sum() / ((v.sum() + 1e-8) * v.size), 0, 1))


def roi(masked, brain, max_regions=1):
    """Box around the strongest high-activation component inside the brain. Marks the saliency peak, not the lesion."""
    pos = masked[masked > 0]
    if pos.size < 20:
        return []
    hot = np.where(masked >= np.percentile(pos, 88), masked, 0.0)
    pos = hot[hot > 0]
    if hot.max() <= 1e-8 or pos.size < 20:
        return []
    lab, n = label(hot >= max(0.45 * hot.max(), np.percentile(pos, 92)), structure=np.ones((3, 3), int))
    h, w = hot.shape
    min_area, margin = max(20, int(0.0015 * h * w)), max(6, int(0.04 * min(h, w)))
    boxes = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero(lab == c)
        if len(xs) < min_area or brain[ys, xs].mean() < 0.98:
            continue
        if xs.min() <= margin or ys.min() <= margin or xs.max() >= w - margin or ys.max() >= h - margin:
            continue
        boxes.append((float(hot[ys, xs].sum()), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return [b for _, b in sorted(boxes, reverse=True)[:max_regions]]


def analyse(img, maps):
    """Masks and saliency measures for one slice. img: 224x224x3 in 0..255; maps: output of HybridCAM."""
    head, skull, brain = brain_mask(img)
    out = dict(head=head, skull=skull, brain=brain)
    for k, raw in maps.items():
        smooth = gaussian_filter(raw.astype(np.float32), SIGMA)
        masked = smooth * brain
        out[k] = dict(smooth=smooth, masked=masked, ohr=ohr(smooth, brain), ac_raw=ac(smooth), ac_masked=ac(masked),
                      roi=roi(masked, brain))
    return out
