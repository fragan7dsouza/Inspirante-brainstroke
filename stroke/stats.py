"""Metrics, bootstrap CIs and paired tests (DeLong, McNemar, Brier t-test) with Holm correction."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

CI_KEYS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]


def metrics(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return dict(Accuracy=(tp + tn) / len(y), Precision=tp / (tp + fp) if tp + fp else np.nan,
                Recall=tp / (tp + fn) if tp + fn else np.nan, Specificity=tn / (tn + fp) if tn + fp else np.nan,
                F1=2 * tp / (2 * tp + fp + fn), AUC=roc_auc_score(y, p),
                LogLoss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), Brier=float(np.mean((p - y) ** 2)),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def boot_ci(y, p, B=2000, seed=42, groups=None):
    """Percentile bootstrap CIs. With `groups` (e.g. patient IDs), whole groups are resampled."""
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in CI_KEYS}
    if groups is not None:
        members = list(pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices.values())
    for _ in range(B):
        if groups is None:
            i = rng.integers(0, len(y), len(y))
        else:
            i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[i])) < 2:
            continue
        m = metrics(y[i], p[i])
        for k in CI_KEYS:
            acc[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in acc.items()}


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    T = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[J] = T
    return out


def delong(y, p1, p2):
    """Fast DeLong (Sun & Xu 2014). Returns (auc1, auc2), z, two-sided p."""
    X = np.vstack([p1[y == 1], p2[y == 1]])
    Y = np.vstack([p1[y == 0], p2[y == 0]])
    m, n = X.shape[1], Y.shape[1]
    tx = np.array([_midrank(r) for r in X])
    ty = np.array([_midrank(r) for r in Y])
    tz = np.array([_midrank(np.concatenate([X[k], Y[k]])) for k in range(2)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    S = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1 - (tz[:, m:] - ty) / m) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return aucs, z, 2 * stats.norm.sf(abs(z))


def mcnemar(y, p1, p2):
    c1, c2 = (p1 >= .5) == y, (p2 >= .5) == y
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def compare(y, P, ref):
    """Paired tests of model `ref` against every other model in P, Holm-corrected per test family."""
    rows = []
    for n in P:
        if n == ref:
            continue
        aucs, z, p_d = delong(y, P[ref], P[n])
        b, c, p_m = mcnemar(y, P[ref], P[n])
        se1, se2 = (P[ref] - y) ** 2, (P[n] - y) ** 2
        t, p_t = stats.ttest_rel(se1, se2)
        rows.append(dict(model=n, auc_ref=aucs[0], auc_model=aucs[1], dauc=aucs[0] - aucs[1], z=z, p_delong=p_d,
                         mcnemar_b=b, mcnemar_c=c, p_mcnemar=p_m, brier_ref=se1.mean(), brier_model=se2.mean(), t=t, p_brier=p_t))
    for key in ["p_delong", "p_mcnemar", "p_brier"]:
        for r, a in zip(rows, holm([r[key] for r in rows])):
            r[key + "_holm"] = float(a)
    return rows
