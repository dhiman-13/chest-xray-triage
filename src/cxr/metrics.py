"""Evaluation metrics used in medical-imaging papers, implemented on numpy arrays.

y: (N, K) binary ground truth, p: (N, K) predicted probabilities.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free form of 1 / (1 + e^-x)


def _per_class(fn, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        if 0 < y[:, k].sum() < len(y):  # undefined when a class has only one value
            out[k] = fn(y[:, k], p[:, k])
    return out


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return _per_class(roc_auc_score, y, p)


def per_class_auprc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return _per_class(average_precision_score, y, p)


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, specificity: float = 0.9) -> np.ndarray:
    """Recall achieved when the false-positive rate is held at 1 - specificity."""

    def fn(yk, pk):
        fpr, tpr, _ = roc_curve(yk, pk)
        return float(np.interp(1 - specificity, fpr, tpr))

    return _per_class(fn, y, p)


def thresholds_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.9) -> np.ndarray:
    """Per-class operating points chosen on validation data.

    A triage tool must not miss disease, so we fix sensitivity (e.g. 90%) and accept the
    resulting false-positive rate, rather than using a naive 0.5 cut-off.
    """
    th = np.full(y.shape[1], 0.5)
    for k in range(y.shape[1]):
        pos = np.sort(p[y[:, k] == 1, k])
        if len(pos) > 0:
            th[k] = float(pos[int(np.floor((1 - target) * len(pos)))])
    return th


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric=per_class_auroc, n_boot: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile bootstrap CI per class and for the macro average (last element).

    Reporting a point estimate without uncertainty is the most common weakness reviewers
    flag in medical-AI papers; rare classes on a 25k test set have wide intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = metric(y[idx], p[idx])
        samples.append(np.append(vals, np.nanmean(vals)))
    samples = np.array(samples)
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE over all (image, label) pairs: |accuracy - confidence| averaged over probability bins."""
    y, p = y.ravel(), p.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    y, p = y.ravel(), p.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    conf, acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p > lo) & (p <= hi)
        if m.sum() >= 20:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
    return np.array(conf), np.array(acc)


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising NLL on validation data.

    Dividing logits by T > 1 softens over-confident predictions without changing ranking,
    so AUROC is untouched while probabilities become trustworthy.
    """
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
