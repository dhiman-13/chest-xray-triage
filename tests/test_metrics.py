import numpy as np

from cxr.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    per_class_auroc,
    sensitivity_at_specificity,
    sigmoid,
    thresholds_for_sensitivity,
)

rng = np.random.default_rng(0)


def test_auroc_perfect_and_undefined():
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.5], [0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])
    auc = per_class_auroc(y, p)
    assert auc[0] == 1.0 and np.isnan(auc[1])  # all-positive column has no AUROC


def test_thresholds_hit_target_sensitivity():
    y = rng.integers(0, 2, (2000, 3))
    p = np.clip(y * 0.3 + rng.random((2000, 3)) * 0.7, 0, 1)
    th = thresholds_for_sensitivity(y, p, target=0.9)
    sens = ((p >= th) & (y == 1)).sum(0) / y.sum(0)
    assert np.all(sens >= 0.89)


def test_sensitivity_at_specificity_range():
    y = rng.integers(0, 2, (500, 2))
    s = sensitivity_at_specificity(y, rng.random((500, 2)), 0.9)
    assert np.all((s >= 0) & (s <= 1))


def test_bootstrap_ci_contains_point_estimate():
    y = rng.integers(0, 2, (800, 2))
    p = np.clip(y * 0.4 + rng.random((800, 2)) * 0.6, 0, 1)
    lo, hi = bootstrap_ci(y, p, n_boot=100)
    point = per_class_auroc(y, p)
    assert np.all(lo[:-1] <= point) and np.all(point <= hi[:-1])


def test_temperature_scaling_fixes_overconfidence():
    true_logits = rng.normal(0, 1.5, (4000, 4))
    y = (rng.random((4000, 4)) < sigmoid(true_logits)).astype(float)
    overconfident = true_logits * 3.0
    t = fit_temperature(overconfident, y)
    assert 2.4 < t < 3.6
    assert expected_calibration_error(y, sigmoid(overconfident / t)) < expected_calibration_error(y, sigmoid(overconfident))
