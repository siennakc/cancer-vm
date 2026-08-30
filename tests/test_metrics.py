"""Metric validation against known cases (T-1.4)."""

import numpy as np

from oncoscope.eval.metrics import (
    auroc,
    clustered_bootstrap_ci,
    ece_adaptive,
    partial_auc,
    sensitivity_at_specificity,
)


def test_auroc_perfect_and_reversed():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0


def test_auroc_ties_give_half_credit():
    y = np.array([0, 1])
    assert auroc(y, np.array([0.5, 0.5])) == 0.5


def test_sensitivity_at_specificity_perfect_separation():
    y = np.array([0] * 50 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 10)])
    assert sensitivity_at_specificity(y, scores, 0.96) == 1.0


def test_partial_auc_bounds():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.concatenate([np.zeros(50), np.ones(50)])
    assert partial_auc(y, perfect, 0.9) > 0.99
    random_scores = np.random.default_rng(0).random(100)
    assert 0.0 <= partial_auc(y, random_scores, 0.9) <= 1.0


def test_ece_perfectly_calibrated_extremes():
    y = np.array([0] * 100 + [1] * 100)
    probs = np.array([0.0] * 100 + [1.0] * 100)
    assert ece_adaptive(y, probs) < 1e-9


def test_clustered_bootstrap_keeps_patients_together():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    pids = ["A", "A", "B", "B", "C", "C"]
    point, lo, hi = clustered_bootstrap_ci(y, scores, pids, auroc, iterations=200)
    assert lo <= point <= hi
    assert point == 1.0
