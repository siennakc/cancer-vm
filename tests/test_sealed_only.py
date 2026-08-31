"""Sealed test set accounting (T-1.3). Gate tests moved to Onco-Harness."""

import numpy as np
import pytest

from oncoscope.eval.sealed import QueryBudgetExhausted, SealedTestSet



def _sealed(tmp_path, budget=3):
    return SealedTestSet(
        manifest_path=tmp_path / "manifest.json",
        access_log_path=tmp_path / "access.jsonl",
        query_budget=budget,
    )


def test_sealed_set_scores_and_accounts(tmp_path):
    s = _sealed(tmp_path)
    ids = [f"c{i}" for i in range(20)]
    s.seal(ids)
    y = np.array([0] * 15 + [1] * 5)
    scores = np.concatenate([np.linspace(0, 0.4, 15), np.linspace(0.6, 1, 5)])
    out = s.score(ids, y, scores, caller="test")
    assert out["auroc"] == 1.0
    assert out["queries_remaining"] == 2.0


def test_sealed_set_rejects_wrong_membership(tmp_path):
    s = _sealed(tmp_path)
    s.seal([f"c{i}" for i in range(10)])
    with pytest.raises(ValueError):
        s.score([f"c{i}" for i in range(9)], np.zeros(9), np.zeros(9), caller="test")


def test_query_budget_exhausts(tmp_path):
    s = _sealed(tmp_path, budget=2)
    ids = [f"c{i}" for i in range(10)]
    s.seal(ids)
    y = np.array([0] * 8 + [1] * 2)
    scores = np.linspace(0, 1, 10)
    s.score(ids, y, scores, caller="a")
    s.score(ids, y, scores, caller="b")
    with pytest.raises(QueryBudgetExhausted):
        s.score(ids, y, scores, caller="c")


def _cohort(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    # A strong model, calibrated by construction: p = sigmoid(4x), y ~ Bernoulli(p).
    # The gate's ECE check passes for an honest model and fails for a distorted one.
    x = rng.normal(0, 1, n)
    good = 1.0 / (1.0 + np.exp(-4.0 * x))
    y = (rng.random(n) < good).astype(int)
    pids = [f"P{i//2}" for i in range(n)]
    sites = ["site_a" if i % 2 else "site_b" for i in range(n)]
    return y, good, pids, sites


