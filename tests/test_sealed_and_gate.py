"""Sealed test set accounting (T-1.3) and the promotion gate (T-3.1)."""

import numpy as np
import pytest
import yaml

from oncoscope.eval.gate import run_gate
from oncoscope.eval.sealed import QueryBudgetExhausted, SealedTestSet

RULES = yaml.safe_load(open("gates/gate_rules.yaml"))


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


def test_gate_passes_equivalent_candidate():
    y, scores, pids, sites = _cohort()
    result = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        subgroups={"site": sites}, candidate_scores_rerun=scores.copy(),
    )
    assert result.passed, result.summary()


def test_gate_fails_degraded_candidate():
    y, good, pids, sites = _cohort()
    rng = np.random.default_rng(1)
    degraded = np.clip(good + rng.normal(0, 0.35, len(good)), 0, 1)  # much noisier
    result = run_gate(RULES, y, degraded, champion_scores=good, patient_ids=pids)
    assert not result.passed


def test_gate_fails_nondeterministic_candidate():
    y, scores, pids, _ = _cohort()
    jittered = scores + 1e-3
    result = run_gate(
        RULES, y, scores, champion_scores=scores.copy(), patient_ids=pids,
        candidate_scores_rerun=jittered,
    )
    assert any(c.name == "determinism_double_run" and not c.passed for c in result.checks)
