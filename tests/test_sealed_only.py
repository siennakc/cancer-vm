"""Sealed test set accounting + split provenance (T-1.3). Gate tests moved to Onco-Harness."""

import json
from pathlib import Path

import numpy as np
import pytest

from oncoscope.data.splits import SplitManifest, make_splits, save_manifest
from oncoscope.eval.sealed import (
    QueryBudgetExhausted,
    SealedProvenanceError,
    SealedTestSet,
)

FRACTIONS = {
    "train": 0.6,
    "calibration": 0.1,
    "threshold": 0.1,
    "slice_discovery": 0.05,
    "test": 0.15,
}


def _world(tmp_path, budget=3):
    """A sealed set + case table + fit manifest whose test patients ARE the sealed ones.

    Split manifests key patients as ``site/patient_id`` (the convention every
    real script uses); the case table carries the two fields separately.
    """
    patients = {f"site_a/P{i:03d}": "site_a" for i in range(40)}
    manifest = make_splits(patients, FRACTIONS)
    manifest_path = tmp_path / "splits.json"
    save_manifest(manifest, manifest_path)

    case_table = tmp_path / "cases.jsonl"
    with case_table.open("w") as fh:
        for key in patients:
            pid = key.split("/", 1)[1]
            fh.write(json.dumps({"case_id": f"c-{pid}", "site": "site_a",
                                 "patient_id": pid}) + "\n")

    sealed_ids = sorted(
        f"c-{p.split('/', 1)[1]}"
        for p, s in manifest.assignment.items() if s == "test"
    )
    s = SealedTestSet(
        manifest_path=tmp_path / "manifest.json",
        access_log_path=tmp_path / "access.jsonl",
        query_budget=budget,
        case_table_path=case_table,
    )
    s.seal(sealed_ids)
    return s, sealed_ids, manifest, manifest_path, tmp_path


def _scores(ids):
    y = np.array([i % 3 == 0 for i in range(len(ids))], dtype=float)
    return y, np.clip(y * 0.8 + 0.1, 0, 1)


def test_sealed_set_scores_and_accounts(tmp_path):
    s, ids, _, manifest_path, _ = _world(tmp_path)
    y, scores = _scores(ids)
    out = s.score(ids, y, scores, caller="test", fit_manifest=manifest_path)
    assert out["auroc"] == 1.0
    assert out["queries_remaining"] == 2.0


def test_sealed_set_rejects_wrong_membership(tmp_path):
    s, ids, _, manifest_path, _ = _world(tmp_path)
    y, scores = _scores(ids[:-1])
    with pytest.raises(ValueError):
        s.score(ids[:-1], y, scores, caller="test", fit_manifest=manifest_path)


def test_query_budget_exhausts(tmp_path):
    s, ids, _, manifest_path, _ = _world(tmp_path, budget=2)
    y, scores = _scores(ids)
    s.score(ids, y, scores, caller="a", fit_manifest=manifest_path)
    s.score(ids, y, scores, caller="b", fit_manifest=manifest_path)
    with pytest.raises(QueryBudgetExhausted):
        s.score(ids, y, scores, caller="c", fit_manifest=manifest_path)


# --- split provenance ------------------------------------------------------
#
# Membership hashing cannot see a re-split: after splits_v2's fresh seed, 381
# of the 499 v1-sealed patients landed in v2 fitting splits, so a v3/v4 model
# could pass every membership check while having trained on 76% of the sealed
# set. score() therefore demands the candidate's fit manifest (checked), or an
# explicit external=True for cohorts outside the internal split universe.

def test_scoring_without_provenance_is_refused(tmp_path):
    s, ids, _, _, _ = _world(tmp_path)
    y, scores = _scores(ids)
    with pytest.raises(SealedProvenanceError):
        s.score(ids, y, scores, caller="test")


def test_external_benchmark_needs_no_manifest(tmp_path):
    s, ids, _, _, _ = _world(tmp_path)
    y, scores = _scores(ids)
    out = s.score(ids, y, scores, caller="bench:mias", external=True)
    assert out["auroc"] == 1.0


def test_contaminating_resplit_is_refused(tmp_path):
    s, ids, manifest, _, root = _world(tmp_path)
    # Re-split under a new seed: sealed patients scatter into fitting splits.
    patients = {p: "site_a" for p in manifest.assignment}
    resplit = make_splits(patients, FRACTIONS, seed=999, version="v2")
    resplit_path = root / "splits_resplit.json"
    save_manifest(resplit, resplit_path)
    sealed_patients = {p for p, sp in manifest.assignment.items() if sp == "test"}
    assert any(
        resplit.assignment[p] not in ("test",) for p in sealed_patients
    ), "fixture should scatter sealed patients"

    y, scores = _scores(ids)
    with pytest.raises(SealedProvenanceError, match="fitting split"):
        s.score(ids, y, scores, caller="test", fit_manifest=resplit_path)


def test_unmapped_sealed_case_fails_closed(tmp_path):
    s, ids, _, manifest_path, root = _world(tmp_path)
    # Corrupt the case table: drop the row of a SEALED case specifically.
    lines = [l for l in Path(s.case_table_path).read_text().splitlines()
             if json.loads(l)["case_id"] != ids[0]]
    Path(s.case_table_path).write_text("\n".join(lines) + "\n")
    y, scores = _scores(ids)
    with pytest.raises(SealedProvenanceError, match="missing from the case table"):
        s.score(ids, y, scores, caller="test", fit_manifest=manifest_path)


# --- integration against the real committed artifacts ----------------------

REAL = {
    "sealed": Path("data/processed/sealed_test_v1.json"),
    "log": Path("data/processed/sealed_access_log.jsonl"),
    "cases": Path("data/processed/cases_v1.jsonl"),
    "v1": Path("data/processed/splits_v1.json"),
    "v2": Path("data/processed/splits_v2.json"),
}


@pytest.mark.skipif(not all(p.exists() for p in REAL.values()),
                    reason="committed data files not present")
def test_real_sealed_set_provenance_verdicts():
    """splits_v1 models may score the sealed set; splits_v2 models may not."""
    s = SealedTestSet(REAL["sealed"], REAL["log"], case_table_path=REAL["cases"])
    s.verify_provenance(REAL["v1"])  # v1/v2-era models: clean by construction
    with pytest.raises(SealedProvenanceError, match=r"of \d+ sealed patients"):
        s.verify_provenance(REAL["v2"])  # v3/v4 are disqualified here
