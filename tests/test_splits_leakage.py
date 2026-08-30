"""Split grouping and leakage audit as failing tests (T-1.2, axiom A9)."""

import numpy as np
import pytest

from oncoscope.data.phantom import generate_dataset
from oncoscope.data.splits import load_manifest, make_splits, save_manifest
from oncoscope.eval.leakage import audit

FRACTIONS = {
    "train": 0.6,
    "calibration": 0.1,
    "threshold": 0.1,
    "slice_discovery": 0.05,
    "test": 0.15,
}


def test_every_patient_in_exactly_one_split():
    patients = {f"P{i:04d}": ("site_a" if i % 2 else "site_b") for i in range(200)}
    manifest = make_splits(patients, FRACTIONS)
    assert set(manifest.assignment) == set(patients)
    # deterministic given the seed
    again = make_splits(patients, FRACTIONS)
    assert again.assignment == manifest.assignment


def test_manifest_hash_roundtrip(tmp_path):
    patients = {f"P{i:04d}": "site_a" for i in range(50)}
    manifest = make_splits(patients, FRACTIONS)
    path = tmp_path / "splits.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded.assignment == manifest.assignment

    # Tampering must be detected.
    tampered = path.read_text().replace('"train"', '"test"', 1)
    path.write_text(tampered)
    with pytest.raises(ValueError):
        load_manifest(path)


def test_leakage_audit_catches_cross_split_duplicate():
    cases = generate_dataset(n_patients=12, images_per_patient=1, seed=3)
    split_of = {c.patient_id: ("train" if i % 2 else "test") for i, c in enumerate(cases)}
    images = [(c.case_id, c.patient_id, c.pixels) for c in cases]
    # Plant an exact duplicate of a train image under a test patient.
    train_case = next(c for i, c in enumerate(cases) if i % 2)
    test_case = next(c for i, c in enumerate(cases) if not i % 2)
    images.append((test_case.case_id + "_dup", test_case.patient_id, train_case.pixels.copy()))

    report = audit(split_of, images)
    assert not report.clean
    assert any(train_case.case_id in pair for pair in report.near_duplicates)


def test_clean_data_passes_audit():
    rng = np.random.default_rng(0)
    images = [
        (f"c{i}", f"P{i}", rng.random((64, 64)).astype(np.float32)) for i in range(10)
    ]
    split_of = {f"P{i}": ("train" if i < 5 else "test") for i in range(10)}
    assert audit(split_of, images).clean
