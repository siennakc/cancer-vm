"""Case-table construction: the label mapping and the patient-grouping trap."""

from __future__ import annotations

import json

import pytest

from oncoscope.data.mammography import (
    MammoCase,
    _series_uid_from_cbis_path,
    build_cbis_cases,
    read_case_table,
    write_case_table,
)
from oncoscope.data.splits import make_splits

CSV_HEADER = (
    "patient_id,breast_density,left or right breast,image view,abnormality id,"
    "abnormality type,mass shape,mass margins,assessment,pathology,subtlety,"
    "image file path,cropped image file path,ROI mask file path\n"
)


def _row(patient, lat, view, pathology, series_uid, density=3):
    tcia_pid = f"Mass-Training_{patient}_{lat}_{view}"
    path = f"{tcia_pid}/1.2.3.study/{series_uid}/000000.dcm"
    return (f"{patient},{density},{lat},{view},1,mass,IRREGULAR,SPICULATED,4,"
            f"{pathology},4,{path},{path},{path}\n")


@pytest.fixture()
def fake_dataset(tmp_path):
    """Four views of two patients, mirroring the real CBIS layout."""
    meta, raw = tmp_path / "metadata", tmp_path / "raw"
    meta.mkdir()
    rows = [
        ("P_00001", "LEFT", "CC", "MALIGNANT", "uid-1"),
        ("P_00001", "LEFT", "MLO", "MALIGNANT", "uid-2"),
        ("P_00001", "RIGHT", "CC", "BENIGN", "uid-3"),
        ("P_00002", "RIGHT", "MLO", "BENIGN_WITHOUT_CALLBACK", "uid-4"),
    ]
    (meta / "mass_case_description_train_set.csv").write_text(
        CSV_HEADER + "".join(_row(*r) for r in rows)
    )
    manifest = meta / "manifest.jsonl"
    with manifest.open("w") as fh:
        for *_, uid in rows:
            relpath = f"CBIS-DDSM/{uid}"
            series_dir = raw / relpath
            series_dir.mkdir(parents=True)
            (series_dir / "000000.dcm").write_bytes(b"not-a-real-dicom")
            fh.write(json.dumps({"series_uid": uid, "collection": "CBIS-DDSM",
                                 "relpath": relpath, "sha256": "x"}) + "\n")
    return meta, manifest, raw


def test_series_uid_parsed_from_cbis_path():
    path = "Mass-Training_P_00001_LEFT_CC/1.2.3.study/1.2.3.series/000000.dcm"
    assert _series_uid_from_cbis_path(path) == "1.2.3.series"
    assert _series_uid_from_cbis_path("too/short") is None


def test_patient_id_is_the_woman_not_the_view(fake_dataset):
    """The A9 trap: TCIA's PatientID is per-view, so grouping on it leaks.

    Three of these four images belong to one woman. If the patient key were
    taken from the TCIA PatientID (``Mass-Training_P_00001_LEFT_CC``) they would
    look like three independent patients and could land in different splits.
    """
    meta, manifest, raw = fake_dataset
    cases = build_cbis_cases(meta, manifest, raw)

    assert len(cases) == 4
    assert {c.patient_id for c in cases} == {"P_00001", "P_00002"}
    assert sum(c.patient_id == "P_00001" for c in cases) == 3
    assert all("_LEFT_" not in c.patient_id for c in cases)


def test_benign_without_callback_is_a_negative(fake_dataset):
    """Folding BENIGN_WITHOUT_CALLBACK into the positives would inflate prevalence."""
    meta, manifest, raw = fake_dataset
    by_uid = {c.case_id.removeprefix("ddsm-"): c for c in build_cbis_cases(meta, manifest, raw)}

    assert by_uid["uid-1"].label == 1
    assert by_uid["uid-3"].label == 0
    assert by_uid["uid-4"].label == 0


def test_grouped_splits_keep_a_patients_views_together(fake_dataset):
    """End-to-end: case table -> split manifest -> no patient spans two splits."""
    meta, manifest, raw = fake_dataset
    cases = build_cbis_cases(meta, manifest, raw)
    patients = {c.patient_id: c.site for c in cases}

    splits = make_splits(patients, {"train": 0.5, "test": 0.5}, seed=0)
    per_patient = {}
    for case in cases:
        per_patient.setdefault(case.patient_id, set()).add(splits.split_of(case.patient_id))

    assert all(len(s) == 1 for s in per_patient.values())


def test_case_table_round_trips(tmp_path):
    case = MammoCase(
        case_id="ddsm-uid-1", patient_id="P_00001", site="ddsm",
        dicom_path="CBIS-DDSM/uid-1/000000.dcm", label=1, laterality="L",
        view="CC", abnormality="mass", density_band="c", age_band=None,
        source_split="train",
    )
    path = write_case_table([case], tmp_path / "cases.jsonl")
    assert read_case_table(path) == [case]


def _mc(case_id, patient, path, label, site="cmmd"):
    return MammoCase(case_id=case_id, patient_id=patient, site=site,
                     dicom_path=path, label=label, laterality="L", view=None,
                     abnormality="mass", density_band=None, age_band=None,
                     source_split=None)


def test_content_audit_twins_and_conflicts(tmp_path):
    """Byte-identical files across patients: merge if labels agree, drop if not."""
    from oncoscope.data.mammography import content_audit

    raw = tmp_path
    (raw / "a.dcm").write_bytes(b"IMAGE-ALPHA")
    (raw / "a2.dcm").write_bytes(b"IMAGE-ALPHA")      # same bytes, other patient
    (raw / "b.dcm").write_bytes(b"IMAGE-BETA")
    (raw / "b2.dcm").write_bytes(b"IMAGE-BETA")       # same bytes, conflicting label
    (raw / "c.dcm").write_bytes(b"IMAGE-GAMMA")       # unique, innocent bystander

    cases = [
        _mc("cmmd-uid1", "D1-0002", "a.dcm", 1),      # twin pair, labels agree
        _mc("cmmd-uid2", "D1-0001", "a2.dcm", 1),
        _mc("cmmd-uid3", "D1-0010", "b.dcm", 0),      # twin pair, labels CONFLICT
        _mc("cmmd-uid4", "D2-0020", "b2.dcm", 1),
        _mc("cmmd-uid5", "D1-0099", "c.dcm", 0),
    ]
    clean, audit = content_audit(cases, raw)

    # conflicting pair: every image of both patients dropped
    kept_patients = {c.patient_id for c in clean}
    assert "D1-0010" not in kept_patients and "D2-0020" not in kept_patients

    # consistent pair: one copy kept, both ids collapsed to the alias root
    alpha = [c for c in clean if c.patient_id == "D1-0001"]
    assert len(alpha) == 1
    assert audit["merged_patient_groups"] == [["cmmd/D1-0001", "cmmd/D1-0002"]]

    # content-addressed ids: identical bytes can never be two cases again
    assert alpha[0].case_id.startswith("cmmd-") and len(alpha[0].case_id) == 21

    # bystander untouched
    assert any(c.patient_id == "D1-0099" for c in clean)
    assert audit["n_kept"] == 2 and audit["n_dropped"] == 3
