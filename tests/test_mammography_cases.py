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


# --- density column spelling (regression) ---------------------------------
#
# CBIS-DDSM spells the density column two different ways in its own CSVs:
# ``breast_density`` in the mass files, ``breast density`` in the calc files.
# Reading only the first silently nulls the band for all 1,501 calcification
# cases, which quietly reduces every density-stratified result to a mass-only
# analysis. These tests pin both spellings, and pin the real shipped headers so
# a future CSV revision cannot reintroduce the bug unnoticed.

CALC_CSV_HEADER = (
    "patient_id,breast density,left or right breast,image view,abnormality id,"
    "abnormality type,calc type,calc distribution,assessment,pathology,subtlety,"
    "image file path,cropped image file path,ROI mask file path\n"
)


def test_reads_both_density_spellings():
    from oncoscope.data.mammography import _read_density_band

    assert _read_density_band({"breast_density": "3"}) == "c"
    assert _read_density_band({"breast density": "3"}) == "c"
    assert _read_density_band({"breast density": "1"}) == "a"
    # Absent, blank, and out-of-range (BI-RADS density is 1-4) stay None
    # rather than being guessed.
    assert _read_density_band({}) is None
    assert _read_density_band({"breast density": ""}) is None
    assert _read_density_band({"breast density": "0"}) is None
    assert _read_density_band({"breast_density": "junk"}) is None


def test_calc_cases_get_a_density_band(tmp_path):
    """A calcification CSV must yield a band, not a silent None."""
    meta, raw = tmp_path / "metadata", tmp_path / "raw"
    meta.mkdir()
    uid = "uid-calc-1"
    tcia_pid = "Calc-Training_P_00010_LEFT_CC"
    path = f"{tcia_pid}/1.2.3.study/{uid}/000000.dcm"
    (meta / "calc_case_description_train_set.csv").write_text(
        CALC_CSV_HEADER
        + f"P_00010,3,LEFT,CC,1,calcification,PLEOMORPHIC,CLUSTERED,4,MALIGNANT,4,"
        f"{path},{path},{path}\n"
    )
    series = raw / "CBIS" / uid
    series.mkdir(parents=True)
    (series / "000000.dcm").write_bytes(b"stub")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"series_uid": uid, "relpath": f"CBIS/{uid}"}) + "\n")

    cases = build_cbis_cases(meta, manifest, raw)
    assert len(cases) == 1
    assert cases[0].abnormality == "calcification"
    assert cases[0].density_band == "c", "calc density band was dropped"


def test_shipped_csv_headers_still_carry_a_known_density_spelling():
    """Guards against an upstream CSV revision renaming the column again."""
    import csv
    from pathlib import Path

    from oncoscope.data.mammography import _DENSITY_KEYS

    meta = Path("data/metadata")
    csvs = sorted(meta.glob("*_case_description_*.csv"))
    if not csvs:
        pytest.skip("label CSVs not present")
    for path in csvs:
        with path.open(newline="") as fh:
            fields = next(csv.reader(fh))
        assert any(k in fields for k in _DENSITY_KEYS), f"{path.name}: no density column"


# --- any-malignant labels (regression) ------------------------------------
#
# The CSVs carry one row per ABNORMALITY. First-wins dedup labeled an image
# benign whenever its first listed finding was benign, even with a
# biopsy-proven malignant row later — 11 real images. Image-level gold is
# any-malignant.

def test_multi_abnormality_image_is_any_malignant(tmp_path):
    meta, raw = tmp_path / "metadata", tmp_path / "raw"
    meta.mkdir()
    uid = "uid-multi-1"
    tcia_pid = "Mass-Training_P_00020_LEFT_CC"
    path = f"{tcia_pid}/1.2.3.study/{uid}/000000.dcm"
    # benign row FIRST, malignant row second — first-wins would say benign
    (meta / "mass_case_description_train_set.csv").write_text(
        CSV_HEADER
        + f"P_00020,3,LEFT,CC,1,mass,IRREGULAR,SPICULATED,4,BENIGN,4,{path},{path},{path}\n"
        + f"P_00020,3,LEFT,CC,2,mass,IRREGULAR,SPICULATED,5,MALIGNANT,4,{path},{path},{path}\n"
    )
    series = raw / "CBIS" / uid
    series.mkdir(parents=True)
    (series / "000000.dcm").write_bytes(b"stub")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"series_uid": uid, "relpath": f"CBIS/{uid}"}) + "\n")

    cases = build_cbis_cases(meta, manifest, raw)
    assert len(cases) == 1, "two abnormality rows must stay one image case"
    assert cases[0].label == 1, "any-malignant: a malignant row anywhere wins"


def test_all_benign_rows_stay_benign(tmp_path):
    meta, raw = tmp_path / "metadata", tmp_path / "raw"
    meta.mkdir()
    uid = "uid-multi-2"
    tcia_pid = "Mass-Training_P_00021_LEFT_CC"
    path = f"{tcia_pid}/1.2.3.study/{uid}/000000.dcm"
    (meta / "mass_case_description_train_set.csv").write_text(
        CSV_HEADER
        + f"P_00021,3,LEFT,CC,1,mass,OVAL,CIRCUMSCRIBED,2,BENIGN,4,{path},{path},{path}\n"
        + f"P_00021,3,LEFT,CC,2,mass,OVAL,CIRCUMSCRIBED,2,BENIGN_WITHOUT_CALLBACK,4,{path},{path},{path}\n"
    )
    series = raw / "CBIS" / uid
    series.mkdir(parents=True)
    (series / "000000.dcm").write_bytes(b"stub")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"series_uid": uid, "relpath": f"CBIS/{uid}"}) + "\n")

    cases = build_cbis_cases(meta, manifest, raw)
    assert len(cases) == 1
    assert cases[0].label == 0
