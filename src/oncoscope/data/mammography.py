"""Real mammography case tables: CBIS-DDSM + CMMD (T-1.1, T-1.2).

Two public collections, deliberately: they are different countries, different
scanners, and different label provenance, so they give the split machinery a
genuine ``site`` axis (axiom A9) and the eval a real external-validation arm
instead of a random subset of one distribution.

| site     | source    | n patients | label provenance             |
|----------|-----------|-----------:|------------------------------|
| ``ddsm`` | CBIS-DDSM |      ~1566 | biopsy-proven pathology      |
| ``cmmd`` | CMMD      |       1775 | biopsy-proven pathology      |

**The patient-id trap.** TCIA's ``PatientID`` for CBIS-DDSM is per-*view*
(``Mass-Training_P_01239_RIGHT_CC``), not per-patient. Grouping splits on it
would scatter one woman's four views across train and test — exactly the
image-level leakage axiom A9 forbids, worth 2-20+ fake AUROC points. The real
key is the ``patient_id`` column of the case-description CSVs (``P_01239``),
and that is what this module emits.

Pixels are never loaded here; a case carries a path, and
``dicom_canonical.load_canonical`` remains the one decoder.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# BENIGN_WITHOUT_CALLBACK is benign tissue that was not even recalled — it is a
# negative for a detection task. Folding it into the positive class is a common
# way to quietly inflate prevalence and wreck calibration.
_CBIS_MALIGNANT = {"MALIGNANT"}
_CBIS_BENIGN = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}

_DENSITY_BAND = {1: "a", 2: "b", 3: "c", 4: "d"}


@dataclass(frozen=True)
class MammoCase:
    """One mammogram with a biopsy-proven label. Mirrors ``PhantomCase`` minus pixels."""

    case_id: str
    patient_id: str            # TRUE patient, the grouping key for splits
    site: str                  # "ddsm" | "cmmd"
    dicom_path: str            # relative to the raw-data root
    label: int                 # 1 = malignant
    laterality: str | None     # "L" | "R"
    view: str | None           # "CC" | "MLO" | None (CMMD carries no view tag)
    abnormality: str | None    # "mass" | "calcification" | "both"
    density_band: str | None   # "a".."d" (CBIS only)
    age_band: str | None       # decade band (CMMD only)
    source_split: str | None   # CBIS ships its own train/test division

    def to_json(self) -> dict:
        return asdict(self)


def _age_band(age) -> str | None:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    lo = max(20, min(80, (a // 10) * 10))
    return f"{lo}-{lo + 9}"


def _load_manifest(manifest_path: Path | str) -> dict[str, dict]:
    """series_uid -> manifest row, successful downloads only."""
    rows: dict[str, dict] = {}
    path = Path(manifest_path)
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            rows[row["series_uid"]] = row
    return rows


def _series_uid_from_cbis_path(image_file_path: str) -> str | None:
    """CBIS CSV paths look like ``<TciaPatientID>/<StudyUID>/<SeriesUID>/000000.dcm``."""
    parts = [p for p in image_file_path.strip().split("/") if p]
    return parts[2] if len(parts) >= 3 else None


def build_cbis_cases(metadata_dir: Path | str, manifest_path: Path | str,
                     raw_root: Path | str) -> list[MammoCase]:
    """Join the four CBIS-DDSM case-description CSVs to downloaded series."""
    metadata_dir, raw_root = Path(metadata_dir), Path(raw_root)
    manifest = _load_manifest(manifest_path)
    cases: list[MammoCase] = []
    seen: set[str] = set()

    for csv_name in (
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ):
        csv_path = metadata_dir / csv_name
        if not csv_path.exists():
            continue
        source_split = "test" if "test_set" in csv_name else "train"
        abnormality = "mass" if csv_name.startswith("mass") else "calcification"

        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                pathology = (row.get("pathology") or "").strip().upper()
                if pathology in _CBIS_MALIGNANT:
                    label = 1
                elif pathology in _CBIS_BENIGN:
                    label = 0
                else:
                    continue  # unlabelled row — dropped, never guessed

                series_uid = _series_uid_from_cbis_path(row.get("image file path", ""))
                entry = manifest.get(series_uid or "")
                if entry is None:
                    continue  # not downloaded (yet)

                dcm = sorted((raw_root / entry["relpath"]).rglob("*.dcm"))
                if not dcm:
                    continue

                # One CSV row per abnormality, but the *image* is the unit here:
                # two masses in one view must not become two duplicate cases.
                case_id = f"ddsm-{series_uid}"
                if case_id in seen:
                    continue
                seen.add(case_id)

                try:
                    density = _DENSITY_BAND.get(int(row.get("breast_density") or 0))
                except ValueError:
                    density = None

                cases.append(MammoCase(
                    case_id=case_id,
                    patient_id=(row.get("patient_id") or "").strip(),
                    site="ddsm",
                    dicom_path=str(dcm[0].relative_to(raw_root)),
                    label=label,
                    laterality=(row.get("left or right breast") or "").strip()[:1] or None,
                    view=(row.get("image view") or "").strip() or None,
                    abnormality=abnormality,
                    density_band=density,
                    age_band=None,
                    source_split=source_split,
                ))
    return cases


def build_cmmd_cases(clinical_xlsx: Path | str, manifest_path: Path | str,
                     raw_root: Path | str) -> list[MammoCase]:
    """Join CMMD clinical rows to downloaded series via (patient, laterality).

    CMMD labels each *breast*, and a series may hold both breasts' images, so
    the join key is the ``ImageLaterality`` tag of each individual file.
    """
    import pandas as pd  # optional dependency, only needed to build CMMD cases
    import pydicom

    raw_root = Path(raw_root)
    manifest = _load_manifest(manifest_path)
    clinical = pd.read_excel(clinical_xlsx)

    by_breast: dict[tuple[str, str], dict] = {}
    for _, r in clinical.iterrows():
        key = (str(r["ID1"]).strip(), str(r["LeftRight"]).strip()[:1].upper())
        by_breast[key] = r.to_dict()

    cases: list[MammoCase] = []
    for entry in manifest.values():
        pid = str(entry.get("tcia_patient_id", "")).strip()
        for dcm_path in sorted((raw_root / entry["relpath"]).rglob("*.dcm")):
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            except Exception:
                continue
            lat = str(getattr(ds, "ImageLaterality", "") or
                      getattr(ds, "Laterality", "") or "").strip()[:1].upper()
            meta = by_breast.get((pid, lat))
            if meta is None:
                continue  # no clinical row for this breast — dropped, never guessed

            classification = str(meta.get("classification", "")).strip().lower()
            if classification == "malignant":
                label = 1
            elif classification == "benign":
                label = 0
            else:
                continue

            cases.append(MammoCase(
                case_id=f"cmmd-{ds.SOPInstanceUID}",
                patient_id=pid,
                site="cmmd",
                dicom_path=str(dcm_path.relative_to(raw_root)),
                label=label,
                laterality=lat or None,
                view=str(getattr(ds, "ViewPosition", "") or "").strip() or None,
                abnormality=str(meta.get("abnormality", "")).strip() or None,
                density_band=None,
                age_band=_age_band(meta.get("Age")),
                source_split=None,
            ))
    return cases


def write_case_table(cases: list[MammoCase], path: Path | str) -> Path:
    """Persist the case table as JSONL — the versioned input to splits and eval."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for case in sorted(cases, key=lambda c: c.case_id):
            fh.write(json.dumps(case.to_json()) + "\n")
    return path


def read_case_table(path: Path | str) -> list[MammoCase]:
    return [MammoCase(**json.loads(line))
            for line in Path(path).read_text().splitlines() if line.strip()]
