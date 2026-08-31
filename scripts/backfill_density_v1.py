"""Backfill CBIS calcification density bands into cases_v1.jsonl.

``build_cbis_cases`` read only the mass CSVs' ``breast_density`` spelling, so
every calcification case was written with ``density_band: null`` — silently
reducing every density-stratified result to a mass-only analysis. The parser is
fixed in ``mammography.py``; this repairs the already-committed case table
without a full rebuild, which would need the 126 GB of DICOMs.

The join is the same one the builder uses (series UID out of the CSV's image
file path -> ``ddsm-<series_uid>`` case id), so this is deterministic and
carries no new assumptions. Labels, ids, and every other field are untouched.

Idempotent: re-running changes nothing.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from oncoscope.data.mammography import (  # noqa: E402
    _read_density_band,
    _series_uid_from_cbis_path,
)

CASES = Path("data/processed/cases_v1.jsonl")
META = Path("data/metadata")
CSVS = (
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
)


def main() -> int:
    # case_id -> density band, first non-null wins (rows repeat per abnormality)
    bands: dict[str, str] = {}
    for name in CSVS:
        with (META / name).open(newline="") as fh:
            for row in csv.DictReader(fh):
                uid = _series_uid_from_cbis_path(row.get("image file path", ""))
                if not uid:
                    continue
                band = _read_density_band(row)
                if band:
                    bands.setdefault(f"ddsm-{uid}", band)

    rows = [json.loads(line) for line in CASES.open()]
    changed = conflicts = 0
    for row in rows:
        band = bands.get(row["case_id"])
        if band is None:
            continue
        if row.get("density_band") in (None, ""):
            row["density_band"] = band
            changed += 1
        elif row["density_band"] != band:
            conflicts += 1

    if conflicts:
        print(f"REFUSING: {conflicts} cases disagree with the CSV band", file=sys.stderr)
        return 1

    if changed:
        with CASES.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    filled = sum(1 for r in rows if r["site"] == "ddsm" and r.get("density_band"))
    total = sum(1 for r in rows if r["site"] == "ddsm")
    print(f"backfilled {changed} cases; ddsm density coverage {filled}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
