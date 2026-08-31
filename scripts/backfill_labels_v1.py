"""Backfill any-malignant CBIS labels into cases_v1.jsonl.

``build_cbis_cases`` deduplicated one-row-per-abnormality CSVs by series with
first-wins, so an image whose FIRST listed finding was benign kept label 0
even when a later row recorded a biopsy-proven malignancy — 11 images, 4 of
them inside the public benchmark. The builder is fixed to any-malignant; this
repairs the already-committed case table from the committed CSVs alone (the
126 GB of DICOMs are not needed: the join is the same series-UID one the
builder uses).

Direction of the error was conservative (malignancies labeled benign depress
measured AUROC), but wrong gold is wrong gold. Numbers computed before this
fix are annotated as stale in their result files, not rewritten.

Idempotent: re-running changes nothing. Refuses on any disagreement it cannot
explain (a table label 1 with no malignant CSV row).
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from oncoscope.data.mammography import (  # noqa: E402
    _CBIS_BENIGN,
    _CBIS_MALIGNANT,
    _series_uid_from_cbis_path,
)

CASES = Path("data/processed/cases_v1.jsonl")
META = Path("data/metadata")
CSVS = (
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
)


def main() -> int:
    # case_id -> any-malignant label derived from EVERY row of that series
    gold: dict[str, int] = {}
    for name in CSVS:
        with (META / name).open(newline="") as fh:
            for row in csv.DictReader(fh):
                uid = _series_uid_from_cbis_path(row.get("image file path", ""))
                if not uid:
                    continue
                pathology = (row.get("pathology") or "").strip().upper()
                if pathology in _CBIS_MALIGNANT:
                    label = 1
                elif pathology in _CBIS_BENIGN:
                    label = 0
                else:
                    continue
                case_id = f"ddsm-{uid}"
                gold[case_id] = max(gold.get(case_id, 0), label)

    rows = [json.loads(line) for line in CASES.open()]
    flips: list[str] = []
    unexplained: list[str] = []
    for row in rows:
        want = gold.get(row["case_id"])
        if want is None:
            continue
        if row["label"] == want:
            continue
        if row["label"] == 0 and want == 1:
            row["label"] = 1
            flips.append(row["case_id"])
        else:
            unexplained.append(row["case_id"])  # table says 1, CSVs never do

    if unexplained:
        print(f"REFUSING: {len(unexplained)} cases labeled 1 with no malignant "
              f"CSV row: {unexplained[:5]}", file=sys.stderr)
        return 1

    if flips:
        with CASES.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        for cid in flips:
            print(f"flipped to malignant: {cid}")
    print(f"backfilled {len(flips)} labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
