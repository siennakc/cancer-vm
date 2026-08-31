"""Build the versioned case table and grouped split manifest from downloaded data.

Run after the fetch scripts. Idempotent, and safe to run against a partial
download — it reports coverage so a short dataset is visible rather than silent.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from oncoscope.data.mammography import (build_cbis_cases, build_cmmd_cases,
                                         content_audit, write_case_table)
from oncoscope.data.splits import make_splits, save_manifest

META, RAW, OUT = "data/metadata", "data/raw", "data/processed"

# Disjoint calibration / threshold / slice-discovery splits are required by
# T-1.3: a threshold picked on the test set is the classic invalidating bug.
FRACTIONS = {
    "train": 0.60,
    "calibration": 0.10,
    "threshold": 0.10,
    "slice_discovery": 0.05,
    "test": 0.15,
}


def main() -> None:
    cases = []
    cbis = build_cbis_cases(META, f"{META}/manifest_cbis.jsonl", RAW)
    print(f"CBIS-DDSM: {len(cbis)} cases, {len({c.patient_id for c in cbis})} patients")
    cases += cbis

    cmmd = build_cmmd_cases(f"{META}/CMMD_clinicaldata_revision.xlsx",
                            f"{META}/manifest_cmmd.jsonl", RAW)
    print(f"CMMD:      {len(cmmd)} cases, {len({c.patient_id for c in cmmd})} patients")
    cases += cmmd

    if not cases:
        raise SystemExit("no cases built — run the fetch scripts first")

    # Byte-level duplicate audit: merge consistent twins, drop conflicted ones.
    cases, audit = content_audit(cases, RAW)
    audit_path = Path(f"{OUT}/duplicates_audit_v1.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=1))
    print(f"audit: kept={audit['n_kept']} dropped={audit['n_dropped']} "
          f"merged_groups={len(audit['merged_patient_groups'])} "
          f"conflicted={len(audit['conflicted_patients_dropped'])}")

    table = write_case_table(cases, f"{OUT}/cases_v1.jsonl")
    print(f"\nwrote {table} ({len(cases)} cases)")

    by_site = collections.Counter(c.site for c in cases)
    by_label = collections.Counter((c.site, c.label) for c in cases)
    for site in sorted(by_site):
        pos, neg = by_label[(site, 1)], by_label[(site, 0)]
        print(f"  {site:5} n={by_site[site]:6}  malignant={pos:6} ({pos / (pos + neg):.1%})")

    # Patient ids are unique per site by construction, but namespacing them keeps
    # the grouping honest if a future collection reuses an id string.
    patients = {f"{c.site}/{c.patient_id}": c.site for c in cases}
    splits = make_splits(patients, FRACTIONS, version="v1")
    path = f"{OUT}/splits_v1.json"
    save_manifest(splits, path)
    print(f"\nwrote {path} — {len(patients)} patients, sha256={splits.sha256[:12]}")
    print("  " + "  ".join(f"{k}={v}" for k, v in
                           sorted(collections.Counter(splits.assignment.values()).items())))


if __name__ == "__main__":
    main()
