"""splits_v2: quarantine the official CBIS-DDSM test patients (public benchmark).

Every patient appearing in the official mass/calc test CSVs is assigned to a
reserved ``public_bench`` split that no fitting stage may touch; the remainder
is re-split with the standard fractions. This exists because splits_v1 ignored
the official division, so v2's encoder trained on 202 official-test patients —
fine for our internal sealed test, disqualifying for the public benchmark.
"""
import csv, sys, collections
sys.path.insert(0, "src")
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import make_splits, save_manifest

official = set()
for f in ("mass_case_description_test_set.csv", "calc_case_description_test_set.csv"):
    for row in csv.DictReader(open(f"data/metadata/{f}")):
        official.add(row["patient_id"].strip())

cases = read_case_table("data/processed/cases_v1.jsonl")
patients = {f"{c.site}/{c.patient_id}": c.site for c in cases}
bench = {p for p in patients if p.startswith("ddsm/") and p.split("/", 1)[1] in official}
rest = {p: s for p, s in patients.items() if p not in bench}

m = make_splits(rest, {"train": 0.60, "calibration": 0.10, "threshold": 0.10,
                       "slice_discovery": 0.05, "test": 0.15},
                seed=20260831, version="v2")
assignment = dict(m.assignment)
assignment.update({p: "public_bench" for p in bench})
import hashlib, json
payload = json.dumps({"version": "v2", "assignment": assignment}, sort_keys=True)
from oncoscope.data.splits import SplitManifest
manifest = SplitManifest(version="v2", assignment=assignment,
                         sha256=hashlib.sha256(payload.encode()).hexdigest())
save_manifest(manifest, "data/processed/splits_v2.json")
print(f"splits_v2: {len(bench)} public_bench patients quarantined;",
      dict(collections.Counter(assignment.values())))
