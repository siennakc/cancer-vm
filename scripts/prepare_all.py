"""Driver: drain retries -> case table + splits -> SEAL TEST SET -> embeddings.

Sealing happens before any embedding or training so the test membership is
hash-locked before a model exists (T-1.3, axiom A6).
"""

import subprocess
import sys
import time

sys.path.insert(0, "src")

PY = ".venv/bin/python"


def run(cmd):
    print(f"\n===== {' '.join(cmd)} =====", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {cmd}")


# 1. wait for any in-flight retry sweep, then drain to zero (max 4 passes)
while subprocess.run(["pgrep", "-f", "retry_failed.py"], capture_output=True).returncode == 0:
    print("[prepare] waiting for in-flight retry sweep...", flush=True)
    time.sleep(30)

from oncoscope.data.tcia import failed_series  # noqa: E402

for sweep in range(4):
    n = len(failed_series("data/metadata/manifest_cbis.jsonl"))
    if n == 0:
        break
    print(f"[prepare] sweep {sweep + 1}: {n} failed CBIS series", flush=True)
    run([PY, "scripts/retry_failed.py", "CBIS-DDSM",
         "data/metadata/manifest_cbis.jsonl", "full mammogram images"])

remaining = len(failed_series("data/metadata/manifest_cbis.jsonl"))
print(f"[prepare] CBIS failures after sweeps: {remaining}", flush=True)

# 2. case table + grouped splits
run([PY, "scripts/build_dataset.py"])

# 3. seal the test split NOW — before embeddings, before any head is fit
from oncoscope.data.mammography import read_case_table  # noqa: E402
from oncoscope.data.splits import load_manifest  # noqa: E402
from oncoscope.eval.sealed import SealedTestSet  # noqa: E402

cases = read_case_table("data/processed/cases_v1.jsonl")
splits = load_manifest("data/processed/splits_v1.json")
test_ids = sorted(c.case_id for c in cases
                  if splits.split_of(f"{c.site}/{c.patient_id}") == "test")
sealed = SealedTestSet("data/processed/sealed_test_v1.json",
                       "data/processed/sealed_access_log.jsonl", query_budget=50)
sha = sealed.seal(test_ids, version="v1")
print(f"[prepare] sealed test set: {len(test_ids)} cases, sha256={sha[:16]}…", flush=True)

# 4. embeddings for every case (test included — caching is not scoring)
run([PY, "scripts/cache_embeddings.py"])

print("[prepare] ALL DONE", flush=True)
