#!/bin/bash
# Wait for validation -> pick aggregation on TRAIN slides -> launch full official test.
set -e
cd "${ONCOSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
while pgrep -f "infer_c16_slide.*c16_val" > /dev/null; do sleep 60; done
.venv/bin/python - <<'PY'
import json, glob
import numpy as np
recs = [json.load(open(f)) for f in glob.glob("runs/c16/scores/*.json")]
recs = [r for r in recs if r["slide"].startswith(("tumor", "normal"))]
y = np.array([1.0 if r["slide"].startswith("tumor") else 0.0 for r in recs])
print(f"[agg] {len(recs)} labeled train slides")
best, best_auc = None, -1
for key in ["max", "top5_mean", "top20_mean", "top50_mean", "top100_mean", "frac_over_0.5", "frac_over_0.9"]:
    s = np.array([r[key] for r in recs])
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    pos = y == 1
    auc = (rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    print(f"[agg] {key:15} train-slide AUC = {auc:.4f}")
    if auc > best_auc: best, best_auc = key, auc
json.dump({"aggregation": best, "train_slide_auc": best_auc, "n": len(recs)},
          open("runs/c16/aggregation_choice.json", "w"), indent=1)
print(f"[agg] chosen: {best} ({best_auc:.4f})")
PY
grep "^test_" data/raw/CAMELYON16/evaluation/reference.csv | cut -d, -f1 | sed 's/.tif$//' > runs/c16/test_list.txt
wc -l runs/c16/test_list.txt
echo "=== FULL TEST RUN ==="
.venv/bin/python scripts/bench/infer_c16_slide.py --list-file runs/c16/test_list.txt --batch 192
echo C16-TEST-DONE
