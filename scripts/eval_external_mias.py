"""External validation on MIAS — a public dataset, evaluated plainly.

Not a benchmark of ours: MIAS (322 UK film-screen images, official 2012
distribution) serves as out-of-distribution external validation, the way
papers report a second-site table. Image-level malignant (severity M) vs
negative (B/NORM), standard metrics, film-pair patient clustering for CIs.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from oncoscope.bench.mias import build_mias_cases, read_pgm_canonical
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci, ece_adaptive,
                                    sensitivity_at_specificity)
from oncoscope.models.encoder import FrozenEncoder
from oncoscope.models.head import LogisticHead

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=None)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--gray-stats", action="store_true")
ap.add_argument("--head", required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()

kw = {"mean": (0.449,) * 3, "std": (0.226,) * 3} if args.gray_stats else {}
enc = FrozenEncoder(tag="mias_eval", weights_path=args.weights,
                    normalize=not args.raw, **kw)
head = LogisticHead.load(args.head)

cases = build_mias_cases(Path("data/raw/MIAS"))
embs = []
for i in range(0, len(cases), 8):
    embs.append(enc.embed_batch([read_pgm_canonical(Path("data/raw/MIAS") / c.image_path)
                                 for c in cases[i:i + 8]]))
X = np.concatenate(embs)
y = np.array([c.label for c in cases], float)
p = head.predict_proba(X)
pids = [c.patient_id for c in cases]

a = clustered_bootstrap_ci(y, p, pids, auroc, iterations=2000)
s = clustered_bootstrap_ci(y, p, pids, sensitivity_at_specificity, iterations=2000)
report = {"dataset": "MIAS v1.21 (external validation, never trained on)",
          "model": args.name, "n": len(y), "prevalence": float(y.mean()),
          "auroc": round(a[0], 4), "auroc_ci95": [round(a[1], 4), round(a[2], 4)],
          "sens_at_spec96": round(s[0], 4),
          "sens_at_spec96_ci95": [round(s[1], 4), round(s[2], 4)],
          "ece_adaptive": round(float(ece_adaptive(y, p)), 4)}
out = Path(f"results/external_mias/{args.name}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
