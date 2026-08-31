"""Finalize v2: import the fine-tuned network's own fc as the calibrated head.

The 5-seed refit on L2-normed embeddings lost the magnitude signal the trained
fc uses (cal AUROC 0.708 vs the network's 0.807). The fc IS a LogisticHead;
import it, prior-correct from the balanced sampler's 0.5, temperature-scale on
calibration, threshold on the threshold split, one sealed query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import torch

from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci, ece_adaptive,
                                    sensitivity_at_specificity)
from oncoscope.eval.sealed import SealedTestSet
from oncoscope.models.head import LogisticHead

EMB = Path("data/embeddings/resnet50_ft_v2_448_raw")
RUN = Path("runs/finetune_v2b_head")
RUN.mkdir(parents=True, exist_ok=True)

cases = read_case_table("data/processed/cases_v1.jsonl")
splits = load_manifest("data/processed/splits_v1.json")
data = {}
for c in cases:
    p = EMB / f"{c.case_id}.npy"
    s = splits.split_of(f"{c.site}/{c.patient_id}")
    d = data.setdefault(s, {"X": [], "y": [], "pid": [], "site": [], "cid": []})
    d["X"].append(np.load(p)); d["y"].append(c.label)
    d["pid"].append(f"{c.site}/{c.patient_id}"); d["site"].append(c.site)
    d["cid"].append(c.case_id)
for d in data.values():
    d["X"] = np.stack(d["X"]); d["y"] = np.array(d["y"], float); d["site"] = np.array(d["site"])

ck = torch.load("runs/finetune_v2/best_model.pt", map_location="cpu", weights_only=False)
head = LogisticHead(weights=ck["model"]["fc.weight"].numpy().ravel().astype(np.float64),
                    bias=float(ck["model"]["fc.bias"].numpy()[0]))
cal, thr, sd, te = data["calibration"], data["threshold"], data["slice_discovery"], data["test"]

shift = head.apply_prior_correction(0.5, float(cal["y"].mean()))
temp = head.calibrate_temperature(cal["X"], cal["y"])
print(f"cal_auroc={auroc(cal['y'], head.predict_proba(cal['X'])):.4f} "
      f"shift={shift:.4f} temp={temp:.4f}")

p_thr = head.predict_proba(thr["X"])
neg = np.sort(p_thr[thr["y"] == 0])
k = min(max(int(np.ceil(0.96 * len(neg))), 1), len(neg))
tau = float(neg[k - 1])
op = {"threshold": tau, "target_specificity": 0.96,
      "threshold_split_sens": float((p_thr[thr["y"] == 1] > tau).mean()),
      "threshold_split_spec": float((p_thr[thr["y"] == 0] <= tau).mean())}
print(f"tau={tau:.4f} sens={op['threshold_split_sens']:.3f} spec={op['threshold_split_spec']:.3f}")


def block(name, y, scores, pids):
    a = clustered_bootstrap_ci(y, scores, pids, auroc, iterations=1000)
    s = clustered_bootstrap_ci(y, scores, pids, sensitivity_at_specificity, iterations=1000)
    out = {"n": int(len(y)), "prevalence": float(y.mean()), "auroc": a[0],
           "auroc_ci": [a[1], a[2]], "sens_at_spec96": s[0],
           "sens_at_spec96_ci": [s[1], s[2]], "ece_adaptive": float(ece_adaptive(y, scores))}
    print(f"{name:22} n={out['n']:4} auroc={out['auroc']:.4f} [{a[1]:.4f},{a[2]:.4f}] "
          f"sens@96={out['sens_at_spec96']:.3f} ece={out['ece_adaptive']:.4f}")
    return out


metrics = {"threshold_split": block("threshold", thr["y"], p_thr, thr["pid"])}
p_sd = head.predict_proba(sd["X"])
metrics["slice_discovery"] = block("slice_discovery", sd["y"], p_sd, sd["pid"])
for site in ("ddsm", "cmmd"):
    m = sd["site"] == site
    metrics[f"slice_discovery/{site}"] = block(
        f"  {site}", sd["y"][m], p_sd[m],
        [p for p, s2 in zip(sd["pid"], sd["site"]) if s2 == site])

sealed = SealedTestSet("data/processed/sealed_test_v1.json",
                       "data/processed/sealed_access_log.jsonl", query_budget=50)
order = np.argsort(np.array(te["cid"]))
sm = sealed.score(sorted(te["cid"]), te["y"][order],
                  head.predict_proba(te["X"])[order],
                  caller="finetune_v2b_fc_import_fixedstats",
                  fit_manifest="data/processed/splits_v1.json")
print("SEALED TEST:", json.dumps(sm))
metrics["sealed_test"] = sm

head.save(RUN / "head.json")
(RUN / "operating_point.json").write_text(json.dumps(op, indent=1))
(RUN / "metrics.json").write_text(json.dumps({
    "encoder": "resnet50_ft_v2_448_raw", "head": "imported network fc",
    "prior_logit_shift": shift, "temperature": temp,
    "splits_sha256": splits.sha256, "metrics": metrics}, indent=1))
print(f"saved {RUN}/")
