"""Quarantined heads for the public benchmark (no internal sealed queries).

Two arms, both fit strictly inside splits_v2's fitting splits:
- frozen_v1q: stock-IN1K embeddings + 5-seed balanced refit (control arm)
- finetune_v3: retrained encoder's fc imported, prior-shifted, temperatured
Threshold at spec>=0.96 chosen on the v2 threshold split for each.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import torch
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc
from oncoscope.models.head import LogisticHead

cases = read_case_table("data/processed/cases_v1.jsonl")
splits = load_manifest("data/processed/splits_v2.json")

def arrays(tag):
    d = {}
    for c in cases:
        s = splits.split_of(f"{c.site}/{c.patient_id}")
        if s == "public_bench":
            continue
        e = d.setdefault(s, {"X": [], "y": []})
        e["X"].append(np.load(f"data/embeddings/{tag}/{c.case_id}.npy"))
        e["y"].append(c.label)
    return {k: {"X": np.stack(v["X"]), "y": np.array(v["y"], float)} for k, v in d.items()}

def finish(head, d, out):
    shift = head.apply_prior_correction(0.5, float(d["calibration"]["y"].mean()))
    temp = head.calibrate_temperature(d["calibration"]["X"], d["calibration"]["y"])
    p = head.predict_proba(d["threshold"]["X"])
    neg = np.sort(p[d["threshold"]["y"] == 0])
    tau = float(neg[min(max(int(np.ceil(0.96 * len(neg))), 1), len(neg)) - 1])
    cal_auc = auroc(d["calibration"]["y"], head.predict_proba(d["calibration"]["X"]))
    Path(out).mkdir(parents=True, exist_ok=True)
    head.save(Path(out) / "head.json")
    (Path(out) / "operating_point.json").write_text(json.dumps(
        {"threshold": tau, "target_specificity": 0.96}, indent=1))
    print(f"{out}: cal_auroc={cal_auc:.4f} shift={shift:.4f} temp={temp:.4f} tau={tau:.4f}")

# frozen control
d = arrays("resnet50_in1k_v2_448")
heads = [LogisticHead.fit(d["train"]["X"], d["train"]["y"], class_balanced=True,
                          seed=s, epochs=800) for s in range(5)]
merged = LogisticHead(weights=np.mean([h.weights for h in heads], axis=0),
                      bias=float(np.mean([h.bias for h in heads])))
finish(merged, d, "runs/frozen_v1q_head")

# retrained encoder
d = arrays("resnet50_ft_v5_448_raw")
ck = torch.load("runs/finetune_v5/best_model.pt", map_location="cpu", weights_only=False)
fc = LogisticHead(weights=ck["model"]["fc.weight"].numpy().ravel().astype(np.float64),
                  bias=float(ck["model"]["fc.bias"].numpy()[0]))
finish(fc, d, "runs/finetune_v5_head")
