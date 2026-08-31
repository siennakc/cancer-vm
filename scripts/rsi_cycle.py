"""One real RSI cycle: mine the weak slice -> DFR candidate -> conjunctive gate.

The self-improvement loop, run for the first time on the real trained model
instead of phantoms. Champion: the v4 calibrated head. Failure slice (from the
benchmark subgroup grid): density-c. Candidate: a Deep-Feature-Reweighting
refit of the head on a (site x density)-balanced sample of the train split,
recalibrated identically. The Onco-Harness gate — non-inferiority, subgroup
floors, calibration ceiling, negative-flip, determinism — decides promotion.
The gate's verdict is the output either way; the loop is self-improving,
never self-certifying.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import torch
from oncoharness.gate import load_rules, run_gate
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc, sensitivity_at_specificity
from oncoscope.models.head import LogisticHead, dfr_refit

EMB = Path("data/embeddings/resnet50_ft_v4_1152x896_raw")
cases = read_case_table("data/processed/cases_v1.jsonl")
splits = load_manifest("data/processed/splits_v2.json")

def arrays(split):
    rows = [c for c in cases if splits.split_of(f"{c.site}/{c.patient_id}") == split]
    return {
        "X": np.stack([np.load(EMB / f"{c.case_id}.npy") for c in rows]),
        "y": np.array([c.label for c in rows], float),
        "pid": [f"{c.site}/{c.patient_id}" for c in rows],
        "site": [c.site for c in rows],
        "density": [(c.density_band or "na") for c in rows],
        "age": [(c.age_band or "na") for c in rows],
    }

tr, cal, thr = arrays("train"), arrays("calibration"), arrays("threshold")

# champion
ck = torch.load("runs/posttrain_v4/best_model.pt", map_location="cpu", weights_only=False)
champ = LogisticHead(weights=ck["model"]["fc.weight"].numpy().ravel().astype(np.float64),
                     bias=float(ck["model"]["fc.bias"].numpy()[0]))
champ.apply_prior_correction(0.5, float(cal["y"].mean()))
champ.calibrate_temperature(cal["X"], cal["y"])

# candidate: DFR on (site x density) groups, then the identical calibration chain
groups = np.array([f"{s}/{d}" for s, d in zip(tr["site"], tr["density"])])
sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}
print("[rsi] DFR groups:", sizes, flush=True)
cand = dfr_refit(tr["X"], tr["y"], groups, seed=0)
cand.apply_prior_correction(0.5, float(cal["y"].mean()))
cand.calibrate_temperature(cal["X"], cal["y"])

s_champ, s_cand = champ.predict_proba(thr["X"]), cand.predict_proba(thr["X"])
print(f"[rsi] threshold-split AUROC: champion {auroc(thr['y'], s_champ):.4f}  "
      f"candidate {auroc(thr['y'], s_cand):.4f}", flush=True)
for attr in ("site", "density"):
    for lvl in sorted(set(thr[attr])):
        m = np.array(thr[attr]) == lvl
        if m.sum() >= 30 and len(set(thr["y"][m])) > 1:
            print(f"[rsi]   {attr}={lvl:4} n={m.sum():3}  champ {auroc(thr['y'][m], s_champ[m]):.3f}"
                  f"  cand {auroc(thr['y'][m], s_cand[m]):.3f}", flush=True)

rules = load_rules("/Users/mike/onco-harness/gates/gate_rules.yaml")
result = run_gate(
    rules, thr["y"], s_cand, s_champ, thr["pid"],
    subgroups={"site": thr["site"], "density_band": thr["density"], "age_band": thr["age"]},
    candidate_scores_rerun=cand.predict_proba(thr["X"]),
)
print("\n[rsi] GATE:", "PASS — candidate promotable" if result.passed else "FAIL — champion retained", flush=True)
for c in result.checks:
    print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}", flush=True)

out = {"cycle": 1, "champion": "finetune_v4_head", "candidate": "dfr_site_x_density",
       "gate_passed": result.passed,
       "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks]}
Path("results/rsi").mkdir(parents=True, exist_ok=True)
Path("results/rsi/cycle1.json").write_text(json.dumps(out, indent=1))
if result.passed:
    cand.save("results/rsi/cycle1_candidate_head.json")
print("[rsi] recorded results/rsi/cycle1.json", flush=True)
