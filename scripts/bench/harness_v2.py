"""Harness v2 for the mammography A/B — every component gated, per the spec.

Components (each with its falsifier, evaluated before it ships):
  1. 4-flip TTA (identity, h, v, hv), mean of probabilities.
     Gate: must raise calibration-split AUROC, else identity only.
  2. Mondrian per-class conformal deferral (two-sided band): per-class quantiles
     of s = 1 - p_hat_y on the calibration split, alpha = 0.10 each class.
     |set|=1 -> act; |set|=2 -> defer; |set|=0 -> OOD, defer.
     Falsifier: risk-coverage curve flat/inverted -> deferral uninformative.
  3. Asymmetric veto is trivial single-stream (nothing downstream lowers scores).

Arms on the official CBIS-DDSM test split (public_bench):
  A  = v4 identity scores (the 0.771 baseline)
  B2 = harness v2 (gated TTA + conformal deferral)
Reported: full-set paired delta, non-deferred AUROC vs A on same subset,
coverage, deferral informativeness. Honest either way.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import torch
from oncoscope.data.dicom_canonical import load_canonical
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc, clustered_bootstrap_ci, paired_bootstrap_delta_ci
from oncoscope.models.encoder import FrozenEncoder
from oncoscope.models.head import LogisticHead

RAW = Path("data/raw")
VIEWS = ("id", "h", "v", "hv")


def tta_embed(split_cases, enc, view):
    """Embed a split under a flip view, cached per view."""
    out_dir = Path(f"data/embeddings/resnet50_ft_v4_1152x896_raw_tta_{view}")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in split_cases if not (out_dir / f"{c.case_id}.npy").exists()]
    print(f"[h2] view={view}: {len(todo)} to embed", flush=True)
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        pixels = []
        for c in chunk:
            px = load_canonical(RAW / c.dicom_path).pixels
            if "h" in view: px = px[:, ::-1]
            if "v" in view: px = px[::-1, :]
            pixels.append(np.ascontiguousarray(px))
        embs = enc.embed_batch(pixels)
        for c, e in zip(chunk, embs):
            np.save(out_dir / f"{c.case_id}.npy", e.astype(np.float32))
    return out_dir


def main():
    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest("data/processed/splits_v2.json")
    of = lambda c: splits.split_of(f"{c.site}/{c.patient_id}")
    cal = [c for c in cases if of(c) == "calibration"]
    bench = sorted([c for c in cases if of(c) == "public_bench"], key=lambda c: c.case_id)
    head = LogisticHead.load("runs/finetune_v4_head/head.json")
    enc = FrozenEncoder(tag="h2", weights_path="runs/posttrain_v4/best_model.pt",
                        normalize=False, input_size=(1152, 896),
                        mean=(0.449,) * 3, std=(0.226,) * 3)

    def view_probs(group, view):
        if view == "id":
            d = Path("data/embeddings/resnet50_ft_v4_1152x896_raw")
        else:
            d = tta_embed(group, enc, view)
        X = np.stack([np.load(d / f"{c.case_id}.npy") for c in group])
        return head.predict_proba(X)

    y_cal = np.array([c.label for c in cal], float)
    p_cal = {v: view_probs(cal, v) for v in VIEWS}
    id_auc = auroc(y_cal, p_cal["id"])
    tta_cal = np.mean([p_cal[v] for v in VIEWS], axis=0)
    tta_auc = auroc(y_cal, tta_cal)
    use_tta = tta_auc > id_auc
    print(f"[h2] cal AUROC identity={id_auc:.4f} 4-flip TTA={tta_auc:.4f} "
          f"-> TTA {'ENABLED' if use_tta else 'DISABLED (falsified)'}", flush=True)

    p_cal_final = tta_cal if use_tta else p_cal["id"]
    # Mondrian per-class conformal quantiles (two-sided band)
    alpha = 0.10
    q = {}
    for c_ in (0, 1):
        s = 1 - np.where(c_ == 1, p_cal_final, 1 - p_cal_final)[y_cal == c_]
        n = len(s)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q[c_] = float(np.sort(s)[k - 1])
    print(f"[h2] conformal quantiles: q_neg={q[0]:.4f} q_pos={q[1]:.4f}", flush=True)

    y = np.array([c.label for c in bench], float)
    pids = [c.patient_id for c in bench]
    p_id = view_probs(bench, "id")
    p_b2 = np.mean([view_probs(bench, v) for v in VIEWS], axis=0) if use_tta else p_id

    in_pos = (1 - p_b2) <= q[1]
    in_neg = p_b2 <= q[0]
    setsize = in_pos.astype(int) + in_neg.astype(int)
    act = setsize == 1
    cov = float(act.mean())

    a_all = clustered_bootstrap_ci(y, p_id, pids, auroc, iterations=2000)
    b_all = clustered_bootstrap_ci(y, p_b2, pids, auroc, iterations=2000)
    delta = paired_bootstrap_delta_ci(y, p_b2, p_id, pids, auroc, iterations=2000)
    sub = {}
    if act.sum() > 50 and len(set(y[act])) > 1:
        pid_act = [p for p, a in zip(pids, act) if a]
        sub = {"auroc_B2_nondeferred": round(auroc(y[act], p_b2[act]), 4),
               "auroc_A_same_subset": round(auroc(y[act], p_id[act]), 4),
               "auroc_A_deferred_subset": round(auroc(y[~act], p_id[~act]), 4)
               if (~act).sum() > 30 and len(set(y[~act])) > 1 else None}
    # risk-coverage falsifier: retained accuracy at tau=0.5 vs overall
    err_all = float(np.mean((p_b2 > 0.5) != (y == 1)))
    err_ret = float(np.mean((p_b2[act] > 0.5) != (y[act] == 1))) if act.sum() else 1.0
    informative = err_ret < err_all

    out = {"benchmark": "CBIS-DDSM official test", "n": len(y),
           "tta_enabled": bool(use_tta),
           "cal_auroc_identity": round(id_auc, 4), "cal_auroc_tta": round(tta_auc, 4),
           "armA_auroc": round(a_all[0], 4), "armA_ci": [round(a_all[1], 4), round(a_all[2], 4)],
           "armB2_auroc": round(b_all[0], 4), "armB2_ci": [round(b_all[1], 4), round(b_all[2], 4)],
           "paired_delta_B2_minus_A": {"point": round(delta[0], 4),
                                       "ci95": [round(delta[1], 4), round(delta[2], 4)]},
           "conformal": {"alpha": alpha, "q_neg": q[0], "q_pos": q[1],
                         "coverage": round(cov, 4),
                         "deferral_rate": round(1 - cov, 4),
                         "ood_flags": int((setsize == 0).sum()),
                         "err_retained_at_0.5": round(err_ret, 4),
                         "err_overall_at_0.5": round(err_all, 4),
                         "informative": bool(informative)},
           "selective": sub}
    Path("results/ab_harness").mkdir(parents=True, exist_ok=True)
    Path("results/ab_harness/report_v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
