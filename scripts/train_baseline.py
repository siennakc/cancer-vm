"""Baseline v1: frozen ResNet-50 embeddings -> 5-seed logistic ensemble (T-2.1, T-2.2).

Axiom A11's control arm. Sequence, in the order A10 requires:

  1. fit 5 seeds class-balanced on ``train`` (rebalanced => outputs on a 50/50 prior)
  2. merge — linear heads average exactly: mean of member logits == logits of
     the averaged head, so the ensemble ships as one ``LogisticHead`` file
  3. prior-correct the merged head from 0.5 to the cohort prevalence measured
     on the *calibration* split (rebalancing without this is the 50-200x pitfall)
  4. temperature-scale on ``calibration``
  5. choose the operating threshold on ``threshold`` at spec >= 0.96
  6. report dev metrics on ``slice_discovery`` + per-site breakdown,
     patient-clustered bootstrap CIs
  7. spend ONE sealed-set query for the headline number

Never touches ``test`` outside the sealed scoring service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import (
    auroc,
    clustered_bootstrap_ci,
    ece_adaptive,
    sensitivity_at_specificity,
)
from oncoscope.eval.sealed import SealedTestSet
from oncoscope.models.head import LogisticHead

SEEDS = (0, 1, 2, 3, 4)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="resnet50_in1k_v2_448")
_ap.add_argument("--run", default="baseline_v1")
_ap.add_argument("--caller", default="baseline_v1_resnet50_5seed")
ARGS = _ap.parse_args()
EMB = Path("data/embeddings") / ARGS.tag
RUN = Path("runs") / ARGS.run


def load_split_arrays():
    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest("data/processed/splits_v1.json")
    data = {}
    missing = 0
    for c in cases:
        path = EMB / f"{c.case_id}.npy"
        if not path.exists():
            missing += 1
            continue
        split = splits.split_of(f"{c.site}/{c.patient_id}")
        d = data.setdefault(split, {"X": [], "y": [], "pid": [], "site": [], "cid": []})
        d["X"].append(np.load(path))
        d["y"].append(c.label)
        d["pid"].append(f"{c.site}/{c.patient_id}")
        d["site"].append(c.site)
        d["cid"].append(c.case_id)
    if missing:
        print(f"[train] WARNING: {missing} cases missing embeddings", flush=True)
    for d in data.values():
        d["X"] = np.stack(d["X"])
        d["y"] = np.array(d["y"], dtype=np.float64)
        d["site"] = np.array(d["site"])
    return data


def block(name, y, scores, pids):
    a = clustered_bootstrap_ci(y, scores, pids, auroc, iterations=1000)
    s = clustered_bootstrap_ci(y, scores, pids, sensitivity_at_specificity, iterations=1000)
    out = {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": a[0], "auroc_ci": [a[1], a[2]],
        "sens_at_spec96": s[0], "sens_at_spec96_ci": [s[1], s[2]],
        "ece_adaptive": float(ece_adaptive(y, scores)),
    }
    print(f"[train] {name:24} n={out['n']:5} prev={out['prevalence']:.2f} "
          f"auroc={out['auroc']:.4f} [{a[1]:.4f},{a[2]:.4f}] "
          f"sens@96={out['sens_at_spec96']:.3f} ece={out['ece_adaptive']:.4f}", flush=True)
    return out


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    data = load_split_arrays()
    tr, cal, thr, sd = data["train"], data["calibration"], data["threshold"], data["slice_discovery"]
    print(f"[train] train n={len(tr['y'])} prev={tr['y'].mean():.3f}", flush=True)

    # 1) five seeds, class-balanced
    heads = [LogisticHead.fit(tr["X"], tr["y"], class_balanced=True, seed=s, epochs=800)
             for s in SEEDS]
    seed_auroc = [auroc(cal["y"], h.predict_proba(cal["X"])) for h in heads]
    print(f"[train] per-seed cal AUROC: {[round(a, 4) for a in seed_auroc]} "
          f"(spread {max(seed_auroc) - min(seed_auroc):.5f})", flush=True)

    # 2) exact ensemble merge (linear => mean logits == merged head)
    merged = LogisticHead(
        weights=np.mean([h.weights for h in heads], axis=0),
        bias=float(np.mean([h.bias for h in heads])),
    )

    # 3) prior correction: balanced 0.5 -> cohort prevalence (calibration split)
    shift = merged.apply_prior_correction(0.5, float(cal["y"].mean()))
    # 4) temperature on the calibration split
    temp = merged.calibrate_temperature(cal["X"], cal["y"])
    print(f"[train] prior_shift={shift:.4f} temperature={temp:.4f}", flush=True)

    # 5) operating threshold at spec>=0.96, chosen on the threshold split
    p_thr = merged.predict_proba(thr["X"])
    neg = np.sort(p_thr[thr["y"] == 0])
    k = min(max(int(np.ceil(0.96 * len(neg))), 1), len(neg))
    tau = float(neg[k - 1])
    op = {
        "threshold": tau, "target_specificity": 0.96,
        "threshold_split_sens": float((p_thr[thr["y"] == 1] > tau).mean()),
        "threshold_split_spec": float((p_thr[thr["y"] == 0] <= tau).mean()),
    }
    print(f"[train] tau={tau:.4f} sens={op['threshold_split_sens']:.3f} "
          f"spec={op['threshold_split_spec']:.3f} (threshold split)", flush=True)

    # 6) dev metrics
    metrics = {"threshold_split": block("threshold", thr["y"], p_thr, thr["pid"])}
    p_sd = merged.predict_proba(sd["X"])
    metrics["slice_discovery"] = block("slice_discovery", sd["y"], p_sd, sd["pid"])
    for site in ("ddsm", "cmmd"):
        m = sd["site"] == site
        if m.sum() > 30:
            metrics[f"slice_discovery/{site}"] = block(
                f"  {site}", sd["y"][m], p_sd[m], [p for p, s in zip(sd["pid"], sd["site"]) if s == site])

    # 7) one sealed query — the only touch of test
    te = data["test"]
    sealed = SealedTestSet("data/processed/sealed_test_v1.json",
                           "data/processed/sealed_access_log.jsonl", query_budget=50)
    order = np.argsort(np.array(te["cid"]))
    sealed_metrics = sealed.score(
        case_ids=sorted(te["cid"]),
        y_true=te["y"][order],
        scores=merged.predict_proba(te["X"])[order],
        caller=ARGS.caller,
    )
    print(f"[train] SEALED TEST: {json.dumps(sealed_metrics)}", flush=True)
    metrics["sealed_test"] = sealed_metrics

    merged.save(RUN / "head.json")
    (RUN / "operating_point.json").write_text(json.dumps(op, indent=1))
    (RUN / "metrics.json").write_text(json.dumps({
        "encoder": ARGS.tag, "seeds": list(SEEDS),
        "per_seed_cal_auroc": seed_auroc,
        "prior_logit_shift": shift, "temperature": temp,
        "splits_sha256": load_manifest("data/processed/splits_v1.json").sha256,
        "metrics": metrics,
    }, indent=1))
    print(f"[train] saved to {RUN}/", flush=True)


if __name__ == "__main__":
    main()
