"""Ensemble + flip-TTA on the public benchmark — the last cheap levers.

Both are evidence-backed (Shen et al.: 4-model averaging 0.88 -> 0.91;
4-flip TTA +0.01-0.03 AUC) and neither trains anything. Two modes:

CACHED (default, seconds, no GPU): average each member's calibrated
probabilities computed from its already-cached bench embeddings. Members
default to v3 + v4 (+ v5 when its cache exists).

TTA (--tta, needs GPU + data/raw): per member with weights, embed each bench
image 4 ways (identity, horizontal flip, vertical flip, both), average the
member's probabilities over the views, then ensemble. Flips are
label-invariant for malignancy; identity is always included (the TTA
literature's one hard rule). Laterality-sensitive OUTPUTS are unaffected —
this scores, it never localizes.

Discipline mirrors eval_public_cbis.py: quarantined bench membership from
splits_v2, patient-clustered CIs, paired delta vs the best single member, and
per-member encoder-lineage verification when checkpoints are given. This is
an EVAL-TIME change: no gate is bypassed because nothing is promoted — the
result JSON is the artifact, and the README only changes if the paired delta's
CI clears zero.

    .venv/bin/python scripts/eval_ensemble.py                # cached mode
    .venv/bin/python scripts/eval_ensemble.py --tta          # + flip TTA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci, ece_adaptive,
                                    paired_bootstrap_delta_ci,
                                    sensitivity_at_specificity)
from oncoscope.models.head import LogisticHead

# name -> (embedding tag, committed head, weights for TTA/lineage, input size)
DEFAULT_MEMBERS = {
    "finetune_v3": ("resnet50_ft_v3_448_raw", "results/finetune_v3/head.json",
                    "runs/finetune_v3/best_model.pt", 448),
    "posttrain_v4": ("resnet50_ft_v4_1152x896_raw", "results/posttrain_v4/head.json",
                     "runs/posttrain_v4/best_model.pt", (1152, 896)),
    "finetune_v5": ("resnet50_ft_v5_448_raw", "runs/finetune_v5_head/head.json",
                    "runs/finetune_v5/best_model.pt", 448),
}
FLIPS = ("identity", "h", "v", "hv")


def flip(pixels: np.ndarray, mode: str) -> np.ndarray:
    if mode == "h":
        return pixels[:, ::-1]
    if mode == "v":
        return pixels[::-1, :]
    if mode == "hv":
        return pixels[::-1, ::-1]
    return pixels


def verify_lineage(weights_path: Path, splits_sha: str, name: str) -> None:
    import torch
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if ck.get("tainted"):
        raise SystemExit(f"{name}: checkpoint is TAINTED — refusing")
    shas = [ck.get("splits_sha256")] + [
        e.get("splits_sha256") for e in ck.get("init_lineage", [])]
    bad = [s for s in shas if s != splits_sha]
    if bad:
        raise SystemExit(
            f"{name}: lineage includes manifests other than splits_v2 "
            f"({[str(s)[:12] for s in bad]}) — a fitting stage may have touched "
            "bench patients")


def member_probs_cached(tag: str, head: LogisticHead, cases) -> np.ndarray:
    X = np.stack([np.load(f"data/embeddings/{tag}/{c.case_id}.npy") for c in cases])
    return head.predict_proba(X)


def member_probs_tta(weights: Path, input_size, head: LogisticHead, cases,
                     batch: int = 8) -> np.ndarray:
    """Mean probability over the 4 flip views, identity included."""
    from oncoscope.data.dicom_canonical import load_canonical
    from oncoscope.models.encoder import FrozenEncoder

    enc = FrozenEncoder(tag="tta", weights_path=str(weights), normalize=False,
                        input_size=input_size, mean=(0.449,) * 3, std=(0.226,) * 3)
    per_view = {m: [] for m in FLIPS}
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        pixels = [np.ascontiguousarray(
            load_canonical(Path("data/raw") / c.dicom_path).pixels.astype(np.float32))
            for c in chunk]
        for mode in FLIPS:
            emb = enc.embed_batch([np.ascontiguousarray(flip(p, mode)) for p in pixels])
            per_view[mode].append(head.predict_proba(emb))
        if (i // batch) % 10 == 0:
            print(f"[tta] {i + len(chunk)}/{len(cases)}", flush=True)
    views = np.stack([np.concatenate(per_view[m]) for m in FLIPS])
    return views.mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=None,
                    help=f"subset of {sorted(DEFAULT_MEMBERS)} (default: all with "
                         "an existing embedding cache)")
    ap.add_argument("--tta", action="store_true",
                    help="4-flip test-time augmentation per member (needs GPU + data/raw)")
    ap.add_argument("--skip-lineage", action="store_true",
                    help="skip checkpoint lineage verification (only when a "
                         "member's checkpoint file is genuinely unavailable)")
    ap.add_argument("--name", default=None, help="report name override")
    args = ap.parse_args()

    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest("data/processed/splits_v2.json")
    bench = sorted((c for c in cases
                    if splits.split_of(f"{c.site}/{c.patient_id}") == "public_bench"),
                   key=lambda c: c.case_id)
    y = np.array([c.label for c in bench], float)
    pids = [c.patient_id for c in bench]
    print(f"[ens] {len(bench)} bench cases", flush=True)

    wanted = args.members or [
        n for n, (tag, head, _, _) in DEFAULT_MEMBERS.items()
        if Path(f"data/embeddings/{tag}").exists() and Path(head).exists()
    ]
    if len(wanted) < 2:
        raise SystemExit(f"need >=2 members, found {wanted} — check embedding "
                         "caches and head paths (see DEFAULT_MEMBERS)")

    member_scores: dict[str, np.ndarray] = {}
    for name in wanted:
        tag, head_path, weights, input_size = DEFAULT_MEMBERS[name]
        head = LogisticHead.load(head_path)
        if not args.skip_lineage:
            if not Path(weights).exists():
                raise SystemExit(f"{name}: {weights} missing — download the release "
                                 "or pass --skip-lineage (and say so in the writeup)")
            verify_lineage(Path(weights), splits.sha256, name)
        if args.tta:
            member_scores[name] = member_probs_tta(Path(weights), input_size, head, bench)
        else:
            member_scores[name] = member_probs_cached(tag, head, bench)
        a = auroc(y, member_scores[name])
        print(f"[ens] {name}: AUROC {a:.4f}" + (" (4-flip TTA)" if args.tta else ""),
              flush=True)

    ens = np.mean(np.stack(list(member_scores.values())), axis=0)
    best_single = max(member_scores, key=lambda n: auroc(y, member_scores[n]))

    a = clustered_bootstrap_ci(y, ens, pids, auroc, iterations=2000)
    s = clustered_bootstrap_ci(y, ens, pids, sensitivity_at_specificity, iterations=2000)
    delta = paired_bootstrap_delta_ci(y, ens, member_scores[best_single], pids,
                                      auroc, iterations=2000)
    report = {
        "benchmark": "CBIS-DDSM official test split",
        "model": args.name or ("ensemble_tta" if args.tta else "ensemble"),
        "members": {n: round(auroc(y, sc), 4) for n, sc in member_scores.items()},
        "aggregation": "mean of calibrated probabilities"
                       + ("; per-member mean over 4 flip views (identity included)"
                          if args.tta else ""),
        "overall": {
            "n": len(bench),
            "auroc": round(a[0], 4), "auroc_ci95": [round(a[1], 4), round(a[2], 4)],
            "sens_at_spec96": round(s[0], 4),
            "sens_at_spec96_ci95": [round(s[1], 4), round(s[2], 4)],
            "ece_adaptive": round(float(ece_adaptive(y, ens)), 4),
        },
        "paired_delta_vs_best_single": {
            "best_single": best_single,
            "point": round(delta[0], 4),
            "ci95": [round(delta[1], 4), round(delta[2], 4)],
            "reportable_gain": bool(delta[1] > 0),
        },
        "lineage": ("verified per member" if not args.skip_lineage
                    else "SKIPPED — noted"),
    }
    out = Path(f"results/public_cbis/{report['model']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"[ens] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
