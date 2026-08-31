"""A/B on the public benchmark: v4 alone vs v4 inside the Onco-Harness pipeline.

Arm A (without harness): the calibrated v4 classifier's whole-image score —
identical to the number in results/public_cbis/posttrain_v4.json.

Arm B (with harness): the same weights driving the deterministic harness
(preflight QC -> window-grid detect -> TTA self-consistency -> zoom re-verify
-> symmetric FP/FN hunt -> aggregate -> rule-based adjudication). The v4
adapter proposes candidates from a 6-window grid (whole image, 2x2 overlapping
quadrants, center); the blindspot FN-hunter deliberately keeps the harness's
own second detector family (DoG) per axioms A4/A5. Cost columns are reported
beside accuracy — the harness must pay for what it buys.

Same 709 quarantined benchmark images, same metrics, paired delta CI.

Fairness contract (revised 2026-08-31): **both arms see native-resolution
pixels.** The first recorded run (delta -0.071) pre-shrank Arm B's input to
max side 1600 while Arm A's cached embeddings came from native DICOMs. The
shrink docstring claimed the letterbox made this lossless; that is true for
the whole-image window and FALSE for every sub-window — a 0.62-scale quadrant
of a 1600px image is ~992px and gets UPSAMPLED into the encoder, where the
same quadrant of a native image carries real detail. It also quietly nullified
the zoom re-verify stage, whose entire premise (axiom A1) is re-examining
candidates at native resolution. Some unknown share of the recorded -0.071 is
that handicap. ``--shrink N`` remains available as an explicit, recorded
speed knob for smoke runs; headline numbers must come from ``--shrink 0``
(the default), and the report JSON records which one ran.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

from oncoharness.ledger import EvidenceLedger
from oncoharness.reference.detector import Candidate
from oncoharness.state_machine import HarnessPipeline
from oncoharness.store import ArtifactStore
from oncoharness.tools import Toolbelt

from oncoscope.data.dicom_canonical import load_canonical
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci,
                                    paired_bootstrap_delta_ci,
                                    sensitivity_at_specificity)
from oncoscope.models.encoder import FrozenEncoder
from oncoscope.models.head import LogisticHead


def _shrink(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """OPTIONAL pre-resize, off by default — see the fairness contract above.

    Not harmless: sub-windows and zoom crops of a shrunk image reach the
    encoder upsampled. Use only for smoke runs, never for reported numbers.
    """
    h, w = pixels.shape
    if max_side <= 0 or max(h, w) <= max_side:
        return pixels
    import torch
    import torch.nn.functional as F
    s = max_side / max(h, w)
    t = torch.from_numpy(np.ascontiguousarray(pixels))[None, None]
    return F.interpolate(t, size=(round(h * s), round(w * s)),
                         mode="area")[0, 0].numpy()


def _n_calls(cost: dict) -> float:
    tc = (cost or {}).get("tool_calls", 0)
    return float(sum(tc.values())) if isinstance(tc, dict) else float(tc)


class V4WindowDetector:
    """v4 classifier as a candidate proposer: 6 fixed windows, batched on MPS."""

    def __init__(self, weights: str, head_path: str):
        self.enc = FrozenEncoder(tag="ab", weights_path=weights, normalize=False,
                                 input_size=(1152, 896),
                                 mean=(0.449,) * 3, std=(0.226,) * 3)
        self.head = LogisticHead.load(head_path)
        self.forwards = 0

    @staticmethod
    def _windows(h, w):
        half_h, half_w = h // 2, w // 2
        qh, qw = int(h * 0.62), int(w * 0.62)          # overlapping quadrants
        return [
            (0, 0, w, h),                               # whole image
            (0, 0, qw, qh), (w - qw, 0, w, qh),         # top L/R
            (0, h - qh, qw, h), (w - qw, h - qh, w, h), # bottom L/R
            ((w - half_w) // 2, (h - half_h) // 2,
             (w + half_w) // 2, (h + half_h) // 2),     # center
        ]

    def propose(self, pixels: np.ndarray) -> list[Candidate]:
        boxes = self._windows(*pixels.shape)
        crops = [pixels[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
        probs = self.head.predict_proba(self.enc.embed_batch(crops))
        self.forwards += len(crops)
        return [Candidate(box=b, score=float(p)) for b, p in zip(boxes, probs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/ab_harness/report.json")
    ap.add_argument("--shrink", type=int, default=0,
                    help="max side for Arm B input; 0 (default) = native, the only "
                         "fair setting — nonzero is a smoke-run speed knob and is "
                         "recorded in the report")
    args = ap.parse_args()
    if args.shrink:
        print(f"[ab] WARNING: --shrink {args.shrink} handicaps Arm B; "
              "this run's numbers are not comparable to Arm A", flush=True)

    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest("data/processed/splits_v2.json")
    bench = [c for c in cases
             if splits.split_of(f"{c.site}/{c.patient_id}") == "public_bench"]
    bench.sort(key=lambda c: c.case_id)
    if args.limit:
        bench = bench[: args.limit]
    y = np.array([c.label for c in bench], float)
    pids = [c.patient_id for c in bench]
    print(f"[ab] {len(bench)} benchmark cases", flush=True)

    # ---- Arm A: model alone (cached HR embeddings -> calibrated head) ----
    head = LogisticHead.load("runs/finetune_v4_head/head.json")
    Xa = np.stack([np.load(f"data/embeddings/resnet50_ft_v4_1152x896_raw/{c.case_id}.npy")
                   for c in bench])
    scores_a = head.predict_proba(Xa)

    # ---- Arm B: full harness around the same weights ----
    detector = V4WindowDetector("runs/posttrain_v4/best_model.pt",
                                "runs/finetune_v4_head/head.json")
    root = Path("runs/ab_harness"); root.mkdir(parents=True, exist_ok=True)
    pipeline = HarnessPipeline(
        Toolbelt(ArtifactStore(str(root / "artifacts")),
                 EvidenceLedger(str(root / "ledger.jsonl")),
                 detector=detector),
        consistency_reads=3, min_reproduced=2,
        policy_id="ab_v4_rule_adjudicated_v1",
    )
    scores_b, decisions, tool_calls, wall = [], [], [], []
    t0 = time.time()
    for i, c in enumerate(bench, 1):
        px = _shrink(load_canonical(Path("data/raw") / c.dicom_path).pixels,
                     args.shrink)
        t1 = time.time()
        report = pipeline.run_case(c.case_id, px)
        wall.append((time.time() - t1) * 1000)
        scores_b.append(report.score)
        decisions.append(str(report.decision))
        tool_calls.append(_n_calls(report.cost))
        if i % 25 == 0:
            rate = i / (time.time() - t0)
            print(f"[ab] {i}/{len(bench)}  {rate:.2f} case/s  "
                  f"eta {(len(bench) - i) / rate / 60:.0f}m", flush=True)
    scores_b = np.array(scores_b)

    def block(scores):
        a = clustered_bootstrap_ci(y, scores, pids, auroc, iterations=2000)
        s = clustered_bootstrap_ci(y, scores, pids, sensitivity_at_specificity,
                                   iterations=2000)
        return {"auroc": round(a[0], 4), "auroc_ci95": [round(a[1], 4), round(a[2], 4)],
                "sens_at_spec96": round(s[0], 4),
                "sens_at_spec96_ci95": [round(s[1], 4), round(s[2], 4)]}

    delta = paired_bootstrap_delta_ci(y, scores_b, scores_a, pids, auroc,
                                      iterations=2000)

    # Deferral informativeness, computed here so the CARD's analysis is
    # reproducible from the report alone: if Arm A performs the same on the
    # cases Arm B deferred as on the ones it answered, the deferrals bought
    # nothing (they were not selecting hard cases).
    deferred = np.array([d.endswith("defer_to_human") for d in decisions])
    def _subset(mask):
        if mask.sum() == 0 or len(set(y[mask])) < 2:
            return {"n": int(mask.sum()), "auroc_arm_a": None}
        return {"n": int(mask.sum()),
                "auroc_arm_a": round(auroc(y[mask], scores_a[mask]), 4),
                "mean_abs_error_arm_a": round(
                    float(np.mean(np.abs(scores_a[mask] - y[mask]))), 4)}
    deferral_analysis = {
        "deferred": _subset(deferred),
        "answered": _subset(~deferred),
        "deferral_rate": round(float(deferred.mean()), 4),
    }

    out = {
        "benchmark": "CBIS-DDSM official test split",
        "n": len(bench),
        "arm_b_input_max_side": args.shrink or "native",
        "fair_comparison": not bool(args.shrink),
        "arm_a_model_alone": block(scores_a),
        "arm_b_harness_rule_adjudicated": {
            **block(scores_b),
            "mean_tool_calls": round(float(np.mean(tool_calls)), 1),
            "mean_wall_ms": round(float(np.mean(wall)), 0),
            "model_forwards_total": detector.forwards,
            "decisions": {d: decisions.count(d) for d in sorted(set(decisions))},
        },
        "paired_delta_auroc_b_minus_a": {
            "point": round(delta[0], 4),
            "ci95": [round(delta[1], 4), round(delta[2], 4)],
        },
        "deferral_analysis": deferral_analysis,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
