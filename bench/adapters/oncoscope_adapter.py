"""Benchmark adapter: FrozenEncoder + calibrated LogisticHead, offline.

Runs inside the hermetic sandbox: no network (torchvision weights load from a
local checkpoint, never the hub), inputs are bare pixel arrays, output is one
JSONL row per case. Model selection via BENCH_MODEL:

  baseline_v1  — stock ResNet-50 features (L2-normed) + refit ensemble head
  finetune_v2  — fine-tuned backbone (raw features, grayscale stats) + fc head

Abstention: |p - 0.5| < margin emits abstain=true (score still reported — a
deferral ships evidence, axiom A13).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["BENCH_REPO_ROOT"])
sys.path.insert(0, str(ROOT / "src"))

from oncoscope.models.encoder import FrozenEncoder  # noqa: E402
from oncoscope.models.head import LogisticHead      # noqa: E402

MODELS = {
    "baseline_v1": dict(
        weights=ROOT / "runs/frozen_v1_backbone.pt", normalize=True,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        head=ROOT / "results/baseline_v1/head.json"),
    "finetune_v2": dict(
        weights=ROOT / "runs/finetune_v2/best_model.pt", normalize=False,
        mean=(0.449, 0.449, 0.449), std=(0.226, 0.226, 0.226),
        head=ROOT / "runs/finetune_v2b_head/head.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--abstain-margin", type=float, default=0.05)
    args = ap.parse_args()

    cfg = MODELS[os.environ.get("BENCH_MODEL", "finetune_v2")]
    enc = FrozenEncoder(tag="bench", weights_path=str(cfg["weights"]),
                        normalize=cfg["normalize"], mean=cfg["mean"], std=cfg["std"])
    # determinism first: the double-run check compares scores to 1e-6
    enc._torch.use_deterministic_algorithms(True, warn_only=True)
    head = LogisticHead.load(cfg["head"])

    staged = Path(args.input)
    ids = json.loads((staged / "manifest.json").read_text())["case_ids"]
    with open(args.output, "w") as out:
        for i in range(0, len(ids), 8):
            chunk = ids[i : i + 8]
            pixels = [np.load(staged / "cases" / f"{c}.npy").astype(np.float32)
                      for c in chunk]
            embs = enc.embed_batch(pixels)
            probs = head.predict_proba(embs)
            for cid, p in zip(chunk, probs):
                out.write(json.dumps({
                    "case_id": cid, "score": float(np.clip(p, 0.0, 1.0)),
                    "abstain": bool(abs(p - 0.5) < args.abstain_margin),
                }) + "\n")


if __name__ == "__main__":
    main()
