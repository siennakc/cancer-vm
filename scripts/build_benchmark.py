"""Build the sealed external benchmark from MIAS (site never trained on)."""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.bench.hermetic import (assert_no_byte_leakage, build_bench,
                                      near_duplicate_scan, training_content_hashes)
from oncoscope.bench.mias import build_mias_cases, read_pgm_canonical
from oncoscope.models.encoder import FrozenEncoder

RAW = Path("data/raw/MIAS")
OUT = Path("data/bench/mias_v1")


def main() -> None:
    tar = RAW / "all-mias.tar.gz"
    if not any(RAW.rglob("mdb*.pgm")):
        print(f"[bench] extracting {tar}")
        with tarfile.open(tar) as tf:
            tf.extractall(RAW / "extracted", filter="data")

    cases = build_mias_cases(RAW)
    n_pos = sum(c.label for c in cases)
    print(f"[bench] MIAS cases: {len(cases)} ({n_pos} malignant, "
          f"{len(cases) - n_pos} negative)")

    # -- leakage gate 1: byte-hash intersection with the FULL training corpus
    train_hashes = training_content_hashes(
        "data/processed/cases_v1.jsonl", "data/raw",
        "data/processed/train_content_hashes.json")
    assert_no_byte_leakage([RAW / c.image_path for c in cases], train_hashes)
    print(f"[bench] byte-leakage: 0 hits against {len(train_hashes)} training files")

    # -- leakage gate 2: near-duplicate scan in frozen-v1 embedding space
    enc = FrozenEncoder(weights_path="runs/frozen_v1_backbone.pt")
    bench_embs = []
    for i in range(0, len(cases), 8):
        chunk = cases[i : i + 8]
        bench_embs.append(enc.embed_batch(
            [read_pgm_canonical(RAW / c.image_path) for c in chunk]))
    bench_embs = np.concatenate(bench_embs)
    emb_dir = Path("data/embeddings/resnet50_in1k_v2_448")
    train_embs = np.stack([np.load(p) for p in sorted(emb_dir.glob("*.npy"))])
    worst = near_duplicate_scan(bench_embs, train_embs)
    print(f"[bench] near-duplicate scan: max cosine {worst:.4f} (fail>0.999)")

    seal = build_bench(cases, lambda c: read_pgm_canonical(RAW / c.image_path),
                       OUT, version="mias_v1")
    print(f"[bench] sealed: {json.dumps(seal)}")


if __name__ == "__main__":
    main()
