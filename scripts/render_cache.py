"""Pre-render every case to a 448x448 float16 tensor once (training input cache).

Same breast_crop + letterbox as FrozenEncoder.preprocess — one preprocessing
path, cached before augmentation. ~2.7 GB for the corpus vs 126 GB of DICOM.
Resumable like the embedding cache.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.data.dicom_canonical import load_canonical
from oncoscope.data.mammography import read_case_table
from oncoscope.models.encoder import breast_crop, letterbox

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="data/cache/render448")
_ap.add_argument("--height", type=int, default=448)
_ap.add_argument("--width", type=int, default=448)
_A = _ap.parse_args()
RAW, OUT, SIZE = Path("data/raw"), Path(_A.out), (_A.height, _A.width)


def render(case):
    out = OUT / f"{case.case_id}.npy"
    if out.exists():
        return True
    try:
        px = load_canonical(RAW / case.dicom_path).pixels
        np.save(out, letterbox(breast_crop(px.astype(np.float32)), SIZE).astype(np.float16))
        return True
    except Exception as exc:
        print(f"[render] FAIL {case.case_id}: {exc}", flush=True)
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_case_table("data/processed/cases_v1.jsonl")
    t0, ok = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for i, good in enumerate(pool.map(render, cases), 1):
            ok += bool(good)
            if i % 500 == 0:
                print(f"[render] {i}/{len(cases)} ({i/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[render] done ok={ok}/{len(cases)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
