"""Cache frozen-encoder embeddings for every case (T-2.1).

Resumable: a case whose .npy already exists under the encoder tag is skipped,
so interrupting and re-running is safe. Decode (pydicom, the bottleneck) runs
in a small thread pool feeding MPS batches.
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
from oncoscope.models.encoder import FrozenEncoder

RAW, OUT = Path("data/raw"), Path("data/embeddings")
BATCH = 8


def main() -> None:
    cases = read_case_table("data/processed/cases_v1.jsonl")
    enc = FrozenEncoder()
    out_dir = OUT / enc.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [c for c in cases if not (out_dir / f"{c.case_id}.npy").exists()]
    print(f"[embed] {len(cases)} cases, {len(todo)} to embed "
          f"(encoder={enc.tag}, device={enc.device})", flush=True)

    def decode(case):
        try:
            return case, load_canonical(RAW / case.dicom_path).pixels
        except Exception as exc:
            return case, exc

    done = failed = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        pending, batch = [], []
        it = iter(todo)
        for case in it:
            pending.append(pool.submit(decode, case))
            if len(pending) < BATCH * 3:
                continue
            fut = pending.pop(0)
            case, pixels = fut.result()
            if isinstance(pixels, Exception):
                print(f"[embed] DECODE FAIL {case.case_id}: {pixels}", flush=True)
                failed += 1
                continue
            batch.append((case, pixels))
            if len(batch) == BATCH:
                embs = enc.embed_batch([p for _, p in batch])
                for (c, _), e in zip(batch, embs):
                    np.save(out_dir / f"{c.case_id}.npy", e.astype(np.float32))
                done += len(batch)
                batch = []
                if done % 96 == 0:
                    rate = done / (time.time() - t0)
                    eta = (len(todo) - done) / max(rate, 1e-9) / 60
                    print(f"[embed] {done}/{len(todo)}  {rate:.1f}/s  eta {eta:.0f}m", flush=True)
        # drain the tail
        for fut in pending:
            case, pixels = fut.result()
            if isinstance(pixels, Exception):
                print(f"[embed] DECODE FAIL {case.case_id}: {pixels}", flush=True)
                failed += 1
                continue
            batch.append((case, pixels))
        for i in range(0, len(batch), BATCH):
            chunk = batch[i : i + BATCH]
            embs = enc.embed_batch([p for _, p in chunk])
            for (c, _), e in zip(chunk, embs):
                np.save(out_dir / f"{c.case_id}.npy", e.astype(np.float32))
            done += len(chunk)

    print(f"[embed] complete: ok={done} failed={failed} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
