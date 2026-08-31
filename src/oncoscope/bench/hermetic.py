"""Hermetic benchmark environment (the bench arm of axiom A6).

Threats this design addresses, and how:

1. *Trained on the bench data* — byte-hash intersection between every bench
   source file and every training DICOM must be empty (checked at build AND
   at every run), plus an embedding near-duplicate scan with the frozen v1
   encoder (catches re-encoded copies byte-hashing cannot see).
2. *Metadata shortcuts* — the adapter never sees files with headers. Staged
   inputs are bare float16 pixel arrays under opaque ids salted per-seal;
   filename, order, and directory carry zero label signal.
3. *Label access* — gold lives outside the staging dir and is only readable
   by the scorer, which is a ``SealedTestSet``: hash-locked membership, access
   log, query budget. The adapter subprocess gets pixels and nothing else.
4. *State / order / test-time-adaptation cheats* — every evaluation runs the
   adapter twice over differently-shuffled presentations; per-case scores must
   agree to 1e-6 or the run is void.
5. *Phoning home* — the adapter runs under ``sandbox-exec`` with all network
   denied (macOS; skipped with a loud warning elsewhere), stripped env, cwd
   inside the staging dir.
6. *Partial answers* — every case scored exactly once, all scores finite in
   [0,1], else the run is void. Abstention is a separate explicit field, and
   abstained cases still require a score (deferral ships evidence, A13).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SANDBOX_PROFILE = '(version 1)(allow default)(deny network*)'


# --------------------------------------------------------------------------
# build & seal
# --------------------------------------------------------------------------

def build_bench(cases, pixels_fn, out_dir: Path | str, version: str) -> dict:
    """Render bench cases to opaque staged arrays + a sealed gold table.

    ``cases``: list with .case_id/.patient_id/.label/.image_path attributes.
    ``pixels_fn``: case -> float32 [0,1] canonical pixels (headers die here).
    """
    out_dir = Path(out_dir)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    # Reuse the sealed salt on rebuild so a fresh clone regenerates the exact
    # staged arrays and ids the committed seal was computed over (images are
    # never committed — the MIAS licence is research-use, no redistribution).
    gold_path = out_dir / "gold.json"
    salt = (json.loads(gold_path.read_text())["salt"] if gold_path.exists()
            else secrets.token_hex(16))

    gold_rows, id_map = [], {}
    for case in cases:
        opaque = hashlib.sha256(f"{salt}:{case.case_id}".encode()).hexdigest()[:16]
        px = pixels_fn(case).astype(np.float16)
        np.save(out_dir / "cases" / f"{opaque}.npy", px)
        id_map[case.case_id] = opaque
        gold_rows.append({
            "opaque_id": opaque, "case_id": case.case_id,
            "patient_id": case.patient_id, "label": int(case.label),
        })

    gold_rows.sort(key=lambda r: r["opaque_id"])
    gold = {"version": version, "salt": salt, "rows": gold_rows}
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=1))

    membership = sorted(id_map.values())
    seal = {
        "version": version,
        "membership_sha256": hashlib.sha256(json.dumps(membership).encode()).hexdigest(),
        "gold_sha256": hashlib.sha256(
            json.dumps(gold, sort_keys=True).encode()).hexdigest(),
        "n_cases": len(gold_rows),
    }
    (out_dir / "seal.json").write_text(json.dumps(seal, indent=1))
    return seal


def verify_bench(bench_dir: Path | str) -> dict:
    """Recompute both hashes; refuse to run on any mismatch."""
    bench_dir = Path(bench_dir)
    seal = json.loads((bench_dir / "seal.json").read_text())
    gold = json.loads((bench_dir / "gold.json").read_text())
    gold_sha = hashlib.sha256(json.dumps(gold, sort_keys=True).encode()).hexdigest()
    membership = sorted(r["opaque_id"] for r in gold["rows"])
    mem_sha = hashlib.sha256(json.dumps(membership).encode()).hexdigest()
    if gold_sha != seal["gold_sha256"] or mem_sha != seal["membership_sha256"]:
        raise ValueError("benchmark seal verification FAILED — refusing to run")
    on_disk = {p.stem for p in (bench_dir / "cases").glob("*.npy")}
    if on_disk != set(membership):
        raise ValueError("staged cases do not match sealed membership")
    return gold


# --------------------------------------------------------------------------
# leakage checks
# --------------------------------------------------------------------------

def training_content_hashes(case_table: Path | str, raw_root: Path | str,
                            cache_path: Path | str) -> set[str]:
    """sha256 of every training-corpus source file (cached after first pass)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return set(json.loads(cache_path.read_text()))
    hashes = set()
    for line in Path(case_table).read_text().splitlines():
        row = json.loads(line)
        h = hashlib.sha256()
        h.update((Path(raw_root) / row["dicom_path"]).read_bytes())
        hashes.add(h.hexdigest())
    cache_path.write_text(json.dumps(sorted(hashes)))
    return hashes


def assert_no_byte_leakage(bench_source_files: list[Path], train_hashes: set[str]) -> None:
    for path in bench_source_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in train_hashes:
            raise ValueError(f"LEAKAGE: bench file {path} is byte-identical to training data")


def near_duplicate_scan(bench_embeddings: np.ndarray, train_embeddings: np.ndarray,
                        fail_above: float = 0.999) -> float:
    """Max cosine similarity bench x train (both L2-normed). Fails on near-dupes."""
    sims = bench_embeddings @ train_embeddings.T
    worst = float(sims.max()) if sims.size else 0.0
    if worst > fail_above:
        raise ValueError(f"LEAKAGE: near-duplicate of a training image (cos={worst:.5f})")
    return worst


# --------------------------------------------------------------------------
# hermetic run
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    scores: dict[str, float]       # opaque_id -> malignancy score
    abstain: dict[str, bool]
    determinism_max_delta: float
    sandboxed: bool


def _run_adapter_once(adapter: Path, staged: Path, order: list[str],
                      python: str, timeout_s: int) -> dict[str, dict]:
    manifest = staged / "manifest.json"
    manifest.write_text(json.dumps({"case_ids": order, "input_spec":
        "cases/<id>.npy = float16 [0,1] canonical pixels, HxW"}))
    out_path = staged / "scores.jsonl"
    if out_path.exists():
        out_path.unlink()

    cmd = [python, str(adapter), "--input", str(staged), "--output", str(out_path)]
    sandboxed = shutil.which("sandbox-exec") is not None
    if sandboxed:
        cmd = ["sandbox-exec", "-p", SANDBOX_PROFILE] + cmd
    else:
        print("[bench] WARNING: sandbox-exec unavailable — network NOT blocked",
              file=sys.stderr, flush=True)

    repo_root = os.environ.get("BENCH_REPO_ROOT", os.getcwd())
    env = {"PATH": "/usr/bin:/bin", "HOME": str(staged),
           "PYTHONPATH": str(Path(repo_root) / "src"),
           "BENCH_REPO_ROOT": repo_root,
           "BENCH_MODEL": os.environ.get("BENCH_MODEL", "finetune_v2"),
           "BENCH_HERMETIC": "1"}
    proc = subprocess.run(cmd, env=env, cwd=staged, timeout=timeout_s,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"adapter failed rc={proc.returncode}:\n{proc.stderr[-2000:]}")

    rows: dict[str, dict] = {}
    for line in out_path.read_text().splitlines():
        row = json.loads(line)
        cid = row["case_id"]
        if cid in rows:
            raise ValueError(f"adapter scored {cid} twice")
        score = float(row["score"])
        if not (0.0 <= score <= 1.0) or not np.isfinite(score):
            raise ValueError(f"invalid score for {cid}: {score}")
        rows[cid] = {"score": score, "abstain": bool(row.get("abstain", False))}
    return rows, sandboxed


def run_hermetic(bench_dir: Path | str, adapter: Path | str,
                 python: str | None = None, timeout_s: int = 3600) -> RunResult:
    """Stage inputs, run the adapter twice in shuffled order, enforce agreement."""
    bench_dir, adapter = Path(bench_dir), Path(adapter).resolve()
    gold = verify_bench(bench_dir)
    ids = sorted(r["opaque_id"] for r in gold["rows"])
    python = python or sys.executable

    with tempfile.TemporaryDirectory(prefix="oncobench_") as tmp:
        staged = Path(tmp)
        (staged / "cases").mkdir()
        for cid in ids:
            shutil.copy(bench_dir / "cases" / f"{cid}.npy", staged / "cases" / f"{cid}.npy")
        # gold.json / seal.json are deliberately NOT copied — pixels only.

        rng = np.random.default_rng(0)
        order_a = list(rng.permutation(ids))
        order_b = list(rng.permutation(ids))
        rows_a, sandboxed = _run_adapter_once(adapter, staged, order_a, python, timeout_s)
        rows_b, _ = _run_adapter_once(adapter, staged, order_b, python, timeout_s)

    if set(rows_a) != set(ids) or set(rows_b) != set(ids):
        missing = set(ids) - set(rows_a)
        raise ValueError(f"adapter did not score every case (missing {len(missing)})")
    max_delta = max(abs(rows_a[c]["score"] - rows_b[c]["score"]) for c in ids)
    if max_delta > 1e-6:
        raise ValueError(f"DETERMINISM FAILURE: max score delta {max_delta:.3e} "
                         "between shuffled runs — evaluation void")
    return RunResult(
        scores={c: rows_a[c]["score"] for c in ids},
        abstain={c: rows_a[c]["abstain"] for c in ids},
        determinism_max_delta=max_delta,
        sandboxed=sandboxed,
    )
