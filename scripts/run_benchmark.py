"""Run an adapter against the sealed external benchmark, hermetically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.bench.hermetic import run_hermetic, verify_bench
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci, ece_adaptive,
                                    sensitivity_at_specificity)
from oncoscope.eval.sealed import SealedTestSet

BENCH = Path("data/bench/mias_v1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["baseline_v1", "finetune_v2"])
    args = ap.parse_args()

    import os
    os.environ["BENCH_MODEL"] = args.model
    os.environ.setdefault("BENCH_REPO_ROOT", str(Path.cwd()))

    result = run_hermetic(BENCH, "bench/adapters/oncoscope_adapter.py",
                          python=str(Path.cwd() / ".venv/bin/python"))
    gold = verify_bench(BENCH)
    rows = gold["rows"]
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    scores = np.array([result.scores[r["opaque_id"]] for r in rows])
    abst = np.array([result.abstain[r["opaque_id"]] for r in rows])
    pids = [r["patient_id"] for r in rows]

    scorer = SealedTestSet(BENCH / "seal_score_manifest.json",
                           BENCH / "access_log.jsonl", query_budget=20)
    if not (BENCH / "seal_score_manifest.json").exists():
        scorer.seal(sorted(r["opaque_id"] for r in rows), version="mias_v1")
    ids_sorted = sorted(r["opaque_id"] for r in rows)
    order = np.argsort([r["opaque_id"] for r in rows])
    headline = scorer.score(ids_sorted, y[order], scores[order],
                            caller=f"bench:{args.model}")

    a = clustered_bootstrap_ci(y, scores, pids, auroc, iterations=2000)
    s = clustered_bootstrap_ci(y, scores, pids, sensitivity_at_specificity, iterations=2000)
    cov = float(1 - abst.mean())
    nonabst = ~abst
    report = {
        "model": args.model, "n": len(y), "prevalence": float(y.mean()),
        "sandboxed": result.sandboxed,
        "determinism_max_delta": result.determinism_max_delta,
        "auroc": a[0], "auroc_ci95": [a[1], a[2]],
        "sens_at_spec96": s[0], "sens_at_spec96_ci95": [s[1], s[2]],
        "ece_adaptive": float(ece_adaptive(y, scores)),
        "coverage": cov,
        "auroc_nonabstained": float(auroc(y[nonabst], scores[nonabst]))
                              if nonabst.sum() > 10 else None,
        "headline": headline,
    }
    out = Path(f"results/bench_mias_v1/{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
