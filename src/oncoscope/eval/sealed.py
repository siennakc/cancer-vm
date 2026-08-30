"""Hash-sealed test set with query budget and access accounting (T-1.3, A6).

Adaptive overfitting needs no gradient access — repeatedly querying a holdout
overfits it through human choices alone. The sealed set is therefore reachable
only through this scoring service: membership is hash-locked, every scoring
call is logged, a pre-registered query budget is enforced, and responses are
aggregate metrics only (never per-case results).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import auroc, ece_adaptive, sensitivity_at_specificity


class QueryBudgetExhausted(RuntimeError):
    pass


class SealedTestSet:
    def __init__(
        self,
        manifest_path: str | Path,
        access_log_path: str | Path,
        query_budget: int = 50,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.access_log_path = Path(access_log_path)
        self.query_budget = query_budget

    # -- sealing ---------------------------------------------------------
    def seal(self, case_ids: list[str], version: str = "v1") -> str:
        ids = sorted(case_ids)
        sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {"version": version, "sha256": sha, "case_ids": ids},
                indent=1,
            )
        )
        return sha

    def verify(self, case_ids: list[str]) -> None:
        raw = json.loads(self.manifest_path.read_text())
        sha = hashlib.sha256(json.dumps(sorted(case_ids)).encode()).hexdigest()
        if sha != raw["sha256"]:
            raise ValueError(
                "case set does not match the sealed manifest — refusing to score"
            )

    # -- accounting ------------------------------------------------------
    def _queries_spent(self) -> int:
        if not self.access_log_path.exists():
            return 0
        return sum(1 for _ in self.access_log_path.open())

    def _log_access(self, caller: str, gold_version: str) -> None:
        self.access_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.access_log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        "caller": caller,
                        "gold_version": gold_version,
                    }
                )
                + "\n"
            )

    # -- the only scoring path ------------------------------------------
    def score(
        self,
        case_ids: list[str],
        y_true: np.ndarray,
        scores: np.ndarray,
        caller: str,
        specificity: float = 0.96,
        gold_version: str = "v1",
    ) -> dict[str, float]:
        """Aggregate metrics only. One budget unit per call. Every call logged."""
        self.verify(case_ids)
        if self._queries_spent() >= self.query_budget:
            raise QueryBudgetExhausted(
                f"sealed-set query budget of {self.query_budget} exhausted; "
                "further access is a governance decision, not an API call"
            )
        self._log_access(caller, gold_version)
        return {
            "auroc": auroc(y_true, scores),
            "sensitivity_at_specificity": sensitivity_at_specificity(
                y_true, scores, specificity
            ),
            "ece_adaptive": ece_adaptive(y_true, scores),
            "n_cases": float(len(case_ids)),
            "queries_remaining": float(self.query_budget - self._queries_spent()),
        }
