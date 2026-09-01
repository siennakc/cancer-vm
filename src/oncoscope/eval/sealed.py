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


class SealedProvenanceError(RuntimeError):
    """The candidate's fitting data touches the sealed patients."""


FITTING_SPLITS = ("train", "calibration", "threshold", "slice_discovery")


class SealedTestSet:
    def __init__(
        self,
        manifest_path: str | Path,
        access_log_path: str | Path,
        query_budget: int = 50,
        case_table_path: str | Path = "data/processed/cases_v1.jsonl",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.access_log_path = Path(access_log_path)
        self.query_budget = query_budget
        self.case_table_path = Path(case_table_path)

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

    def verify_provenance(self, fit_manifest_path: str | Path) -> None:
        """Refuse to score a model whose fitting splits touch sealed patients.

        Membership hashing alone cannot catch this: after a re-split under a
        new seed (splits_v1 -> splits_v2), 381 of the 499 v1-sealed patients
        landed in v2 fitting splits, so a v3/v4 model could be scored here and
        pass every membership check while having trained on 76% of the sealed
        set. This check derives the sealed PATIENTS from the case table and
        rejects any fit manifest that assigns one of them to a fitting split.
        """
        from ..data.splits import load_manifest  # hash-verifies the manifest

        sealed_ids = set(json.loads(self.manifest_path.read_text())["case_ids"])
        if not self.case_table_path.exists():
            raise SealedProvenanceError(
                f"case table {self.case_table_path} not found — cannot map sealed "
                "cases to patients, refusing to score"
            )
        patients: set[str] = set()
        unmapped = set(sealed_ids)
        with self.case_table_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                if row["case_id"] in sealed_ids:
                    patients.add(f"{row['site']}/{row['patient_id']}")
                    unmapped.discard(row["case_id"])
        if unmapped:
            raise SealedProvenanceError(
                f"{len(unmapped)} sealed case(s) missing from the case table "
                f"(e.g. {sorted(unmapped)[:3]}) — refusing to score"
            )
        assignment = load_manifest(fit_manifest_path).assignment
        contaminated = sorted(
            p for p in patients if assignment.get(p) in FITTING_SPLITS
        )
        if contaminated:
            raise SealedProvenanceError(
                f"{len(contaminated)} of {len(patients)} sealed patients sit in a "
                f"fitting split of {fit_manifest_path} (e.g. {contaminated[:3]}). "
                "A model fit under that manifest is disqualified on this sealed set."
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
        fit_manifest: str | Path | list | None = None,
        external: bool = False,
    ) -> dict[str, float]:
        """Aggregate metrics only. One budget unit per call. Every call logged.

        Provenance is not optional: pass ``fit_manifest`` (the split manifest
        the candidate was fit under — checked against the sealed patients), or
        declare ``external=True`` for a benchmark whose cases cannot appear in
        any internal fitting split (e.g. MIAS). Silence is not an option.

        A warm-started model has MORE THAN ONE fit manifest — its own plus
        every entry in its checkpoint's ``init_lineage`` — and each one can
        independently contaminate the sealed set. Pass them all as a list;
        every manifest must be clean for scoring to proceed.
        """
        if not external:
            if fit_manifest is None:
                raise SealedProvenanceError(
                    "declare the candidate's fit manifest(s) (fit_manifest=..., "
                    "including every init_lineage manifest for a warm-started "
                    "model) or mark the benchmark external=True — scoring "
                    "without split provenance is how a sealed set gets quietly "
                    "burned"
                )
            manifests = (fit_manifest if isinstance(fit_manifest, (list, tuple))
                         else [fit_manifest])
            if not manifests:
                raise SealedProvenanceError("fit_manifest list is empty")
            for m in manifests:
                self.verify_provenance(m)
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
