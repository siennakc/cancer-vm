"""Deterministic outer state machine (T-4.2, T-4.3).

ingest -> preflight QC -> screen -> detect -> verify -> aggregate ->
adjudicate -> report

Control flow is code, never the model. The LLM sits only at the *adjudicate*
decision node, behind the ``Adjudicator`` protocol; a rule-based adjudicator
ships as the default so the entire pipeline runs, tests, and gates without an
LLM in the loop (and serves as the control arm in T-4.5 ablations).

Verification (T-4.3 inference stack v1) uses harness-orchestrated TTA:
label-preserving translations only (never laterality flips), inverted before
aggregation, with IoU clustering. Findings must reproduce in >= k of N reads;
the disagreement rate is a first-class output driving escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..models.features import embed_crop
from ..models.head import LogisticHead
from .schemas import (
    Adjudication,
    Assessment,
    CaseDecision,
    CaseReport,
    Finding,
    QCVerdict,
)
from .tools import Toolbelt


class Adjudicator(Protocol):
    def adjudicate(self, request: dict) -> Adjudication: ...


@dataclass
class RuleBasedAdjudicator:
    """Deterministic adjudication: thresholds + deferral band + abstention.

    Also the "detector-alone" arm of the T-4.5 ablation.
    """

    recall_threshold: float = 0.65
    deferral_band: tuple[float, float] = (0.35, 0.65)
    max_disagreement: float = 0.4

    def adjudicate(self, request: dict) -> Adjudication:
        score = float(request["consistency"]["case_score"])
        disagreement = float(request["consistency"]["disagreement_rate"])
        per_candidate: dict[str, Assessment] = {}
        for cand in request["candidates"]:
            if cand["reproduced_fraction"] >= 0.6:
                per_candidate[cand["candidate_id"]] = (
                    Assessment.present if cand["score"] >= 0.5 else Assessment.uncertain
                )
            else:
                per_candidate[cand["candidate_id"]] = Assessment.absent

        if request["qc"] == QCVerdict.inadequate_defer.value:
            decision = CaseDecision.defer_to_human
            rationale = "image quality inadequate for assessment"
        elif disagreement > self.max_disagreement:
            decision = CaseDecision.defer_to_human
            rationale = f"self-consistency disagreement {disagreement:.2f} above band"
        elif self.deferral_band[0] <= score <= self.deferral_band[1]:
            decision = CaseDecision.defer_to_human
            rationale = f"score {score:.2f} inside deferral band"
        elif score > self.deferral_band[1]:
            decision = CaseDecision.recall
            rationale = f"score {score:.2f} above recall threshold"
        else:
            decision = CaseDecision.no_recall
            rationale = f"score {score:.2f} below deferral band"
        return Adjudication(
            per_candidate=per_candidate,
            decision=decision,
            rationale=rationale,
            cited_evidence=[c.get("evidence_ref", "") for c in request["candidates"]],
        )


class HarnessPipeline:
    def __init__(
        self,
        toolbelt: Toolbelt,
        adjudicator: Adjudicator | None = None,
        head: LogisticHead | None = None,
        consistency_reads: int = 5,
        min_reproduced: int = 3,
    ) -> None:
        self.tools = toolbelt
        self.adjudicator = adjudicator or RuleBasedAdjudicator()
        self.head = head
        self.consistency_reads = consistency_reads
        self.min_reproduced = min_reproduced

    # -- stages ----------------------------------------------------------
    def preflight_qc(self, pixels: np.ndarray) -> QCVerdict:
        """Technical adequacy before any diagnostic reasoning (2G)."""
        dynamic_range = float(pixels.max() - pixels.min())
        saturated = float((pixels >= 0.999).mean())
        if dynamic_range < 0.05:
            return QCVerdict.inadequate_defer
        if saturated > 0.5:
            return QCVerdict.degraded
        return QCVerdict.adequate

    def _detect_with_shift(self, pixels: np.ndarray, dy: int, dx: int) -> list[dict]:
        """One TTA read: translate, detect, invert the transform on the boxes."""
        shifted = np.roll(np.roll(pixels, dy, axis=0), dx, axis=1)
        info = self.tools.store.put(shifted, kind="image", meta={"tta": [dy, dx]})
        result = self.tools.call("run_detector", image_handle=info.handle)
        out = []
        for cand in result["candidates"]:
            x0, y0, x1, y1 = cand["box"]
            out.append(
                {
                    "box": [x0 - dx, y0 - dy, x1 - dx, y1 - dy],
                    "score": cand["score"],
                    "evidence_ref": result["evidence_ref"],
                }
            )
        return out

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def verify(self, pixels: np.ndarray, base_candidates: list[dict]) -> dict:
        """Self-consistency across TTA reads; IoU-cluster and count reproduction."""
        shifts = [(0, 0), (3, 0), (0, 3), (-3, 0), (0, -3)][: self.consistency_reads]
        reads = [self._detect_with_shift(pixels, dy, dx) for dy, dx in shifts]
        verified = []
        for i, cand in enumerate(base_candidates):
            reproduced = sum(
                1
                for read in reads
                if any(self._iou(cand["box"], other["box"]) >= 0.3 for other in read)
            )
            verified.append(
                {
                    "candidate_id": cand.get("candidate_id", f"c{i}"),
                    "box": cand["box"],
                    "score": cand["score"],
                    "evidence_ref": cand.get("evidence_ref", ""),
                    "reproduced": reproduced,
                    "reproduced_fraction": reproduced / len(reads),
                    "kept": reproduced >= self.min_reproduced,
                }
            )
        per_read_positive = [
            any(c["score"] >= 0.5 for c in read) for read in reads
        ]
        disagreement = float(np.mean(per_read_positive) * (1 - np.mean(per_read_positive)) * 4)
        return {"verified": verified, "disagreement_rate": round(disagreement, 4)}

    def aggregate(self, pixels: np.ndarray, verified: list[dict]) -> float:
        """Case-level suspicion score in [0,1]; calibrated head when available."""
        kept = [c for c in verified if c["kept"]]
        detector_score = max((c["score"] for c in kept), default=0.0)
        if self.head is not None:
            emb = embed_crop(pixels).reshape(1, -1)
            head_score = float(self.head.predict_proba(emb)[0])
            return round(0.5 * detector_score + 0.5 * head_score, 4)
        return round(float(detector_score), 4)

    # -- the whole case --------------------------------------------------
    def run_case(self, case_id: str, pixels: np.ndarray, pixel_spacing_mm=(0.1, 0.1)) -> CaseReport:
        info = self.tools.store.put(pixels, kind="image", meta={"case_id": case_id})
        qc = self.preflight_qc(pixels)
        if qc == QCVerdict.inadequate_defer:
            self.tools.call(
                "submit_review", case_id=case_id, reason="failed preflight QC", ranked_regions=[]
            )
            return CaseReport(
                case_id=case_id, qc=qc, findings=[], decision=CaseDecision.defer_to_human,
                score=0.5, disagreement_rate=1.0, deferral_reason="preflight QC",
            )

        detect = self.tools.call("run_detector", image_handle=info.handle)
        for i, cand in enumerate(detect["candidates"]):
            cand["evidence_ref"] = detect["evidence_ref"]
        consistency = self.verify(pixels, detect["candidates"])
        case_score = self.aggregate(pixels, consistency["verified"])

        request = {
            "case_id": case_id,
            "qc": qc.value,
            "candidates": consistency["verified"],
            "consistency": {
                "case_score": case_score,
                "disagreement_rate": consistency["disagreement_rate"],
            },
            "atlas_neighbors": [],
            "guideline_notes": [],
        }
        adjudication = self.adjudicator.adjudicate(request)
        self.tools.ledger.append("decision", adjudication.model_dump(mode="json"))

        findings = []
        for cand in consistency["verified"]:
            if not cand["kept"]:
                continue
            measure = self.tools.call(
                "measure",
                image_handle=info.handle,
                box=cand["box"],
                pixel_spacing_mm=list(pixel_spacing_mm),
            )
            findings.append(
                Finding(
                    finding_id=cand["candidate_id"],
                    box=tuple(int(v) for v in cand["box"]),
                    size_mm=measure["long_axis_mm"],
                    detector_score=cand["score"],
                    assessment=adjudication.per_candidate.get(
                        cand["candidate_id"], Assessment.uncertain
                    ),
                    evidence_refs=[cand["evidence_ref"], measure["evidence_ref"]],
                )
            )

        if adjudication.decision == CaseDecision.defer_to_human:
            self.tools.call(
                "submit_review",
                case_id=case_id,
                reason=adjudication.rationale,
                ranked_regions=[
                    {"box": f.box, "size_mm": f.size_mm, "score": f.detector_score}
                    for f in findings
                ],
            )

        report = CaseReport(
            case_id=case_id,
            qc=qc,
            findings=findings,
            decision=adjudication.decision,
            score=case_score,
            disagreement_rate=consistency["disagreement_rate"],
            deferral_reason=(
                adjudication.rationale
                if adjudication.decision == CaseDecision.defer_to_human
                else ""
            ),
            evidence_refs=[detect["evidence_ref"]],
        )
        self.tools.ledger.append("claim", report.model_dump(mode="json"))
        return report
