"""Deterministic tools (T-4.1). The only place work happens.

The LLM plans and adjudicates; these tools perceive, measure, and retrieve.
Every call writes a ``tool_call`` + ``tool_result`` pair into the evidence
ledger, and every numeric fact downstream must cite one of those entries.

Registry is deny-by-default: only tools registered here are callable, whether
invoked by the deterministic state machine or exposed to the LLM over MCP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..models.detector import DoGBlobDetector
from ..models.features import embed_crop
from .ledger import EvidenceLedger
from .store import ArtifactStore


class Toolbelt:
    """All tools bound to one case-processing context (store + ledger)."""

    def __init__(
        self,
        store: ArtifactStore,
        ledger: EvidenceLedger,
        atlas_path: str | Path | None = None,
        criteria_path: str | Path | None = None,
        detector: DoGBlobDetector | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.detector = detector or DoGBlobDetector()
        self.atlas_path = Path(atlas_path) if atlas_path else None
        self.criteria_path = Path(criteria_path) if criteria_path else None
        self._registry: dict[str, Callable[..., dict]] = {
            "describe_store": self.describe_store,
            "run_detector": self.run_detector,
            "crop_region": self.crop_region,
            "measure": self.measure,
            "retrieve_similar": self.retrieve_similar,
            "lookup_criteria": self.lookup_criteria,
            "submit_review": self.submit_review,
        }

    # -- plumbing --------------------------------------------------------
    def call(self, tool_name: str, **kwargs: Any) -> dict:
        """Single audited entry point. Unknown tools are refused loudly."""
        if tool_name not in self._registry:
            raise PermissionError(f"tool {tool_name!r} is not in the allowlist")
        call_ref = self.ledger.append("tool_call", {"tool": tool_name, "args": _safe(kwargs)})
        result = self._registry[tool_name](**kwargs)
        result_ref = self.ledger.append(
            "tool_result", {"tool": tool_name, "call_ref": call_ref, "result": _safe(result)}
        )
        result["evidence_ref"] = result_ref
        return result

    # -- tools -----------------------------------------------------------
    def describe_store(self) -> dict:
        return {"artifacts": self.store.describe()}

    def run_detector(self, image_handle: str) -> dict:
        pixels = self.store.get(image_handle)
        candidates = self.detector.propose(pixels)
        return {
            "image_handle": image_handle,
            "candidates": [
                {"candidate_id": f"c{i}", "box": list(c.box), "score": c.score}
                for i, c in enumerate(candidates)
            ],
        }

    def crop_region(self, image_handle: str, box: list[int]) -> dict:
        pixels = self.store.get(image_handle)
        x0, y0, x1, y1 = (int(v) for v in box)
        h, w = pixels.shape
        x0, x1 = max(0, x0), min(w, x1 + 1)
        y0, y1 = max(0, y0), min(h, y1 + 1)
        crop = pixels[y0:y1, x0:x1]
        info = self.store.put(crop, kind="crop", meta={"source": image_handle, "box": [x0, y0, x1, y1]})
        return {"crop_handle": info.handle, "shape": list(crop.shape)}

    def measure(self, image_handle: str, box: list[int], pixel_spacing_mm: list[float]) -> dict:
        """All sizes in mm are computed here, never authored by the LLM."""
        x0, y0, x1, y1 = (int(v) for v in box)
        sy, sx = float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1])
        width_mm = abs(x1 - x0) * sx
        height_mm = abs(y1 - y0) * sy
        pixels = self.store.get(image_handle)
        region = pixels[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        return {
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "long_axis_mm": round(max(width_mm, height_mm), 2),
            "mean_intensity": round(float(region.mean()) if region.size else 0.0, 4),
            "contrast": round(
                float(region.mean() - pixels.mean()) if region.size else 0.0, 4
            ),
        }

    def retrieve_similar(self, crop_handle: str, k: int = 5) -> dict:
        """kNN case atlas (2F): distance-aware — no neighbors means abstain."""
        if self.atlas_path is None or not self.atlas_path.exists():
            return {"neighbors": [], "note": "atlas empty — treat as no-neighbor abstain signal"}
        atlas = json.loads(self.atlas_path.read_text())
        q = embed_crop(self.store.get(crop_handle))
        scored = []
        for entry in atlas["entries"]:
            d = float(np.linalg.norm(q - np.array(entry["embedding"])))
            scored.append({"case_id": entry["case_id"], "label": entry["label"], "distance": round(d, 4)})
        scored.sort(key=lambda e: e["distance"])
        return {"neighbors": scored[:k]}

    def lookup_criteria(self, topic: str) -> dict:
        """Guidelines applied from a version-pinned local corpus, never weights."""
        if self.criteria_path is None or not self.criteria_path.exists():
            return {"topic": topic, "entries": [], "corpus_version": None}
        corpus = json.loads(self.criteria_path.read_text())
        entries = [e for e in corpus.get("entries", []) if topic.lower() in e["topic"].lower()]
        return {"topic": topic, "entries": entries, "corpus_version": corpus.get("version")}

    def submit_review(self, case_id: str, reason: str, ranked_regions: list[dict]) -> dict:
        """Structured human handoff (2G): regions + measurements, not a bare flag."""
        self.ledger.append(
            "decision",
            {"action": "defer_to_human", "case_id": case_id, "reason": reason,
             "ranked_regions": ranked_regions},
        )
        return {"queued": True, "case_id": case_id}


def _safe(obj: Any) -> Any:
    """Ledger payloads are JSON: arrays and numpy scalars are summarized, not embedded."""
    if isinstance(obj, np.ndarray):
        return {"__array__": list(obj.shape)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj
