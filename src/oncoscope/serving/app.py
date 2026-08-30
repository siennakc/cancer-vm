"""FastAPI wrapper (T-2.3). Imports the SAME preprocessing as training.

Preprocessing skew between training and serving is the largest silent
accuracy killer; this module is deliberately thin and owns no image logic.
Install with ``pip install -e '.[serving]'``.
"""

from __future__ import annotations

import tempfile

import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install serving extras: pip install -e '.[serving]'") from exc

from ..data.dicom_canonical import load_canonical
from ..harness.ledger import EvidenceLedger
from ..harness.state_machine import HarnessPipeline
from ..harness.store import ArtifactStore
from ..harness.tools import Toolbelt

app = FastAPI(title="oncoscope", version="0.1.0")

_workdir = tempfile.mkdtemp(prefix="oncoscope_serve_")
_pipeline = HarnessPipeline(
    Toolbelt(ArtifactStore(f"{_workdir}/artifacts"), EvidenceLedger(f"{_workdir}/ledger.jsonl"))
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".dcm") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        canonical = load_canonical(tmp.name)
    report = _pipeline.run_case(
        case_id=canonical.sop_uid or "unknown",
        pixels=np.asarray(canonical.pixels),
        pixel_spacing_mm=canonical.pixel_spacing_mm or (0.1, 0.1),
    )
    return report.model_dump(mode="json")
