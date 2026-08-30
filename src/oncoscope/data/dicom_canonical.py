"""Canonical DICOM loading (T-1.1).

One shared loader for training, serving, and the harness. Preprocessing skew
between training and serving is the largest silent accuracy killer, so nothing
else in the codebase is allowed to touch pydicom directly.

Handles the correctness traps from the tasksheet pitfall register:
- MONOCHROME1 photometric interpretation (inverted grayscale)
- RescaleSlope / RescaleIntercept
- VOI LUT / windowing
- PixelSpacing propagation (sizes are computed in mm, never in pixels)
- Laterality tags
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels import apply_voi_lut


@dataclass
class CanonicalImage:
    """A DICOM rendered to a canonical float32 array plus the facts tools need."""

    pixels: np.ndarray                 # float32, [0, 1], MONOCHROME2 orientation
    pixel_spacing_mm: tuple[float, float] | None
    laterality: str | None             # "L" | "R" | None
    view: str | None                   # e.g. "CC" | "MLO"
    modality: str | None
    patient_id: str | None
    study_uid: str | None
    sop_uid: str | None
    meta: dict = field(default_factory=dict)


def _pixel_spacing(ds: Dataset) -> tuple[float, float] | None:
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(ds, tag, None)
        if value is not None and len(value) == 2:
            return (float(value[0]), float(value[1]))
    return None


def load_canonical(path_or_dataset: str | Dataset) -> CanonicalImage:
    """Load a DICOM file (or in-memory dataset) into canonical form."""
    ds = (
        path_or_dataset
        if isinstance(path_or_dataset, Dataset)
        else pydicom.dcmread(path_or_dataset)
    )

    arr = ds.pixel_array.astype(np.float64)

    # Modality LUT: rescale slope/intercept (CT HU values depend on this).
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # VOI LUT / window when present.
    if getattr(ds, "VOILUTSequence", None) or getattr(ds, "WindowCenter", None) is not None:
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float64)
        except Exception:
            pass  # fall through to min-max normalization on the rescaled values

    # MONOCHROME1 means low pixel value = bright; canonicalize to MONOCHROME2.
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(arr.min()), float(arr.max())
    pixels = ((arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)).astype(np.float32)

    return CanonicalImage(
        pixels=pixels,
        pixel_spacing_mm=_pixel_spacing(ds),
        laterality=getattr(ds, "ImageLaterality", None) or getattr(ds, "Laterality", None),
        view=getattr(ds, "ViewPosition", None),
        modality=getattr(ds, "Modality", None),
        patient_id=str(ds.PatientID) if getattr(ds, "PatientID", None) else None,
        study_uid=str(ds.StudyInstanceUID) if getattr(ds, "StudyInstanceUID", None) else None,
        sop_uid=str(ds.SOPInstanceUID) if getattr(ds, "SOPInstanceUID", None) else None,
    )
