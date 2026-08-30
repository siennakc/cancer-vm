"""De-identification with a tag allowlist (T-1.1, Part 5 injection surfaces).

Allowlist, not blocklist: any tag not explicitly listed is dropped, which
removes both PHI *and* the free-text fields that carry prompt injection
(private tags, SR content, overlays, unknown vendor text). The LLM never sees
raw header text either way — this gate protects downstream storage and any
human-facing export.

Burned-in pixel text (OCR redaction) is a Phase 6 gate; `preflight_qc` in the
harness flags suspiciously text-like regions until then.
"""

from __future__ import annotations

from pydicom.dataset import Dataset

# Tags a screening-detection pipeline actually needs. Everything else is dropped.
ALLOWED_KEYWORDS: frozenset[str] = frozenset(
    {
        # Identity is replaced, not carried
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PixelSpacing",
        "ImagerPixelSpacing",
        "RescaleSlope",
        "RescaleIntercept",
        "WindowCenter",
        "WindowWidth",
        "VOILUTFunction",
        "ImageLaterality",
        "Laterality",
        "ViewPosition",
        "PatientSex",
        "PixelData",
    }
)


def deidentify(ds: Dataset, pseudo_patient_id: str) -> Dataset:
    """Return a new dataset containing only allowlisted tags.

    ``pseudo_patient_id`` replaces PatientID; the mapping lives in a separated
    crosswalk vault outside the repository, never beside the images.
    """
    out = Dataset()
    for elem in ds:
        if elem.tag.is_private:
            continue
        if elem.keyword in ALLOWED_KEYWORDS:
            out.add(elem)
    out.PatientID = pseudo_patient_id
    out.PatientName = ""
    # Preserve pydicom file meta needs
    out.file_meta = getattr(ds, "file_meta", None)
    if hasattr(ds, "is_little_endian"):
        out.is_little_endian = ds.is_little_endian
        out.is_implicit_VR = ds.is_implicit_VR
    return out
