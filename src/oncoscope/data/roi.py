"""CBIS-DDSM ROI annotation table: join abnormality rows to mask pixels.

The patch-pretraining stage (Shen et al. 2019 — the missing core of the
published 0.88 recipe, and the localizing detector the harness A/B diagnosed
as absent) needs, per abnormality: the full image it lives in, its binary ROI
mask, and a 5-class patch label.

Two CBIS-specific traps this module exists to absorb:

1. **The crop and the mask usually share one series — but not always.**
   3,464 of 3,568 abnormality rows name the SAME series UID in the "cropped
   image file path" and "ROI mask file path" columns (two instance files);
   the other 104 (2.9%) keep them in two single-file series. And within a
   shared series the CSV's file-level assignment is ~50/50 arbitrary
   (mask at ``000000.dcm`` in 1,871 rows, ``000001.dcm`` in 1,593). Files
   are therefore gathered from BOTH columns' series and classified by
   CONTENT: the mask is the (near-)binary raster; the crop is a grayscale
   cutout. Filenames, file order, and series descriptions are never trusted.
2. **Mask dimensions do not always match the full image.** Known data-quality
   issue. A small mismatch is recorded and resampled at patch time (nearest);
   a gross mismatch marks the record unusable rather than guessing geometry.
   When several binary rasters compete, agreement with the full image's dims
   breaks the tie — "largest wins" would pick a saturated crop over a
   reduced-dims mask, inverting the roles exactly on the defective records.
   If the full image's dims cannot be established at all, the record is
   marked unusable (fail closed) rather than skipping the dims guard.

CMMD has no lesion annotations, so the patch stage is DDSM-only by design.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom

from .mammography import _CBIS_BENIGN, _CBIS_MALIGNANT, _series_uid_from_cbis_path

# 5-class patch taxonomy (Shen et al.): background + abnormality x pathology.
PATCH_CLASSES = (
    "background",
    "benign_calcification",
    "malignant_calcification",
    "benign_mass",
    "malignant_mass",
)


def patch_class(abnormality: str, label: int) -> int:
    base = 1 if abnormality == "calcification" else 3
    return base + int(label)


@dataclass(frozen=True)
class RoiRecord:
    """One abnormality with a usable (or explicitly unusable) mask."""

    roi_id: str                # "<case_id>#<abnormality id>"
    case_id: str               # ddsm-<full image series uid>
    patient_id: str
    abnormality: str           # mass | calcification
    label: int | None          # 1 = biopsy-proven malignant; None = sentinel row
    cls: int | None            # index into PATCH_CLASSES; None = sentinel row
    mask_relpath: str | None   # under data/raw
    crop_relpath: str | None
    image_shape: tuple[int, int] | None
    mask_shape: tuple[int, int] | None
    status: str                # ok | mask_dims_mismatch | <error reason>

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "mask_dims_mismatch")


def _dims_from_header(path: Path) -> tuple[int, int] | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return int(ds.Rows), int(ds.Columns)
    except Exception:
        return None


def _is_binary(arr: np.ndarray) -> bool:
    """A mask raster has (at most) two levels; real tissue has many.

    Sampled uniques keep this cheap on multi-megapixel masks. A true binary
    mask can never fail this test (any sample of a 2-valued array has <= 2
    uniques); the residual risk is a near-uniform CROP passing it, which the
    dims-aware tie-break in ``classify_roi_files`` exists to absorb.
    """
    flat = arr.reshape(-1)
    sample = flat[:: max(1, flat.size // 10_000)]
    return np.unique(sample).size <= 2


def classify_roi_files(
    dcm_paths: list[Path],
    image_shape: tuple[int, int] | None = None,
) -> tuple[Path | None, Path | None, str]:
    """(mask_path, crop_path, status) by pixel content, never by filename.

    Ties between multiple binary candidates (rare; usually a saturated crop
    sneaking past the sampled-uniques test) resolve in layers:

    1. a CONSTANT raster (min == max over every pixel) is never a mask — a
       mask with zero foreground marks nothing; this alone disposes of truly
       saturated crops, and the full scan is cheap because ties are rare;
    2. with ``image_shape`` known, closest ASPECT RATIO to the image wins —
       scale-invariant, so it stays correct for the defective records whose
       real mask is stored at reduced dims (reduction preserves aspect;
       cutouts have arbitrary aspect);
    3. otherwise, larger raster wins (the residual heuristic).
    """
    decoded: list[tuple[Path, np.ndarray]] = []
    for p in sorted(dcm_paths):
        try:
            decoded.append((p, pydicom.dcmread(p).pixel_array))
        except Exception as exc:
            return None, None, f"undecodable: {p.name}: {exc}"
    if not decoded:
        return None, None, "empty series"

    binary = [(p, a) for p, a in decoded if _is_binary(a)]
    gray = [(p, a) for p, a in decoded if not _is_binary(a)]
    if len(binary) == 1:
        crop = gray[0][0] if gray else None
        return binary[0][0], crop, "ok"
    if len(binary) > 1:
        pool = [(p, a) for p, a in binary if a.min() != a.max()] or binary
        if image_shape is not None:
            img_aspect = image_shape[0] / image_shape[1]
            pool.sort(key=lambda pa: (
                abs(pa[1].shape[0] / pa[1].shape[1] - img_aspect), -pa[1].size))
        else:
            pool.sort(key=lambda pa: -pa[1].size)
        chosen = pool[0][0]
        others = [p for p, _ in binary if p != chosen]
        crop = gray[0][0] if gray else (others[0] if others else None)
        return chosen, crop, "ok"
    return None, None, "no binary raster in ROI series"


def build_roi_table(
    metadata_dir: Path | str,
    roi_manifest_path: Path | str,
    raw_root: Path | str,
    known_case_ids: set[str] | None = None,
    max_dims_ratio: float = 1.25,
    full_manifest_path: Path | str = "data/metadata/manifest_cbis.jsonl",
) -> list[RoiRecord]:
    """Join the four case CSVs to downloaded ROI series.

    ``known_case_ids`` (from cases_v1.jsonl) restricts output to images that
    exist in the case table, so patches can never be sampled for an image the
    label audit dropped.
    """
    import csv

    from .tcia import Manifest

    metadata_dir, raw_root = Path(metadata_dir), Path(raw_root)
    manifest = Manifest.load(roi_manifest_path)
    # Full-image locations come from the FULL manifest's relpath — the
    # authoritative record of where download_series put the bytes — never
    # from an assumed directory convention.
    full_manifest = Manifest.load(full_manifest_path)
    records: list[RoiRecord] = []

    for csv_name in (
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ):
        csv_path = metadata_dir / csv_name
        if not csv_path.exists():
            continue
        abnormality = "mass" if csv_name.startswith("mass") else "calcification"
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                image_uid = _series_uid_from_cbis_path(row.get("image file path", ""))
                mask_uid = _series_uid_from_cbis_path(row.get("ROI mask file path", ""))
                crop_uid = _series_uid_from_cbis_path(row.get("cropped image file path", ""))
                if not image_uid:
                    continue  # cannot even attribute the row to an image
                case_id = f"ddsm-{image_uid}"
                if known_case_ids is not None and case_id not in known_case_ids:
                    continue
                roi_id = f"{case_id}#{(row.get('abnormality id') or '?').strip()}"
                who = (row.get("patient_id") or "").strip()

                pathology = (row.get("pathology") or "").strip().upper()
                label = (1 if pathology in _CBIS_MALIGNANT
                         else 0 if pathology in _CBIS_BENIGN else None)
                # A row of a KNOWN image that cannot be resolved (unparseable
                # pathology, no mask column) still marks a lesion somewhere on
                # that image. Dropping it silently would let background patches
                # land on it, so a sentinel record keeps the image's exclusion
                # completeness honestly unprovable (the whole image is skipped).
                # Zero such rows exist in the committed CSVs — this is armor
                # against a future CSV revision, not a live condition.
                if label is None or not mask_uid:
                    records.append(RoiRecord(
                        roi_id, case_id, who, abnormality, label, None,
                        None, None, None, None,
                        "unresolvable row — image excluded from sampling"))
                    continue
                cls = patch_class(abnormality, label)

                # 104 of 3,568 rows keep mask and crop in two single-file
                # series, and the CSV's series-level assignment is as
                # untrustworthy as its file-level one — so gather files from
                # BOTH columns' series and let content classification decide.
                dcm_paths: list[Path] = []
                missing_uids: list[str] = []
                for uid in dict.fromkeys((mask_uid, crop_uid)):
                    if not uid:
                        continue
                    entry = manifest.entries.get(uid)
                    if entry is None or "error" in entry:
                        missing_uids.append(uid)
                        continue
                    dcm_paths += sorted((raw_root / entry["relpath"]).rglob("*.dcm"))
                if not dcm_paths:
                    records.append(RoiRecord(
                        roi_id, case_id, who, abnormality, label, cls,
                        None, None, None, None, "roi series not downloaded"))
                    continue

                # Image dims first: they inform the mask tie-break, and their
                # absence must fail the record closed, never skip the guard.
                full_entry = full_manifest.entries.get(image_uid)
                full_dcms = (
                    sorted((raw_root / full_entry["relpath"]).rglob("*.dcm"))
                    if full_entry and "error" not in full_entry else []
                )
                image_shape = _dims_from_header(full_dcms[0]) if full_dcms else None

                mask_path, crop_path, status = classify_roi_files(
                    dcm_paths, image_shape=image_shape)
                if status == "no binary raster in ROI series" and missing_uids:
                    status = "roi series partially downloaded — mask missing"

                mask_shape = None
                if status == "ok" and mask_path is not None:
                    mask_shape = _dims_from_header(mask_path)
                    if image_shape is None:
                        status = "full image dims unavailable — unusable"
                    elif mask_shape is None:
                        status = "mask dims unreadable — unusable"
                    elif image_shape != mask_shape:
                        ratio = max(image_shape[0] / mask_shape[0],
                                    mask_shape[0] / image_shape[0],
                                    image_shape[1] / mask_shape[1],
                                    mask_shape[1] / image_shape[1])
                        status = ("mask_dims_mismatch" if ratio <= max_dims_ratio
                                  else f"mask dims off by {ratio:.2f}x — unusable")

                records.append(RoiRecord(
                    roi_id, case_id, who, abnormality, label, cls,
                    str(mask_path.relative_to(raw_root)) if mask_path else None,
                    str(crop_path.relative_to(raw_root)) if crop_path else None,
                    image_shape, mask_shape, status))
    return records


def write_roi_table(records: list[RoiRecord], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")


def read_roi_table(path: Path | str) -> list[RoiRecord]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            for key in ("image_shape", "mask_shape"):
                if row.get(key) is not None:
                    row[key] = tuple(row[key])
            out.append(RoiRecord(**row))
    return out
