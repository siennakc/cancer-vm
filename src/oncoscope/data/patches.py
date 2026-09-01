"""Patch sampling in render space (the Shen et al. patch-pretraining stage).

Patches are sampled from the SAME rendered geometry the whole-image model
sees (breast_crop -> letterbox onto an HxW canvas), so patch statistics match
whole-image training and the backbone transfers. The ROI mask rides through
the IDENTICAL transform, with two rules that keep coordinates honest:

- the crop box comes from the IMAGE (``encoder.crop_box``), never from the
  mask's own pixels;
- the mask is resampled nearest-neighbor (a mask has no fractional membership).

Everything here is numpy-only so the sampler is testable and CI-runnable
without torch; only the training script needs a GPU stack.

**Split discipline**: patches may be sampled only from patients in an allowed
fitting split. ``public_bench``, ``test``, ``threshold``, and
``slice_discovery`` patients are refused with :class:`SplitViolation` — a
patch dataset that silently touched the benchmark would poison every number
downstream of the patch model (axioms A6/A9). The enforcement point is
``select_sampleable_images`` (which also refuses images whose background
exclusion cannot be proven complete); the raw sampling functions below carry
no patient identity, so any NEW consumer of them must route image selection
through that helper rather than around it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.encoder import crop_box
from .splits import SplitManifest

DEFAULT_ALLOWED_SPLITS = ("train", "calibration")


class SplitViolation(RuntimeError):
    """A patch was requested for a patient outside the allowed fitting splits."""


def assert_sampleable(patient_key: str, splits: SplitManifest,
                      allowed=DEFAULT_ALLOWED_SPLITS) -> str:
    split = splits.assignment.get(patient_key)
    if split not in allowed:
        raise SplitViolation(
            f"{patient_key} is in split {split!r} — patch sampling is allowed "
            f"only from {allowed}. Refusing rather than leaking."
        )
    return split


def select_sampleable_images(
    records,
    cases_by_id: dict,
    splits: SplitManifest,
    allowed=DEFAULT_ALLOWED_SPLITS,
) -> tuple[dict, dict]:
    """Group ROI records by image, admitting only images that are BOTH
    split-clean and exclusion-complete.

    Two refusal rules, each load-bearing:

    - **Split discipline**: only :class:`SplitViolation` is treated as a
      legitimate skip (counted per split). Any other exception — a corrupted
      manifest, a key-format drift — propagates: a broken guard must halt the
      build, not masquerade as a quarantine.
    - **Exclusion completeness**: an image with ANY unusable ROI record is
      dropped wholesale. Sampling its usable lesions would be fine, but its
      background patches could land on the lesion whose mask we could not
      read — a possibly malignant region labeled class 0 (the multi-finding
      trap). No provable exclusion union, no patches.

    Returns (case_id -> [usable RoiRecords], stats).
    """
    stats = {"split_skips": {}, "incomplete_images": 0, "unknown_case": 0,
             "selected": 0}
    grouped: dict[str, list] = {}
    for r in records:
        if r.case_id not in cases_by_id:
            stats["unknown_case"] += 1
            continue
        grouped.setdefault(r.case_id, []).append(r)

    selected: dict[str, list] = {}
    for case_id, rois in sorted(grouped.items()):
        case = cases_by_id[case_id]
        key = f"{case.site}/{case.patient_id}"
        try:
            assert_sampleable(key, splits, allowed)
        except SplitViolation:
            name = splits.assignment.get(key) or "unassigned"
            stats["split_skips"][name] = stats["split_skips"].get(name, 0) + 1
            continue
        if any(not r.usable for r in rois):
            stats["incomplete_images"] += 1
            continue
        selected[case_id] = rois
    stats["selected"] = len(selected)
    return selected, stats


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_shard_provenance(
    shard_dir: Path | str,
    splits: SplitManifest,
    allowed=DEFAULT_ALLOWED_SPLITS,
    allow_tainted: bool = False,
) -> dict:
    """Load-time gate for the patch shards: the LAST check before gradients.

    The build-time guards can be routed around (a smoke-test flag, a stale
    shard from an earlier manifest, a resumed run, a hand-edited file), so
    training re-derives the facts instead of trusting the directory:

    1. the build report must exist and must not be tainted by
       ``--allow-unquarantined`` (unless ``allow_tainted`` — smoke tests only);
    2. the report's ``splits_sha256`` must equal the manifest the caller holds;
    3. every shard file must hash to what the report recorded (a shard the
       report does not list, or a listed shard whose bytes changed, refuses);
    4. every patch's patient in every meta file must sit in EXACTLY the split
       whose shard it is in, under this manifest.

    Returns the verified build report. Raises :class:`SplitViolation` on any
    provenance failure — same class as the sampling guard, because it is the
    same crime at a later hour.
    """
    shard_dir = Path(shard_dir)
    report_path = shard_dir / "build_report.json"
    if not report_path.exists():
        raise SplitViolation(f"{report_path} missing — unbuilt or interrupted build")
    report = json.loads(report_path.read_text())

    if report.get("config", {}).get("allow_unquarantined") and not allow_tainted:
        raise SplitViolation(
            "shards were built with --allow-unquarantined (smoke-test taint); "
            "refusing to train a real model on them"
        )
    if report.get("splits_sha256") != splits.sha256:
        raise SplitViolation(
            f"shards built under splits sha {str(report.get('splits_sha256'))[:12]}… "
            f"but training holds {splits.sha256[:12]}… — rebuild the shards"
        )

    counts = report.get("counts", {})
    for split in allowed:
        patches_path = shard_dir / f"patches_{split}.npy"
        meta_path = shard_dir / f"meta_{split}.json"
        entry = counts.get(split)
        if entry is None:
            if patches_path.exists() or meta_path.exists():
                raise SplitViolation(
                    f"stale shard files for split {split!r} not covered by the "
                    "build report — clean the shard directory and rebuild"
                )
            continue
        for path, key in ((patches_path, "patches_sha256"), (meta_path, "meta_sha256")):
            if not path.exists():
                raise SplitViolation(f"{path} listed in the build report but missing")
            recorded = entry.get(key)
            if recorded is None or file_sha256(path) != recorded:
                raise SplitViolation(
                    f"{path} does not match the build report's {key} — stale or "
                    "edited shard"
                )
        meta = json.loads(meta_path.read_text())
        wrong = {m["patient"] for m in meta
                 if splits.assignment.get(m["patient"]) != split}
        if wrong:
            raise SplitViolation(
                f"{len(wrong)} patient(s) in the {split} shard are not in split "
                f"{split!r} of this manifest (e.g. {sorted(wrong)[:3]})"
            )
    return report


# --- geometry -------------------------------------------------------------

@dataclass(frozen=True)
class RenderGeometry:
    """The exact breast_crop + letterbox transform for one image."""

    box: tuple[int, int, int, int]      # r0, r1, c0, c1 in source pixels
    scale: float
    offset: tuple[int, int]             # top, left placement on the canvas
    canvas: tuple[int, int]             # (H, W)

    def source_to_render(self, r: float, c: float) -> tuple[float, float]:
        r0, _, c0, _ = self.box
        return ((r - r0) * self.scale + self.offset[0],
                (c - c0) * self.scale + self.offset[1])

    def render_to_source(self, y: float, x: float) -> tuple[float, float]:
        r0, _, c0, _ = self.box
        return ((y - self.offset[0]) / self.scale + r0,
                (x - self.offset[1]) / self.scale + c0)


def render_geometry(pixels: np.ndarray, size: int | tuple[int, int]) -> RenderGeometry:
    """Mirror of ``encoder.breast_crop`` + ``encoder.letterbox`` arithmetic.

    Must stay byte-for-byte consistent with those functions. The crop-box half
    of the equivalence is pinned by torch-free tests that run everywhere; the
    letterbox half needs torch, so ``test_render_geometry_matches_letterbox``
    SKIPS in the torch-less CI and runs on the training machine — run the
    suite there before training (the runbook says so too).
    """
    th, tw = (size, size) if isinstance(size, int) else size
    box = crop_box(pixels)
    r0, r1, c0, c1 = box
    h, w = r1 - r0, c1 - c0
    scale = min(th / h, tw / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return RenderGeometry(
        box=box, scale=scale,
        offset=((th - nh) // 2, (tw - nw) // 2),
        canvas=(th, tw),
    )


def _resize_mask_any(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Any-overlap mask resize: an output pixel is True if ANY source pixel in
    its block is True.

    Nearest-neighbor sampling is wrong for masks at downsampling scales: at
    the production render scale (~0.28) it visits ~1 of every 3.5 source
    rows/cols, so a small lesion can miss every sample and VANISH — dropping
    it from both lesion sampling and the background-exclusion union (69% of
    2x2 px lesions annihilated in the review's repro). Any-overlap is
    conservative in the safe direction: masks only ever grow by less than one
    output pixel, never disappear.
    """
    oh, ow = out_shape
    ih, iw = mask.shape
    mask = np.ascontiguousarray(mask, dtype=bool)
    row_edges = np.minimum((np.arange(oh) * ih) // oh, ih - 1).astype(np.intp)
    col_edges = np.minimum((np.arange(ow) * iw) // ow, iw - 1).astype(np.intp)
    rows = np.logical_or.reduceat(mask, row_edges, axis=0)
    return np.logical_or.reduceat(rows, col_edges, axis=1)


def mask_to_render(mask: np.ndarray, geometry: RenderGeometry,
                   image_shape: tuple[int, int]) -> np.ndarray:
    """Binary source-space mask -> binary canvas-space mask (any-overlap).

    A mask whose dims disagree with the image (known CBIS defect) is first
    resampled onto the image grid so the image-derived crop box applies to it
    meaningfully. Both resamples use any-overlap semantics — a lesion smaller
    than the sampling stride must dilate slightly, never vanish.
    """
    mask = (np.asarray(mask) > 0)
    if mask.shape != tuple(image_shape):
        mask = _resize_mask_any(mask, tuple(image_shape))
    r0, r1, c0, c1 = geometry.box
    cropped = mask[r0:r1, c0:c1]
    th, tw = geometry.canvas
    oy, ox = geometry.offset
    nh = max(1, round((r1 - r0) * geometry.scale))
    nw = max(1, round((c1 - c0) * geometry.scale))
    canvas = np.zeros((th, tw), dtype=bool)
    canvas[oy:oy + nh, ox:ox + nw] = _resize_mask_any(cropped, (nh, nw))
    return canvas


# --- sampling -------------------------------------------------------------

@dataclass(frozen=True)
class PatchSpec:
    y: int          # top-left corner on the render canvas
    x: int
    cls: int        # index into roi.PATCH_CLASSES; 0 = background
    roi_id: str | None


def _clamp_corner(cy: float, cx: float, patch: int, canvas: tuple[int, int]) -> tuple[int, int]:
    y = int(round(cy)) - patch // 2
    x = int(round(cx)) - patch // 2
    y = max(0, min(canvas[0] - patch, y))
    x = max(0, min(canvas[1] - patch, x))
    return y, x


def sample_lesion_patches(
    render_mask: np.ndarray,
    cls: int,
    roi_id: str,
    patch: int = 224,
    n: int = 10,
    jitter: float = 0.35,
    rng: np.random.Generator | None = None,
) -> list[PatchSpec]:
    """One centroid-centered patch + jittered patches centered on lesion pixels.

    Every accepted patch window CONTAINS the lesion pixel it was centered on
    (guaranteed by construction: centers are lesion pixels, corners clamped to
    the canvas, and jitter is bounded to keep the anchor inside the window).
    Returns [] for an empty mask rather than inventing background labeled as
    lesion.
    """
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(render_mask)
    if ys.size == 0:
        return []
    canvas = render_mask.shape
    if canvas[0] < patch or canvas[1] < patch:
        return []  # a window cannot fit; short is honest
    # Seed on the lesion pixel NEAREST the centroid, not the centroid itself:
    # for a multi-component mask (calcification clusters) the raw centroid can
    # sit between the blobs, yielding a "lesion" window with zero lesion
    # pixels. Anchoring on an actual pixel restores the containment guarantee.
    ci = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    specs = [PatchSpec(*_clamp_corner(ys[ci], xs[ci], patch, canvas), cls, roi_id)]
    max_off = int(patch * min(jitter, 0.49))
    for _ in range(max(0, n - 1)):
        i = int(rng.integers(ys.size))
        cy = ys[i] + int(rng.integers(-max_off, max_off + 1))
        cx = xs[i] + int(rng.integers(-max_off, max_off + 1))
        y, x = _clamp_corner(cy, cx, patch, canvas)
        # Clamping may push the window off the anchor lesion pixel near
        # borders; re-anchor on the pixel itself when that happens.
        if not (y <= ys[i] < y + patch and x <= xs[i] < x + patch):
            y, x = _clamp_corner(ys[i], xs[i], patch, canvas)
        specs.append(PatchSpec(y, x, cls, roi_id))
    return specs


def sample_background_patches(
    render: np.ndarray,
    exclusion_mask: np.ndarray,
    patch: int = 224,
    n: int = 10,
    breast_threshold: float = 0.02,
    min_breast_frac: float = 0.5,
    rng: np.random.Generator | None = None,
    max_tries: int = 200,
) -> list[PatchSpec]:
    """Background patches: inside the breast, ZERO overlap with any lesion.

    ``exclusion_mask`` must be the union of every ROI on the image — excluding
    only one lesion's mask would label the other lesion "background", the
    classic multi-finding trap. Fewer than ``n`` returns is normal on small
    breasts; short is honest, mislabeled is not.
    """
    rng = rng or np.random.default_rng(0)
    breast = render > breast_threshold
    canvas = render.shape
    if canvas[0] < patch or canvas[1] < patch:
        return []
    specs: list[PatchSpec] = []
    for _ in range(max_tries):
        if len(specs) >= n:
            break
        y = int(rng.integers(0, canvas[0] - patch + 1))
        x = int(rng.integers(0, canvas[1] - patch + 1))
        window_excl = exclusion_mask[y:y + patch, x:x + patch]
        if window_excl.any():
            continue
        if breast[y:y + patch, x:x + patch].mean() < min_breast_frac:
            continue
        specs.append(PatchSpec(y, x, 0, None))
    return specs


def extract(render: np.ndarray, spec: PatchSpec, patch: int = 224) -> np.ndarray:
    """Cut the patch window as uint8 [0, 255] (compact on disk, lossless enough
    for training — the render itself is already float16-quantized)."""
    window = render[spec.y:spec.y + patch, spec.x:spec.x + patch]
    return np.clip(np.asarray(window, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
