"""CAMELYON16 streaming toolkit: fetch one slide, tile tissue at 10x, score, delete.

Slides are BigTIFF pyramids on public S3 (CC0). Nothing here stores more than
one slide at a time (peak disk < 5 GB/worker against a 340 GB corpus).
Patch geometry matches PCam exactly: 96x96 at ~0.97 um/px, so the PCam-trained
classifier sees its native distribution.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

BUCKET = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16"
TARGET_MPP = 0.972
PATCH = 96


def fetch(rel: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["curl", "-sL", "--retry", "5", "-C", "-", "-o", str(dest),
                        f"{BUCKET}/{rel}"], check=True)
    return dest


def open_slide(path: Path):
    import tiffslide
    return tiffslide.TiffSlide(str(path))


def pick_level(slide) -> tuple[int, float]:
    """Level whose mpp is closest to TARGET_MPP, plus its scale vs target."""
    mpp0 = float(slide.properties.get("tiffslide.mpp-x") or 0.243)
    best, best_d = 0, 1e9
    for lvl, ds in enumerate(slide.level_downsamples):
        d = abs(mpp0 * ds - TARGET_MPP)
        if d < best_d:
            best, best_d = lvl, d
    return best, (mpp0 * slide.level_downsamples[best]) / TARGET_MPP


def tissue_tiles(slide, level: int, stride: int = PATCH):
    """Grid coords (level space) whose thumbnail cell looks like tissue (HSV)."""
    thumb_level = len(slide.level_downsamples) - 1
    tw, th = slide.level_dimensions[thumb_level]
    thumb = np.asarray(slide.read_region((0, 0), thumb_level, (tw, th)).convert("RGB"),
                       np.float32) / 255.0
    mx, mn = thumb.max(2), thumb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    mask = (sat > 0.07) & (mx > 0.1) & (mn < 0.95)
    W, H = slide.level_dimensions[level]
    fx, fy = tw / W, th / H
    coords = []
    for y in range(0, H - PATCH + 1, stride):
        ty0, ty1 = int(y * fy), max(int((y + PATCH) * fy), int(y * fy) + 1)
        row = mask[ty0:ty1]
        for x in range(0, W - PATCH + 1, stride):
            tx0, tx1 = int(x * fx), max(int((x + PATCH) * fx), int(x * fx) + 1)
            if row[:, tx0:tx1].mean() > 0.25:
                coords.append((x, y))
    return coords


def read_patches(slide, level: int, coords, scale: float):
    """Yield float32 [0,1] RGB 96x96 patches (resampled if level mpp != target)."""
    from PIL import Image
    ds = slide.level_downsamples[level]
    side = PATCH if abs(scale - 1) < 0.02 else int(round(PATCH * scale))
    for x, y in coords:
        img = slide.read_region((int(x * ds), int(y * ds)), level, (side, side)).convert("RGB")
        if side != PATCH:
            img = img.resize((PATCH, PATCH), Image.BILINEAR)
        yield np.asarray(img, np.float32) / 255.0
