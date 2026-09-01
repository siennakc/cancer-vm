"""Build the 5-class patch dataset from ROI masks (train + calibration only).

fetch_cbis_roi.py -> build_roi_table -> sample patches in render space ->
one memmap-friendly uint8 array per split + a metadata sidecar.

Split discipline (enforced in oncoscope.data.patches, re-asserted here):
patches come from ``train`` and ``calibration`` patients under splits_v2
ONLY. public_bench / test / threshold / slice_discovery patients never enter
this dataset — the patch model feeds the whole-image encoder and the harness
detector, so a leak here poisons everything downstream.

Refusals are structural, not advisory: a splits manifest with no
``public_bench`` quarantine is rejected outright (``--allow-unquarantined``
exists for synthetic smoke tests only), split checks raise through anything
that is not a clean SplitViolation, and an image with any unusable ROI is
dropped wholesale so background patches can never land on a lesion whose
mask could not be read.

Outputs: the ROI table at --roi-table (default data/processed/roi_v1.jsonl —
commit it), and under --out (default data/cache/patches224_v1/):
    patches_<split>.npy           (N, 224, 224) uint8
    meta_<split>.json             per-patch class / case / patient / coords
    build_report.json             config, counts, splits sha, skip stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from oncoscope.data.mammography import read_case_table
from oncoscope.data.patches import (
    DEFAULT_ALLOWED_SPLITS,
    extract,
    file_sha256,
    mask_to_render,
    render_geometry,
    sample_background_patches,
    sample_lesion_patches,
    select_sampleable_images,
)
from oncoscope.data.roi import build_roi_table, read_roi_table, write_roi_table
from oncoscope.data.splits import load_manifest
from oncoscope.data.dicom_canonical import load_canonical

RAW = Path("data/raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/splits_v2.json")
    ap.add_argument("--roi-table", default="data/processed/roi_v1.jsonl")
    ap.add_argument("--rebuild-roi-table", action="store_true")
    ap.add_argument("--render-size", default="1152x896",
                    help="HxW; must match the render the whole-image model trains on")
    ap.add_argument("--patch", type=int, default=224)
    ap.add_argument("--lesion-per-roi", type=int, default=10)
    ap.add_argument("--background-per-image", type=int, default=10)
    ap.add_argument("--out", default="data/cache/patches224_v1")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--allow-unquarantined", action="store_true",
                    help="permit a splits manifest without a public_bench split "
                         "(synthetic smoke tests only — never real data)")
    args = ap.parse_args()

    size = tuple(int(v) for v in args.render_size.split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Stale-shard defense: a rebuild must never leave files from a previous
    # manifest beside a fresh report. The report goes first so an interrupted
    # rebuild leaves an UNVERIFIABLE directory (train refuses), never a mixed
    # one that verifies.
    for stale in [out / "build_report.json",
                  *out.glob("patches_*.npy"), *out.glob("meta_*.json")]:
        stale.unlink(missing_ok=True)
    splits = load_manifest(args.splits)
    if "public_bench" not in set(splits.assignment.values()) and not args.allow_unquarantined:
        raise SystemExit(
            f"refusing: {args.splits} has no public_bench quarantine split. "
            "The patch model warm-starts the whole-image encoder, so building "
            "patches under an unquarantined manifest poisons the public benchmark "
            "(this is how v2 got disqualified). Use splits_v2 or later."
        )
    cases = {c.case_id: c for c in read_case_table("data/processed/cases_v1.jsonl")}

    roi_path = Path(args.roi_table)
    if args.rebuild_roi_table or not roi_path.exists():
        print("[patches] building ROI table (decodes every ROI series once)…", flush=True)
        records = build_roi_table("data/metadata", "data/metadata/manifest_cbis_roi.jsonl",
                                  RAW, known_case_ids=set(cases))
        write_roi_table(records, roi_path)
    records = read_roi_table(roi_path)
    status = Counter(r.status for r in records)
    print(f"[patches] ROI records: {len(records)}; status: {dict(status)}", flush=True)

    # Split discipline + exclusion completeness live in this one helper; a
    # broken manifest raises through it rather than counting as a "skip".
    by_case, select_stats = select_sampleable_images(
        records, cases, splits, DEFAULT_ALLOWED_SPLITS)
    print(f"[patches] images selected: {select_stats['selected']}  "
          f"split skips: {select_stats['split_skips']}  "
          f"incomplete (unusable ROI on image): {select_stats['incomplete_images']}  "
          f"unknown case: {select_stats['unknown_case']}", flush=True)
    total_images = len({r.case_id for r in records})
    if select_stats["selected"] == 0 or (
        select_stats["selected"] < 0.2 * max(total_images, 1)
    ):
        print("[patches] WARNING: selection kept "
              f"{select_stats['selected']}/{total_images} images — a wholesale "
              "drop usually means the wrong splits manifest or a broken ROI "
              "fetch, not biology. Investigate before training.", flush=True)

    rng = np.random.default_rng(args.seed)
    shards: dict[str, dict] = {s: {"patches": [], "meta": []} for s in DEFAULT_ALLOWED_SPLITS}
    failures: list[str] = []

    for i, (case_id, rois) in enumerate(sorted(by_case.items()), 1):
        case = cases[case_id]
        split = splits.split_of(f"{case.site}/{case.patient_id}")
        if split not in shards:
            # Selection admitted this image, so a non-allowed split here means
            # the guard itself is broken — halt, never file under "failures".
            raise RuntimeError(
                f"{case_id} routed to split {split!r} outside {tuple(shards)} "
                "after selection — the split guard is broken")
        try:
            pixels = load_canonical(RAW / case.dicom_path).pixels.astype(np.float32)
            geometry = render_geometry(pixels, size)
            # Render via the SAME numpy pipeline geometry; pixel values come
            # from the canonical loader (one decoder, axiom of T-1.1).
            from oncoscope.models.encoder import breast_crop, letterbox
            render = letterbox(breast_crop(pixels), size)

            render_masks = []
            for r in rois:
                import pydicom
                mask = pydicom.dcmread(RAW / r.mask_relpath).pixel_array
                render_masks.append((r, mask_to_render(mask, geometry, pixels.shape)))
            exclusion = np.zeros(size, dtype=bool)
            for _, m in render_masks:
                exclusion |= m

            specs = []
            for r, m in render_masks:
                specs += sample_lesion_patches(m, r.cls, r.roi_id, patch=args.patch,
                                               n=args.lesion_per_roi, rng=rng)
            specs += sample_background_patches(render, exclusion, patch=args.patch,
                                               n=args.background_per_image, rng=rng)
            for s in specs:
                shards[split]["patches"].append(extract(render, s, patch=args.patch))
                shards[split]["meta"].append({
                    "case_id": case_id, "patient": f"{case.site}/{case.patient_id}",
                    "cls": s.cls, "y": s.y, "x": s.x, "roi_id": s.roi_id,
                })
        except Exception as exc:
            failures.append(f"{case_id}: {exc}")
        if i % 100 == 0:
            print(f"[patches] {i}/{len(by_case)} images", flush=True)

    report = {"config": vars(args), "splits_sha256": splits.sha256,
              "images": len(by_case), "selection": select_stats,
              "failures": failures[:50],
              "n_failures": len(failures), "counts": {}}
    for split, shard in shards.items():
        if not shard["patches"]:
            continue
        arr = np.stack(shard["patches"])
        patches_path = out / f"patches_{split}.npy"
        meta_path = out / f"meta_{split}.json"
        np.save(patches_path, arr)
        meta_path.write_text(json.dumps(shard["meta"]))
        balance = Counter(m["cls"] for m in shard["meta"])
        report["counts"][split] = {
            "n": int(arr.shape[0]),
            "by_class": {str(k): v for k, v in sorted(balance.items())},
            # Per-shard hashes bind these exact bytes to this report; training
            # verifies them (verify_shard_provenance) before touching a batch.
            "patches_sha256": file_sha256(patches_path),
            "meta_sha256": file_sha256(meta_path),
        }
        print(f"[patches] {split}: {arr.shape[0]} patches, class balance {dict(balance)}",
              flush=True)
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    if failures:
        print(f"[patches] {len(failures)} image failures recorded in build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
