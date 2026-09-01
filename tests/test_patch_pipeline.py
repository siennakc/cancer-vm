"""Patch-pipeline invariants (the pre-flight for the patch-pretraining stage).

Everything here runs without torch and without the 126 GB of DICOMs: geometry
is pure arithmetic, ROI classification is exercised on synthetic DICOMs, and
the sampler on synthetic renders. The two invariants that matter most:

1. geometry equivalence — patch coordinates live in the exact breast_crop +
   letterbox frame the whole-image model sees (a drift here silently
   mislocates every lesion);
2. split discipline — patch sampling REFUSES patients outside train/
   calibration, so the patch model can never touch public_bench or the
   sealed/threshold splits.
"""

from __future__ import annotations

import json

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from oncoscope.data.patches import (
    DEFAULT_ALLOWED_SPLITS,
    SplitViolation,
    assert_sampleable,
    extract,
    mask_to_render,
    render_geometry,
    sample_background_patches,
    sample_lesion_patches,
)
from oncoscope.data.roi import PATCH_CLASSES, classify_roi_files, patch_class
from oncoscope.data.splits import make_splits
from oncoscope.models.encoder import breast_crop, crop_box

FRACTIONS = {"train": 0.6, "calibration": 0.1, "threshold": 0.1,
             "slice_discovery": 0.05, "test": 0.15}


# --- geometry --------------------------------------------------------------

def _breast_image(h=400, w=300, seed=0):
    """Dark field with a bright 'breast' region and soft texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    img[40:360, 30:250] = 0.3 + 0.2 * rng.random((320, 220)).astype(np.float32)
    return img


def test_crop_box_matches_breast_crop():
    img = _breast_image()
    r0, r1, c0, c1 = crop_box(img)
    assert np.array_equal(breast_crop(img), img[r0:r1, c0:c1])


def test_crop_box_blank_image_degrades_to_full_frame():
    blank = np.zeros((64, 48), dtype=np.float32)
    assert crop_box(blank) == (0, 64, 0, 48)


def test_render_geometry_matches_letterbox():
    torch = pytest.importorskip("torch")  # noqa: F841 — letterbox needs it
    from oncoscope.models.encoder import letterbox

    img = _breast_image()
    size = (256, 192)
    geo = render_geometry(img, size)
    rendered = letterbox(breast_crop(img), size)
    # The rendered content occupies exactly the region the geometry claims.
    oy, ox = geo.offset
    r0, r1, c0, c1 = geo.box
    nh = max(1, round((r1 - r0) * geo.scale))
    nw = max(1, round((c1 - c0) * geo.scale))
    content = rendered[oy:oy + nh, ox:ox + nw]
    assert content.shape == (nh, nw)
    assert content.mean() > 0.1                       # content landed inside
    border = rendered.copy()
    border[oy:oy + nh, ox:ox + nw] = 0
    assert border.max() == 0.0                        # nothing outside it


def test_geometry_roundtrip_source_render_source():
    geo = render_geometry(_breast_image(), (256, 192))
    for r, c in ((50, 40), (200, 120), (350, 240)):
        y, x = geo.source_to_render(r, c)
        rr, cc = geo.render_to_source(y, x)
        assert abs(rr - r) < 1e-6 and abs(cc - c) < 1e-6


def test_mask_to_render_lands_on_lesion():
    img = _breast_image()
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[100:130, 80:110] = 255                       # lesion in source space
    geo = render_geometry(img, (256, 192))
    rmask = mask_to_render(mask, geo, img.shape)
    ys, xs = np.nonzero(rmask)
    assert ys.size > 0
    # The rendered mask centroid maps back to the source lesion box.
    rr, cc = geo.render_to_source(float(ys.mean()), float(xs.mean()))
    assert 95 <= rr <= 135 and 75 <= cc <= 115


def test_mismatched_mask_dims_are_resampled():
    img = _breast_image(400, 300)
    small = np.zeros((200, 150), dtype=np.uint8)      # half-size mask (CBIS defect)
    small[50:65, 40:55] = 1                           # -> source ~(100:130, 80:110)
    geo = render_geometry(img, (256, 192))
    rmask = mask_to_render(small, geo, img.shape)
    ys, xs = np.nonzero(rmask)
    assert ys.size > 0
    rr, cc = geo.render_to_source(float(ys.mean()), float(xs.mean()))
    assert 90 <= rr <= 140 and 70 <= cc <= 120


# --- ROI content classification -------------------------------------------

def _dicom(pixels: np.ndarray, path):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_classify_by_content_not_filename(tmp_path):
    rng = np.random.default_rng(0)
    mask_px = (rng.random((300, 200)) > 0.7).astype(np.uint16) * 65535
    crop_px = (rng.random((80, 60)) * 4000).astype(np.uint16)
    # Adversarial naming: the mask sits at 000000.dcm (the "crop" slot).
    mask_file = _dicom(mask_px, tmp_path / "000000.dcm")
    crop_file = _dicom(crop_px, tmp_path / "000001.dcm")
    found_mask, found_crop, status = classify_roi_files([crop_file, mask_file])
    assert status == "ok"
    assert found_mask == mask_file
    assert found_crop == crop_file


def test_classify_refuses_series_with_no_mask(tmp_path):
    rng = np.random.default_rng(1)
    a = _dicom((rng.random((80, 60)) * 4000).astype(np.uint16), tmp_path / "a.dcm")
    b = _dicom((rng.random((90, 70)) * 4000).astype(np.uint16), tmp_path / "b.dcm")
    mask, crop, status = classify_roi_files([a, b])
    assert mask is None and "no binary raster" in status


def test_patch_class_taxonomy():
    assert PATCH_CLASSES[0] == "background"
    assert patch_class("calcification", 0) == 1
    assert patch_class("calcification", 1) == 2
    assert patch_class("mass", 0) == 3
    assert patch_class("mass", 1) == 4


# --- sampling --------------------------------------------------------------

def _render_world(patch=32):
    rng = np.random.default_rng(2)
    render = np.clip(0.3 + 0.1 * rng.random((256, 192)), 0, 1).astype(np.float32)
    mask = np.zeros((256, 192), dtype=bool)
    mask[60:90, 50:80] = True
    return render, mask, patch


def test_lesion_patches_contain_their_anchor():
    _, mask, patch = _render_world()
    specs = sample_lesion_patches(mask, cls=4, roi_id="r1", patch=patch, n=10,
                                  rng=np.random.default_rng(3))
    assert len(specs) == 10
    for s in specs:
        assert s.cls == 4
        # every window overlaps the lesion (its anchor pixel is inside)
        assert mask[s.y:s.y + patch, s.x:s.x + patch].any()


def test_empty_mask_yields_no_lesion_patches():
    assert sample_lesion_patches(np.zeros((64, 64), bool), 4, "r", patch=16) == []


def test_background_patches_never_touch_any_lesion():
    render, mask, patch = _render_world()
    specs = sample_background_patches(render, mask, patch=patch, n=10,
                                      rng=np.random.default_rng(4))
    assert specs, "background sampling failed on an easy image"
    for s in specs:
        assert s.cls == 0
        assert not mask[s.y:s.y + patch, s.x:s.x + patch].any()


def test_sampling_is_deterministic():
    render, mask, patch = _render_world()
    a = sample_lesion_patches(mask, 2, "r", patch=patch, rng=np.random.default_rng(7))
    b = sample_lesion_patches(mask, 2, "r", patch=patch, rng=np.random.default_rng(7))
    assert a == b
    ba = sample_background_patches(render, mask, patch=patch, rng=np.random.default_rng(7))
    bb = sample_background_patches(render, mask, patch=patch, rng=np.random.default_rng(7))
    assert ba == bb


def test_extract_is_uint8_window():
    render, mask, patch = _render_world()
    spec = sample_lesion_patches(mask, 4, "r", patch=patch)[0]
    out = extract(render, spec, patch=patch)
    assert out.shape == (patch, patch) and out.dtype == np.uint8


# --- split discipline ------------------------------------------------------

def test_quarantined_patients_are_refused():
    patients = {f"ddsm/P{i:03d}": "ddsm" for i in range(40)}
    manifest = make_splits(patients, FRACTIONS)
    # Simulate the public benchmark quarantine on one patient.
    forced = dict(manifest.assignment)
    victim = next(iter(forced))
    forced[victim] = "public_bench"
    object.__setattr__(manifest, "assignment", forced)

    with pytest.raises(SplitViolation):
        assert_sampleable(victim, manifest)
    for key, split in manifest.assignment.items():
        if split in DEFAULT_ALLOWED_SPLITS:
            assert assert_sampleable(key, manifest) == split
        else:
            with pytest.raises(SplitViolation):
                assert_sampleable(key, manifest)


def test_unknown_patient_is_refused():
    manifest = make_splits({"ddsm/P000": "ddsm"}, {"train": 1.0})
    with pytest.raises(SplitViolation):
        assert_sampleable("ddsm/GHOST", manifest)


# --- build report contract -------------------------------------------------

def test_build_script_importable_and_documents_discipline():
    """The build script must exist, parse, and carry the split-discipline
    contract in its module docstring (greppable governance, cheap to keep)."""
    import ast
    from pathlib import Path

    src = Path("scripts/build_patch_dataset.py").read_text()
    module = ast.parse(src)
    doc = ast.get_docstring(module) or ""
    assert "public_bench" in doc and "train" in doc
    assert json is not None  # keep the import honest


# --- review findings (regression) -----------------------------------------
#
# From the adversarial review of this pipeline: 104 of 3,568 CBIS rows keep
# mask and crop in separate single-file series; multiple binary rasters must
# tie-break on image dims (largest-wins inverts on the reduced-dims-mask
# defect); and image selection must drop whole images whose exclusion union
# cannot be proven.

def test_classify_single_file_binary_series(tmp_path):
    mask_px = (np.random.default_rng(5).random((120, 90)) > 0.8).astype(np.uint16) * 255
    mask_file = _dicom(mask_px, tmp_path / "000000.dcm")
    mask, crop, status = classify_roi_files([mask_file])
    assert status == "ok" and mask == mask_file and crop is None


def test_saturated_constant_crop_never_beats_a_real_mask(tmp_path):
    rng = np.random.default_rng(6)
    # The REAL mask: reduced dims (the known CBIS defect), 2-valued.
    real_mask = _dicom(((rng.random((150, 100)) > 0.7) * 255).astype(np.uint16),
                       tmp_path / "a.dcm")
    # A saturated CONSTANT crop, larger than the mask: passes the sampled
    # binary test, but a constant raster can never be a mask.
    saturated = _dicom(np.full((200, 160), 4000, dtype=np.uint16), tmp_path / "b.dcm")
    for shape in ((600, 400), None):
        mask, _, status = classify_roi_files([saturated, real_mask], image_shape=shape)
        assert status == "ok"
        assert mask == real_mask, f"constant raster won the tie (image_shape={shape})"


def test_two_nonconstant_binaries_tie_break_on_aspect(tmp_path):
    rng = np.random.default_rng(7)
    image_shape = (600, 400)                      # aspect 1.5
    # Real mask: half-dims reduction, aspect preserved (1.5), SMALLER raster.
    real_mask = _dicom(((rng.random((300, 200)) > 0.7) * 255).astype(np.uint16),
                       tmp_path / "a.dcm")
    # Two-valued near-square cutout, LARGER than the mask (aspect ~1.05).
    binaryish_crop = _dicom(((rng.random((400, 380)) > 0.5) * 255).astype(np.uint16),
                            tmp_path / "b.dcm")
    mask, _, status = classify_roi_files([binaryish_crop, real_mask],
                                         image_shape=image_shape)
    assert status == "ok"
    assert mask == real_mask, "aspect agreement must beat raster size"
    # Without image dims the residual size heuristic picks the larger raster —
    # pinned so the limitation stays documented rather than silent.
    mask_no_dims, _, _ = classify_roi_files([binaryish_crop, real_mask])
    assert mask_no_dims == binaryish_crop


def _roi_record(case_id, status="ok", roi_id=None):
    from oncoscope.data.roi import RoiRecord
    return RoiRecord(roi_id or f"{case_id}#1", case_id, case_id.split("-")[-1],
                     "mass", 1, 4, "m.dcm", None, (100, 80), (100, 80), status)


class _FakeCase:
    def __init__(self, case_id, patient_id):
        self.case_id, self.site, self.patient_id = case_id, "ddsm", patient_id


def _selection_world():
    from oncoscope.data.splits import make_splits
    patients = {f"ddsm/P{i}": "ddsm" for i in range(20)}
    manifest = make_splits(patients, FRACTIONS)
    cases = {f"ddsm-c{i}": _FakeCase(f"ddsm-c{i}", f"P{i}") for i in range(20)}
    return manifest, cases


def test_selection_drops_whole_image_with_unusable_roi():
    from oncoscope.data.patches import select_sampleable_images
    manifest, cases = _selection_world()
    fitting = [cid for cid in cases
               if manifest.assignment[f"ddsm/{cases[cid].patient_id}"] in DEFAULT_ALLOWED_SPLITS]
    good, bad = fitting[0], fitting[1]
    records = [
        _roi_record(good, "ok", f"{good}#1"),
        _roi_record(bad, "ok", f"{bad}#1"),
        _roi_record(bad, "undecodable: x", f"{bad}#2"),  # one bad ROI on the image
    ]
    selected, stats = select_sampleable_images(records, cases, manifest)
    assert good in selected
    assert bad not in selected, "an image with an unusable ROI must be dropped wholesale"
    assert stats["incomplete_images"] == 1


def test_selection_counts_split_skips_by_name_and_raises_on_broken_guard():
    from oncoscope.data.patches import select_sampleable_images
    manifest, cases = _selection_world()
    quarantined = next(cid for cid in cases
                       if manifest.assignment[f"ddsm/{cases[cid].patient_id}"] == "test")
    selected, stats = select_sampleable_images(
        [_roi_record(quarantined)], cases, manifest)
    assert not selected
    assert stats["split_skips"] == {"test": 1}

    class BrokenManifest:
        @property
        def assignment(self):
            raise RuntimeError("manifest backend exploded")

    with pytest.raises(RuntimeError):
        select_sampleable_images([_roi_record(quarantined)], cases, BrokenManifest())


def test_partially_downloaded_and_dims_unavailable_are_unusable():
    from oncoscope.data.roi import RoiRecord
    for status in ("roi series partially downloaded — mask missing",
                   "full image dims unavailable — unusable",
                   "mask dims unreadable — unusable"):
        r = _roi_record("ddsm-x", status)
        assert not r.usable, status
    assert _roi_record("ddsm-x", "mask_dims_mismatch").usable


# --- shard provenance (second review round) --------------------------------
#
# The load-time gate: taint, manifest sha, per-shard hashes, and per-patch
# membership are re-derived at training time instead of trusted from the
# directory — closing the smoke-flag, stale-shard, and resume laundering
# paths the review demonstrated end-to-end.

def _shard_world(tmp_path, taint=False, patch=8):
    from oncoscope.data.patches import file_sha256
    from oncoscope.data.splits import make_splits

    patients = {f"ddsm/P{i}": "ddsm" for i in range(30)}
    manifest = make_splits(patients, FRACTIONS)
    rng = np.random.default_rng(0)
    counts = {}
    for split in DEFAULT_ALLOWED_SPLITS:
        members = [p for p, s in manifest.assignment.items() if s == split][:3]
        arr = (rng.random((len(members), patch, patch)) * 255).astype(np.uint8)
        meta = [{"case_id": f"c-{p}", "patient": p, "cls": 4, "y": 0, "x": 0,
                 "roi_id": None} for p in members]
        np.save(tmp_path / f"patches_{split}.npy", arr)
        (tmp_path / f"meta_{split}.json").write_text(json.dumps(meta))
        counts[split] = {"n": len(members), "by_class": {"4": len(members)},
                         "patches_sha256": file_sha256(tmp_path / f"patches_{split}.npy"),
                         "meta_sha256": file_sha256(tmp_path / f"meta_{split}.json")}
    report = {"config": {"allow_unquarantined": taint},
              "splits_sha256": manifest.sha256, "counts": counts}
    (tmp_path / "build_report.json").write_text(json.dumps(report))
    return manifest


def test_shard_provenance_accepts_clean_world(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    manifest = _shard_world(tmp_path)
    report = verify_shard_provenance(tmp_path, manifest)
    assert report["splits_sha256"] == manifest.sha256


def test_shard_provenance_refuses_taint_unless_opted_in(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    manifest = _shard_world(tmp_path, taint=True)
    with pytest.raises(SplitViolation, match="allow-unquarantined"):
        verify_shard_provenance(tmp_path, manifest)
    verify_shard_provenance(tmp_path, manifest, allow_tainted=True)  # smoke path


def test_shard_provenance_refuses_cross_manifest(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    from oncoscope.data.splits import make_splits
    _shard_world(tmp_path)
    other = make_splits({f"ddsm/P{i}": "ddsm" for i in range(30)}, FRACTIONS, seed=99)
    with pytest.raises(SplitViolation, match="rebuild the shards"):
        verify_shard_provenance(tmp_path, other)


def test_shard_provenance_refuses_edited_and_stale_files(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    manifest = _shard_world(tmp_path)
    # Edited shard: flip one byte.
    path = tmp_path / "patches_train.npy"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(SplitViolation, match="does not match the build report"):
        verify_shard_provenance(tmp_path, manifest)

    # Stale shard: a file the report does not cover.
    manifest = _shard_world(tmp_path)  # rebuild clean
    report = json.loads((tmp_path / "build_report.json").read_text())
    del report["counts"]["calibration"]
    (tmp_path / "build_report.json").write_text(json.dumps(report))
    with pytest.raises(SplitViolation, match="stale shard"):
        verify_shard_provenance(tmp_path, manifest)


def test_shard_provenance_refuses_wrong_split_membership(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    from oncoscope.data.patches import file_sha256
    manifest = _shard_world(tmp_path)
    # Swap a calibration patient into the train meta (a hand-edit / routing bug).
    meta_path = tmp_path / "meta_train.json"
    meta = json.loads(meta_path.read_text())
    intruder = next(p for p, s in manifest.assignment.items() if s == "calibration")
    meta[0]["patient"] = intruder
    meta_path.write_text(json.dumps(meta))
    report = json.loads((tmp_path / "build_report.json").read_text())
    report["counts"]["train"]["meta_sha256"] = file_sha256(meta_path)
    (tmp_path / "build_report.json").write_text(json.dumps(report))
    with pytest.raises(SplitViolation, match="not in split 'train'"):
        verify_shard_provenance(tmp_path, manifest)


def test_shard_provenance_refuses_missing_report(tmp_path):
    from oncoscope.data.patches import verify_shard_provenance
    from oncoscope.data.splits import make_splits
    manifest = make_splits({"ddsm/P0": "ddsm"}, {"train": 1.0})
    with pytest.raises(SplitViolation, match="build_report.json missing"):
        verify_shard_provenance(tmp_path, manifest)


def test_sentinel_rows_are_unusable_and_roundtrip():
    from oncoscope.data.roi import RoiRecord
    from oncoscope.data.roi import read_roi_table, write_roi_table
    import tempfile
    sentinel = RoiRecord("ddsm-x#1", "ddsm-x", "P1", "mass", None, None,
                         None, None, None, None,
                         "unresolvable row — image excluded from sampling")
    assert not sentinel.usable
    with tempfile.TemporaryDirectory() as d:
        write_roi_table([sentinel], f"{d}/t.jsonl")
        back = read_roi_table(f"{d}/t.jsonl")
    assert back == [sentinel]


# --- final review round: geometry + detector -------------------------------
#
# Confirmed by repro in the adversarial review: nearest-resampling annihilated
# small masks (69% of 2x2px lesions vanished at production scale, letting
# background patches cover real lesions); the centroid seed missed
# multi-component masks; the detector's NMS was directionally asymmetric and
# its box conversion could emit inverted boxes for padding windows.

def test_small_masks_survive_rendering():
    rng = np.random.default_rng(11)
    img = np.zeros((1000, 800), dtype=np.float32)
    img[50:950, 40:760] = 0.4
    geo = render_geometry(img, (288, 224))
    survived = 0
    for _ in range(100):
        mask = np.zeros((1000, 800), dtype=np.uint8)
        r = int(rng.integers(60, 940)); c = int(rng.integers(50, 750))
        mask[r:r + 2, c:c + 2] = 1                    # 2x2 px lesion
        if mask_to_render(mask, geo, img.shape).any():
            survived += 1
    assert survived == 100, f"only {survived}/100 tiny lesions survived rendering"


def test_background_exclusion_survives_rendering():
    """The end-to-end version: a tiny lesion must still repel background patches."""
    img = np.zeros((1000, 800), dtype=np.float32)
    img[50:950, 40:760] = 0.4
    geo = render_geometry(img, (288, 224))
    mask = np.zeros((1000, 800), dtype=np.uint8)
    mask[500:502, 400:402] = 1
    rmask = mask_to_render(mask, geo, img.shape)
    assert rmask.any()
    from oncoscope.models.encoder import breast_crop
    render = np.zeros((288, 224), dtype=np.float32)
    render[:] = 0.4                                   # simple bright canvas
    specs = sample_background_patches(render, rmask, patch=32, n=50,
                                      rng=np.random.default_rng(12), max_tries=2000)
    for s in specs:
        assert not rmask[s.y:s.y + 32, s.x:s.x + 32].any()


def test_multicomponent_mask_centroid_patch_contains_lesion():
    mask = np.zeros((256, 192), dtype=bool)
    mask[60:70, 50:60] = True                         # blob A
    mask[180:190, 130:140] = True                     # blob B: centroid is between
    specs = sample_lesion_patches(mask, cls=2, roi_id="r", patch=32, n=10,
                                  rng=np.random.default_rng(13))
    for s in specs:
        assert mask[s.y:s.y + 32, s.x:s.x + 32].any(), "lesion patch without lesion"


def test_symmetric_nms_suppresses_both_directions():
    from oncoscope.models.patch_detector import symmetric_nms
    patch, stride = 32, 16
    corners = [(64, 64), (64, 64 + stride), (64, 64 - stride),
               (64 + stride, 64), (64 - stride, 64)]
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.8])
    picked = symmetric_nms(corners, scores, patch, 0.25, 16)
    assert picked == [0], f"stride-neighbours must all be suppressed, got {picked}"


def test_content_filter_drops_pure_padding_windows():
    from oncoscope.models.patch_detector import content_windows
    img = np.zeros((4000, 700), dtype=np.float32)     # narrow: wide side padding
    img[100:3900, 50:650] = 0.4
    geo = render_geometry(img, (1152, 896))
    grid = [(0, 0), (0, 872), (500, 400)]             # two corners, one center
    kept = content_windows(grid, geo, 24)
    assert (500, 400) in kept
    assert (0, 0) not in kept and (0, 872) not in kept


def test_window_to_source_box_never_inverts():
    from oncoscope.models.patch_detector import window_to_source_box
    img = np.zeros((4000, 700), dtype=np.float32)
    img[100:3900, 50:650] = 0.4
    geo = render_geometry(img, (1152, 896))
    for y in range(0, 1152 - 224 + 1, 112):
        for x in range(0, 896 - 224 + 1, 112):
            box = window_to_source_box(geo, y, x, 224, img.shape)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            assert 0 <= x0 < x1 <= img.shape[1] - 1, box
            assert 0 <= y0 < y1 <= img.shape[0] - 1, box


def test_window_to_source_box_padding_window_is_none():
    from oncoscope.models.patch_detector import window_to_source_box
    img = np.zeros((4000, 700), dtype=np.float32)
    img[100:3900, 50:650] = 0.4
    geo = render_geometry(img, (1152, 896))
    assert window_to_source_box(geo, 0, 0, 224, img.shape) is None


def test_canvas_smaller_than_patch_returns_empty():
    tiny = np.full((16, 16), 0.4, dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    assert sample_lesion_patches(mask, 4, "r", patch=32) == []
    assert sample_background_patches(tiny, np.zeros((16, 16), bool), patch=32) == []
