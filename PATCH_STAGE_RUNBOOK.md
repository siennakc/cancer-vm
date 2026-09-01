# Patch-pretraining stage — runbook

Everything below is written, tested on synthetic fixtures, and ready to run
on the training machine (the one with `data/raw/` and the GPU). Nothing in it
has touched real pixels yet. Total new download ~50–60 GB; total compute
roughly 1–2 h on the M4 Max.

**Why this stage:** it is the missing core of the published 0.88 CBIS recipe
(Shen et al. 2019 — patch classifier first, whole-image model warm-started
from it), *and* it produces the localizing detector whose absence the harness
A/B diagnosed (model 0.771 vs harness 0.700 with 41% uninformative deferrals).
One stage, both open fronts.

## Run order

```sh
# 0. Run the test suite ON THIS MACHINE first: one geometry-equivalence test
#    (render_geometry vs the torch letterbox) needs torch and SKIPS in the
#    torch-less CI — here it actually runs.
.venv/bin/python -m pytest
# 1. Fetch the ROI-mask + cropped series (~50-60 GB, public, resumable).
#    Exits nonzero if any series is still failed after 4 retry sweeps.
.venv/bin/python scripts/fetch_cbis_roi.py

# 2. Build the 5-class patch dataset (train + calibration patients ONLY,
#    splits_v2; public_bench/test/threshold are refused by construction).
#    Also writes data/processed/roi_v1.jsonl — commit that file.
.venv/bin/python scripts/build_patch_dataset.py

# 3. Train the patch classifier (~30-60 min at 224px on MPS).
.venv/bin/python scripts/train_patch_model.py --run runs/patch_v1

# 4. Whole-image v5: warm-start from the patch backbone, same v3 recipe.
.venv/bin/python scripts/finetune_encoder.py \
    --splits data/processed/splits_v2.json \
    --init-weights runs/patch_v1/best_model.pt \
    --run runs/finetune_v5

# then the usual: cache_embeddings -> refit heads -> eval_public_cbis
# (which also regenerates the density grid over all 709 images and re-scores
# against the corrected any-malignant gold — three stale annotations retire
# in one run).
```

## The harness rematch (after step 3)

`oncoscope.models.patch_detector.PatchDetector(weights_path="runs/patch_v1/best_model.pt")`
provides `propose(pixels) -> [PatchCandidate(box, score, cls, cls_probs)]`
with boxes in **source-pixel coordinates**. Wire it in as the harness's
`run_detector` backend in place of the whole-image screener, then re-run
`scripts/ab_harness_bench.py`. The Arm-B 1600px pre-shrink handicap is fixed:
native input is now the default (`--shrink 0`), the setting is stamped into
the report, and the deferral-informativeness analysis is computed in-script.
That number decides whether the harness lane earns its keep. Budget 2–4× the
recorded 50 min for the native-resolution run.

## Lineage discipline (enforced in code, know why it exists)

A warm start is a fitting stage: the init weights carry everything their
training data taught them. splits_v1 and splits_v2 cross memberships for 609
patients, so a patch model trained under one manifest must never initialize a
run held out under the other — the result would have no clean evaluation
surface anywhere. Enforcement, so you don't have to remember any of this:

- `finetune_encoder.py --init-weights` hard-errors unless the init
  checkpoint's `splits_sha256` equals the run's `--splits` sha, and stamps an
  `init_lineage` chain into every checkpoint it writes.
- `build_patch_dataset.py` refuses a splits manifest with no `public_bench`
  quarantine, reports skip counts by split name, and drops any image whose
  ROI masks can't ALL be read (background patches must never land on a lesion
  whose mask failed to decode).
- `SealedTestSet.score` accepts a list of fit manifests — for a warm-started
  model, pass its own manifest plus every `init_lineage` entry.

## What to check before believing any output

- `build_patch_dataset.py` prints per-split class balance and writes
  `build_report.json` (config, splits sha, failures). A high failure count or
  a missing class means stop, not shrug.
- The ROI table records a `status` per abnormality; `mask_dims_mismatch` is
  expected in small numbers (known CBIS defect, handled by nearest-resample),
  `no binary raster` should be rare — if it is common, the series listing
  changed shape and `classify_roi_files` needs eyes.
- The patch model checkpoint stamps `splits_sha256`; it must equal
  `splits_v2.json`'s (`45cc17cb…`). A different sha = wrong quarantine = stop.
- CMMD contributes no patches (it has no lesion annotations); the patch
  stage is DDSM-only by design. The whole-image stage still trains on both.

## Test coverage (runs anywhere, no torch, no data)

`tests/test_patch_pipeline.py`: geometry equivalence with the encoder's
breast_crop/letterbox (a drift silently mislocates every lesion — pinned),
round-trip coordinate mapping, mask resampling incl. the wrong-dims defect,
mask/crop classification by content under adversarial file naming, sampler
invariants (lesion patches contain their anchor, background patches never
touch any lesion, determinism), and split discipline (quarantined and unknown
patients are refused with `SplitViolation`).
