# Training-machine handoff — for the agent running the next session

You are an AI agent working in the `oncoscope` checkout (formerly `cancer-vm` —
renamed 2026-09-02; GitHub redirects the old remote, but update your origin URL)
on the training machine
(Apple Silicon, MPS, `data/raw/` populated with the 126 GB corpus, `.venv` with
torch). This document is your work order. It was written by the agent working
with Sienna on her machine on 2026-08-31/09-01; everything referenced here is
committed on `main`.

## Prime directives (read before any command)

1. **Refusals are features.** The scripts now enforce split provenance, shard
   hashes, taint, and lineage in code. If a script refuses to run, it is
   working — diagnose, never bypass. Specifically: never pass
   `--allow-unquarantined` or `--allow-tainted-shards` on real data, and never
   work around `SealedProvenanceError` or `SplitViolation`.
2. **The internal sealed set (`sealed_test_v1`) is off-limits for every model
   trained under splits_v2** (v3, v4, and everything you train here). The
   scorer now refuses this in code. Do not spend its queries.
3. **The MIAS benchmark has a 20-query budget (2 spent).** Do not score it
   casually; one run per finished model generation, at most.
4. **Commit conventions:** author is Sienna Chen, no AI co-author trailers,
   result JSONs and CARDs are committed, weights go to GitHub releases, raw
   data and caches never enter git. Pull before you start; rebase, never
   force-push.
5. **Expected numbers below are sanity rails, not targets.** If a number lands
   far outside its rail, stop and investigate rather than proceeding.

## Step 0 — Sync and verify (10 min)

```sh
git pull --rebase
.venv/bin/pip install -e '.[dev]' --quiet
.venv/bin/python -m pytest
```

- Expect **80 passed** (the torch geometry-equivalence test SKIPS on machines
  without torch; here it must RUN and pass — it pins patch coordinates to the
  letterbox the models actually see).
- Verify `data/processed/splits_v2.json` exists with sha
  `45cc17cb8593ec2911ffc8c1e3bcdd83c1961ababfa0b70053889d883ce6a4b8`
  (`python -c "import json; print(json.load(open('data/processed/splits_v2.json'))['sha256'])"`).
- Note what changed since you last worked here: any-malignant gold labels
  (11 flips in `cases_v1.jsonl`), density bands backfilled for all calc cases,
  `finetune_encoder.py` defaults to splits_v2, `eval_public_cbis.py` takes
  `--encoder-checkpoint`, `ab_harness_bench.py` defaults to native-resolution
  Arm B input, and the whole patch stage below is new. `git log --oneline
  95ed753..HEAD` lists it all.

## Step 1 — Patch stage (the main event; ~2 h compute + ~50-60 GB download)

Read `PATCH_STAGE_RUNBOOK.md` first — it explains why each guard exists.
Commands, in order:

```sh
# 1a. Fetch ROI-mask + crop series. Resumable; exits nonzero while any series
#     is still failed. Re-run until it prints "complete, zero failed series".
.venv/bin/python scripts/fetch_cbis_roi.py

# 1b. Build the patch dataset (train+calibration patients of splits_v2 ONLY).
.venv/bin/python scripts/build_patch_dataset.py
```

Sanity rails for 1b: ROI table status counts should be overwhelmingly `ok`
(a handful of `mask_dims_mismatch` is the known CBIS defect and fine; more
than ~2% `no binary raster`/`unresolvable` means the fetch is incomplete —
stop). "images selected" should be roughly 1,600–2,000 (ddsm train+calibration
images with usable ROIs); a wholesale drop triggers the built-in warning —
heed it. Commit `data/processed/roi_v1.jsonl` when clean.

```sh
# 1c. Train the 5-class patch classifier (~30-60 min at 224px on MPS).
.venv/bin/python scripts/train_patch_model.py --run runs/patch_v1
```

Sanity rails: calibration macro-AUROC should clear 0.60 within 2 epochs and
plateau somewhere ≥0.75. The checkpoint must show
`splits_sha256 = 45cc17cb…` and `tainted: False`. If the loader is slow or
memory balloons, something regressed in the lazy-memmap dataset — stop.

```sh
# 1d. Whole-image v5: same v3 recipe, warm-started from the patch backbone.
#     (The script verifies the lineage sha itself; splits_v2 is the default.)
.venv/bin/python scripts/finetune_encoder.py \
    --init-weights runs/patch_v1/best_model.pt \
    --run runs/finetune_v5

# 1e. Embeddings + calibrated head for v5 (mirror the v3 pattern):
.venv/bin/python scripts/cache_embeddings.py --tag resnet50_ft_v5_448_raw \
    --weights runs/finetune_v5/best_model.pt --raw --gray-stats
# adapt scripts/refit_heads_v3.py -> refit_heads_v5.py (paths/tag only)

# 1f. Optionally: repeat the v4 high-res post-train on top of v5
#     (scripts/posttrain_hr.py — read its args first) as v5hr, then re-embed.
```

Sanity rails for v5: calibration AUROC at 448px should be ≥ v3's 0.8331; if
the warm start does not beat the cold start, that is itself a reportable
result — record it either way in `TRAINING_HISTORY.md`.

## Step 2 — Re-score the public benchmark (30 min, no training)

The committed density grids are mass-only-stale and the gold labels changed
(4 bench images flipped to malignant — the old error DEPRESSED AUROC).
Embeddings are label-free, so v3/v4 need no recompute:

```sh
.venv/bin/python scripts/eval_public_cbis.py --tag resnet50_ft_v3_448_raw \
    --head runs/finetune_v3_head/head.json --name finetune_v3 \
    --encoder-checkpoint runs/finetune_v3/best_model.pt
.venv/bin/python scripts/eval_public_cbis.py --tag resnet50_ft_v4_1152x896_raw \
    --head runs/finetune_v4_head/head.json --name posttrain_v4 \
    --encoder-checkpoint runs/posttrain_v4/best_model.pt
# then v5 (and v5hr) the same way
```

- Every report must print `encoder lineage verified` — if it refuses, the
  lineage is genuinely wrong; investigate, don't drop the flag.
- Expect v4 to land slightly ABOVE 0.7707 (label-fix direction) with density
  slices now summing to n=709. Update the README results table and
  `TRAINING_HISTORY.md` with the re-scored numbers; the fresh JSONs replace
  the stale-annotated ones.

## Step 3 — The harness A/B rematch (the experiment that decides the lane)

Two runs, in this order:

```sh
# 3a. Handicap-isolation run: unchanged v4 proposer, now at native resolution
#     (the recorded -0.071 fed Arm B 1600px input; this measures the honest
#     gap with the OLD architecture). Budget 2-4x the recorded 50 min.
.venv/bin/python scripts/ab_harness_bench.py --out results/ab_harness/report_native_v4.json
```

```sh
# 3b. The real rematch: patch detector as the proposer. Wire it in
#     ab_harness_bench.py in place of V4WindowDetector with a thin adapter:
#
#   from oncoscope.models.patch_detector import PatchDetector
#   from oncoharness.reference.detector import Candidate
#   class PatchProposer:
#       def __init__(self):
#           self.det = PatchDetector(weights_path="runs/patch_v1/best_model.pt")
#           self.forwards = 0
#       def propose(self, pixels):
#           cands = self.det.propose(pixels)
#           self.forwards += 1
#           return [Candidate(box=c.box, score=c.score) for c in cands]
#
#     Keep the v4 whole-image score as an anchor feature at aggregation if the
#     harness supports fusion; otherwise run detector-pure and note it.
.venv/bin/python scripts/ab_harness_bench.py --out results/ab_harness/report_patch_rematch.json
```

Report BOTH deltas with their CIs in a CARD update, whatever their sign.
The pre-registered question: does the paired ΔAUROC (harness − model) CI
cross zero once the proposer can localize? Either answer ships.

## Step 4 — Small cleanups while queues run

- `scripts/bench/train_pcam.py` now writes `results/bench_pcam/report.json`
  — re-run it so the claimed 0.964 has an artifact (it is validation-split
  only; the report says so itself).
- `gh release create weights-v5 runs/finetune_v5/best_model.pt runs/finetune_v5/checkpoint.pt`
  (sha256 in the notes, matching the v2–v4 releases).
- Commit: result JSONs, CARDs, `roi_v1.jsonl`, `build_report.json` is cache —
  do NOT commit shards/embeddings/weights.

## Reporting back

Update `TRAINING_HISTORY.md` in the established intervention-ledger style:
what was run, wall time, every metric moved (or not), and any refusal a guard
raised with its resolution. If a sanity rail was breached, the writeup of why
matters more than the run itself. Sienna's session will pull and review.
