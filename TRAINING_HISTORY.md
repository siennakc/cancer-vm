# Training history — every change made to the model, and what it did

**Not a medical device. Research artifact only.**

This file is the complete ledger of interventions that made the model better
(or didn't — those are recorded too). Every number is reproducible from the
committed scripts; the source JSON for each figure lives in `results/`.

## The benchmark, precisely

There are no "questions" in the LLM sense. The benchmark is the **official
CBIS-DDSM test split**: 709 mammogram images from 349 patients, fixed by the
dataset authors in `mass_case_description_test_set.csv` +
`calc_case_description_test_set.csv`. Each "question" is one mammogram; the
hidden answer is its biopsy-proven pathology (malignant = 1; benign and
BENIGN_WITHOUT_CALLBACK = 0). The model must score every image; the reported
number is image-level AUROC with patient-clustered 95% CIs. Published results
on this exact split run ≈ 0.75–0.88 (Shen et al. 2019: 0.88 single model,
0.91 ensemble). Two supporting evaluations: an internal hash-sealed test set
(n = 1,002, query-budgeted), and MIAS (n = 322, 1994 UK film-screen) as
never-trained external validation.

## Scoreboard

| version | change | official CBIS test | MIAS external | internal sealed |
|---|---|---|---|---|
| v1 | frozen ImageNet features + calibrated head | 0.628 | 0.512 (chance) | 0.707 |
| v2 | end-to-end fine-tune @448 | *disqualified*¹ | 0.733 | 0.8165 |
| v3 | v2 recipe + official-test quarantine | **0.742** | 0.664 | not spent |
| v4 | + literature post-train @1152×896 | **0.771** | **0.736** | — |

¹ v2 trained on 202/349 official-test patients before the quarantine existed;
reporting it on the benchmark would be leakage.

## Part 1 — data corrections (before any training)

These fixed silent label/leakage errors; every later number depends on them.

1. **Patient-grouping trap.** TCIA's PatientID for CBIS-DDSM is per-*view*
   (`Mass-Training_P_01239_RIGHT_CC`). Grouping splits on it scatters one
   woman's four views across train/test — image-level leakage worth 2–20 fake
   AUROC points in the field. Fixed by keying on the CSV `patient_id`
   (`P_01239`); regression-tested.
2. **Label mapping.** `BENIGN_WITHOUT_CALLBACK` → negative (it is tissue not
   even recalled); folding it into positives inflates prevalence and wrecks
   calibration.
3. **Byte-level twin audit** (`content_audit`). Both collections enroll some
   women twice under different patient IDs with byte-identical DICOMs; two
   CMMD pairs and two CBIS pairs carry *contradictory* labels for the same
   bytes. Consistent twins merged into one split group; conflicted patients
   (8) dropped entirely; 24 duplicate images removed. UID checks cannot see
   any of this. Record: `data/processed/duplicates_audit_v1.json`.
4. **Patient-grouped 5-way splits**, hash-locked: train / calibration /
   threshold / slice_discovery / test — thresholds and temperatures are never
   chosen on data that produced the headline number.
5. **Prevalence honesty.** Both cohorts are biopsy-enriched (~58% malignant
   vs ~0.5% screening reality) → analytic prior-correction shift is applied
   after balanced training, and PPV is never quoted.

## Part 2 — model versions

### v1 — frozen baseline (the control arm)
Stock ResNet-50 (IN1K), frozen; breast-crop + 448 letterbox; L2-normed GAP
embeddings; 5-seed class-balanced logistic head, merged exactly (linear ⇒
mean of logits = averaged head); prior shift + temperature; threshold at
≥96% specificity on the threshold split.
**Result: 0.628 official / 0.512 MIAS.** Its MIAS collapse to chance proved
the internal number was substantially site-shortcut, and set the bar honest
training had to clear.

### v2 — end-to-end fine-tune @448
All 23.56M backbone params unfrozen. Class-balanced sampler; augs hflip /
±10° / scale 0.9–1.1 / intensity jitter; AdamW 1e-4, cosine, 14 epochs,
batch 16; epoch selection on calibration AUROC. ~22 min on an M4 Max (MPS).
**Internal 0.707 → 0.8165; MIAS chance → 0.733.** Within-CMMD dev AUROC went
0.517 → 0.696 — evidence of real domain learning rather than a larger
shortcut.

Two defects found and fixed during v2, both now contracts in code:
- **Preprocessing skew** (the classic train/serve killer): training normalized
  with grayscale stats (0.449/0.226) while the embedding path used ImageNet
  per-channel stats — cost 0.11 AUROC through the serving path until fixed.
  `FrozenEncoder(mean=, std=)` now makes the stats an explicit contract.
- **L2-normalization loss**: unit-norming embeddings discards the magnitude
  signal a fine-tuned fc uses (cal 0.708 vs 0.807). Fine-tuned tags embed raw
  (`normalize=False`), and the network's own fc is imported as the calibrated
  head instead of refitting a weaker one.

### v3 — the quarantine retrain (legitimacy, not accuracy)
Identical recipe to v2, trained under `splits_v2`, which locks all 349
official-test patients into a `public_bench` split untouchable by any fitting
stage. This is what makes benchmark numbers reportable at all.
**Official CBIS test: 0.742 [0.696–0.786], sens@96% spec 0.202.**
Subgroups (BenchX-style): mass 0.754 / calc 0.736; density a 0.836, b 0.795,
**c 0.623 (weak slice)**, d 0.826; CC 0.768 / MLO 0.720.

### v4 — post-training @1152×896 (the literature recipe, applied)
Method in Part 3. Cal AUROC 0.8331 → 0.8528 during training;
**official CBIS test 0.742 → 0.771 [0.726–0.815]** (now inside the published
0.75–0.88 range), **MIAS external 0.664 → 0.736**. The mechanism is visible in
the subgroups: calcifications +0.067 (0.736 → 0.803) while masses stayed flat
(0.754 → 0.748) — higher resolution recovers microcalcification detail, exactly
the effect Shen et al. built their recipe around. Density-c remains the weak
slice (0.629); resolution alone does not fix dense tissue.

## Part 3 — the post-training method (v4)

Derived from the two published recipes that define the state of the art here
(researched 2026-08-31; links in README):

- *Shen et al. 2019, Scientific Reports* — the canonical CBIS-DDSM recipe:
  1152×896 whole images, patch-classifier pretraining from ROI masks, then
  all-layers fine-tune at 1e-5; flips/±25°/zoom/intensity augs. 0.88 AUC.
- *RSNA 2023 Kaggle 1st place* — ROI-cropped high-res ConvNeXt, cosine LR,
  positive-label smoothing, external data (they used CMMD, as we do).

Gap analysis vs our v3, largest first: **input resolution** (448² discards
~5× the pixels of 1152×896; microcalcifications vanish), then the wider aug
set, then label smoothing. Hence v4 = **progressive high-resolution
fine-tune** (`scripts/posttrain_hr.py`):

| element | setting | source |
|---|---|---|
| init | v3 best (domain-adapted @448) | progressive-resize practice |
| resolution | 1152×896 rectangular letterbox | Shen et al. |
| stage | single all-layers stage, LR 1e-5, cosine, 8 epochs | Shen stage 2 |
| augs | h+v flip, ±25°, zoom 0.8–1.2, intensity ±0.08 | Shen et al. |
| labels | positives smoothed to 0.9 | RSNA-2023 1st place |
| sampling | class-balanced (50/50) + prior re-shift afterwards | axiom A10 |
| selection | best calibration-split AUROC | unchanged |
| quarantine | splits_v2 (`public_bench` untouched) | benchmark eligibility |

Post-processing after every version, always in this order: import fc as the
logistic head → prior-correct 0.5 → cohort prevalence → temperature-scale on
the calibration split → pick the ≥96%-specificity threshold on the threshold
split. Calibration is refit per model and would need refitting per site —
MIAS ECE 0.485 shows calibration does not survive domain shift even when
discrimination does.

## Part 4 — what did *not* work (kept on the record)

- L2-normed refit head on fine-tuned embeddings: −0.10 AUROC vs importing fc.
- Grayscale/ImageNet stat mismatch: −0.11 AUROC, silent, through the serving
  path. Two sealed-test queries were spent on stacks later voided by these
  bugs; the sealed access log records them (4/50 spent total).
- Frozen generic features as a screener: chance on any truly foreign domain.

## Part 5 — the harness A/B (run 2026-08-31)

Same v4 weights on the same 709 benchmark images, alone vs inside the
rule-adjudicated Onco-Harness pipeline: **A 0.771 vs B 0.700**, paired
ΔAUROC −0.071 [−0.099, −0.044], at 29 tool calls/case; 41% deferred to human
with deferrals shown to be uninformative (model-alone performs identically on
deferred vs answered cases). Mechanism: the harness presumes a localizing
detector; a whole-image classifier's fragmented window reads are OOD and the
aggregation destroys the calibrated global signal. Full analysis and the fix
list: `results/ab_harness/CARD.md`. The measurement apparatus did its job —
it refused to certify the architecture in this configuration.

## Part 6 — planned next, in expected-yield order

1. **Patch pretraining from the CBIS ROI masks** (the core of Shen's 0.88,
   AND the fix for the harness A/B: it yields the localizing detector the
   harness architecture presumes): ~50 GB mask series + a 5-class patch stage.
2. Multi-seed / multi-backbone ensemble (Shen's 0.88 → 0.91 came from
   4-model averaging).
3. Flip-TTA at inference (deterministic transform set).
4. Per-site calibration refit for any deployment-shaped claim.

## Part 7 — the patch stage, first results (run 2026-09-01, per TRAINING_HANDOFF.md)

**1b, dataset:** ROI join clean — 3,459/3,553 records ok, 5 unusable (0.14%);
31,299 train patches sampled from 1,673 quarantine-respecting images, all
guards quiet (sanity rail was 1,600–2,000 images).

**1d, v5 warm-start: a wash.** Whole-image fine-tune initialized from the
5-class patch backbone matched v3's calibration AUROC to the third decimal
(0.8336 vs 0.8331) and landed at 0.735 [0.688–0.780] on the bench vs v3's
0.7416 — the patch features neither helped nor hurt at 448px. Recorded as a
negative; the literature's patch-stage gains evidently need the rest of the
recipe (high-res conversion, ensembling) and not the warm start alone. Open:
high-res post-train on v5, and the patch model's detector role (below).

**2, corrected-gold re-scores:** v3 0.7416 [0.694–0.787], v4 0.7707
[0.726–0.814] — both effectively unchanged; the 11 label fixes were already
ranked correctly by the models (the error direction depressed nothing
measurable at this n). Density grids now cover all 709 images: v4 density
a/b/c/d = 0.942/0.773/0.727/0.726. The earlier "density-c ≈ chance" reading
was the mass-only-grid artifact; dense breasts are weak, not blind.
Encoder lineage verified in code for every re-score.

**3a, harness A/B handicap isolation: native input made it WORSE.**
Δ −0.0903 [−0.1208, −0.0603] vs −0.0711 with the 1600px shrink — the shrink
had been *softening* the harness's failure, not causing it. Fragmentation of
the calibrated whole-image signal into OOD window reads is now the isolated
mechanism. Deferral rate fell to 22.6% at native res but deferrals remain
uninformative (model-alone AUROC 0.760 deferred vs 0.753 answered). The
pre-registered rematch — patch detector as proposer — is wired
(`ab_harness_bench.py --proposer patch`) and queued.
