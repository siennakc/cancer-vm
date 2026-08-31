# Baseline v1 — frozen ResNet-50 + 5-seed logistic ensemble

**Not a medical device. Research baseline only.**

The axiom-A11 control arm: ImageNet ResNet-50 (IMAGENET1K_V2), frozen, breast-crop
+ 448px letterbox, GAP embeddings, five class-balanced logistic heads merged
exactly (linear ⇒ mean of logits = averaged head), prior-corrected 0.5 → cohort
prevalence, temperature-scaled on the calibration split, threshold chosen on the
threshold split at spec ≥ 0.96. No pixels were fit; the encoder saw zero
mammograms during its training.

## Data
6,821 images / 3,328 patients after the byte-level duplicate audit
(`data/processed/duplicates_audit_v1.json`: 24 images dropped, 6 twin-patient
groups merged, 8 patients dropped for label-conflicting duplicates).
Splits patient-grouped, hash `28c8c7a14e20…`; test sealed before training
(`sealed_test_v1.json`, v1.1-postaudit, budget 50, spent 1).

## Sealed test (1,002 cases, single budgeted query)
| metric | value |
|---|---|
| AUROC | **0.707** |
| Sensitivity @ 96% specificity | **0.179** |
| ECE (adaptive) | 0.043 |

## Dev (slice_discovery split, patient-clustered 95% CI)
| slice | AUROC | sens@96 |
|---|---|---|
| pooled (n=336) | 0.652 [0.568–0.732] | 0.047 |
| ddsm only (n=156) | 0.701 [0.602–0.796] | 0.250 |
| cmmd only (n=180) | **0.517 [0.415–0.628]** | 0.025 |

## Honest readings
- **Within CMMD the model is at chance.** The pooled number is propped up by the
  site-prevalence shortcut (cmmd 70% malignant vs ddsm 44%): generic ImageNet
  features partly identify the *site*, which correlates with the label. Per-site
  rows are the real performance; treat pooled AUROC accordingly.
- Seed spread is 0.0002 — the head is convex, so the "ensemble" is a formality
  at this stage; it exists to keep the T-2.2 plumbing exercised.
- sens@96 of 0.18 is far below any clinical floor. This baseline exists to be
  beaten by the detector-first harness (T-4.3) and a real imaging FM encoder,
  through the gate, not to screen anyone.

## Next levers (in expected-yield order)
1. Real mammography/pathology FM encoder behind `FrozenEncoder` (new tag, re-cache, re-fit).
2. Detector-proposed ROI crops (A1/A2) instead of whole-breast letterbox.
3. Site-stratified DFR refit (`dfr_refit`) once slice discovery flags the gap formally.
