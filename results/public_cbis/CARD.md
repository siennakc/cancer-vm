# Public benchmark: official CBIS-DDSM test split

**Not a medical device. Research results only.**

The recognized public benchmark in our modality (in lieu of BenchX, which is
abdominal CT at multi-TB scale — wrong modality for these models; its subgroup
-grid methodology is adopted here instead). Image-level malignant vs benign
over the official mass+calc test CSVs: **709 images / 349 patients**,
BENIGN_WITHOUT_CALLBACK → negative, patient-clustered 95% CIs.

**Integrity:** splits_v1 had scattered official-test patients into our fitting
splits, so v2 is disqualified here. splits_v2 quarantines all 349 patients as
`public_bench`; both arms below were (re)fit strictly on the remainder
(v3 encoder retrained from scratch, 22 min on-device; heads refit; no internal
sealed queries spent).

| model | AUROC [95% CI] | sens@96% spec | ECE |
|---|---|---|---|
| frozen_v1q (control) | 0.628 [0.577–0.680] | 0.065 | 0.064 |
| **finetune_v3** | **0.742 [0.696–0.786]** | 0.202 | 0.129 |

Literature context: published whole-image CBIS-DDSM classifiers report AUC
≈0.75–0.88 with heavier training and higher resolution. 0.742 from a 448-px,
22-minute laptop fine-tune sits at the low end of that range — reported as-is.

Subgroups (BenchX-style; v3): mass 0.754 / calc 0.736; density a 0.836,
b 0.795, **c 0.623** (the soft spot — dense-c tissue), d 0.826; CC 0.768 /
MLO 0.720. Full grids in the per-model JSONs.

MIAS external validation under the same quarantined arms (v3: 0.664
[0.576–0.741], ECE 0.494): the domain-shift discount and the per-site
calibration failure persist — consistent with results/bench_mias_v1/CARD.md.
Note v3 < v2 on MIAS (0.664 vs 0.733): the quarantine costs ~10% of training
data; that is the price of a clean public-benchmark claim.
