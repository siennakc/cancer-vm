# External benchmark: MIAS v1 (never-trained site)

**Not a medical device. Research benchmark only.**

322 UK film-screen mammograms (MIAS MiniMammographic Database, official 2012
distribution via the Internet Archive's copy of peipa.essex.ac.uk; research-use
licence, images not redistributed here). Third country, third era, third
acquisition chain — zero overlap with training: byte-hash intersection against
all 6,821 training files = 0 hits; frozen-space near-duplicate scan max cosine
0.949 (fail >0.999). Labels: severity M → malignant (52), B/NORM → negative
(270). MIAS truth mixes biopsy and expert reading — stated caveat.

Hermetic protocol per run: opaque salted ids over bare pixel arrays, no gold in
the staging dir, `sandbox-exec` network denial, stripped env, double-run in
shuffled order (max score delta observed: 1e-16), sealed scorer with query
budget 20 and access log.

## Results (patient-clustered 95% CI; internal sealed test for contrast)

| | internal AUROC | **external AUROC** | external sens@96 | external ECE |
|---|---|---|---|---|
| baseline_v1 (frozen IN1K) | 0.707 | **0.504 [0.42–0.59]** | 0.04 | 0.232 |
| finetune_v2 | 0.8165 | **0.733 [0.65–0.81]** | 0.423 | **0.485** |

## What this says, plainly
1. **The frozen baseline's internal 0.707 was substantially shortcut.** On a
   truly foreign domain it is a coin flip. Without this benchmark that number
   would have looked like competence.
2. **Fine-tuning learned something real.** 0.733 on 1990s UK film it never saw,
   sens@96 0.423 — discrimination survives a brutal domain shift.
3. **Calibration does not survive the shift.** ECE 0.485: probabilities fitted
   on enriched digital cohorts (58% prevalence) are meaningless at MIAS's 16%.
   Discrimination transfers; calibration is per-site infrastructure (axiom A10)
   and must be refit before any probability from this model is repeated.
4. Abstention (|p−0.5|<0.05) covered 85–88% of cases; v2's non-abstained AUROC
   rises to 0.756 — the abstain signal is pointing the right way.

Residual risk, stated: a deliberately malicious adapter with filesystem access
could read MIAS's public labels outside the staging dir. The environment
defends against accidental leakage, metadata shortcuts, state/order cheats,
and network egress — adversarial adapter authors are handled by code review.
