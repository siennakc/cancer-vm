# A/B: v4 alone vs v4 inside the harness — official CBIS-DDSM test split

**Not a medical device. Research result only.**

Same weights, same 709 quarantined benchmark images. Arm A: the calibrated v4
whole-image score. Arm B: v4 as the candidate proposer inside Onco-Harness's
rule-adjudicated pipeline (QC → 6-window grid detect → TTA ×3 → zoom re-verify
→ symmetric FP/FN hunt → aggregate → adjudicate), policy
`ab_v4_rule_adjudicated_v1`, everything ledgered.

## Headline: the harness currently subtracts accuracy

| | AUROC | sens@96% spec | cost/case |
|---|---|---|---|
| A — model alone | **0.771 [0.726–0.815]** | 0.219 | 1 forward |
| B — harness (rule-adjudicated) | 0.700 [0.650–0.747] | 0.188 | 29 tool calls, ~60 forwards, 4.3 s |

Paired ΔAUROC (B−A): **−0.071 [−0.099, −0.044]** — significant, not noise.

## The deferral is not earning its keep either

B deferred 291/709 (41%) to a human. Selective analysis from the ledger:

- B on the cases it *chose to answer*: 0.700 — still below A on those same
  cases (0.760). The loss is in the scores themselves, not the case selection.
- A scores deferred cases (0.757) as well as answered ones (0.760), and A's
  per-case error is no higher on the deferred set (0.349 vs 0.364): the
  deferrals are **uninformative** — human review spent without targeting
  difficulty (pitfall register: uninformative deferral).
- Decision level: 7 of 62 `no_recall` cases are cancer (11.3%).

## Why (mechanism, not excuse)

The harness implements A2 — *detector proposes, adjudicator filters* — and its
machinery (TTA reproduction gates, zoom re-verify, FP/FN hunters, window
aggregation) presumes the proposer is a **localizing detector**. v4 is a
whole-image classifier: its calibrated global score was Arm A's entire signal.
Arm B fragments that signal into quadrant/center windows the model never
trained on (out-of-distribution crops), then gates and re-aggregates the
fragments — destroying calibration and ranking that the single forward already
had. The deferral thresholds were tuned on phantom traffic, not real
calibration scores. The ablation runner exists to catch exactly this
(T-4.5: "the ablation that justifies the architecture" — here it declines to).

## What would make the harness earn its keep, in order

1. **A real localizing detector as the proposer** — the patch model trained
   from CBIS ROI masks (the same lever as the literature's 0.88) gives the
   harness genuine candidates; the whole-image v4 score becomes an anchor
   feature fused at aggregation, never replaced.
2. **Fit conformal deferral on real calibration-split scores** so deferrals
   track actual difficulty (Mondrian per site/density).
3. **Crop-augmented training** so window reads are in-distribution.
4. **LLM adjudication arm** (agent.py) — only after the evidence stream it
   would adjudicate is sound; adjudicating today's fragmented evidence would
   measure the LLM's ability to compensate for a broken sensor.

Reproduce: `scripts/ab_harness_bench.py` (full run ≈ 50 min on an M4 Max);
raw report `report.json`; per-case decisions in the run's hash-chained ledger.
