# Finetune v2 — end-to-end fine-tuned ResNet-50 encoder + calibrated fc head

**Not a medical device. Research artifact only.**

First real training run (M4 Max, MPS, 14 epochs ≈ 22 min): ResNet-50 initialized
from IN1K, fine-tuned on the `train` split with class-balanced sampling and
label-invariant augs (hflip / ±10° / scale 0.9–1.1 / intensity jitter), epoch
selection on `calibration` AUROC. The shipped artifact is the backbone as a
`FrozenEncoder` (raw GAP features, grayscale stats) + the network's own fc
imported as the calibrated `LogisticHead` (prior shift 0.5→cohort, temperature)
— identical serving contract to baseline v1. Weights + full resume checkpoint:
GitHub release `weights-v2`.

## Sealed test (n=1,002, one budgeted query)
| metric | baseline v1 | **finetune v2** |
|---|---|---|
| AUROC | 0.707 | **0.8165** |
| Sens @ 96% spec | 0.179 | **0.337** |
| ECE (adaptive) | 0.043 | **0.034** |

## Dev slices (slice_discovery, patient-clustered 95% CI)
| slice | v1 AUROC | v2 AUROC |
|---|---|---|
| pooled (n=336) | 0.652 | 0.736 [0.656–0.798] |
| ddsm (n=156) | 0.701 | 0.755 [0.651–0.849] |
| cmmd (n=180) | **0.517 (chance)** | **0.696 [0.598–0.793]** |

Within-CMMD went from chance to clearly-above-chance — the gain is domain
learning, not a bigger site shortcut.

## Process notes (the honest ledger)
- A preprocessing-skew bug (fine-tune normalized with grayscale stats, encoder
  embedded with ImageNet per-channel stats) initially cost 0.11 AUROC through
  the serving path. Sealed query #3 was spent on that skewed stack and is
  **void**; the access log records it. The skew class now has a named knob
  (`FrozenEncoder.mean/std`) and the card states the contract.
- L2-normalizing fine-tuned embeddings destroys fc-relevant magnitude signal
  (cal 0.708 vs 0.807): `normalize=False` for fine-tuned tags.
- Sealed budget after this run: 46/50 remaining, 4 spent
  (v1, void-v2-refit, void-v2-skew, v2-final).
