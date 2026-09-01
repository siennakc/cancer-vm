# Oncoscope — an open mammography malignancy model

A breast-cancer screening model you can train from scratch on a laptop:
fine-tuned ResNet-50 encoder + calibrated logistic head, trained on two public
mammography collections (126 GB, credential-free), evaluated on the **official
CBIS-DDSM test split** and externally validated on MIAS. Every number below is
reproducible from a clean clone with the commands in this README.

> **Not a medical device.** Research and education only. No output of this
> model may be used for diagnosis, screening, or treatment decisions.

The agent harness that orchestrates this model (state machine, evidence
ledger, eval gates) lives in its own repo: **Onco-Harness**.

## Results

Image-level malignant-vs-benign, AUROC with patient-clustered 95% CIs.

| model | public benchmark: CBIS-DDSM official test¹ | external: MIAS² | internal sealed test³ |
|---|---|---|---|
| v1 frozen IN1K + refit head | 0.628 [0.577–0.680] | 0.512 (chance) | 0.707 |
| v2 fine-tuned (`weights-v2`) | *disqualified⁴* | 0.733 [0.65–0.81] | **0.8165** |
| v3 fine-tuned, quarantined (`weights-v3`) | 0.742 [0.694–0.787] | 0.664 [0.58–0.74] | — |
| **v4 + high-res post-train (`weights-v4`)** | **0.771 [0.726–0.814]** | **0.736 [0.66–0.81]** | — |
| v5 patch-warm-start, 448px | 0.735 [0.688–0.780] — a wash⁵ | — | — |

Public-benchmark numbers re-scored 2026-09-01 against the corrected
any-malignant gold with verified encoder lineage; v3/v4 moved <0.001 (the 11
fixed labels were already ranked correctly by the models).

¹ 709 images from the **349 patients** named in the official mass+calc test
CSVs. That is a patient-complete *superset* of the official 645-image test
split: 31 patients appear in both the official train and test CSVs, and
quarantining them wholesale (64 extra images) is the stricter choice, but it
means this set is composed slightly differently from the one published papers
score. Published whole-image results on the official split run ≈0.75–0.88;
v4 lands in that range after ~80 min of laptop training (v3: 22 min at 448px,
just under it).
² 322 UK film-screen images (1994), never seen in any fitting stage. **Calibration
does not survive this shift** (ECE 0.49): refit per site before quoting probabilities.
³ Hash-sealed, query-budgeted internal split (n=1,002), defined against
`splits_v1`. **v3/v4 are disqualified here, exactly as v2 is on the public
benchmark**: `splits_v2` re-splits with a new seed, so 381 of the 499 sealed
patients now sit in a v3/v4 fitting split. No query has been spent on them,
and since 2026-08-31 the scorer enforces this in code: `SealedTestSet.score`
requires the candidate's fit manifest (or an explicit `external=True`) and
refuses any manifest that puts a sealed patient in a fitting split.
⁴ v2 trained on 202 of the official test split's 349 patients (our patient-grouped
splits predate the quarantine) — its internal/MIAS numbers stand, its official-split
numbers would be leakage and are not reported.
⁵ Warm-starting the whole-image encoder from the 5-class patch model
(31,299 patches, 1,673 quarantine-respecting images) matched v3's calibration
AUROC exactly (0.8336 vs 0.8331) and did not beat it on the bench at 448px —
a reportable negative. The patch model's other role (localizing detector for
the harness rematch) and the high-res post-train on v5 remain open.

Subgroup slices (BenchX-style; v4, official split): mass 0.748 / calc **0.803** ·
CC 0.786 / MLO 0.757. The resolution bump moved calcifications most
(v3: 0.736 → 0.803) — exactly the microcalcification-detail mechanism the
literature predicts.

Density slices (v4, re-scored 2026-09-01, now covering all 709 after the
column-spelling fix): a 0.942, b 0.773, **c 0.727**, d 0.726. The earlier
"density-c indistinguishable from chance" reading was an artifact of the grid
silently covering masses only — with calcifications included, dense-breast
performance is weak but real (c: [0.630–0.813]). Dense breasts (c/d) remain
the soft half of the model. `results/public_cbis/*.json` has the full grid.

## Get the model

GitHub releases (sha256 in notes): [`weights-v4`](../../releases/tag/weights-v4)
(best, benchmark-eligible) · [`weights-v3`](../../releases/tag/weights-v3) ·
[`weights-v2`](../../releases/tag/weights-v2) (best internal-only).
Each ships `best_model.pt` + `checkpoint.pt` (full optimizer state — resume
training with `--resume`). Calibrated heads are committed in `results/*/head.json`.

```python
import torch, numpy as np, sys; sys.path.insert(0, "src")
from oncoscope.models.encoder import FrozenEncoder
from oncoscope.models.head import LogisticHead

enc = FrozenEncoder(tag="v3", weights_path="best_model.pt", normalize=False,
                    mean=(0.449,)*3, std=(0.226,)*3)   # grayscale train stats
head = LogisticHead.load("results/finetune_v3/head.json")
prob = head.predict_proba(enc.embed_batch([pixels]))    # pixels: float [0,1] HxW
```

## How it was trained

**Data — two public TCIA collections, no credentials** (`DATA_LICENSES.md`):

| site | collection | images used | labels |
|---|---|---|---|
| `ddsm` | CBIS-DDSM (US, film→digital) | 3,093 full mammograms | biopsy-proven pathology |
| `cmmd` | CMMD (China, FFDM) | 3,728 | biopsy-proven, per breast |

- `scripts/fetch_cmmd.py` / `fetch_cbis.py` download via the public NBIA API,
  SHA-256 per series into append-only manifests; failures recorded and retried,
  never silently dropped.
- **Byte-level duplicate audit** (`content_audit`): both collections enroll some
  women twice under different patient IDs with byte-identical DICOMs — six twin
  groups merged for splitting, four label-conflicting pairs dropped entirely.
  UID checks cannot see these; content hashing can (`duplicates_audit_v1.json`).
- **Label gold is any-malignant** (revised 2026-08-31): the CSVs carry one row
  per abnormality, and first-wins dedup had labeled 11 multi-finding images
  benign despite a biopsy-proven malignant row (4 inside the public benchmark,
  2 in the sealed set). `scripts/backfill_labels_v1.py` fixed the committed
  table; numbers published before the revision are annotated stale in their
  `results/*.json` and will be re-scored — the old error *depressed* AUROC,
  so re-scoring can only help.
- One DICOM decoder for training, serving, and eval (`data/dicom_canonical.py`):
  MONOCHROME1, rescale, VOI LUT, pixel spacing handled once.

**Splits — patient-grouped always** (`data/splits.py`, axiom A9): train /
calibration / threshold / slice_discovery / test, hash-locked. `splits_v2`
additionally quarantines all 349 official-test patients into `public_bench`,
untouchable by any fitting stage — that quarantine is what makes the v3
benchmark numbers legitimate.

**Recipe** (`scripts/finetune_encoder.py`, Apple M4 Max, MPS, ~22 min):
ResNet-50 from IN1K → breast-crop + 448 letterbox render cache → class-balanced
sampling → hflip / ±10° rotation / 0.9–1.1 scale / intensity jitter →
AdamW 1e-4, cosine, 14 epochs, batch 16 → epoch selection on calibration AUROC.

**Calibration** (`models/head.py`, axiom A10): the network's fc imported as a
logistic head → analytic prior-correction (balanced 0.5 → cohort prevalence) →
temperature scaling on the disjoint calibration split → operating threshold at
≥96% specificity chosen on the disjoint threshold split. Both cohorts are
biopsy-enriched (~58% malignant vs ~0.5% screening reality): never quote PPV
from these test sets, and re-shift the prior before any deployment claim.

**Evaluation discipline:** the internal test set is hash-sealed with a
50-query budget and an access log (4 spent, 2 on stacks later voided — the log
keeps us honest). Public-benchmark numbers come only from quarantined models.

## Reproduce everything

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]' pandas openpyxl
.venv/bin/python scripts/fetch_cmmd.py        # 23 GB
.venv/bin/python scripts/fetch_cbis.py        # 103 GB
.venv/bin/python scripts/build_dataset.py     # case table + audit + splits
.venv/bin/python scripts/make_splits_v2.py    # official-test quarantine
.venv/bin/python scripts/render_cache.py      # 448px training cache (2.6 GB)
.venv/bin/python scripts/finetune_encoder.py --splits data/processed/splits_v2.json --run runs/finetune_v3
.venv/bin/python scripts/cache_embeddings.py --tag resnet50_ft_v3_448_raw --weights runs/finetune_v3/best_model.pt --raw --gray-stats
.venv/bin/python scripts/refit_heads_v3.py    # calibrated heads
.venv/bin/python scripts/eval_public_cbis.py --tag resnet50_ft_v3_448_raw --head runs/finetune_v3_head/head.json --name finetune_v3
.venv/bin/python -m pytest                    # 35 tests, no GPU needed
```

## Layout

| path | contents |
|---|---|
| `src/oncoscope/data/` | TCIA client, DICOM canonicalizer, case tables, audits, grouped splits |
| `src/oncoscope/models/` | encoder (frozen/fine-tuned), calibrated head, DFR refit |
| `src/oncoscope/eval/` | metrics, sealed test set, leakage audit, gate |
| `scripts/` | fetch → build → train → calibrate → evaluate, in order |
| `results/` | committed metrics, cards, calibrated heads — the source of every number above |
| `data/metadata/` | label CSVs + download manifests (images themselves are never committed) |

Data licences and citations: `DATA_LICENSES.md`. Task provenance: `TASKSHEET.md`.
Full intervention-by-intervention ledger (what made it better, what didn't): `TRAINING_HISTORY.md`.

**Harness A/B** (same weights, same benchmark): model alone 0.771 vs the
rule-adjudicated harness — **Δ −0.071** with the original 1600px input, and
**Δ −0.090 [−0.121, −0.060]** in the fair native-resolution re-run
(2026-09-01). The audit's fairness caveat resolved in the unexpected
direction: giving the harness full resolution made it *worse*, which pins the
fault on window fragmentation (out-of-distribution sub-reads destroying the
calibrated whole-image signal), not resolution. The honest answer to "does
the harness help?" remains **no — and the mechanism is now isolated**:
`results/ab_harness/CARD.md`. The decisive rematch (patch detector as
proposer, `--proposer patch`) is queued.
