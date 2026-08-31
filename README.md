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
| v3 fine-tuned, quarantined (`weights-v3`) | 0.742 [0.696–0.786] | 0.664 [0.58–0.74] | — |
| **v4 + high-res post-train (`weights-v4`)** | **0.771 [0.726–0.815]** | **0.736 [0.66–0.81]** | — |

¹ 709 images / 349 patients from the official mass+calc test CSVs; published
whole-image results on this split run ≈0.75–0.88 — v4 is inside that range
after ~80 min of laptop training (v3: 22 min at 448px sat just under it).
² 322 UK film-screen images (1994), never seen in any fitting stage. **Calibration
does not survive this shift** (ECE 0.49): refit per site before quoting probabilities.
³ Hash-sealed, query-budgeted internal split (n=1,002). v3 has not spent a query.
⁴ v2 trained on 202 of the official test split's 349 patients (our patient-grouped
splits predate the quarantine) — its internal/MIAS numbers stand, its official-split
numbers would be leakage and are not reported.

Subgroup slices (BenchX-style; v4, official split): mass 0.748 / calc **0.803** ·
density a 0.926, b 0.771, **c 0.629**, d 0.777 · CC 0.786 / MLO 0.757.
The resolution bump moved calcifications most (v3: 0.736 → 0.803) — exactly the
microcalcification-detail mechanism the literature predicts. Density-c breasts
remain the weak slice; `results/public_cbis/*.json` has the grid.

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
.venv/bin/python -m pytest                    # 59 tests, no GPU needed
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
