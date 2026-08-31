# Oncoscope

> **The harness now lives in its own repo:
> [siennakc/Onco-Harness](https://github.com/siennakc/Onco-Harness)** (package
> `oncoharness`; extracted at `4ce701d`). This repo is the **model & data
> lane**: ingestion, trained encoders, sealed internal test, external
> benchmark. Serving plugs the harness back in via the `harness` extra.

An LLM harness that orchestrates vision models to detect cancer in medical images —
wrapped in a gated self-improvement loop designed to get measurably better every
cycle **without fooling itself**.

Built from the [Oncoscope Build Tasksheet](TASKSHEET.md) (repo copy of the research
synthesis; task IDs like `T-4.2` reference it). Threat model and PHI boundary:
[THREAT_MODEL.md](THREAT_MODEL.md).

## The architecture in one paragraph

A **deterministic state machine** (`ingest → preflight QC → screen → detect →
verify → aggregate → adjudicate → report`) drives every case. Specialist
**detectors propose** candidate findings at high sensitivity; the **LLM adjudicates**
at exactly one decision node — it plans, weighs verified evidence, and decides
recall / no-recall / defer, but it **never sees pixels and never authors a number**.
Images live in a handle-passing **artifact store**; every tool call and claim lands in
an append-only, hash-chained **evidence ledger**; every promotion must pass a
conjunctive **eval gate** whose rules live in a path the harness cannot write.

```
                     ┌──────────────────────────────────────────────┐
                     │        deterministic state machine           │
  DICOM ──canonical──▶  QC ─▶ detect ─▶ TTA verify ─▶ aggregate     │
                     │                                   │          │
                     │             ┌─────────────────────▼────────┐ │
   artifact store ◀──┼── tools ◀───│  LLM adjudication node       │ │
   (pixels, handles) │  (ledgered) │  (text + handles only)       │ │
                     │             └─────────────────────┬────────┘ │
                     │                          report / defer      │
                     └──────────────────────────────────┬───────────┘
                                                        ▼
                        sealed test set ──▶ conjunctive eval gate (gates/)
```

## Layout

| Path | Contents | Tasksheet |
|---|---|---|
| `src/oncoscope/data/` | Canonical DICOM loader, allowlist de-ID, grouped splits, phantom generator | T-1.1, T-1.2, T-3.2 |
| `src/oncoscope/models/` | DoG candidate detector, embedding features, calibrated head + DFR refit | T-2.1, Part 4 |
| `src/oncoscope/eval/` | Metrics, leakage audit, sealed test set, PASS/FAIL gate | T-1.3, T-1.4, T-3.1 |
| `src/oncoscope/harness/` | Artifact store, evidence ledger, toolbelt, state machine, Claude agent | T-4.1 – T-4.4 |
| `src/oncoscope/serving/` | FastAPI wrapper (same preprocessing as training) | T-2.3 |
| `gates/` | Gate rules — **protected path**, never writable by the harness | T-3.3 |
| `tests/` | Golden DICOM fixtures, leakage, metrics, gate, ledger, phantom end-to-end | T-3.2 |

## Quickstart

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # full suite runs in seconds, no GPU, no data
```

Run the whole pipeline on a synthetic phantom:

```sh
.venv/bin/python - <<'EOF'
from oncoscope.data.phantom import generate_dataset
from oncoscope.harness.ledger import EvidenceLedger
from oncoscope.harness.state_machine import HarnessPipeline
from oncoscope.harness.store import ArtifactStore
from oncoscope.harness.tools import Toolbelt

case = next(c for c in generate_dataset() if c.label == 1)
pipeline = HarnessPipeline(Toolbelt(ArtifactStore("runs/demo/artifacts"),
                                    EvidenceLedger("runs/demo/ledger.jsonl")))
report = pipeline.run_case(case.case_id, case.pixels)
print(report.model_dump_json(indent=1))
EOF
```

To use the LLM adjudicator instead of the rule-based one, install the agent extra
(`pip install -e '.[agent]'`), authenticate the Anthropic SDK, and pass
`LLMAdjudicator(toolbelt)` to `HarnessPipeline`.

## Non-negotiables (see TASKSHEET.md Part 1)

- The LLM never authors pixels, coordinates, or numbers (A3).
- Splits are patient- and site-grouped, always; the leakage audit is a failing CI test (A9).
- The sealed test set is hash-locked, query-budgeted, and access-logged (A6).
- Abstention is a first-class output; deferral ships evidence, not a bare flag (A13).
- The harness can never write its own gates — self-improving, never self-certifying.

## Results so far

| model | internal sealed test (n=1,002) | external MIAS bench (n=322) |
|---|---|---|
| baseline_v1 — frozen IN1K ResNet-50 | AUROC 0.707 · sens@96 0.179 | AUROC **0.504** (chance) |
| finetune_v2 — fine-tuned encoder | AUROC **0.8165** · sens@96 0.337 | AUROC **0.733** · sens@96 0.423 |

The external column is the honest one: the frozen baseline's internal score was
substantially shortcut (site prevalence), and calibration does not survive the
domain shift (v2 external ECE 0.485 — refit per site before quoting any
probability). Details: `results/*/CARD.md`; weights: release `weights-v2`.

## Status

**The harness (Phases 0–4) is complete** and tested end-to-end on phantoms
(45 tests): full toolbelt, TTA + zoom + symmetric FP/FN verification,
Mondrian conformal deferral, image-ablated CI control, and the T-4.5 ablation
runner (`python -m oncoscope.eval.ablation`). Real-data ingestion (RSNA
mammography), the flywheel (Phase 5), and hardening (Phase 6) come next;
per-task status lives in [TASKSHEET.md](TASKSHEET.md#part-8--build-plan).
