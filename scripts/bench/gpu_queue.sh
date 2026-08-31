#!/bin/bash
# Sequential benchmark queue: ISIC 2018 Task 3, then PCam.
#
# Run from the repository root, or set ONCOSCOPE_ROOT. Previously this script
# hardcoded another machine's checkout path and blocked on a task-output file
# in that machine's temp directory; both are gone. PCam now waits on its own
# extracted data instead, so the queue is portable and its precondition is a
# fact about this machine rather than about some other run.
set -euo pipefail

ROOT="${ONCOSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY (set PYTHON=...)" >&2; exit 1; }

echo "=== UNZIP ISIC ==="
if [ -d data/raw/ISIC2018 ]; then
  (
    cd data/raw/ISIC2018
    [ -d train_input/ISIC2018_Task3_Training_Input ] || unzip -o -q train_input.zip -d train_input
    [ -d test_input/ISIC2018_Task3_Test_Input ] || unzip -o -q test_input.zip -d test_input
  )
else
  echo "data/raw/ISIC2018 missing — download it first; skipping ISIC" >&2
  SKIP_ISIC=1
fi

if [ -z "${SKIP_ISIC:-}" ]; then
  echo "=== ISIC TRAINING ===" && "$PY" scripts/bench/train_isic.py
  echo "=== ISIC EVAL ===" && "$PY" scripts/bench/eval_isic.py
fi

# PCam gate: wait for the download to land, bounded, instead of polling another
# machine's task log forever.
PCAM_DIR="${PCAM_DIR:-data/raw/PCam}"
WAIT_SECS="${PCAM_WAIT_SECS:-0}"
waited=0
while [ ! -d "$PCAM_DIR" ] && [ "$waited" -lt "$WAIT_SECS" ]; do
  sleep 30
  waited=$((waited + 30))
done

if [ -d "$PCAM_DIR" ]; then
  echo "=== PCAM TRAINING ===" && "$PY" scripts/bench/train_pcam.py
else
  echo "$PCAM_DIR missing — skipping PCam (set PCAM_WAIT_SECS to wait for a download)" >&2
fi

echo QUEUE-DONE
