#!/bin/bash
# Chained behind the night queue: harness v2 A/B, then C16 train-slide npz re-run.
set -e
cd "${ONCOSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
while ! grep -q NIGHT-QUEUE-DONE runs/night_queue.log 2>/dev/null; do sleep 120; done
echo "=== HARNESS V2 ==="
.venv/bin/python scripts/bench/harness_v2.py
echo "=== C16 TRAIN-SLIDE NPZ RERUN (for the TTA-verify layer gate) ==="
rm -f runs/c16/scores/tumor_00*.json runs/c16/scores/normal_00*.json
.venv/bin/python scripts/bench/infer_c16_slide.py --list-file /tmp/c16_val_slides.txt --batch 128
echo LATE-QUEUE-DONE
