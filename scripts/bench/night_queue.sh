#!/bin/bash
# Overnight GPU queue: wait for C16 validation -> MedMNIST suite -> ISIC v2 -> eval.
set -e
cd "${ONCOSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
while pgrep -f infer_c16_slide > /dev/null; do sleep 60; done
echo "=== MEDMNIST SUITE ==="
.venv/bin/python scripts/bench/train_medmnist.py --epochs 6
echo "=== ISIC V2 TRAIN ==="
.venv/bin/python scripts/bench/train_isic_v2.py
echo "=== ISIC V2 EVAL ==="
.venv/bin/python scripts/bench/eval_isic_v2.py
echo NIGHT-QUEUE-DONE
