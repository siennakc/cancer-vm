#!/bin/bash
# Night GPU queue v3: runs CONCURRENTLY with the download-bound C16 test stream.
set -e
cd "${ONCOSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
echo "=== MEDMNIST SUITE ==="
.venv/bin/python scripts/bench/train_medmnist.py --epochs 8
echo "=== ISIC V2 TRAIN ==="
.venv/bin/python scripts/bench/train_isic_v2.py
echo "=== ISIC V2 EVAL ==="
.venv/bin/python scripts/bench/eval_isic_v2.py
echo NIGHT-QUEUE-DONE
