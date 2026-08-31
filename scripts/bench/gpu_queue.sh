#!/bin/bash
# Sequential GPU queue, take 2: ISIC first (data complete), PCam after its download settles.
set -e
cd "/Users/mike/cancer model"
echo "=== UNZIP ISIC ==="
cd data/raw/ISIC2018
[ -d train_input/ISIC2018_Task3_Training_Input ] || unzip -o -q train_input.zip -d train_input
[ -d test_input/ISIC2018_Task3_Test_Input ] || unzip -o -q test_input.zip -d test_input
cd ../../..
echo "=== ISIC TRAINING ==="
.venv/bin/python scripts/bench/train_isic.py
echo "=== ISIC EVAL ==="
.venv/bin/python scripts/bench/eval_isic.py
echo "=== WAIT PCAM DOWNLOAD ==="
T=/private/tmp/claude-502/-Users-mike-cancer-model/cac7d3a3-80b7-49a8-9c5f-c6c1faa8ddbe/tasks/b7mahb3h4.output
while ! grep -q PCAM-COMPLETE "$T" 2>/dev/null; do sleep 30; done
echo "=== PCAM TRAINING ==="
.venv/bin/python scripts/bench/train_pcam.py
echo QUEUE-DONE
