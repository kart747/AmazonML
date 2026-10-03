#!/usr/bin/env bash
# Overnight chain: waits for the running train-blocking process (PID given) to exit on
# its own, then runs the next stages sequentially. Stops at the first failure.
set -euo pipefail
PID=$1
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=../../work
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
grep -q "blocking report" $L/log_block_train.txt || { echo "train blocking did not finish cleanly"; exit 1; }
$PY src/run.py features --split train   > $L/log_features_train.txt 2>&1
$PY src/run.py evaluate --approach 1    > $L/log_eval_A1.txt 2>&1
$PY src/run.py evaluate --approach 2    > $L/log_eval_A2.txt 2>&1
$PY src/run.py block --split test       > $L/log_block_test.txt 2>&1
$PY src/run.py features --split test    > $L/log_features_test.txt 2>&1
echo "chain1 done" > $L/chain1.done
