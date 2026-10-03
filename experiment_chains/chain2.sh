#!/usr/bin/env bash
# Overnight chain (replaces chain1): waits for the running train-blocking process to exit
# on its own, re-runs TEST prep (France address normalization added 21:30), then the
# remaining stages sequentially. Stops at the first failure.
set -euo pipefail
PID=$1
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=../../work
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
grep -q "blocking report" $L/log_block_train.txt || { echo "train blocking did not finish cleanly"; exit 1; }
rm -f $L/prep/s1_test.parquet.done $L/prep/tg_test.parquet.done
$PY src/run.py prepare --split test     > $L/log_prepare_test2.txt 2>&1
$PY src/run.py features --split train   > $L/log_features_train.txt 2>&1
$PY src/run.py evaluate --approach 1    > $L/log_eval_A1.txt 2>&1
$PY src/run.py evaluate --approach 2    > $L/log_eval_A2.txt 2>&1
$PY src/run.py block --split test       > $L/log_block_test.txt 2>&1
$PY src/run.py features --split test    > $L/log_features_test.txt 2>&1
echo "chain2 done" > $L/chain2.done
