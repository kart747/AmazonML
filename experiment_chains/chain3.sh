#!/usr/bin/env bash
# Replaces chain2 (22:17): waits for the running train-features process to exit, then
# re-runs TEST prep (glued "N°24" fix), then evaluation, test blocking, test features.
set -euo pipefail
PID=$1
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=../../work
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
grep -q "END features_train_US" $L/log_features_train.txt || { echo "train features did not finish cleanly"; exit 1; }
rm -f $L/prep/s1_test.parquet.done $L/prep/tg_test.parquet.done
$PY src/run.py prepare --split test     > $L/log_prepare_test3.txt 2>&1
$PY src/run.py evaluate --approach 1    > $L/log_eval_A1.txt 2>&1
$PY src/run.py evaluate --approach 2    > $L/log_eval_A2.txt 2>&1
$PY src/run.py block --split test       > $L/log_block_test.txt 2>&1
$PY src/run.py features --split test    > $L/log_features_test.txt 2>&1
echo "chain3 done" > $L/chain3.done
