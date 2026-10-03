#!/bin/bash
# Parallel-track chain X2: train-side evaluation of base ∪ xs ∪ dn candidates + dn_cos, in the
# isolated work dir ../work_xd (base artifacts untouched; US features symlinked, only India recomputed).
# usage: ./chain_x2.sh <views> <caps>     e.g. ./chain_x2.sh xs,dn dn=4
set -uo pipefail
VIEWS=${1:?views}; CAPS=${2:-}
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=../../work; D=../../work_xd
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
gate 6 "merge train"
$T $PY src/experiments/xscript.py merge --split train --dst $D --views $VIEWS ${CAPS:+--cap $CAPS} > $L/log_xd_merge_train.txt 2>&1 || { echo merge failed; exit 1; }
tail -2 $L/log_xd_merge_train.txt
mkdir -p $D/feat/train
for f in $L/feat/train/US_*.parquet $L/feat/train/US.done; do ln -sf "$(realpath $f)" $D/feat/train/; done
gate 11 "features train (India only)"
BER_WORK_DIR=$(realpath $D) $T $PY src/run.py features --split train > $L/log_xd_features_train.txt 2>&1 || { echo features failed; exit 1; }
tail -2 $L/log_xd_features_train.txt
gate 6 "featdn train"
BER_WORK_DIR=$(realpath $D) $T $PY src/experiments/dense_view.py features --split train > $L/log_xd_featdn_train.txt 2>&1 || { echo featdn failed; exit 1; }
gate 11 "A2 evaluate _xd"
BER_WORK_DIR=$(realpath $D) BER_EXTRA=featdn BER_TAG=_xd $T $PY src/run.py evaluate --approach 2 > $L/log_xd_eval_A2.txt 2>&1 || { echo evaluate failed; exit 1; }
tail -3 $L/log_xd_eval_A2.txt
echo "chain_x2 done $(date +%T)" | tee $L/chain_x2.done
