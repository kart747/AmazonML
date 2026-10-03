#!/bin/bash
# Parallel-track chain X3: test side of work_xd (dense + xs views for test, merge, India features,
# dn_cos). Does NOT predict: prediction is a separate decision after the _xd train-side result.
# usage: ./chain_x3.sh <views> <caps>     e.g. ./chain_x3.sh xs,dn xs=2,dn=1
set -uo pipefail
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE  # other sessions run LightGBM concurrently
VIEWS=${1:?views}; CAPS=${2:-}
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python; DPY=../../.venv-dense/bin/python
L=../../work; D=../../work_xd
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
until [ -f $L/chain_x2.done ] || grep -q failed $L/chain_x2.log; do sleep 30; done
# order agreed with 4f/5e: chain_5f -> x3 (this) -> x4 (s2lab port) -> chain_f49
until [ -f $L/chain_5f.done ]; do sleep 30; done
trap 'touch $L/chain_x3.done' EXIT   # always release chain_f49/x4, also on failure
gate 7 "dense build test (GPU)"
$T $DPY src/experiments/dense_view.py build --split test > $L/log_dense_test.txt 2>&1 || { echo "dense test failed"; exit 1; }
tail -1 $L/log_dense_test.txt
gate 6 "xs build test"
$T $PY src/experiments/xscript.py build --split test > $L/log_xs_build_test.txt 2>&1 || { echo "xs test failed"; exit 1; }
tail -1 $L/log_xs_build_test.txt
gate 7 "merge test"
$T $PY src/experiments/xscript.py merge --split test --dst $D --views $VIEWS ${CAPS:+--cap $CAPS} > $L/log_xd_merge_test.txt 2>&1 || { echo "merge test failed"; exit 1; }
tail -2 $L/log_xd_merge_test.txt
mkdir -p $D/feat/test
for c in US France; do for f in $L/feat/test/${c}_*.parquet $L/feat/test/${c}.done; do ln -sf "$(realpath $f)" $D/feat/test/; done; done
gate 11 "features test (India only)"
BER_WORK_DIR=$(realpath $D) $T $PY src/run.py features --split test > $L/log_xd_features_test.txt 2>&1 || { echo "features test failed"; exit 1; }
tail -2 $L/log_xd_features_test.txt
gate 6 "featdn test"
BER_WORK_DIR=$(realpath $D) $T $PY src/experiments/dense_view.py features --split test > $L/log_xd_featdn_test.txt 2>&1 || { echo "featdn test failed"; exit 1; }
echo "chain_x3 done $(date +%T)" | tee $L/chain_x3.ok
