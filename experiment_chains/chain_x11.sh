#!/bin/bash
# work_loco self-training variants without the harmful target negatives (after 27's chain_lstrip)
set -uo pipefail
export OMP_WAIT_POLICY=PASSIVE BER_THREADS=6 POLARS_MAX_THREADS=6 BER_WORK_DIR=/home/kart/Projects/AmazonML/work_loco
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=/home/kart/Projects/AmazonML/.venv/bin/python; L=/home/kart/Projects/AmazonML/work
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
until [ -f $L/chain_lstrip.done ]; do sleep 30; done
while pgrep -f "pseudo_label.py --ref s2lab_all --source US" >/dev/null; do sleep 20; done
run() { until [ "$(avail)" -ge 10 ]; do sleep 20; done; echo "[$(date +%T)] start $1 (avail $(avail) GB)"
  /usr/bin/time -v $PY src/experiments/pseudo_label.py --ref s2lab_all --source US --target India --no-oracle "${@:2}" > $L/log_st_$1.txt 2>&1
  grep -E "round|base \(" $L/log_st_$1.txt | cut -c1-200; grep "Maximum resident" $L/log_st_$1.txt; }
run r3_posonly --rounds 3 --hi 0.95 --lo -1 --tag _r3_posonly
run r3_lo001 --rounds 3 --hi 0.95 --lo 0.01 --tag _r3_lo001
echo "chain_x11 done $(date +%T)" | tee $L/chain_x11.done
