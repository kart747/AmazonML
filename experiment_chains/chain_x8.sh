#!/bin/bash
# France pseudo-label apply (after the density_train build frees RAM; can overlap its ~3 GB evaluates)
set -uo pipefail
export OMP_WAIT_POLICY=PASSIVE BER_THREADS=8
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=/home/kart/Projects/AmazonML/.venv/bin/python
L=/home/kart/Projects/AmazonML/work; D=/home/kart/Projects/AmazonML/work_xd
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
until [ -f $L/dtrain_build.done ] || ! pgrep -f chain_x7.sh >/dev/null; do sleep 20; done
sleep 60
until [ "$(avail)" -ge 8 ]; do sleep 15; done; echo "[$(date +%T)] start pseudo apply France (avail $(avail) GB)"
BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/pseudo_label.py apply --ref s2lab_all --target France \
    --out /home/kart/Projects/AmazonML/output_variants/xd_s2lab_all_plfr > $L/log_pl_apply_fr.txt 2>&1
grep -E "pseudo labels|plfr:|test pred|subset|Maximum resident|Exit" $L/log_pl_apply_fr.txt
echo "chain_x8 done $(date +%T)" | tee $L/chain_x8.done
