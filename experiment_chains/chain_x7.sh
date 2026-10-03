#!/bin/bash
# Post-reboot relaunch of density-augmented stage 2 on work_xd (was chain_x6, killed by the 09:10 power loss).
set -uo pipefail
export OMP_WAIT_POLICY=PASSIVE
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=/home/kart/Projects/AmazonML/.venv/bin/python
L=/home/kart/Projects/AmazonML/work; D=/home/kart/Projects/AmazonML/work_xd
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 15; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
trap 'touch $L/chain_x7.done' EXIT
gate 13 "density_train build (solo)"
BER_THREADS=16 BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/density_train.py build --stage1-tag _xd --stage1-extra dn_cos \
    --extra featx,featdn > $L/log_dtrain_build.txt 2>&1 || { echo "build FAILED"; tail -5 $L/log_dtrain_build.txt; exit 1; }
grep -E "Maximum resident|Elapsed" $L/log_dtrain_build.txt
touch $L/dtrain_build.done
for mix in "" "--mix"; do
  gate 6 "density_train evaluate $mix"
  BER_THREADS=8 BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/density_train.py evaluate --ref s2lab_all $mix \
      > $L/log_dtrain_eval${mix/--/_}.txt 2>&1; grep -E "(ref|new) +normal|Maximum resident" $L/log_dtrain_eval${mix/--/_}.txt
done
echo "chain_x7 done $(date +%T)"
