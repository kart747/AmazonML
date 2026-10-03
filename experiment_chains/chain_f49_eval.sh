#!/usr/bin/env bash
# Session 5e: after chain_f49 -> evaluate s2lab with reserve-fold rows (+/- target-side context).
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=/home/kart/Projects/AmazonML/work
F=/home/kart/Projects/AmazonML/work_f49
PY="nice -n 5 ../../.venv/bin/python"
S2=src/experiments/stage2_lab.py
memgate() { until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$1" ]; do sleep 20; done; }
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE
until [ -f $L/chain_f49.done ]; do sleep 30; done
# 96's density_train build (chain_x6) runs solo first; it touches dtrain_build.done on success
sleep 60
until [ -f $L/dtrain_build.done ] || ! pgrep -f chain_x6.sh >/dev/null; do sleep 30; done
memgate 7
/usr/bin/time -v $PY $S2 evaluate --variant all --drop dn_cos --more-train $F --tag _f49 > $L/log_f49_eval.txt 2>&1
$PY $S2 tctx --split train --more-train $F >> $L/log_f49_eval.txt 2>&1
$PY $S2 tctx --split test >> $L/log_f49_eval.txt 2>&1
$PY $S2 evaluate --variant all+tctx --drop dn_cos --more-train $F --tag _f49 >> $L/log_f49_eval.txt 2>&1
$PY $S2 evaluate --variant all+tctx --drop dn_cos --tag _t >> $L/log_f49_eval.txt 2>&1
echo "chain_f49_eval done $(date)" > $L/chain_f49_eval.done
