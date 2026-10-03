#!/bin/bash
# Parallel-track chain X1: xs build (train) + recall measures. Waits for the A2 test predict
# (chain5) and for enough free RAM before each step; never runs alongside predict.
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=../../work
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
until [ -f $L/chain5.done ]; do sleep 30; done
sleep 300  # let chain_5e ramp up first, so the RAM gates see its real usage (no start race)
gate 11 "density A2 drop 0.19"
$T $PY src/experiments/density.py --approach 2 --drop 0.19 > $L/log_density_A2_d19.txt 2>&1 && cp $L/reports/density_A2.json $L/reports/density_A2_drop19.json
tail -2 $L/log_density_A2_d19.txt
gate 10 "xs build train"
$T $PY src/experiments/xscript.py build --split train > $L/log_xs_build_train.txt 2>&1 || { echo "xs build failed"; exit 1; }
tail -1 $L/log_xs_build_train.txt
gate 6 "measure xs"
$T $PY src/experiments/xscript.py measure > $L/log_measure_xs.txt 2>&1; tail -1 $L/log_measure_xs.txt
gate 6 "measure xs,dn"
$T $PY src/experiments/xscript.py measure --views xs,dn > $L/log_measure_xsdn.txt 2>&1; tail -1 $L/log_measure_xsdn.txt
echo "chain_x1 done $(date +%T)" | tee $L/chain_x1.done
