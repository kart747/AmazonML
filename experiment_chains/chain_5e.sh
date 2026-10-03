#!/usr/bin/env bash
# Session 5e: after chain5 (baseline A2 test predict) -> featx (train+test) -> stage2_lab build -> variants.
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=../../work
PY="nice -n 5 ../../.venv/bin/python"
S2=src/experiments/stage2_lab.py
memgate() { until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$1" ]; do sleep 20; done; }
until [ -f $L/chain5.done ]; do sleep 30; done
memgate 8; echo "featx start $(date)"
/usr/bin/time -v $PY src/experiments/features_v2.py > $L/log_featx.txt 2>&1
for sp in train test; do
  EX=featx
  if [ "$(ls $L/featdn/$sp/*.done 2>/dev/null | wc -l)" -ge "$(ls $L/feat/$sp/*.parquet | wc -l)" ]; then EX=featx,featdn; fi
  echo "$sp extras $EX"
  memgate 6; /usr/bin/time -v $PY $S2 build --split $sp --extra $EX > $L/log_s2lab_build_$sp.txt 2>&1
done
for args in "--variant base" "--variant base --train-folds 1 --tag _f1" "--variant new" "--variant featx --drop dn_cos" "--variant all --drop dn_cos"; do
  memgate 4; $PY $S2 evaluate $args >> $L/log_s2lab_eval.txt 2>&1
done
echo "chain_5e done $(date)" > $L/chain_5e.done
