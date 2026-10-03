#!/usr/bin/env bash
# Session 5e: density check of s2lab variants -> test predictions (output_variants/) -> resume chain_f49.
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=/home/kart/Projects/AmazonML/work
PY="nice -n 5 ../../.venv/bin/python"
S2=src/experiments/stage2_lab.py
memgate() { until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$1" ]; do sleep 20; done; }
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE
while kill -0 180957 2>/dev/null; do sleep 20; done
memgate 9; echo "density start $(date)"
/usr/bin/time -v $PY $S2 density --variant base,new,featx,all --extra featx > $L/log_s2lab_density.txt 2>&1
for v in featx all; do
  memgate 6; /usr/bin/time -v $PY $S2 predict --variant $v --out /home/kart/Projects/AmazonML/output_variants/s2lab_$v > $L/log_s2lab_predict_$v.txt 2>&1
done
echo "chain_5f done $(date)" > $L/chain_5f.done
nohup bash $L/chain_f49.sh > $L/chain_f49.log 2>&1 &
