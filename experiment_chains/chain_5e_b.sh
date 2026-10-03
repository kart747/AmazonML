#!/usr/bin/env bash
# chain_5e remainder with 4 threads + passive OpenMP (avoid spin contention with a concurrent LightGBM run)
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=../../work
export BER_THREADS=4 OMP_WAIT_POLICY=PASSIVE
for args in "--variant new" "--variant featx --drop dn_cos" "--variant all --drop dn_cos"; do
  ../../.venv/bin/python src/experiments/stage2_lab.py evaluate $args >> $L/log_s2lab_eval.txt 2>&1
done
echo "chain_5e_b done $(date)" >> $L/chain_5e.log
