#!/bin/bash
# address-ambiguity features: normal + dense-mix evaluation on work_xd (after 27's world_sim build)
set -uo pipefail
export OMP_WAIT_POLICY=PASSIVE BER_THREADS=8 POLARS_MAX_THREADS=8
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=/home/kart/Projects/AmazonML/.venv/bin/python
L=/home/kart/Projects/AmazonML/work; export BER_WORK_DIR=/home/kart/Projects/AmazonML/work_xd
trap 'touch $L/chain_x9.done' EXIT   # always release chain_loco (27), also on failure
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
# gate on amazonml-27's world_sim build (15 GB): it touches work/world_sim_build.done at exit
until [ -f $L/world_sim_build.done ]; do sleep 20; done
until [ "$(avail)" -ge 8 ]; do sleep 15; done; echo "[$(date +%T)] start addr_freq (avail $(avail) GB)"
$PY src/experiments/addr_freq.py --src s2lab --dst s2lab_adf > $L/log_adf_build.txt 2>&1 || exit 1
$PY src/experiments/addr_freq.py --src s2lab_dense --dst s2lab_dense_adf --drop 0.19 >> $L/log_adf_build.txt 2>&1 || exit 1
BER_S2LAB_OUT=s2lab_adf /usr/bin/time -v $PY src/experiments/stage2_lab.py evaluate --variant all --drop dn_cos --tag _adf > $L/log_adf_eval.txt 2>&1
grep -E "s2lab_all_adf:|Maximum resident" $L/log_adf_eval.txt
BER_S2LAB_OUT=s2lab_adf BER_S2LAB_DENSE=s2lab_dense_adf /usr/bin/time -v $PY src/experiments/density_train.py evaluate --ref s2lab_all_adf --mix --tag _adf > $L/log_adf_mix.txt 2>&1
grep -E "(ref|new) +normal|Maximum resident" $L/log_adf_mix.txt
echo "chain_x9 done $(date +%T)"
