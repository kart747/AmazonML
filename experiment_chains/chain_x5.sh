#!/bin/bash
# Parallel-track chain X5: P-dense (19% S1 drop) for the ported xd + stage2-lab `all` model.
set -uo pipefail
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=/home/kart/Projects/AmazonML/work; D=/home/kart/Projects/AmazonML/work_xd
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
until [ -f $L/xd_port.done ]; do sleep 20; done
trap 'touch $L/xd_density.done' EXIT
gate 13 "s2lab density (xd, all)"
BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/s2lab_density_tagged.py --stage1-tag _xd --stage1-extra dn_cos \
    --extra featx,featdn --variant all > $L/log_xd_s2lab_density.txt 2>&1 || echo "density FAILED"
grep -E "DENSITY|Maximum resident|Elapsed" $L/log_xd_s2lab_density.txt
echo "chain_x5 done $(date +%T)"
