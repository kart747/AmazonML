#!/bin/bash
# E-G3 world_sim: build (solo, ~15 GB) -> score shipped/dense_mix models in the test-like world.
set -uo pipefail
export BER_THREADS=16 OMP_WAIT_POLICY=PASSIVE
R=/home/kart/Projects/AmazonML; L=$R/work
cd $R/code/business_entity_resolution
PY=../../.venv/bin/python
trap 'touch $L/world_sim_build.done' EXIT
BER_WORK_DIR=$R/work_xd /usr/bin/time -v $PY src/experiments/world_sim.py build --stage1-tag _xd --stage1-extra dn_cos --extra featx,featdn > $L/log_world_build.txt 2>&1 || { echo "build FAILED"; exit 1; }
grep -E "world:|Maximum resident|Elapsed" $L/log_world_build.txt
touch $L/world_sim_build.done
BER_THREADS=8 BER_WORK_DIR=$R/work_xd $PY src/experiments/world_sim.py score --models s2lab_all,s2lab_dense_mix > $L/log_world_score.txt 2>&1 || echo "score FAILED"
grep WORLD $L/log_world_score.txt
echo "chain_world done $(date +%T)"
