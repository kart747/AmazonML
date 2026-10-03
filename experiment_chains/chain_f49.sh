#!/usr/bin/env bash
# Session 5e: stage-2 training rows from reserve folds 4..9 in isolated ../work_f49
# (forward views only, stored rev reused) -> features -> stage-2 parts (averaged p1, as on test)
# -> featx -> s2lab build. Heavy steps gated on free RAM.
set -euo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=/home/kart/Projects/AmazonML/work
F=/home/kart/Projects/AmazonML/work_f49
PY="nice -n 5 ../../.venv/bin/python"
memgate() { until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$1" ]; do sleep 20; done; }
mkdir -p $F/reports
[ -e $F/prep ] || ln -s $L/prep $F/prep
[ -e $F/models ] || ln -s $L/models $F/models
export BER_WORK_DIR=$F BER_QUERY_FOLDS=4,5,6,7,8,9
# 5d xs port (work_xd train side) has priority; it touches work/xd_port.done when finished
until [ -f $L/xd_port.done ] && [ -f $L/chain_x3.done ] && [ -f $L/xd_density.done ]; do sleep 30; done
[ "$(date +%H)" -lt 17 ] || { echo "too late for f49 (deadline plan: nothing heavy after 21:00)"; exit 0; }
memgate 12; echo "block start $(date)"
/usr/bin/time -v $PY src/experiments/block_more_folds.py --src $L > $L/log_f49_block.txt 2>&1
memgate 13; echo "features start $(date)"
/usr/bin/time -v $PY src/run.py features --split train > $L/log_f49_features.txt 2>&1
memgate 8; echo "stage2 parts start $(date)"
/usr/bin/time -v $PY -c "
import sys; sys.path.insert(0,'src')
from ber.approaches import load_model, build_stage2_parts
build_stage2_parts('train', load_model('A2_stage1_foldA'), load_model('A2_stage1_foldB'))" > $L/log_f49_feat2.txt 2>&1
memgate 8; echo "featx start $(date)"
/usr/bin/time -v $PY src/experiments/features_v2.py > $L/log_f49_featx.txt 2>&1
memgate 6; echo "s2lab build start $(date)"
/usr/bin/time -v $PY src/experiments/stage2_lab.py build --split train --extra featx > $L/log_f49_s2lab.txt 2>&1
echo "chain_f49 done $(date)" > $L/chain_f49.done
