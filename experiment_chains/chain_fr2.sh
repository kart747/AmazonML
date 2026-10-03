#!/bin/bash
# E-G1 France re-normalization: rebuild France test parts in ../work_fr with the shipped models.
set -uo pipefail
export BER_VALIDATOR=/home/kart/Projects/AmazonML/work/novalidate.sh BER_THREADS=6 OMP_WAIT_POLICY=PASSIVE
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python; D=/home/kart/Projects/AmazonML/work_fr; L=/home/kart/Projects/AmazonML/work
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
fail() { echo "FAILED: $1"; exit 1; }
while pgrep -f "run.py features --split test" >/dev/null; do sleep 10; done; [ -f $D/feat/test/France.done ] || fail features
tail -1 $L/log_fr_features.txt
BER_WORK_DIR=$D nice $T $PY src/experiments/features_v2.py > $L/log_fr_featx.txt 2>&1 || fail featx
tail -1 $L/log_fr_featx.txt
BER_WORK_DIR=$D BER_TAG=_xd BER_EXTRA=featdn nice $T $PY -c "
import sys; sys.path.insert(0, 'src')
from ber.approaches import load_model, build_stage2_parts
build_stage2_parts('test', load_model('A2_xd_stage1_foldA'), load_model('A2_xd_stage1_foldB'))" > $L/log_fr_feat2.txt 2>&1 || fail feat2
tail -1 $L/log_fr_feat2.txt
BER_WORK_DIR=$D nice $T $PY src/experiments/stage2_lab.py build --split test --extra featx > $L/log_fr_s2lab_build.txt 2>&1 || fail s2lab_build
tail -1 $L/log_fr_s2lab_build.txt
BER_WORK_DIR=$D nice $T $PY src/experiments/stage2_lab.py predict --variant all --out /home/kart/Projects/AmazonML/output_variants/fr_renorm_full > $L/log_fr_predict.txt 2>&1
grep -E "submission:|validator exit" $L/log_fr_predict.txt
$PY src/experiments/fr_renorm.py compare --a ../../work_xd --b ../../work_fr > $L/log_fr_compare.txt 2>&1
echo "chain_fr done $(date +%T)" | tee $L/chain_fr.done
