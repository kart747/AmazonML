#!/bin/bash
# E-G5 full leave-one-country-out (France proxy): stage 1 AND stage 2 trained on US only, India scored.
# Isolated work dir ../work_loco (symlinks to work_xd inputs; own models/feat2_loco/s2lab/reports).
set -uo pipefail
export BER_THREADS=12 OMP_WAIT_POLICY=PASSIVE BER_TAG=_loco BER_EXTRA=featdn
R=/home/kart/Projects/AmazonML; X=$R/work_xd; D=$R/work_loco; L=$R/work
cd $R/code/business_entity_resolution
PY=../../.venv/bin/python
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
fail() { echo "FAILED: $1"; exit 1; }
mkdir -p $D/models $D/reports $D/feat/train $D/featdn/train $D/featx $D/s2lab/train
for d in prep block dense; do [ -e $D/$d ] || ln -s $X/$d $D/$d; done
ln -sfn $X/featx/train $D/featx/train
cp -n $X/models/s2lab_all.txt $D/models/ 2>/dev/null
# stage 1 on US only: only US parts visible
for f in $X/feat/train/US_*.parquet; do ln -sf $f $D/feat/train/; done
for f in $X/featdn/train/US_*; do ln -sf $f $D/featdn/train/; done
BER_WORK_DIR=$D $T $PY -c "
import sys; sys.path.insert(0,'src')
from ber.approaches import a2_stage1_models; a2_stage1_models()" > $L/log_loco_stage1.txt 2>&1 || fail stage1
tail -1 $L/log_loco_stage1.txt
# now expose India parts and build stage-2 parts for all (US OOF, India averaged)
for f in $X/feat/train/India_*.parquet; do ln -sf $f $D/feat/train/; done
for f in $X/featdn/train/India_*; do ln -sf $f $D/featdn/train/; done
BER_WORK_DIR=$D $T $PY -c "
import sys; sys.path.insert(0,'src')
from ber.approaches import load_model, build_stage2_parts
build_stage2_parts('train', load_model('A2_loco_stage1_foldA'), load_model('A2_loco_stage1_foldB'))" > $L/log_loco_feat2.txt 2>&1 || fail feat2
tail -1 $L/log_loco_feat2.txt
BER_WORK_DIR=$D BER_S2LAB_SRC=feat2_loco $T $PY src/experiments/stage2_lab.py build --split train --extra featx > $L/log_loco_s2lab.txt 2>&1 || fail s2lab
tail -1 $L/log_loco_s2lab.txt
BER_WORK_DIR=$D $T $PY src/experiments/loco_s2.py --src US --dst India > $L/log_loco_eval.txt 2>&1 || fail eval
grep -E "unseen|in_country" $L/log_loco_eval.txt
echo "chain_loco done $(date +%T)" | tee $L/chain_loco.done
