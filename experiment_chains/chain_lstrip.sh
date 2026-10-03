#!/bin/bash
# E-G7 legal-form stripping, India in the LOCO world (US-only stage 1+2), no re-blocking.
set -uo pipefail
export BER_THREADS=12 OMP_WAIT_POLICY=PASSIVE BER_TAG=_loco BER_EXTRA=featdn
R=/home/kart/Projects/AmazonML; D=$R/work_lstrip; L=$R/work
cd $R/code/business_entity_resolution
PY=../../.venv/bin/python
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
fail() { echo "FAILED: $1"; exit 1; }
echo "start lstrip $(date +%T)"
[ -e $D ] || $PY src/experiments/legal_strip.py setup --src ../../work_loco --dst $D --split train --country India --feat2 feat2_loco > $L/log_lstrip_setup.txt 2>&1 || fail setup
grep stripped $L/log_lstrip_setup.txt
BER_WORK_DIR=$D $T $PY src/run.py features --split train > $L/log_lstrip_feat.txt 2>&1 || fail feat
tail -1 $L/log_lstrip_feat.txt
BER_WORK_DIR=$D $T $PY -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/experiments')
import features_v2 as f
from ber import config
m = f.learn_dictionary(config.WORK_DIR); f.build('train', m)" > $L/log_lstrip_featx.txt 2>&1 || fail featx
tail -1 $L/log_lstrip_featx.txt
BER_WORK_DIR=$D $T $PY -c "
import sys; sys.path.insert(0,'src')
from ber.approaches import load_model, build_stage2_parts
build_stage2_parts('train', load_model('A2_loco_stage1_foldA'), load_model('A2_loco_stage1_foldB'))" > $L/log_lstrip_feat2.txt 2>&1 || fail feat2
tail -1 $L/log_lstrip_feat2.txt
BER_WORK_DIR=$D BER_S2LAB_SRC=feat2_loco $T $PY src/experiments/stage2_lab.py build --split train --extra featx > $L/log_lstrip_s2lab.txt 2>&1 || fail s2lab
tail -1 $L/log_lstrip_s2lab.txt
echo "chain_lstrip built $(date +%T)" | tee $L/chain_lstrip.done
