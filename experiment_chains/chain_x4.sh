#!/bin/bash
# Parallel-track chain X4: port the xs/dn candidates (work_xd) onto the stage-2 lab pipeline
# (session 5e, E-M3), train side: featx in work_xd -> s2lab build -> evaluate all --drop dn_cos.
# Like-for-like with work/reports/s2lab_all.json. Writes only inside ../work_xd (own models/reports).
set -uo pipefail
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=../../.venv/bin/python
L=/home/kart/Projects/AmazonML/work; D=/home/kart/Projects/AmazonML/work_xd
T="/usr/bin/time -f %Uuser_%esec_peak%MKB"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
trap 'touch $L/xd_port.done' EXIT   # always release chain_f49, also on failure
fail() { echo "FAILED: $1"; exit 1; }
# order agreed with 4f/5e: chain_5f -> x3 (test side of work_xd) -> x4 (this) -> chain_f49
until [ -f $L/chain_x3.done ]; do sleep 30; done
[ -f $L/chain_x3.ok ] || fail "chain_x3 did not finish cleanly"
[ -L $D/models ] && fail "work_xd/models is a symlink"
# featx: US parts are identical to the base run (same candidates, same prep) -> symlink them
mkdir -p $D/featx/train $D/featx/test
for f in $L/featx/train/US_*; do ln -sf "$f" $D/featx/train/; done
for f in $L/featx/test/US_* $L/featx/test/France_*; do ln -sf "$f" $D/featx/test/; done
gate 8 "featx (features_v2) train+test in work_xd"
BER_WORK_DIR=$D $T $PY src/experiments/features_v2.py > $L/log_xd_featx.txt 2>&1 || fail "featx"
tail -2 $L/log_xd_featx.txt
[ -e $D/feat2 ] || ln -s $D/feat2_xd $D/feat2          # stage2_lab reads feat2/: the _xd stage-2 parts
gate 7 "s2lab build train in work_xd"
BER_WORK_DIR=$D $T $PY src/experiments/stage2_lab.py build --split train --extra featx > $L/log_xd_s2lab_build.txt 2>&1 || fail "s2lab build"
tail -2 $L/log_xd_s2lab_build.txt
gate 5 "s2lab evaluate all --drop dn_cos in work_xd"
BER_WORK_DIR=$D $T $PY src/experiments/stage2_lab.py evaluate --variant all --drop dn_cos > $L/log_xd_s2lab_all.txt 2>&1 || fail "s2lab evaluate"
grep -E "s2lab_all:" $L/log_xd_s2lab_all.txt
# test side of the port (prediction goes to output_variants/, never output/)
gate 8 "stage-2 parts test (_xd stage-1 models)"
BER_WORK_DIR=$D BER_TAG=_xd BER_EXTRA=featdn $T $PY -c "
import sys; sys.path.insert(0, 'src')
from ber.approaches import load_model, build_stage2_parts
build_stage2_parts('test', load_model('A2_xd_stage1_foldA'), load_model('A2_xd_stage1_foldB'))" > $L/log_xd_feat2_test.txt 2>&1 || fail "feat2 test"
gate 7 "s2lab build test in work_xd"
BER_WORK_DIR=$D $T $PY src/experiments/stage2_lab.py build --split test --extra featx > $L/log_xd_s2lab_build_test.txt 2>&1 || fail "s2lab build test"
gate 6 "s2lab predict all (work_xd)"
BER_WORK_DIR=$D $T $PY src/experiments/stage2_lab.py predict --variant all --out /home/kart/Projects/AmazonML/output_variants/xd_s2lab_all > $L/log_xd_s2lab_predict.txt 2>&1 || fail "predict"
grep -E "submission:|validator exit" $L/log_xd_s2lab_predict.txt
echo "chain_x4 done $(date +%T)" | tee $L/chain_x4.ok
