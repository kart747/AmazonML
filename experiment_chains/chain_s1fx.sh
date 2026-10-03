#!/usr/bin/env bash
# stage-1 with featx -> feat2_s1fx (train+test) -> s2lab_s1fx build -> evaluate all/featx -> density -> predict
set -euo pipefail
C=/home/kart/Projects/AmazonML/code/business_entity_resolution
PY="nice -n 5 /home/kart/Projects/AmazonML/.venv/bin/python"
cd $C
L=/home/kart/Projects/AmazonML/work
memgate() { until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$1" ]; do sleep 20; done; }
export BER_THREADS=8 OMP_WAIT_POLICY=PASSIVE
memgate 8
while pgrep -f "run.py features" >/dev/null; do sleep 20; done
memgate 8
export BER_EXTRA=featx BER_TAG=_s1fx
$PY -c "
import sys; sys.path.insert(0,'src')
from ber.approaches import a2_stage1_models, build_stage2_parts
ma, mb = a2_stage1_models()
build_stage2_parts('train', ma, mb)
build_stage2_parts('test', ma, mb)"
export BER_S2LAB_SRC=feat2_s1fx BER_S2LAB_OUT=s2lab_s1fx
for sp in train test; do $PY src/experiments/stage2_lab.py build --split $sp; done
$PY src/experiments/stage2_lab.py evaluate --variant all --tag _s1fx --drop dn_cos
$PY src/experiments/stage2_lab.py density --variant all_s1fx --extra featx
$PY src/experiments/stage2_lab.py predict --variant all --tag _s1fx --out ${OUTV:-/home/kart/Projects/AmazonML/output_variants}/s2lab_all_s1fx
echo "chain_s1fx done $(date)" > $L/chain_s1fx.done
