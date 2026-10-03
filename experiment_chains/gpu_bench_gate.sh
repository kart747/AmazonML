#!/usr/bin/env bash
# Run gpu_topk bench once the dense build has released the GPU and RAM allows.
set -euo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=../../work
until grep -q "^done" $L/dense_gate.log 2>/dev/null; do sleep 15; done
until [ "$(free -g | awk '/^Mem:/ {print $7}')" -ge 5 ]; do sleep 15; done
echo "start $(date)"
nice -n 10 ../../.venv-dense/bin/python src/experiments/gpu_topk.py bench > $L/log_gpu_bench.txt 2>&1
echo "done $(date)"
