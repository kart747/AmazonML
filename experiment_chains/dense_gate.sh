#!/usr/bin/env bash
# Run the parallel session's dense-view train build (GPU-bound) during chain2's
# train-features stage, only once enough RAM is free.
set -euo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=../../work
until [ -f $L/log_features_train.txt ]; do sleep 20; done
until [ "$(free -g | awk '/^Mem:/ {print $7}')" -ge 6 ]; do sleep 20; done
echo "start $(date)"
nice -n 10 ../../.venv-dense/bin/python src/experiments/dense_view.py build --split train > $L/log_dense_train.txt 2>&1
echo "done $(date)"
