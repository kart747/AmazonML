#!/usr/bin/env bash
# After chain3 (test blocking + features): baseline A2 test prediction + official validator.
set -uo pipefail
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
L=../../work
until [ -f $L/chain3.done ]; do sleep 30; done
../../.venv/bin/python src/run.py predict --approach 2 > $L/log_predict_A2.txt 2>&1
echo "chain5 exit $?" > $L/chain5.done
