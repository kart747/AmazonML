#!/bin/bash
# Validate+lock the sub03p package once no heavy job can collide: after chain_f49 (features, featx
# peaks 10-11 GB) AND after chain_x6 (density build 10-12 GB + its own probe validators), so
# only one official validator runs at a time. Then MemAvailable >= 11 GB.
cd /home/kart/Projects/AmazonML
until [ -f work/chain_f49.done ] || ! pgrep -f chain_f49.sh >/dev/null; do sleep 20; done
until [ -f work/chain_x6.done ] || ! pgrep -f chain_x6.sh >/dev/null; do sleep 20; done
echo "[$(date +%T)] f49 and x6 done; waiting for >=11 GB free"
submission/validate_package.sh submission/sub03p_xd_s2lab_all 11
