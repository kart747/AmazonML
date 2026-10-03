#!/bin/bash
L=/home/kart/Projects/AmazonML/work
until [ -f $L/world_sim_build.done ] && [ -f $L/chain_x9.done ]; do sleep 30; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 12 ]; do sleep 30; done
echo "start loco $(date +%T)"
exec $L/chain_loco.sh
