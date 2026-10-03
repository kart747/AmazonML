#!/bin/bash
# Parallel-track chain X6: density-augmented stage 2 on work_xd (+ validator runs for the LB probes).
set -uo pipefail
export OMP_WAIT_POLICY=PASSIVE
cd /home/kart/Projects/AmazonML/code/business_entity_resolution
PY=/home/kart/Projects/AmazonML/.venv/bin/python
L=/home/kart/Projects/AmazonML/work; D=/home/kart/Projects/AmazonML/work_xd; O=/home/kart/Projects/AmazonML/output_variants
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "[$(date +%T)] start $2 (avail $(avail) GB)"; }
until [ -f $L/chain_f49.done ]; do sleep 30; done
gate 13 "density_train build (solo)"
BER_THREADS=16 BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/density_train.py build --stage1-tag _xd --stage1-extra dn_cos \
    --extra featx,featdn > $L/log_dtrain_build.txt 2>&1 || { echo "build FAILED"; tail -5 $L/log_dtrain_build.txt; exit 1; }
grep -E "Maximum resident|Elapsed" $L/log_dtrain_build.txt
touch $L/dtrain_build.done
gate 9 "validator probe tau080"
$PY /home/kart/Projects/AmazonML/student_resource/utils/validate_submission.py --matching $O/xd_s2lab_all_tau080/matching_results.tsv \
    --candidate $O/xd_s2lab_all_tau080/candidate_pairs.tsv --test-dir /home/kart/Projects/AmazonML/student_resource/dataset/test --check-ids \
    > $L/log_validate_tau080.txt 2>&1; tail -1 $L/log_validate_tau080.txt
for mix in "" "--mix"; do
  gate 6 "density_train evaluate $mix"
  BER_THREADS=8 BER_WORK_DIR=$D /usr/bin/time -v $PY src/experiments/density_train.py evaluate --ref s2lab_all $mix \
      > $L/log_dtrain_eval${mix/--/_}.txt 2>&1; grep -E "^\S+ .*(ref|new) +normal|Maximum resident" $L/log_dtrain_eval${mix/--/_}.txt
done
gate 9 "validator probe tau055"
$PY /home/kart/Projects/AmazonML/student_resource/utils/validate_submission.py --matching $O/xd_s2lab_all_tau055/matching_results.tsv \
    --candidate $O/xd_s2lab_all_tau055/candidate_pairs.tsv --test-dir /home/kart/Projects/AmazonML/student_resource/dataset/test --check-ids \
    > $L/log_validate_tau055.txt 2>&1; tail -1 $L/log_validate_tau055.txt
echo "chain_x6 done $(date +%T)" | tee $L/chain_x6.done
