#!/usr/bin/env bash
# Cross-script (xs) retrieval view: build -> measure -> gate -> isolated full evaluation.
#
#   bash src/experiments/xs_pipeline.sh [MIN_RECOVERED_SHARE]
#
# Needs the base run's work/prep, work/block/{train,test} and (for the comparison)
# work/reports/eval_A2.json. Writes ONLY: work/block/<split>/xs_* (new files) and the isolated
# dir ${BER_WORK_DIR:-work}_xs/. Base artifacts are never modified.
# Gate: continue past `measure` only if xs recovers at least MIN_RECOVERED_SHARE (default 0.10)
# of the base blocking misses in some country; otherwise stop (the view is not worth a re-run).
# Cost on 16 threads: build ≈10 min, merge ≈6 min, features ≈40 min, evaluate A2 ≈60 min,
# test features ≈30 min, predict ≈10 min. Peak RAM ≈ the base pipeline's (≈10 GB).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)            # code/business_entity_resolution
PY=${PY:-python}
MIN_SHARE=${1:-0.10}
BASE=${BER_WORK_DIR:-$(cd "$HERE/../.." && pwd)/work}
XS=${BASE}_xs
cd "$HERE"
export BER_WORK_DIR=$BASE
$PY src/experiments/xscript.py build --split train
$PY src/experiments/xscript.py measure
$PY - "$BASE/reports/xscript_recall.json" "$MIN_SHARE" <<'EOF'
import json, sys
rep, thr = json.load(open(sys.argv[1])), float(sys.argv[2])
best = max((r["recovered_pairs"] / max(1, r["base_misses"]) for r in rep.values()), default=0.0)
print(f"xs gate: best recovered share of base misses = {best:.3f} (threshold {thr})")
sys.exit(0 if best >= thr else 3)
EOF
$PY src/experiments/xscript.py build --split test
$PY src/experiments/xscript.py merge --split train --dst "$XS"
$PY src/experiments/xscript.py merge --split test --dst "$XS"
export BER_WORK_DIR=$XS
$PY src/run.py features --split train
$PY src/run.py evaluate --approach 2
$PY - "$BASE/reports/eval_A2.json" "$XS/reports/eval_A2.json" <<'EOF'
import json, os, sys
if not os.path.exists(sys.argv[1]):
    print(f"no base report at {sys.argv[1]}; run `run.py evaluate --approach 2` on the base work dir to compare")
    sys.exit(0)
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("macro_f05", "macro_precision", "macro_recall", "pair_recall", "singleton_accuracy", "false_positive_pairs"):
    print(f"{k:22} base {a['eval'][k]}  xs {b['eval'][k]}")
print(f"{'blocking oracle':22} base {a['eval_blocking_oracle']['macro_f05']}  xs {b['eval_blocking_oracle']['macro_f05']}")
EOF
$PY src/run.py features --split test
BER_OUT_DIR="$XS/output" $PY src/run.py predict --approach 2
echo "xs pipeline done: compare above; the submission candidate is in $XS/output"
