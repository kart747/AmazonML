#!/bin/bash
# Validate a packaged submission with the OFFICIAL validator, record the output, verify hashes and
# the matched-subset-of-candidates rule, then make the package read-only.
# usage: submission/validate_package.sh submission/<id> [min_free_GB]
# The validator with --check-ids peaks at ~8 GB on the full test set. The script waits until
# MemAvailable >= min_free_GB (default 11) so it never pushes a running experiment into swap.
set -euo pipefail
P=$(realpath "${1:?package dir}"); NEED=${2:-11}
R=/home/kart/Projects/AmazonML
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
until [ "$(avail)" -ge "$NEED" ]; do sleep 5; done
cd "$P"
sha256sum -c SHA256SUMS
{
  echo "# official validator, $(date '+%F %T %Z')"
  echo "# python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids"
  echo "# validator sha256: $(sha256sum $R/student_resource/utils/validate_submission.py | cut -c1-64)"
  set +e
  python3 $R/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir $R/student_resource/dataset/test --check-ids 2>&1
  echo "# exit code: $?"
  set -e
  echo
  echo "# hard check: matched ids subset of candidates (redecode.py check)"
  $R/.venv/bin/python $R/code/business_entity_resolution/src/experiments/redecode.py check --out "$P" 2>&1 | grep -v Deprecation | grep -v "^  "
  echo "# sha256 after validation"
  sha256sum output/matching_results.tsv output/candidate_pairs.tsv
} > validation.txt 2>&1
cat validation.txt
grep -q "^PASS" validation.txt && grep -q "# exit code: 0" validation.txt || { echo "VALIDATION FAILED: package left writable"; exit 1; }
chmod -R a-w "$P"
echo "validated and locked: $P"
