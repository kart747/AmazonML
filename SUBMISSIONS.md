# Leaderboard submission log

The challenge rules require a version history of every submission, and shortlisting is
based on the submitted solutions. Add one row per upload, **before** uploading.
Record `sha256sum <dir>/matching_results.tsv` so each score maps to an exact file, and the
git commit plus decision (`decision.json` for `redecode.py` variants) so each file can be regenerated.

Limits: 5 uploads per day. The public leaderboard is a subset of test; the private leaderboard decides.
Never tune repeatedly against the public score; use it to check direction only.

| # | Date/time (IST) | Git commit | Pipeline / decision | Local fold-0 macro F0.5 | sha256 (first 12) | Public LB F0.5 | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-27 01:46 (built) | 98cdcf3 pipeline (run.py predict --approach 2) | A2 baseline: stage-1 OOF → stage-2, threshold+excl τ=0.67 (tune fold 3) | 0.98045 | 20ade8c0db29 | | Files in `output/sub01_A2_base/`. Validator PASS. Test: 85.38M candidates (49.3/S1), 5.87M matched pairs = 3.39/S1 (eval 3.34), 94.2% of S1 non-empty. |
| | | | | | | | |
