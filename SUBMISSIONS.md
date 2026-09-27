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
| 2 | 2026-09-27 05:03 (built) | 36c39ee+ (stage2_lab.py; now a387fb0), model models/s2lab_all.txt | A2 stage-1 → stage2_lab "all" (54 A2 + 17 new + 21 featx features, no dn_cos), p1 ≥ 1e-3 cascade, threshold+excl τ=0.69 (tune fold 3: 0.98579); `decision.json` alongside | 0.98550 | c67c4d408dc4 | | Files in `output_variants/s2lab_all/`. Validator PASS (re-run independently). Density drop 0.19: eval 0.98323 (−0.0023; A2 −0.0033). Test 3.36 pred/S1 (A2 3.39), 94.2% S1 non-empty. Watch France on LB (pred/S1 3.447→3.404). |
| 3 | 2026-09-27 07:30 (built) | fc01362 (xscript.py merge --cap xs=2,dn=1; stage2_lab.py a387fb0), models work_xd/models/{A2_xd_stage1_fold*,s2lab_all}.txt | base ∪ xs<2 ∪ dn<1 India candidates (work_xd) → A2_xd stage-1 → stage2_lab "all" (92 features, no dn_cos), p1 ≥ 1e-3 cascade, threshold+excl τ=0.68 (tune fold 3: 0.98783); `decision.json` alongside | 0.98742 | f9ff0a26696c | | Files in `output_variants/xd_s2lab_all/`. Validator PASS. Density drop 0.19: eval 0.98498 at normal τ (−0.0024; sub 2 −0.0023). Test: 86.40M candidates (49.9/S1; +1.02M India xs/dn pairs), 5.86M matched = 3.38/S1, 94.3% S1 non-empty. |
| | | | | | | | |

**Audit reserve used (2026-09-27, amazonml-5e):** S1 folds 4–9 are consumed as extra stage-2 training rows (`work_f49/`, chain_f49.sh). From this point only tune fold 3 and eval fold 0 remain held out. Any submission built on f49-trained models has no untouched audit fold.
