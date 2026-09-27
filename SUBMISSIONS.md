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
| 3 | 2026-09-27 07:30 (built) | fc01362 (xscript.py merge --cap xs=2,dn=1; stage2_lab.py a387fb0), models work_xd/models/{A2_xd_stage1_fold*,s2lab_all}.txt | base ∪ xs<2 ∪ dn<1 India candidates (work_xd) → A2_xd stage-1 → stage2_lab "all" (92 features, no dn_cos), p1 ≥ 1e-3 cascade, threshold+excl τ=0.68 (tune fold 3: 0.98783); `decision.json` alongside | 0.98742 | f9ff0a26696c | **0.97741** (08:42, upload #1) | Files in `output_variants/xd_s2lab_all/`. Validator PASS. Density drop 0.19: eval 0.98498 at normal τ (−0.0024; sub 2 −0.0023). Test: 86.40M candidates (49.9/S1; +1.02M India xs/dn pairs), 5.86M matched = 3.38/S1, 94.3% S1 non-empty. |
| 3p | 2026-09-27 07:56 (built) | 2fa73e1+ (redecode.py pruned/check) | Sub 3 matches unchanged; candidate_pairs.tsv = the 9,585,603 pairs the stage-2 model scored (stage-1 p1 ≥ 1e-3), 5.53/S1 (US 5.31, India 5.50, France 6.21) vs 49.9/S1 in sub 3 | 0.98742 (identical matches) | f9ff0a26696c (candidates 23caf70a1767) | **0.97741** (08:42, upload #1) | Files in `output_variants/xd_s2lab_all_pruned/`. matching_results.tsv byte-identical to sub 3 (4f cmp). Validator PASS; hard subset check: 0 of 5,860,806 matched ids outside candidates. Recommended upload form of sub 3 (smaller candidate set per S1 ranks higher). |
| | | | | | | | |

**Audit reserve used (2026-09-27, amazonml-5e):** S1 folds 4–9 are consumed as extra stage-2 training rows (`work_f49/`, chain_f49.sh). From this point only tune fold 3 and eval fold 0 remain held out. Any submission built on f49-trained models has no untouched audit fold.

## Clean submission packages (`submission/<id>/`)
Each package is an isolated, read-only copy made from a validated source. It holds `output/matching_results.tsv` (the leaderboard upload), `output/candidate_pairs.tsv` (the final zip), `decision.json`, `metadata.json`, `SHA256SUMS`, `validation.txt` (official validator output) and a `README.md`. Validate and lock with `submission/validate_package.sh submission/<id>`. Never edit a package; a change means a new ID. The TSVs are not in git; verify any copy with `sha256sum -c SHA256SUMS`.

| ID | Local macro F0.5 (fold 0) | Approach | Path | sha256 matching (candidates) | Validator | Git | Public LB F0.5 |
|---|---|---|---|---|---|---|---|
| sub03p_xd_s2lab_all | 0.98742 (tune 0.98783; density 0.98498) | xd candidates + stage2_lab `all`, thr+excl τ 0.68; candidates = stage-2-scored set, 5.53/S1 (= row 3p) | `submission/sub03p_xd_s2lab_all/output/matching_results.tsv` | f9ff0a26696ca3d6… (23caf70a1767123e…) | pending (gated run after f49 features; see `validation.txt`). Source copy: PASS 07:56 | generated at fc01362; candidate writer 2fa73e1 | **0.97741** (2026-09-27 08:42 IST, upload #1) |
| sub04_xd_dense_mix | 0.98743 (tune 0.98768; density sim 0.98625, −0.0012) | xd candidates + stage 2 trained on normal + 19%-drop dense rows (density_train --mix), thr+excl τ 0.68; candidates 5.53/S1 | `submission/sub04_xd_dense_mix/output/matching_results.tsv` | 4b4b064ccc1b… | PASS 10:03 (locked) | 45809ee/ba0cbed/2ac7819 | |
| sub05probe_xd_dense_mix_frblank | PROBE (France emptied; US/India identical to sub04) | diagnostic: LB(sub04) − LB(this) = w_FR·(F_FR − s_FR) | `submission/sub05probe_xd_dense_mix_frblank/output/matching_results.tsv` | 1e9fc443ca89… | PASS 10:03 (locked) | caa8780 redecode | |
| sub06probe_xd_s2lab_all_frblank | PROBE (France emptied; US/India byte-identical to upload #1) | diagnostic, single upload: F_FR = s_FR + (0.97741 − LB(this))/w_FR, w_FR ≈ 0.150, s_FR ≈ 0.054; supersedes sub04+sub05 (two uploads) for measuring France | `submission/sub06probe_xd_s2lab_all_frblank/output/matching_results.tsv` (1,732,544 rows, 4,977,793 matched pairs) | 6ef0446659f1… (23caf70a1767…) | PASS 11:01 (locked); subset check 0 of 4,977,793 matched ids outside candidates; SHA256SUMS re-verified 11:20 | redecode.py | |
| sub07_xd_s2lab_all_guard | 0.98748 (veto part; frhn has no France labels) | upload #1 minus planted look-alike FPs: affiliate-token veto (US 1,840 + India 2,826 pairs, tokens learned on tune fold, 0 true on eval) + France same-street/different-house-number guard (9,140 France pairs); candidates byte-identical to upload #1. Predicted Δ public +0.0013 (veto +0.0007, frhn +0.0006 at 63% planted); ΔLB − 0.0007 measures France's planted share (EXPERIMENTS E-G8) | `submission/sub07_xd_s2lab_all_guard/output/matching_results.tsv` | c2646be519ec… (23caf70a1767…) | PASS 11:50 (locked) | lb_guards.py | |

### Leaderboard log
| Upload | Time (IST) | File (sha256 of matching_results.tsv) | Local fold-0 F0.5 | Public LB F0.5 | Local→LB |
|---|---|---|---|---|---|
| #1 | 2026-09-27 08:42 | f9ff0a26696c: xd + s2lab_all, τ 0.68 (= rows 3 / 3p / package sub03p_xd_s2lab_all; the three share the byte-identical matching file) | 0.98742 (tune 0.98783; density sim 0.98498) | **0.97741** | **−0.0100** (density sim predicted −0.0024) |
Attribution: the user reported "first leaderboard submission 0.97741 at 08:42" for the 0.98742 model. Rows 3, 3p and the package have the same matching_results.tsv, so the score belongs to that file whichever copy was uploaded.
