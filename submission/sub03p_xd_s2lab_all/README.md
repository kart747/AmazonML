# sub03p_xd_s2lab_all

**What to upload to the leaderboard portal:** `output/matching_results.tsv`, and nothing else.
**What goes into the final zip:** both files in `output/`. `code/business_entity_resolution/src/package.py` builds the zip.

| | |
|---|---|
| Approach | xd candidates + stage2_lab `all` (the team's best local pipeline on 2026-09-27) |
| Local macro F0.5 (eval fold 0, reported once) | **0.98742** (P 0.99665, R 0.97018, 2,328 FP pairs, singleton accuracy 0.99045) |
| Tune fold 3 | 0.98783 (the decision was chosen here) |
| Test-like density check (19% of S1 dropped) | 0.98498 (−0.0024) |
| Decoder | threshold + target exclusivity, τ = 0.68 (`decision.json`) |
| Rows | 1,732,544 S1 rows + header, both files; 5,860,806 matched pairs; 1,633,049 S1 non-empty |
| Candidates | 9,585,603 pairs = **5.53 per S1** (`decision_candidates.json`) |
| sha256 | `SHA256SUMS` (matching `f9ff0a26…`, candidates `23caf70a…`) |
| Official validator | `validation.txt` |
| Git | generated at `fc01362`; candidate writer committed in `2fa73e1`; details in `metadata.json` |

## What it is
1. **Retrieval, per country.** Exact sparse TF-IDF top-K: name char 3-gram (5), address words (5), name+address combo (10), and the reverse combo target→S1 (8). India adds two views over non-Latin-script targets: **xs**, a native→Latin token dictionary learned from training-fold pairs (rank < 2), and **dn**, multilingual-e5-small, MIT, rank < 1. On test this gives 49.9 pairs per S1.
2. **Stage 1 (learned filter).** A2_xd LightGBM (pairwise + candidate-graph context + e5 cosine, cross-fitted). Only pairs with p1 ≥ 1e-3 go on. That is 5.53 per S1 on test; on eval fold 0 the filter drops 61 of 765,153 true pairs.
3. **Stage 2 (matcher).** stage2_lab `all` LightGBM, 92 features: p1 profile/sibling, cluster support, twins, glued names, learned-dictionary name similarity, house-number edit types. It runs only on the stage-1 set.
4. **Decision.** Keep a pair if p ≥ 0.68 and the pair is the target's highest-scoring S1.

`output/candidate_pairs.tsv` is exactly the set stage 2 scores (the rules: "whatever your model actually runs inference over"). Disclosure: stage 1 scores the full retrieved list, and several stage-2 features aggregate those stage-1 scores. `output/matching_results.tsv` is byte-identical to submission 3 (`output_variants/xd_s2lab_all/`). Only the candidate file differs from submission 3, which listed 49.9 pairs per S1.

## Integrity
- Both TSVs were copied from `output_variants/xd_s2lab_all_pruned/` and checked byte-identical with `cmp`.
- The official validator was run on these copies (`validation.txt`).
- A hard check that every matched id is among its S1's candidates found 0 violations (`redecode.py check`; the official validator only warns on this).
- After validation, the directory is read-only (`chmod -R a-w`). Do not edit it. A changed submission gets a new directory and a new ID.
- The TSVs are not in git (they exceed GitHub's file limits). `SHA256SUMS` and `metadata.json` are, so any copy can be verified with `sha256sum -c SHA256SUMS`.

## Reproduce
See `metadata.json` → `reproduce`. The chain scripts actually used are `work/chain_x2.sh`, `chain_x3.sh` and `chain_x4.sh` (see EXPERIMENTS.md E-X1/E-X2).
