# Deep GitHub Reconnaissance Report: Amazon ML Challenge 2026 Business Entity Resolution

## Executive Summary

Inspected **7 repositories** with actual source code. Our pipeline (TF-IDF/cross-script retrieval → A2_xd stage-1 → s2lab_all stage-2 → 92 features → one-S1-per-target τ≈0.68 → affiliate veto → France pseudo-label) differs **fundamentally** from top public approaches in:

1. **Candidate generation**: We use ~50 candidates/S1 (retrieval + p1≥1e-3 filter); others use 12-30 with multi-channel inverted indices
2. **Feature count**: We have 92; others use 7-72 with heavy emphasis on address token-set ratio and IDF-weighted overlaps
3. **Decision engine**: We use `exclusive_mask` + global τ; others use **greedy bipartite max-weight matching**, **expected-F0.5 per entity**, or **per-country thresholds frozen on OOF**
4. **Ensemble**: We use single LightGBM; multiple repos use **XGBoost + LightGBM + CatBoost** tri-model ensembles
5. **Threshold calibration**: We tune on fold-3 (τ=0.68); others freeze on OOF CV (0.76-0.94) with per-country adaptation
6. **Singleton handling**: We implicitly handle via threshold; others have **explicit singleton gates** (threshold 0.80-0.88, or expected-F floor)

---

## Repository Analysis

### 1. purvanshjoshi/business-entity-resolution
**URL**: https://github.com/purvanshjoshi/business-entity-resolution  
**Key files**: `blocking.py`, `features.py`, `model.py`, `postprocessing.py`

| Aspect | Implementation |
|--------|----------------|
| **Blocking** | Shard-streaming TF-IDF char 3-4-gram cosine, top-k=12 per S1, ~4-6 GB peak RAM |
| **Features** | **7 only**: TF-IDF score, Jaro-Winkler name, token-sort name, token-set name, token-sort addr, postal match (1/0.5/0), name length ratio |
| **Model** | LightGBM, GroupKFold CV, threshold grid search 0.35-0.85 for macro F0.5 |
| **Decision** | **Greedy maximum weighted bipartite matching** (1-to-1 cardinality), singleton protection (τ_singleton=0.30), match cap 11 |
| **Reported** | No LB score disclosed |

**Genuinely different from us**: Minimal features (7 vs 92), explicit greedy bipartite matching, singleton gate, tiny candidate set (12 vs 50).

---

### 2. AvinashMalladi/amazon-ml-challenge-2026-entity-resolution
**URL**: https://github.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution  
**Key files**: `features.py`, `train_model.py`, `infer_submission.py`, `normalization.py`

| Aspect | Implementation |
|--------|----------------|
| **Blocking** | **Multi-channel inverted indices** per country: rare name tokens (IDF-prioritized, top-4), compact name prefix (8-char, +40 weight), substantive numbers (+15), address signatures (+30), rare addr tokens (+12), sole-occupant exact address (+50). Top-16 candidates. |
| **Features** | **14**: Jaccard name, fuzz ratio/token_sort/token_set name, compact ratio/match, first-token match, Jaccard addr, fuzz ratio/token_sort addr, number match/conflict, has-both-addr, exact-addr-sole |
| **Model** | LightGBM (num_leaves=31, lr=0.05), binary_logloss, 250 rounds, no CV shown |
| **Decision** | Per-batch **1-to-many best-match assignment**: sort candidates by score, greedy assign unclaimed targets to highest-scoring S1. Street-number conflict gate (has_num_conflict=1 & token_sort<0.88 → skip). Sole-occupant anchor gets score=10.0+p. |
| **Normalization** | `anyascii` transliteration, legal suffix map (inc→incorporated, pvt→private, sarl/sas/eurl/sci), DBA stripping, domain stripping |
| **Reported** | Claims "F_0.5 > 0.98" (unclear if val or LB) |

**Genuinely different from us**: Multi-channel blocking with **weighted scoring per channel** (not just TF-IDF), per-country indices, **sole-occupant exact address anchor** (+50 weight), **street-number conflict gate**, compact name prefix channel, top-16 not 50.

---

### 3. harshgitty58/Amazon_ML_Challenge
**URL**: https://github.com/harshgitty58/Amazon_ML_Challenge  
**Key files**: `stack.py`, `features.py`, `train.py`, `postprocess.py`  
**Reported**: Val 0.9842 (two-stage + sibling), LB 0.969. **Top LB 0.9906 at handoff (2026-09-26).**

| Aspect | Implementation |
|--------|----------------|
| **Stage-1** | Two LightGBM halves trained on disjoint qhash splits (OOF), each 800 rounds, 127 leaves, lr=0.15. Every training pair gets OOF probability. |
| **Stage-2** | Stage-1 probs + **competition features** (p_rank_q, p_other_q, p_sum_q, p_rank_s1, p_top_s1, p_sum_s1, n_win_s1) + **sibling features** (13 features comparing each candidate to other confident siblings of same S1: n_sib, sib_rn, sib_ra, sib_nn, sib_an, sib_exn, sib_exa, sib_p_ra, etc.). 1000 rounds, 63 leaves, lr=0.08. |
| **Expansion** | `expand.py` adds candidates found through confidently matched siblings (scored by stage-1 like any pair) |
| **Decision** | **Expected-F0.5 per entity** (replaces global threshold): for each S1, keep top-k maximizing 1.25×Σp₁..ₖ/(0.25×Σall p + k); compare vs P(no match)=Π(1-pᵢ). Floor p≥0.05. |
| **Features** | ~40+ pairwise: n_ratio, n_tset, n_tsort, c_ratio, c_tset, c_tsort, c_partial, c_wratio, cp_jw, cp_lev, sk_tset, sk_ratio, dom_ratio, dom_partial, alt_tset, c_overlap, c_inter, first_tok_ratio, name_num, a_tset, a_tsort, a_ratio, a_partial, aw_tset, aw_overlap, num_overlap, num_inter, num_r_extra, num_l_extra, num_jacc, first_num_eq, first_num_lev, first_num_in, postal_eq, state_eq, rank, gap_top, margin, n_cand_q, rank_s1, gap_top_s1, n_cand_s1, n_top1_s1 |

**Genuinely different from us**: **Two-stage with sibling features** (E-SC1 rejected by us but their implementation differs - they use stage-1 probs to find siblings, we used raw features), **expected-F0.5 per-entity decision** (not global τ), **OOF stage-1 probabilities** (no leakage), competition features from stage-1 probs, n_top1_s1 singleton signal.

---

### 4. Noyonika16/Amazon_ML_Challenge_2026
**URL**: https://github.com/Noyonika16/Amazon_ML_Challenge_2026  
**Reported**: Val macro F0.5 0.7337, Precision 95.9%, Recall 59.8%

| Aspect | Implementation |
|--------|----------------|
| **Blocking** | Multi-channel: N_EXACT (exact core name), PRE4 (4-char prefix), N_TOK (rare name tokens ≥4), ADDR_BI (address bigrams), NUM (numeric tokens ≥3). **Top 30 per S1**, >96% recall ceiling, >99.999% reduction. |
| **Features** | **72** (but top importance dominated by address): a_tset (#1, 1.66M gain), a_partial (#2), a_tsort (#3), is_name_match_missing_addr (#4), name_x_addr (#5), n_partial (#6), num_jaccard (#7), num_conflict (#8), n_tset (#9) |
| **Model** | LightGBM (300 trees, lr=0.08, leaves=45, subsample=0.8, colsample=0.8) |
| **Threshold** | Grid search 0.20-0.90 on macro F0.5 → **optimal θ=0.80** (precision 95.9%) |
| **Singleton** | 5.58% in train; empty set scores 1.0, any FP scores 0.0 |

**Genuinely different from us**: **Address features dominate** (a_tset 12× more important than name features), **threshold 0.80** (vs our 0.68), multi-channel blocking with rare tokens + bigrams + numbers, explicit missing-addr flag handling.

---

### 5. Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution
**URL**: https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution  
**Notebook**: Full pipeline in `business_entity_resolution_pipeline.ipynb`

| Aspect | Implementation |
|--------|----------------|
| **Blocking** | **CountryCandidateIndex**: name token inverted index with **IDF/√len(postings)** weighting, postal index (+2.0 weight). **Relative pruning**: drop tokens >1% pool (name), >2% pool (postal). **Sparse fallback**: if no candidates, use rarest token. Max 75 candidates. |
| **Features** | **72** with **IDF-weighted overlaps**: idf_name_jaccard, idf_addr_jaccard, idf_name_diff_penalty (unshared brand tokens), idf_shared_name_weight, idf_shared_addr_weight, plus 9 composite interactions (name×addr, partial×partial, etc.) |
| **Negatives** | **Graded hard negatives**: 10 per entity (top-4 by retrieval score + 3 mid + 3 low) |
| **Model** | **Tri-model ensemble**: XGBoost (40%, depth-wise), LightGBM (35%, leaf-wise), CatBoost (25%, oblivious). 5-fold GroupKFold. **Stacking meta-learner** (LogisticRegression on OOF probs). scale_pos_weight = 1.5×√(N_neg/N_pos) ≈ 3.13. |
| **Threshold** | **Frozen strictly on training OOF**: global 0.830, per-segment (US 0.800, India 0.760, France 0.820). **Never retuned on cold/validation.** |
| **Decision** | **Global Consistency (Fix 1)**: each target awarded to highest-probability claimant (bipartite max-weight matching). Max 10 matches/entity (GT max=11). Conservative France threshold 0.880. |
| **Validation** | **Zero-leakage cold virgin set**: 10,000 entities from row 1M+, completely disjoint from 15k training. **Reported cold macro F0.5: 96.88%**. Rule attribution: 97.61% decisions by GBDT, 2.39% by exact-match rules. |

**Genuinely different from us**: **Tri-model ensemble + stacking**, **IDF-weighted Jaccard**, **graded hard negatives**, **relative frequency pruning** (drops common tokens), **disparity penalty for unshared brand tokens**, **OOF-frozen thresholds per country**, **cold virgin benchmark discipline**, **max-weight bipartite matching** for global consistency.

---

### 6. vigneshkomma/amazon-ml-challenge
**URL**: https://github.com/vigneshkomma/amazon-ml-challenge  
**Simple pipeline**: Token-key blocking (4-char name prefix × country) + TF-IDF char n-gram (2-4) cosine kNN. 7 features (Jaccard, Levenshtein, token-sort, partial-ratio, first-token, numeric overlap, length delta, same-country). LightGBM, threshold tuning on exact metric.

**Genuinely different**: Very minimal, similar to purvanshjoshi but simpler.

---

### 7. Dhruv716-stack/business-entity-resolution-v1
**URL**: https://github.com/Dhruv716-stack/business-entity-resolution-v1  
**Reported**: Val 0.9102, LB 0.894

| Aspect | Implementation |
|--------|----------------|
| **Blocking** | 4-pass: (1) 5 compound exact-match keys, (2) TF-IDF char n-gram NN fallback, (1b) 4 hashed order-independent token keys (no-space name, house#×street-word, sorted name tokens, address-word pairs). **93.1% recall** at ~32 candidates/S1. |
| **Features** | ~44: fuzzy string metrics, token overlap, structured-field matches, typo-normalized comparisons, per-entity candidate-ranking features |
| **Model** | XGBoost / LightGBM / CatBoost, GPU auto-detection with CPU fallback. **No ensemble beat plain XGBoost** (threshold 0.940). |
| **Decision** | **Greedy global 1-to-1 assignment** (each candidate id claimed by at most one S1) + Macro-F0.5-optimal threshold search |
| **Key insight** | "Verified from training ground truth that every S2/S3 id belongs to at most one S1 entity" — same as our exclusive_mask |

**Genuinely different from us**: Multi-pass blocking with hashed token keys (catches what exact+TF-IDF miss), explicit greedy global assignment, threshold 0.94 (much higher than our 0.68), no per-country adaptation.

---

## Comparison Matrix: Our Pipeline vs Public Repos

| Dimension | Our Pipeline | Public Consensus |
|-----------|--------------|------------------|
| **Candidates/S1** | ~50 (retrieval + p1≥1e-3) | **12-30** (multi-channel inverted indices, top-k) |
| **Blocking** | TF-IDF char n-gram cosine + cross-script | **Multi-channel**: rare tokens, prefixes, numbers, signatures, sole-occupant, bigrams |
| **Features** | 92 (A2_xd + s2lab_all) | **7-72**, address token-set ratio dominates importance |
| **Model** | Single LightGBM (2-stage) | **Tri-model ensemble** (XGB+LGB+Cat) common; 2-stage with sibling features (harshgitty58) |
| **Threshold** | τ≈0.68 (fold-3 tune) | **0.76-0.94**, **frozen on OOF CV**, per-country |
| **Decision** | exclusive_mask + global τ | **Greedy max-weight bipartite matching**, **expected-F0.5 per entity**, singleton gate |
| **Singleton** | Implicit via τ | **Explicit gate** (τ_singleton=0.30-0.88, or expected-F floor) |
| **France** | Pseudo-label round (E-X7) | **Country-agnostic features**, per-country threshold, or conservative France τ (0.82-0.88) |
| **Calibration** | None explicit | **scale_pos_weight = 1.5×√(N_neg/N_pos)**, OOF probability blending |
| **Hard negatives** | Cascade p1≥1e-3 | **Graded: top-k retrieval + mid + low** (Aamod007: 10/entity) |

---

## TOP 10 Genuinely Different Techniques

| # | Technique | Source Repo | Status vs Our Pipeline |
|---|-----------|-------------|------------------------|
| 1 | **Multi-channel inverted index blocking** (rare tokens, compact prefixes, numbers, signatures, sole-occupant exact address, bigrams) | AvinashMalladi, Noyonika16, Aamod007 | **NOT IMPLEMENTED** — we use single TF-IDF channel |
| 2 | **Tri-model ensemble + stacking meta-learner** (XGBoost 40% + LightGBM 35% + CatBoost 25%) | Aamod007, Dhruv716 | **NOT IMPLEMENTED** — we use single LightGBM |
| 3 | **Expected-F0.5 per-entity decision** (optimizes macro F0.5 locally per S1) | harshgitty58 | **NOT IMPLEMENTED** — we use global τ |
| 4 | **Greedy max-weight bipartite matching** (1-to-1 cardinality enforcement) | purvanshjoshi, Dhruv716, Aamod007 | **PARTIALLY IMPLEMENTED** — we have `exclusive_mask` but it's per-S1 not global max-weight |
| 5 | **Sibling features from stage-1 probabilities** (competition + sibling context) | harshgitty58 | **PARTIALLY IMPLEMENTED** — we tried E-SC1 (rejected) but their implementation uses **stage-1 probs**, not raw features |
| 6 | **IDF-weighted Jaccard + disparity penalty** for unshared brand tokens | Aamod007 | **NOT IMPLEMENTED** — we have raw Jaccard but not IDF-weighted + diff penalty |
| 7 | **Relative frequency pruning** (drop tokens >1-2% of pool) + sparse fallback | Aamod007 | **NOT IMPLEMENTED** — we don't prune blocking vocabulary |
| 8 | **OOF-frozen per-country thresholds** (never retune on cold/val) | Aamod007, Noyonika16 | **NOT IMPLEMENTED** — we tune τ on fold-3 (single split) |
| 9 | **Sole-occupant exact address anchor** (exact address match when unique in S1) | AvinashMalladi, Aamod007 | **NOT IMPLEMENTED** — we have address features but not this specific gate |
| 10 | **Street-number conflict gate** (has_num_conflict=1 & token_sort<0.88 → reject) | AvinashMalladi | **NOT IMPLEMENTED** — we have number_conflict detector but not as hard gate |

---

## Which 3 Have Highest Potential to Explain Our LB Gap (0.98742 → 0.97741/0.978)

### 1. **Per-Country OOF-Frozen Thresholds + Conservative France Gate** (Aamod007)
**Why**: Our τ=0.68 is a single global threshold tuned on fold-3. France (unseen in train) likely has different precision/recall tradeoff. Aamod007 uses France τ=0.880, India 0.760, US 0.800 frozen on OOF. This directly addresses **distribution shift** — the #1 candidate for local→LB gap.
- **Evidence**: Noyonika16 also uses θ=0.80 (much higher than our 0.68) because macro F0.5 weights precision 2×.
- **Our risk**: Low threshold on France admits FPs that destroy per-entity F0.5.

### 2. **Expected-F0.5 Per-Entity Decision** (harshgitty58)
**Why**: Global τ is suboptimal for macro F0.5 because entities have different candidate count and score distributions. Expected-F selection adapts k per entity: keep top-k maximizing 1.25×Σp₁..ₖ/(0.25×Σp + k). This is **mathematically optimal** for the metric.
- **Evidence**: harshgitty58 reports 0.984 val with this; our ablation on decision rules (Table 15) showed only 0.00022 spread — but we only tested global τ variants, not per-entity expected-F.

### 3. **Multi-Channel Blocking with Sole-Occupant & Number Channels** (AvinashMalladi, Aamod007)
**Why**: Our blocking recall ceiling may be lower than we think. Aamod007's relative pruning + sparse fallback ensures **no entity stranded**. AvinashMalladi's sole-occupant exact address (+50 weight) catches matches TF-IDF misses (transliteration, missing addr). Our p1≥1e-3 filter may drop true matches with low model probability but strong structural signals.
- **Evidence**: Dhruv716 reports 93.1% blocking recall; we don't measure ours on test.

---

## Cheapest Falsification Experiments

| Technique | Experiment | Compute/RAM | Needs LB Submission? |
|-----------|------------|-------------|---------------------|
| **Per-country OOF thresholds** | Re-run stage2_lab with τ_US=0.80, τ_IN=0.76, τ_FR=0.82 (from Aamod007) on fold-0 eval; compare macro F0.5 vs τ=0.68 | ~2 min, <1 GB (reuse scores_s2lab_all.parquet) | **NO** |
| **Expected-F0.5 per entity** | Implement `postprocess.py::ef_select` on fold-0 scores_s2lab_all.parquet; compare macro F0.5 vs τ=0.68 | ~30 sec, <500 MB | **NO** |
| **Sole-occupant address feature** | Add `exact_addr_sole` binary (1 if S1 addr unique in train & matches candidate addr) to features; retrain s2lab_all; eval fold-0 | ~3 min, ~3 GB | **NO** |
| **IDF-weighted Jaccard + diff penalty** | Add `idf_name_jaccard`, `idf_addr_jaccard`, `idf_name_diff_penalty` to s2lab_all feature set; retrain | ~5 min, ~4 GB | **NO** |
| **Tri-model ensemble** | Train XGBoost + CatBoost on s2lab_all features (same CV); blend 40/35/25; eval fold-0 | ~15 min, ~6 GB | **NO** |
| **Multi-channel blocking** | Implement AvinashMalladi's 6-channel inverted index; generate candidates; measure recall@k vs TF-IDF on fold-0 GT | ~10 min, ~4 GB | **NO** |
| **Greedy max-weight bipartite matching** | Replace `exclusive_mask` with purvanshjoshi/Aamod007 global assignment on fold-0 probs; eval | ~1 min, <1 GB | **NO** |
| **Sibling features (stage-1 probs)** | Compute harshgitty58's competition + sibling features from s2lab_all fold-0 probs; train stage-2 | ~5 min, ~3 GB | **NO** |
| **Graded hard negatives** | Retrain s2lab_all with Aamod007's 10 graded negatives/entity (top-4 retrieval + 3 mid + 3 low) | ~5 min, ~3 GB | **NO** |
| **Conservative France τ + max-weight matching** | Combine per-country τ + Aamod007 global consistency on fold-0 | ~2 min, <1 GB | **NO** |

**All experiments run on existing fold-0 scores / features — zero leaderboard consumption.**

---

## Files/Functions Worth Studying (Exact References)

| Repo | File | Function/Class | What to Extract |
|------|------|----------------|-----------------|
| AvinashMalladi | `infer_submission.py` | `run_inference()` lines 88-180 | Multi-channel candidate scoring with weighted channels; sole-occupant logic |
| AvinashMalladi | `features.py` | `extract_pairwise_features()` | 14-feature set; `has_num_conflict`, `exact_addr_sole` |
| harshgitty58 | `postprocess.py` | `ef_select()` | Expected-F0.5 per-entity selection (20 lines) |
| harshgitty58 | `stack.py` | `prob_context()`, `sibling_features()` | Competition + sibling features from stage-1 probs |
| purvanshjoshi | `postprocessing.py` | `apply_graph_postprocessing()` | Greedy max-weight bipartite matching + singleton gate |
| Aamod007 | Notebook cells 5-7 | `CountryCandidateIndex.query()`, `compute_pair_features()` | IDF-weighted Jaccard, relative pruning, graded negatives |
| Aamod007 | Notebook cell 8 | Threshold freezing on OOF, per-segment thresholds | Frozen threshold discipline |
| Noyonika16 | `README.md` §4 | Feature importance table | Address features dominate — prioritize a_tset, a_partial, a_tsort |
| Dhruv716 | `blocking.py` (not fetched but described) | Multi-pass: exact keys + TF-IDF + 4 hashed token keys | Blocking recall 93.1% — highest reported |
| Our code | `src/experiments/stage2_lab.py` | `evaluate()` with `--variant all --tag` | Retrain point for all feature/threshold experiments |

---

## Conclusion & Recommended Next Steps

**Priority 1 (Highest ROI, zero LB risk)**:
1. **Per-country OOF thresholds** — implement τ_US=0.80, τ_IN=0.76, τ_FR=0.88 on existing fold-0 scores; measure macro F0.5 delta
2. **Expected-F0.5 per entity** — port harshgitty58's 20-line `ef_select` to our fold-0 scores; measure delta
3. **Sole-occupant exact address feature** — add to s2lab_all, retrain, eval fold-0

**Priority 2 (Medium effort, high signal)**:
4. **IDF-weighted Jaccard + brand-token disparity penalty** — add 3 features, retrain
5. **Tri-model ensemble (XGB+LGB+Cat) + stacking** — train on s2lab_all features
6. **Global max-weight bipartite matching** — replace `exclusive_mask` with purvanshjoshi/Aamod007 assignment

**Priority 3 (Structural changes)**:
7. **Multi-channel blocking** — implement AvinashMalladi's 6-channel index; measure recall@30 vs our 50
8. **Two-stage with sibling features** — re-implement harshgitty58's stack.py using our stage-1 OOF probs

**Decision**: All Priority 1-2 experiments are **offline falsifiable** on fold-0. Run them sequentially; any showing >+0.001 macro F0.5 on fold-0 warrants a leaderboard submission (we have 3 remaining).