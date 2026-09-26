# Experiment log

Every number here was measured on this laptop (Ryzen 7 5800H 16 threads, 19 GB RAM,
RTX 3050 Laptop 4 GB, unused by the pipeline) unless marked otherwise. Configurations
are in `code/business_entity_resolution/src/ber/config.py`. Per-stage runtime and peak
RSS are appended to `work/telemetry.jsonl`. Reports are in `work/reports/*.json`.

## Fixed validation protocol
- `fold = int(S1 id digits) % 10` (S1-grouped; a target belongs to ≤1 S1, so no true pair spans folds).
- Train folds {1,2}. Tune fold {3} (early stopping, threshold, decision variant). Eval fold {0} (reported only; never used for any choice). Folds 4–9 are untouched (audit reserve).
- Retrieval always runs against the **full** train target pool (10.3M S2+S3) and the full S1 pool (for the reverse view).
- Metric: exact challenge macro F0.5 over **every** S1 of the fold, including S1s whose true matches blocking missed ("realistic" score). Self-test reproduces the spec example (0.714).

---

## E-B0: retrieval pruning sweep (India, 3,000 sampled S1 queries vs the full India pool)
Goal: make exact TF-IDF top-K retrieval affordable on a 16-thread laptop.

| name view | addr max_df | name R@5 | combo R@10 | rev R@8 | union | est. rev time (4.13M targets) |
|---|---|---|---|---|---|---|
| char_wb3, max_df 0.01 | 0.01 | 0.279 | 0.815 | 0.930 | 0.934 | 36 min |
| char_wb3, max_df 0.01 | 0.002 | 0.279 | 0.799 | 0.900 | 0.903 | 22 min |
| char_wb3, max_df 0.002 | 0.01 | 0.113 | 0.836 | 0.863 | 0.886 | 15 min |
| char_wb3, max_df 0.002 | 0.002 | 0.113 | 0.778 | 0.819 | 0.838 | 3 min |

**Conclusion: discard pruning.** Dropping frequent n-grams/tokens destroys recall (0.84–0.93 union).

## E-B1: column-blocked exact top-K (same sample; max_df 0.05 for both fields)
`sp_matmul_topn` keeps a dense accumulator the size of the item set. With 4.1M targets that is ~33 MB per thread, so every multiply-add is a cache miss (~300 ns/op measured). Splitting the items into column blocks and merging the per-block top-K gives **identical results**.

| view | one call | block 262k | **block 65k** | block 16k | recall |
|---|---|---|---|---|---|
| fwd name (3k q) | 17.81 ms/q | 5.78 | **1.02** | 1.02 | 0.420 |
| fwd combo (3k q) | 25.43 ms/q | 8.13 | **1.47** | 1.81 | 0.891 |
| rev combo (10.5k targets) | 3.74 ms/q | 1.48 | **0.25** | 0.27 | 0.969 |

**Kept:** item_block = 65,536. A **17× speed-up** with no loss, which makes unpruned retrieval (max_df 0.05) feasible: about 35 min for India train.

## E-B2: full train blocking (configuration `BLOCK` in config.py)
Views per country: name char_wb-3 top-5, address words top-5, combo top-10 (S1 → target, query folds 0–3), reverse combo top-8 (every target → all S1 of the country).

| partition | query S1 | targets | name | addr | combo | rev pairs | candidates | per query S1 | time | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| India | 353,705 | 4,133,346 | 1.77M | 1.77M | 3.54M | 33.1M | 14.78M | 41.8 | 34.5 min | 8.8 GB |
| US | 529,104 | 6,186,873 | 2.65M | 2.65M | 5.29M | 49.5M | 21.66M | 40.9 | 88.3 min* | 9.0 GB |

*US was slowed by a swap event (a parallel job peaked at 9 GB at 21:23).

**Train candidate quality (query folds 0–3, 882,809 S1, 3,056,221 true pairs; labels for measurement only):**

| metric | value |
|---|---|
| pair recall (union) | **0.98301** (India 0.9732, US ≈0.989) |
| entity full recall | 0.94599 |
| view recall: rev / combo / addr / name | **0.98232** / 0.93549 / 0.74179 / 0.4664 |
| candidates per S1 | mean 41.3, p50 29, p95 100, max 6,399, zero-candidate S1: 0 |

Reading: the reverse view alone finds 98.2% of true pairs; the three forward views add only +0.07 pt. India's recall gap is mainly cross-script (E-B3), with the xs/dn views queued. For reference, Ayan [R] reported 0.989 at ~45 candidates/S1 with a learned dictionary.

## E-B3: India blocking misses → cross-script supplementary views (parallel track)
**Miss analysis** (India train, query folds 0–3, read from `block/train/cand_India.parquet`; labels used for measurement only):

| metric | value |
|---|---|
| pair recall | **0.9732** (views: rev 0.9720, combo 0.8918, addr 0.7332, name 0.4117) |
| entity full recall | 0.9193 |
| oracle macro F0.5 (perfect matcher on these candidates) | **0.99054** |
| candidates per query S1 | mean 41.8, p50 29, p95 105, p99 217, max 6,399 |
| true pairs at reverse rank 0 | 94.5% of all true pairs |

| share of pairs with... | missed (32,933) | found (1,193,708) |
|---|---|---|
| Indic-script target name | **53.3%** | 22.7% |
| empty target address | **33.4%** | 3.1% |
| S1 name shared by >1 S1 | 78.3% | 43.2% |

Mechanism: anyascii turns "लक्ष्मी" into "lksmi", which has almost no 3-grams in common with "laxmi", so every name view fails. Empty-address targets with a name shared by 30–56 S1 are undecidable by design (left as FN).

**Built (not yet run on real data; sequenced after chain2 by the pipeline session):**
- `xscript.py` (xs): learned native→Latin dictionary (training folds only) + reverse target→S1 view for non-Latin-script targets.
- `dense_view.py` (dn): multilingual-e5-small (MIT, 118M) embeddings, reverse top-15 on the GPU, plus a `dn_cos` pair feature (`BER_EXTRA=featdn`).
- `xscript.py measure/merge --views xs,dn --cap dn=K`: recall gain vs base, and an isolated merged work dir.

**Synthetic check only** (tests/synth_data.py, 25% of India copies in Devanagari; the synthetic vocabulary is tiny, so absolutes are meaningless and xs finds nothing there because max_df removes most n-grams):

| A1, synthetic fold 0 | macro F0.5 | macro P | macro R | oracle |
|---|---|---|---|---|
| base | 0.92425 | 0.9983 | 0.7832 | 0.9251 |
| + xs + dn (rank < 8) | 0.92639 | 0.9853 | 0.8088 | 0.9385 |
| + dn_cos feature | **0.92945** | 0.9887 | 0.8084 | 0.9385 |

External evidence [S], Karthikatstuffmama/amazon-ml-challenge-2026, 3% dev slice: e5 dense ∪ key blocking gave recall 0.974 → 0.993 and holdout F0.5 +0.0065 (India +0.0128). Per-country τ was flat or worse. Public LB top-10 cut 0.986. A second repo (Dhruv716-stack) reports local 0.9102 → leaderboard 0.894 (−0.016 with France).

**Label-free test token audit (redecode motivation):** unmapped target-only tokens. France: `ndeg` (N°, 3.3%), `st`/saint, `crs`, `q` (fixed in 23c8756 for test prep). India: anyascii'd native state names (`mharastr` 5.2%, `dilli` 3.3%, `uttr prdes`, `krnatk`, …; ≈25% of targets) are not mapped to S1 state codes, so only the RapidFuzz address features are affected (TF-IDF drops frequent state tokens). Logged for a future train+test re-run.

## E-B4. Blocking runtime profile and the GPU top-K test (parallel track): no-go

**Profile** (train blocking log): sparse top-K takes about 97% of blocking time. India: name 4.9 + addr 2.6 + combo 7.8 + rev 18.0 of 34.5 min. US: 15.9 + 3.6 + 20.3 + 47.1 of 88.3 min. TF-IDF fitting, transforms and the union are under 3% together. Top-K is therefore the only stage where a speed-up would matter.

**GPU implementation** (`src/experiments/gpu_topk.py`, opt-in `BER_TOPK=gpu` in `blocking.topk`, commit b6dc27a):
- The item CSR is stored int32 on the GPU.
- Each 512-query chunk is scattered into a dense V×q matrix.
- cuSPARSE SpMM runs per item block, followed by top-k and a merge.
- It is the exact same computation (positive scores only). Self-test on random data: Jaccard 1.0.

**Benchmark** (RTX 3050 Laptop, 4 GB; India train, all 883k S1, a 1M-target pool containing every true target of the sampled queries; the CPU was contended by A1 training and is still faster):

| view | queries | CPU ms/q | GPU ms/q | speed-up | pair-set Jaccard | recall CPU / GPU |
|---|---|---|---|---|---|---|
| fwd name k5 | 20,000 | 0.832 | 2.245 | 0.37× | 0.787 | 0.4911 / 0.4905 |
| fwd combo k10 | 20,000 | 1.239 | 2.351 | 0.53× | 0.871 | 0.9079 / 0.9080 |
| rev combo k8 | 97,869 | 0.916 | 2.055 | 0.45× | 0.914 | 0.9712 / 0.9712 |

**Decision: no-go.** The pre-registered criteria were ≥2× faster on rev and combo, unchanged recall, and Jaccard ≥0.999. The GPU is 2–2.7× *slower*. Recall is equal, but the pair sets differ at the k-th tie, which matters for names duplicated across many targets. Why it loses: cuSPARSE needs a dense right-hand side, so the GPU performs nnz(items)×q multiply-adds, almost all against zero query entries. sparse_dot_topn touches only the overlapping n-grams. Test blocking stays on the CPU. The opt-in code is inert by default and stays in place.

Other GPU candidates were not pursued: the remaining stages are below 3% of blocking time. The GPU is used where it helps, which is dense e5 encoding (about 3,000 texts/s) and dense top-k.

## E-M1: A1 vs A2 on the eval fold (fold 0, 220,845 S1, 9.09M candidate pairs)
Train folds {1,2}, tuning fold {3} (early stopping and the choice of decision variant and τ), eval fold {0} reported once. Candidates come from E-B2 blocking. Reports are in `work/reports/eval_A{1,2}.json`.

| | features | macro F0.5 | macro P | macro R | pair P | pair R | FP pairs | singleton acc | F0.5 (1 true) | F0.5 (2-3) | F0.5 (4+) | pred pairs/S1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocking oracle | – | 0.99430 | 1.0 | 0.98279 | 1.0 | 0.98298 | 0 | 1.0 | 0.98182 | 0.99383 | 0.99544 | 3.406 |
| A1 pairwise LGBM, threshold+excl τ=0.66 | 29 | 0.96428 | 0.98907 | 0.91923 | 0.98923 | 0.91945 | 7,657 | 0.96010 | 0.90226 | 0.96285 | 0.97292 | 3.220 |
| **A2 stage-1 OOF → stage-2, threshold+excl τ=0.67** | 54 | **0.98045** | **0.99334** | **0.95898** | 0.99369 | 0.95915 | 4,661 | 0.98009 | 0.92929 | 0.98052 | 0.98615 | 3.344 |
| A2, threshold only τ=0.69 | 54 | 0.98036 | 0.99361 | 0.95823 | 0.99394 | 0.95841 | 4,470 | 0.98050 | 0.92848 | 0.98046 | 0.98606 | 3.341 |

- A2 improves macro F0.5 by **+0.0162** over A1, with both recall (+4.0 pt) and precision (+0.4 pt) rising. Tuning-fold F0.5 was 0.98055 against 0.98045 on the eval fold, so there is no sign of tuning overfit.
- Exclusivity (many-to-one) adds only +0.0001 for A2, down from +0.0007 for A1: the stage-2 context features already capture most of it.
- Validation logloss: A1 0.01398 (1,500 iterations, still falling at the cap), A2 stage-1 0.00769, A2 stage-2 0.00638 (816 iterations).
- A2's top features by gain are p1, rev_margin, p1_qgap and r_rev, all context-dependent. Because the test pool has about 2× more distractors, test predicted pairs per S1 must be checked against the 3.34 seen on eval before submitting A2.
- Remaining gap to the blocking oracle is 0.0139, mostly recall. S1 with exactly one true match (0.929) are the weakest group.
- Runtime: A1 evaluate 25.2 min with 5.84 GB peak RSS; A2 evaluate 29.9 min with 9.8 GB peak RSS.
