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

## E-M2. A2 gap decomposition and decoder lab (session 5e; eval fold 0, τ=0.67 thr+excl)
Where the 0.0139 gap to the blocking oracle comes from. **It is entirely matcher error on pairs already in the candidate set.** Blocking misses are already inside the oracle's 0.9943.

| bucket (fold 0) | pairs | note |
|---|---|---|
| true pairs among candidates | 752,132 | |
| FN (true, p < τ) | 18,222 (2.4%) | 15,959 S1 partially missed, 695 S1 fully missed |
| … of which target address empty | 8,861 (49%) | often an identical-name empty-address lookalike from another S1 exists: largely undecidable |
| … of which target is not its reverse-best S1 | 8,759 (48%) | name shared by several S1 |
| … of which address matches (a_tset ≥ 0.9, non-empty) | 6,720 | typos plus house-number noise inside TRUE pairs (620/619, 454/54, 103/91), phonetic transliteration ("hoyait gyalaksi eksports praibhet" = white galaxy exports private), coined DBA names at the same address ("caloflux", "evobrix") |
| … coined single-token OOV target names | 1,858 FN / 264 FP | including glued names ("earnosethroat") |
| FP (false, p ≥ τ) | 4,721 | **78% of FP targets belong to no S1**: orphan lookalikes, same address plus a one-character name edit (meentech→meentex). Test has ~23% more targets per S1, so more of these |

FN p distribution: median 0.32, 10% below 0.05; they are not near-misses at τ.

**Decoder lab** (`decoder_lab.py --approach 2`, parameters chosen on fold 3, reported on fold 0, paired bootstrap SE):

| decoder | params | eval F0.5 | Δ vs thr+excl (SE) | pair P | pair R | singleton acc |
|---|---|---|---|---|---|---|
| threshold+excl (A2 default) | τ 0.67 | 0.98045 | 0 | 0.9937 | 0.9591 | 0.9801 |
| threshold | 0.69 | 0.98036 | −0.00009 (0.00004) | 0.9939 | 0.9584 | 0.9805 |
| relative+excl | 0.5, α 0.7 | 0.98063 | +0.00018 (0.00007) | 0.9940 | 0.9583 | 0.9714 |
| per_source+excl | 0.65/0.65 | 0.98040 | −0.00004 | 0.9933 | 0.9599 | 0.9789 |
| caps(S2≤5,S3≤6) | 0.67 | 0.98045 | +0.00001 | 0.9937 | 0.9591 | 0.9801 |
| hn_adjust | 0.68, b 0.05 | 0.98035 | −0.00009 | 0.9936 | 0.9591 | 0.9799 |
| expected_f05 calibrated+excl | λ 0 | 0.98064 | +0.00019 (0.00008) | 0.9944 | 0.9569 | 0.9689 |

**Decision: decoding is saturated** (≤ +0.0002, and the two best rules lower singleton accuracy, which is risky under test density). Not adopted. The remaining gain must come from better scores.

**Cascade finding:** stage-2 rows with p1 < 1e-3 are 88.6% of rows (train folds 0–3) but hold 0.01% of positives. On fold 0, no pair with p1 < 1e-3 reaches p ≥ 0.67 under the full A2. A pruned stage 2 (`src/experiments/stage2_lab.py`) therefore trains in ~2 GB instead of 9.8 GB. Status: [IMPLEMENTED] [SMOKE TESTED on synthetic data]. Real-data run is queued after test blocking (RAM).

## E-D1. A2 under test-like density, plus label-free test diagnostics (parallel track)
**Setup.** Run with `density.py --approach 2 --drop 0.19`, using the A2 models from E-M1 unchanged. It removes 19% of train S1 at random, so their true targets become owner-less distractors. It then recomputes every pool-dependent context feature and the p1/sibling features, rescores folds 3 and 0, and re-selects τ on fold 3.
Why 0.19: train has 3.465 true matches and 4.68 targets per S1, which leaves 1.22 orphan targets per S1. If test S1 have the same number of true matches, test's 5.75 targets per S1 give 2.29 orphans per S1. Dropping 19% gives (1.22 + 0.19·3.465)/0.81 = 2.31.

| fold 0 | macro F0.5 | macro P | macro R | FP pairs/S1 | pred pairs/S1 | singleton acc |
|---|---|---|---|---|---|---|
| A2 normal density, τ 0.67 | 0.98045 | 0.99334 | 0.95898 | 0.0211 | 3.344 | 0.9801 |
| A2 test-like density, τ 0.67 | **0.97716** (−0.0033) | 0.98949 | 0.95844 | 0.0340 | 3.356 | 0.9673 |
| A2 test-like density, τ re-selected (0.73) | 0.97730 (+0.00014) | 0.99089 | 0.95581 | 0.0292 | 3.342 | 0.9701 |

The τ curve under density is flat (0.70: 0.97732, 0.75: 0.97727, 0.67: 0.9772). **Keep τ 0.67.** Density costs precision (FP per S1 +61%, singleton accuracy −1.3 pt), and no threshold recovers it. The fix has to come from scores: features that tell an orphan lookalike from a true copy.

**Label-free test vs eval** (`redecode.py diagnose --approach 2`):

| | cand/S1 | E[m]/S1 (Σp) | pred/S1 @0.7 |
|---|---|---|---|
| India eval / test | 41.8 / 50.2 | 3.375 / 3.438 | 3.309 / 3.339 |
| US eval / test | 40.7 / 48.8 | 3.427 / 3.494 | 3.359 / 3.403 |
| France test | 47.8 | 3.582 | 3.438 |

Test pred/S1 rises by +0.03 to +0.045 and Σp by +0.065, but the density simulation raises pred/S1 by only +0.012. There are two possible explanations, and nothing label-free can tell them apart:
- test S1 have slightly more true matches, which is harmless;
- test orphans are harder lookalikes than simulated ones, which would mean up to about 4× the simulated FP increase.

Expected local-to-LB drop is at least 0.003. An LB score near 0.97 would point to harder orphans.

**Cost:** density.py peaked at **13.5 GB RSS** in 9.5 min (its README says ~8 GB), and it pushed the shared box 4 GB into swap. Run it as a solo job.

## E-M3. Stage-2 lab: pruned stage 2 + new feature sets (session 5e; `src/experiments/stage2_lab.py`)
Same protocol as E-M1: train folds {1,2}, early stopping, variant choice and τ on tune fold 3, reported on eval fold 0. Stage-2 rows are cascade-pruned to p1 ≥ 1e-3 (E-M2): 2.07M training rows instead of 18.2M. All other rows are rejected. Stage-1 models and p1 are the unchanged A2 ones. Extra columns are hstacked row-aligned from work/featx (features_v2.py, first real-data run) and computed in stage2_lab.augment (17 "new" p1-aware features: cluster support vs the S1's confident set, identical twins, per-source p1 rank, space-stripped name, exact/contained address, target-name OOV share). dn_cos is excluded (`--drop dn_cos`; 5d owns that ablation).

| variant | features | tune F0.5 | eval F0.5 | macro P | macro R | FP pairs | singleton acc | F0.5 (1 true) | pred/S1 | iters | fit RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 (E-M1, reference) | 54 | 0.98055 | 0.98045 | 0.99334 | 0.95898 | 4,661 | 0.9801 | 0.9293 | 3.344 | 816 | 9.8 GB |
| s2lab_base (pruned, same 54) | 54 | 0.98073 | 0.98047 | 0.99331 | 0.95946 | 4,755 | 0.9818 | – | 3.347 | 645 | 1.5 GB |
| s2lab_base, fold 1 only (learning curve) | 54 | 0.98010 | 0.97987 | 0.99332 | 0.95812 | 4,817 | 0.9832 | – | 3.343 | 289 | 1.4 GB |
| s2lab_new | 71 | 0.98182 | 0.98145 | 0.99413 | 0.96077 | 4,137 | 0.9849 | – | 3.350 | 921 | 1.4 GB |
| s2lab_featx | 75 | 0.98549 | 0.98527 | 0.99634 | 0.96505 | 2,572 | 0.9896 | 0.9467 | 3.356 | 830 | 1.8 GB |
| s2lab_all (new + featx) | 92 | **0.98579** | **0.98550** | 0.99686 | 0.96487 | **2,142** | 0.9908 | – | 3.354 | 690 | 1.6 GB |

Reading:
- The cascade prune is lossless (base ≥ A2 on both folds).
- Stage 2 is data-limited: halving the training data costs 0.0006 on tune. This motivates the reserve folds 4–9 (`block_more_folds.py`, chain_f49; this uses up the audit reserve, noted in SUBMISSIONS).
- featx is the big gain (+0.0048 tune over base, FP −46%). Top featx gains: t_unm_maxidf, q_unm_maxidf (IDF of the rarest one-sided name token, which is exactly the orphan-lookalike signature from E-M2), n_map_ratio, t_match_idf. All are label-free, except that the transliteration dictionary is learned from train-fold labels only.
- all vs featx is only +0.0003 on tune, below the +0.001 gate. The tie is broken by the density simulation (the "new" features target orphan lookalikes). **Eval fold 0 is not used for this choice.**
- Status: [REAL DATA TESTED] [MACRO F0.5 VERIFIED on fold 0] for all rows. Density check and test predictions are pending (chain_5f).
- Runtime note: a concurrent LightGBM fit with 16 OpenMP threads each (two jobs, 16 cores) slowed both about 10× (spin-wait contention). Concurrent fits now use BER_THREADS≤8 and OMP_WAIT_POLICY=PASSIVE.

## E-X1. Cross-script candidate views: xs (learned dictionary) and dn (e5 dense), train side (parallel track)
**Recall** (India, train query folds 0–3, 353,705 S1, 1,226,641 true pairs; base pair recall 0.97315, 32,933 misses). `xscript.py measure` plus a cap sweep. Both views are reverse (target→S1) and cover only targets with a non-Latin-script name (752,869 India train targets).

| added to base | new pairs | new/S1 | misses recovered | India pair recall |
|---|---|---|---|---|
| dn rank<1 | 71,802 | 0.20 | 5,923 | 0.97798 |
| dn rank<4 | 892,702 | 2.52 | 8,630 | 0.98019 |
| dn rank<50 | 14.58M | 41.2 | 12,665 | 0.98348 |
| xs rank<1 | – | – | – | 0.98368 |
| **xs rank<2 ∪ dn rank<1** | **344,568** | **0.97** | **14,791** | **0.98521** |
| xs rank<4 ∪ dn rank<1 | 881,617 | 2.49 | 15,355 | 0.98567 |
| xs rank<8 ∪ dn rank<8 | 3.95M | 11.2 | 15,850 | 0.98607 |

- xs (802 learned native→Latin token maps, train folds only) is far more efficient than dense e5: at rank<2 about 5% of its new pairs are true, against 0.2% for dn rank<4.
- dn adds only +0.0004 recall on top of xs rank<2. Chosen caps: xs 2, dn 1 (`--cap xs=2,dn=1`).
- Costs: xs build 4.5 min, 3.3 GB. Measures 5–9 s, ≤4.3 GB. Merge 79 s, 5.0 GB, and the refit reproduces the stored cosines (check passed).

**Matcher effect** (A2 retrained end to end in the isolated `../work_xd`: merged India candidates, India features recomputed at 101 s and 9.8 GB, US features symlinked, `BER_EXTRA=featdn` e5 cosine, `BER_TAG=_xd`). τ and variant chosen on fold 3, reported on fold 0:

| fold 0 | tune F0.5 | eval F0.5 | macro P | macro R | pair R | FP pairs | singleton acc | F0.5 (1 true) | oracle | eval cand pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| A2 (E-M1) | 0.98055 | 0.98045 | 0.99334 | 0.95898 | 0.95915 | 4,661 | 0.98009 | 0.92929 | 0.99430 | 9,089,404 |
| **A2 _xd** (τ 0.68 thr+excl) | **0.98233** | **0.98234** (+0.0019) | 0.99361 | 0.96321 | 0.96330 | 4,564 | 0.98229 | 0.93088 | 0.99631 | 9,176,025 (+1.0%) |

- The matcher keeps ~95% of the oracle gain (+0.0020 oracle, +0.0019 actual). Precision does not drop, and FP pairs fall by 97.
- dn_cos is not in the top-20 gain, so the gain comes from the candidates, not the feature. Its separate ablation on the stage-2 lab is E-X2.
- Runtime: A2 _xd evaluate took 2 h 23 min, against ~30 min solo, because of LightGBM OpenMP oversubscription with a concurrent 16-thread fit. Concurrent LightGBM jobs now run with `BER_THREADS≤8 OMP_WAIT_POLICY=PASSIVE`.
- **Decision: keep** (tune +0.0018 ≥ +0.001 gate). Still pending: a density check at drop 0.19 (solo), the port onto the stage-2-lab pipeline (E-M3), and the test side (`work/chain_x3.sh`).
