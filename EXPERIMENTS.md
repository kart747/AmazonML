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

### E-M3b. Density check of stage2_lab variants and the test prediction (`stage2_lab.py density`, `predict`)
Same 19% S1 drop as density.py (seed 42). Pool-dependent context is recomputed, then stage 1 → stage-2 features → augment/prune → each model. τ is re-chosen on the dense tune fold.

| variant | normal tune | dense tune (τ) | dense eval, dense τ | dense eval, normal τ | loss vs normal eval | dense FP (normal τ) |
|---|---|---|---|---|---|---|
| A2 (5d, density.py) | 0.98055 | – (0.73) | 0.97730 | 0.97716 | −0.0033 | – |
| s2lab_base | 0.98073 | 0.97751 (0.74) | 0.97749 | 0.97728 | −0.0032 | 6,148 |
| s2lab_new | 0.98182 | 0.97834 (0.71) | 0.97834 | 0.97826 | −0.0032 | 5,770 |
| s2lab_featx | 0.98549 | 0.98285 (0.78) | 0.98307 | 0.98281 | −0.0025 | 3,898 |
| **s2lab_all** | 0.98579 | **0.98315** (0.76) | 0.98328 | **0.98323** | **−0.0023** | 3,558 |

**Selected: s2lab_all.** It leads featx on both the normal and the dense tune (+0.0003 each), loses the least under density, and is the most robust at the normal τ. τ stays at 0.69 (tune-selected): under density the dense τ changes the score by only 0.00005. Density peak RSS 3.4 GB (rev filtered to needed targets), 10 min.

**Test prediction** (`output_variants/s2lab_all/`, validator PASS, sha256 c67c4d40…b55ff):

| country | pred/S1 A2 → all | ≥1-match A2 → all | summed p/S1 A2 → all |
|---|---|---|---|
| France | 3.447 → 3.404 | 0.9492 → 0.9462 | 3.582 → 3.514 |
| India | 3.348 → 3.332 | 0.9400 → 0.9400 | 3.438 → 3.393 |
| US | 3.411 → 3.390 | 0.9425 → 0.9422 | 3.494 → 3.459 |
| total | 3.39 → 3.36 | 0.942 → 0.942 | |

The direction matches the density simulation (fewer orphan-lookalike FPs). France moves most; it is unseen, so this is the thing to watch on the leaderboard. Status: [MACRO F0.5 VERIFIED fold 0: 0.98550] [density-checked] [test file validated]. Handed to 4f as submission 2.

### E-M4. Stage 1 retrained with featx (BER_EXTRA=featx BER_TAG=_s1fx) → s2lab `all` (`work/chain_s1fx.sh`)
Stage-1 validation logloss 0.00769 → **0.00513 / 0.00507** (fold A / B, −34%). But stage 2 already used featx, so little carries over:

| | tune | eval | FP | dense tune | dense eval (normal τ) | test pred/S1 |
|---|---|---|---|---|---|---|
| s2lab_all (submission 2) | 0.98579 | 0.98550 | 2,142 | 0.98315 | 0.98323 | 3.36 |
| s2lab_all_s1fx | 0.98594 | 0.98579 | 2,053 | 0.98337 | 0.98351 | 3.37 |

Δ tune +0.00015 and Δ dense tune +0.0002, both below the +0.001 gate. **Not adopted.** The test file is kept as a validated backup (`output_variants/s2lab_all_s1fx/`, PASS). Runtime ~80 min: stage-1 fits 7–8 min each at 8 threads (6.2 GB); feat2 train+test is the bulk.

## E-X2. Cross-script candidates on the stage-2 lab pipeline, plus the dn_cos ablation (parallel track)
**dn_cos ablation** on the base candidates: 5e's s2lab `all` (E-M3) with dn_cos added (`--variant all --tag _dn`, no --drop, 93 features):

| fold 3 / fold 0 | tune F0.5 | eval F0.5 | eval P | eval R | FP |
|---|---|---|---|---|---|
| s2lab all (−dn_cos) | 0.98579 | 0.98550 | 0.99686 | 0.96487 | 2,142 |
| s2lab all + dn_cos | 0.98583 | 0.98570 | 0.99699 | 0.96495 | 2,070 |

Tune shows +0.00004, which is noise, so **dn_cos is dropped**. The e5 embedding covers only non-Latin India targets, and featx's n_map_* features already score those.

**Port.** The work_xd candidates (base ∪ xs<2 ∪ dn<1 for India, E-X1) run through the whole stage-2 lab pipeline: stage 1 = A2_xd; featx recomputed in work_xd (US parts symlinked, 11 min, 11.6 GB peak); `stage2_lab build --extra featx` (2 min, 6.6 GB); `evaluate --variant all --drop dn_cos` (2.5 min, 8 threads). Same folds, same 92 features:

| | tune F0.5 | eval F0.5 | eval P | eval R | pair R | FP | singleton acc | F0.5 (1 true) | pairs/S1 |
|---|---|---|---|---|---|---|---|---|---|
| s2lab all, base candidates | 0.98579 | 0.98550 | 0.99686 | 0.96487 | 0.96528 | 2,142 | 0.99078 | 0.94498 | 3.354 |
| **s2lab all, xd candidates** | **0.98783** | **0.98742** | 0.99665 | 0.97018 | 0.97038 | 2,328 | 0.99045 | 0.95019 | 3.373 |

**Result: +0.0020 tune / +0.0019 eval, fully additive with E-M3.** Recall rises 0.53 pt, and FP rises by 186 pairs. The density check (`src/experiments/s2lab_density_tagged.py`, which uses stage2_lab.density with the A2_xd stage 1) is running; the ship gate is a drop no larger than s2lab all's. Test side: India test gains 1.02M candidate pairs (1.26/S1, against 0.97/S1 on train), merge 7.5 GB, India test features 11.1 GB. The prediction goes to output_variants/xd_s2lab_all.

### E-M5. Stage-2 hyperparameters, and 5d's xd candidates under s2lab (cross-reference)
- `s2lab_all_hp1` (lr 0.04, 255 leaves, min_data 100, feature_fraction 0.7): tune 0.98583 (+0.00004 vs s2lab_all), eval 0.98562, FP 1,848, 1,133 iterations. Below the gate, **not adopted**. Stage-2 hyperparameters are saturated.
- 5d's port (E-X*): s2lab `all --drop dn_cos` on work_xd candidates (base ∪ xs<2 ∪ dn<1 for India, stage 1 A2_xd): tune **0.98783** / eval **0.98742** (+0.0020 / +0.0019 vs s2lab_all). Density (same protocol, tag-aware stage 1): dense tune 0.98515 (vs 0.98315), dense eval at normal τ 0.98498, loss −0.0024. **Passes both gates → submission 3 candidate** (5d owns the test file and the SUBMISSIONS row).
- Reserve folds 4–9 (chain_f49): block started 07:42. Measured on base candidates first.
- **Blend lab** (`blend_lab.py`, mean of saved scores, missing pair = 0, τ and weights chosen on tune): best blend all + all_s1fx + all_dn scores tune 0.98615 / eval 0.98595. That is +0.00021 tune over the best single model (all_s1fx 0.98594) and +0.00036 over s2lab_all. The five stage-2 models are too correlated; below the gate, **not adopted**.

## E-X3. Candidate-set size, remaining retrieval misses, and the train→test gap (parallel track)
**Candidate-set size.** The organisers' update says smaller candidate sets per S1 rank higher (SUBMISSION_REQUIREMENTS.md l.11, l.107). Eval fold 0, xd candidates, 765,153 true pairs:

| candidate set | per S1 | pair recall | true pairs lost |
|---|---|---|---|
| retrieval (all views) | 41.55 | 0.98787 | 0 |
| rank caps name<3 addr<3 combo<8 rev<8 | 39.57 | 0.98764 | 174 |
| rank caps name<5 addr<5 combo<10 **rev<4** | 24.28 | 0.98561 | 1,727 |
| top-10 by stage-1 p1 per S1 | 10.00 | 0.98775 | 92 |
| **stage-1 p1 ≥ 1e-3 (what stage 2 already scores)** | **4.77** | **0.98779** | **61** |
| stage-1 p1 ≥ 1e-2 | 4.02 | 0.98696 | 700 |

Rank caps lose recall quickly, because reverse-view ranks carry the rare true pairs. The stage-1 learned filter keeps recall at ~1/9 of the size. **Adopted for shipping:** `candidate_pairs.tsv` = the pairs stage 2 scores (test 5.53/S1: US 5.31, India 5.50, France 6.21) instead of 49.9/S1. The matches are identical. `redecode.py pruned` builds it, and `redecode.py check` is a hard matched ⊆ candidates check, whereas the official validator only warns. The pipeline is disclosed in Documentation §3.

**Remaining retrieval misses** (fold 0, after xs/dn): India 4,534 pairs (62% empty target address, 10% non-Latin), US 4,746 (90% empty address; 2,886 have name ≥ 90 and address < 50). Only ~650 pairs are "coined name at the same address" (name < 50, address ≥ 80), and that is also the planted-distractor pattern. **Retrieval is saturated for decidable pairs.** No further views were built.

**Train→test gap evidence (no LB scores of ours yet).**
- A public repo with a pipeline like ours (harshgitty58/Amazon_ML_Challenge, two-stage + siblings) reports val 0.9842 → public LB 0.969. Their earlier model went 0.9743 → 0.967, so +0.010 val gave +0.002 LB. Their country-blanking probes (France 0.836, India 0.580, US 0.578) imply public-LB weights of about US 43% / India 43% / France 15%. Swapping France predictions changed nothing. Public top on 09-26: 0.9906.
- Ours, label-free: we match 58.8% of test targets against ~74% true in train. Test pred/S1 is +0.046, against +0.012 in the density simulation.
- Stage-2 score mass per S1 by p bin (xd + s2lab_all):

| | <0.01 | 0.01–0.05 | 0.05–0.2 | 0.2–0.5 | 0.5–0.68 | 0.68–0.8 | 0.8–0.95 | 0.95–0.99 | >0.99 |
|---|---|---|---|---|---|---|---|---|---|
| eval India | 1.096 | 0.195 | 0.100 | 0.072 | 0.024 | 0.016 | 0.051 | 0.097 | 3.211 |
| test India | 1.565 | 0.281 | 0.151 | 0.095 | 0.030 | 0.022 | 0.062 | 0.108 | 3.181 |
| eval US | 0.926 | 0.167 | 0.127 | 0.093 | 0.023 | 0.014 | 0.038 | 0.071 | 3.249 |
| test US | 1.331 | 0.290 | 0.172 | 0.102 | 0.029 | 0.018 | 0.044 | 0.082 | 3.245 |
| test France | 1.951 | 0.389 | 0.257 | 0.149 | 0.052 | 0.041 | 0.110 | 0.149 | 3.112 |
| eval precision in bin | 0.002 | 0.025 | 0.116 | 0.327 | 0.569 | 0.748 | 0.898 | 0.978 | 0.9997 |

The confident mass (> 0.99) is unchanged, and the mid-range grows by 10–30% (France about 2×). With eval calibration this would add only ~0.003 FP/S1. So an LB drop of the size above would mean calibration itself breaks on test, most likely on orphan copies of absent S1 entities (hypothesis H1). Thresholds cannot fix that: τ 0.55 → 0.80 moves only 1.3% of test pairs. The fix is density-trained stage 2 (`density_train.py`, queued).
- LB direction probes, all on sub-3 scores with pruned candidates and subset-checked: `output_variants/xd_s2lab_all_tau080` (3.36/S1), `_tau055` (3.40/S1), `_fr090` (France τ 0.90; 5,837,458 matches). Decisions rest on tune + density, not on public-LB deltas.

## E-X4. Unseen-country adaptation by pseudo-labelling: US→India proxy for France (parallel track)
France is 15% of test (~15% of the public LB) and has no labels. Proxy: India is treated as unlabeled (`src/experiments/pseudo_label.py`, work_xd s2lab parts, `all` features without dn_cos). τ/variant are always chosen on the **US** tune fold, and results are reported on the **India** eval fold (fold 0). Stage 1 was trained on both countries, so this measures stage-2 transfer only.

| stage 2 trained on | India eval F0.5 | P | R | FP | singleton acc | pairs/S1 |
|---|---|---|---|---|---|---|
| US only (unseen-country baseline) | 0.97754 | 0.99276 | 0.95266 | 2,101 | 0.98393 | 3.331 |
| **US + India pseudo-labels** (p ≥ 0.95 & target's best S1 → 1, p ≤ 0.05 → 0) | **0.98033** (+0.0028) | 0.99274 | 0.95950 | 2,173 | 0.98313 | 3.355 |
| US + true India labels (oracle) | 0.98667 | 0.99630 | 0.96875 | 1,049 | 0.98996 | 3.372 |

- Pseudo-labels: 1.16M rows, 852,554 positives with 99.76% precision, negatives 1.46% wrong, 117k middle rows dropped.
- Pseudo-labelling recovers **31%** of the unseen-country gap (0.0028 of 0.0091), all of it through recall. Precision and FP stay flat.
- Implication for France: the shipped model is in the "US+India → France" position, so a France stage-2 gap of the same order (~0.009 on France, ~0.0014 on the LB at 15% weight) is plausible. Pseudo-labelling France test pairs with the shipped model and retraining stage 2 on train + France pseudo-rows could recover about a third of it. It cannot be validated on France itself; the proxy above is the only evidence.
- Runtime 13.6 min, 7.2 GB peak at 4 threads.

## E-X5. LB ground truth for upload #1, and the local→LB gap (parallel track, post-reboot)
**Upload #1** (08:42 IST): xd + s2lab_all, matching sha f9ff0a26696c (= SUBMISSIONS rows 3/3p = `submission/sub03p_xd_s2lab_all`). **Public LB 0.97741** against local fold 0 0.98742: **−0.0100**. The density simulation predicted −0.0024, which explains only a quarter.
What the gap is NOT (label-free and local evidence, all on the shipped model):
- **Not the country mix.** Local per country: India 0.98665 (P 0.99613, R 0.96917), US 0.98794 (P 0.99700, R 0.97085).
- **Not France retrieval.** The best-candidate combo cosine per test S1 is highest in France (median 1.00, q10 0.894; India 0.972/0.795; US 0.991/0.827). The share of S1 whose best candidate is < 0.5 is 0.03–0.09% everywhere.
- **Not visible prediction volume.** Test pred/S1: France 3.40, India 3.37, US 3.39. Empty: 5.4/5.8/5.8%. ≥6 matches: 10.8/10.3/10.5%. Fold 0: 3.37 pred vs 3.47 true; 5.8% empty vs 5.6% true.
- **Not more true copies per S1.** p > 0.99 mass per S1 is unchanged (E-X3).

Arithmetic: if US/India lose the density-sim −0.0024 on test and France is ~15% of the public set, then 0.97741 ⇒ **France ≈ 0.93**. France is unseen in every learned stage: stage 1, stage 2, featx dictionary and normalization. The stage-2-only US→India proxy (E-X4) already shows −0.009, so France is the prime suspect for most of the gap, with harder test orphans second.

## E-X6. Density-augmented stage 2 (`density_train.py`, work_xd; 19% S1 dropped with the same mask as density.py)
Dense-world s2lab parts for folds 0–3: build 10.9 min, **peak 15.9 GB** (run solo). Same 92 features as s2lab_all.

| stage 2 | normal tune | normal eval F0.5 | normal FP | dense tune | dense eval @ normal τ | dense FP | density loss |
|---|---|---|---|---|---|---|---|
| s2lab_all (shipped, upload #1) | 0.98783 | 0.98742 | 2,328 | 0.98515 | 0.98498 | 3,800 | −0.0024 |
| dense-only training | 0.98660 | 0.98642 | 3,063 | 0.98630 | 0.98631 | 2,521 | −0.0001 |
| **dense + normal training (`--mix`)** | 0.98768 | **0.98743** | 2,162 | **0.98634** | **0.98625** | 2,542 | **−0.0012** |

- **dense_mix is the strict improvement**: normal eval equal (+0.00001), test-like density **+0.0013**, density loss halved, FP down at both densities.
- Test (`output_variants/xd_dense_mix`, τ 0.68 from the dense tune, pruned candidates, subset check 0): 5,849,442 matches (−11,364 vs upload #1). pred/S1 changes: France 3.403 → 3.379 (−0.025), India −0.004, US −0.003. The model drops the most France merges.
- Evaluate: 6.7 min each, 3.4–3.7 GB.

## E-X7. France pseudo-labelling on test (`pseudo_label.py apply`)
The shipped s2lab_all labels France test pairs: p ≥ 0.95 and the target's best S1 → 1 (845,543); p ≤ 0.05 → 0 (607,085); 158,717 dropped. Stage 2 is retrained on train folds 1,2 plus the France pseudo rows, with τ on tune fold 3.
- Normal: tune 0.98760 (−0.0002), **eval 0.98744** (+0.00002), FP 2,190. US/India are unaffected, as intended.
- Test: France pred/S1 3.403 → 3.427 (+0.024), mass/S1 3.503 → 3.468. The France distribution gets sharper (more recall, fewer uncertain pairs), matching the proxy (E-X4: +0.0028 on the unseen country, all through recall).
- File: `output_variants/xd_s2lab_all_plfr` (5,862,360 matches, subset check 0). France-only effect on the LB ≈ w_FR · ΔF_FR.
- Running: the combination (dense_mix labels France, then dense+normal+France pseudo rows) → `output_variants/xd_dense_mix_plfr`.

## E-G0. LB-gap audit, label-free (main session; upload #1 = 0.97741 vs local 0.98742)
Train vs test structure (`work/prep`):

| | train | test |
|---|---|---|
| S1 (US / India / France) | 1,323,633 / 883,188 / **0** | 663,106 / 809,986 / **259,452** |
| targets per S1 (US / India / France) | 4.67 / 4.68 / – | 5.76 / 5.82 / 5.53 |
| S1 pool density vs train | 1 | **US 0.50×**, India 0.92× |
| S1 sharing an exact address with another S1 | US 5.6%, India 5.1% | US 4.0%, India 5.0%, **France 11.9%** |

- **No leakage**: S1/target entity ids don't overlap across splits, there are 0 exact name+address test-S1 duplicates of train S1, and there's no id or row-order correlation within true pairs.
- **France is unseen** (15% of test S1). It has 18 cities, S1 addresses carry a multi-token region while targets often carry the department, and French suffix tokens (sarl/sas/eurl/sci, groupe, développement, & Fils, & Cie, & Associés) are new.
- **US test has more planted look-alikes.** Targets carrying affiliate tokens (group/partners/holdings/ltd/co/corp) go from 0.90 to 1.45 per S1, and branch tokens (eastgate, southside, …) from 0.32 to 0.57 per S1. In train, extra group/holdings/branch tokens are 0% true and carry shifted house numbers (hn_eq ~2%). In US test the mid-band (0.2–0.99) affiliate mass is 3× eval (0.016 → 0.051 per S1), while non-affiliate mid mass is unchanged.
- The **score distribution** above p 0.999 is unchanged per S1 (US 2.99 vs 3.01, India 2.87 vs 2.90), so there are not more true copies. The mid-band grows 10–30% for US/India. For France it is about 2× (0.78 vs ~0.52 per S1 in 0.2–0.999).
- **France extra-name-token kept rates** (the target adds the token vs the S1): fils 70%, et 67%, cie 61%, associes 47%, groupe 25%, développement 24%; holding/international/distribution/participations/snc about 0.2%. In US/India train, the analogous "and / associates / sons / group / holdings" extras are 0–1% true.
  - Their house numbers differ: fils/cie/associes/et have the same house number 84–90% of the time, the rejected ones ~15%. Without labels this remains ambiguous: they could be true suffix variants (France recall) or same-address siblings (precision).
- Per-source match counts and the S2-only / S3-only / n=1 / n≥6 shares of test predictions match train truth in every country, France included. 185 S1 exceed the S2 cap of 5 and 24 exceed the S3 cap of 6 (negligible).

## E-G1. France-only re-normalization (`fr_renorm.py`, isolated `work_fr`, same models)
Each region/department comma-segment becomes one region code (hdf/naq/pdl, like US/India state codes), and "et" becomes "and" in names. Only France feat/featx/feat2/s2lab test parts are rebuilt; the candidates are the same rows.
- Confident France pairs had address token Jaccard 0.74 (US 0.83, India 0.79).
- Effect: France kept 883,013 → 882,573, kept in both 876,026 (~0.8% of France kept pairs change), mid mass slightly lower, >0.999 mass 2.83 → 2.86. US/India are byte-identical.
- **Decision: too small to spend an upload on alone.** It is kept only as a possible add-on to a France candidate. Runtime ~20 min, peak 10 GB (features_v2).

## E-G2. Covariate-shift reweighting (`shift_reweight.py`)
Eval-fold S1s are reweighted so their label-free covariate histogram matches test: candidate count, affiliate candidates, mid-band count, >0.99 count, affiliate mid-band.
- Estimated test F0.5: US 0.9861, India 0.9850, France (from pooled US+India cells) 0.979; overall 0.9845 vs eval 0.98742.
- So candidate-structure shift explains about −0.003, like the 19% density sim (E-D1). The remaining ≈ −0.007 must come from P(error | covariates) changing on test: new error patterns that are not in train.

Rejected: an **affiliate guard** (drop kept pairs whose target adds an affiliate/branch token AND whose house number differs). On eval, the pairs it hits are 95% true (fold 0: US 381 TP vs 21 FP, India 516 vs 12) because house numbers are noisy in true pairs.

## E-G3. Test-world simulation (`world_sim.py`, built, run pending)
Per region, test US equals the train world with half the S1 entities deleted **together with their true copies**, while the orphans stay: 1.22 / 0.5 = 2.44 orphans per S1, matching test's ~2.3. The 19% sim instead keeps a 0.81×-dense S1 pool.
- Build: US drops 50% of S1 and deletes their 2.29M true targets. India drops 19% and orphanizes them (India's orphan density cannot be reached by deletion).
- It recomputes the reverse context, name frequencies, forward ranks r_name/r_addr/r_combo, and q_rank/q_gap/q_ncand after the deletions, followed by stage 1 → stage 2 features.

**E-X7 addendum: dense_mix + France pseudo-labels** (`pseudo_label.py apply --ref s2lab_dense_mix --mix-dense`): dense_mix labels France test pairs (832,912 pos / 622,268 neg / 156,165 dropped). Stage 2 is trained on normal + dense folds 1,2 + France pseudo rows (5.40M rows), with early stopping on the dense tune fold and τ 0.71 on the normal tune fold.
Normal: tune 0.98759, **eval 0.98743**, FP 2,042. Dense eval @ τ: **0.98632** (dense_mix 0.98625). Test pred/S1: France 3.396 (dense_mix 3.379, upload #1 3.403), India 3.363, US 3.382. File: `output_variants/xd_dense_mix_plfr` (5,847,281 matches, subset check 0). Held back until upload #2 (dense_mix) and #3 (France-blank probe) show where the LB gap sits.

**Upload queue (agreed with amazonml-27):** U2 = `submission/sub04_xd_dense_mix` (validator PASS, locked, sha 4b4b064ccc1b). U3 = `submission/sub05probe_xd_dense_mix_frblank` (France emptied; US/India byte-identical to U2), so LB(U2) − LB(U3) = w_FR · (F_FR − s_FR).
- **Result (10:22, build 7.4 min, peak 10.7 GB):** eval fold 0 in the test-like world at the normal τ 0.68:

| model | world eval | US | India | world-tuned τ → eval |
|---|---|---|---|---|
| s2lab_all (upload #1) | 0.98592 | **0.98847** (normal 0.98794) | 0.98357 (normal 0.98665) | 0.66 → 0.98593 |
| s2lab_dense_mix (U2) | **0.98688** | 0.98882 | 0.98508 | 0.64 → 0.98694 |

- **The US half-density hypothesis is rejected.** Deleting half the US S1 together with their true copies makes US slightly *easier*: their siblings' look-alikes disappear with them. India loses the usual −0.003 from orphanization.
- Neither the S1-pool density nor orphan density reproduces the −0.010 LB gap. With US/India near −0.002, the unexplained loss points to France (unseen) or to error types absent from train.
- dense_mix beats the shipped model in this world too (+0.0010 overall, +0.0015 India), which supports U2.

## E-X8. Address-ambiguity features (`addr_freq.py`), rejected
Idea (from amazonml-27's France audit: 11.9% of France test S1 share an exact normalized address with another S1, against 4–5.6% elsewhere): give stage 2 the address analogue of name frequency. q_ad_n / q_ad_rate = S1 sharing this S1's address; t_ad_s1n = S1 at the target's address; t_ad_others = competing S1 at the target's address. These are extra s2lab columns, computed in the normal and in the dense (19% drop) world.

| work_xd, 96 features | normal tune | normal eval | FP | dense tune | dense eval @ normal τ |
|---|---|---|---|---|---|
| s2lab_all (reference) | 0.98783 | 0.98742 | 2,328 | 0.98515 | 0.98498 |
| s2lab_all + adf | 0.98774 | 0.98748 | 2,335 | 0.98512 | 0.98495 |
| dense_mix (U2) | 0.98768 | 0.98743 | 2,162 | 0.98634 | 0.98625 |
| dense_mix + adf | 0.98772 | 0.98751 | 2,109 | 0.98643 | 0.98633 |

All deltas are within ±0.0001 (noise), and none of the four features is in the top-25 gain. **Rejected.** Train has too few shared-address S1 for the model to learn the interaction, so a France transfer can't be expected from this. Runtime ~12 min, ≤4.9 GB.

## E-G4. Unseen-country stage-2 transfer (`loco_s2.py`, France proxy)
- **Hypothesis:** France has no training rows, so the France loss can be approximated by scoring a country the stage-2 model never saw.
- **Implementation:** train the s2lab_all feature set on source-country folds 1,2 (early stop on source fold 3) and score the target country's eval fold 0 over a τ grid (thr+excl), next to the in-country s2lab_all. Stage 1 still saw both countries, so this is only a partial transfer: the real France loss is at least this large.
- **Result (work_xd, a few minutes each, <6 GB):**

| transfer | F0.5 @ τ 0.68 | P | R | FP pairs | in-country F0.5 | best τ → F0.5 |
|---|---|---|---|---|---|---|
| US → India | 0.97787 | 0.99168 | 0.95565 | 2,430 | 0.98665 | 0.68 → 0.97787 |
| India → US | 0.97949 | 0.98749 | 0.96869 | 5,513 | 0.98794 | 0.85 → 0.98172 |

- **Reading:** an unseen country costs about −0.009 with stage 2 alone. The failure mode is country-specific: recall for India (tokens/scripts US never shows), precision for US (2.4× the in-country FP). No single τ fixes both: raising τ helps the precision-limited direction (+0.0022 at 0.85) and cannot help the recall-limited one.
- **Implication for France:** with w_FR ≈ 0.15, a France F0.5 of ~0.93–0.96 is plausible and could account for most of the −0.010 LB gap. Whether France is precision- or recall-limited decides the fix (France τ up vs pseudo-labels/renorm). U3 (France-blank probe) measures its F0.5. The full LOCO (stage 1 also trained on US only, `work/chain_loco.sh`) is running as E-G5.

## E-X9. Adversarial validation of stage-2 inputs and the name-frequency scale (`adversarial.py`)
Train (folds 0–3) vs test stage-2 rows of the same country, p1 ≥ 0.05, 200k per side, LightGBM, 30% held-out AUC:
- **All features: AUC 1.0000 (US and India).** The separators are q_nm_freq, t_nm_freq and the featx IDF features (q/t_unm_maxidf, q/t_match_idf). These are computed **per split**: name frequency is count·1e6 / n_S1(country, split). A unique name is 0.76 in US train (1.32M S1), 1.51 in US test (663k) and **3.85 in France test** (259k), which is what a name shared by ~5 S1 looks like in US train.
- **Without freq/idf: AUC 0.728 (US), 0.665 (India).** The shift is density context: q_ncand 41.6 → 49.3 (US), p1_qsum, t_gap12, rev_margin, r_combo, q_rank. The density simulations already cover this (E-D1, E-X6, amazonml-27's E-G3: ≤ −0.003).

**Counterfactual** (dense_mix stage 2, test q_nm_freq/t_nm_freq multiplied back to the train count scale; stage 1 untouched):

| test country | factor | kept (p ≥ 0.68)/S1 | pairs flipping | >0.99 mass/S1 |
|---|---|---|---|---|
| France (US scale) | ×0.196 | 3.384 → 3.407 | 7,852 | 3.058 → 3.079 |
| France (India scale) | ×0.294 | 3.384 → 3.409 | 8,671 | 3.058 → 3.077 |
| US | ×0.501 | 3.387 → 3.397 | 8,533 | 3.208 → 3.239 |
| India | ×0.917 | 3.370 → 3.370 | 987 | 3.152 → 3.153 |

The pool-size-dependent rate makes names look more ambiguous on test, and the model holds back confident US/France matches. The proper fix is raw counts in stages 1+2 plus a retrain. The cheap test is a label-backed check in amazonml-27's world_sim (US pool 0.5×, same inflation): rescale there and compare F. Status: [HYPOTHESIS] pending that check.

## E-G5. Full leave-one-country-out: both stages trained on US only, scored on India (France's actual position)
- **Hypothesis:** France is unseen by stage 1 AND stage 2. E-G4 (stage 2 only) understates the loss because p1, 72% of stage-2 gain, was still calibrated on the target country.
- **Implementation** (`work/chain_loco.sh`, isolated `work_loco`, symlinked work_xd inputs):
  1. Stage 1 (`A2_loco_stage1_fold{A,B}`, +dn_cos) is trained with only the US feat/featdn parts visible. The India parts are exposed afterwards.
  2. `build_stage2_parts` writes US OOF / India-averaged p1 → `stage2_lab build`.
  3. A US-only stage 2 (s2lab_all features) is scored on India eval fold 0.
  - Runtime: 15 min total (stage 1 4.6 min, feat2 7.6 min, s2lab 2 min), peak 6.6 GB.
- **Result** (India eval fold 0, thr+excl τ 0.68):

| model in the LOCO world | India F0.5 | P | R |
|---|---|---|---|
| US-only stage 2 (unseen) | **0.92193** | 0.96048 | 0.88926 |
| shipped s2lab_all fed US-only p1 | 0.95839 | 0.97882 | 0.93720 |
| + one pseudo-label round (`pseudo_label.py`, p≥0.95 & best → 1, p≤0.05 → 0; pos precision 0.978, neg error 0.045) | 0.92922 | | |
| stage 2 trained on true India labels (oracle) | 0.98325 | | |

  Normal world, in-country: 0.98665.
- **Stratified (unseen model):**

| India S1 stratum | share | unseen F0.5 | shipped model, US-only p1 |
|---|---|---|---|
| true targets all Latin script | 49.5% | 0.93875 | 0.96462 |
| mixed script | 39.3% | 0.93118 | 0.96521 |
| all native script | 5.6% | 0.75382 | 0.89299 |
| singletons | 5.6% | 0.87766 | 0.92145 |

  Candidate recall after the US-only p1 ≥ 1e-3 cascade: 97.6% of true pairs remain.
- **Reading:**
  - An unseen country costs ~0.06 F0.5 when both stages are unseen, about 7× E-G4's stage-2-only estimate.
  - It is not a script effect: Latin-only S1 lose 0.048 too.
  - A US-only stage 1 is almost harmless once stage 2 has target labels (oracle 0.98325). The damage is in stage 2's learned mapping from a shifted p1/context to a decision.
  - One pseudo-label round recovers only 0.008 of 0.062.
- **Implication:** France is plausibly ~0.93. Then with w_FR ≈ 0.15, LB = 0.85·0.985 + 0.15·0.93 ≈ 0.977, which is exactly the observed 0.97741. work_loco is now the label-backed test bed for every France fix: amazonml-46 runs the freq rescale and iterative self-training there, E-G6 ablates feature groups.
