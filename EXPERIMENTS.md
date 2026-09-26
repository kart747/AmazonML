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
