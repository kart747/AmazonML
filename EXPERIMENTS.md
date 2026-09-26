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
