# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [[TBD: team name]]
**Team Members:** [[TBD: members]]
**Submission Date:** [[TBD]]

> Draft status: every number below was measured on the provided training data by our own
> code, unless it is marked `[[TBD]]`. Placeholders are filled from `work/reports/*.json` and
> `EXPERIMENTS.md` when the final runs finish. No external data, APIs or pretrained models are used.

---

## 1. Executive Summary
We treat the task as large-scale retrieval followed by precision-oriented matching.
Per country, candidates are generated with exact sparse TF-IDF top-K retrieval over several
views. The key view is **reverse retrieval**: each Source-2/3 record looks for its best
Source-1 records. It exploits the fact that every target belongs to at most one Source-1
entity. A LightGBM matcher then scores the candidates. It combines pairwise similarities with
label-free candidate-graph context (does this target prefer a different Source-1 record?).
The decision layer uses a threshold tuned on held-out macro F0.5, plus target exclusivity.
[[TBD: final validation macro F0.5 and leaderboard score]]

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full training data (2,206,821 S1; 5,034,616 S2; 5,285,603 S3; 7,638,365 true pairs):

| Finding | Value | Consequence |
|---|---|---|
| S1 with no match (singletons) | 5.58% | Predicting nothing scores only 0.056. The problem needs recall as well as precision |
| Matches per S1 | mean 3.46, max 11; 72% have ≥3 | Variable-size sets, never top-1 |
| Targets linked to >1 S1 | 0 of 7.64M | Many-to-one: each target has ≤1 owner (an S1 can have many) |
| True pairs crossing countries | 0 | Country is a lossless partition (label-agnostic, so it works for unseen France) |
| Unmatched ("distractor") targets | S2 26.6%, S3 25.4% | Hard negatives are abundant |
| S1 names shared with another S1 | 39.3% | A name alone rarely identifies an entity; the address decides |
| True pairs: exact normalized name / share ≥1 name token | 25.7% / 85.5% | Exact-key blocking is hopeless; fuzzy retrieval is needed |
| True pairs: share ≥1 address token / ≥1 number | 95.6% / 79.8% | Address is the most reliable field |
| True pairs with a non-Latin target name | 7.3% (India targets ≈23%) | Cross-script bridge needed |
| Empty target address | S2 3.4%, S3 3.3% | Name-only evidence; often undecidable for shared names |
| Test composition | India 810k, US 663k, **France 259k (15%, unseen)** | Open-set countries; no France labels |
| Targets per S1 | train 4.68 → test 5.75 | Test is denser in distractors; train-tuned thresholds are optimistic |

Noise observed: typos and leetspeak (`C0astal`), token reorder, legal forms moved or bracketed (`[LLC] Coastal Tungsten`),
truncation, domain names (`coastaltungsten.com`), `M/s` prefixes, Indic-script names and state
names, perturbed or shortened house numbers (`2454`→`245`), city swaps, `H.NO`/`#`/`NO.` prefixes, component
reordering. In France (test only): accents injected or removed, `R.`/`AV.`/`BD`/`N°`, région↔département swaps.

### 2.2 Solution Strategy
**Approach Type:** Blocking (multi-view sparse retrieval) + gradient-boosted classifier + F0.5-aware decision
**Core Innovation:**
1. Reverse (target→S1) retrieval as the primary blocking view, justified by the many-to-one structure.
2. Cache-blocked exact sparse top-K, which makes exact retrieval over 10M records feasible on a 16-thread laptop.
3. Label-free many-to-one competition features for the matcher.

Pipeline: `prepare → block → features → evaluate (fit on train folds, tune on fold 3, report fold 0) → predict`.

---

## 3. Candidate Generation (Blocking)
- **Normalization (multi-view, raw text kept):** Unicode folding and transliteration (anyascii), lowercase, `&`→and,
  junk-prefix/`M/s`/bracket removal, leetspeak zeros, domain→core name, spaced-initials gluing (`L.L.C.`→`llc`),
  null-token removal, house-number prefix removal, leading zeros, and a small hand-written street-type/state lexicon.
- **Partition:** by the country label (from the data; unseen labels are handled identically).
- **Views (per country):**

| View | Representation | Direction | K |
|---|---|---|---|
| name | char_wb 3-gram TF-IDF of the normalized name | S1 → target | 5 |
| addr | word TF-IDF of the normalized address | S1 → target | 5 |
| combo | [√.5·name ‖ √.5·addr], cosine = mean of the two | S1 → target | 10 |
| **rev** | combo | **target → S1 (over all S1 of the country)** | 8 |

- **Exact top-K at scale:** the sparse top-K kernel keeps an item-sized accumulator per thread. Splitting the items into
  65,536-row column blocks and merging the per-block top-K is **17× faster with identical results**
  (India, 4.1M targets: 17.8 → 1.0 ms/query). Pruning frequent n-grams was rejected: it cost 5–15 points of recall.
- **Candidate pairs generated:** train query folds [[TBD: total from reports/blocking_train.json]]; test [[TBD]].
  India train (353,705 query S1): 14.78M pairs, 41.8 per S1 (p50 29, p95 105, p99 217).
- **How we ensured true matches were not lost:** recall is measured against the full target pool.
  India train pair recall **0.9732** (views: rev 0.972, combo 0.892, addr 0.733, name 0.412).
  94.5% of true pairs are the target's reverse top-1. Upper bound on India macro F0.5 with perfect matching: **0.9905**.
  The remaining misses are dominated by Indic-script target names (53% of misses vs 23% of found pairs) and by empty
  target addresses combined with names shared by many S1s (undecidable by design).
  [[TBD: US numbers; the cross-script supplementary view result, see §5]]

---

## 4. Matching Model

**Features used** (all label-free functions of the pair or the unlabeled candidate graph; identical for train and test):
- Name: exact TF-IDF cosines (char 3-gram, space-less), RapidFuzz ratio / token-sort / token-set / partial / Jaro-Winkler,
  token Jaccard and overlap, token counts, non-ASCII flag.
- Address: word TF-IDF cosine, RapidFuzz ratio / token-set / partial, token Jaccard, house-number agreement
  (Jaccard, first-number equality, log difference, counts), empty-address flag, lengths.
- Context (many-to-one competition): per-view retrieval ranks, the target's best reverse score and runner-up gap, whether this
  S1 is the target's best S1, the margin over the best other S1, rank and gap inside the S1's own candidate list, list
  size, and name frequency among S1 (ambiguity, as a rate per 1M S1 of the country).
- [[TBD: stage-2 collective features and features v2 if kept]]

**Model type:** LightGBM (MIT licence; far below the 8B-parameter limit). Two variants were compared on the same candidates and split:
- A1: independent pair classifier (pairwise features only).
- A2: two-stage relational classifier. Stage 1 uses pairwise + context features and is cross-fitted across two folds
  (out-of-fold probabilities). Stage 2 adds each S1's probability profile and sibling support (similarity of a
  target to the S1's other strongest targets).

**Threshold selection method:** S1-grouped folds (`int(id) % 10`): train folds 1–2, tuning fold 3, reporting fold 0; folds 4–9 untouched.
τ (grid 0.30–0.95) and the exclusivity variant (keep a target only for its highest-scoring S1) are chosen
on fold 3 by exact macro F0.5, and reported once on fold 0. [[TBD: chosen τ; density-adjusted τ for test]]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, fold 0, every S1 counted incl. blocking misses):** [[TBD: A1 / A2]]
- **Precision / recall / singleton accuracy:** [[TBD]]
- **Common false positives (wrong merges):** [[TBD from reports/errors_A2.md]]
- **Common false negatives (missed matches):** blocking misses as in §3. [[TBD scoring FNs]]

| Experiment | Macro F0.5 (fold 0) | Kept? |
|---|---|---|
| [[TBD]] | | |

---

## 6. Conclusion
[[TBD]]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/run.py` is the CLI (`prepare`, `block`, `features`, `evaluate`, `predict`).
The `src/ber/` library contains io, normalize, blocking, features, approaches, metric, decoders and utils.
`src/experiments/` holds the ablation and analysis tools; `tests/` has a synthetic-data smoke test.
Reproduce with:
`python src/run.py prepare && python src/run.py block && python src/run.py features && python src/run.py evaluate --approach 2 && python src/run.py predict --approach 2`.
Every stage is resumable and writes timing and peak RAM to `work/telemetry.jsonl`.

### B. Additional Results
[[TBD]]
