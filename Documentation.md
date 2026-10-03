# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Kali
**Team Members:** Karthik (GitHub: kart747)
**Submission Date:** 2026-10-02 (final upload, leaderboard file `sub09`, 27 Sep 2026)

> Every number below was measured on the provided training data by our own code. The only pretrained model is
> multilingual-e5-small (MIT licence, 118M parameters, far below the 8B limit), used as a blocking view for non-Latin
> India targets. No external data or APIs are used. Full experiment log: `EXPERIMENTS.md`; upload log: `SUBMISSIONS.md`.

---

## 1. Executive Summary
We treat the task as large-scale retrieval followed by precision-oriented matching.
Per country, candidates are generated with exact sparse TF-IDF top-K retrieval over several
views. The key view is **reverse retrieval**: each Source-2/3 record looks for its best
Source-1 records. It exploits the fact that every target belongs to at most one Source-1
entity. A LightGBM matcher then scores the candidates. It combines pairwise similarities with
label-free candidate-graph context (does this target prefer a different Source-1 record?).
The decision layer uses a threshold tuned on held-out macro F0.5, plus target exclusivity.
The submitted file (`sub09`) adds cross-script (xs / e5) retrieval for India, a cascade-pruned stage-2 matcher with
extra features, a train-learned affiliate-token veto, and one round of pseudo-labelling for the unseen country (France).
Local macro F0.5 on held-out fold 0: **0.98748** (US/India; France has no labels). Public leaderboard: **0.978**
(first upload 0.97741). The final candidate set is **5.53 pairs per Source-1 entity**, down from 49.9 at retrieval.

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
- **Cross-script views (India):** 53% of India blocking misses had an Indic-script target name. anyascii spells
  "लक्ष्मी" as "lksmi", which shares almost no 3-grams with "laxmi". Two reverse views run only for non-Latin-script targets:
  **xs** maps native tokens through a native→Latin token dictionary learned from true pairs of the *training folds only*
  (802 maps), then runs combo retrieval, target → S1, keeping rank < 2. **dn** is multilingual-e5-small (MIT, 118M)
  embeddings, target → S1, keeping rank < 1. Together they add 0.97 pairs per India S1 and raise India pair recall from
  0.97315 to 0.98521 (train query folds).
- **Three stages up to the final candidate set.** `candidate_pairs.tsv` holds the last set, the one the final model scores:
  1. **Retrieval** (the views above): 41.6 pairs per S1 on eval fold 0, 49.9 per S1 on test (86.4M pairs).
  2. **Learned filter** (stage 1, a LightGBM over pairwise and retrieval-context features, cross-fitted): scores every
     retrieved pair and keeps p1 ≥ 1e-3.
  3. **Matcher** (stage 2): runs inference only on the kept pairs, **4.77 per S1 on eval fold 0 and 5.53 per S1 on test**
     (US 5.31, India 5.50, France 6.21; 9.59M pairs). The filter loses 61 of 765,153 eval-fold true pairs: pair recall is
     0.98779 against 0.98787 for the retrieved set.
  *Disclosure:* the 5.5 per S1 is not pure retrieval output. Stage 1 scores the full retrieved list, and several stage-2
  context features aggregate those stage-1 scores over the S1's full list (rank, gap, sum, count above 0.5, best other
  target). Every matched pair is in `candidate_pairs.tsv`; this is checked with `redecode.py check`.
- **Candidate pairs generated (retrieval stage):** eval fold 0: 9.18M (41.6/S1); test 86.4M (49.9/S1).
  India train (353,705 query S1, lexical views only): 14.78M pairs, 41.8 per S1 (p50 29, p95 105, p99 217).
- **How we ensured true matches were not lost:** recall is measured against the full target pool.
  India train pair recall **0.9732** (views: rev 0.972, combo 0.892, addr 0.733, name 0.412).
  94.5% of true pairs are the target's reverse top-1. Upper bound on India macro F0.5 with perfect matching: **0.9905**.
  The remaining misses are dominated by Indic-script target names (53% of misses vs 23% of found pairs) and by empty
  target addresses combined with names shared by many S1s (undecidable by design).
  All countries, eval fold 0, lexical views plus xs/dn: pair recall **0.98787**; blocking-oracle macro F0.5 **0.99631**
  (0.99430 without xs/dn). US misses are 90% empty-address targets of common names, which are undecidable, so no US-specific
  view was added.

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
- Features v2 (`featx`, kept): learned native→Latin token dictionary (train folds only), consonant skeletons, IDF of the
  rarest one-sided name token (the orphan look-alike signature), learned core-token overlap, number conflicts. It was the
  largest single gain (+0.0048 tune, FP −46%).
- Stage-2 lab (`all`, 92 features, kept): cluster support against the S1's confident set, twin counts, p1 rank/confidence
  within source, space-stripped name similarity, exact/contained address, out-of-vocabulary name share.

**Model type:** LightGBM (MIT licence; far below the 8B-parameter limit). Two variants were compared on the same candidates and split:
- A1: independent pair classifier (pairwise features only).
- A2: two-stage relational classifier. Stage 1 uses pairwise + context features and is cross-fitted across two folds
  (out-of-fold probabilities). Stage 2 adds each S1's probability profile and sibling support (similarity of a
  target to the S1's other strongest targets).

**Threshold selection method:** S1-grouped folds (`int(id) % 10`): train folds 1–2, tuning fold 3, reporting fold 0; folds 4–9 untouched.
τ (grid 0.30–0.95) and the exclusivity variant (keep a target only for its highest-scoring S1) are chosen
on fold 3 by exact macro F0.5, and reported once on fold 0. Final: τ = 0.68 with target exclusivity (tune fold 3: 0.98783). The τ curve stays flat under the test's higher
target density, so the same τ is used on test. A post-hoc veto drops 4,667 pairs whose only name difference is an
affiliate token (`group`, `holdings`, `ventures`, ...) learned on the tune fold (true rate ≤ 2%; 0 true pairs removed on eval).
France decisions come from a stage-2 model retrained with France pseudo-labels (p ≥ 0.95 positive, p ≤ 0.05 negative, τ 0.70).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, fold 0, every S1 counted incl. blocking misses):** A1 0.96428 → A2 0.98045 → final xd + stage-2 lab 0.98748 (tune 0.98787). Leaderboard 0.978.
- **Precision / recall / singleton accuracy:** Precision 0.99665 / recall 0.97018 (macro); pair precision 0.99687, pair recall 0.97038; singleton accuracy 0.99045; 3.37 predicted pairs per S1.
- **Common false positives (wrong merges):** 78% of false-positive targets belong to no S1: orphan look-alikes with the same address and a one-character or one-token name edit. On test, planted affiliate-token look-alikes are the main source; the veto removes them.
- **Common false negatives (missed matches):** blocking misses as in §3. Remaining misses are partial misses inside multi-match S1 (shared names, empty target addresses, coined or glued names, perturbed house numbers); 10% of FN pairs score below 0.05.

| Experiment | Macro F0.5 (fold 0) | Kept? |
|---|---|---|
| A1 pairwise | 0.96428 | baseline |
| A2 two-stage relational | 0.98045 | yes |
| + xs/dn India blocking + stage-2 lab `all` (final) | 0.98748 | **yes** |
| + affiliate veto | 0.98748 (0 true removed) | yes |
| + France pseudo-labelling (France only) | US/India unchanged; unseen-country proxy 0.92142 → 0.92979 | yes |
| Dense-mix stage 2, hyper-parameter search, France house-number guard | no gain / unsupported | no |

---

## 6. Conclusion
We cast entity resolution as exact reverse retrieval, a learned candidate filter, a two-stage LightGBM matcher and an F0.5-tuned decision
with target exclusivity, all label-agnostic so the unseen country France is handled identically. The final candidate set is about 5.5
pairs per Source-1 entity with 98.8% pair recall. The lesson was that precision against planted look-alikes, not recall,
bounds the score, and that train-to-test shift (density, an unseen country) needs explicit proxies.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/run.py` is the CLI (`prepare`, `block`, `features`, `evaluate`, `predict`).
The `src/ber/` library contains io, normalize, blocking, features, approaches, metric, decoders and utils.
`src/experiments/` holds the ablation and analysis tools; `tests/` has a synthetic-data smoke test.
Reproduce with:
`python src/run.py prepare && python src/run.py block && python src/run.py features && python src/run.py evaluate --approach 2 && python src/run.py predict --approach 2` gives the base pipeline.
The submitted file adds the steps in the README section "Final submission (sub09)".
Every stage is resumable and writes timing and peak RAM to `work/telemetry.jsonl`.

### B. Additional Results
Full experiment history, ablations and leaderboard log: `EXPERIMENTS.md`, `SUBMISSIONS.md` in the project repository.
Test candidate statistics (final): 9,585,603 candidate pairs for 1,732,544 S1 entities (5.53/S1); 5,862,245 matched pairs;
every matched pair is in `candidate_pairs.tsv` (checked by `redecode.py check`); the official validator passes.
