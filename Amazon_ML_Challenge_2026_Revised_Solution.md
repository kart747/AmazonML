# Amazon ML Challenge 2026 — Revised Entity Resolution Solution

## Purpose

This document records the changes recommended to the reviewed solution for the Amazon ML Challenge 2026 large-scale Business Entity Resolution / Record Linkage problem.

The original review proposed a two-stage retrieval + matching system:

1. Country blocking.
2. Fuzzy blocking using LSH / MinHash or dense embeddings such as `all-MiniLM-L6-v2`.
3. Re-ranking using a cross-encoder or LightGBM / CatBoost over pairwise similarity features.
4. A conservative global probability threshold around 0.65–0.70 because the metric is macro-averaged F0.5 and therefore precision-heavy.

The revised solution keeps the core two-stage architecture, but changes the retrieval, training, calibration, and final decoding strategy.

---

# 1. Executive Summary of the Changes

## Original reviewed approach

```text
Country block
    ↓
LSH / MinHash / dense embeddings
    ↓
Top-K candidates (e.g. K=20)
    ↓
Cross-encoder / LightGBM
    ↓
Global threshold (e.g. 0.65–0.70)
    ↓
Final matches
```

## Revised approach

```text
                    Raw S1 / S2 / S3
                           ↓
                 Canonical normalization
                           ↓
              Native + Latin representations
                           ↓
                   Country partition
                           ↓
              Multi-view candidate retrieval
                           ↓
              Candidate union + deduplication
                           ↓
                  Pair feature generation
                           ↓
                Hard-negative-aware LightGBM
                           ↓
                 OOF / cross-fitted scores
                           ↓
               Collective/context features
                           ↓
                    LightGBM stage 2
                           ↓
                  Probability calibration
                           ↓
               Target-side exclusivity
                           ↓
              Expected-F0.5-aware decoding
                           ↓
                    Final submission
```

The main conceptual change is:

> **Use cheap, diverse retrieval views to maximize candidate recall; use a strong tabular matcher to maximize precision; then make the final decision at the entity/constraint level rather than using one global probability threshold.**

---

# 2. Change #1 — Replace Embedding-First Blocking with Multi-View Retrieval

## Original

The reviewed solution suggested:

```text
country
  ↓
LSH / MinHash / dense embeddings
  ↓
top-K candidates
```

with `sentence-transformers/all-MiniLM-L6-v2` as a possible dense retrieval model.

## Revised

Use multiple cheap retrieval mechanisms in parallel:

```text
                Country partition
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Name TF-IDF     Char n-grams      Exact keys
       ↓               ↓                ↓
   Name+number     Name+city       Rare-token index
       ↓               ↓                ↓
           Transliteration retrieval
                       ↓
             Optional dense retrieval
                       ↓
                      UNION
                       ↓
               Candidate set
```

### Why this change

Dense retrieval is useful, but it should not be the only or primary candidate generator. Business entity resolution is dominated by noisy lexical variations, abbreviations, transliteration, address components, typos, punctuation, and structured identifiers.

A single embedding model can miss an exact-but-obfuscated lexical relationship. Sparse and exact retrieval views provide complementary recall.

### Recommended blocking views

1. Normalized full business name.
2. Character 3–5 gram TF-IDF similarity.
3. Core business-name token index after removing common legal suffixes.
4. Name + house/building number.
5. Name + city/location tokens.
6. Rare-token inverted index.
7. Native-script / Latin-transliterated name views.
8. Optional dense embedding retrieval.
9. Reverse retrieval from S2/S3 → S1.

### Design principle

Blocking should be **high recall and relatively permissive**.

The matcher, not the blocker, should be responsible for being strict.

---

# 3. Change #2 — Keep Country Blocking, but Make It Open-Set Safe

## Original

The solution correctly identified country as a powerful first-stage blocker because the public dataset analysis reports zero cross-country ground-truth matches.

## Revised

Keep country blocking, but do not hard-code the known training countries.

Bad:

```python
if country == "US":
    ...
elif country == "India":
    ...
```

Better:

```python
partition_by_country(record.country)
```

This lets unseen test-time countries, such as France in the reported dataset analysis, automatically receive the same retrieval pipeline.

### Add a fallback for:

- missing country
- unknown country
- corrupted country values
- country parsing failures

Fallback:

```text
invalid / missing country
        ↓
broader multilingual candidate retrieval
```

### Important principle

Country is primarily a **retrieval partition**, not a reason to memorize country-specific heuristics.

---

# 4. Change #3 — Make Normalization a Multi-View System

## Original

The reviewed solution emphasized fuzzy string similarity but did not make representation diversity central enough.

## Revised

Maintain several versions of each field.

### Business name

```text
raw_name
normalized_name
core_name
native_script_name
latin_transliterated_name
```

### Address

```text
raw_address
normalized_address
house_number
postal_code
city
state_or_region
street_tokens
unit_number
```

Do **not** replace native script with transliteration. Keep both.

Example:

```text
native representation
        +
Latin transliteration
        +
normalized token representation
```

This is particularly important for India and any multilingual test-time domain shift.

---

# 5. Change #4 — Add Strong Address Decomposition

## Original

The reviewed approach treated address as a fuzzy text field.

## Revised

Treat addresses as a structured combination of components.

Create explicit features for:

```text
house_number_exact
postal_code_exact
city_exact
state_exact
street_overlap
address_token_similarity
address_char_similarity
number_conflict
unit_number_match
```

### Why

Two businesses can have almost identical names while being different entities:

```text
Sai Motors
14 MG Road
```

versus:

```text
Sai Motors
17 MG Road
```

A name-only or embedding-heavy approach may assign high similarity. The explicit number conflict should push the pair score down.

This also gives LightGBM interpretable structured signals instead of asking one similarity score to encode everything.

---

# 6. Change #5 — Add Missingness Features

Missingness is information.

Add:

```text
name_missing_s1
name_missing_s2_or_s3
address_missing_s1
address_missing_s2_or_s3
country_missing_s1
country_missing_target
```

Distinguish:

```text
missing ↔ missing
missing ↔ present
present ↔ conflicting
present ↔ matching
```

A missing value is not equivalent to a contradictory value.

---

# 7. Change #6 — Make Hard Negatives a Core Training Component

This is one of the most important upgrades.

## Original

The reviewed approach proposed pairwise classification but did not make hard-negative construction central.

## Revised

Train the matcher mostly on difficult non-matches, not just random negatives.

### Random negative

```text
S1: Apollo Medical Center
S2: Krishna Hardware
```

Easy and low value.

### Hard negative

```text
S1: Apollo Medical Center
S2: Apollo Medical Centre
same city
similar address
```

or:

```text
S1: Sai Motors, 14 MG Road
S2: Sai Motors, 17 MG Road
```

or:

```text
S1: Shree Krishna Jewellers
S2: Shree Krishna Jewellery
same locality
```

These examples train the model to separate near-duplicates that matter for precision.

### Negative sampling sources

Generate hard negatives from:

- top retrieval candidates that are not ground-truth matches
- same-city candidates
- same-name candidates
- similar-address candidates
- same-brand / same-core-name candidates
- near-duplicate lexical candidates

The closer the negative is to the positive, the more useful it is for improving the precision-heavy metric.

---

# 8. Change #7 — Prefer LightGBM as the Primary Pair Matcher

## Original

The reviewed solution gave equal conceptual weight to a cross-encoder and tree-based methods.

## Revised

Make LightGBM the primary matcher for the first strong baseline.

Why:

The pair representation is naturally tabular:

```text
name_ratio
name_token_jaccard
name_char_cosine
address_ratio
city_match
postal_match
house_number_match
number_conflict
rare_token_overlap
transliteration_similarity
length_difference
candidate_rank
candidate_margin
...
```

This is exactly the kind of heterogeneous feature space where gradient-boosted trees are useful.

A neural cross-encoder can be tested later as an ensemble component, but it should not be the dependency that determines whether the pipeline works.

---

# 9. Change #8 — Expand the Pairwise Feature Set

Recommended feature families:

## Name

- RapidFuzz ratio
- WRatio
- token-sort ratio
- token-set ratio
- Levenshtein ratio
- Jaro-Winkler
- char n-gram cosine
- word TF-IDF cosine
- token Jaccard
- token containment
- acronym compatibility
- rare-token overlap
- normalized length difference

## Address

- character similarity
- token similarity
- token Jaccard
- city exact match
- state exact match
- postal exact match
- house-number exact match
- unit-number match
- street overlap
- conflicting-number indicator

## Cross-field / structural

- country conflict
- script compatibility
- transliteration agreement
- name/address interaction
- number/name interaction
- city/name interaction

## Missingness

- field missing indicators

---

# 10. Change #9 — Add Candidate-Context Features

Do not score a pair in isolation.

Suppose the candidate scores are:

```text
A = 0.98
B = 0.97
C = 0.31
```

A is much more ambiguous than:

```text
A = 0.98
B = 0.51
C = 0.30
```

Therefore include:

```text
top1_score
top2_score
top1_minus_top2
candidate_rank
candidate_count
score_percentile
```

These features tell the model whether a candidate is clearly dominant or competing against near-equals.

---

# 11. Change #10 — Add Reverse Retrieval

Do not only perform:

```text
S1 → S2/S3
```

Also perform:

```text
S2/S3 → S1
```

Then take the union of candidates.

Why:

A retrieval mechanism that is asymmetric can miss candidates in one direction because of indexing or normalization differences.

Reverse retrieval provides another route to recall.

---

# 12. Change #11 — Add a Second Matching Stage Only After the Baseline Works

The revised system can become a stacked model.

## Stage 1

```text
candidate pair
   ↓
LightGBM-1
   ↓
OOF probability
```

## Stage 2

Use Stage-1 probabilities plus contextual features:

```text
stage1_probability
top1_margin
candidate_density
sibling_support
source_type
retrieval_view_count
```

→ LightGBM-2

Do not start here.

First establish a clean Stage-1 baseline and verify that every advanced component genuinely improves entity-level validation F0.5.

---

# 13. Change #12 — Use Out-of-Fold / Cross-Fitted Scores for Collective Features

Collective or sibling features can be powerful, but they introduce leakage risk.

Wrong:

```text
train model on entity
        ↓
predict same entity
        ↓
use prediction as feature
```

Correct:

```text
fold 1 model → predicts fold 2
fold 2 model → predicts fold 1
...
        ↓
OOF probabilities
        ↓
collective features
```

Only use cross-fitted predictions for training-time meta-features.

---

# 14. Change #13 — Add Collective / Sibling Evidence

If multiple target records independently provide evidence for the same S1 entity, that can become useful contextual information.

Example:

```text
S1-A
 ├── S2-X high probability
 └── S2-Y high probability

S2-X and S2-Y share identity evidence
```

This can generate features such as:

```text
sibling_support
mean sibling probability
max sibling probability
number of supporting siblings
cluster consistency
```

However, the feature must be computed without leakage using OOF/cross-fitted scores.

---

# 15. Change #14 — Do Not Hard-Code a Global 0.65–0.70 Threshold

The original reviewed solution suggested a conservative threshold around 0.65–0.70 because F0.5 is precision-heavy.

The revised position is:

> The threshold must be learned from validation rather than assumed.

Public implementations report different useful thresholds and threshold values depend on the exact validation setup.

Potentially test:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

but choose based on **entity-level macro F0.5**, not pairwise accuracy.

Also test source-specific thresholds if S2 and S3 behave differently.

---

# 16. Change #15 — Calibrate Probabilities Before Decoding

Tree-model probabilities are not guaranteed to be perfectly calibrated.

Use a held-out calibration stage such as:

- isotonic regression
- Platt scaling / logistic calibration

The goal is for a score to behave more like an interpretable probability.

Then use the calibrated score in final decoding.

---

# 17. Change #16 — Make the Decoder F0.5-Aware

This is a major conceptual improvement.

Do not simply do:

```python
prediction = probability > threshold
```

Instead, for each S1 entity, consider the candidate set and estimate which prediction subset produces the best expected F0.5 under the calibrated probabilities.

Conceptually:

```text
S1 entity
   ↓
candidate probabilities
   ↓
possible selected subsets
   ↓
expected F0.5
   ↓
best decision
```

For practical scale, this can be simplified to a ranked-prefix or dynamic-selection strategy rather than exhaustive subset search.

---

# 18. Change #17 — Enforce Target-Side Exclusivity

A target record from S2/S3 should not be assigned to multiple S1 entities.

Example:

```text
S2-X
 ├── S1-A = 0.93
 └── S1-B = 0.91
```

The system must resolve ownership rather than independently accepting both predictions.

This can be implemented as:

1. rank candidate assignments for each target;
2. retain the strongest calibrated assignment where justified;
3. pass the resulting assignments into S1-level F0.5 decoding.

Important distinction:

> S1 can have multiple linked S2/S3 records; the exclusivity constraint applies to the target record side.

---

# 19. Revised End-to-End Pipeline

```text
RAW TSV FILES
│
├── S1
├── S2
└── S3
        │
        ▼
CANONICAL NORMALIZATION
│
├── raw views
├── normalized views
├── core tokens
├── native script
├── Latin transliteration
└── parsed address components
        │
        ▼
COUNTRY PARTITION
        │
        ▼
MULTI-VIEW CANDIDATE RETRIEVAL
│
├── name TF-IDF
├── character n-grams
├── exact / rare-token keys
├── name + number
├── name + city
├── transliterated views
├── optional embeddings
└── reverse retrieval
        │
        ▼
UNION + DEDUPLICATE
        │
        ▼
PAIR FEATURE GENERATION
        │
        ▼
HARD-NEGATIVE-AWARE LIGHTGBM
        │
        ▼
OOF PROBABILITIES
        │
        ▼
COLLECTIVE / CONTEXT FEATURES
        │
        ▼
LIGHTGBM STAGE 2
        │
        ▼
CALIBRATION
        │
        ▼
TARGET-SIDE EXCLUSIVITY
        │
        ▼
EXPECTED-F0.5 DECODING
        │
        ▼
SUBMISSION
```

---

# 20. Validation Strategy

## Primary validation

Split by S1 entity rather than by individual pair.

```text
S1 entities
   ↓
train entities / validation entities
```

All associated target records remain grouped with their S1 entity.

This avoids leakage where records from the same entity appear in both train and validation.

## Metrics to monitor

Do not monitor only final F0.5.

Track:

```text
candidate recall
mean candidates/entity
pair precision
pair recall
final precision
final recall
macro F0.5
singleton accuracy
runtime
memory usage
```

Candidate recall is especially important because missing a true candidate at retrieval time creates an unrecoverable false negative.

---

# 21. Stress Validation for Test-Time Domain Shift

The reported public dataset analysis indicates an unseen-France test component relative to US/India training.

Do not train a France-specific model.

Instead, test how robust the representation is to domain shift.

Build stress splits that hold out:

- regions
- cities
- naming patterns
- scripts
- address styles

The goal is to detect models that are memorizing geography or training-source quirks rather than learning entity correspondence.

---

# 22. Computational Architecture

## Data layer

Use:

- Polars
- DuckDB
- Parquet where possible

Do not materialize every possible pair.

## Retrieval

Use inverted indexes / sparse matrices / ANN indexes only where beneficial.

## Matching

Use LightGBM with batched feature computation.

## Storage

Recommended intermediate artifacts:

```text
normalized_s1.parquet
normalized_s2.parquet
normalized_s3.parquet
candidate_pairs.parquet
features_part_*.parquet
predictions.parquet
submission.csv
```

This makes experiments reproducible and avoids recomputing expensive upstream stages.

---

# 23. Recommended Experiment Order

Do not attempt the full architecture immediately.

## Experiment 0 — Sanity baseline

Country + exact normalized name.

Purpose: understand the dataset.

## Experiment 1 — Multi-view retrieval

Add:

- char TF-IDF
- token TF-IDF
- name+number
- name+city

Measure candidate recall.

## Experiment 2 — Strong LightGBM

Add the complete name/address feature set.

Measure pairwise and entity-level F0.5.

## Experiment 3 — Hard negatives

Replace random negatives with retrieval-generated hard negatives.

Measure precision gain.

## Experiment 4 — Calibration

Add isotonic or Platt calibration.

Measure decoder performance.

## Experiment 5 — Context features

Add candidate margins, density and source-aware features.

## Experiment 6 — Collective features

Add OOF sibling/cluster evidence.

## Experiment 7 — Exclusivity + F0.5-aware decoder

Optimize the final prediction process.

## Experiment 8 — Optional embeddings

Add dense retrieval only if validation shows meaningful candidate-recall or final-score gains.

---

# 24. What We Are Explicitly NOT Doing

## Not using a single global embedding model as the blocker

Too brittle for noisy structured records.

## Not using a fixed 0.70 probability threshold

Thresholds are learned from validation.

## Not training on mostly random negatives

Hard negatives are required.

## Not treating addresses as one plain string

Address components are extracted separately.

## Not hard-coding US/India/France branches

Country handling is open-set and data-driven.

## Not relying on pairwise accuracy

The actual objective is macro F0.5 over entities.

## Not leaking OOF/collective predictions

Cross-fitting is required.

## Not assigning targets independently to multiple S1 entities

Target-side exclusivity must be enforced.

---

# 25. Priority Ranking of Components

| Component | Priority |
|---|---|
| Entity-level validation | Critical |
| Multi-view blocking | Critical |
| Hard-negative generation | Critical |
| Name/address feature engineering | Critical |
| Probability calibration | Very High |
| F0.5-aware decoding | Very High |
| Target-side exclusivity | Very High |
| Transliteration / multilingual normalization | Very High |
| Candidate-context features | High |
| Collective / sibling features | High |
| Dense embeddings | Medium |
| Cross-encoder | Low initially |
| Large custom neural model | Very Low |

---

# 26. Final Revised Strategy

The revised solution is therefore:

> **High-recall multi-view retrieval + hard-negative LightGBM matching + calibrated entity-level F0.5-aware decoding.**

The most important practical changes from the original reviewed solution are:

1. Replace embedding-first blocking with **multi-view sparse/exact retrieval**.
2. Keep **country blocking**, but make it open-set safe.
3. Add **native + transliterated representations** rather than a single normalized string.
4. Decompose addresses into **structured components**.
5. Add **missingness and contradiction features**.
6. Make **hard-negative generation** central to training.
7. Prefer **LightGBM** as the initial pair matcher.
8. Add **candidate-context / margin features**.
9. Add **reverse retrieval**.
10. Use **OOF/cross-fitting** for collective features.
11. Add **probability calibration**.
12. Replace the fixed global threshold with **validation-learned decoding**.
13. Explicitly enforce **target-side exclusivity**.
14. Optimize the **final entity-level macro F0.5**, not pairwise accuracy.

---

# 27. Bottom Line

The original reviewed solution had the correct high-level idea: blocking first, matching second, conservative decisions because F0.5 penalizes false positives strongly.

The revised solution makes the pipeline more competition-oriented:

```text
RETRIEVAL
maximize recall
        ↓
MATCHING
maximize precision
        ↓
CALIBRATION
make probabilities trustworthy
        ↓
DECODING
respect entity-level F0.5 + target exclusivity
```

The biggest strategic shift is that **retrieval, matching, and decoding are treated as three separate optimization problems** rather than one pipeline ending in a probability threshold.

---

# Sources / Public Implementations Reviewed

- Ayan Ahmed Khan — Amazon ML Challenge repository:
  https://github.com/AyanAhmedKhan/amazon-ml-challenge
- Akash Bardia — Amazon ML Challenge 2026:
  https://github.com/Akash-bardia/amazon-ml-challenge-2026
- jvikram — Business Entity Resolution:
  https://github.com/jvikramsrd/business_entity_resolution
- Sid-techweb — AmazonML-New:
  https://github.com/Sid-techweb/AmazonML-New
- Amazon ML Challenge 2026 dataset packaging / analysis:
  https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026

Note: performance figures reported by these public repositories are self-reported validation results under their respective validation methodologies, not official leaderboard rankings.
