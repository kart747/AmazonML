# Amazon ML Challenge 2026 — Business Entity Resolution
## Detailed Solution Methodology, Reasoning, Evidence, and Implementation Plan

**Prepared for:** Pranam B  
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Primary objective:** Maximize the official macro-averaged F_0.5 score while preserving a reproducible, auditable, fair-play-compliant pipeline.

---

# 1. Executive Summary

The challenge asks us to identify, for every Source 1 business, all corresponding records in Source 2 and Source 3. Source 1 is the deduplicated reference source. A Source 1 entity may have zero, one, or many matches.

The supplied data is large:

- Train Source 1: 2,206,822 lines including header
- Train Source 2: 5,034,617 lines including header
- Train Source 3: 5,285,604 lines including header
- Train ground truth: 2,206,822 lines including header
- Test Source 1: 1,732,545 lines including header
- Test Source 2: 4,887,274 lines including header
- Test Source 3: 5,082,317 lines including header

This makes exhaustive pairwise comparison infeasible. The solution therefore uses a **retrieval → candidate generation → pairwise scoring → entity-level decision** architecture.

## Proposed pipeline

```text
Raw TSVs
   |
   v
Streaming ingestion + normalization
   |
   +----------------------------+
   |                            |
   v                            v
S1 -> S2 retrieval         S1 -> S3 retrieval
   |                            |
   +------------+---------------+
                |
                v
        Multi-block candidate
             generation
                |
                v
       Candidate union + dedupe
                |
                v
       Pairwise feature engine
                |
                v
       ML pair scoring model
                |
                v
       Score calibration / QA
                |
                v
       Entity-level thresholding
                |
                v
     matching_results.tsv
                |
                +--> candidate_pairs.tsv
                |
                v
       Official validator
```

The key design principle is:

> **Do not ask the ML model to search millions of records. First retrieve a high-recall, manageable candidate set; then let the model make precision-focused decisions.**

This follows directly from the official problem statement's emphasis that blocking determines the upper bound on recall, while F_0.5 places greater weight on precision.

---

# 2. What the Challenge Actually Requires

The official documentation defines the task as business entity resolution across three independent, noisy sources.

Source 1 is the deduplicated reference source. For each Source 1 entity, we must find matching records in Source 2 and/or Source 3.

The records contain:

- `entity_id`
- `business_name`
- `business_address`
- `country`

The training ground truth contains:

- `source1_entity_id`
- `matched_entity_ids`

The matched IDs are comma-separated and may contain both S2 and S3 records.

The official documentation explicitly states that a Source 1 entity may match:

- zero records
- one record
- many records

Therefore this is **not a one-to-one nearest-neighbour problem**.

---

# 3. Dataset Reality and Its Consequences

The extracted training/test data is approximately 2.5 GB of TSV data and contains roughly 26 million rows across all files.

The largest files are approximately:

- S2 train: 489 MB
- S3 train: 504 MB
- S2 test: 509 MB
- S3 test: 506 MB

The approximate pair-space for one S1 source against one secondary source is:

2.2M × 5.0M ≈ 11 trillion pairs

Doing this for both S2 and S3 would produce more than 22 trillion comparisons before even considering test inference.

Therefore:

## Exhaustive pairwise matching is rejected

A solution that computes all S1 × S2 and S1 × S3 similarities is not a practical architecture.

Instead we require:

1. fast indexes over S2/S3
2. multiple retrieval/blocking rules
3. candidate union
4. ML scoring only on candidates

---

# 4. Official Evidence Driving the Architecture

The challenge documentation provides several important constraints.

## 4.1 Blocking is explicitly important

The official tips state:

> Invest in a strong blocking/candidate generation strategy — it determines the upper bound for recall.

This is the strongest justification for making candidate retrieval a first-class subsystem.

### Consequence

We measure two different things:

**Blocking recall**

```text
true matches that entered candidate set
---------------------------------------
all true matches
```

and:

**Final match quality**

```text
quality after the ML scorer and decision stage
```

If blocking recall is poor, improving the classifier cannot recover the lost true matches.

---

## 4.2 The metric is F_0.5

The official metric is:

```text
F_0.5 = (1.25 × Precision × Recall)
        --------------------------------
        (0.25 × Precision + Recall)
```

F_0.5 weights precision more strongly than recall.

Therefore:

- false merges are especially dangerous
- threshold selection matters
- indiscriminate fuzzy matching is dangerous
- singleton handling matters

---

## 4.3 F_0.5 is macro-averaged per Source 1 entity

The documentation says F_0.5 is calculated per Source 1 entity and then averaged over all Source 1 entities.

This is a critical design constraint.

The model should therefore not be optimized solely for pairwise accuracy.

The actual optimization target is:

> **quality of the complete predicted match set for each S1 entity.**

---

## 4.4 Singletons matter

A Source 1 entity with no true matches receives:

- 1.0 when we correctly predict an empty list
- 0.0 if we predict any incorrect match

Therefore, the system must be conservative when evidence is weak.

This strongly supports a precision-aware decision layer after pair scoring.

---

## 4.5 The test set contains France

Training contains US and India, while test additionally contains France.

Therefore:

- country cannot be hard-coded to US/India
- preprocessing must treat country as an open set
- no country-based test filtering may be used
- retrieval should not assume the training country vocabulary is complete

Country can still be used as a feature or blocking component where useful.

---

## 4.6 External lookup is prohibited

The challenge prohibits external databases, APIs, geocoding services, government registries, and internet-based business-data augmentation.

Therefore the system must learn from the supplied data only.

We must not:

- Google a business
- query a business database
- geocode an address
- use an external entity-resolution API
- augment records from the internet

This is both a fair-play requirement and an architecture constraint.

---

# 5. Proposed Solution

## High-level design

We divide the solution into six major stages:

### Stage A — Data ingestion and normalization

Read TSV files in chunks and create multiple normalized representations.

### Stage B — Index construction

Build retrieval indexes for S2 and S3.

### Stage C — Multi-block candidate generation

Retrieve candidates using several independent signals.

### Stage D — Pairwise feature generation

Compute name, address, country, token, character, and structural similarity features.

### Stage E — ML scoring

Train a precision-oriented pair classifier/ranker.

### Stage F — Entity-level decision

Convert pair probabilities into a final set of matches using validation-optimized rules.

---

# 6. Stage A — Normalization

Normalization should not destroy information.

Instead of producing one canonical string, create multiple representations.

## Business name representations

For every record:

```text
name_raw
name_lower
name_unicode_normalized
name_alnum
name_tokens
name_sorted_tokens
name_without_common_legal_suffixes
name_character_ngrams
```

Potential normalization operations:

- Unicode normalization
- lowercase
- whitespace normalization
- punctuation normalization
- repeated-symbol removal where appropriate
- conservative legal-suffix normalization
- tokenization
- optional transliteration representation

Examples:

```text
"Pvt. Ltd."
"Private Limited"
"PVT LTD"
```

can share a representation while the raw name remains available.

### Important principle

Never replace the original field.

We retain:

```text
raw
normalized
derived
```

representations simultaneously.

This protects against over-normalization collisions.

---

# 7. Address Normalization

Addresses are significantly noisier than names.

Create:

```text
address_raw
address_lower
address_normalized
address_tokens
address_alnum
numeric_tokens
alphabetic_tokens
possible_postal_tokens
```

Normalize common formatting differences without assuming a universal address grammar.

Examples:

```text
Road -> road
Rd.  -> rd
M.G. Road -> mg road
```

But we should avoid external geocoding or address databases.

---

# 8. Country Handling

Country should be represented as a normal categorical/string feature.

Do not build:

```python
if country not in {"US", "India"}:
    reject()
```

Do not construct a pipeline that can only represent two countries.

The test set contains France, so the system must generalize to an unseen country label.

Country can be used for:

- candidate retrieval
- feature generation
- score analysis

but should not become a hard eligibility rule unless validation proves that rule safe.

---

# 9. Stage B — Retrieval Indexes

We need indexes over Source 2 and Source 3.

For each source, build structures for:

### Exact normalized name

```text
normalized_name -> entity IDs
```

### Name tokens

```text
token -> entity IDs
```

### Rare name tokens

Rare tokens are particularly valuable because common words such as:

```text
shop
restaurant
company
services
india
limited
```

produce enormous candidate sets.

Instead, prioritize informative tokens.

### Address tokens

```text
address_token -> entity IDs
```

### Character n-gram retrieval

Use character n-grams to tolerate:

- spelling variation
- punctuation differences
- abbreviations
- minor typos

### Country-aware indexes

Where useful:

```text
country + key
```

should be indexed, but country should not be a mandatory global restriction.

---

# 10. Stage C — Multi-Block Candidate Generation

This is the most important subsystem.

We should not rely on one blocking rule.

## Block 1 — Exact normalized name

```text
country + normalized_name
```

High precision and extremely cheap.

---

## Block 2 — Name token retrieval

Use informative name tokens.

For example:

```text
"Shree Krishna Bakery Private Limited"
```

may generate candidates through:

```text
shree
krishna
bakery
```

but common tokens are down-weighted or ignored for retrieval.

---

## Block 3 — Character n-gram retrieval

Retrieve records with similar character fragments.

Useful for:

```text
"International South Consultants"
"International South Consultant Pvt Ltd"
```

and typo variants.

---

## Block 4 — Address token retrieval

Retrieve records sharing informative address tokens.

Examples:

```text
MG
Road
Mangalore
575001
```

---

## Block 5 — Numeric/address components

Business addresses often contain strong identifying numbers:

```text
1795
12
570
575001
```

Shared distinctive numbers can provide useful candidate evidence.

---

## Block 6 — Name + address intersection

A candidate sharing both a name signal and an address signal should receive higher priority.

We can construct candidate sets such as:

```text
name_candidates ∩ address_candidates
```

or simply add intersection membership as a feature.

---

## Candidate union

The final candidate set is:

```text
C =
C_exact_name
∪ C_name_token
∪ C_char_ngram
∪ C_address
∪ C_numeric
∪ C_combined
```

Then deduplicate IDs.

This gives us high recall while avoiding exhaustive matching.

---

# 11. Why Multiple Blocking Rules?

A single blocking rule fails on different noise types.

Example:

### Name badly corrupted

Address may recover the match.

### Address incomplete

Name may recover the match.

### Both contain minor spelling differences

Character n-gram retrieval may recover it.

### Transliteration

Token/character retrieval may recover a candidate where exact matching fails.

Therefore the union of multiple retrieval mechanisms is safer than a single block.

The validation objective is:

> maximize true-match recall subject to a manageable candidate count.

---

# 12. Candidate Budgeting

Retrieval must not explode into millions of candidates per S1.

For each retrieval mechanism, impose candidate limits based on validation.

Example starting ranges:

```text
exact name       : all
rare-token       : top 20–50
char retrieval   : top 20–100
address retrieval: top 20–100
```

These are **initial experimental values**, not final values.

We will measure:

- blocking recall
- average candidates/S1
- 95th percentile candidates/S1
- maximum candidates/S1
- reduction ratio

and choose the smallest candidate budget that preserves excellent blocking recall.

---

# 13. Blocking Evaluation

For validation:

```text
True matches
     |
     | did blocking retrieve them?
     v
+----+----+
|         |
YES       NO
|         |
v         v
model     blocking bug
```

Calculate:

```text
Blocking Recall =
# true pairs present in candidate set
--------------------------------------
# true pairs
```

Also calculate:

```text
Reduction Ratio =
1 - (# candidate pairs / # possible pairs)
```

The ideal system has:

- very high blocking recall
- extremely small candidate fraction

---

# 14. Stage D — Pairwise Feature Engineering

For every S1-candidate pair, compute features.

## Name features

### Exact equality

```text
raw_name_exact
normalized_name_exact
```

### Token Jaccard

```text
|A ∩ B|
------
|A ∪ B|
```

### Character similarity

Character-level similarity is useful for typos and formatting differences.

### Edit similarity

Derived from edit distance.

### TF-IDF cosine similarity

Useful for comparing token/character representations.

### Token containment

Useful when one business name contains the informative portion of another.

---

# 15. Address Features

Compute:

```text
address_exact
normalized_address_exact
address_jaccard
address_character_similarity
address_edit_similarity
address_tfidf_cosine
numeric_token_overlap
alphabetic_token_overlap
```

Also:

```text
shared_number_count
shared_numeric_ratio
```

These can distinguish:

```text
same street / different building
```

from:

```text
same building / same business
```

---

# 16. Cross-Field Features

Some of the most useful features should represent interactions.

Examples:

```text
country_match

name_similarity * address_similarity

name_exact AND address_similarity_high

name_high AND numeric_overlap

name_low AND address_high
```

This lets the ML model learn patterns such as:

> A moderately similar name plus highly similar address can be strong evidence.

and:

> An exact generic name without address evidence may not be enough.

---

# 17. Candidate Provenance Features

For every candidate, record which blocks produced it.

Example:

```text
from_exact_name = 1
from_name_token = 1
from_char_retrieval = 0
from_address = 1
from_numeric = 0
```

This is valuable.

A candidate independently retrieved through:

```text
name + address + character similarity
```

is more credible than one retrieved through a single weak token.

Therefore block provenance itself becomes a model feature.

---

# 18. Stage E — Training Data Construction

The ground truth gives positive matches.

For every S1:

```text
positive pairs = ground truth matches
```

But the majority of candidate pairs will be negatives.

We should create **hard negatives**, not just random negatives.

## Negative types

### Random negatives

Easy negatives.

Useful mainly for calibration.

### Same-name negatives

Two businesses with the same/similar names but different addresses.

Very important.

### Same-address negatives

Businesses sharing an address but having different names.

Also important.

### High-similarity negatives

Candidates with high fuzzy similarity but known to be different.

These are the most useful examples for a precision-heavy competition.

---

# 19. Why Hard Negative Mining Matters

The competition penalizes false merges more strongly.

A model trained only on easy random negatives learns:

```text
obviously unrelated -> negative
```

but the leaderboard problem is:

```text
very similar but different -> negative
```

Therefore the training set should deliberately contain difficult negatives.

This should improve precision in the high-score region where the final decision threshold operates.

---

# 20. Train/Validation Split

Do not randomly split individual pairs without thinking about entity leakage.

A better initial protocol:

1. split Source 1 entities into train/validation
2. construct candidate pairs within each split
3. train on train S1 entities
4. evaluate complete match sets on validation S1 entities

This keeps entity-level evaluation aligned with the competition.

---

# 21. Stage E — ML Model

Start with a strong, efficient gradient-boosted tree model.

Recommended experiment order:

## Model 0 — Heuristic baseline

Rules such as:

```text
exact normalized name + strong address -> match
```

This establishes a sanity baseline.

## Model 1 — Logistic Regression

Useful as an interpretable baseline.

## Model 2 — Gradient Boosted Trees

Primary model.

Candidate features are tabular and heterogeneous, making tree boosting a natural fit.

Possible implementations include LightGBM/XGBoost-compatible approaches subject to final license and environment verification.

The final submitted model must comply with the challenge's model licensing/parameter constraint.

---

# 22. Why Gradient Boosting?

The input is not raw text alone.

It is structured evidence:

```text
name_edit = 0.91
name_jaccard = 0.72
address_edit = 0.38
address_jaccard = 0.85
country_match = 1
numeric_overlap = 0.67
from_address_block = 1
from_name_block = 1
```

Tree models are good at learning nonlinear combinations such as:

```text
IF name similarity is high
AND address similarity is high
AND country agrees
THEN strong match
```

without requiring a huge language model.

This is also computationally attractive for millions of candidate pairs.

---

# 23. Optional Advanced Model

Only after establishing the feature-based baseline should we test a small text encoder/reranker.

Possible architecture:

```text
S1 name + address
        +
candidate name + address
        |
        v
small encoder
        |
        v
pair embedding / score
```

But this should be treated as an experiment, not the default architecture.

Reasons:

- millions of candidates
- inference cost
- model licensing requirement
- reproducibility
- likely strong performance from engineered similarity features

The competition allows models up to 8B parameters but does not require using a large model.

---

# 24. Probability Calibration

Raw classifier scores are not necessarily calibrated probabilities.

We can calibrate on validation data if useful.

Why?

Because the final system needs decisions such as:

```text
score = 0.99 -> strong match
score = 0.55 -> uncertain
```

Calibration makes threshold selection more stable.

---

# 25. Stage F — Entity-Level Decision Layer

This stage converts candidate scores into the final set.

For each S1:

```text
candidate A -> 0.997
candidate B -> 0.991
candidate C -> 0.621
candidate D -> 0.031
```

Possible final output:

```text
A,B
```

not simply:

```text
top 2
```

because the true number of matches varies.

The system therefore needs a **match threshold**, not a fixed top-k assumption.

---

# 26. Threshold Optimization

Search thresholds on the validation set.

For example:

```text
0.50
0.55
0.60
...
0.99
```

For every threshold:

1. convert pair scores to predicted match lists
2. calculate F_0.5 per S1
3. macro-average
4. select the best threshold

The final threshold should be chosen based on the official metric, not ordinary accuracy.

---

# 27. Potential Adaptive Decision Rules

After establishing a global threshold, investigate whether validation supports additional rules.

Examples:

### Strong exact-name evidence

May require less address evidence.

### Generic names

Require stronger address evidence.

### Missing address

Require stronger name evidence.

### Very high candidate ambiguity

Raise the threshold.

### Very low candidate ambiguity

Potentially allow a lower threshold if validation supports it.

These rules should only be retained if they improve held-out F_0.5.

---

# 28. Singleton Protection

A critical rule:

If no candidate exceeds the match threshold:

```text
matched_entity_ids = empty
```

This is not a failure.

It may be the correct answer.

The official evaluation explicitly gives correctly predicted singletons a score of 1.0 for that entity.

Therefore:

> **Never force a match.**

---

# 29. Cross-Source Treatment

We should initially model:

```text
S1 -> S2
S1 -> S3
```

separately but with the same feature architecture.

Why?

S2 and S3 may exhibit different noise distributions.

We can compare:

```text
Model_S2
Model_S3
```

against:

```text
Shared_Model
```

on validation.

If source-specific models materially improve F_0.5, use them.

If not, use the shared model for simplicity.

---

# 30. Potential Cross-Source Information

Because S2 and S3 can both match the same S1, there may be useful indirect evidence.

Example:

```text
S1
 |
 +---- S2 candidate
 |
 +---- S3 candidate
```

If both independently have extremely strong evidence against the same S1, that can increase confidence.

However, we should **not assume S2↔S3 matches are required**.

The official output is defined around S1 and matched S2/S3 IDs.

Therefore cross-source consistency should be an optional second-stage feature/experiment, not a prerequisite.

---

# 31. Validation Strategy

Build an evaluation framework that produces:

## Overall metrics

```text
macro F_0.5
mean precision
mean recall
```

## Entity statistics

```text
singleton accuracy
0-match F0.5
1-match F0.5
multi-match F0.5
```

## Blocking statistics

```text
blocking recall
average candidates/S1
median candidates/S1
P95 candidates/S1
max candidates/S1
candidate reduction ratio
```

## Error analysis

Inspect:

```text
false positives
false negatives
high-confidence false positives
high-confidence false negatives
```

---

# 32. Error Analysis Loop

Every experiment should follow:

```text
Train
  |
  v
Generate candidates
  |
  v
Score
  |
  v
Evaluate F0.5
  |
  v
Find errors
  |
  v
Classify error
  |
  +---- blocking failure
  |
  +---- feature failure
  |
  +---- model failure
  |
  +---- threshold failure
  |
  v
Modify one component
  |
  v
Re-evaluate
```

Do not change everything simultaneously.

This makes improvements attributable.

---

# 33. Experiment Priority

We should prioritize experiments in this order.

## Experiment 1

Exact normalized name + exact/strong address rules.

Purpose:

Establish an interpretable baseline.

## Experiment 2

Add fuzzy name/address features.

## Experiment 3

Add multi-block retrieval.

## Experiment 4

Hard-negative training.

## Experiment 5

Gradient boosting.

## Experiment 6

Threshold optimization for macro F_0.5.

## Experiment 7

Source-specific models.

## Experiment 8

Adaptive thresholds/rules.

## Experiment 9

Optional small encoder reranker.

This order avoids spending compute on complicated models before understanding the data.

---

# 34. Implementation Architecture

Recommended repository:

```text
business_entity_resolution/
|
├── src/
│   ├── config.py
│   ├── io.py
│   ├── normalize.py
│   ├── indexes.py
│   ├── blocking.py
│   ├── features.py
│   ├── training_data.py
│   ├── model.py
│   ├── calibration.py
│   ├── decision.py
│   ├── evaluation.py
│   ├── inference.py
│   └── pipeline.py
|
├── scripts/
│   ├── profile_dataset.py
│   ├── build_indexes.py
│   ├── build_training_pairs.py
│   ├── train.py
│   ├── validate.py
│   └── predict.py
|
├── configs/
│   └── baseline.yaml
|
├── README.md
└── requirements.txt
```

---

# 35. Dataset Profiling Script

Before modelling, implement:

```text
profile_dataset.py
```

It should stream through TSV files and calculate:

### Schema

```text
columns
types
```

### Missingness

```text
missing_name
missing_address
missing_country
```

### Length distributions

```text
name length
address length
```

### Country distribution

```text
US
India
France
...
```

### Duplicate rates

```text
exact name duplicates
normalized name duplicates
exact address duplicates
```

### Ground truth

```text
singleton rate
matches per S1
S2 match count
S3 match count
total match count
```

This should run without loading the complete dataset into RAM.

---

# 36. Memory Strategy

The files are too large for careless pandas loading on a 16 GB laptop.

Use:

- chunked TSV reads
- streaming preprocessing
- compact dtypes
- dictionary/inverted-index structures
- on-disk indexes where necessary
- Parquet/Arrow intermediate data if useful
- SQLite/DuckDB for analysis if convenient

Do not create enormous Python object graphs unnecessarily.

---

# 37. Candidate Generation at Scale

Potential implementation options:

### Exact indexes

Python dictionaries or compact serialized mappings.

### Token indexes

Inverted indexes.

### Character retrieval

A scalable approximate-nearest-neighbour or sparse TF-IDF retrieval layer can be evaluated.

### Sparse TF-IDF

Potentially useful because it can support fast sparse similarity without comparing every pair.

The exact implementation should be selected after profiling the actual token distributions.

---

# 38. Storage Strategy

Avoid materializing every possible pair.

Only persist:

```text
S1 ID
candidate ID
candidate provenance
```

for candidates that survive blocking.

Then compute features in batches.

Potential intermediate format:

```text
candidate_pairs.parquet
```

while the final required submission remains TSV.

---

# 39. Candidate-Pair Training Pipeline

Training should approximately be:

```text
Ground truth
     |
     v
positive pairs
     |
     +--------------------+
     |                    |
     v                    v
candidate retrieval    hard-negative retrieval
     |                    |
     +---------+----------+
               |
               v
         labeled pairs
               |
               v
         feature matrix
               |
               v
              model
```

This makes the training distribution resemble inference.

---

# 40. Avoiding Training/Inference Mismatch

A common mistake would be:

```text
train classifier on random all-dataset negatives
```

but inference only scores retrieved candidates.

Instead:

> Train on candidates produced by the same or a very similar blocking pipeline used during inference.

This makes the model learn the actual decision boundary it will encounter.

---

# 41. Leakage Prevention

Do not use:

- test labels
- external information
- future leaderboard information to alter validation improperly
- business identities obtained externally

Use only:

```text
provided training records
provided ground truth
```

for supervised learning.

---

# 42. Public Leaderboard Strategy

The public leaderboard is based on a subset of test data; final ranking uses the private leaderboard.

Therefore we should not overfit to repeated public submissions.

Recommended process:

```text
Local validation
      |
      v
Strong improvement
      |
      v
One leaderboard submission
      |
      v
Compare direction
      |
      v
Return to local validation
```

The public score should be treated as an external sanity check, not the sole optimization target.

---

# 43. Submission Generation

Generate:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

## matching_results.tsv

Exactly:

```text
source1_entity_id    matched_entity_ids
```

Every S1 test entity appears exactly once.

Empty matches remain empty.

## candidate_pairs.tsv

Exactly:

```text
source1_entity_id    candidate_entity_ids
```

Every final match must exist in the candidate list.

The official documentation explicitly requires this relationship.

---

# 44. Validation

Before every submission:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Do this automatically at the end of the inference pipeline.

The validator catches formatting and structural errors before submission.

---

# 45. Reproducibility

Final submission package:

```text
<team_name>_submission.zip
|
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
|
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
|
└── Documentation_template.md
```

Pin dependencies.

Record:

- model version
- configuration
- threshold
- blocking parameters
- feature version
- random seed

---

# 46. AWS Builder / AWS Role

AWS Builder Center is not part of the entity-resolution algorithm.

It can be useful as:

- a learning resource
- a community resource
- a place to learn AWS/ML tooling
- a source of workshops and educational material

AWS infrastructure can optionally be used for compute if allowed by the challenge environment and available resources.

But:

> Do not add AWS services merely because this is an Amazon competition.

The solution should be judged on the quality of the entity-resolution pipeline.

---

# 47. What We Should NOT Do

## Do not

- brute-force all pairs
- use Google/business search
- use external business databases
- geocode addresses
- force one match per S1
- assume every S1 has a match
- hard-code US/India
- train only on easy random negatives
- optimize only pairwise accuracy
- blindly use a huge LLM
- ignore candidate recall
- submit without local validation

---

# 48. Expected Winning Characteristics

The strongest practical system should have:

### 1. High blocking recall

True matches should almost always enter the candidate set.

### 2. Strong hard-negative discrimination

The model must distinguish similar-but-different businesses.

### 3. Precision-focused decision making

Because F_0.5 penalizes false merges strongly.

### 4. Entity-level evaluation

Because the metric is macro-averaged over Source 1 entities.

### 5. Singleton awareness

No forced matches.

### 6. Robust normalization

Multiple representations rather than destructive normalization.

### 7. Efficient inference

Millions of records require retrieval, not exhaustive comparison.

---

# 49. Implementation Roadmap

## Phase 0 — Dataset reconnaissance

Build the profiler.

Deliverables:

```text
dataset_profile.json
dataset_profile.md
```

Questions to answer:

- exact country counts
- missingness
- duplicate rates
- name/address distributions
- match-count distribution
- singleton percentage
- S2/S3 contribution distribution

---

## Phase 1 — Baseline

Implement:

```text
normalization
+
exact name retrieval
+
simple address similarity
+
rules
```

Measure F_0.5.

---

## Phase 2 — Blocking

Implement:

```text
exact-name block
token block
address block
character retrieval
numeric block
```

Measure:

```text
blocking recall
candidate volume
```

---

## Phase 3 — Feature engine

Implement all pairwise features.

Persist a manageable training feature table.

---

## Phase 4 — ML matcher

Train:

```text
Logistic Regression
Gradient Boosting
```

Compare validation F_0.5.

---

## Phase 5 — Hard negatives

Mine difficult negative candidates.

Retrain.

Measure whether precision improves.

---

## Phase 6 — Threshold optimization

Optimize directly for:

```text
macro F_0.5
```

Investigate:

- global threshold
- source-specific threshold
- country-aware threshold
- evidence-aware threshold

---

## Phase 7 — Error analysis

Analyze:

```text
top false positives
top false negatives
singleton errors
high-similarity errors
blocking misses
```

Then improve the weakest subsystem.

---

## Phase 8 — Optional advanced reranker

Only if the feature model has plateaued:

```text
candidate pair
      |
      v
small text encoder
      |
      v
reranking score
```

Benchmark against the simpler model.

---

## Phase 9 — Full test inference

Run the optimized pipeline over:

```text
test_source1
test_source2
test_source3
```

Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

---

## Phase 10 — Submission QA

Run the official validator.

Then package the reproducible submission.

---

# 50. The Core Hypothesis

Our central hypothesis is:

> **A high-recall multi-block retrieval system combined with a feature-rich, hard-negative-trained gradient-boosted matcher and an entity-level F_0.5-optimized decision layer will provide a better performance/compute/reproducibility trade-off than either brute-force fuzzy matching or a large language-model-first approach.**

This is a hypothesis.

It must be tested against validation data.

We should not assume it wins until the experiments demonstrate it.

---

# 51. The Three Metrics We Should Watch Constantly

Every experiment should report:

```text
                 BLOCKING        MATCHING
                   │                 │
                   v                 v
              Recall ceiling     F_0.5
                   │                 │
                   v                 v
              candidate count   precision/recall
```

The most important dashboard is:

```text
F0.5
Blocking Recall
Avg Candidates / S1
P95 Candidates / S1
Singleton Accuracy
```

A model that increases F_0.5 from 0.90 to 0.905 while multiplying candidates by 20 may not be worth keeping.

---

# 52. First Concrete Task

Before writing the complete matcher, run:

```bash
python3 scripts/profile_dataset.py
```

against the training files.

The first report should answer:

1. How many unique normalized names exist?
2. How many S1 records have exact-name candidates?
3. How large are exact-name candidate sets?
4. How many S1s are singletons?
5. What is the distribution of number of matches per S1?
6. What fraction of matches are S2 vs S3?
7. How often do true matches share exact normalized names?
8. How often do true matches share useful address tokens?
9. How many duplicate names correspond to different businesses?
10. What percentage of candidates would each block generate?

These results will determine the actual blocking configuration.

---

# 53. Final Architecture

The intended final system is:

```text
                 ┌─────────────────────┐
                 │  S1 / S2 / S3 TSVs  │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Streaming ingestion │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Multi-view normalize│
                 └──────────┬──────────┘
                            |
              ┌─────────────┴─────────────┐
              v                           v
      ┌───────────────┐           ┌───────────────┐
      │ S2 indexes    │           │ S3 indexes    │
      └───────┬───────┘           └───────┬───────┘
              |                           |
              v                           v
      ┌───────────────┐           ┌───────────────┐
      │ Multi-block   │           │ Multi-block   │
      │ retrieval     │           │ retrieval     │
      └───────┬───────┘           └───────┬───────┘
              |                           |
              └─────────────┬─────────────┘
                            v
                 ┌─────────────────────┐
                 │ Candidate union     │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Pairwise features   │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ ML matcher          │
                 │ gradient boosting   │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Entity-level        │
                 │ F0.5 decision layer │
                 └──────────┬──────────┘
                            |
                   ┌────────┴────────┐
                   v                 v
          matching_results.tsv   candidate_pairs.tsv
```

---

# 54. Bottom Line

The competition is fundamentally a **large-scale retrieval problem followed by a precision-sensitive matching problem**.

The winning direction is therefore not:

> "Find a bigger model."

It is:

> **"Retrieve the right candidates extremely reliably, teach the matcher to reject hard false matches, and optimize the final entity-level decision for the exact F_0.5 metric."**

Our first implementation should therefore focus on **dataset profiling and blocking recall**, before spending serious effort on model complexity.

