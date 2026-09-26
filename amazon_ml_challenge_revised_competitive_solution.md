# Amazon ML Challenge 2026 — Revised Competitive Solution

## Executive decision

After reviewing the two public implementations shared by the team:

- [AyanAhmedKhan/amazon-ml-challenge](https://github.com/AyanAhmedKhan/amazon-ml-challenge)
- [jvikramsrd/business_entity_resolution](https://github.com/jvikramsrd/business_entity_resolution)

I would **upgrade our original solution substantially**.

Our original plan was:

```text
normalization
→ multi-block retrieval
→ pair features
→ LightGBM
→ threshold
```

The revised plan is:

```text
strict ingestion
→ multiview normalization
→ learned transliteration / abbreviation views
→ high-recall bounded retrieval
→ pairwise features
→ LightGBM stage 1
→ out-of-fold collective / sibling features
→ LightGBM stage 2
→ held-out probability calibration
→ per-S1 expected-F0.5 decoding
→ target-record exclusivity
→ final submission
```

This is not a blind copy of either repository. We combine the strongest ideas and make expensive components conditional on our own validation results.

---

# 1. Evidence from the two public implementations

## AyanAhmedKhan

The repository describes a pipeline using:

- multi-view retrieval
- learned native-to-Latin dictionaries
- learned abbreviation maps
- five sparse TF-IDF views per country partition
- reverse target→S1 retrieval
- static multilingual embeddings
- RapidFuzz/string/IDF/number features
- stage-1 LightGBM
- 3-fold cross-fit predictions
- collective/sibling-support features
- stage-2 LightGBM
- thresholding
- many-to-one exclusivity

It also documents that its full-scale pipeline can reach roughly 90M candidate pairs and recommends substantially more compute than our laptop provides.

Source: https://github.com/AyanAhmedKhan/amazon-ml-challenge

## jvikramsrd

The repository describes:

```text
strict TSV ingestion
→ multiview normalization
→ transliteration of 9 Indic scripts
→ bounded inverted-index blocking
→ string/number/context features
→ LightGBM
→ held-out isotonic calibration
→ expected-F0.5 decoding
→ record exclusivity
```

It is explicitly CPU-capable and emphasizes resumability, validation, manifests, and hardware-aware execution.

Source: https://github.com/jvikramsrd/business_entity_resolution

The fact that two independent implementations converge on bounded retrieval, GBDT matching, calibration/decision optimization, and structural constraints is strong evidence that these components deserve priority. It is not proof that every component will improve our leaderboard score; we will verify each through ablation.

---

# 2. What changes from our original solution

The most important upgrade is the **decision layer**.

A simple rule:

```text
score > threshold → match
```

does not fully exploit the metric.

For one S1 we may have:

```text
S2-A  0.98
S2-B  0.94
S3-C  0.81
S3-D  0.27
```

The real question is not just whether each pair crosses a threshold.

The real question is:

```text
Should the final set be:
{}
{A}
{A,B}
{A,B,C}
...
```

The public Ayan implementation explicitly implements expected-F0.5 decoding with a Poisson-binomial dynamic program and allows the empty set. jvikramsrd also uses expected-F0.5 decoding.

Therefore our revised system should optimize the **complete predicted match set per S1**, not just pair classification.

---

# 3. Revised architecture

```text
                         RAW DATA
                            |
                            v
                 ┌─────────────────────┐
                 │ Strict TSV ingestion│
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Multiview normalize│
                 │ raw / canonical     │
                 │ transliterated      │
                 │ abbreviation-aware  │
                 │ tokens / ngrams     │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Retrieval / Blocking│
                 │ exact / token /     │
                 │ TF-IDF / char /     │
                 │ address / numeric / │
                 │ reverse retrieval   │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Candidate union     │
                 │ + bounded top-K     │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Pairwise features   │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Stage-1 LightGBM    │
                 └──────────┬──────────┘
                            |
                     OOF probabilities
                            |
                            v
                 ┌─────────────────────┐
                 │ Collective/context  │
                 │ sibling features    │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Stage-2 LightGBM    │
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Isotonic calibration│
                 └──────────┬──────────┘
                            |
                            v
                 ┌─────────────────────┐
                 │ Expected-F0.5       │
                 │ entity decoder      │
                 │ + exclusivity       │
                 └──────────┬──────────┘
                            |
                            v
                   final submission
```

---

# 4. Multiview normalization

Do not create one destructive normalized string.

Maintain:

```text
raw
canonical
alphanumeric
tokens
character n-grams
transliterated
abbreviation-normalized
```

This is especially important because our actual data already contains Indic-script business names.

For example:

```text
राम मार्केटिंग प्राइवेट लिमिटेड
आदित्य प्रॉपर्टीज एलएलपी
```

A Latin-only matcher can miss cross-script matches.

The public implementations independently emphasize multilingual normalization/transliteration, with Ayan learning native→Latin mappings and jvikramsrd describing nine Indic scripts.

### Rule

Keep the original representation and add derived views. Never replace the raw value.

---

# 5. Learned transliteration

Rather than depending on an external dictionary, learn recurring transformations from the supplied training data.

Maintain:

```text
native representation
+
learned Latin representation
+
raw representation
```

This keeps us compliant with the no-external-data requirement and prevents transliteration errors from destroying information.

We should measure whether learned mappings improve:

- blocking recall
- candidate volume
- final F0.5

---

# 6. Learned abbreviation maps

Learn high-frequency equivalences from the supplied data where supported.

Examples may include:

```text
pvt ↔ private
ltd ↔ limited
co ↔ company
corp ↔ corporation
```

These should create additional views rather than overwrite the source text.

---

# 7. Retrieval / blocking

Blocking remains the recall ceiling.

We use several retrieval families:

### A. Exact normalized name

Very fast and high precision.

### B. Token inverted index

Use informative tokens and avoid enormous common-token blocks.

### C. Sparse TF-IDF / character retrieval

Handles spelling noise, punctuation differences, and abbreviations.

### D. Address token retrieval

Useful when the name is corrupted.

### E. Numeric retrieval

House numbers, unit numbers, postal codes, etc.

### F. Name + address retrieval

High-precision combined evidence.

### G. Reverse target→S1 retrieval

Benchmark this because the public Ayan implementation uses it.

---

# 8. Country partitioning

Country can dramatically reduce the retrieval universe.

However:

**Never hard-code US/India.**

The test set introduces France, so the implementation must dynamically handle unseen country labels.

Correct:

```text
partition by whatever country values occur
```

Incorrect:

```text
if country in {"US", "India"}:
```

Country can be used as:

- a retrieval partition
- a model feature
- an error-analysis dimension

but should not become a hard rejection rule for unseen countries.

---

# 9. Candidate budgets

Ayan reports a full-scale pipeline with around 90M candidate pairs and recommends high-end compute.

We should not blindly reproduce that on a 16 GB laptop.

Instead we will optimize the **recall/candidate-count Pareto frontier**.

For each configuration measure:

```text
blocking recall
average candidates/S1
P95 candidates/S1
P99 candidates/S1
total candidates
runtime
RAM
```

Then select the smallest candidate set that preserves excellent recall.

---

# 10. Pairwise features

## Name

```text
exact equality
normalized equality
token Jaccard
character similarity
edit similarity
TF-IDF cosine
token containment
length ratio
shared informative tokens
```

## Address

```text
exact equality
normalized equality
token Jaccard
character similarity
edit similarity
TF-IDF cosine
numeric overlap
postal-like overlap
city/state token overlap
```

## Cross-field

```text
country_match
name_similarity × address_similarity
name_high AND numeric_overlap
name_high AND address_high
```

## Retrieval provenance

```text
from_exact_name
from_token
from_tfidf
from_char
from_address
from_numeric
from_reverse
```

## Rank/density

```text
retrieval_rank
number_of_candidates
number_of_high_name_candidates
number_of_high_address_candidates
score_margin_to_next_candidate
```

---

# 11. Why candidate density matters

Consider:

```text
S1-A:
best candidate = 0.92
only 2 plausible candidates

S1-B:
best candidate = 0.92
100 plausible candidates
```

The raw score has different meaning.

Therefore candidate density and score-margin features can help the model distinguish an isolated strong match from an ambiguous generic-name situation.

---

# 12. Training pairs and hard negatives

Ground truth provides positives.

But random negatives are not enough.

Create:

### Easy negatives

Random candidate pairs.

### Same-name negatives

Similar/exact name but different entity.

### Same-address negatives

Same address but different business.

### Hard negatives

Very high retrieval similarity but known false match.

### Structural negatives

Pairs that compete for the same target record.

This is particularly important because F0.5 weights precision more heavily than recall.

---

# 13. Avoid training/inference mismatch

Do not train on arbitrary random pairs if inference scores only retrieved candidates.

Instead:

```text
same retrieval system
→ candidate distribution
→ labeled candidate pairs
→ model training
```

This makes the classifier learn the decision boundary it will actually see.

Ayan also describes density-matched training; this principle should be adopted.

---

# 14. Stage-1 LightGBM

Use LightGBM on the structured pair features.

Why:

- heterogeneous numeric features
- nonlinear interactions
- fast inference
- efficient training
- strong fit for tabular similarity features
- easier to scale than a large text model

The first objective is a strong pairwise probability estimate.

---

# 15. Cross-fitted Stage-1 predictions

Use out-of-fold predictions when constructing downstream contextual features.

Example:

```text
fold 1:
train on folds 2,3
predict fold 1

fold 2:
train on folds 1,3
predict fold 2

fold 3:
train on folds 1,2
predict fold 3
```

This prevents downstream collective features from simply consuming predictions made by a model that trained on those same rows.

This is one of the strongest ideas in the Ayan implementation.

---

# 16. Collective / sibling features

Some source records may be near-copies or multiple noisy observations of the same business.

For an S1:

```text
S1
 ├── S2-A
 ├── S2-B
 └── S3-C
```

we can derive:

```text
sibling_count
mean sibling probability
max sibling probability
source diversity
cluster support
number of high-confidence neighbors
```

These become stage-2 features.

### Important

This is an experiment, not a guaranteed improvement.

If held-out F0.5 does not improve, remove it.

---

# 17. Stage-2 LightGBM

Stage 2 sees:

```text
stage-1 probability
+
original pair features
+
candidate density
+
collective/sibling support
+
retrieval provenance
```

Conceptually:

```text
Stage 1:
"Is this pair plausible?"

Stage 2:
"Is this pair plausible given the surrounding candidate evidence?"
```

This is the main advanced model experiment.

---

# 18. Calibration

Raw LightGBM probabilities are not guaranteed to be calibrated.

This matters because our decoder will use probabilities to estimate expected F0.5.

Therefore benchmark:

```text
no calibration
sigmoid calibration
isotonic calibration
```

on a held-out calibration split.

jvikramsrd specifically uses held-out isotonic calibration, making this a strong candidate.

---

# 19. Expected-F0.5 decoding

For each S1:

```text
p1 ≥ p2 ≥ p3 ≥ ...
```

consider:

```text
k = 0
k = 1
k = 2
...
```

and select:

```text
k* = argmax E[F0.5(top-k)]
```

The empty set is explicitly allowed.

A Poisson-binomial dynamic program can compute the expected score efficiently.

This is superior to assuming:

```text
threshold = 0.5
```

because the actual competition objective is set-level macro F0.5.

---

# 20. Singleton protection

If all candidates have insufficient evidence:

```text
prediction = {}
```

Do not force the top candidate.

A correctly predicted singleton is valuable under the competition metric.

---

# 21. Target-record exclusivity

The public solutions use the structural rule that each S2/S3 target record belongs to at most one S1.

We should first verify this directly in our supplied ground truth.

If confirmed:

```text
S2-X:
S1-A = 0.96
S1-B = 0.94
```

should not normally be assigned to both.

This is a strong precision prior and should be evaluated as part of the decoder.

Do not enable it blindly before verifying the training data.

---

# 22. Optional multilingual embeddings

Ayan uses the MIT-licensed `minishlab/potion-multilingual-128M` static embedding model.

We should **not start here**.

Run:

```text
baseline without embeddings
vs
embedding-enhanced model
```

only after the retrieval + LightGBM + decoder pipeline is strong.

Keep the embedding model only if its validation improvement justifies:

- runtime
- RAM
- complexity
- compliance review

---

# 23. Comparison of approaches

| Component | Original plan | Ayan | jvikramsrd | Our revised plan |
|---|---|---|---|---|
| Multiview normalization | Basic | Strong | Strong | **Strong** |
| Indic transliteration | Optional | Learned | 9 scripts | **Learned** |
| Abbreviation learning | No | Yes | Yes | **Yes** |
| Bounded retrieval | Yes | Yes | Strong | **Yes** |
| TF-IDF retrieval | Optional | Strong | Less central | **Benchmark/core** |
| Reverse retrieval | No | Yes | Not central | **Benchmark** |
| Pair features | Yes | Strong | Strong | **Strong** |
| LightGBM | Yes | Yes | Yes | **Yes** |
| Cross-fit | No | Yes | Not central | **Yes** |
| Collective features | No | Yes | Not central | **Experimental** |
| Stage 2 | No | Yes | No | **Experimental/core if useful** |
| Calibration | No | Limited | Isotonic | **Yes** |
| Expected F0.5 | No | Yes | Yes | **Core** |
| Exclusivity | No | Yes | Yes | **Core if verified** |
| Static embeddings | Optional | Yes | No | **Optional** |
| External data | No | No | No | **No** |

---

# 24. What we should NOT copy blindly

## Do not reproduce a 90M-candidate pipeline

Our hardware is much smaller.

## Do not add every feature

Every feature must pass ablation.

## Do not start with an embedding model

The retrieval + structured model may already be excellent.

## Do not assume exclusivity

Verify it from the actual ground truth.

## Do not use fixed thresholds as the final decoder

Expected-F0.5 decoding should be benchmarked.

## Do not use external transliteration/business databases

Learn from the supplied data only.

---

# 25. Hardware-aware design

Our local machine has approximately:

- 16 GB RAM
- RTX 4050 6 GB VRAM

Therefore the pipeline should be CPU-first.

Use tools such as:

```text
Polars
PyArrow
RapidFuzz
scikit-learn
LightGBM
Numba
```

where appropriate.

Do not load all 5M-row tables as Python object-heavy pandas DataFrames.

Do not try to put the whole dataset on the GPU.

---

# 26. Storage and resumability

Recommended structure:

```text
AMA/
├── student_resource/
│   └── dataset/
├── er_work/
│   ├── normalized/
│   ├── indexes/
│   ├── candidates/
│   ├── features/
│   ├── models/
│   ├── calibration/
│   ├── reports/
│   └── submissions/
└── business_entity_resolution/
    ├── src/
    ├── scripts/
    ├── configs/
    └── tests/
```

Each expensive stage should be resumable:

```text
profile
normalize
build_index
retrieve_train
retrieve_test
features_train
features_test
train
crossfit
calibrate
decode
validate
package
```

This is another strong engineering lesson from jvikramsrd.

---

# 27. Validation split strategy

Use Source-1-level splits.

Recommended:

```text
A — development
B — validation
C — final local validation
D — untouched audit
```

Avoid optimizing every decision on the same validation partition.

Before final submission, freeze the configuration and evaluate once on the audit split.

---

# 28. Metrics dashboard

Every experiment should report:

```text
Macro F0.5
Macro precision
Macro recall

Blocking recall
Average candidates / S1
P95 candidates / S1
P99 candidates / S1

Singleton accuracy
S2-specific performance
S3-specific performance

High-confidence false positives
High-confidence false negatives

Runtime
Peak RAM
Candidate count
```

---

# 29. Ablation plan

Run in this order:

```text
E0  exact rules
E1  + fuzzy features
E2  + multi-block retrieval
E3  + TF-IDF retrieval
E4  + transliteration
E5  + LightGBM
E6  + hard negatives
E7  + calibration
E8  + expected-F0.5 decoder
E9  + exclusivity
E10 + cross-fit collective features
E11 + stage-2 LightGBM
E12 + static embeddings
```

For every experiment record:

```text
ΔF0.5
Δblocking recall
Δcandidate count
Δruntime
ΔRAM
```

Only keep changes that improve held-out performance.

---

# 30. Implementation roadmap

## Phase 0 — Profile

Create:

```text
scripts/profile_dataset.py
```

Measure:

1. country distribution
2. missingness
3. duplicate names
4. duplicate addresses
5. name/address lengths
6. ground-truth match-count distribution
7. singleton percentage
8. S2/S3 match distribution
9. target exclusivity
10. exact-name coverage
11. candidate volume per block

---

## Phase 1 — Normalize

Create:

```text
src/normalize.py
```

Implement:

```text
raw
canonical
alnum
tokens
character ngrams
transliterated
abbreviation-aware
```

---

## Phase 2 — Retrieval

Create:

```text
src/blocking.py
```

Start with:

```text
exact name
token index
address index
numeric index
```

Then add:

```text
sparse TF-IDF
reverse retrieval
```

Measure blocking recall after each addition.

---

## Phase 3 — Feature engine

Create:

```text
src/features.py
```

Implement:

```text
name
address
numeric
country
retrieval provenance
candidate density
rank/margin
```

---

## Phase 4 — Baseline model

Train:

```text
LightGBM
```

with hard negatives and entity-level splits.

---

## Phase 5 — Calibration

Compare:

```text
none
sigmoid
isotonic
```

---

## Phase 6 — F0.5 decoder

Implement:

```text
src/decision.py
```

with:

```text
expected F0.5
Poisson-binomial DP
empty-set option
```

---

## Phase 7 — Exclusivity

Measure:

```text
max S1 owners per target
```

in ground truth.

If confirmed and beneficial:

```text
enable exclusivity
```

---

## Phase 8 — Collective stage

Generate OOF stage-1 probabilities and add:

```text
sibling support
source diversity
cluster support
candidate density
```

Train stage 2.

---

## Phase 9 — Optional embeddings

Only now benchmark a compliant multilingual static embedding model.

---

## Phase 10 — Full inference

Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

---

## Phase 11 — Official validation

Run:

```bash
python3 student_resource/utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir student_resource/dataset/test
```

No submission is complete until this succeeds.

---

# 31. Final recommended architecture

```text
S1 / S2 / S3
     |
     v
multiview normalization
     |
     +-----------------------------+
     |                             |
     v                             v
exact/token indexes          TF-IDF/char retrieval
     |                             |
     +-------------+---------------+
                   |
                   v
            candidate union
                   |
                   v
             pair features
                   |
                   v
              LightGBM #1
                   |
                OOF p
                   |
                   v
          collective/context
              features
                   |
                   v
              LightGBM #2
                   |
                   v
              calibration
                   |
                   v
         expected-F0.5 decoder
                   |
                   v
              exclusivity
                   |
                   v
          final match sets
                   |
          +--------+--------+
          |                 |
          v                 v
 matching_results      candidate_pairs
```

---

# 32. Why this is stronger than our original plan

Original:

```text
retrieve
→ classify
→ threshold
```

Revised:

```text
retrieve
→ classify
→ calibrate
→ use candidate context
→ enforce validated structural constraints
→ optimize the complete match set for F0.5
```

The final stage is particularly important because the competition is evaluated at the Source-1 entity level.

---

# 33. Core hypothesis

Our central hypothesis is:

> A high-recall multilingual multi-view retrieval system, followed by hard-negative-trained LightGBM scoring, calibrated probabilities, and per-entity expected-F0.5 decoding with validated target exclusivity, will provide a stronger performance/compute trade-off than brute-force fuzzy matching or a large-model-first approach.

This remains a hypothesis.

The implementation must prove it through held-out experiments.

---

# 34. Immediate next step

Do **not** start with the full model.

First build the profiler and answer:

```text
1. How many S1 entities are singletons?
2. What is the match-count distribution?
3. Is target exclusivity perfect?
4. What percentage of true matches have exact normalized names?
5. How many require address evidence?
6. How many require transliteration?
7. What candidate volume does each block produce?
8. What blocking recall do we get?
9. What candidate budget can our machine handle?
```

Those measurements determine the actual configuration.

---

# 35. Bottom line

The strongest evidence from the two public solutions is not that we should copy their code.

It is that **both independently converge on the same high-level strategy**:

```text
high-recall retrieval
+
structured ML scoring
+
metric-aware decision making
+
precision-oriented structural constraints
```

Our competitive advantage should come from implementing that architecture carefully, validating every component on our own data, and avoiding unnecessary compute.

The key principle is:

> **Retrieve the right candidates extremely reliably, reject hard false matches, calibrate the resulting probabilities, and choose the complete prediction set that best serves the actual macro-F0.5 objective.**
