# Amazon ML Challenge 2026 — Business Entity Resolution

Given business records from 3 independent sources (noisy, no shared identifiers), determine which records refer to the same real-world business.

- **Source 1** is the deduplicated reference source.
- For each Source 1 entity, find all matching records in **Source 2** and **Source 3**.
- A Source 1 entity may match zero, one, or many S2/S3 records.

## Data

All files are **tab-separated** (`pd.read_csv(path, sep="\t")`) — addresses and ID lists contain commas.

Source files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) columns:

| Column | Notes |
|---|---|
| `entity_id` | Prefix gives the source: `S1-`, `S2-`, `S3-` |
| `business_name` | Abbreviations, legal suffixes, typos, transliterations |
| `business_address` | Partial addresses, format variations, missing components, landmark references |
| `country` | Train: US, India. **Test adds France.** Treat as an open set — do not hard-code, filter, or one-hot to {US, India}. Every test entity, France included, must appear in the submission. |

`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs; empty for no match).

### Noise patterns
- **Names:** Corp/Corporation, Pvt/Private, Ltd/Limited, legal suffix inconsistencies, DBA/trade names, `&` vs `and`, word-order transpositions, typos.
- **Addresses:** Rd/Road, St/Street, transliteration variants, missing components (no PIN, no state), landmark references ("Near SBI ATM"), municipal numbering formats, component reordering.

### Files
- `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source{1,2,3}.tsv` — generate matches for every entity in `test_source1.tsv`. No test ground truth; hold out a validation split from train.

## Output (both in `output/`, tab-separated)

### `matching_results.tsv` — the only scored file (uploaded to the Portal)
Columns: `source1_entity_id`, `matched_entity_ids`

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	
S1-00003	S3-00004
```

- Exactly one row per test Source 1 entity.
- Empty list for singletons.
- No duplicate IDs within a list; no duplicate `source1_entity_id` rows.
- Only S2/S3 IDs that exist in the test set.

### `candidate_pairs.tsv` — blocking output (not scored on the leaderboard, but ranked)
Columns: `source1_entity_id`, `candidate_entity_ids`

> **Update (2026-09-27):** `candidate_pairs.tsv` and the code that produces it are reviewed for the final ranking. **A smaller candidate set per Source 1 entity is ranked higher**, beyond the public/private leaderboard. Blocking must scale (billions of records). Full verbatim text: `SUBMISSION_REQUIREMENTS.md`.

- The **final** candidate set the matching model actually runs inference over (last filtering stage, not an early raw block).
- Same rules as above. Every matched ID must also appear as a candidate.
- Used by organisers to analyse recall ceiling / reduction ratio.

### Validation
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Run from `student_resource/`. Prints PASS (exit 0) or numbered issues (exit 1). Does not compute score.

## Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # final matches (same as leaderboard upload)
│   └── candidate_pairs.tsv        # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                   # all source code
│       ├── README.md              # exact end-to-end run instructions (data → blocking → matching → output)
│       └── requirements.txt       # pinned dependencies
└── Documentation_template.md      # filled-in methodology write-up (.md or .pdf)
```

Methodology document must cover: methodology, candidate generation/blocking strategy, model architecture and feature engineering, other relevant details. No page limit.

## Constraints
1. Exact output format; failed validation is not evaluated (look for SCORED status).
2. Matched IDs only from S2/S3; no self-matches to S1; no non-existent IDs.
3. Every Source 1 entity must appear.
4. No duplicate IDs in a list; no duplicate `source1_entity_id` rows.
5. **Final model: MIT/Apache 2.0 license, ≤ 8B parameters.**

## Evaluation: F_0.5, macro-averaged per Source 1 entity

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

- Computed per Source 1 entity, then averaged. Singletons included.
- Singleton: predict empty → 1.0; predict anything → 0.0.
- Example: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] → P = 2/3, R = 1 → F_0.5 = 0.714.

## Leaderboards
- **Public:** subset of test during the challenge.
- **Private:** remaining test after the challenge. **Final ranking uses private.**
- Submit predictions for the full test set in both cases.

## Fair play — STRICTLY PROHIBITED
No external databases, APIs, or services to look up businesses or resolve entities: no commercial ER APIs, government registries, geocoding APIs, or internet data augmentation. Evidence → immediate disqualification. Use only the provided training data.

## Tips (from organisers)
- Strong blocking sets the recall ceiling.
- String similarity features: Jaccard, Levenshtein, TF-IDF cosine.
- Country-specific address patterns.
- F_0.5 favours precision.
- Don't neglect singletons — correct "no match" is worth 1.0.
- Validate output format before submitting.
