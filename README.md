# Business Entity Resolution at Scale

Solution for the **Amazon ML Challenge 2026**: for every business record in Source 1, find the
matching records in two other noisy sources, across the US, India and France, at 10M+ record scale.

## Results

| Metric | Value |
| --- | --- |
| Local macro F0.5 (held-out fold 0) | 0.98748 |
| Public leaderboard F0.5 (final upload) | 0.978 |
| Test candidates scored | 85.4M pairs, pruned to 9.6M for stage 2 |

All numbers are from this repository's own runs; the full history is in
[`EXPERIMENTS.md`](EXPERIMENTS.md) and [`SUBMISSIONS.md`](SUBMISSIONS.md).

## Approach

1. **Normalisation**: multi-view text cleaning with transliteration, noise repair and an abbreviation lexicon.
2. **Blocking**: per-country TF-IDF retrieval (name character 3-grams, address words, combined view)
   with reverse target-to-source retrieval, plus a multilingual-e5-small view for non-Latin India records.
3. **Stage 1 matcher**: LightGBM over pairwise similarity features, trained out-of-fold.
4. **Stage 2 matcher**: a relational LightGBM model over 92 pair and candidate-graph context features.
5. **Decision**: F0.5-tuned threshold with target exclusivity, an affiliate-token veto learned on the
   training data, and pseudo-labelling for France, which has no labels.

## Repository layout

| Path | Contents |
| --- | --- |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | The pipeline, CLI and tests; setup and reproduction steps are in its README |
| [`experiment_chains/`](experiment_chains/) | Shell scripts that ran each experiment chain |
| [`submission/`](submission/) | Metadata, checksums and validator output for each leaderboard package |
| [`Documentation.md`](Documentation.md) | Solution write-up submitted with the final upload |
| [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) | Validation protocol and experiment design |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Log of every experiment and its outcome |
| [`SUBMISSIONS.md`](SUBMISSIONS.md) | Log of every leaderboard upload |

## Running it

The challenge dataset is not included. See
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for setup,
the end-to-end commands and a two-minute smoke test on synthetic data.

## Stack

Python 3.11, scikit-learn, LightGBM, Polars, RapidFuzz, sparse-dot-topn, PyTorch (optional, for the e5 blocking view).
