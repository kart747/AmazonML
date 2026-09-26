# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business, find its matching Source-2/Source-3 records.
Pipeline: normalization → multi-view TF-IDF retrieval with reverse retrieval (blocking) →
pair features → LightGBM matcher → F0.5-tuned decision with target exclusivity.
CPU only. No external data, APIs or pretrained models.

## Setup
Python 3.11.
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Paths default to the challenge layout next to this repo. Override them with environment variables:
`BER_DATA_DIR` (folder with `train/` and `test/`), `BER_WORK_DIR` (artifacts, ~30 GB),
`BER_OUT_DIR` (submission files), `BER_VALIDATOR` (official `validate_submission.py`),
`BER_THREADS` (CPU threads).

## Reproduce end to end
```bash
python src/run.py prepare              # TSV -> normalized Parquet (≈3 min)
python src/run.py block                # candidate generation, train + test (≈4 h on 16 threads)
python src/run.py features             # pair + context features (train + test)
python src/run.py evaluate --approach 2   # fit, tune on fold 3, report fold 0
python src/run.py predict  --approach 2   # test scoring -> output/*.tsv + official validator
```
Every stage is resumable. Finished outputs carry a `.done` marker and are skipped on rerun.
Delete a marker to force a stage to rerun.

Outputs: `output/matching_results.tsv` (scored file) and `output/candidate_pairs.tsv`
(the exact candidate set the matcher scores; every match is a candidate).

## Layout
- `src/run.py`: CLI
- `src/ber/config.py`: paths, validation protocol, blocking and model parameters
- `src/ber/io.py`: strict TSV ingestion (`QUOTE_NONE`), integer ids, ground-truth pairs
- `src/ber/normalize.py`: multi-view text normalization (transliteration, noise repair, abbreviation lexicon)
- `src/ber/blocking.py`: per-country TF-IDF views (name char 3-gram, address words, combo) + reverse target→S1 retrieval with cache-blocked exact top-K
- `src/ber/features.py`: pairwise similarity features and candidate-graph context features
- `src/ber/approaches.py`: the matchers (A1 pairwise, A2 relational two-stage), decision, test prediction, submission writer
- `src/ber/metric.py`: exact macro F0.5 evaluator and diagnostics

## Validation protocol
S1-grouped folds `int(id) % 10`: train {1,2}, tune {3}, report {0}, 4–9 untouched.
Retrieval always runs against the full target pool. See `../../EXPERIMENTS.md` for results.
