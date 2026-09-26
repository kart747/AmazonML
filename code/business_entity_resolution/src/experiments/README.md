# Experiments (team runbook)

Independent, runnable experiments on top of the main pipeline (`src/run.py`). Each one
reads the shared artifacts in `BER_WORK_DIR` and writes its own tagged outputs, so
several people or machines can run different experiments without overwriting each other.

## Prerequisites (per machine)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export BER_DATA_DIR=/path/to/student_resource/dataset   # train/ and test/
export BER_WORK_DIR=/path/to/work                      # shared or copied artifacts
export BER_VALIDATOR=/path/to/student_resource/utils/validate_submission.py
python tests/test_smoke.py                              # ~2 min, synthetic data, must print SMOKE TEST PASSED
```
All of these are CPU-only. A GPU is not needed or used.

**Sharing artifacts instead of recomputing them.** Blocking takes about 2 h per split on 16 threads.
Copy `work/prep/`, `work/block/` and `work/feat/` from the machine that produced them
(train about 6 GB; test similar). Run tags keep models and reports separate:
`BER_TAG=_something` suffixes every model, report and stage-2 directory.

| Experiment | Needs | Command | Output | Est. cost (16 thr) |
|---|---|---|---|---|
| Main A1 / A2 | work/feat/train | `python src/run.py evaluate --approach 1` (or `2`) | reports/eval_A1.json, scores_A1.parquet | A1 ~20 min, A2 ~60 min; 8–12 GB RAM |
| **Decoder lab** | reports/scores_A<n>.parquet | `python src/experiments/decoder_lab.py --approach 2` | reports/decoders_A2.json + table | ~5 min, ~4 GB |
| **P-dense** (test-like density) | A<n> models + work/feat/train + work/block/train | `python src/experiments/density.py --approach 2` | reports/density_A2.json | ~20 min, ~8 GB |
| **LOCO** (France proxy) | work/feat/train | `python src/experiments/loco.py --features context` | reports/loco_context.json | ~30 min, ~8 GB |
| **Features v2** | work/feat/{train,test} + prep | `python src/experiments/features_v2.py` then `BER_EXTRA=featx BER_TAG=_fx python src/run.py evaluate --approach 2` | work/featx/, reports/eval_A2_fx.json | features ~30 min; eval as main |

## What each answers
- **decoder_lab.py**: which decision rule maximizes held-out macro F0.5 on the same scores. Compares threshold, threshold + target exclusivity, relative-to-best, per-source thresholds, and calibrated expected-F0.5 prefix decoding (isotonic calibration fitted on the tune fold). Parameters are chosen on fold 3 and reported on fold 0, with a paired bootstrap SE against threshold+excl. Keep a rule only if Δ > 2 SE.
- **density.py**: the test has about 23% more targets per S1 than train. It drops 19% of S1 from the graph, recomputes all pool-dependent context features, rescores with the trained models and reports (a) the F0.5 loss under test-like density with the normal τ and (b) the τ re-selected under density. Use (b) for the test submission if it differs.
- **loco.py**: trains on US and evaluates on India, and the reverse, with τ chosen on the source country. Measures the unseen-country gap and how much τ should rise for unseen labels (France = 15% of test).
- **features_v2.py**: adds a learned cross-script dictionary (from train-fold pairs only), consonant skeletons, rarest-unmatched-token IDF, learned core-token overlap and number conflicts. Ablate with `BER_EXTRA=featx` against the same run without it.

## Parallel-track modules (owned by the second session; smoke test: `python tests/test_parallel_modules.py`)

| Experiment | Needs | Command | Output | Est. cost (16 thr) |
|---|---|---|---|---|
| **Cross-script retrieval (xs)** | work/prep + work/block/<split>/cand_* | `python src/experiments/xscript.py build --split train` then `xscript.py measure` | block/train/xs_<country>.parquet, reports/xscript_recall.json | ≈5 min per country with Indic names; peak ≈4 GB |
| **xs end to end (one command)** | base run finished (work/block/{train,test}, reports/eval_A2.json) | `bash src/experiments/xs_pipeline.sh [0.10]`: build → measure → gate (stops unless ≥10% of base misses are recovered) → merge → features → evaluate A2 → predict, all in `work_xs/` | base vs xs comparison printed; `work_xs/output/*.tsv` | ≈2.5–3 h, ≈10 GB peak |
| xs into the pipeline (only if `measure` shows a gain) | both splits built | `xscript.py merge --split train --dst ../work_xs` (and `--split test`), then `BER_WORK_DIR=../work_xs python src/run.py features` / `evaluate` / `predict` | an isolated work dir; base artifacts untouched | merge ≈3 min/split; then a full features + evaluate run |
| **Dense multilingual view (dn, GPU)** | work/prep + work/block/<split>/cand_*; a venv with torch + transformers (`requirements-dense.txt`) | `.venv-dense/bin/python src/experiments/dense_view.py build --split train` then `xscript.py measure --views xs,dn` | work/dense/<split>/*.npy, block/<split>/dn_<country>.parquet, reports/supp_recall_xs_dn.json | encode ≈1.8M texts/split (India S1 + non-Latin targets); peak RAM ≈2 GB, VRAM <2 GB |
| dn/xs into the pipeline + `dn_cos` feature | both splits built | `xscript.py merge --split train --dst ../work_xd --views xs,dn --cap dn=8` (and test), `BER_WORK_DIR=../work_xd run.py features`, `BER_WORK_DIR=../work_xd .venv-dense/bin/python src/experiments/dense_view.py features --split train` (and test), then `BER_EXTRA=featdn BER_TAG=_dn run.py evaluate --approach 2` | isolated work_xd/ | a full features + evaluate run |
| **Re-decode / test diagnostics** | reports/test_A<n>_pairs.parquet (from `run.py predict`) | `python src/experiments/redecode.py diagnose --approach 2` | reports/test_diagnostics_A2.json + table | ~1 min, ~3 GB |
| Variant submission (no re-scoring) | same | `redecode.py write --approach 2 --tau 0.8 --country-tau France=0.85 --out ../output_variants/<name>` | complete submission dir + decision.json, validator run | ~1 min |

- **xscript.py**: 53% of India blocking misses have an Indic-script target name. anyascii spells "लक्ष्मी" as "lksmi", so every name view fails. This module maps native tokens through the learned dictionary (the same `features_v2.learn_dictionary`, training folds only) and adds a reverse target→S1 view for non-Latin-script targets. It is a no-op for Latin-script countries (US, France). `merge` refits the base vectorizers and asserts that they reproduce the stored cosines before adding new pairs. Pair it with `BER_EXTRA=featx` so the matcher also sees the mapped-name similarity.
- **dense_view.py**: the learned alternative to xs. It uses multilingual-e5-small (MIT, 118M) embeddings of raw name + address, with reverse target→S1 top-k on the GPU, for non-Latin-script targets against all S1 of the country. The external evidence (Karthikatstuffmama, 3% dev slice) is recall 0.974 → 0.993 and holdout F0.5 +0.0065 (India +0.0128). `features` adds `dn_cos` for every candidate pair with an embedded target (NaN otherwise). On synthetic data it recovered precision after adding candidates (A1 0.92639 → 0.92945).
- **redecode.py**: France has no labels, so its decision can only be checked label-free. `diagnose` compares predicted and expected matches per S1 by country (test vs the validation eval fold). `write` produces leaderboard A/B variants, such as a per-country τ, from saved scores in seconds. It never overwrites `output/` (`--out` is required).

## Suggested allocation across the team machines
Only the RAM and CPU of these machines matter (no GPU stages).
1. The machine that already has `work/feat/train`: main A1/A2 evaluation, then decoder lab.
2. Second machine with a copy of `work/{prep,block,feat}/train`: LOCO, then features v2 + its ablation.
3. Third machine: P-dense after the A2 models exist (copy `work/models/`).
