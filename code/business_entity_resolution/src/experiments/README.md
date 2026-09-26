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

## Suggested allocation across the team machines
Only the RAM and CPU of these machines matter (no GPU stages).
1. The machine that already has `work/feat/train`: main A1/A2 evaluation, then decoder lab.
2. Second machine with a copy of `work/{prep,block,feat}/train`: LOCO, then features v2 + its ablation.
3. Third machine: P-dense after the A2 models exist (copy `work/models/`).
