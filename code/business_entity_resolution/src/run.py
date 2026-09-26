"""Pipeline entry point.

Stages (each is resumable: finished outputs carry a '.done' marker and are skipped):
  prepare  --split train|test     TSV -> normalized Parquet with integer ids
  block    --split train|test     multi-view TF-IDF + reverse retrieval -> candidate pairs
  features --split train|test     pairwise (A1) + context (A2) features
  evaluate --approach 1|2         fit on train folds, tune threshold on fold 3, report fold 0
  predict  --approach 1|2         score test, write output/*.tsv, run official validator
  all      --approach 1|2         everything above in order

Example: python src/run.py all --approach 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "block", "features", "evaluate", "predict", "all"])
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--approach", type=int, choices=[1, 2], default=None)
    a = ap.parse_args()
    config.ensure_dirs()
    splits = [a.split] if a.split else ["train", "test"]

    if a.stage in ("prepare", "all"):
        from ber.io import prepare
        for s in splits:
            prepare(s)
    if a.stage in ("block", "all"):
        from ber.blocking import block
        for s in splits:
            block(s)
    if a.stage in ("features", "all"):
        from ber.features import build_features
        for s in splits:
            build_features(s)
    if a.stage in ("evaluate", "all"):
        from ber.approaches import evaluate
        for ap_id in ([a.approach] if a.approach else [1, 2]):
            evaluate(ap_id)
    if a.stage in ("predict", "all"):
        from ber.approaches import predict
        predict(a.approach or 2)


if __name__ == "__main__":
    main()
