"""Hard-negative mining experiment for the existing A2 stage-2 matcher.

This is deliberately isolated from the production pipeline. It reuses ``work/feat2``
(the existing OOF stage-1 features), trains the baseline stage-2 LightGBM, mines
high-scoring false candidate pairs from TRAIN_FOLDS only, adds those rows without
removing any original training rows, and evaluates both models with the normal
fold-3 tuning / fold-0 reporting protocol.

Usage after the normal A2 feature artifacts exist:
    python src/experiments/hard_negative_lab.py --approach 2

Outputs:
    work/reports/hard_negative_A2.json
    work/models/hard_negative_A2.txt
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import (S2_FEATS, _pred, choose_decision, exclusive_mask,
                            extra_features, load_xy, make_evaluator)  # noqa: E402
from ber.utils import log, save_json, stage  # noqa: E402


def fit_model(X_train, y_train, X_tune, y_tune, features, name, rounds=None):
    params = dict(config.LGB_PARAMS)
    with stage(f"hard_negative_train_{name}", {"rows": len(y_train), "features": len(features)}):
        train_set = lgb.Dataset(X_train, y_train, feature_name=features, free_raw_data=True)
        tune_set = lgb.Dataset(X_tune, y_tune, reference=train_set)
        model = lgb.train(
            params,
            train_set,
            rounds or config.LGB_ROUNDS,
            valid_sets=[tune_set],
            callbacks=[
                lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        (config.WORK_DIR / "models").mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.WORK_DIR / "models" / f"{name}.txt"))
    return model


def score_and_report(model, features, tag):
    """Tune on fold 3 and report fold 0; fold 0 is never used for mining or tuning."""
    meta_tune, X_tune, y_tune = load_xy("train", [config.TUNE_FOLD], features, sub="feat2")
    meta_eval, X_eval, y_eval = load_xy("train", [config.EVAL_FOLD], features, sub="feat2")
    p_tune = _pred(model, X_tune)
    p_eval = _pred(model, X_eval)
    all_gid = np.concatenate([meta_tune.get_column("gid").to_numpy(), meta_eval.get_column("gid").to_numpy()])
    all_p = np.concatenate([p_tune, p_eval])
    exclusion = exclusive_mask(all_gid, all_p)
    tune_exclusion = exclusion[:len(p_tune)]
    eval_exclusion = exclusion[len(p_tune):]
    choice = choose_decision(
        make_evaluator(meta_tune, y_tune, config.TUNE_FOLD),
        p_tune,
        meta_tune.get_column("gid").to_numpy(),
        tune_exclusion,
    )
    selected = max(choice, key=lambda key: choice[key]["tune_f05"])
    tau = choice[selected]["tau"]
    evaluator = make_evaluator(meta_eval, y_eval, config.EVAL_FOLD)
    keep = (p_eval >= tau) & (eval_exclusion if selected == "threshold+excl" else True)
    result = evaluator.report(keep)
    return {
        "tag": tag,
        "decision": {"variant": selected, "tau": tau, "tune_f05": choice[selected]["tune_f05"]},
        "eval": result,
        "eval_candidate_pairs": int(len(y_eval)),
    }


def mine_rows(model, features, threshold, max_per_s1, max_rows):
    """Mine only false candidates from folds 1/2; folds 0 and 3 are never read here."""
    meta, X, y = load_xy("train", list(config.TRAIN_FOLDS), features, sub="feat2")
    p = _pred(model, X)
    hard = meta.with_columns(y=pl.Series(y), p=pl.Series(p)).filter(
        (pl.col("y") == 0) & (pl.col("p") >= threshold)
    )
    hard = hard.sort(["rid", "p"], descending=[False, True]).with_columns(
        rank=pl.col("p").rank("ordinal", descending=True).over("rid")
    ).filter(pl.col("rank") <= max_per_s1).drop("rank")
    if hard.height > max_rows:
        hard = hard.sort("p", descending=True).head(max_rows)
    return hard, len(y), int((y == 0).sum())


def run(mine_threshold, max_per_s1, max_rows):
    features = S2_FEATS + extra_features()
    _, X_tune, y_tune = load_xy("train", [config.TUNE_FOLD], features, sub="feat2")
    _, X_train, y_train = load_xy("train", list(config.TRAIN_FOLDS), features, sub="feat2")
    baseline = fit_model(X_train, y_train, X_tune, y_tune, features, "hard_negative_baseline")
    baseline_report = score_and_report(baseline, features, "baseline")

    hard, train_rows, train_negatives = mine_rows(
        baseline, features, mine_threshold, max_per_s1, max_rows
    )
    if hard.height:
        hard_meta, X_hard, y_hard = load_xy("train", list(config.TRAIN_FOLDS), features, sub="feat2")
        hard_keys = hard.select("rid", "gid")
        selected = hard_meta.with_row_index("_idx").join(hard_keys, on=["rid", "gid"], how="inner")
        X_mined = X_hard[selected.get_column("_idx").to_numpy()]
        y_mined = y_hard[selected.get_column("_idx").to_numpy()]
        X_aug = np.concatenate([X_train, X_mined])
        y_aug = np.concatenate([y_train, y_mined])
    else:
        X_aug, y_aug = X_train, y_train

    mined_model = fit_model(X_aug, y_aug, X_tune, y_tune, features, "hard_negative_A2")
    mined_report = score_and_report(mined_model, features, "hard_negative")
    report = {
        "approach": 2,
        "features": len(features),
        "mining": {
            "source_folds": list(config.TRAIN_FOLDS),
            "excluded_folds": [config.TUNE_FOLD, config.EVAL_FOLD],
            "prediction_threshold": mine_threshold,
            "max_per_s1": max_per_s1,
            "max_rows": max_rows,
            "candidate_rows_scored": train_rows,
            "original_negative_rows": train_negatives,
            "mined_negative_rows": hard.height,
            "augmented_training_rows": len(y_aug),
        },
        "baseline": baseline_report,
        "hard_negative": mined_report,
    }
    save_json(report, config.WORK_DIR / "reports" / "hard_negative_A2.json")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", type=int, choices=[2], default=2)
    parser.add_argument("--mine-threshold", type=float, default=0.50)
    parser.add_argument("--max-per-s1", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=500_000)
    args = parser.parse_args()
    config.ensure_dirs()
    run(args.mine_threshold, args.max_per_s1, args.max_rows)


if __name__ == "__main__":
    main()
