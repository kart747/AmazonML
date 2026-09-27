"""Hard-negative mining experiment for the existing A2 stage-2 matcher.

This is deliberately isolated from the production pipeline. It reuses ``work/feat2``
(the existing OOF stage-1 features), trains the baseline stage-2 LightGBM, mines
high-scoring false candidate pairs from TRAIN_FOLDS only (scored out-of-fold, so no
row is ever scored by a model that trained on it), adds those rows without removing
any original training rows, and evaluates both models with the normal fold-3 tuning /
fold-0 reporting protocol.

Usage after the normal A2 feature artifacts exist:
    python src/experiments/hard_negative_lab.py --approach 2

Outputs:
    work/reports/hard_negative_A2.json
    work/models/hard_negative_A2.txt

Requires len(config.TRAIN_FOLDS) >= 2 so each fold can be scored by a model trained
on the *other* train folds (true out-of-fold mining). With a single train fold this
falls back to in-sample scoring and prints a warning.
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
from ber.utils import save_json, stage  # noqa: E402


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


def _cap_symmetric(scored: pl.DataFrame, max_per_group: int) -> pl.DataFrame:
    """Keep top-`max_per_group` negatives per S1 (rid) UNION top-`max_per_group` per
    target (gid), so both 'strong competitor for the same S1' and 'strong competitor
    for the same target' failure modes are represented."""
    by_s1 = (scored
        .with_columns(rank=pl.col("p").rank("ordinal", descending=True).over("rid"))
        .filter(pl.col("rank") <= max_per_group))
    by_target = (scored
        .with_columns(rank=pl.col("p").rank("ordinal", descending=True).over("gid"))
        .filter(pl.col("rank") <= max_per_group))
    return pl.concat([by_s1, by_target]).unique(subset=["rid", "gid"]).drop("rank")


def mine_rows_oof(features, X_tune, y_tune, tau, max_per_group, max_rows):
    """Score every TRAIN_FOLDS row out-of-fold: for each train fold, predict it with a
    model trained on the *other* train folds only, so mining never scores a row with a
    model that saw it during training. Folds TUNE_FOLD and EVAL_FOLD are never read
    here. Returns (meta, X, y) for all TRAIN_FOLDS rows plus the mined hard-negative
    subset, so callers don't need to reload the data.
    """
    meta, X, y = load_xy("train", list(config.TRAIN_FOLDS), features, sub="feat2")
    fold_col = meta.get_column("fold").to_numpy()
    p_oof = np.zeros(len(y), dtype=float)

    train_folds = list(config.TRAIN_FOLDS)
    if len(train_folds) < 2:
        print("WARNING: only one TRAIN_FOLD configured — cannot mine out-of-fold; "
              "falling back to in-sample scoring (mined negatives may be optimistic).")
        submodel = fit_model(X, y, X_tune, y_tune, features, "hard_negative_oof_fallback")
        p_oof = _pred(submodel, X)
    else:
        for held_out in train_folds:
            fit_folds = [f for f in train_folds if f != held_out]
            fit_mask = np.isin(fold_col, fit_folds)
            held_mask = fold_col == held_out
            # Use the configured tune fold for early stopping. It is not mined and
            # remains separate from the held-out train fold being scored.
            sub_model = fit_model(
                X[fit_mask], y[fit_mask], X_tune, y_tune, features,
                f"hard_negative_oof_fold{held_out}",
            )
            p_oof[held_mask] = _pred(sub_model, X[held_mask])

    scored = meta.with_columns(y=pl.Series(y), p=pl.Series(p_oof))
    hard_pool = scored.filter((pl.col("y") == 0) & (pl.col("p") >= tau))
    hard = _cap_symmetric(hard_pool, max_per_group)
    if hard.height > max_rows:
        hard = hard.sort("p", descending=True).head(max_rows)
    return meta, X, y, hard, len(y), int((y == 0).sum())


def run(max_per_group, max_rows, mine_threshold_override):
    features = S2_FEATS + extra_features()
    _, X_tune, y_tune = load_xy("train", [config.TUNE_FOLD], features, sub="feat2")
    _, X_train, y_train = load_xy("train", list(config.TRAIN_FOLDS), features, sub="feat2")
    baseline = fit_model(X_train, y_train, X_tune, y_tune, features, "hard_negative_baseline")
    baseline_report = score_and_report(baseline, features, "baseline")

    # Hard negative = "would have been a false positive at the production threshold",
    # not an arbitrary fixed cutoff.
    tau = mine_threshold_override if mine_threshold_override is not None \
        else baseline_report["decision"]["tau"]

    meta, X_all, y_all, hard, train_rows, train_negatives = mine_rows_oof(
        features, X_tune, y_tune, tau, max_per_group, max_rows
    )

    if hard.height:
        indexed = meta.with_row_index("_idx")
        selected = indexed.join(hard.select("rid", "gid"), on=["rid", "gid"], how="inner")
        idx = selected.get_column("_idx").to_numpy()
        X_aug = np.concatenate([X_train, X_all[idx]])
        y_aug = np.concatenate([y_train, y_all[idx]])
    else:
        X_aug, y_aug = X_train, y_train

    mined_model = fit_model(X_aug, y_aug, X_tune, y_tune, features, "hard_negative_A2")
    mined_report = score_and_report(mined_model, features, "hard_negative")
    report = {
        "approach": 2,
        "features": len(features),
        "mining": {
            "method": "out-of-fold within TRAIN_FOLDS" if len(config.TRAIN_FOLDS) >= 2
                      else "in-sample fallback (single train fold)",
            "source_folds": list(config.TRAIN_FOLDS),
            "excluded_folds": [config.TUNE_FOLD, config.EVAL_FOLD],
            "prediction_threshold": tau,
            "threshold_source": "override" if mine_threshold_override is not None
                                 else "baseline_tuned_tau",
            "max_per_group": max_per_group,
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
    parser.add_argument(
        "--mine-threshold", type=float, default=None,
        help="Override the mining threshold. Default: use the baseline model's own "
             "tuned decision tau, so 'hard negative' means 'would have been a false "
             "positive in production'.",
    )
    parser.add_argument("--max-per-group", type=int, default=5,
                        help="Cap on mined negatives per S1 AND per target (union of both).")
    parser.add_argument("--max-rows", type=int, default=500_000)
    args = parser.parse_args()
    config.ensure_dirs()
    run(args.max_per_group, args.max_rows, args.mine_threshold)


if __name__ == "__main__":
    main()