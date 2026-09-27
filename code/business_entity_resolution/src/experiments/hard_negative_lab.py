"""Hard-negative mining experiment for the existing A2 stage-2 matcher.

This is deliberately isolated from the production pipeline. It reuses ``work/feat2{TAG}``
(the existing OOF stage-1 features), takes the baseline stage-2 LightGBM, mines
high-scoring false candidate pairs from TRAIN_FOLDS only (scored out-of-fold, so no
row is ever scored by a model that trained on it), upweights those rows without removing
any original training rows, and evaluates both models with the normal fold-3 tuning /
fold-0 reporting protocol.

Usage after the normal A2 feature artifacts exist (same BER_TAG / BER_EXTRA as that run):
    BER_TAG=_xd python src/experiments/hard_negative_lab.py --approach 2

Outputs:
    work/reports/hard_negative_A2{TAG}.json
    work/models/hard_negative_A2{TAG}.txt

Requires len(config.TRAIN_FOLDS) >= 2 so each fold can be scored by a model trained
on the *other* train folds (true out-of-fold mining). With a single train fold this
falls back to in-sample scoring and prints a warning.

Fixes on top of the original branch (origin/hard-negative-mining, Yashwin69):
  * reads ``feat2{TAG}`` like ``approaches.evaluate``; the hard-coded ``feat2`` silently
    used the untagged (pre-xd) stage-2 parts under BER_TAG=_xd, and every tag wrote the
    same model/report names.
  * each fold is loaded once (train folds were loaded twice, tune three times, plus a full
    concatenated copy for the augmented set: ~3x the train matrix at peak).
  * mined rows get a sample weight (``--weight``, default 2.0) instead of being appended a
    second time: same expected loss as the duplicate row, no matrix copy.
  * the baseline is the already trained ``A2{TAG}_stage2`` model when it exists (same
    folds, features and parameters as this script's baseline fit); --retrain-baseline refits.
  * ``--mine-rule decision`` (default) mines the negatives the baseline's selected decision
    would actually accept: with threshold+excl only pairs that also win their target.
    ``--mine-rule threshold`` is the original p >= tau rule.
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
from ber.approaches import (S2_FEATS, TAG, _pred, choose_decision, decide, exclusive_mask,
                            extra_features, load_xy, make_evaluator)  # noqa: E402
from ber.utils import save_json, stage  # noqa: E402

SUB = f"feat2{TAG}"


def fit_model(X_train, y_train, X_tune, y_tune, features, name, rounds=None, weight=None):
    params = dict(config.LGB_PARAMS)
    with stage(f"hard_negative_train_{name}", {"rows": len(y_train), "features": len(features)}):
        train_set = lgb.Dataset(X_train, y_train, weight=weight, feature_name=features, free_raw_data=True)
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


def baseline_model(features, X_train, y_train, X_tune, y_tune, retrain):
    """The production A2 stage-2 model if present and built on the same features, else a fit."""
    path = config.WORK_DIR / "models" / f"A2{TAG}_stage2.txt"
    if not retrain and path.exists():
        model = lgb.Booster(model_file=str(path))
        if model.feature_name() == features:
            print(f"baseline: reusing {path.name}")
            return model
        print(f"baseline: {path.name} has different features, refitting")
    return fit_model(X_train, y_train, X_tune, y_tune, features, f"hard_negative_baseline{TAG}")


def score_and_report(model, tune, evalf, tag):
    """Tune on fold 3 and report fold 0; fold 0 is never used for mining or tuning."""
    (meta_tune, X_tune, y_tune), (meta_eval, X_eval, y_eval) = tune, evalf
    p_tune = _pred(model, X_tune)
    p_eval = _pred(model, X_eval)
    g_tune, g_eval = meta_tune.get_column("gid").to_numpy(), meta_eval.get_column("gid").to_numpy()
    exclusion = exclusive_mask(np.concatenate([g_tune, g_eval]), np.concatenate([p_tune, p_eval]))
    tune_exclusion = exclusion[:len(p_tune)]
    eval_exclusion = exclusion[len(p_tune):]
    choice = choose_decision(make_evaluator(meta_tune, y_tune, config.TUNE_FOLD), p_tune, g_tune, tune_exclusion)
    selected = max(choice, key=lambda key: choice[key]["tune_f05"])
    tau = choice[selected]["tau"]
    evaluator = make_evaluator(meta_eval, y_eval, config.EVAL_FOLD)
    result = evaluator.report(decide(p_eval, g_eval, selected, tau, eval_exclusion))
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


def mine_rows_oof(features, meta, X, y, X_tune, y_tune, tau, variant, max_per_group, max_rows):
    """Score every TRAIN_FOLDS row out-of-fold: for each train fold, predict it with a
    model trained on the *other* train folds only, so mining never scores a row with a
    model that saw it during training. Folds TUNE_FOLD and EVAL_FOLD are never read
    here. Returns the row indices of the mined hard negatives and their table.
    """
    fold_col = meta.get_column("fold").to_numpy()
    p_oof = np.zeros(len(y), dtype=float)

    train_folds = list(config.TRAIN_FOLDS)
    if len(train_folds) < 2:
        print("WARNING: only one TRAIN_FOLD configured — cannot mine out-of-fold; "
              "falling back to in-sample scoring (mined negatives may be optimistic).")
        submodel = fit_model(X, y, X_tune, y_tune, features, f"hard_negative_oof_fallback{TAG}")
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
                f"hard_negative_oof_fold{held_out}{TAG}",
            )
            p_oof[held_mask] = _pred(sub_model, X[held_mask])

    # a negative is "hard" if the chosen decision would accept it (see --mine-rule)
    accept = decide(p_oof, meta.get_column("gid").to_numpy(), variant, tau)
    scored = meta.with_row_index("_idx").with_columns(y=pl.Series(y), p=pl.Series(p_oof))
    hard_pool = scored.filter((pl.col("y") == 0) & pl.Series(accept))
    hard = _cap_symmetric(hard_pool, max_per_group)
    if hard.height > max_rows:
        hard = hard.sort("p", descending=True).head(max_rows)
    return hard.get_column("_idx").to_numpy(), hard


def run(max_per_group, max_rows, mine_threshold_override, mine_rule, weight, retrain):
    features = S2_FEATS + extra_features()
    tune = load_xy("train", [config.TUNE_FOLD], features, sub=SUB)
    evalf = load_xy("train", [config.EVAL_FOLD], features, sub=SUB)
    meta, X_train, y_train = load_xy("train", list(config.TRAIN_FOLDS), features, sub=SUB)
    _, X_tune, y_tune = tune
    baseline = baseline_model(features, X_train, y_train, X_tune, y_tune, retrain)
    baseline_report = score_and_report(baseline, tune, evalf, "baseline")

    # Hard negative = "would have been a false positive at the production threshold",
    # not an arbitrary fixed cutoff.
    tau = mine_threshold_override if mine_threshold_override is not None \
        else baseline_report["decision"]["tau"]
    variant = baseline_report["decision"]["variant"] if mine_rule == "decision" else "threshold"

    idx, hard = mine_rows_oof(
        features, meta, X_train, y_train, X_tune, y_tune, tau, variant, max_per_group, max_rows
    )
    w = np.ones(len(y_train), dtype=np.float32)
    w[idx] = weight

    mined_model = fit_model(X_train, y_train, X_tune, y_tune, features, f"hard_negative_A2{TAG}", weight=w)
    mined_report = score_and_report(mined_model, tune, evalf, "hard_negative")
    report = {
        "approach": 2,
        "tag": TAG,
        "features": len(features),
        "mining": {
            "method": "out-of-fold within TRAIN_FOLDS" if len(config.TRAIN_FOLDS) >= 2
                      else "in-sample fallback (single train fold)",
            "source_folds": list(config.TRAIN_FOLDS),
            "excluded_folds": [config.TUNE_FOLD, config.EVAL_FOLD],
            "prediction_threshold": tau,
            "threshold_source": "override" if mine_threshold_override is not None
                                 else "baseline_tuned_tau",
            "mine_rule": variant,
            "max_per_group": max_per_group,
            "max_rows": max_rows,
            "candidate_rows_scored": len(y_train),
            "original_negative_rows": int((y_train == 0).sum()),
            "mined_negative_rows": hard.height,
            "mined_row_weight": weight,
        },
        "baseline": baseline_report,
        "hard_negative": mined_report,
    }
    save_json(report, config.WORK_DIR / "reports" / f"hard_negative_A2{TAG}.json")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", type=int, choices=[2], default=2)
    parser.add_argument(
        "--mine-threshold", type=float, default=None,
        help="Override the mining threshold. Default: use the baseline model's own "
             "tuned decision tau, so 'hard negative' means 'would have been a false "
             "positive in production'.",
    )
    parser.add_argument("--mine-rule", choices=["decision", "threshold"], default="decision",
                        help="decision: the baseline's selected variant (exclusivity included); "
                             "threshold: p >= tau only (the original rule).")
    parser.add_argument("--weight", type=float, default=2.0,
                        help="Sample weight of mined negatives; 2.0 = the original duplicate row.")
    parser.add_argument("--retrain-baseline", action="store_true")
    parser.add_argument("--max-per-group", type=int, default=5,
                        help="Cap on mined negatives per S1 AND per target (union of both).")
    parser.add_argument("--max-rows", type=int, default=500_000)
    args = parser.parse_args()
    config.ensure_dirs()
    run(args.max_per_group, args.max_rows, args.mine_threshold, args.mine_rule, args.weight,
        args.retrain_baseline)


if __name__ == "__main__":
    main()
