"""Out-of-fold hard-negative upweighting for the stage2_lab / dense_mix stage 2.

Idea from origin/hard-negative-mining (hard_negative_lab.py, Yashwin69): score the training
folds out-of-fold, find negatives the model would accept, and give them more weight in the
final fit. That branch works on the old A2 feat2 parts (54 features, 18M unpruned rows) and
re-appends the mined rows, which duplicates rows already in the training set. Duplication is
the same as weight 2, so here it is written as an explicit sample weight, on the pipeline
that is actually shipped (work_xd s2lab parts, 92 features, optional dense+normal mix).

  mine      for each training fold f in {1,2}: fit on the other training fold (same rows/world
            mix as the final model), early stop on the tune fold, score fold f. Saves p_oof.
  evaluate  weight = W for negatives with p_oof >= --mine-tau (optionally capped per S1 and
            per target), 1 otherwise; fit on folds 1,2; score exactly like density_train
            (normal world and 19%-dense world, tau from the tune fold). Writes
            reports/density_train_<name>.json, so density_train.py predict and
            world_sim.py score work unchanged on models/s2lab_<name>.txt.

    BER_WORK_DIR=../../work_xd python src/experiments/hard_negative_s2.py mine --mix
    BER_WORK_DIR=../../work_xd python src/experiments/hard_negative_s2.py evaluate --mix --weight 2
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ber.approaches as ap  # noqa: E402
import density_train as dt  # noqa: E402
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.utils import log, save_json, stage  # noqa: E402


def _rows(feats, folds, mix):
    m, X = dt._load(s2.OUT, folds, feats)
    m = m.with_columns(world=pl.lit(0, pl.Int8))
    if mix:
        m2, X2 = dt._load(dt.DENSE, folds, feats)
        m, X = pl.concat([m, m2.with_columns(world=pl.lit(1, pl.Int8))]), np.concatenate([X, X2])
        del X2
    return m, X


def _fit(X, y, w, Xtu, ytu, feats):
    dtr = lgb.Dataset(X, y, weight=w, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xtu, ytu, reference=dtr)
    return lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(200)])


def _tune(feats, mix):
    # early stopping on the dense tune fold when training on the mix (as density_train does)
    return dt._load(dt.DENSE if mix else s2.OUT, [config.TUNE_FOLD], feats)


def _oof_path(mix):
    return config.WORK_DIR / "reports" / f"hn_oof{'_mix' if mix else ''}.parquet"


def mine(ref, mix):
    feats = lgb.Booster(model_file=str(config.WORK_DIR / "models" / f"{ref}.txt")).feature_name()
    mtr, Xtr = _rows(feats, list(config.TRAIN_FOLDS), mix)
    mtu, Xtu = _tune(feats, mix)
    y, fold = mtr.get_column("y").to_numpy(), mtr.get_column("fold").to_numpy()
    p = np.zeros(len(y), np.float32)
    with stage(f"hn_mine{'_mix' if mix else ''}"):
        for f in config.TRAIN_FOLDS:
            fit, held = fold != f, fold == f
            m = _fit(Xtr[fit], y[fit], None, Xtu, mtu.get_column("y").to_numpy(), feats)
            p[held] = ap._pred(m, Xtr[held])
            log(f"hn_mine: fold {f} scored by a model on the other fold ({m.best_iteration} iters)")
    mtr.with_columns(p_oof=pl.Series(p)).write_parquet(_oof_path(mix))
    neg = y == 0
    for t in (0.3, 0.5, 0.68, 0.8):
        log(f"OOF negatives with p >= {t}: {(neg & (p >= t)).sum():,}  positives below: {((~neg) & (p < t)).sum():,}")


def _cap(df, k):
    """top-k mined negatives per S1 union top-k per target (the branch's symmetric cap)."""
    a = df.filter(pl.col("p_oof").rank("ordinal", descending=True).over(["world", "rid"]) <= k)
    b = df.filter(pl.col("p_oof").rank("ordinal", descending=True).over(["world", "gid"]) <= k)
    return pl.concat([a, b]).unique(["world", "rid", "gid"])


def evaluate(ref, mix, weight, mine_tau, cap, pos_weight, tag):
    WD = config.WORK_DIR
    feats = lgb.Booster(model_file=str(WD / "models" / f"{ref}.txt")).feature_name()
    name = f"dense_mix_hn{tag}" if mix else f"hn{tag}"
    s1, dropped = dt.dropped_s1(0.19)
    with stage(f"hn_evaluate_{name}"):
        mtr, Xtr = _rows(feats, list(config.TRAIN_FOLDS), mix)
        oof = pl.read_parquet(_oof_path(mix))
        assert oof.select("rid", "gid", "world").equals(mtr.select("rid", "gid", "world")), "OOF rows not aligned"
        hard = oof.filter((pl.col("y") == 0) & (pl.col("p_oof") >= mine_tau))
        if cap:
            hard = _cap(hard, cap)
        mk = (mtr.select("rid", "gid", "world").with_row_index("_i")
                 .join(hard.select("rid", "gid", "world"), on=["rid", "gid", "world"], how="inner")
                 .get_column("_i").to_numpy())
        w = np.ones(len(mtr), np.float32)
        w[mk] = weight
        y = mtr.get_column("y").to_numpy()
        if pos_weight != 1.0:  # hard positives: true pairs the OOF model would reject
            w[(y == 1) & (oof.get_column("p_oof").to_numpy() < mine_tau)] = pos_weight
        log(f"{name}: {len(y):,} rows, {len(mk):,} hard negatives (p_oof >= {mine_tau}, cap {cap}) weighted {weight}")
        mtu, Xtu = _tune(feats, mix)
        m = _fit(Xtr, y, w, Xtu, mtu.get_column("y").to_numpy(), feats)
        m.save_model(str(WD / "models" / f"s2lab_{name}.txt"))
        del Xtr, Xtu
        out = {"ref": ref, "mix": mix, "weight": weight, "mine_tau": mine_tau, "cap": cap, "pos_weight": pos_weight,
               "hard_negatives": int(len(mk)), "best_iteration": m.best_iteration, "new": {}}
        out["new"]["normal"] = dt._score(m, feats, s2.OUT, config.EVAL_FOLD, config.TUNE_FOLD, False, dropped, s1)
        out["new"]["dense"] = dt._score(m, feats, dt.DENSE, config.EVAL_FOLD, config.TUNE_FOLD, True, dropped, s1)
        out["new"]["dense"]["eval_at_normal_tau"] = dt._score(
            m, feats, dt.DENSE, config.EVAL_FOLD, config.TUNE_FOLD, True, dropped, s1,
            tau_normal=out["new"]["normal"]["tune"]["tau"])["eval_at_tau"]
    save_json(out, WD / "reports" / f"density_train_{name}.json")
    o = out["new"]
    log(f"{name}: normal tune {o['normal']['tune']['tune_f05']} tau {o['normal']['tune']['tau']} eval "
        f"{o['normal']['eval']['macro_f05']} FP {o['normal']['eval']['false_positive_pairs']} | dense tune "
        f"{o['dense']['tune']['tune_f05']} eval(normal tau) {o['dense']['eval_at_normal_tau']['macro_f05']} "
        f"FP {o['dense']['eval_at_normal_tau']['false_positive_pairs']}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["mine", "evaluate"])
    a.add_argument("--ref", default="s2lab_dense_mix", help="model whose feature list is used")
    a.add_argument("--mix", action="store_true", help="train on dense + normal folds 1,2 (dense_mix)")
    a.add_argument("--weight", type=float, default=2.0, help="2.0 = the branch's duplicate-row scheme")
    a.add_argument("--mine-tau", type=float, default=0.68)
    a.add_argument("--cap", type=int, default=0, help="0 = no cap; the branch used 5")
    a.add_argument("--pos-weight", type=float, default=1.0)
    a.add_argument("--tag", default="")
    x = a.parse_args()
    if x.cmd == "mine":
        mine(x.ref, x.mix)
    else:
        evaluate(x.ref, x.mix, x.weight, x.mine_tau, x.cap, x.pos_weight, x.tag)


if __name__ == "__main__":
    main()
