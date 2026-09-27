"""E-G4: unseen-country stage-2 transfer and its threshold curve (France proxy).

Train the shipped stage-2 feature set (s2lab_all) on source-country folds 1,2 (early stop on
source fold 3) and score the target country's folds 0/3. Reports the target eval-fold F0.5 over
a tau grid (thr+excl), next to the in-country model s2lab_all. Stage 1 was trained on both
countries, so this is a lower bound on the France-like loss.

  BER_WORK_DIR=../../work_xd python src/experiments/loco_s2.py --src US --dst India
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ber.approaches as ap  # noqa: E402
from ber import config  # noqa: E402
from ber.utils import log, save_json  # noqa: E402


def load(feats, country, folds):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "country"]).filter(pl.col("country") == country)
    df = (pl.scan_parquet(config.WORK_DIR / "s2lab" / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
            .select(["rid", "gid", "fold", "y"] + feats).collect().join(s1.select("rid"), on="rid", how="semi"))
    return df.select("rid", "gid", "fold", "y"), np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="US")
    a.add_argument("--dst", default="India")
    x = a.parse_args()
    WD = config.WORK_DIR
    ref = lgb.Booster(model_file=str(WD / "models" / "s2lab_all.txt"))
    feats = ref.feature_name()
    mtr, Xtr = load(feats, x.src, list(config.TRAIN_FOLDS))
    mtu, Xtu = load(feats, x.src, [config.TUNE_FOLD])
    log(f"{x.src}->{x.dst}: train rows {len(mtr):,}")
    dtr = lgb.Dataset(Xtr, mtr.get_column("y").to_numpy(), feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xtu, mtu.get_column("y").to_numpy(), reference=dtr)
    m = lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False)])
    del Xtr, dtr
    mt, Xt = load(feats, x.dst, [config.TUNE_FOLD, config.EVAL_FOLD])
    s1c = pl.read_parquet(WD / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(WD / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    from ber.metric import Evaluator
    out = {}
    taus = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    for label, model in (("unseen", m), ("in_country_ref", ref)):
        p = ap._pred(model, Xt)
        ex = ap.exclusive_mask(mt.get_column("gid").to_numpy(), p)
        msk = (mt.get_column("fold") == config.EVAL_FOLD).to_numpy()
        me = mt.filter(pl.Series(msk))
        pop = (s1c.filter((pl.col("fold") == config.EVAL_FOLD) & (pl.col("country") == x.dst)).select("rid")
                  .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
        idx = me.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
        ev = Evaluator(idx.get_column("idx").to_numpy(), me.get_column("y").to_numpy(), pop.get_column("n_true").to_numpy())
        res = {}
        for t in taus:
            r = ev.report((p[msk] >= t) & ex[msk])
            res[str(t)] = {k: r[k] for k in ("macro_f05", "macro_precision", "macro_recall", "false_positive_pairs", "predicted_pairs_per_s1")}
        out[label] = res
        best = max(res, key=lambda k: res[k]["macro_f05"])
        log(f"{label}: {x.dst} eval F0.5 @0.68 {res['0.68']['macro_f05']} (P {res['0.68']['macro_precision']} R {res['0.68']['macro_recall']}) "
            f"| best tau {best} -> {res[best]['macro_f05']}")
        log("  curve " + " ".join(f"{t}:{res[str(t)]['macro_f05']}" for t in taus))
    save_json(out, WD / "reports" / f"loco_s2_{x.src}_to_{x.dst}.json")


if __name__ == "__main__":
    main()
