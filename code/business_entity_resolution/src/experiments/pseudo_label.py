"""Unseen-country adaptation by pseudo-labelling, measured with a leave-one-country-out proxy.

France is 15% of test S1 (~15% of the public LB) and has no labels. Proxy: pretend one train
country (--target, default India) is unlabeled.
  base    stage 2 (stage2_lab features of --ref) trained on the SOURCE country's train folds only,
          tau from the source country's tune fold, reported on the TARGET country's eval fold.
  pseudo  the base model scores the target country's train+tune folds. Confident pairs become
          pseudo-labels (p >= --hi and the target's best S1 -> 1; p <= --lo -> 0; the rest are
          dropped), and stage 2 is retrained on source rows + pseudo rows. Same tau protocol, same report.
  oracle  trained with the target country's TRUE labels (upper bound; = the normal all-country model
          restricted to the target's rows).
If pseudo recovers a meaningful part of (oracle - base), the same recipe applies to France on
test: pseudo-label France test pairs with the shipped model, then retrain.
Caveat: stage 1 (p1 and everything derived from it) was trained on both countries, so this
measures stage-2 transfer only.

    BER_WORK_DIR=../work_xd python src/experiments/pseudo_label.py --ref s2lab_all --target India
Output: <work>/reports/pseudo_label_<target>.json. RAM ~3 GB at 4 threads.
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
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json, stage  # noqa: E402


def country_rows(feats, folds):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "country"])
    df = (pl.scan_parquet(config.WORK_DIR / s2.OUT / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
            .select(["rid", "gid", "fold", "y"] + feats).collect().join(s1, on="rid", how="left"))
    return df


def evaluator(meta, fold, country):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(config.WORK_DIR / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter((pl.col("fold") == fold) & (pl.col("country") == country)).select("rid")
             .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), meta.get_column("y").to_numpy(), pop.get_column("n_true").to_numpy())


def fit(tr, tu, feats):
    X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
    dtr = lgb.Dataset(X(tr), tr.get_column("y").to_numpy(), feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X(tu), tu.get_column("y").to_numpy(), reference=dtr)
    return lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(0)])


def report(m, feats, tu, ev, src, tgt):
    """tau/variant chosen on the SOURCE tune fold, reported on the TARGET eval fold."""
    X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
    uf, ef = config.TUNE_FOLD, config.EVAL_FOLD
    p_tu, p_ev = ap._pred(m, X(tu)), ap._pred(m, X(ev))
    g_tu, g_ev = tu.get_column("gid").to_numpy(), ev.get_column("gid").to_numpy()
    ch = ap.choose_decision(evaluator(tu, uf, src), p_tu, g_tu)
    v = max(ch, key=lambda k: ch[k]["tune_f05"])
    e = evaluator(ev, ef, tgt).report(ap.decide(p_ev, g_ev, v, ch[v]["tau"]))
    return {"variant": v, "tau": ch[v]["tau"], "source_tune_f05": ch[v]["tune_f05"], "target_eval": e}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ref", default="s2lab_all")
    a.add_argument("--target", default="India")
    a.add_argument("--source", default="US")
    a.add_argument("--hi", type=float, default=0.95)
    a.add_argument("--lo", type=float, default=0.05)
    x = a.parse_args()
    feats = lgb.Booster(model_file=str(config.WORK_DIR / "models" / f"{x.ref}.txt")).feature_name()
    tf, uf, ef = list(config.TRAIN_FOLDS), config.TUNE_FOLD, config.EVAL_FOLD
    by = lambda d, c: d.filter(pl.col("country") == c)
    out = {"ref": x.ref, "source": x.source, "target": x.target, "hi": x.hi, "lo": x.lo}
    with stage(f"pseudo_label_{x.target}"):
        train = country_rows(feats, tf)
        tune = country_rows(feats, [uf])
        ev = by(country_rows(feats, [ef]), x.target)
        src_tr, src_tu, tgt_tr, tgt_tu = by(train, x.source), by(tune, x.source), by(train, x.target), by(tune, x.target)
        base = fit(src_tr, src_tu, feats)
        out["base"] = report(base, feats, src_tu, ev, x.source, x.target)
        log(f"base ({x.source} only): {x.target} eval F0.5 {out['base']['target_eval']['macro_f05']}")
        # pseudo-labels on the target's unlabeled train + tune folds
        pool = pl.concat([tgt_tr, tgt_tu])
        p = ap._pred(base, np.ascontiguousarray(pool.select(feats).to_numpy(), dtype=np.float32))
        best = ap.exclusive_mask(pool.get_column("gid").to_numpy(), p)
        lab = np.where((p >= x.hi) & best, 1, np.where(p <= x.lo, 0, -1))
        pool = (pool.with_columns(y_true=pl.col("y"), y=pl.Series(lab)).filter(pl.col("y") >= 0)
                    .with_columns(pl.col("y").cast(src_tr.schema["y"])))
        yt, yp = pool.get_column("y_true").to_numpy(), pool.get_column("y").to_numpy()
        out["pseudo_labels"] = {"rows": int(len(yp)), "dropped_middle": int((lab < 0).sum()), "pos": int(yp.sum()),
                                "pos_precision": round(float(yt[yp == 1].mean()), 5),
                                "neg_error": round(float(yt[yp == 0].mean()), 5)}
        log(f"pseudo labels: {out['pseudo_labels']}")
        ps = fit(pl.concat([src_tr, pool.drop("y_true")]), src_tu, feats)
        out["pseudo"] = report(ps, feats, src_tu, ev, x.source, x.target)
        log(f"pseudo: {x.target} eval F0.5 {out['pseudo']['target_eval']['macro_f05']}")
        orc = fit(pl.concat([src_tr, tgt_tr]), pl.concat([src_tu, tgt_tu]), feats)
        out["oracle"] = report(orc, feats, src_tu, ev, x.source, x.target)
        log(f"oracle (true {x.target} labels): {x.target} eval F0.5 {out['oracle']['target_eval']['macro_f05']}")
    save_json(out, config.WORK_DIR / "reports" / f"pseudo_label_{x.target}.json")


if __name__ == "__main__":
    main()
