"""E-G6: which stage-2 feature groups break unseen-country transfer (France proxy).

Run in the full leave-one-country-out world (work_loco: stage 1 AND stage 2 trained on the
source country only). For each feature-group ablation, stage 2 is trained on source folds 1,2
(early stop on source fold 3), tau is picked on the SOURCE tune fold (what we can do for
France), and the target's eval fold 0 is reported at that tau and at its best tau (oracle).

  BER_WORK_DIR=../../work_loco python src/experiments/loco_ablate.py --src US --dst India
Output: <work>/reports/loco_ablate_<src>_to_<dst>.json
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
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json  # noqa: E402

GROUPS = {
    "p1": ["p1", "p1_qrank", "p1_qgap", "p1_qsum", "p1_qn50", "p1_qmax_other", "sib_p1", "twin_p1max",
           "p1_srank", "p1_sn50"],
    "freq": ["q_nm_freq", "t_nm_freq", "q_unm_maxidf", "t_unm_maxidf", "q_match_idf", "t_match_idf",
             "core_inter", "core_jacc", "q_core_n", "t_core_n", "t_oov"],
    "ctx": ["r_name", "r_addr", "r_combo", "r_rev", "t_top1", "t_gap12", "rev_margin", "t_is_best",
            "q_rank", "q_gap", "q_ncand"],
    "src": ["t_src", "t_nonascii"],
}
CONFIGS = {"all": [], "-p1": ["p1"], "-freq": ["freq"], "-ctx": ["ctx"], "-src": ["src"],
           "-p1-freq": ["p1", "freq"], "-p1-ctx": ["p1", "ctx"], "pairwise_only": ["p1", "freq", "ctx", "src"]}
TAUS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load(feats, country, folds):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "country"]).filter(pl.col("country") == country)
    return (pl.scan_parquet(config.WORK_DIR / "s2lab" / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
              .select(["rid", "gid", "fold", "y"] + feats).collect().join(s1.select("rid"), on="rid", how="semi"))


def evaluator(meta, fold, country):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(config.WORK_DIR / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter((pl.col("fold") == fold) & (pl.col("country") == country)).select("rid")
             .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), meta.get_column("y").to_numpy(), pop.get_column("n_true").to_numpy())


def curve(model, feats, df, ev):
    p = ap._pred(model, np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32))
    ex = ap.exclusive_mask(df.get_column("gid").to_numpy(), p)
    return {str(t): ev.report((p >= t) & ex) for t in TAUS}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="US")
    a.add_argument("--dst", default="India")
    a.add_argument("--configs", default=",".join(CONFIGS))
    x = a.parse_args()
    WD = config.WORK_DIR
    allf = lgb.Booster(model_file=str(WD / "models" / "s2lab_all.txt")).feature_name()
    tr = load(allf, x.src, list(config.TRAIN_FOLDS))
    tu = load(allf, x.src, [config.TUNE_FOLD])
    te = load(allf, x.dst, [config.EVAL_FOLD])
    ev_tu, ev_te = evaluator(tu, config.TUNE_FOLD, x.src), evaluator(te, config.EVAL_FOLD, x.dst)
    out = {}
    for name in x.configs.split(","):
        drop = {f for g in CONFIGS[name] for f in GROUPS[g]}
        feats = [f for f in allf if f not in drop]
        X = np.ascontiguousarray(tr.select(feats).to_numpy(), dtype=np.float32)
        dtr = lgb.Dataset(X, tr.get_column("y").to_numpy(), feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(np.ascontiguousarray(tu.select(feats).to_numpy(), dtype=np.float32),
                          tu.get_column("y").to_numpy(), reference=dtr)
        m = lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False)])
        del X, dtr, dva
        c_tu, c_te = curve(m, feats, tu, ev_tu), curve(m, feats, te, ev_te)
        t_src = max(c_tu, key=lambda t: c_tu[t]["macro_f05"])
        t_best = max(c_te, key=lambda t: c_te[t]["macro_f05"])
        r = c_te[t_src]
        out[name] = {"n_feats": len(feats), "src_tau": float(t_src), "src_tune_f05": c_tu[t_src]["macro_f05"],
                     "dst_f05": r["macro_f05"], "dst_P": r["macro_precision"], "dst_R": r["macro_recall"],
                     "dst_fp": r["false_positive_pairs"], "dst_best_tau": float(t_best),
                     "dst_best_f05": c_te[t_best]["macro_f05"],
                     "curve": {t: c_te[t]["macro_f05"] for t in c_te}}
        log(f"{name:14s} {len(feats)} feats | {x.src} tune {c_tu[t_src]['macro_f05']} @tau {t_src} | "
            f"{x.dst} eval {r['macro_f05']} (P {r['macro_precision']} R {r['macro_recall']}) | "
            f"best tau {t_best} -> {c_te[t_best]['macro_f05']}")
        save_json(out, WD / "reports" / f"loco_ablate_{x.src}_to_{x.dst}.json")


if __name__ == "__main__":
    main()
