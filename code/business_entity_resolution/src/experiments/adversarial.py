"""Adversarial validation of the stage-2 inputs: which features shift between train and test?

A LightGBM classifier separates stage-2 rows of train (folds 0-3) from test rows of the same
country (US / India; France has no train counterpart, so it is scored against both pooled). AUC
~0.5 means no covariate shift; the gain ranking names the shifted features. Only kept-region
rows matter for the metric, so both sides are restricted to p1 >= --p1 (default 0.05: the
candidates stage 2 actually has to decide).

    BER_WORK_DIR=../work_xd python src/experiments/adversarial.py --ref s2lab_all [--n 300000]
Output: <work>/reports/adversarial_<ref>.json. RAM ~2 GB.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.utils import log, save_json  # noqa: E402


def sample(split, country, feats, n, p1min, seed):
    WD = config.WORK_DIR
    s1 = pl.read_parquet(WD / "prep" / f"s1_{split}.parquet", columns=["rid", "country"]).filter(pl.col("country") == country)
    lf = pl.scan_parquet(WD / s2.OUT / split / f"{country}_*.parquet").filter(pl.col("p1") >= p1min)
    if split == "train":
        lf = lf.filter(pl.col("fold").is_in([0, 1, 2, 3]))
    df = lf.select(["rid"] + feats).collect().join(s1.select("rid"), on="rid", how="semi")
    return df.sample(min(n, df.height), seed=seed).drop("rid")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ref", default="s2lab_all")
    a.add_argument("--n", type=int, default=300000)
    a.add_argument("--p1", type=float, default=0.05)
    a.add_argument("--exclude", default="", help="comma-separated substrings of features to leave out, e.g. freq,idf")
    a.add_argument("--tag", default="")
    x = a.parse_args()
    feats = lgb.Booster(model_file=str(config.WORK_DIR / "models" / f"{x.ref}.txt")).feature_name()
    ex = [e for e in x.exclude.split(",") if e]
    feats = [f for f in feats if not any(e in f for e in ex)]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=config.N_THREADS, seed=config.SEED)
    out = {"ref": x.ref, "p1_min": x.p1, "n": x.n, "exclude": ex, "n_features": len(feats)}
    for country in ("US", "India"):
        tr = sample("train", country, feats, x.n, x.p1, 1)
        te = sample("test", country, feats, x.n, x.p1, 2)
        X = np.ascontiguousarray(pl.concat([tr, te]).to_numpy(), dtype=np.float32)
        y = np.r_[np.zeros(tr.height), np.ones(te.height)]
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        itr, iva = idx[:cut], idx[cut:]
        m = lgb.train(params, lgb.Dataset(X[itr], y[itr], feature_name=feats), 300)
        auc = roc_auc_score(y[iva], m.predict(X[iva]))
        gain = dict(sorted(zip(feats, m.feature_importance("gain").round().astype(int).tolist()), key=lambda kv: -kv[1])[:15])
        # per-feature single-variable shift: mean train vs test (top gain features)
        shift = {f: [round(float(tr.get_column(f).mean() or 0), 4), round(float(te.get_column(f).mean() or 0), 4)] for f in list(gain)[:10]}
        out[country] = {"rows_train": tr.height, "rows_test": te.height, "auc": round(float(auc), 4), "gain_top15": gain,
                        "mean_train_test_top10": shift}
        log(f"adversarial {country}: AUC {auc:.4f}; top shifted {list(gain)[:8]}")
        for f, (mt, mv) in shift.items():
            log(f"   {f:22} train {mt:>10} test {mv:>10}")
    save_json(out, config.WORK_DIR / "reports" / f"adversarial_{x.ref}{x.tag}.json")


if __name__ == "__main__":
    main()
