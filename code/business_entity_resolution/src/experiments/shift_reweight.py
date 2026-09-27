"""E-G2: covariate-shift (importance-weighted) estimate of test macro F0.5.

Per S1 of the eval fold we know its F0.5 under the shipped decision. Test S1s differ in
label-free covariates (candidate counts, affiliate look-alike candidates, score mass in the
uncertain band). Reweighting eval S1s so the covariate histogram matches test estimates
test F0.5, assuming P(error | covariates, country) transfers. Covariates are computed
identically on eval and test from the pruned stage-2 pairs + scores.

usage: BER_WORK_DIR=../../work_xd python src/experiments/shift_reweight.py [--tau 0.68]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import exclusive_mask  # noqa: E402

AFF = ["group", "partners", "holdings", "ltd", "co", "corp", "eastgate", "westgate", "northside", "southside",
       "lakeside", "riverside", "midtown", "downtown", "central", "greater", "metro", "valley", "harbor", "summit",
       "coastal", "limited", "pvt", "private", "llp", "company", "ventures", "industries", "enterprises"]


def covariates(pairs, s1, tg):
    d = (pairs.join(s1.select("rid", "nm"), on="rid").join(tg.select("gid", t_nm="nm"), on="gid")
         .with_columns(aff=pl.col("t_nm").str.split(" ").list.set_difference(pl.col("nm").str.split(" "))
                       .list.set_intersection(pl.lit(AFF)).list.len().gt(0)))
    return d.group_by("rid").agg(
        n_c=pl.len(), n_aff=pl.col("aff").sum(),
        n_mid=pl.col("p").is_between(0.05, 0.99).sum(), n_hi=(pl.col("p") > 0.99).sum(),
        n_affmid=(pl.col("aff") & pl.col("p").is_between(0.05, 0.99)).sum())


def bucket(df):
    return df.with_columns(
        b_c=pl.col("n_c").fill_null(0).clip(0, 12) // 3, b_aff=pl.col("n_aff").fill_null(0).clip(0, 3),
        b_mid=pl.col("n_mid").fill_null(0).clip(0, 3), b_hi=pl.col("n_hi").fill_null(0).clip(0, 6),
        b_affmid=pl.col("n_affmid").fill_null(0).clip(0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.68)
    ap.add_argument("--test", default="reports/test_s2lab_all_pairs.parquet")
    ap.add_argument("--scores", default="reports/scores_s2lab_all.parquet")
    ap.add_argument("--keys", default="country,b_c,b_aff,b_mid,b_hi,b_affmid")
    a = ap.parse_args()
    W, P = config.WORK_DIR, config.WORK_DIR / "prep"
    s1tr = pl.read_parquet(P / "s1_train.parquet", columns=["rid", "nm", "country", "fold"])
    tgtr = pl.read_parquet(P / "tg_train.parquet", columns=["gid", "nm"])
    sc = pl.read_parquet(W / a.scores)
    ex = exclusive_mask(sc.get_column("gid").to_numpy(), sc.get_column("p").to_numpy())
    sc = sc.with_columns(keep=pl.Series(ex) & (pl.col("p") >= a.tau))
    ev = sc.filter(pl.col("fold") == config.EVAL_FOLD)
    gt = pl.read_parquet(P / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1tr.filter(pl.col("fold") == config.EVAL_FOLD).select("rid", "country")
           .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)))
    per = ev.group_by("rid").agg(tp=(pl.col("keep") & (pl.col("y") == 1)).sum(), npred=pl.col("keep").sum())
    pop = pop.join(per, on="rid", how="left").fill_null(0).with_columns(
        f=pl.when((pl.col("n_true") == 0) & (pl.col("npred") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("tp") / (pl.col("npred") + 0.25 * pl.col("n_true"))))
    pop = bucket(pop.join(covariates(ev, s1tr, tgtr), on="rid", how="left"))
    s1te = pl.read_parquet(P / "s1_test.parquet", columns=["rid", "nm", "country"])
    tgte = pl.read_parquet(P / "tg_test.parquet", columns=["gid", "nm"])
    te = pl.read_parquet(W / a.test)
    tpop = bucket(s1te.select("rid", "country").join(covariates(te, s1te, tgte), on="rid", how="left"))
    keys = a.keys.split(",")
    ce = pop.group_by(keys).agg(n_e=pl.len(), f_e=pl.col("f").mean())
    ct = tpop.group_by(keys + ([] if "country" in keys else ["country"])).agg(n_t=pl.len())
    j = ct.join(ce, on=keys, how="left")
    miss = j.filter(pl.col("n_e").is_null())
    print(f"test S1 in covariate cells unseen in eval: {miss['n_t'].sum():,} of {j['n_t'].sum():,}")
    # unseen cells: fall back to country mean
    cm = pop.group_by("country").agg(f_c=pl.col("f").mean())
    j = j.join(cm, on="country", how="left").with_columns(f_hat=pl.col("f_e").fill_null(pl.col("f_c")).fill_null(pop["f"].mean()))
    print("eval macro F0.5 by country (actual):")
    print(pop.group_by("country").agg(pl.col("f").mean(), pl.len()).sort("country"))
    print("reweighted estimate on test covariates:")
    est = j.group_by("country").agg(f_est=(pl.col("f_hat") * pl.col("n_t")).sum() / pl.col("n_t").sum(), n=pl.col("n_t").sum()).sort("country")
    print(est)
    tot = (j["f_hat"] * j["n_t"]).sum() / j["n_t"].sum()
    print(f"overall (test S1 mix, France via US+India cells): {tot:.5f}; eval actual {pop['f'].mean():.5f}")


if __name__ == "__main__":
    main()
