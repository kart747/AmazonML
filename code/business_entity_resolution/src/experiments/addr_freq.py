"""Address-ambiguity features for stage 2: how many S1 share an address.

Why: the model has name frequency (q_nm_freq, t_nm_freq) but no address frequency. On test, 11.9%
of France S1 share an exact normalized address with another S1, against 4-5.6% in US/India/train. There,
"same address, different name" is much weaker evidence for a match, and the France mid-band score
mass is 5x the US's (amazonml-27 audit). Train has enough shared-address S1 to learn the
interaction, so the feature can transfer to France.

Features (label-free; counts over the S1 pool of the country in the same split/world):
  q_ad_n        # S1 with exactly this S1's normalized address (incl. itself), null if empty
  t_ad_s1n      # S1 whose normalized address equals the TARGET's address, null if empty
  t_ad_others   t_ad_s1n minus 1 if the query S1 itself has that address: competing S1 at the
                target's address
  q_ad_rate     q_ad_n per 1M S1 of the country (pool-size invariant, like q_nm_freq)

  python src/experiments/addr_freq.py --src s2lab --dst s2lab_adf [--split train|test] [--drop 0.19]
(--drop: the same dense-world S1 mask as density.py, for s2lab_dense parts). Output rows are the
source rows plus the 4 columns, part by part. Then: BER_S2LAB_OUT=s2lab_adf stage2_lab.py evaluate ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.utils import done, log, mark_done  # noqa: E402

NEW = ["q_ad_n", "t_ad_s1n", "t_ad_others", "q_ad_rate"]


def tables(split, drop):
    WD = config.WORK_DIR
    s1 = pl.read_parquet(WD / "prep" / f"s1_{split}.parquet", columns=["rid", "country", "ad"])
    if drop:
        rng = np.random.default_rng(config.SEED)   # same mask as density.py / density_train.py
        s1 = s1.filter(pl.Series(rng.random(s1.height) >= drop))
    s1 = s1.with_columns(pl.when(pl.col("ad") == "").then(None).otherwise(pl.col("ad")).alias("ad"))
    cnt = s1.filter(pl.col("ad").is_not_null()).group_by("country", "ad").len("n")
    ncty = s1.group_by("country").len("nc")
    q = (s1.join(cnt, on=["country", "ad"], how="left").join(ncty, on="country")
           .select("rid", "country", _qad="ad", q_ad_n="n", q_ad_rate=pl.col("n") * 1e6 / pl.col("nc")))
    tg = pl.read_parquet(WD / "prep" / f"tg_{split}.parquet", columns=["gid", "country", "ad"]) \
        .with_columns(pl.when(pl.col("ad") == "").then(None).otherwise(pl.col("ad")).alias("ad"))
    t = tg.join(cnt, on=["country", "ad"], how="left").select("gid", _tad="ad", t_ad_s1n=pl.col("n").fill_null(0))
    t = t.with_columns(pl.when(pl.col("_tad").is_null()).then(None).otherwise(pl.col("t_ad_s1n")).alias("t_ad_s1n"))
    return q, t


def run(src, dst, split, drop):
    WD = config.WORK_DIR
    q, t = tables(split, drop)
    out = WD / dst / split
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted((WD / src / split).glob("*.parquet")):
        o = out / p.name
        if done(o):
            continue
        df = pl.read_parquet(p)
        n = df.height
        df = (df.with_row_index("_r").join(q.drop("country"), on="rid", how="left")
                .join(t, on="gid", how="left").sort("_r")
                .with_columns(t_ad_others=(pl.col("t_ad_s1n") - (pl.col("_qad") == pl.col("_tad")).cast(pl.UInt32).fill_null(0)))
                .drop("_r", "_qad", "_tad")
                .with_columns([pl.col(c).cast(pl.Float32) for c in NEW]))
        assert df.height == n
        df.write_parquet(o)
        mark_done(o)
        log(f"addr_freq {dst}/{split}/{p.name}: {n:,} rows")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="s2lab")
    a.add_argument("--dst", default="s2lab_adf")
    a.add_argument("--split", default="train")
    a.add_argument("--drop", type=float, default=0.0)
    x = a.parse_args()
    run(x.src, x.dst, x.split, x.drop)


if __name__ == "__main__":
    main()
