"""Candidates for extra train S1 folds (default 4..9), reusing the stored reverse view.

Stage 2 is trained on folds {1,2} only. Folds 4..9 (the reserve) can add training rows
whose p1 comes from the averaged stage-1 models, exactly as on test. blocking.py would
recompute the reverse view (about 65 of 123 min on train), but that view already covers
every S1 of every fold (work/block/train/rev_*.parquet). This script refits the identical
seeded TF-IDF vectorizers and computes only the forward views for the extra query folds,
then writes cand_<country>.parquet into an isolated work dir. rev_* is symlinked there.

Usage:
  BER_WORK_DIR=../../work_f49 BER_QUERY_FOLDS=4,5,6,7,8,9 python src/experiments/block_more_folds.py --src ../../work
  (the dst work dir needs prep/ and models/ symlinked to the source's)
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.blocking import _sample, fit_transform, rowdot, topk  # noqa: E402
from ber.utils import done, log, mark_done, stage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source work dir holding block/train/rev_*.parquet")
    a = ap.parse_args()
    src_b, bdir = Path(a.src) / "block" / "train", config.WORK_DIR / "block" / "train"
    assert src_b.resolve() != bdir.resolve(), "dst must be an isolated work dir"
    bdir.mkdir(parents=True, exist_ok=True)
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_train.parquet", columns=["rid", "country", "fold", "nm", "ad"])
    tg = pl.read_parquet(pdir / "tg_train.parquet", columns=["gid", "country", "nm", "ad"])
    B = config.BLOCK
    for country in sorted(s1.get_column("country").unique().to_list(), key=lambda c: s1.filter(pl.col("country") == c).height):
        fc, fr = bdir / f"cand_{country}.parquet", bdir / f"rev_{country}.parquet"
        if not fr.exists():
            os.symlink((src_b / f"rev_{country}.parquet").resolve(), fr)
            mark_done(fr)
        if done(fc):
            continue
        s1c, tgc = s1.filter(pl.col("country") == country), tg.filter(pl.col("country") == country)
        S_rid, T_gid = s1c.get_column("rid").to_numpy(), tgc.get_column("gid").to_numpy()
        q_loc = np.flatnonzero(s1c.get_column("fold").is_in(config.QUERY_FOLDS).to_numpy())
        s_nm, t_nm = s1c.get_column("nm").to_list(), tgc.get_column("nm").to_list()
        s_ad, t_ad = s1c.get_column("ad").to_list(), tgc.get_column("ad").to_list()
        with stage(f"block_more_{country}", {"queries": len(q_loc)}):
            vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=B["name_max_df"],
                                 sublinear_tf=True, lowercase=False, dtype=np.float32)
            NS, NT = fit_transform(vn, _sample(s_nm + t_nm, B["fit_sample"]), s_nm, t_nm)
            va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=B["addr_max_df"],
                                 sublinear_tf=True, lowercase=False, dtype=np.float32)
            AS, AT = fit_transform(va, _sample(s_ad + t_ad, B["fit_sample"]), s_ad, t_ad)
            h = np.float32(math.sqrt(0.5))
            CS, CT = sp.hstack([NS * h, AS * h]).tocsr(), sp.hstack([NT * h, AT * h]).tocsr()
            views = {}
            for v, (A, Bm, k) in {"name": (NS[q_loc], NT, B["k_name"]), "addr": (AS[q_loc], AT, B["k_addr"]),
                                  "combo": (CS[q_loc], CT, B["k_combo"])}.items():
                q, t, _, r = topk(A, Bm, k)
                views[v] = pl.DataFrame({"sl": q_loc[q].astype(np.int32), "tl": t.astype(np.int32), f"r_{v}": r})
                log(f"{country} view {v}: {len(q):,} pairs")
            # stored reverse view, mapped back to local row indices, restricted to the query S1
            rloc = pl.DataFrame({"rid": S_rid, "sl": np.arange(len(S_rid), dtype=np.int32)})
            tloc = pl.DataFrame({"gid": T_gid, "tl": np.arange(len(T_gid), dtype=np.int32)})
            qset = pl.DataFrame({"sl": q_loc.astype(np.int32)})
            rev_q = (pl.read_parquet(fr, columns=["rid", "gid", "rrank"])
                       .join(rloc, on="rid").join(qset, on="sl").join(tloc, on="gid")
                       .select("sl", "tl", r_rev="rrank"))
            cand = views["name"].join(views["addr"], on=["sl", "tl"], how="full", coalesce=True)
            cand = cand.join(views["combo"], on=["sl", "tl"], how="full", coalesce=True)
            cand = cand.join(rev_q, on=["sl", "tl"], how="full", coalesce=True)
            sl, tl = cand.get_column("sl").to_numpy(), cand.get_column("tl").to_numpy()
            cand = cand.with_columns(
                cos_name=rowdot(NS, NT, sl, tl), cos_addr=rowdot(AS, AT, sl, tl),
                rid=pl.Series(S_rid[sl]).cast(pl.Int32), gid=pl.Series(T_gid[tl]).cast(pl.Int32),
            ).drop("sl", "tl")
            cand.write_parquet(fc)
            mark_done(fc)
            log(f"candidates {country}: {cand.height:,} ({cand.height / max(1, len(q_loc)):.1f} per query S1)")


if __name__ == "__main__":
    main()
