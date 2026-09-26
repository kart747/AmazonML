"""Cross-script supplementary retrieval view ("xs"): recover true matches whose target
name is written in a non-Latin script.

Why (measured on India train, query folds 0-3, EXPERIMENTS.md E-B3): base blocking misses
2.7% of true pairs, and 53% of the misses have an Indic-script target name (vs 23% of the
found pairs). anyascii transliterates "लक्ष्मी" to "lksmi", which shares almost no
character 3-grams with the S1 spelling "laxmi", so the name part of every view fails.

Method (label-free at inference; the dictionary is learned from training folds only):
  1. native->Latin token dictionary learned from aligned true pairs of TRAIN_FOLDS
     (reuses features_v2.learn_dictionary, so blocking and features share one dictionary)
  2. targets whose raw name contains a non-Latin letter get a mapped name (tokens
     replaced through the dictionary). Accented Latin (French, US diacritics) is NOT
     affected, so the view is a no-op for countries without such scripts
  3. per country: [sqrt(.5)*name char_wb-3 TF-IDF | sqrt(.5)*address word TF-IDF] and
     reverse retrieval target -> S1 top-k over ALL S1 of the country (same design as the
     base reverse view, which finds 97% of true pairs on its own)
  4. keep pairs whose S1 is a query S1 (train: QUERY_FOLDS, test: all)

Commands (all resumable; nothing here modifies the base artifacts):
  python src/experiments/xscript.py build   --split train|test
        -> work/block/<split>/xs_<country>.parquet   (rid, gid, r_xs, s_xs)
  python src/experiments/xscript.py measure [--country India]
        -> reports/xscript_recall.json: recall gain vs the base candidates (train, labels
           used for measurement only), candidates added per query S1
  python src/experiments/xscript.py merge   --split train|test --dst /path/work_xs
        -> an isolated work dir: prep/ and rev_* are symlinked to the base; cand_* =
           base ∪ xs, where new pairs get exact base-view cos_name/cos_addr. Then run the
           normal pipeline there:
           BER_WORK_DIR=/path/work_xs python src/run.py features
           BER_WORK_DIR=/path/work_xs python src/run.py evaluate --approach 2
Cost (16 threads): build ≈ 5 min per country with Indic names; merge ≈ 3 min per split.
Peak RAM ≈ 4 GB (the dictionary step reads the full train prep tables).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber import config  # noqa: E402
from ber.blocking import _sample, fit_transform, rowdot, topk  # noqa: E402
from ber.utils import done, log, mark_done, save_json, stage  # noqa: E402
from features_v2 import learn_dictionary  # noqa: E402

NONLATIN = r"[\p{L}--\p{Latin}]"   # a letter from a non-Latin script
K_XS = 8


def _vec(kind):
    B = config.BLOCK
    if kind == "name":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=B["name_max_df"],
                               sublinear_tf=True, lowercase=False, dtype=np.float32)
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=B["addr_max_df"],
                           sublinear_tf=True, lowercase=False, dtype=np.float32)


def get_dictionary():
    """Learned native->Latin token map, cached in work/reports/xscript_dictionary.parquet."""
    f = config.WORK_DIR / "reports" / "xscript_dictionary.parquet"
    if f.exists():
        d = pl.read_parquet(f)
    else:
        m = learn_dictionary(config.WORK_DIR)
        d = pl.DataFrame({"t": list(m), "s": list(m.values())}, schema={"t": pl.String, "s": pl.String})
        f.parent.mkdir(parents=True, exist_ok=True)
        d.write_parquet(f)
    return d.get_column("t").to_list(), d.get_column("s").to_list()


def map_names(nm: pl.Series, old, new) -> pl.Series:
    return (nm.str.split(" ").list.eval(pl.element().replace(old, new)).list.join(" ")) if old else nm


def build(split):
    W = config.WORK_DIR
    bdir = W / "block" / split
    bdir.mkdir(parents=True, exist_ok=True)
    old, new = get_dictionary()
    log(f"xs dictionary: {len(old):,} token maps")
    s1 = pl.read_parquet(W / "prep" / f"s1_{split}.parquet", columns=["rid", "country", "fold", "nm", "ad"])
    tg = (pl.scan_parquet(W / "prep" / f"tg_{split}.parquet")
            .filter(pl.col("name").str.contains(NONLATIN))
            .select("gid", "country", "nm", "ad").collect())
    tg = tg.with_columns(nm=map_names(tg.get_column("nm"), old, new))
    log(f"xs {split}: {tg.height:,} targets with non-Latin names; by country {tg.get_column('country').value_counts().rows()}")
    for country in sorted(tg.get_column("country").unique()):
        out = bdir / f"xs_{country}.parquet"
        if done(out):
            log(f"xs {split}/{country}: cached")
            continue
        s1c, tgc = s1.filter(pl.col("country") == country), tg.filter(pl.col("country") == country)
        if s1c.height == 0:
            continue
        with stage(f"xs_build_{split}_{country}", {"targets": tgc.height, "s1": s1c.height}):
            s_nm, t_nm = s1c.get_column("nm").to_list(), tgc.get_column("nm").to_list()
            s_ad, t_ad = s1c.get_column("ad").to_list(), tgc.get_column("ad").to_list()
            fs = config.BLOCK["fit_sample"]
            NS, NT = fit_transform(_vec("name"), _sample(s_nm + t_nm, fs), s_nm, t_nm)
            AS, AT = fit_transform(_vec("addr"), _sample(s_ad + t_ad, fs), s_ad, t_ad)
            h = np.float32(math.sqrt(0.5))
            CS, CT = sp.hstack([NS * h, AS * h]).tocsr(), sp.hstack([NT * h, AT * h]).tocsr()
            tq, sq, sc, rk = topk(CT, CS, K_XS)
            qmask = (s1c.get_column("fold").is_in(config.QUERY_FOLDS).to_numpy() if split == "train"
                     else np.ones(s1c.height, bool))
            keep = qmask[sq]
            res = pl.DataFrame({
                "rid": s1c.get_column("rid").to_numpy()[sq[keep]].astype(np.int32),
                "gid": tgc.get_column("gid").to_numpy()[tq[keep]].astype(np.int32),
                "r_xs": rk[keep], "s_xs": sc[keep]})
            res.write_parquet(out)
            mark_done(out)
            log(f"xs {split}/{country}: {res.height:,} pairs")


def measure(countries=None):
    """Recall gain of base ∪ xs over base, on train query folds (labels for measurement only)."""
    W = config.WORK_DIR
    bdir = W / "block" / "train"
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"]) \
        .filter(pl.col("fold").is_in(config.QUERY_FOLDS))
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet").join(s1, on="rid")
    rep = {}
    for xf in sorted(bdir.glob("xs_*.parquet")):
        c = xf.stem[3:]
        if countries and c not in countries:
            continue
        cf = bdir / f"cand_{c}.parquet"
        if not cf.exists():
            log(f"measure {c}: base candidates missing ({cf}), skipped")
            continue
        base = pl.read_parquet(cf, columns=["rid", "gid"]).with_columns(in_base=pl.lit(True))
        xs = pl.read_parquet(xf).with_columns(in_xs=pl.lit(True))
        g = gt.filter(pl.col("country") == c).select("rid", "gid")
        h = (g.join(base, on=["rid", "gid"], how="left").join(xs, on=["rid", "gid"], how="left")
               .with_columns(pl.col("in_base").fill_null(False), pl.col("in_xs").fill_null(False)))
        new_pairs = xs.join(base, on=["rid", "gid"], how="anti").height
        n_q = s1.filter(pl.col("country") == c).height
        by_k = {k: round(float((h.get_column("in_base") | (h.get_column("in_xs") & (h.get_column("r_xs") < k)).fill_null(False)).mean()), 5)
                for k in (1, 2, 4, 8)}
        rep[c] = {
            "true_pairs": h.height,
            "base_recall": round(float(h.get_column("in_base").mean()), 5),
            "union_recall_by_k": by_k,
            "recovered_pairs": int((h.get_column("in_xs") & ~h.get_column("in_base")).sum()),
            "base_misses": int((~h.get_column("in_base")).sum()),
            "xs_pairs": xs.height, "xs_new_pairs": new_pairs,
            "new_candidates_per_query_s1": round(new_pairs / max(1, n_q), 2),
        }
        log(f"xs measure {c}: {rep[c]}")
    save_json(rep, W / "reports" / "xscript_recall.json")
    return rep


def merge(split, dst: Path):
    """Isolated work dir with cand = base ∪ xs (new pairs get exact base-view cosines)."""
    W = config.WORK_DIR
    src_b, dst_b = W / "block" / split, dst / "block" / split
    dst_b.mkdir(parents=True, exist_ok=True)
    if not (dst / "prep").exists():
        os.symlink((W / "prep").resolve(), dst / "prep")
    # rev/cand files of countries without an xs file are linked unchanged
    for f in sorted(src_b.glob("rev_*.parquet")) + sorted(src_b.glob("cand_*.parquet")):
        c = f.stem.split("_", 1)[1]
        if f.name.startswith("cand_") and (src_b / f"xs_{c}.parquet").exists():
            continue
        for x in (f, Path(str(f) + ".done")):
            if x.exists() and not (dst_b / x.name).exists():
                os.symlink(x.resolve(), dst_b / x.name)
    s1 = pl.read_parquet(W / "prep" / f"s1_{split}.parquet", columns=["rid", "country", "fold", "nm", "ad"])
    tg = pl.read_parquet(W / "prep" / f"tg_{split}.parquet", columns=["gid", "country", "nm", "ad"])
    for xf in sorted(src_b.glob("xs_*.parquet")):
        c = xf.stem[3:]
        out = dst_b / f"cand_{c}.parquet"
        if done(out):
            continue
        with stage(f"xs_merge_{split}_{c}"):
            base = pl.read_parquet(src_b / f"cand_{c}.parquet")
            xs = pl.read_parquet(xf)
            new = xs.join(base.select("rid", "gid"), on=["rid", "gid"], how="anti")
            # refit the base blocking vectorizers exactly (same record order and sample seed as
            # blocking.block_country) so cosines of new pairs are on the same scale
            s1c, tgc = s1.filter(pl.col("country") == c), tg.filter(pl.col("country") == c)
            s_nm, t_nm = s1c.get_column("nm").to_list(), tgc.get_column("nm").to_list()
            s_ad, t_ad = s1c.get_column("ad").to_list(), tgc.get_column("ad").to_list()
            srow = pl.DataFrame({"rid": s1c.get_column("rid")}).with_row_index("sl")
            trow = pl.DataFrame({"gid": tgc.get_column("gid")}).with_row_index("tl")
            check = base.select("rid", "gid", "cos_name", "cos_addr").sample(min(2000, base.height), seed=0)

            def locate(df):
                d = df.join(srow, on="rid", how="left", maintain_order="left") \
                      .join(trow, on="gid", how="left", maintain_order="left")
                return d.get_column("sl").to_numpy(), d.get_column("tl").to_numpy()

            (sl, tl), (csl, ctl) = locate(new), locate(check)
            us, ut = np.unique(np.r_[sl, csl]), np.unique(np.r_[tl, ctl])
            fs = config.BLOCK["fit_sample"]
            vn, va = _vec("name"), _vec("addr")
            vn.fit(_sample(s_nm + t_nm, fs))
            va.fit(_sample(s_ad + t_ad, fs))
            NS, NT = vn.transform([s_nm[i] for i in us]).tocsr(), vn.transform([t_nm[i] for i in ut]).tocsr()
            AS, AT = va.transform([s_ad[i] for i in us]).tocsr(), va.transform([t_ad[i] for i in ut]).tocsr()
            cos = lambda X, Y, a, b: rowdot(X, Y, np.searchsorted(us, a), np.searchsorted(ut, b))
            # the refit must reproduce the stored base cosines, or new pairs would be on another scale
            err = max(np.abs(cos(NS, NT, csl, ctl) - check.get_column("cos_name").to_numpy()).max(),
                      np.abs(cos(AS, AT, csl, ctl) - check.get_column("cos_addr").to_numpy()).max())
            if err > 1e-4:
                raise RuntimeError(f"refit vectorizers do not reproduce base cosines (max err {err:.2e})")
            new = new.with_columns(cos_name=cos(NS, NT, sl, tl), cos_addr=cos(AS, AT, sl, tl))
            merged = pl.concat([base.join(xs.select("rid", "gid", "r_xs"), on=["rid", "gid"], how="left"),
                                new.drop("s_xs")], how="diagonal_relaxed")
            merged = merged.select(base.columns + ["r_xs"])
            merged.write_parquet(out)
            mark_done(out)
            log(f"xs merge {split}/{c}: base {base.height:,} + new {new.height:,} = {merged.height:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "measure", "merge"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--country", action="append", default=None)
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split)
    elif a.cmd == "measure":
        measure(a.country)
    else:
        if not a.dst:
            sys.exit("--dst is required for merge")
        merge(a.split, Path(a.dst).resolve())


if __name__ == "__main__":
    main()
