"""Pair features for every candidate pair, computed in bounded-memory chunks.

PAIR_FEATURES (Approach 1 and 2): functions of the two records only.
CONTEXT_FEATURES (Approach 2 only): label-free context from the candidate graph:
  forward retrieval ranks, the target's reverse list (does the target prefer a
  different S1? = many-to-one competition), rank/gap inside the S1's own list, and
  name ambiguity (how many S1 share this normalized name).

Output: work/feat/<split>/<country>_<part>.parquet with rid, gid, fold, y (train), features.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config
from .utils import done, log, mark_done, stage

PAIR_FEATURES = [
    "cos_name", "cos_addr", "cos_combo",
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "cat_ratio",
    "a_ratio", "a_tset", "a_partial",
    "n_jacc", "n_inter", "a_jacc", "a_inter",
    "num_jacc", "num_inter", "num_first_eq", "num_first_logdiff", "q_num_n", "t_num_n",
    "q_ntok", "t_ntok", "q_alen", "t_alen", "t_addr_empty", "t_src", "t_nonascii",
]
CONTEXT_FEATURES = [
    "r_name", "r_addr", "r_combo", "r_rev",
    "t_top1", "t_gap12", "rev_margin", "t_is_best",
    "q_rank", "q_gap", "q_ncand",
    "q_nm_freq", "t_nm_freq",
]
CHUNK_PAIRS = 4_000_000


def _fuzz(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / np.float32(100)


def _rev_summary(rev: pl.DataFrame) -> pl.DataFrame:
    """Per target: best S1 (rid, score) and runner-up score from its reverse list."""
    return (rev.sort(["gid", "rrank"])
               .group_by("gid", maintain_order=True)
               .agg(t_best_rid=pl.col("rid").first(), t_top1=pl.col("rscore").first(),
                    t_top2=pl.col("rscore").get(1, null_on_oob=True)))


def _pair_block(df: pl.DataFrame) -> pl.DataFrame:
    """Pure pairwise features. df has q_*/t_* string columns."""
    qn, tn = df.get_column("q_nm").to_list(), df.get_column("t_nm").to_list()
    qa, ta = df.get_column("q_ad").to_list(), df.get_column("t_ad").to_list()
    qc, tc = df.get_column("q_cat").to_list(), df.get_column("t_cat").to_list()
    df = df.with_columns(
        n_ratio=_fuzz(qn, tn, fuzz.ratio), n_tsort=_fuzz(qn, tn, fuzz.token_sort_ratio),
        n_tset=_fuzz(qn, tn, fuzz.token_set_ratio), n_partial=_fuzz(qn, tn, fuzz.partial_ratio),
        n_jw=_fuzz(qn, tn, JaroWinkler.normalized_similarity) * np.float32(100),
        cat_ratio=_fuzz(qc, tc, fuzz.ratio),
        a_ratio=_fuzz(qa, ta, fuzz.ratio), a_tset=_fuzz(qa, ta, fuzz.token_set_ratio),
        a_partial=_fuzz(qa, ta, fuzz.partial_ratio),
    )
    tok = lambda c: pl.col(c).str.split(" ")
    num = lambda c: pl.col(c).str.extract_all(r"\d+")
    df = df.with_columns(q_nt=tok("q_nm"), t_nt=tok("t_nm"), q_at=tok("q_ad"), t_at=tok("t_ad"),
                         q_num=num("q_ad"), t_num=num("t_ad"))
    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = lambda a, b: pl.col(a).list.set_union(pl.col(b)).list.len()
    first_num = lambda c: pl.col(c).list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
    df = df.with_columns(
        n_inter=inter("q_nt", "t_nt"), n_jacc=inter("q_nt", "t_nt") / union("q_nt", "t_nt"),
        a_inter=inter("q_at", "t_at"), a_jacc=inter("q_at", "t_at") / union("q_at", "t_at").clip(1),
        num_inter=inter("q_num", "t_num"),
        num_jacc=inter("q_num", "t_num") / union("q_num", "t_num").clip(1),
        num_first_eq=(pl.col("q_num").list.first() == pl.col("t_num").list.first()).cast(pl.Float32),
        num_first_logdiff=(first_num("q_num") - first_num("t_num")).abs().cast(pl.Float64).log1p(),
        q_num_n=pl.col("q_num").list.len(), t_num_n=pl.col("t_num").list.len(),
        q_ntok=pl.col("q_nt").list.len(), t_ntok=pl.col("t_nt").list.len(),
        q_alen=pl.col("q_ad").str.len_chars(), t_alen=pl.col("t_ad").str.len_chars(),
        t_addr_empty=(pl.col("t_ad") == "").cast(pl.Float32),
        t_nonascii=(~pl.col("t_name").str.contains(r"^[\x00-\x7F]*$")).cast(pl.Float32),
        cos_combo=(pl.col("cos_name") + pl.col("cos_addr")) * 0.5,
    )
    return df


def _context_block(df: pl.DataFrame) -> pl.DataFrame:
    """Label-free candidate-graph context (Approach 2)."""
    df = df.with_columns(
        [pl.col(f"r_{v}").fill_null(99).cast(pl.Float32) for v in ("name", "addr", "combo", "rev")]
    ).with_columns(
        t_is_best=(pl.col("rid") == pl.col("t_best_rid")).cast(pl.Float32),
        t_gap12=(pl.col("t_top1") - pl.col("t_top2").fill_null(0)),
    ).with_columns(
        # margin of this S1 over the best *other* S1 competing for the same target
        rev_margin=pl.when(pl.col("t_is_best") == 1)
        .then(pl.col("cos_combo") - pl.col("t_top2").fill_null(0))
        .otherwise(pl.col("cos_combo") - pl.col("t_top1").fill_null(0)),
        q_rank=pl.col("cos_combo").rank("ordinal", descending=True).over("rid").cast(pl.Float32),
        q_gap=pl.col("cos_combo").max().over("rid") - pl.col("cos_combo"),
        q_ncand=pl.len().over("rid").cast(pl.Float32),
    )
    return df


def build_features(split: str):
    pdir, bdir = config.WORK_DIR / "prep", config.WORK_DIR / "block" / split
    fdir = config.WORK_DIR / "feat" / split
    fdir.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(pdir / f"s1_{split}.parquet", columns=["rid", "fold", "country", "nm", "ad", "cat"])
    tg = pl.read_parquet(pdir / f"tg_{split}.parquet", columns=["gid", "src", "name", "nm", "ad", "cat"])
    # name ambiguity: number of S1 records (same country) sharing a normalized name
    freq = s1.group_by(["country", "nm"]).len("nm_freq")
    gt = pl.read_parquet(pdir / "gt_train.parquet").with_columns(y=pl.lit(1, pl.Int8)) if split == "train" else None

    for cf in sorted(bdir.glob("cand_*.parquet")):
        country = cf.stem[len("cand_"):]
        if done(fdir / f"{country}"):
            log(f"features {split}/{country}: cached")
            continue
        with stage(f"features_{split}_{country}"):
            cand = pl.read_parquet(cf)
            revs = _rev_summary(pl.read_parquet(bdir / f"rev_{country}.parquet"))
            s1c = s1.filter(pl.col("country") == country).join(
                freq.filter(pl.col("country") == country).drop("country"), on="nm", how="left")
            cand = cand.sort("rid")
            rids = cand.get_column("rid").to_numpy()
            # chunk boundaries on S1 changes so per-S1 context features stay complete
            bounds = [0]
            while bounds[-1] < len(rids):
                e = min(bounds[-1] + CHUNK_PAIRS, len(rids))
                while e < len(rids) and rids[e] == rids[e - 1]:
                    e += 1
                bounds.append(e)
            for part, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
                df = (cand.slice(a, b - a)
                      .join(s1c.select("rid", "fold", q_nm="nm", q_ad="ad", q_cat="cat", q_nm_freq="nm_freq"),
                            on="rid", how="left")
                      .join(tg.select("gid", t_src="src", t_name="name", t_nm="nm", t_ad="ad", t_cat="cat"),
                            on="gid", how="left")
                      .join(revs, on="gid", how="left"))
                df = df.join(freq.filter(pl.col("country") == country).select(t_nm="nm", t_nm_freq="nm_freq"),
                             on="t_nm", how="left").with_columns(pl.col("t_nm_freq").fill_null(0))
                df = _context_block(_pair_block(df))
                if gt is not None:
                    df = df.join(gt, on=["rid", "gid"], how="left").with_columns(pl.col("y").fill_null(0))
                keep = ["rid", "gid", "fold"] + (["y"] if gt is not None else []) + PAIR_FEATURES + CONTEXT_FEATURES
                df.select(keep).with_columns(
                    [pl.col(c).cast(pl.Float32) for c in PAIR_FEATURES + CONTEXT_FEATURES]
                ).write_parquet(fdir / f"{country}_{part:03d}.parquet")
                log(f"features {split}/{country} part {part}: {df.height:,} pairs")
            mark_done(fdir / f"{country}")
