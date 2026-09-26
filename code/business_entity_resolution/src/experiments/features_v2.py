"""Features v2: extra pair features for existing candidates (no re-blocking).

Writes work/featx/<split>/<part>.parquet, row-aligned with work/feat/<split>/<part>.parquet
(rid, gid, fold + the columns below). Enable in any matcher with BER_EXTRA=featx and give
the run its own BER_TAG, e.g.

    python src/experiments/features_v2.py                       # train + test
    BER_EXTRA=featx BER_TAG=_fx python src/run.py evaluate --approach 2

Features (each targets an observed error type):
  n_map_tset, n_map_ratio   name similarity after a learned native->Latin token dictionary
                            (cross-script Indian names; learned from true pairs of the
                            TRAINING folds only, so tune/eval scores are not leaked)
  n_skel_ratio, n_skel_tset consonant-skeleton similarity (transliteration/typos: "praivet"
                            ~ "private" -> "prvt")
  q_unm_maxidf, t_unm_maxidf  IDF of the rarest name token present on one side only (a sibling
                            lookalike typically differs by one distinctive word)
  q_match_idf, t_match_idf  share of each side's name IDF mass that is matched
  core_inter, core_jacc, q_core_n, t_core_n
                            overlap of "core" name tokens (document frequency <= 1% within
                            the split, learned, so generic words like pvt/llc/sarl drop out)
  num_symdiff, num_reldiff  address-number conflicts (count of unshared numbers; relative
                            gap between first numbers: 245 vs 2454 -> large, 340 vs 345 -> small)
  num_q_extra, num_t_extra  numbers present on only one side
  hn_*                      house-number (first number) edit type. True copies show truncation,
                            padding or suffix noise (2454->245, 1014->014); planted lookalike
                            distractors substitute digits in a same-length number (340->345).
                            On test (≈2x distractors per S1) these lookalikes drive false merges
                            (RESEARCH_DESIGN.md §2.6).
"""
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.utils import done, log, mark_done, save_json, stage  # noqa: E402

OUT = "featx"
FEATS = ["n_map_tset", "n_map_ratio", "n_skel_ratio", "n_skel_tset", "q_unm_maxidf", "t_unm_maxidf",
         "q_match_idf", "t_match_idf", "core_inter", "core_jacc", "q_core_n", "t_core_n",
         "num_symdiff", "num_reldiff", "num_q_extra", "num_t_extra",
         "hn_prefix", "hn_suffix", "hn_lev", "hn_samelen", "hn_subst_small"]
CORE_DF = 0.01


_VOWEL = re.compile(r"\B[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


def skel(s: str) -> str:
    """Consonant skeleton per token: keep first letter, drop later vowels, collapse repeats."""
    return _REPEAT.sub(r"\1", _VOWEL.sub("", s))


def skeleton_col(df, col):
    u = df.get_column(col).unique()
    m = pl.DataFrame({col: u, "_sk": [skel(x or "") for x in u.to_list()]})
    return df.join(m, on=col, how="left", maintain_order="left").get_column("_sk")


def learn_dictionary(W):
    """Native-script-token (after anyascii) -> S1 token, from aligned true pairs of TRAIN_FOLDS."""
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "nm"])
    tg = pl.read_parquet(W / "prep" / "tg_train.parquet", columns=["gid", "name", "nm"])
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet")
    d = (gt.join(s1.filter(pl.col("fold").is_in(list(config.TRAIN_FOLDS))), on="rid")
           .join(tg.filter(~pl.col("name").str.contains(r"^[\x00-\x7F]*$")), on="gid", suffix="_t")
           .select(s=pl.col("nm").str.split(" "), t=pl.col("nm_t").str.split(" "))
           .filter(pl.col("s").list.len() == pl.col("t").list.len())
           .explode(["s", "t"]).filter(pl.col("s") != pl.col("t")))
    tot = d.group_by("t").len("n_t")
    m = (d.group_by(["t", "s"]).len("n").join(tot, on="t")
          .filter((pl.col("n") >= 3) & (pl.col("n") / pl.col("n_t") >= 0.5))
          .sort("n", descending=True).unique("t", keep="first"))
    log(f"learned dictionary: {m.height:,} token maps from {d.height:,} aligned token pairs; "
        f"examples {m.head(8).select('t', 's', 'n').rows()}")
    return dict(zip(m.get_column("t").to_list(), m.get_column("s").to_list()))


def token_idf(W, split):
    s1 = pl.read_parquet(W / "prep" / f"s1_{split}.parquet", columns=["country", "nm"])
    tg = pl.read_parquet(W / "prep" / f"tg_{split}.parquet", columns=["country", "nm"])
    recs = pl.concat([s1, tg]).with_row_index("r")
    n = recs.group_by("country").len("N")
    df = (recs.select("r", "country", tok=pl.col("nm").str.split(" ")).explode("tok").unique(["r", "tok"])
              .group_by(["country", "tok"]).len("df").join(n, on="country")
              .with_columns(idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32),
                            core=(pl.col("df") / pl.col("N") <= CORE_DF))
              .select("country", "tok", "idf", "core"))
    return df


def _side(ex_a, ex_b, prefix):
    """Per-row aggregates of side A tokens w.r.t. side B."""
    j = ex_a.join(ex_b.select("row", "tok", m=pl.lit(True)), on=["row", "tok"], how="left") \
        .with_columns(pl.col("m").fill_null(False))
    return j.group_by("row").agg(
        **{f"{prefix}_unm_maxidf": pl.col("idf").filter(~pl.col("m")).max().fill_null(0.0),
           f"{prefix}_match_idf": pl.col("idf").filter(pl.col("m")).sum() / pl.col("idf").sum().clip(1e-6),
           f"{prefix}_core_n": pl.col("core").sum(),
           f"_{prefix}_core_inter": (pl.col("core") & pl.col("m")).sum()})


def build(split, mapping):
    W = config.WORK_DIR
    src_dir, out_dir = W / "feat" / split, W / OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(W / "prep" / f"s1_{split}.parquet", columns=["rid", "country", "nm", "ad"])
    tg = pl.read_parquet(W / "prep" / f"tg_{split}.parquet", columns=["gid", "nm", "ad"])
    idf = token_idf(W, split)
    max_idf = idf.group_by("country").agg(mx=pl.col("idf").max())
    old, new = list(mapping), list(mapping.values())
    for p in sorted(src_dir.glob("*.parquet")):
        dst = out_dir / p.name
        if done(dst):
            continue
        base = pl.read_parquet(p, columns=["rid", "gid", "fold"]).with_row_index("row")
        df = (base.join(s1, on="rid", how="left", maintain_order="left")
                  .join(tg.rename({"nm": "t_nm", "ad": "t_ad"}), on="gid", how="left", maintain_order="left"))
        df = df.with_columns(q_nt=pl.col("nm").str.split(" "), t_nt=pl.col("t_nm").str.split(" "))
        df = df.with_columns(t_map=pl.col("t_nt").list.eval(pl.element().replace(old, new)).list.join(" "))
        df = df.with_columns(q_sk=skeleton_col(df, "nm"), t_sk=skeleton_col(df, "t_nm"))
        qn, tm = df.get_column("nm").to_list(), df.get_column("t_map").to_list()
        qs, ts = df.get_column("q_sk").to_list(), df.get_column("t_sk").to_list()
        cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32) / np.float32(100)
        df = df.with_columns(n_map_tset=cp(qn, tm, fuzz.token_set_ratio), n_map_ratio=cp(qn, tm, fuzz.ratio),
                             n_skel_ratio=cp(qs, ts, fuzz.ratio), n_skel_tset=cp(qs, ts, fuzz.token_set_ratio))
        # token IDF / core aggregates (per country IDF; unknown tokens get the max IDF)
        ex = lambda col: (df.select("row", "country", tok=pl.col(col)).explode("tok").unique(["row", "tok"])
                            .join(idf, on=["country", "tok"], how="left").join(max_idf, on="country", how="left")
                            .with_columns(idf=pl.col("idf").fill_null(pl.col("mx")), core=pl.col("core").fill_null(True)))
        eq, et = ex("q_nt"), ex("t_nt")
        agg = _side(eq, et, "q").join(_side(et, eq, "t"), on="row", how="full", coalesce=True)
        df = df.join(agg, on="row", how="left", maintain_order="left").with_columns(
            core_inter=pl.col("_q_core_inter"),
            core_jacc=pl.col("_q_core_inter") / (pl.col("q_core_n") + pl.col("t_core_n") - pl.col("_q_core_inter")).clip(1))
        num = lambda c: pl.col(c).str.extract_all(r"\d+")
        first = lambda c: pl.col(c).list.first().str.slice(0, 9).cast(pl.Float64, strict=False)
        df = df.with_columns(qn_=num("ad"), tn_=num("t_ad")).with_columns(
            num_symdiff=pl.col("qn_").list.set_symmetric_difference(pl.col("tn_")).list.len(),
            num_reldiff=(first("qn_") - first("tn_")).abs() / pl.max_horizontal(first("qn_"), first("tn_")).clip(1),
            num_q_extra=pl.col("qn_").list.set_difference(pl.col("tn_")).list.len(),
            num_t_extra=pl.col("tn_").list.set_difference(pl.col("qn_")).list.len(),
            h1=pl.col("qn_").list.first(), h2=pl.col("tn_").list.first())
        both = pl.col("h1").is_not_null() & pl.col("h2").is_not_null()
        neq = pl.col("h1") != pl.col("h2")
        df = df.with_columns(
            hn_prefix=pl.when(both).then((pl.col("h1").str.starts_with(pl.col("h2")) | pl.col("h2").str.starts_with(pl.col("h1"))) & neq),
            hn_suffix=pl.when(both).then((pl.col("h1").str.ends_with(pl.col("h2")) | pl.col("h2").str.ends_with(pl.col("h1"))) & neq),
            hn_samelen=pl.when(both).then(pl.col("h1").str.len_chars() == pl.col("h2").str.len_chars()),
            hn_lev=pl.Series(process.cpdist(df.get_column("h1").fill_null("").to_list(), df.get_column("h2").fill_null("").to_list(),
                                            scorer=Levenshtein.distance, workers=-1)).cast(pl.Float32))
        dnum = (first("h1_l") - first("h2_l")).abs()
        df = df.with_columns(h1_l=pl.concat_list("h1"), h2_l=pl.concat_list("h2")).with_columns(
            hn_lev=pl.when(both).then(pl.col("hn_lev")),
            hn_subst_small=pl.when(both).then(pl.col("hn_samelen") & neq & (dnum <= 10)))
        out = df.sort("row").select("rid", "gid", "fold", *[pl.col(f).cast(pl.Float32) for f in FEATS])
        assert out.get_column("gid").equals(base.get_column("gid")), "row alignment broken"
        out.write_parquet(dst)
        mark_done(dst)
        log(f"featx {split}/{p.name}: {out.height:,} pairs")


def main():
    W = config.WORK_DIR
    mapping = learn_dictionary(W)
    save_json({"n_maps": len(mapping), "sample": dict(list(mapping.items())[:50])}, W / "reports" / "featx_dictionary.json")
    for split in ("train", "test"):
        if (W / "feat" / split).exists():
            with stage(f"features_v2_{split}"):
                build(split, mapping)


if __name__ == "__main__":
    main()
