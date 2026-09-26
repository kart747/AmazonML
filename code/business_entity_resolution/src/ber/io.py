"""Ingestion: challenge TSVs -> normalized Parquet tables with integer ids.

Per split (train/test) this writes to work/prep/:
  s1_{split}.parquet  rid, entity_id, name, addr, country, fold, nm, ad, cat
  tg_{split}.parquet  gid, entity_id, src (2/3), name, addr, country, nm, ad, cat
  gt_{split}.parquet  rid, gid  (train only; one row per true pair)
Integer ids (rid for S1, gid for S2+S3 targets) keep every later table compact.
"""
import polars as pl

from . import config
from .normalize import parallel_map
from .utils import done, log, mark_done, stage


def read_tsv(path):
    """Strict TSV read: tab separator, no quoting (6 train rows contain '"'), all strings."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Add nm/ad/cat views, normalizing each unique string once."""
    out = {}
    for col, fn, dst in (("business_name", "name", "nm"), ("business_address", "addr", "ad")):
        uniq = df.get_column(col).unique().to_list()
        normed = parallel_map(fn, uniq)
        m = pl.DataFrame({col: uniq, dst: normed})
        out[dst] = df.select(col).join(m, on=col, how="left", maintain_order="left").get_column(dst)
    return df.with_columns(nm=out["nm"], ad=out["ad"]).with_columns(cat=pl.col("nm").str.replace_all(" ", ""))


def prepare(split: str):
    pdir = config.WORK_DIR / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    f1, ft = pdir / f"s1_{split}.parquet", pdir / f"tg_{split}.parquet"
    if done(f1) and done(ft):
        log(f"prepare {split}: cached")
        return
    d = config.DATA_DIR / split
    with stage(f"prepare_{split}"):
        s1 = read_tsv(d / f"{split}_source1.tsv")
        s1 = s1.with_row_index("rid").with_columns(
            fold=(pl.col("entity_id").str.slice(3).cast(pl.Int64) % config.N_FOLDS).cast(pl.Int8)
            if split == "train" else pl.lit(-1, pl.Int8))
        tg = pl.concat([read_tsv(d / f"{split}_source2.tsv").with_columns(src=pl.lit(2, pl.Int8)),
                        read_tsv(d / f"{split}_source3.tsv").with_columns(src=pl.lit(3, pl.Int8))])
        tg = tg.with_row_index("gid")
        for name, df in (("s1", s1), ("tg", tg)):
            df = df.with_columns(pl.col("country").str.strip_chars())
            df = _normalize(df).rename({"business_name": "name", "business_address": "addr"})
            path = pdir / f"{name}_{split}.parquet"
            df.write_parquet(path)
            mark_done(path)
            log(f"{name}_{split}: {df.height:,} rows; countries {df.get_column('country').value_counts().rows()}")
        if split == "train":
            gt = read_tsv(d / "train_ground_truth.tsv")
            gt = (gt.with_columns(t=pl.col("matched_entity_ids").str.split(",")).explode("t")
                    .filter(pl.col("t") != "")
                    .join(s1.select("rid", pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
                    .join(tg.select("gid", pl.col("entity_id").alias("t")), on="t")
                    .select("rid", "gid"))
            gpath = pdir / "gt_train.parquet"
            gt.write_parquet(gpath)
            mark_done(gpath)
            log(f"gt_train: {gt.height:,} true pairs")
