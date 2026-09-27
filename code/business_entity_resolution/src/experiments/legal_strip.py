"""E-G7: legal-form stripping for an unseen country (label-free transfer fix).

Why (E-G5/E-G6): a stage 2 trained on another country misses obvious true pairs whose names
differ only by a legal form or its abbreviation ("Pvt. Ltd." vs "Pvt. Limited", "X" vs
"X SNC"). norm_name keeps these tokens, and whether a one-sided legal token is benign is
only learned from labels of that country. Stripping the country's legal-form tokens from the
normalized name (both sides) removes that dependency.

Setup: an isolated work dir where only the target country's normalized names (`nm`) of one
split are re-derived; blocking (candidates, cos_*, reverse ranks) and dense features are
reused unchanged, so featdn stays row-aligned. Every other country's parts are symlinked, and
the target country's feat / featx / feat2<tag> / s2lab parts are left to be rebuilt:

  python src/experiments/legal_strip.py setup --src ../../work_loco --dst ../../work_lstrip \
      --split train --country India --feat2 feat2_loco
  BER_WORK_DIR=../../work_lstrip python src/run.py features --split train
  BER_WORK_DIR=../../work_lstrip python src/experiments/features_v2.py
  BER_WORK_DIR=../../work_lstrip BER_TAG=_loco BER_EXTRA=featdn python -c "...build_stage2_parts('train', ...)"
  BER_WORK_DIR=../../work_lstrip BER_S2LAB_SRC=feat2_loco python src/experiments/stage2_lab.py build --split train --extra featx
"""
import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber.utils import log  # noqa: E402

LEGAL = {
    "India": {"pvt", "private", "ltd", "limited", "llp", "co", "company", "corp", "corporation",
              "inc", "incorporated", "pvtltd"},
    "France": {"sarl", "sas", "sasu", "sa", "snc", "sci", "eurl", "scop", "gie", "selarl", "scp", "scm",
               "associes", "fils", "cie", "compagnie", "et", "and"},
    "US": {"llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
           "lp", "llp", "pllc", "pc"},
}


def strip_name(nm: str, legal: set) -> str:
    toks = [t for t in nm.split() if t not in legal]
    return " ".join(toks) if toks else nm


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src.resolve())


def _link_dir_except(src_dir: Path, dst_dir: Path, country: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.iterdir()):
        if not f.name.startswith(country):
            _link(f, dst_dir / f.name)


def setup(src: Path, dst: Path, split: str, country: str, feat2: str):
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    legal = LEGAL[country]
    (dst / "prep").mkdir(parents=True)
    for f in (src / "prep").iterdir():
        if not f.name.startswith((f"s1_{split}", f"tg_{split}")):
            _link(f, dst / "prep" / f.name)
    for name in (f"s1_{split}", f"tg_{split}"):
        df = pl.read_parquet(src / "prep" / f"{name}.parquet")
        c = pl.col("country") == country
        u = df.filter(c).get_column("nm").unique()
        m = pl.DataFrame({"nm": u, "_nm": [strip_name(x or "", legal) for x in u.to_list()]})
        n_chg = int((m.get_column("nm") != m.get_column("_nm")).sum())
        df = (df.join(m, on="nm", how="left", maintain_order="left")
                .with_columns(nm=pl.when(c).then(pl.col("_nm")).otherwise(pl.col("nm"))).drop("_nm"))
        df.write_parquet(dst / "prep" / f"{name}.parquet")
        (dst / "prep" / f"{name}.parquet.done").touch()
        log(f"{name}: {country} names stripped ({n_chg:,} of {len(u):,} distinct names changed)")
    for d in ("block", "dense", "featdn"):
        if (src / d).exists():
            _link(src / d, dst / d)
    (dst / "models").mkdir()
    for f in (src / "models").iterdir():
        _link(f, dst / "models" / f.name)
    other = {"train": "test", "test": "train"}[split]
    for d in ("feat", "featx", feat2, "s2lab"):
        if (src / d / other).exists():
            _link(src / d / other, dst / d / other)
        if (src / d / split).exists():
            _link_dir_except(src / d / split, dst / d / split, country)
    (dst / "reports").mkdir()
    for f in ("s2lab_all.json", "featx_dictionary.json"):   # copies: later stages rewrite reports
        if (src / "reports" / f).exists():
            shutil.copy(src / "reports" / f, dst / "reports" / f)
    log(f"{dst} ready: rebuild {split} {country} feat/featx/{feat2}/s2lab parts")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["setup"])
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    a.add_argument("--split", default="train")
    a.add_argument("--country", default="India")
    a.add_argument("--feat2", default="feat2_loco")
    x = a.parse_args()
    setup(Path(x.src), Path(x.dst), x.split, x.country, x.feat2)


if __name__ == "__main__":
    main()
