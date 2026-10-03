"""E-SC1: sibling-conflict features (idea 3), smallest useful set.

Writes a row-aligned extra feature set for the built stage-2 parts, to be consumed
by `stage2_lab.py --variant all` through the existing `BER_S2LAB_OUT` mechanism.
No house-number feature and no pool-level address-degree feature (E-X8) is used.

  sc_nmatch    other candidates of this S1 with n_tset >= 0.99  (name twins)
  sc_amatch    other candidates of this S1 with a_tset >= 0.99  (address twins)
  sc_conflict  other candidates with n_tset >= 0.99 and a_tset < 0.9
               (name says yes, address says no -> the sibling conflict)

Usage:  python src/experiments/sibconf.py [--split train]
Output: work_xd/<BER_S2LAB_OUT>_sc/<split>/<same>.parquet
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.utils import log, stage  # noqa: E402

NEW = ["sc_nmatch", "sc_amatch", "sc_conflict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    src = config.WORK_DIR / "s2lab" / a.split
    dst = config.WORK_DIR / "s2lab_sc" / a.split
    dst.mkdir(parents=True, exist_ok=True)
    with stage(f"sibconf_{a.split}"):
        for p in sorted(src.glob("*.parquet")):
            out = dst / p.name
            if out.exists():
                continue
            df = pl.read_parquet(p)
            n = (pl.col("n_tset") >= 0.99).fill_null(False)
            ad = (pl.col("a_tset") >= 0.99).fill_null(False)
            conf = n & ~(ad)
            df = df.with_columns(
                sc_nmatch=(n.sum().over("rid") - n.cast(pl.Int32)).cast(pl.Float32),
                sc_amatch=(ad.sum().over("rid") - ad.cast(pl.Int32)).cast(pl.Float32),
                sc_conflict=(conf.sum().over("rid") - conf.cast(pl.Int32)).cast(pl.Float32))
            df.write_parquet(out)
            log(f"sibconf {a.split}/{p.stem}: {df.height:,} rows, "
                f"{df.get_column('sc_conflict').cast(pl.Int64).sum():,} conflict slots")


if __name__ == "__main__":
    main()
