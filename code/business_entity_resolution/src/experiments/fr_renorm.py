"""E-G1: France-only re-normalization in an isolated work dir (test split only).

Why: France has no training rows. Its addresses end in a multi-token region ("Pays de la
Loire") in S1, while targets often carry the department instead ("Loire-Atlantique",
"Gironde", "Nord", "Pas-de-Calais"). In US/India the same slot is a one-token state code
that both sides share, so French true pairs look like address mismatches the model never
saw in training (token Jaccard of p>0.99 pairs: France 0.74 vs US 0.83). French names
also join with "et" where the normalizer maps "&" to "and".

Fix (France rows only; US/India and every train artifact are untouched):
  address: each comma segment that is a region or one of its departments -> one region
           code token (hdf / naq / pdl), mirroring the US/India state codes.
  name:    token "et" -> "and"; with --legal-alias also "compagnie" -> "cie" and
           "etablissements" -> "ets" (E-X11: the only legal-form aliases that swap inside
           France test pairs; legal forms themselves are NOT stripped, because in US train
           a different legal form on an otherwise equal name is 36% true vs 93% for the same
           form, so the form is evidence, not noise).
Candidates (rid, gid) are NOT re-blocked: blocking already uses max_df 0.05, which drops
these frequent tokens, and keeping the rows keeps featdn/dn_cos aligned. Only the
features that read nm/ad are rebuilt for France: feat, featx, feat2 (stage-1), s2lab.

usage (from code/business_entity_resolution):
  python src/experiments/fr_renorm.py setup   --src ../../work_xd --dst ../../work_fr
  BER_WORK_DIR=../../work_fr python src/run.py features --split test
  BER_WORK_DIR=../../work_fr python src/experiments/features_v2.py
  BER_WORK_DIR=../../work_fr BER_TAG=_xd BER_EXTRA=featdn python -c "...build_stage2_parts('test', ...)"
  BER_WORK_DIR=../../work_fr python src/experiments/stage2_lab.py build --split test --extra featx
  BER_WORK_DIR=../../work_fr python src/experiments/stage2_lab.py predict --variant all --out <dir>
  python src/experiments/fr_renorm.py compare --a ../../work_xd --b ../../work_fr
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber.normalize import norm_addr, norm_name  # noqa: E402
from ber.utils import log  # noqa: E402

REGION = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl",
}


def norm_addr_fr(raw: str) -> str:
    if not raw:
        return ""
    out = []
    for seg in raw.split(","):
        n = norm_addr(seg)
        out.append(REGION.get(n, n))
    return " ".join(x for x in out if x)


NAME_TOKENS = {"et": "and"}
LEGAL_ALIAS = {"compagnie": "cie", "etablissements": "ets"}


def norm_name_fr(raw: str, legal_alias: bool = False) -> str:
    m = {**NAME_TOKENS, **LEGAL_ALIAS} if legal_alias else NAME_TOKENS
    return " ".join(m.get(t, t) for t in norm_name(raw).split())


def _renorm(df: pl.DataFrame, legal_alias: bool = False) -> pl.DataFrame:
    fr = pl.col("country") == "France"
    sub = df.filter(fr)
    names = sub.get_column("name").unique().to_list()
    addrs = sub.get_column("addr").unique().to_list()
    nmap = pl.DataFrame({"name": names, "_nm": [norm_name_fr(x or "", legal_alias) for x in names]})
    amap = pl.DataFrame({"addr": addrs, "_ad": [norm_addr_fr(x or "") for x in addrs]})
    df = df.join(nmap, on="name", how="left", maintain_order="left").join(amap, on="addr", how="left", maintain_order="left")
    return df.with_columns(nm=pl.when(fr).then(pl.col("_nm")).otherwise(pl.col("nm")),
                           ad=pl.when(fr).then(pl.col("_ad")).otherwise(pl.col("ad"))).drop("_nm", "_ad")


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src.resolve())


def _link_parts(src_dir: Path, dst_dir: Path, skip_prefix="France"):
    """Symlink every file of src_dir except the France parts (and their .done markers)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.iterdir()):
        if not f.name.startswith(skip_prefix):
            _link(f, dst_dir / f.name)


def setup(src: Path, dst: Path, legal_alias: bool = False):
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    (dst / "prep").mkdir(parents=True)
    for f in (src / "prep").iterdir():
        if f.name.startswith(("s1_test", "tg_test")):
            continue
        _link(f, dst / "prep" / f.name)
    for name in ("s1_test", "tg_test"):
        df = _renorm(pl.read_parquet(src / "prep" / f"{name}.parquet"), legal_alias)
        df.write_parquet(dst / "prep" / f"{name}.parquet")
        (dst / "prep" / f"{name}.parquet.done").touch()
        log(f"{name}: France rows re-normalized")
    for d in ("block", "featdn", "dense", "models"):
        if (src / d).exists():
            _link(src / d, dst / d)
    for d in ("feat", "featx", "feat2_xd", "s2lab"):
        _link(src / d / "train", dst / d / "train")
        _link_parts(src / d / "test", dst / d / "test")
    (dst / "feat2").symlink_to("feat2_xd")
    (dst / "reports").mkdir()
    for f in ("s2lab_all.json", "featx_dictionary.json", "eval_A2_xd.json"):
        if (src / "reports" / f).exists():
            shutil.copy(src / "reports" / f, dst / "reports" / f)
    log(f"work dir {dst} ready: France feat/featx/feat2/s2lab test parts to be rebuilt")


def compare(a: Path, b: Path):
    bins = [0.001, 0.01, 0.05, 0.2, 0.5, 0.68, 0.9, 0.99, 0.999]
    s1 = pl.read_parquet(a / "prep" / "s1_test.parquet", columns=["rid", "country"])
    n = s1.group_by("country").len("n")
    out = []
    for tag, w in (("before", a), ("after", b)):
        t = pl.read_parquet(w / "reports" / "test_s2lab_all_pairs.parquet").join(s1, on="rid")
        g = (t.with_columns(b=pl.col("p").cut(bins)).group_by(["country", "b"]).len()
              .join(n, on="country").with_columns((pl.col("len") / pl.col("n")).round(4).alias(tag))
              .select("country", "b", tag))
        out.append(g)
        k = t.filter(pl.col("keep")).group_by("country").len("kept").join(n, on="country")
        log(f"{tag}: kept per S1 " + str(k.with_columns((pl.col("kept") / pl.col("n")).round(4)).rows()))
    with pl.Config(tbl_rows=40):
        print(out[0].join(out[1], on=["country", "b"], how="full", coalesce=True).sort(["country", "b"]))
    ta = pl.read_parquet(a / "reports" / "test_s2lab_all_pairs.parquet").join(s1, on="rid").filter(pl.col("country") == "France")
    tb = pl.read_parquet(b / "reports" / "test_s2lab_all_pairs.parquet").join(s1, on="rid").filter(pl.col("country") == "France")
    j = ta.join(tb, on=["rid", "gid"], how="full", suffix="_b", coalesce=True)
    log(f"France pairs: before {ta.height:,} after {tb.height:,}; kept before {ta['keep'].sum():,} after {tb['keep'].sum():,}; "
        f"kept in both {int((j['keep'].fill_null(False) & j['keep_b'].fill_null(False)).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "compare"])
    ap.add_argument("--src", default="../../work_xd")
    ap.add_argument("--dst", default="../../work_fr")
    ap.add_argument("--a", default="../../work_xd")
    ap.add_argument("--b", default="../../work_fr")
    ap.add_argument("--legal-alias", action="store_true", help="also map compagnie->cie, etablissements->ets")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(Path(a.src), Path(a.dst), a.legal_alias)
    else:
        compare(Path(a.a), Path(a.b))


if __name__ == "__main__":
    os.environ.setdefault("POLARS_MAX_THREADS", "8")
    main()
