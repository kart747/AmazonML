"""Re-score saved stage-2 test parts with per-country feature rescaling (no retraining).

Use case (E-X9): q_nm_freq / t_nm_freq are rates per 1M S1 of the country IN THE SPLIT, so the same
name count maps to a different value when the S1 pool is smaller (US test 0.50x train, France 259k).
Multiplying a country's test values by n_test/n_train_ref puts them back on the count scale the
model was trained on.

    BER_WORK_DIR=../work_xd python src/experiments/rescore.py --model s2lab_dense_mix --tau 0.68 \
        --scale "US:q_nm_freq,t_nm_freq=0.501" --scale "France:q_nm_freq,t_nm_freq=0.196" \
        --tag _frq --out ../output_variants/xd_dense_mix_frq
Writes reports/test_<model><tag>_pairs.parquet and a submission (candidates = scored pairs, subset-checked).
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ber.approaches as ap  # noqa: E402
import redecode  # noqa: E402
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.utils import log, save_json  # noqa: E402


def parse_scale(specs):
    out = {}   # country -> {feature: factor}
    for s in specs:
        country, rest = s.split(":", 1)
        feats, factor = rest.rsplit("=", 1)
        for f in feats.split(","):
            out.setdefault(country, {})[f] = float(factor)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True)
    a.add_argument("--tau", type=float, required=True)
    a.add_argument("--variant", default="threshold+excl")
    a.add_argument("--scale", action="append", default=[], help='"Country:feat1,feat2=factor"')
    a.add_argument("--tag", required=True)
    a.add_argument("--out", required=True)
    x = a.parse_args()
    WD = config.WORK_DIR
    m = lgb.Booster(model_file=str(WD / "models" / f"{x.model}.txt"))
    feats = m.feature_name()
    scale = parse_scale(x.scale)
    for c, fs in scale.items():
        for f in fs:
            assert f in feats, f"{f} is not a feature of {x.model}"
    metas, ps = [], []
    for part in sorted((WD / s2.OUT / "test").glob("*.parquet")):
        country = part.name.rsplit("_", 1)[0]
        d = pl.read_parquet(part, columns=["rid", "gid"] + feats)
        if country in scale:
            d = d.with_columns([(pl.col(f) * k).cast(pl.Float32) for f, k in scale[country].items()])
        ps.append(ap._pred(m, np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)))
        metas.append(d.select("rid", "gid"))
    meta, p = pl.concat(metas), np.concatenate(ps)
    keep = ap.decide(p, meta.get_column("gid").to_numpy(), x.variant, x.tau)
    pairs = meta.with_columns(p=pl.Series(p), keep=pl.Series(keep))
    name = f"{x.model}{x.tag}"
    pairs.write_parquet(WD / "reports" / f"test_{name}_pairs.parquet")
    s1t = pl.read_parquet(WD / "prep" / "s1_test.parquet", columns=["rid", "country"])
    by = pairs.join(s1t, on="rid").group_by("country").agg(k=pl.col("keep").sum()).join(s1t.group_by("country").len("n"), on="country")
    per = {r["country"]: round(r["k"] / r["n"], 4) for r in by.iter_rows(named=True)}
    config.OUT_DIR = Path(x.out).resolve()
    ap.write_submission(pairs.select("rid", "gid"), keep)
    bad = redecode.subset_check(config.OUT_DIR)
    dec = {"model": x.model, "variant": x.variant, "tau": x.tau, "scale": scale, "matched_pairs": int(keep.sum()),
           "pred_per_s1": per, "matched_not_in_candidates": bad}
    save_json(dec, config.OUT_DIR / "decision.json")
    log(f"rescore {name}: {json.dumps(dec)}")


if __name__ == "__main__":
    main()
