"""Leave-one-country-out (LOCO): the local proxy for the unseen test country (France).

Train on the training folds of country A, choose tau on A's tune fold (France labels
never exist, so tau cannot be tuned on the target country), report on the eval fold of
country B. Also reports B in-domain (a model trained on B) and B's oracle tau, which
gives (a) the unseen-country F0.5 gap and (b) the tau shift for unseen countries.

Usage (after `run.py features --split train`):
    python src/experiments/loco.py --features context   # PAIR+CONTEXT (A2 stage-1 inputs)
    python src/experiments/loco.py --features pair      # PAIR only (A1)
Output: work/reports/loco_<features><tag>.json. Cost: 4 LightGBM fits on ~1/2 of the train folds.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import (CONTEXT_FEATURES, PAIR_FEATURES, TAG, _pred, choose_decision,  # noqa: E402
                            decide, extra_features, load_xy, train_lgb)
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json, stage  # noqa: E402


def country_slice(meta, X, y, rid_country, country):
    m = meta.select("rid").join(rid_country, on="rid", how="left", maintain_order="left") \
        .get_column("country").eq(country).to_numpy()
    return meta.filter(pl.Series(m)), X[m], y[m]


def evaluator(meta, y, fold, country):
    W = config.WORK_DIR / "prep"
    pop = (pl.read_parquet(W / "s1_train.parquet", columns=["rid", "fold", "country"])
             .filter((pl.col("fold") == fold) & (pl.col("country") == country)).select("rid")
             .join(pl.read_parquet(W / "gt_train.parquet").group_by("rid").len("n_true"), on="rid", how="left")
             .with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), y, pop.get_column("n_true").to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["pair", "context"], default="context")
    a = ap.parse_args()
    feats = PAIR_FEATURES + (CONTEXT_FEATURES if a.features == "context" else []) + extra_features()
    rid_country = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "country"])
    countries = sorted(rid_country.get_column("country").unique().to_list())
    data = {r: load_xy("train", f, feats) for r, f in
            (("train", config.TRAIN_FOLDS), ("tune", [config.TUNE_FOLD]), ("eval", [config.EVAL_FOLD]))}
    out = {"features": a.features, "runs": {}}
    models, taus = {}, {}
    with stage(f"loco_{a.features}{TAG}"):
        for c in countries:
            tr = country_slice(*data["train"], rid_country, c)
            tu = country_slice(*data["tune"], rid_country, c)
            models[c] = train_lgb(tr[1], tr[2], tu[1], tu[2], feats, f"loco_{a.features}{TAG}_{c}")
            ev = evaluator(tu[0], tu[2], config.TUNE_FOLD, c)
            ch = choose_decision(ev, _pred(models[c], tu[1]), tu[0].get_column("gid").to_numpy())
            taus[c] = ("threshold+excl", ch["threshold+excl"]["tau"])
        for src in countries:
            for dst in countries:
                me, Xe, ye = country_slice(*data["eval"], rid_country, dst)
                ev = evaluator(me, ye, config.EVAL_FOLD, dst)
                p = _pred(models[src], Xe)
                g = me.get_column("gid").to_numpy()
                var, tau = taus[src]
                oracle = max((ev.macro_f05(decide(p, g, var, t)), t) for t in config.THRESHOLD_GRID)
                rep = ev.report(decide(p, g, var, tau))
                out["runs"][f"{src}->{dst}"] = {"tau_from_source": tau, "macro_f05": rep["macro_f05"],
                                                 "pair_precision": rep["pair_precision"], "pair_recall": rep["pair_recall"],
                                                 "singleton_accuracy": rep["singleton_accuracy"],
                                                 "oracle_tau": oracle[1], "oracle_tau_f05": round(oracle[0], 5)}
                log(f"LOCO {src}->{dst}: {out['runs'][f'{src}->{dst}']}")
    save_json(out, config.WORK_DIR / "reports" / f"loco_{a.features}{TAG}.json")


if __name__ == "__main__":
    main()
