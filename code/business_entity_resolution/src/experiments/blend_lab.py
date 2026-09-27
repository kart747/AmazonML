"""Blend lab: average the saved tune/eval scores of several stage-2 models, then decide.

Input: reports/scores_<name>.parquet (rid, gid, fold, y, p on tune + eval folds), as written
by stage2_lab.evaluate. Row sets may differ between models (different stage-1 → different
cascade prune). The union is used and a missing pair counts as p = 0 for that model (it was
pruned, i.e. rejected, by that model's cascade).

Every blend's weights (equal or rank-mean) and tau are chosen on the tune fold; the eval
fold is reported for the tune-best blend only (the others are printed for reference, marked).

Usage: python src/experiments/blend_lab.py --models s2lab_all,s2lab_all_hp1,s2lab_all_s1fx [--work DIR]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import choose_decision, decide, exclusive_mask, make_evaluator  # noqa: E402
from ber.utils import log, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    names = a.models.split(",")
    R = config.WORK_DIR / "reports"
    df = None
    for n in names:
        s = pl.read_parquet(R / f"scores_{n}.parquet").select("rid", "gid", "fold", "y", pl.col("p").alias(n))
        df = s if df is None else df.join(s, on=["rid", "gid", "fold", "y"], how="full", coalesce=True)
    df = df.with_columns([pl.col(n).fill_null(0.0) for n in names])
    uf, ef = config.TUNE_FOLD, config.EVAL_FOLD
    tu, ev = df.filter(pl.col("fold") == uf), df.filter(pl.col("fold") == ef)
    E_tu = make_evaluator(tu.select("rid"), tu.get_column("y").to_numpy(), uf)
    E_ev = make_evaluator(ev.select("rid"), ev.get_column("y").to_numpy(), ef)
    g_tu, g_ev = tu.get_column("gid").to_numpy(), ev.get_column("gid").to_numpy()
    res = {}
    combos = [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)
              if k == 1 or k >= a.min_size]
    for c in combos:
        p_tu = tu.select(pl.mean_horizontal(list(c))).to_series().to_numpy()
        p_ev = ev.select(pl.mean_horizontal(list(c))).to_series().to_numpy()
        ex = exclusive_mask(np.r_[g_tu, g_ev], np.r_[p_tu, p_ev])
        ch = choose_decision(E_tu, p_tu, g_tu, ex[:len(g_tu)])
        v = max(ch, key=lambda k: ch[k]["tune_f05"])
        rep = E_ev.report(decide(p_ev, g_ev, v, ch[v]["tau"], ex[len(g_tu):]))
        key = "+".join(c)
        res[key] = {"variant": v, "tau": ch[v]["tau"], "tune_f05": ch[v]["tune_f05"], "eval": rep}
        log(f"{key}: tune {ch[v]['tune_f05']} (tau {ch[v]['tau']}) | eval {rep['macro_f05']} FP {rep['false_positive_pairs']}")
    best = max(res, key=lambda k: res[k]["tune_f05"])
    single = max((k for k in res if "+" not in k), key=lambda k: res[k]["tune_f05"])
    log(f"TUNE-BEST blend: {best} tune {res[best]['tune_f05']} eval {res[best]['eval']['macro_f05']} | "
        f"best single {single} tune {res[single]['tune_f05']} eval {res[single]['eval']['macro_f05']} | "
        f"delta tune {res[best]['tune_f05'] - res[single]['tune_f05']:+.5f}")
    save_json({"models": names, "best": best, "best_single": single, "results": res},
              R / f"blend{a.tag}.json")


if __name__ == "__main__":
    main()
