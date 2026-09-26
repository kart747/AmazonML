"""Decoder lab: compare F0.5 decision rules on saved validation scores.

Input: work/reports/scores_A<n><tag>.parquet (written by `run.py evaluate`), which
holds scored candidate pairs of the tune fold and the eval fold. Every decoder's
parameters are chosen on the tune fold; the eval fold is only reported. Each variant
is compared with the baseline (threshold + exclusivity) by a paired bootstrap over S1.

Usage:
    python src/experiments/decoder_lab.py --approach 2 [--tag _v2]
Output: work/reports/decoders_A<n><tag>.json and a printed table.
Cost: a few minutes on one CPU; needs ~2x the scores file in RAM.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import make_evaluator  # noqa: E402
from ber.decoders import Calibrator, exclusive, expected_f05, relative  # noqa: E402
from ber.utils import log, save_json  # noqa: E402


def paired_bootstrap(fa, fb, n_boot=200, seed=0):
    """Mean and SE of (fa - fb) over S1 resamples."""
    rng = np.random.default_rng(seed)
    d = fa - fb
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return float(d.mean()), float(d[idx].mean(axis=1).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", type=int, default=2)
    ap.add_argument("--tag", default=config.RUN_TAG)
    a = ap.parse_args()
    name = f"A{a.approach}{a.tag}"
    sc = pl.read_parquet(config.WORK_DIR / "reports" / f"scores_{name}.parquet")
    src = pl.read_parquet(config.WORK_DIR / "prep" / "tg_train.parquet", columns=["gid", "src"])
    sc = sc.join(src, on="gid", how="left", maintain_order="left")

    data = {}
    for role, fold in (("tune", config.TUNE_FOLD), ("eval", config.EVAL_FOLD)):
        d = sc.filter(pl.col("fold") == fold)
        ev = make_evaluator(d.select("rid"), d.get_column("y").to_numpy(), fold)
        data[role] = dict(ev=ev, p=d.get_column("p").to_numpy().astype(np.float64),
                          gid=d.get_column("gid").to_numpy(), src=d.get_column("src").to_numpy(),
                          y=d.get_column("y").to_numpy())
    cal = Calibrator().fit(data["tune"]["p"], data["tune"]["y"])
    for d in data.values():
        d["pc"] = cal(d["p"]).astype(np.float64)
        d["excl"] = exclusive(d["gid"], d["p"])

    taus = config.THRESHOLD_GRID
    decoders = {
        "threshold": (lambda d, t: d["p"] >= t, [(t,) for t in taus]),
        "threshold+excl": (lambda d, t: d["excl"] & (d["p"] >= t), [(t,) for t in taus]),
        "relative+excl": (lambda d, t, al: d["excl"] & relative(d["ev"].s1_idx, d["p"], d["ev"].n, t, al),
                          list(itertools.product(taus[::2], [0.5, 0.7, 0.8, 0.9, 0.95]))),
        "per_source+excl": (lambda d, t2, t3: d["excl"] & (d["p"] >= np.where(d["src"] == 2, t2, t3)),
                            list(itertools.product(np.arange(0.4, 0.96, 0.05).round(2), repeat=2))),
        "expected_f05_calibrated+excl": (
            lambda d, lam: d["excl"] & expected_f05(d["ev"].s1_idx, np.where(d["excl"], d["pc"], 0.0), d["ev"].n, lam),
            [(x,) for x in (0.0, 0.05, 0.1, 0.2, 0.4)]),
    }
    results, per_entity = {}, {}
    for dname, (fn, grid) in decoders.items():
        tu = data["tune"]
        best_f, best_params = max((tu["ev"].macro_f05(fn(tu, *g)), g) for g in grid)
        ev = data["eval"]
        keep = fn(ev, *best_params)
        per_entity[dname] = ev["ev"].per_entity(keep)
        rep = ev["ev"].report(keep)
        results[dname] = {"params": [float(x) for x in best_params], "tune_f05": round(best_f, 5),
                          "eval_f05": rep["macro_f05"], "eval_pair_precision": rep["pair_precision"],
                          "eval_pair_recall": rep["pair_recall"], "eval_singleton_acc": rep["singleton_accuracy"],
                          "eval_fp_pairs": rep["false_positive_pairs"]}
        log(f"{dname}: {results[dname]}")
    base = per_entity["threshold+excl"]
    for dname in results:
        m, se = paired_bootstrap(per_entity[dname], base)
        results[dname]["delta_vs_threshold+excl"] = round(m, 5)
        results[dname]["delta_se"] = round(se, 5)
    save_json(results, config.WORK_DIR / "reports" / f"decoders_{name}.json")
    print(f"\n| decoder ({name}) | params (chosen on fold {config.TUNE_FOLD}) | eval F0.5 | Δ vs thr+excl (SE) | pair P | pair R | singleton acc |")
    print("|---|---|---|---|---|---|---|")
    for dname, r in results.items():
        print(f"| {dname} | {r['params']} | {r['eval_f05']:.5f} | {r['delta_vs_threshold+excl']:+.5f} ({r['delta_se']:.5f}) "
              f"| {r['eval_pair_precision']:.4f} | {r['eval_pair_recall']:.4f} | {r['eval_singleton_acc']} |")


if __name__ == "__main__":
    main()
