"""E-AB1: entity-level abstention (idea 1), offline falsification.

Hypothesis: kept false positives concentrate in a small set of entities (S1 or
target) that are identifiable by label-free structure, so dropping those entities'
matches raises macro F0.5.

Protocol: every rule and threshold is selected on the TUNE fold (3) only; the
EVAL fold (0) is reported once, untouched. Baseline is the unchanged s2lab_all
scores at tau=0.68, variant threshold+excl, scored with ber.metric.f05_per_entity.

No submission is written, no existing artifact is modified.

Usage:
  python src/experiments/abstain_lab.py scan   # signal separation, both folds
  python src/experiments/abstain_lab.py run    # rules, threshold picked on fold 3
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import exclusive_mask  # noqa: E402
from ber.metric import f05_per_entity  # noqa: E402
from ber.utils import log, save_json  # noqa: E402

TAU = 0.68
NAME = "s2lab_all"
FEAT_COLS = ["r_rev", "t_is_best", "q_ncand", "q_nm_freq", "t_nm_freq",
             "num_first_eq", "t_addr_empty", "cos_name", "rev_margin", "t_gap12"]
GRID = {"t_contest": [1, 2, 3], "t_ncand": [4, 8, 16, 32],
        "s1_over": [1, 2, 3], "t_margin": [0.0, 0.05, 0.10]}


def load():
    W = config.WORK_DIR
    sc = pl.read_parquet(W / "reports" / f"scores_{NAME}.parquet")
    sc = sc.with_columns(keep=pl.Series(exclusive_mask(sc.get_column("gid").to_numpy(),
                                                       sc.get_column("p").to_numpy()))
                         & (pl.col("p") >= TAU))
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet")
    s1t = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    n_true = gt.group_by("rid").len("n_true")
    sq = sc.group_by("rid").agg(q_sum_p=pl.col("p").sum(),
                                q_nmid=((pl.col("p") >= 0.2) & (pl.col("p") < 0.99)).sum())
    sc = sc.join(sq, on="rid", how="left")
    # target-level structure (numpy: how many S1s claim this target, and the runner-up)
    gid, p = sc.get_column("gid").to_numpy(), sc.get_column("p").to_numpy()
    _, inv, cnt = np.unique(gid, return_inverse=True, return_counts=True)
    t_ncand = cnt[inv]
    order = np.lexsort((-p, gid))
    gs, ps = gid[order], p[order]
    first = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    second = first + 1
    ok = second < len(gs)
    ok[ok] &= gs[second[ok]] == gs[first[ok]]
    uniq, first_p, second_p = gs[first], ps[first], np.full(len(first), -1.0)
    second_p[ok] = ps[second[ok]]
    srt = np.argsort(uniq)
    pos = np.searchsorted(uniq[srt], gid)
    t_2nd = second_p[srt][pos]
    sc = sc.with_columns(t_ncand=pl.Series(t_ncand), t_2ndp=pl.Series(t_2nd),
                         t_nsecond=pl.Series(t_2nd >= TAU),
                         t_margin=pl.Series(p - t_2nd))
    ctx = (pl.scan_parquet(str(W / "feat" / "train" / "*.parquet"))
           .filter(pl.col("fold").is_in([0, 3])).select(["rid", "gid", "fold"] + FEAT_COLS).collect())
    sc = sc.join(ctx, on=["rid", "gid", "fold"], how="left")
    out = {}
    for f in (3, 0):
        pop = (s1t.filter(pl.col("fold") == f).select("rid", "country")
                   .join(n_true, on="rid", how="left")
                   .with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
        sub = sc.filter(pl.col("fold") == f)
        s1 = (sub.select("rid").join(pop.select("rid", "idx"), on="rid", how="left",
                                     maintain_order="left").get_column("idx").to_numpy())
        d = {"df": sub, "s1": s1, "pop": pop,
             "n_true": pop.get_column("n_true").to_numpy().astype(int),
             "y": sub.get_column("y").to_numpy().astype(bool),
             "keep": sub.get_column("keep").to_numpy().astype(bool),
             "p": sub.get_column("p").to_numpy(),
             "gid": sub.get_column("gid").to_numpy()}
        nk = np.bincount(s1, minlength=len(d["n_true"]))
        qsum = np.bincount(s1, weights=sub.get_column("q_sum_p").to_numpy(),
                           minlength=len(d["n_true"])) / np.maximum(1, nk)
        d["s1_sum_p"], d["s1_nkeep"] = qsum, nk
        out[f] = d
    return out


def report(d, drop):
    keep = d["keep"] & ~drop
    s1, n = d["s1"], len(d["n_true"])
    n_pred = np.bincount(s1[keep], minlength=n)
    tp = np.bincount(s1[keep & d["y"]], minlength=n)
    f = f05_per_entity(tp, n_pred, d["n_true"])
    hit = d["keep"] & drop
    return {"f05": float(f.mean()),
            "pair_P": float(d["y"][keep].mean()) if keep.any() else 0.0,
            "pair_R": float(tp.sum() / d["n_true"].sum()),
            "removed": int(hit.sum()), "rem_tp": int((hit & d["y"]).sum()),
            "rem_fp": int((hit & ~d["y"]).sum()),
            "aff_s1": int(len(np.unique(s1[hit]))) if hit.any() else 0,
            "aff_s1_bad": int(len(np.unique(s1[hit & ~d["y"]]))) if hit.any() else 0,
            "aff_tg": int(len(np.unique(d["gid"][hit]))) if hit.any() else 0,
            "kept": int(keep.sum()), "fp_base": int((d["keep"] & ~d["y"]).sum()),
            "tp_base": int((d["keep"] & d["y"]).sum())}


def drops(d):
    df, keep, s1 = d["df"], d["keep"], d["s1"]
    tn = df.get_column("t_ncand").to_numpy()
    tns = df.get_column("t_nsecond").to_numpy().astype(int)
    tm = df.get_column("t_margin").to_numpy()
    out = {f"t_contest>={k}": keep & (tns >= k) for k in GRID["t_contest"]}
    out.update({f"t_ncand>={k}": keep & (tn >= k) for k in GRID["t_ncand"]})
    out.update({f"t_margin<={k}": keep & (tm <= k) for k in GRID["t_margin"]})
    for h in GRID["s1_over"]:
        bad = d["s1_nkeep"] > np.ceil(d["s1_sum_p"]) + h
        out[f"s1_over>={h}"] = bad[s1] & keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "run"])
    a = ap.parse_args()
    t0 = time.time()
    data = load()
    z = lambda d: np.zeros(len(d["keep"]), bool)
    base = {f: report(data[f], z(data[f])) for f in (3, 0)}
    for f in (3, 0):
        b = base[f]
        print(f"fold {f} baseline: F0.5={b['f05']:.5f} P={b['pair_P']:.5f} R={b['pair_R']:.5f} "
              f"kept={b['kept']:,} (TP {b['tp_base']:,} / FP {b['fp_base']:,})")

    if a.mode == "scan":
        rows = []
        for f in (3, 0):
            d, k, y = data[f], data[f]["keep"], data[f]["y"]
            sig = {"t_ncand": d["df"].get_column("t_ncand").to_numpy(),
                   "t_nsecond": d["df"].get_column("t_nsecond").to_numpy().astype(float),
                   "t_margin": d["df"].get_column("t_margin").to_numpy(),
                   "t_2ndp": d["df"].get_column("t_2ndp").to_numpy(),
                   "q_sum_p": d["s1_sum_p"][d["s1"]],
                   "q_nmid": d["df"].get_column("q_nmid").to_numpy(),
                   "r_rev": d["df"].get_column("r_rev").to_numpy(),
                   "t_is_best": d["df"].get_column("t_is_best").to_numpy().astype(float),
                   "q_nm_freq": d["df"].get_column("q_nm_freq").to_numpy(),
                   "t_nm_freq": d["df"].get_column("t_nm_freq").to_numpy(),
                   "p": d["p"]}
            for name, v in sig.items():
                for lab, sel in (("TP", k & y), ("FP", k & ~y)):
                    s = np.asarray(v[sel], dtype=float)
                    s = s[~np.isnan(s)]
                    if s.size:
                        rows.append(f"| {f} | {name} | {lab} | {s.size:,} | {np.median(s):.4f} | "
                                    f"{np.mean(s):.4f} | {np.percentile(s, 90):.4f} |")
            fp = np.bincount(d["s1"][k & ~y], minlength=len(d["n_true"]))
            pred = np.bincount(d["s1"][k], minlength=len(d["n_true"]))
            order = np.argsort(-fp)
            tot = max(1, int(fp.sum()))
            for x in (0.01, 0.05, 0.10):
                m = int(len(fp) * x)
                rows.append(f"| {f} | fp_top{x:.0%}_of_S1 | conc | {fp[order[:m]].sum():,}/{tot:,} | "
                            f"{fp[order[:m]].sum() / tot:.3f} of FP | those S1 keep {pred[order[:m]].sum():,} pairs | — |")
        print("\n| fold | signal | class | n | median | mean | p90 |")
        print("|---|---|---|---|---|---|---|")
        print("\n".join(rows))
        return

    names = [f"t_contest>={k}" for k in GRID["t_contest"]] + \
            [f"t_ncand>={k}" for k in GRID["t_ncand"]] + \
            [f"t_margin<={k}" for k in GRID["t_margin"]] + \
            [f"s1_over>={h}" for h in GRID["s1_over"]]
    res = []
    print("\n| rule | fold3 F0.5 | d3 | fold3 rem T/F | fold0 F0.5 | d0 | fold0 rem T/F | fold0 aff S1 |")
    print("|---|---|---|---|---|---|---|---|")
    for rule in names:
        r3 = report(data[3], drops(data[3])[rule])
        r0 = report(data[0], drops(data[0])[rule])
        res.append((rule, r3, r0))
        print(f"| {rule} | {r3['f05']:.5f} | {r3['f05'] - base[3]['f05']:+.5f} | "
              f"{r3['rem_tp']:,}/{r3['rem_fp']:,} | {r0['f05']:.5f} | {r0['f05'] - base[0]['f05']:+.5f} | "
              f"{r0['rem_tp']:,}/{r0['rem_fp']:,} | {r0['aff_s1']:,} | {r0['aff_s1_bad']:,} | "
              f"{r0['pair_P']:.5f} | {r0['pair_R']:.5f} |")
    best = max(res, key=lambda r: r[1]["f05"])
    print(f"\nselected on fold 3: {best[0]}  fold3 {best[1]['f05'] - base[3]['f05']:+.5f} "
          f"-> fold 0 {best[2]['f05'] - base[0]['f05']:+.5f}")
    save_json({"name": "E-AB1", "tau": TAU, "scores": NAME,
               "baseline": {"fold3": base[3], "fold0": base[0]},
               "rules": [{"rule": r, "fold3": x, "fold0": y} for r, x, y in res],
               "selected_on_fold3": best[0],
               "seconds": round(time.time() - t0, 1),
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)},
              config.WORK_DIR / "reports" / "abstain_lab.json")
    log(f"done {time.time() - t0:.1f}s peak RSS "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
