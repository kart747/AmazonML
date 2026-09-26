"""Failure analysis on the eval fold: where do false positives and false negatives come from?

Uses the scored pairs saved by `run.py evaluate` (reports/scores_A<n><tag>.parquet) and
the decision chosen on the tune fold. Each error gets label-free detector flags, and
we report counts plus readable samples.

Usage:  python src/experiments/error_analysis.py --approach 2 [--samples 25]
Output: work/reports/errors_A<n><tag>.md
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import TAG, decide  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", type=int, default=2)
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()
    W, name = config.WORK_DIR, f"A{a.approach}{TAG}"
    rep = json.load(open(W / "reports" / f"eval_{name}.json"))
    var, tau = rep["decision_selected_on_tune_fold"]["variant"], rep["decision_selected_on_tune_fold"]["tau"]
    sc = pl.read_parquet(W / "reports" / f"scores_{name}.parquet").filter(pl.col("fold") == config.EVAL_FOLD)
    sc = sc.with_columns(keep=pl.Series(decide(sc.get_column("p").to_numpy(), sc.get_column("gid").to_numpy(), var, tau)))
    ctx = (pl.scan_parquet(W / "feat" / "train" / "*.parquet").filter(pl.col("fold") == config.EVAL_FOLD)
             .select("rid", "gid", "t_is_best", "q_nm_freq", "t_addr_empty", "t_nonascii", "num_first_eq",
                     "r_rev", "cos_name", "cos_addr").collect())
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "name", "addr", "country"])
    tg = pl.read_parquet(W / "prep" / "tg_train.parquet", columns=["gid", "src", "name", "addr"])
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet").join(s1.filter(pl.col("fold") == config.EVAL_FOLD).select("rid"), on="rid")
    n_true = gt.group_by("rid").len("n_true")

    fp = sc.filter(pl.col("keep") & (pl.col("y") == 0))
    fn_c = sc.filter(~pl.col("keep") & (pl.col("y") == 1))
    miss = gt.join(sc.select("rid", "gid"), on=["rid", "gid"], how="anti")
    flags = lambda d: (d.join(ctx, on=["rid", "gid"], how="left").join(n_true, on="rid", how="left")
                        .with_columns(pl.col("n_true").fill_null(0))
                        .with_columns(singleton=pl.col("n_true") == 0, shared_name=pl.col("q_nm_freq") > 1,
                                      not_best_for_target=pl.col("t_is_best") == 0, empty_t_addr=pl.col("t_addr_empty") == 1,
                                      cross_script=pl.col("t_nonascii") == 1, number_conflict=pl.col("num_first_eq") == 0))
    det = ["singleton", "shared_name", "not_best_for_target", "empty_t_addr", "cross_script", "number_conflict"]
    fp, fn_c = flags(fp), flags(fn_c)
    lines = [f"# Error analysis: {name}, eval fold {config.EVAL_FOLD}, decision {var} tau={tau}\n",
             f"- false positives: {fp.height:,}  (on singletons: {fp.get_column('singleton').sum():,})",
             f"- false negatives among candidates (scoring misses): {fn_c.height:,}",
             f"- false negatives from blocking (never candidates): {miss.height:,}\n",
             "| detector | FP count | FP share | FN(cand) count | FN share |", "|---|---|---|---|---|"]
    for d in det:
        f1, f2 = int(fp.get_column(d).fill_null(False).sum()), int(fn_c.get_column(d).fill_null(False).sum())
        lines.append(f"| {d} | {f1:,} | {f1 / max(1, fp.height):.2f} | {f2:,} | {f2 / max(1, fn_c.height):.2f} |")

    def show(d, title, sort_desc):
        d = (d.sort("p", descending=sort_desc).head(a.samples)
               .join(s1.select("rid", q_name="name", q_addr="addr", country="country"), on="rid", how="left")
               .join(tg.select("gid", "src", t_name="name", t_addr="addr"), on="gid", how="left"))
        out = [f"\n## {title}\n", "| p | flags | S1 name / address | target name / address |", "|---|---|---|---|"]
        for r in d.iter_rows(named=True):
            fl = ",".join(k for k in det if r.get(k))
            out.append(f"| {r['p']:.3f} | {fl} | {r['q_name']} / {r['q_addr']} ({r['country']}) | "
                       f"S{r['src']}: {r['t_name']} / {r['t_addr']} |".replace("\n", " "))
        return out

    lines += show(fp, "Highest-confidence false positives", True)
    lines += show(fn_c, "Lowest-scored false negatives among candidates", False)
    m = (miss.head(a.samples).join(s1.select("rid", q_name="name", q_addr="addr"), on="rid")
             .join(tg.select("gid", "src", t_name="name", t_addr="addr"), on="gid"))
    lines += ["\n## Blocking misses (sample)\n", "| S1 name / address | target name / address |", "|---|---|"]
    lines += [f"| {r['q_name']} / {r['q_addr']} | S{r['src']}: {r['t_name']} / {r['t_addr']} |" for r in m.iter_rows(named=True)]
    out = W / "reports" / f"errors_{name}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"written {out}")


if __name__ == "__main__":
    main()
