"""P-dense: validation under test-like distractor density.

The test set has ~23% more targets per S1 than train (train 4.68, test 5.75). Remove
a random DROP_FRAC of train S1 from the graph: their true targets become owner-less
distractors, exactly like test targets whose entity is absent from S1. Every label-free
context feature that depends on the S1 pool is recomputed (reverse-list rank/scores,
"is this S1 the target's best", margin to the best other S1, name frequencies), the
already-trained models rescore the tune/eval folds, and the threshold is re-selected.

Answers: how much F0.5 drops under test density, and which tau to use on test.
Forward retrieval is unchanged (the target pool is identical); reverse-view candidates
that would newly enter an S1's top-k after the drop cannot be added (small effect).

Usage (after `run.py evaluate --approach N`):
    python src/experiments/density.py --approach 2 [--drop 0.19]
Output: work/reports/density_A<n><tag>.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import (P1_FEATURES, PAIR_FEATURES, S1_FEATS, TAG, _pred, _with_extras,  # noqa: E402
                            add_stage2_features, choose_decision, decide, extra_features,
                            load_model, make_evaluator, parts)
from ber.utils import log, save_json, stage  # noqa: E402


def dense_context(df, rev_rank, rev_sum, freq_rid, freq_name, tg_nm):
    """Recompute pool-dependent context features for the pairs in df."""
    df = df.drop("t_top1", "t_gap12", "rev_margin", "t_is_best", "r_rev", "q_nm_freq", "t_nm_freq")
    df = (df.join(rev_sum, on="gid", how="left")
            .join(rev_rank, on=["gid", "rid"], how="left")
            .join(freq_rid, on="rid", how="left")
            .join(tg_nm, on="gid", how="left")
            .join(freq_name, on=["country", "t_nm"], how="left"))
    return df.with_columns(
        r_rev=pl.col("r_rev").fill_null(99).cast(pl.Float32),
        t_nm_freq=pl.col("t_nm_freq").fill_null(0),
        t_is_best=(pl.col("rid") == pl.col("t_best_rid")).cast(pl.Float32),
        t_gap12=pl.col("t_top1") - pl.col("t_top2").fill_null(0),
    ).with_columns(
        rev_margin=pl.when(pl.col("t_is_best") == 1).then(pl.col("cos_combo") - pl.col("t_top2").fill_null(0))
        .otherwise(pl.col("cos_combo") - pl.col("t_top1").fill_null(0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.19)
    a = ap.parse_args()
    W, folds = config.WORK_DIR, [config.TUNE_FOLD, config.EVAL_FOLD]
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country", "nm"])
    rng = np.random.default_rng(config.SEED)
    dropped = s1.select("rid").filter(pl.Series(rng.random(s1.height) < a.drop))
    kept = s1.join(dropped, on="rid", how="anti")
    log(f"dropping {dropped.height:,} of {s1.height:,} S1 ({a.drop:.0%})")
    freq_name = kept.group_by(["country", "nm"]).len("t_nm_freq").rename({"nm": "t_nm"})
    freq_rid = kept.join(freq_name.rename({"t_nm": "nm", "t_nm_freq": "q_nm_freq"}), on=["country", "nm"]) \
        .select("rid", "country", "q_nm_freq")
    tg = pl.read_parquet(W / "prep" / "tg_train.parquet", columns=["gid", "src", "nm", "ad"])
    tg_nm = tg.select("gid", t_nm="nm")

    # reverse lists without the dropped S1: re-rank, summarize
    revs = []
    for f in sorted((W / "block" / "train").glob("rev_*.parquet")):
        r = pl.read_parquet(f).join(dropped, on="rid", how="anti")
        revs.append(r.with_columns(rrank=pl.col("rscore").rank("ordinal", descending=True).over("gid") - 1))
    rev = pl.concat(revs)
    rev = rev.filter(pl.col("rrank") < config.BLOCK["k_rev"])
    rev_rank = rev.select("gid", "rid", r_rev="rrank")
    rev_sum = (rev.sort(["gid", "rrank"]).group_by("gid", maintain_order=True)
                  .agg(t_best_rid=pl.col("rid").first(), t_top1=pl.col("rscore").first(),
                       t_top2=pl.col("rscore").get(1, null_on_oob=True)))
    del revs, rev

    extras = extra_features()
    if a.approach == 2:
        m_a, m_b = load_model(f"A2{TAG}_stage1_foldA"), load_model(f"A2{TAG}_stage1_foldB")
        m2 = load_model(f"A2{TAG}_stage2")
    else:
        m1 = load_model(f"A1{TAG}")
    out = {}
    with stage(f"density_A{a.approach}{TAG}"):
        metas, ys, ps = [], [], []
        for p in parts("train"):
            df = _with_extras(pl.read_parquet(p), "train", part_name=p.name)
            df = df.filter(pl.col("fold").is_in(folds)).join(dropped, on="rid", how="anti")
            if df.height == 0:
                continue
            cols = df.columns
            df = dense_context(df, rev_rank, rev_sum, freq_rid, freq_name, tg_nm).select(cols)
            if a.approach == 1:
                prob = _pred(m1, np.ascontiguousarray(df.select(PAIR_FEATURES + extras).to_numpy(), dtype=np.float32))
            else:
                X = np.ascontiguousarray(df.select(S1_FEATS + extras).to_numpy(), dtype=np.float32)
                df = df.with_columns(p1=pl.Series(0.5 * (_pred(m_a, X) + _pred(m_b, X))))
                df = add_stage2_features(df, tg)
                prob = _pred(m2, np.ascontiguousarray(
                    df.select(S1_FEATS + P1_FEATURES + extras).to_numpy(), dtype=np.float32))
            metas.append(df.select("rid", "gid", "fold"))
            ys.append(df.get_column("y").to_numpy())
            ps.append(prob)
        meta, y, prob = pl.concat(metas), np.concatenate(ys), np.concatenate(ps)

        from ber.metric import Evaluator
        res = {}
        for role, fold in (("tune", config.TUNE_FOLD), ("eval", config.EVAL_FOLD)):
            m = (meta.get_column("fold") == fold).to_numpy()
            mf = meta.filter(pl.Series(m))
            pop = (s1.filter(pl.col("fold") == fold).join(dropped, on="rid", how="anti").select("rid")
                     .join(pl.read_parquet(W / "prep" / "gt_train.parquet").group_by("rid").len("n_true"),
                           on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
            idx = mf.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
            res[role] = (Evaluator(idx.get_column("idx").to_numpy(), y[m], pop.get_column("n_true").to_numpy()),
                         prob[m], mf.get_column("gid").to_numpy())
        ev_tu, p_tu, g_tu = res["tune"]
        ev_ev, p_ev, g_ev = res["eval"]
        choice = choose_decision(ev_tu, p_tu, g_tu)
        variant = max(choice, key=lambda v: choice[v]["tune_f05"])
        normal = __import__("json").load(open(W / "reports" / f"eval_A{a.approach}{TAG}.json"))
        n_var, n_tau = normal["decision_selected_on_tune_fold"]["variant"], normal["decision_selected_on_tune_fold"]["tau"]
        out = {
            "drop_frac": a.drop, "dropped_s1": dropped.height,
            "dense_selected": {"variant": variant, "tau": choice[variant]["tau"], "all": choice},
            "eval_dense_with_dense_tau": ev_ev.report(decide(p_ev, g_ev, variant, choice[variant]["tau"])),
            "eval_dense_with_normal_tau": ev_ev.report(decide(p_ev, g_ev, n_var, n_tau)),
            "normal_eval_f05": normal["eval"]["macro_f05"], "normal_tau": n_tau,
        }
    save_json(out, W / "reports" / f"density_A{a.approach}{TAG}.json")
    log(f"P-dense A{a.approach}{TAG}: normal F0.5 {out['normal_eval_f05']} (tau {n_tau}) | dense with normal tau "
        f"{out['eval_dense_with_normal_tau']['macro_f05']} | dense with dense tau "
        f"{out['eval_dense_with_dense_tau']['macro_f05']} (tau {choice[variant]['tau']}, {variant})")


if __name__ == "__main__":
    main()
