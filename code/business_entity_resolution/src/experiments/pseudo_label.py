"""Unseen-country adaptation by pseudo-labelling, measured with a leave-one-country-out proxy.

France is 15% of test S1 (~15% of the public LB) and has no labels. Proxy: pretend one train
country (--target, default India) is unlabeled.
  base    stage 2 (stage2_lab features of --ref) trained on the SOURCE country's train folds only,
          tau from the source country's tune fold, reported on the TARGET country's eval fold.
  pseudo  the base model scores the target country's train+tune folds. Confident pairs become
          pseudo-labels (p >= --hi and the target's best S1 -> 1; p <= --lo -> 0; the rest are
          dropped), and stage 2 is retrained on source rows + pseudo rows. Same tau protocol, same report.
  oracle  trained with the target country's TRUE labels (upper bound; = the normal all-country model
          restricted to the target's rows).
If pseudo recovers a meaningful part of (oracle - base), the same recipe applies to France on
test: pseudo-label France test pairs with the shipped model, then retrain.
Caveat: stage 1 (p1 and everything derived from it) was trained on both countries, so this
measures stage-2 transfer only.

    BER_WORK_DIR=../work_xd python src/experiments/pseudo_label.py --ref s2lab_all --target India
    BER_WORK_DIR=../work_xd python src/experiments/pseudo_label.py apply --target France --out <dir>
      (apply: pseudo-label France TEST pairs with the shipped model, retrain on train folds + them,
       tau on tune fold 3, report normal fold 0, predict all test -> submission with scored-pair candidates)
Output: <work>/reports/pseudo_label_<target>.json. RAM ~3 GB at 4 threads.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ber.approaches as ap  # noqa: E402
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json, stage  # noqa: E402


def country_rows(feats, folds):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "country"])
    df = (pl.scan_parquet(config.WORK_DIR / s2.OUT / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
            .select(["rid", "gid", "fold", "y"] + feats).collect().join(s1, on="rid", how="left"))
    return df


def evaluator(meta, fold, country):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(config.WORK_DIR / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter((pl.col("fold") == fold) & (pl.col("country") == country)).select("rid")
             .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), meta.get_column("y").to_numpy(), pop.get_column("n_true").to_numpy())


def fit(tr, tu, feats, weight=None):
    X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
    dtr = lgb.Dataset(X(tr), tr.get_column("y").to_numpy(), weight=weight, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X(tu), tu.get_column("y").to_numpy(), reference=dtr)
    return lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(0)])


def report(m, feats, tu, ev, src, tgt):
    """tau/variant chosen on the SOURCE tune fold, reported on the TARGET eval fold."""
    X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
    uf, ef = config.TUNE_FOLD, config.EVAL_FOLD
    p_tu, p_ev = ap._pred(m, X(tu)), ap._pred(m, X(ev))
    g_tu, g_ev = tu.get_column("gid").to_numpy(), ev.get_column("gid").to_numpy()
    ch = ap.choose_decision(evaluator(tu, uf, src), p_tu, g_tu)
    v = max(ch, key=lambda k: ch[k]["tune_f05"])
    e = evaluator(ev, ef, tgt).report(ap.decide(p_ev, g_ev, v, ch[v]["tau"]))
    return {"variant": v, "tau": ch[v]["tau"], "source_tune_f05": ch[v]["tune_f05"], "target_eval": e}


def apply(ref, target, hi, lo, out_dir, tag, target_src=None, mix_dense=False):
    """The real thing: pseudo-label the TARGET country's TEST pairs with the shipped model, retrain
    stage 2 on train folds + those rows, choose tau on tune fold 3, report normal fold 0, predict test."""
    from ber.approaches import make_evaluator, write_submission
    import redecode
    WD = config.WORK_DIR
    refm = lgb.Booster(model_file=str(WD / "models" / f"{ref}.txt"))
    feats = refm.feature_name()
    s1t = pl.read_parquet(WD / "prep" / "s1_test.parquet", columns=["rid", "country"])
    # target_src: another work dir whose s2lab/test/<target>_*.parquet replace this one's (e.g. re-normalised
    # France features); every other country's test parts still come from WD
    TS = Path(target_src) if target_src else WD
    tparts = sorted((TS / s2.OUT / "test").glob(f"{target}_*.parquet"))
    assert tparts, f"no {target} test parts in {TS / s2.OUT / 'test'}"
    pool = pl.concat([pl.read_parquet(p, columns=["rid", "gid"] + feats) for p in tparts])
    p = ap._pred(refm, np.ascontiguousarray(pool.select(feats).to_numpy(), dtype=np.float32))
    best = ap.exclusive_mask(pool.get_column("gid").to_numpy(), p)
    lab = np.where((p >= hi) & best, 1, np.where(p <= lo, 0, -1))
    info = {"target": target, "test_rows": pool.height, "pos": int((lab == 1).sum()), "neg": int((lab == 0).sum()),
            "dropped_middle": int((lab < 0).sum()), "hi": hi, "lo": lo}
    log(f"pseudo labels on {target} test: {info}")
    tr = pl.scan_parquet(WD / s2.OUT / "train" / "*.parquet").filter(pl.col("fold").is_in(list(config.TRAIN_FOLDS))) \
        .select(["rid", "gid", "fold", "y"] + feats).collect()
    pool = (pool.with_columns(y=pl.Series(lab)).filter(pl.col("y") >= 0)
                .with_columns(pl.col("y").cast(tr.schema["y"]), fold=pl.lit(-1, dtype=tr.schema["fold"]))
                .select(tr.columns))
    tu = pl.scan_parquet(WD / s2.OUT / "train" / "*.parquet").filter(pl.col("fold") == config.TUNE_FOLD) \
        .select(["rid", "gid", "fold", "y"] + feats).collect()
    name = f"{ref}_pl{target[:2].lower()}{tag}"
    fit_tu = tu
    if mix_dense:   # density_train --mix recipe: + dense-world train folds, early stopping on the dense tune fold
        import density_train as dt
        dn = pl.scan_parquet(WD / dt.DENSE / "train" / "*.parquet").select([pl.col(c).cast(tr.schema[c]) for c in tr.columns])
        tr = pl.concat([tr, dn.filter(pl.col("fold").is_in(list(config.TRAIN_FOLDS))).collect()])
        fit_tu = dn.filter(pl.col("fold") == config.TUNE_FOLD).collect()
        log(f"mix_dense: training rows {tr.height:,} (+ {pool.height:,} pseudo), dense tune rows {fit_tu.height:,}")
    with stage(f"pseudo_apply_{name}"):
        m = fit(pl.concat([tr, pool]), fit_tu, feats)
        del tr, pool
        m.save_model(str(WD / "models" / f"{name}.txt"))
        X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
        ev = pl.scan_parquet(WD / s2.OUT / "train" / "*.parquet").filter(pl.col("fold") == config.EVAL_FOLD) \
            .select(["rid", "gid", "fold", "y"] + feats).collect()
        p_tu, p_ev = ap._pred(m, X(tu)), ap._pred(m, X(ev))
        g_tu, g_ev = tu.get_column("gid").to_numpy(), ev.get_column("gid").to_numpy()
        ex = ap.exclusive_mask(np.r_[g_tu, g_ev], np.r_[p_tu, p_ev])
        ch = ap.choose_decision(make_evaluator(tu, tu.get_column("y").to_numpy(), config.TUNE_FOLD), p_tu, g_tu, ex[:len(g_tu)])
        v = max(ch, key=lambda k: ch[k]["tune_f05"])
        tau = ch[v]["tau"]
        e = make_evaluator(ev, ev.get_column("y").to_numpy(), config.EVAL_FOLD).report(ap.decide(p_ev, g_ev, v, tau, ex[len(g_tu):]))
        info.update({"model": name, "variant": v, "tau": tau, "tune_f05": ch[v]["tune_f05"], "eval": e,
                     "best_iteration": m.best_iteration})
        log(f"{name}: tune {ch[v]['tune_f05']} ({v} tau {tau}) | normal eval F0.5 {e['macro_f05']} P {e['macro_precision']} "
            f"R {e['macro_recall']} FP {e['false_positive_pairs']}")
        if mix_dense:
            s1d, dropped = dt.dropped_s1(0.19)
            info["dense"] = dt._score(m, feats, dt.DENSE, config.EVAL_FOLD, config.TUNE_FOLD, True, dropped, s1d, tau_normal=tau)
            log(f"{name}: dense tune {info['dense']['tune']['tune_f05']} | dense eval at tau {tau}: "
                f"{info['dense']['eval_at_tau']['macro_f05']} FP {info['dense']['eval_at_tau']['false_positive_pairs']}")
        del tu, ev, fit_tu
        metas, ps = [], []
        for part in sorted(q for q in (WD / s2.OUT / "test").glob("*.parquet") if not q.name.startswith(target)) + tparts:
            d = pl.read_parquet(part, columns=["rid", "gid"] + feats)
            ps.append(ap._pred(m, X(d)))
            metas.append(d.select("rid", "gid"))
        meta, pt = pl.concat(metas), np.concatenate(ps)
        keep = ap.decide(pt, meta.get_column("gid").to_numpy(), v, tau)
        pairs = meta.with_columns(p=pl.Series(pt), keep=pl.Series(keep))
        pairs.write_parquet(WD / "reports" / f"test_{name}_pairs.parquet")
        by = pairs.join(s1t, on="rid").group_by("country").agg(pred=pl.col("keep").sum(), mass=pl.col("p").sum()) \
            .join(s1t.group_by("country").len("S1"), on="country")
        info["test_pred_per_s1"] = {r["country"]: round(r["pred"] / r["S1"], 4) for r in by.iter_rows(named=True)}
        info["test_mass_per_s1"] = {r["country"]: round(r["mass"] / r["S1"], 4) for r in by.iter_rows(named=True)}
        log(f"test pred/S1 {info['test_pred_per_s1']} | mass/S1 {info['test_mass_per_s1']}")
        if out_dir:
            config.OUT_DIR = Path(out_dir).resolve()
            write_submission(pairs.select("rid", "gid"), keep)
            info["matched_not_in_candidates"] = redecode.subset_check(config.OUT_DIR)
            save_json({"model": name, "variant": v, "tau": tau, "tune_f05": ch[v]["tune_f05"],
                       "pseudo": {k: info[k] for k in ("target", "pos", "neg", "hi", "lo")}}, config.OUT_DIR / "decision.json")
    save_json(info, WD / "reports" / f"pseudo_apply_{name}.json")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        b = argparse.ArgumentParser()
        b.add_argument("cmd")
        b.add_argument("--ref", default="s2lab_all")
        b.add_argument("--target", default="France")
        b.add_argument("--hi", type=float, default=0.95)
        b.add_argument("--lo", type=float, default=0.05)
        b.add_argument("--out", default=None)
        b.add_argument("--tag", default="")
        b.add_argument("--target-src", default=None, help="work dir holding the target country's s2lab test parts")
        b.add_argument("--mix-dense", action="store_true", help="also train on density_train's dense-world rows")
        y = b.parse_args()
        return apply(y.ref, y.target, y.hi, y.lo, y.out, y.tag, y.target_src, y.mix_dense)
    a = argparse.ArgumentParser()
    a.add_argument("--ref", default="s2lab_all")
    a.add_argument("--target", default="India")
    a.add_argument("--source", default="US")
    a.add_argument("--hi", default="0.95", help="positive threshold, or a per-round schedule e.g. 0.99,0.97,0.95")
    a.add_argument("--lo", type=float, default=0.05)
    a.add_argument("--rounds", type=int, default=1)
    a.add_argument("--pseudo-weight", type=float, default=1.0)
    a.add_argument("--no-oracle", action="store_true")
    a.add_argument("--tag", default="")
    x = a.parse_args()
    feats = lgb.Booster(model_file=str(config.WORK_DIR / "models" / f"{x.ref}.txt")).feature_name()
    tf, uf, ef = list(config.TRAIN_FOLDS), config.TUNE_FOLD, config.EVAL_FOLD
    by = lambda d, c: d.filter(pl.col("country") == c)
    out = {"ref": x.ref, "source": x.source, "target": x.target, "hi": x.hi, "lo": x.lo}
    with stage(f"pseudo_label_{x.target}"):
        train = country_rows(feats, tf)
        tune = country_rows(feats, [uf])
        ev = by(country_rows(feats, [ef]), x.target)
        src_tr, src_tu, tgt_tr, tgt_tu = by(train, x.source), by(tune, x.source), by(train, x.target), by(tune, x.target)
        base = fit(src_tr, src_tu, feats)
        out["base"] = report(base, feats, src_tu, ev, x.source, x.target)
        log(f"base ({x.source} only): {x.target} eval F0.5 {out['base']['target_eval']['macro_f05']}")
        # self-training: each round pseudo-labels the target's unlabeled train+tune folds with the
        # latest model (confidence schedule x.hi per round) and retrains source + pseudo rows
        pool0 = pl.concat([tgt_tr, tgt_tu])
        Xpool = np.ascontiguousarray(pool0.select(feats).to_numpy(), dtype=np.float32)
        gpool, ytrue = pool0.get_column("gid").to_numpy(), pool0.get_column("y").to_numpy()
        his = [float(h) for h in str(x.hi).split(",")]
        his += [his[-1]] * (x.rounds - len(his))
        cur, out["rounds"] = base, []
        for r in range(x.rounds):
            p = ap._pred(cur, Xpool)
            best = ap.exclusive_mask(gpool, p)
            lab = np.where((p >= his[r]) & best, 1, np.where(p <= x.lo, 0, -1))
            keep = lab >= 0
            pool = pool0.filter(pl.Series(keep)).with_columns(y=pl.Series(lab[keep]).cast(src_tr.schema["y"]))
            yt, yp = ytrue[keep], lab[keep]
            info = {"round": r + 1, "hi": his[r], "rows": int(keep.sum()), "dropped_middle": int((~keep).sum()),
                    "pos": int(yp.sum()), "pos_precision": round(float(yt[yp == 1].mean()), 5),
                    "pos_recall_of_true": round(float((yp[yt == 1] == 1).mean()), 5) if (yt == 1).any() else None,
                    "neg_error": round(float(yt[yp == 0].mean()), 5)}
            w = None
            if x.pseudo_weight != 1.0:
                w = np.r_[np.ones(src_tr.height), np.full(pool.height, x.pseudo_weight)]
            cur = fit(pl.concat([src_tr, pool]), src_tu, feats, weight=w)
            info["target_eval"] = report(cur, feats, src_tu, ev, x.source, x.target)
            out["rounds"].append(info)
            log(f"round {r + 1} (hi {his[r]}): labels pos {info['pos']:,} prec {info['pos_precision']} "
                f"neg_err {info['neg_error']} -> {x.target} eval F0.5 {info['target_eval']['target_eval']['macro_f05']}")
        out["pseudo"] = out["rounds"][0]["target_eval"]
        if x.no_oracle:
            save_json(out, config.WORK_DIR / "reports" / f"pseudo_label_{x.target}{x.tag}.json")
            return
        orc = fit(pl.concat([src_tr, tgt_tr]), pl.concat([src_tu, tgt_tu]), feats)
        out["oracle"] = report(orc, feats, src_tu, ev, x.source, x.target)
        log(f"oracle (true {x.target} labels): {x.target} eval F0.5 {out['oracle']['target_eval']['macro_f05']}")
    save_json(out, config.WORK_DIR / "reports" / f"pseudo_label_{x.target}{x.tag}.json")


if __name__ == "__main__":
    main()
