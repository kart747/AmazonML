"""Density-augmented stage-2 training: learn from test-like distractor density.

Why: test has 5.75 targets per S1 against 4.68 in train. The density simulation (density.py,
stage2_lab density: drop 19% of train S1 at random, so their true targets become owner-less
copies) costs the best model 0.0024 macro F0.5. A public two-stage pipeline reports a
0.015 validation -> leaderboard drop. Re-selecting tau under density recovers nothing, so the model
itself has to learn what an orphan copy with a same-name "twin" S1 looks like. Here stage 2
is trained on the dense world.

  build     for every S1 fold 0-3: drop the same 19% of S1 as density.py (config.SEED), recompute
            every pool-dependent context feature (reverse ranks/scores, best-S1 flags, name
            frequencies), re-score stage 1 (out-of-fold for the two training folds, the averaged
            model elsewhere), recompute the stage-2 p1/sibling features and the stage2_lab
            features/pruning -> <work>/s2lab_dense/train/<part>.parquet
  evaluate  train stage 2 (stage2_lab `all` features) on dense folds 1,2 [+ normal folds 1,2 with
            --mix], early-stop on dense fold 3, choose tau on dense fold 3, and report on
            (a) dense fold 0 (kept S1 population) and (b) normal fold 0 with the normal-tune tau,
            next to the reference model scored the same way.

    BER_WORK_DIR=../work_xd python src/experiments/density_train.py build --stage1-tag _xd \
        --stage1-extra dn_cos --extra featx,featdn
    BER_WORK_DIR=../work_xd python src/experiments/density_train.py evaluate --ref s2lab_all [--mix]
Outputs: <work>/s2lab_dense/, models/s2lab_dense_<name>.txt, reports/density_train_<name>.json
"""
import argparse
import json
import os
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
from ber.utils import done, log, mark_done, save_json, stage  # noqa: E402
from density import dense_context  # noqa: E402

DENSE = "s2lab_dense"


def dropped_s1(frac):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "fold"])
    rng = np.random.default_rng(config.SEED)            # the same mask as density.py / stage2_lab density
    return s1, s1.select("rid").filter(pl.Series(rng.random(s1.height) < frac))


def build(frac, stage1_tag, stage1_extra, extras):
    WD, folds = config.WORK_DIR, [0, 1, 2, 3]
    out = WD / DENSE / "train"
    out.mkdir(parents=True, exist_ok=True)
    s1full = pl.read_parquet(WD / "prep" / "s1_train.parquet", columns=["rid", "fold", "country", "nm", "ad"])
    _, dropped = dropped_s1(frac)
    kept = s1full.join(dropped, on="rid", how="anti")
    freq_name = (kept.group_by(["country", "nm"]).len("t_nm_freq")
                     .join(kept.group_by("country").len("n_c"), on="country")
                     .with_columns(t_nm_freq=pl.col("t_nm_freq") * 1e6 / pl.col("n_c"))
                     .drop("n_c").rename({"nm": "t_nm"}))
    freq_rid = kept.join(freq_name.rename({"t_nm": "nm", "t_nm_freq": "q_nm_freq"}), on=["country", "nm"]) \
        .select("rid", "country", "q_nm_freq")
    revs = []
    for f in sorted((WD / "block" / "train").glob("rev_*.parquet")):
        r = pl.read_parquet(f).join(dropped, on="rid", how="anti")
        revs.append(r.with_columns(rrank=pl.col("rscore").rank("ordinal", descending=True).over("gid") - 1))
    rev = pl.concat(revs).filter(pl.col("rrank") < config.BLOCK["k_rev"])
    del revs
    rev_rank = rev.select("gid", "rid", r_rev="rrank")
    rev_sum = (rev.sort(["gid", "rrank"]).group_by("gid", maintain_order=True)
                  .agg(t_best_rid=pl.col("rid").first(), t_top1=pl.col("rscore").first(),
                       t_top2=pl.col("rscore").get(1, null_on_oob=True)))
    del rev
    tgx = s2.target_table("train")
    tg = tgx.select("gid", src="_src", nm="_tnm", ad="_tad")
    tg_nm = tg.select("gid", t_nm="nm")
    s1x = s1full.select("rid", "nm", "ad")
    m_a, m_b = ap.load_model(f"A2{stage1_tag}_stage1_foldA"), ap.load_model(f"A2{stage1_tag}_stage1_foldB")
    feats1 = ap.S1_FEATS + stage1_extra
    fa, fb = config.TRAIN_FOLDS
    for p in ap.parts("train"):
        dst = out / p.name
        if done(dst):
            continue
        with stage(f"dense_build_{p.stem}"):
            df = s2.hstack_extras(pl.read_parquet(p), "train", p.name, extras)
            df = df.filter(pl.col("fold").is_in(folds)).join(dropped, on="rid", how="anti")
            cols = df.columns
            df = dense_context(df, rev_rank, rev_sum, freq_rid, freq_name, tg_nm).select(cols)
            X = np.ascontiguousarray(df.select(feats1).to_numpy(), dtype=np.float32)
            fold = df.get_column("fold").to_numpy()
            p1 = np.empty(len(fold), np.float32)
            ia, ib = fold == fa, fold == fb
            io = ~(ia | ib)
            if ia.any():
                p1[ia] = ap._pred(m_b, X[ia])        # out-of-fold, as in approaches.build_stage2_parts
            if ib.any():
                p1[ib] = ap._pred(m_a, X[ib])
            if io.any():
                p1[io] = 0.5 * (ap._pred(m_a, X[io]) + ap._pred(m_b, X[io]))
            del X
            df = ap.add_stage2_features(df.with_columns(p1=pl.Series(p1)), tg)
            df = s2.augment(df, tgx, s1x)
            df.write_parquet(dst)
            mark_done(dst)
            log(f"dense {p.name}: {df.height:,} pruned pairs")


def _load(sub, folds, feats):
    lf = pl.scan_parquet(config.WORK_DIR / sub / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
    df = lf.select(["rid", "gid", "fold", "y"] + feats).collect()
    return df.select("rid", "gid", "fold", "y"), np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)


def _dense_evaluator(meta, y, fold, dropped, s1):
    gt = pl.read_parquet(config.WORK_DIR / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter(pl.col("fold") == fold).join(dropped, on="rid", how="anti").select("rid")
             .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), y, pop.get_column("n_true").to_numpy())


def _score(m, feats, sub, ev_fold, tu_fold, dense, dropped, s1, tau_normal=None):
    """Tune tau on tu_fold, report ev_fold, in the dense or the normal world."""
    mtu, Xtu = _load(sub, [tu_fold], feats)
    mev, Xev = _load(sub, [ev_fold], feats)
    p_tu, p_ev = ap._pred(m, Xtu), ap._pred(m, Xev)
    y_tu, y_ev = mtu.get_column("y").to_numpy(), mev.get_column("y").to_numpy()
    g_tu, g_ev = mtu.get_column("gid").to_numpy(), mev.get_column("gid").to_numpy()
    ex = ap.exclusive_mask(np.r_[g_tu, g_ev], np.r_[p_tu, p_ev])
    ex_tu, ex_ev = ex[:len(g_tu)], ex[len(g_tu):]
    mk = (lambda mm, yy, ff: _dense_evaluator(mm, yy, ff, dropped, s1)) if dense else ap.make_evaluator
    ch = ap.choose_decision(mk(mtu, y_tu, tu_fold), p_tu, g_tu, ex_tu)
    v = max(ch, key=lambda k: ch[k]["tune_f05"])
    ev = mk(mev, y_ev, ev_fold)
    res = {"tune": {"variant": v, "tau": ch[v]["tau"], "tune_f05": ch[v]["tune_f05"]},
           "eval": ev.report(ap.decide(p_ev, g_ev, v, ch[v]["tau"], ex_ev))}
    if tau_normal is not None:
        res["eval_at_tau"] = {"tau": tau_normal, **ev.report(ap.decide(p_ev, g_ev, "threshold+excl", tau_normal, ex_ev))}
    return res


def evaluate(ref, mix, frac, tag):
    WD = config.WORK_DIR
    s1, dropped = dropped_s1(frac)
    refm = lgb.Booster(model_file=str(WD / "models" / f"{ref}.txt"))
    feats = refm.feature_name()
    uf, ef = config.TUNE_FOLD, config.EVAL_FOLD
    name = f"dense{'_mix' if mix else ''}{tag}"
    with stage(f"density_train_{name}"):
        mtr, Xtr = _load(DENSE, list(config.TRAIN_FOLDS), feats)
        if mix:
            m2, X2 = _load(s2.OUT, list(config.TRAIN_FOLDS), feats)
            mtr, Xtr = pl.concat([mtr, m2]), np.concatenate([Xtr, X2])
            del X2
        mtu, Xtu = _load(DENSE, [uf], feats)
        log(f"{name}: training rows {len(mtr):,}, dense tune rows {len(mtu):,}")
        dtr = lgb.Dataset(Xtr, mtr.get_column("y").to_numpy(), feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xtu, mtu.get_column("y").to_numpy(), reference=dtr)
        m = lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
        m.save_model(str(WD / "models" / f"s2lab_{name}.txt"))
        del Xtr, dtr, Xtu, dva
        out = {"ref": ref, "mix": mix, "drop_frac": frac, "best_iteration": m.best_iteration}
        for label, model in (("new", m), ("ref", refm)):
            out[label] = {
                "normal": _score(model, feats, s2.OUT, ef, uf, False, dropped, s1),
                "dense": _score(model, feats, DENSE, ef, uf, True, dropped, s1),
            }
            # dense fold 0 at the tau that model would ship with (normal-world tune choice)
            out[label]["dense"]["eval_at_normal_tau"] = _score(
                model, feats, DENSE, ef, uf, True, dropped, s1,
                tau_normal=out[label]["normal"]["tune"]["tau"])["eval_at_tau"]
    save_json(out, WD / "reports" / f"density_train_{name}.json")
    for label in ("ref", "new"):
        o = out[label]
        log(f"{label:3} normal: tune {o['normal']['tune']['tune_f05']} tau {o['normal']['tune']['tau']} eval "
            f"{o['normal']['eval']['macro_f05']} FP {o['normal']['eval']['false_positive_pairs']} | dense: tune "
            f"{o['dense']['tune']['tune_f05']} eval(dense tau {o['dense']['tune']['tau']}) {o['dense']['eval']['macro_f05']} "
            f"eval(normal tau) {o['dense']['eval_at_normal_tau']['macro_f05']} FP {o['dense']['eval_at_normal_tau']['false_positive_pairs']}")
    return out


def predict(name, world, out):
    """Score the stage2_lab test parts with models/s2lab_<name>.txt; tau from the chosen world's tune fold.
    Writes reports/test_s2lab_<name>_pairs.parquet and a full submission (candidates = scored pairs)."""
    WD = config.WORK_DIR
    rep = json.load(open(WD / "reports" / f"density_train_{name}.json"))["new"][world]["tune"]
    m = lgb.Booster(model_file=str(WD / "models" / f"s2lab_{name}.txt"))
    df = pl.scan_parquet(WD / s2.OUT / "test" / "*.parquet").select(["rid", "gid"] + m.feature_name()).collect()
    p = ap._pred(m, np.ascontiguousarray(df.select(m.feature_name()).to_numpy(), dtype=np.float32))
    keep = ap.decide(p, df.get_column("gid").to_numpy(), rep["variant"], rep["tau"])
    pairs = df.select("rid", "gid").with_columns(p=pl.Series(p), keep=pl.Series(keep))
    pairs.write_parquet(WD / "reports" / f"test_s2lab_{name}_pairs.parquet")
    config.OUT_DIR = Path(out).resolve()
    ap.write_submission(pairs.select("rid", "gid"), keep)
    save_json({"model": f"s2lab_{name}", "tau_world": world, **rep, "matched_pairs": int(keep.sum())},
              config.OUT_DIR / "decision.json")
    log(f"predict s2lab_{name}: tau {rep['tau']} ({world} tune) -> {config.OUT_DIR}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["build", "evaluate", "predict"])
    a.add_argument("--world", choices=["normal", "dense"], default="normal", help="predict: tau from this tune fold")
    a.add_argument("--name", default="dense", help="predict: model name (models/s2lab_<name>.txt)")
    a.add_argument("--out", default=None)
    a.add_argument("--drop", type=float, default=0.19)
    a.add_argument("--stage1-tag", default="")
    a.add_argument("--stage1-extra", default="")
    a.add_argument("--extra", default="")
    a.add_argument("--ref", default="s2lab_all")
    a.add_argument("--mix", action="store_true", help="train on dense + normal training folds")
    a.add_argument("--tag", default="")
    x = a.parse_args()
    if x.cmd == "build":
        build(x.drop, x.stage1_tag, [c for c in x.stage1_extra.split(",") if c], [s for s in x.extra.split(",") if s])
    elif x.cmd == "evaluate":
        evaluate(x.ref, x.mix, x.drop, x.tag)
    else:
        assert x.out, "--out is required (never overwrite output/)"
        predict(x.name, x.world, x.out)


if __name__ == "__main__":
    main()
