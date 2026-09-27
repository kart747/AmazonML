"""E-G3: test-world simulation per country (label-free facts of the test split -> train folds).

Why: the 19% random S1 drop (density.py / density_train.py) matches test's orphans per S1 but
not its S1 pool. Test US has 663k S1 vs 1.32M in the train pool (0.50x) with ~2.3 orphans per
S1. Per region that is the train world with HALF of the S1 entities deleted TOGETHER with
their true copies while the orphans stay (1.22 / 0.5 = 2.44 orphans per S1). In the 19% sim the
S1 pool is 0.81x as dense, so reverse-view context (r_rev, t_is_best, rev_margin, t_gap12)
is far less shifted than on test. India (810k vs 883k S1, 2.36 orphans per S1) cannot be
matched by deletion alone; the 19% orphanizing drop reproduces its orphan density.

  US:    drop 50% of S1, DELETE their true targets from every candidate list / reverse list
  India: drop 19% of S1, their true targets stay as orphans (= density_train world)

Everything pool-dependent is recomputed: reverse ranks/best/margins, S1 name frequencies, and
(after deleting targets) the S1-side forward ranks r_name/r_addr/r_combo, q_rank/q_gap/q_ncand;
then stage 1 (OOF on training folds), p1 context, and the stage2_lab features/pruning.

  build     -> <work>/s2lab_world/train/<part>.parquet  (folds 0-3)
  score     the given s2lab models on the world eval fold at their normal tau (+ world-tuned tau)
  evaluate  density_train-style training in the world (monkeypatched onto density_train)

  BER_WORK_DIR=../../work_xd python src/experiments/world_sim.py build --stage1-tag _xd \
      --stage1-extra dn_cos --extra featx,featdn
  BER_WORK_DIR=../../work_xd python src/experiments/world_sim.py score --models s2lab_all,s2lab_dense_mix
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
import density_train as dt  # noqa: E402
import stage2_lab as s2  # noqa: E402
from ber import config  # noqa: E402
from ber.utils import done, log, mark_done, save_json, stage  # noqa: E402
from density import dense_context  # noqa: E402

WORLD = "s2lab_world"
DROP = {"US": (0.50, "delete"), "India": (0.19, "orphan")}


def world_drop():
    """Dropped S1 (per-country fraction, fixed seed) and the targets deleted with them."""
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    rng = np.random.default_rng(config.SEED + 7)
    u = pl.Series(rng.random(s1.height))
    frac = s1.get_column("country").replace_strict({c: f for c, (f, _) in DROP.items()}, default=0.0)
    dropped = s1.filter(u < frac).select("rid", "country")
    delete_c = [c for c, (_, how) in DROP.items() if how == "delete"]
    gt = pl.read_parquet(config.WORK_DIR / "prep" / "gt_train.parquet")
    deleted = gt.join(dropped.filter(pl.col("country").is_in(delete_c)), on="rid", how="semi").select("gid").unique()
    return s1, dropped.select("rid"), deleted


def s1_side_context(df):
    """Recompute forward ranks and S1-side candidate context after targets were deleted."""
    upd = []
    for v in ("name", "addr", "combo"):
        c = f"r_{v}"
        upd.append(pl.when(pl.col(c) < 99)
                   .then((pl.col(c).rank("ordinal").over(["rid", pl.col(c) < 99]) - 1).cast(pl.Float32))
                   .otherwise(pl.col(c)).alias(c))
    return df.with_columns(upd).with_columns(
        q_rank=pl.col("cos_combo").rank("ordinal", descending=True).over("rid").cast(pl.Float32),
        q_gap=(pl.col("cos_combo").max().over("rid") - pl.col("cos_combo")).cast(pl.Float32),
        q_ncand=pl.len().over("rid").cast(pl.Float32))


def build(stage1_tag, stage1_extra, extras):
    WD, folds = config.WORK_DIR, [0, 1, 2, 3]
    out = WD / WORLD / "train"
    out.mkdir(parents=True, exist_ok=True)
    s1full = pl.read_parquet(WD / "prep" / "s1_train.parquet", columns=["rid", "fold", "country", "nm", "ad"])
    _, dropped, deleted = world_drop()
    log(f"world: dropped {dropped.height:,} S1, deleted {deleted.height:,} targets "
        f"{s1full.join(dropped, on='rid', how='semi').group_by('country').len().rows()}")
    kept = s1full.join(dropped, on="rid", how="anti")
    freq_name = (kept.group_by(["country", "nm"]).len("t_nm_freq")
                     .join(kept.group_by("country").len("n_c"), on="country")
                     .with_columns(t_nm_freq=pl.col("t_nm_freq") * 1e6 / pl.col("n_c"))
                     .drop("n_c").rename({"nm": "t_nm"}))
    freq_rid = kept.join(freq_name.rename({"t_nm": "nm", "t_nm_freq": "q_nm_freq"}), on=["country", "nm"]) \
        .select("rid", "country", "q_nm_freq")
    need = (pl.scan_parquet(WD / "feat" / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
              .select("gid").unique().collect().join(deleted, on="gid", how="anti"))
    revs = []
    for f in sorted((WD / "block" / "train").glob("rev_*.parquet")):
        r = pl.read_parquet(f).join(need, on="gid", how="semi").join(dropped, on="rid", how="anti")
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
        with stage(f"world_build_{p.stem}"):
            df = s2.hstack_extras(pl.read_parquet(p), "train", p.name, extras)
            df = (df.filter(pl.col("fold").is_in(folds)).join(dropped, on="rid", how="anti")
                    .join(deleted, on="gid", how="anti"))
            cols = df.columns
            df = s1_side_context(dense_context(df, rev_rank, rev_sum, freq_rid, freq_name, tg_nm)).select(cols)
            X = np.ascontiguousarray(df.select(feats1).to_numpy(), dtype=np.float32)
            fold = df.get_column("fold").to_numpy()
            p1 = np.empty(len(fold), np.float32)
            ia, ib = fold == fa, fold == fb
            io = ~(ia | ib)
            if ia.any():
                p1[ia] = ap._pred(m_b, X[ia])
            if ib.any():
                p1[ib] = ap._pred(m_a, X[ib])
            if io.any():
                p1[io] = 0.5 * (ap._pred(m_a, X[io]) + ap._pred(m_b, X[io]))
            del X
            df = ap.add_stage2_features(df.with_columns(p1=pl.Series(p1)), tg)
            df = s2.augment(df, tgx, s1x)
            df.write_parquet(dst)
            mark_done(dst)
            log(f"world {p.name}: {df.height:,} pruned pairs")


def _patch_dt():
    """Point density_train's dense world at this module's world."""
    s1, dropped, _ = world_drop()
    dt.DENSE = WORLD
    dt.dropped_s1 = lambda frac: (pl.read_parquet(config.WORK_DIR / "prep" / "s1_train.parquet",
                                                  columns=["rid", "fold"]), dropped)


def score(models):
    """Each model on the world eval fold (fold 0), per country, at its normal tau and a world-tuned tau."""
    _patch_dt()
    WD = config.WORK_DIR
    s1c = pl.read_parquet(WD / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    s1, dropped = dt.dropped_s1(0)
    out = {}
    for name in models:
        m = lgb.Booster(model_file=str(WD / "models" / f"{name}.txt"))
        tau = json.load(open(WD / "reports" / f"{name}.json"))["decision"]["tau"] \
            if (WD / "reports" / f"{name}.json").exists() else 0.68
        res = dt._score(m, m.feature_name(), WORLD, config.EVAL_FOLD, config.TUNE_FOLD, True, dropped, s1, tau)
        # per-country at the normal tau
        mev, Xev = dt._load(WORLD, [config.EVAL_FOLD], m.feature_name())
        mtu, Xtu = dt._load(WORLD, [config.TUNE_FOLD], m.feature_name())
        p_ev, p_tu = ap._pred(m, Xev), ap._pred(m, Xtu)
        g = np.r_[mtu.get_column("gid").to_numpy(), mev.get_column("gid").to_numpy()]
        ex = ap.exclusive_mask(g, np.r_[p_tu, p_ev])[len(p_tu):]
        keep = ap.decide(p_ev, mev.get_column("gid").to_numpy(), "threshold+excl", tau, ex)
        by = {}
        for c in ("US", "India"):
            sc = s1c.filter(pl.col("country") == c).select("rid")
            msk = mev.select("rid").join(sc.with_columns(_c=pl.lit(True)), on="rid", how="left")["_c"].fill_null(False).to_numpy()
            s1_c = s1.join(sc, on="rid", how="semi")
            ev = dt._dense_evaluator(mev.filter(pl.Series(msk)), mev.get_column("y").to_numpy()[msk],
                                     config.EVAL_FOLD, dropped, s1_c)
            by[c] = ev.report(keep[msk])
        res["by_country_at_tau"] = by
        out[name] = res
        log(f"WORLD {name}: normal-tau {tau} eval {res['eval_at_tau']['macro_f05']} "
            f"(US {by['US']['macro_f05']}, India {by['India']['macro_f05']}) | world-tuned tau "
            f"{res['tune']['tau']} eval {res['eval']['macro_f05']} | pairs/S1 {res['eval_at_tau']['predicted_pairs_per_s1']}")
    save_json(out, WD / "reports" / "world_sim_score.json")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["build", "score", "evaluate"])
    a.add_argument("--stage1-tag", default="_xd")
    a.add_argument("--stage1-extra", default="dn_cos")
    a.add_argument("--extra", default="featx,featdn")
    a.add_argument("--models", default="s2lab_all")
    a.add_argument("--ref", default="s2lab_all")
    a.add_argument("--mix", action="store_true")
    a.add_argument("--tag", default="world")
    x = a.parse_args()
    if x.cmd == "build":
        build(x.stage1_tag, [c for c in x.stage1_extra.split(",") if c], [c for c in x.extra.split(",") if c])
    elif x.cmd == "score":
        score([m for m in x.models.split(",") if m])
    else:
        _patch_dt()
        dt.evaluate(x.ref, x.mix, 0.0, x.tag)


if __name__ == "__main__":
    main()
