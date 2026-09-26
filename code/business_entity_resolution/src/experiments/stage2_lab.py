"""Stage-2 lab: cascade-pruned A2 stage-2 variants with extra p1-aware features.

Why: on the eval fold, A2's gap to the blocking oracle (0.0139) is entirely matcher error
on pairs that are already candidates (18.2k true pairs scored below tau, mostly partial
misses inside multi-match S1). Decoding is saturated (decoder_lab: best rule +0.0002),
so the scores must improve.

Cascade: stage-2 rows with p1 < PRUNE (1e-3) are rejected outright. On the train folds
this keeps 11.4% of rows and 99.99% of positives. On fold 0, no pair with p1 < 1e-3 reached
p >= 0.67 under the full A2. Stage-2 training therefore fits in ~2 GB instead of 9.8 GB.

New features (all label-free and computed on the S1's FULL candidate list before pruning,
so they are identical in construction on train and test):
  cluster support vs the S1's confident set A = {other candidates with p1 >= 0.5}:
    cs_n_max, cs_n_mean, cs_a_max, cs_a_mean   token-set similarity of this target to A
    cs_nsp_max                                 space-stripped name ratio vs A (glued names)
    cs_a_exact, cs_n_exact                     members of A with an identical address / name
    cs_n                                       |A|
  twins, twin_p1max      other candidates of this S1 with identical (name, address)
  p1_srank, p1_sn50      p1 rank / confident count within (S1, source)
  n_nsp_ratio, n_nsp_partial   space-stripped name similarity (cvrindustries ~ cvr industries)
  a_exact, a_contain     identical / contained normalized address (non-empty)
  t_oov                  share of target name tokens absent from every S1 name of the country
                         (coined DBA names such as "caloflux", phonetic transliterations)

Usage:
  python src/experiments/stage2_lab.py build --split train [--extra featx]
  python src/experiments/stage2_lab.py evaluate --variant base|new|featx|all [--tag x]
  python src/experiments/stage2_lab.py predict --variant all --out ../output_variants/<name>
Outputs: work/s2lab/<split>/*.parquet, reports/s2lab_<variant><tag>.json, models/s2lab_*.txt
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import (S2_FEATS, _pred, choose_decision, decide, exclusive_mask,  # noqa: E402
                            make_evaluator, run_validator, write_submission)
from ber.utils import done, log, mark_done, save_json, stage  # noqa: E402

PRUNE = 1e-3
CONF = 0.5
OUT = "s2lab"
NEW_FEATS = ["cs_n_max", "cs_n_mean", "cs_a_max", "cs_a_mean", "cs_nsp_max", "cs_a_exact", "cs_n_exact", "cs_n",
             "twins", "twin_p1max", "p1_srank", "p1_sn50",
             "n_nsp_ratio", "n_nsp_partial", "a_exact", "a_contain", "t_oov"]
W = config.WORKERS if hasattr(config, "WORKERS") else 8


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32) / np.float32(100)


def _s1_vocab(split):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / f"s1_{split}.parquet", columns=["country", "nm"])
    return s1.select("country", tok=pl.col("nm").str.split(" ")).explode("tok").unique()


def target_table(split):
    """Target name/address/source plus t_oov (share of name tokens absent from all S1 names)."""
    WD = config.WORK_DIR
    tg = pl.read_parquet(WD / "prep" / f"tg_{split}.parquet", columns=["gid", "country", "src", "nm", "ad"])
    vocab = _s1_vocab(split).with_columns(_inv=pl.lit(True))
    toov = (tg.select("gid", "country", tok=pl.col("nm").str.split(" ")).explode("tok")
              .join(vocab, on=["country", "tok"], how="left")
              .group_by("gid").agg(t_oov=pl.col("_inv").is_null().mean().cast(pl.Float32)))
    return (tg.join(toov, on="gid", how="left").drop("country")
              .rename({"src": "_src", "nm": "_tnm", "ad": "_tad"}))


def augment(df, tgx, s1):
    """feat2-style frame (full S1 candidate lists, with p1) -> pruned frame + NEW_FEATS."""
    nsp = lambda c: pl.col(c).str.replace_all(" ", "")
    df = (df.with_row_index("_row")
            .join(tgx, on="gid", how="left", maintain_order="left")
            .join(s1.select("rid", _qnm="nm", _qad="ad"), on="rid", how="left", maintain_order="left"))
    # full-list context (before pruning)
    df = df.with_columns(
        twins=(pl.len().over(["rid", "_tnm", "_tad"]) - 1).cast(pl.Float32),
        p1_srank=pl.col("p1").rank("ordinal", descending=True).over(["rid", "_src"]).cast(pl.Float32),
        p1_sn50=(pl.col("p1") > CONF).sum().over(["rid", "_src"]).cast(pl.Float32),
    )
    tw = df.group_by(["rid", "_tnm", "_tad"]).agg(_m1=pl.col("p1").max(), _m2=pl.col("p1").top_k(2).min(), _n=pl.len())
    df = df.join(tw, on=["rid", "_tnm", "_tad"], how="left", maintain_order="left").with_columns(
        twin_p1max=pl.when(pl.col("_n") < 2).then(None)
        .when(pl.col("p1") == pl.col("_m1")).then(pl.col("_m2")).otherwise(pl.col("_m1")).cast(pl.Float32))
    A = df.filter(pl.col("p1") >= CONF).select("rid", _ag="gid", _anm="_tnm", _aad="_tad")
    df = df.filter(pl.col("p1") >= PRUNE)
    # cluster support against A (other members only)
    j = (df.select("_row", "rid", "gid", "_tnm", "_tad").join(A, on="rid", how="inner")
           .filter(pl.col("gid") != pl.col("_ag"))
           .with_columns(_a_ne=(pl.col("_tad") != "") & (pl.col("_aad") != "")))
    j = j.with_columns(
        _sn=pl.Series(_cp(j.get_column("_tnm").to_list(), j.get_column("_anm").to_list(), fuzz.token_set_ratio)),
        _sa=pl.Series(_cp(j.get_column("_tad").to_list(), j.get_column("_aad").to_list(), fuzz.token_set_ratio)),
        _ssp=pl.Series(_cp(j.select(nsp("_tnm")).to_series().to_list(),
                           j.select(nsp("_anm")).to_series().to_list(), fuzz.ratio)),
    ).with_columns(_sa=pl.when(pl.col("_a_ne")).then(pl.col("_sa")))
    agg = j.group_by("_row").agg(
        cs_n_max=pl.col("_sn").max(), cs_n_mean=pl.col("_sn").mean(),
        cs_a_max=pl.col("_sa").max(), cs_a_mean=pl.col("_sa").mean(),
        cs_nsp_max=pl.col("_ssp").max(),
        cs_a_exact=(pl.col("_a_ne") & (pl.col("_tad") == pl.col("_aad"))).sum().cast(pl.Float32),
        cs_n_exact=(pl.col("_tnm") == pl.col("_anm")).sum().cast(pl.Float32),
        cs_n=pl.len().cast(pl.Float32))
    df = df.join(agg, on="_row", how="left", maintain_order="left").with_columns(pl.col("cs_n").fill_null(0))
    qn, tn = df.select(nsp("_qnm")).to_series().to_list(), df.select(nsp("_tnm")).to_series().to_list()
    ne = (pl.col("_qad") != "") & (pl.col("_tad") != "")
    df = df.with_columns(
        n_nsp_ratio=pl.Series(_cp(qn, tn, fuzz.ratio)), n_nsp_partial=pl.Series(_cp(qn, tn, fuzz.partial_ratio)),
        a_exact=pl.when(ne).then(pl.col("_qad") == pl.col("_tad")),
        a_contain=pl.when(ne).then(pl.col("_qad").str.contains(pl.col("_tad"), literal=True)
                                   | pl.col("_tad").str.contains(pl.col("_qad"), literal=True)),
    )
    return df.drop([c for c in df.columns if c.startswith("_")]).with_columns(
        [pl.col(c).cast(pl.Float32) for c in NEW_FEATS])


def hstack_extras(df, split, part_name, extras):
    for s in extras:
        x = pl.read_parquet(config.WORK_DIR / s / split / part_name)
        if not (x.get_column("rid").equals(df.get_column("rid")) and x.get_column("gid").equals(df.get_column("gid"))):
            raise ValueError(f"{s}/{part_name} not row-aligned")
        df = df.hstack(x.drop("rid", "gid", "fold"))
    return df


def build(split, extras):
    WD = config.WORK_DIR
    src_dir, out_dir = WD / "feat2" / split, WD / OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    tgx = target_table(split)
    s1 = pl.read_parquet(WD / "prep" / f"s1_{split}.parquet", columns=["rid", "nm", "ad"])
    for p in sorted(src_dir.glob("*.parquet")):
        dst = out_dir / p.name
        if done(dst):
            continue
        with stage(f"s2lab_build_{split}_{p.stem}"):
            df = augment(hstack_extras(pl.read_parquet(p), split, p.name, extras), tgx, s1)
            df.write_parquet(dst)
            mark_done(dst)
            log(f"s2lab {split}/{p.name}: kept {df.height:,} pairs")


def _extra_cols(split):
    f = next((config.WORK_DIR / OUT / split).glob("*.parquet"))
    cols = list(pl.read_parquet_schema(f))
    known = set(S2_FEATS) | set(NEW_FEATS) | {"rid", "gid", "fold", "y"}
    return [c for c in cols if c not in known]


def variant_feats(variant, split="train"):
    extra = _extra_cols(split)
    return {"base": S2_FEATS, "new": S2_FEATS + NEW_FEATS, "featx": S2_FEATS + extra,
            "all": S2_FEATS + NEW_FEATS + extra}[variant]


def load(split, folds, feats, base=None):
    lf = pl.scan_parquet((Path(base) if base else config.WORK_DIR) / OUT / split / "*.parquet")
    if folds is not None:
        lf = lf.filter(pl.col("fold").is_in(list(folds)))
    cols = ["rid", "gid", "fold"] + (["y"] if split == "train" else []) + feats
    df = lf.select(cols).collect()
    X = np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)
    return df.select(["rid", "gid", "fold"] + (["y"] if split == "train" else [])), X


def evaluate(variant, tag, params_over, train_folds=None, more_train=()):
    """more_train: other work dirs whose s2lab/train rows (any fold) are added to the training set;
    they must hold only S1 folds disjoint from the tune/eval folds (e.g. BER_QUERY_FOLDS=4..9)."""
    feats = variant_feats(variant)
    name = f"s2lab_{variant}{tag}"
    tf, uf, ef = train_folds or config.TRAIN_FOLDS, config.TUNE_FOLD, config.EVAL_FOLD
    params = dict(config.LGB_PARAMS, **params_over)
    with stage(f"{name}_evaluate"):
        mtr, Xtr = load("train", tf, feats)
        for d in more_train:
            m2, X2 = load("train", None, feats, base=d)
            assert not set(m2.get_column("fold").unique().to_list()) & {uf, ef}, "extra dir overlaps tune/eval folds"
            mtr, Xtr = pl.concat([mtr, m2]), np.concatenate([Xtr, X2])
            del X2
        log(f"training rows {len(mtr):,} (folds {sorted(mtr.get_column('fold').unique().to_list())})")
        mtu, Xtu = load("train", [uf], feats)
        dtr = lgb.Dataset(Xtr, mtr.get_column("y").to_numpy(), feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xtu, mtu.get_column("y").to_numpy(), reference=dtr)
        m = lgb.train(params, dtr, config.LGB_ROUNDS * 2, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
        (config.WORK_DIR / "models").mkdir(exist_ok=True)
        m.save_model(str(config.WORK_DIR / "models" / f"{name}.txt"))
        del Xtr, dtr
        mev, Xev = load("train", [ef], feats)
        p_tu, p_ev = _pred(m, Xtu), _pred(m, Xev)
        y_tu, y_ev = mtu.get_column("y").to_numpy(), mev.get_column("y").to_numpy()
        g_tu, g_ev = mtu.get_column("gid").to_numpy(), mev.get_column("gid").to_numpy()
        ex = exclusive_mask(np.r_[g_tu, g_ev], np.r_[p_tu, p_ev])
        ex_tu, ex_ev = ex[:len(g_tu)], ex[len(g_tu):]
        choice = choose_decision(make_evaluator(mtu, y_tu, uf), p_tu, g_tu, ex_tu)
        variant_d = max(choice, key=lambda v: choice[v]["tune_f05"])
        tau = choice[variant_d]["tau"]
        ev = make_evaluator(mev, y_ev, ef)
        keep = decide(p_ev, g_ev, variant_d, tau, ex_ev)
        imp = dict(zip(feats, m.feature_importance("gain").round().astype(int).tolist()))
        res = {"variant": variant, "tag": tag, "n_features": len(feats), "params_override": params_over,
               "train_folds": sorted(mtr.get_column("fold").unique().to_list()), "train_rows": len(mtr),
               "best_iteration": m.best_iteration, "valid_logloss": m.best_score["valid_0"]["binary_logloss"],
               "decision": {"variant": variant_d, "tau": tau, "tune_f05": choice[variant_d]["tune_f05"], "all": choice},
               "eval": ev.report(keep), "eval_pairs_scored": int(len(y_ev)),
               "gain_top25": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:25])}
        pl.DataFrame({"rid": pl.concat([mtu.get_column("rid"), mev.get_column("rid")]),
                      "gid": np.r_[g_tu, g_ev], "fold": np.r_[np.full(len(g_tu), uf), np.full(len(g_ev), ef)].astype(np.int8),
                      "y": np.r_[y_tu, y_ev], "p": np.r_[p_tu, p_ev]}).write_parquet(
            config.WORK_DIR / "reports" / f"scores_{name}.parquet")
    save_json(res, config.WORK_DIR / "reports" / f"{name}.json")
    e = res["eval"]
    log(f"{name}: tune {res['decision']['tune_f05']} ({variant_d} tau {tau}) | eval F0.5 {e['macro_f05']} "
        f"P {e['macro_precision']} R {e['macro_recall']} FP {e['false_positive_pairs']} "
        f"pairs/S1 {e['predicted_pairs_per_s1']} single {e['singleton_accuracy']} | iters {m.best_iteration}")
    return res


def predict(variant, tag, out_dir):
    name = f"s2lab_{variant}{tag}"
    rep = json.load(open(config.WORK_DIR / "reports" / f"{name}.json"))
    m = lgb.Booster(model_file=str(config.WORK_DIR / "models" / f"{name}.txt"))
    feats = m.feature_name()
    with stage(f"{name}_predict_test"):
        meta, X = load("test", None, feats)
        p = _pred(m, X)
        keep = decide(p, meta.get_column("gid").to_numpy(), rep["decision"]["variant"], rep["decision"]["tau"])
        config.OUT_DIR = Path(out_dir)
        meta.with_columns(p=pl.Series(p), keep=pl.Series(keep)).write_parquet(
            config.WORK_DIR / "reports" / f"test_{name}_pairs.parquet")
        # candidate_pairs.tsv must be the full candidate set, not only the pruned rows
        full = pl.scan_parquet(config.WORK_DIR / "feat" / "test" / "*.parquet").select("rid", "gid").collect()
        kept = meta.filter(pl.Series(keep)).select("rid", "gid").with_columns(_k=pl.lit(True))
        full = full.join(kept, on=["rid", "gid"], how="left").with_columns(pl.col("_k").fill_null(False))
        write_submission(full.select("rid", "gid"), full.get_column("_k").to_numpy())
        json.dump({"stage2_lab": name, "decision": rep["decision"], "eval": rep["eval"]},
                  open(Path(out_dir) / "decision.json", "w"), indent=2)
    return run_validator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "evaluate", "predict"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--extra", default="", help="comma-separated row-aligned feature sets, e.g. featx")
    ap.add_argument("--variant", default="new")
    ap.add_argument("--tag", default="")
    ap.add_argument("--params", default="{}", help='JSON LightGBM overrides, e.g. {"learning_rate":0.04}')
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-folds", default="", help="override TRAIN_FOLDS, e.g. 1 (learning curve)")
    ap.add_argument("--more-train", default="", help="comma-separated extra work dirs with s2lab/train")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, [s for s in a.extra.split(",") if s])
    elif a.cmd == "evaluate":
        tfs = tuple(int(x) for x in a.train_folds.split(",") if x) or None
        evaluate(a.variant, a.tag, json.loads(a.params), tfs, [d for d in a.more_train.split(",") if d])
    else:
        assert a.out, "--out is required (never overwrite output/)"
        predict(a.variant, a.tag, a.out)


if __name__ == "__main__":
    main()
