"""The two compared approaches. Both use the same candidates (blocking.py), the same
S1-grouped split, the same metric and the same threshold procedure. Only the matcher
differs:

Approach 1: independent pair classifier
    LightGBM on PAIR_FEATURES (similarity of the two records only).

Approach 2: relational / collective classifier
    Stage 1: LightGBM on PAIR_FEATURES + CONTEXT_FEATURES (candidate-graph context),
             cross-fitted (fold 1 model scores fold 2 and vice versa -> out-of-fold p1).
    Stage 2: LightGBM on stage-1 inputs + p1 + p1 context within the S1's candidate
             list + sibling support (does this target agree with the S1's other
             strongest target?).

Decision (both): keep pairs with p >= tau; the "excl" variant also keeps a target only
for its highest-scoring S1 (every train target has at most one S1). tau and the
variant are chosen on TUNE_FOLD and reported on EVAL_FOLD.
"""
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config
from .features import CONTEXT_FEATURES, PAIR_FEATURES
from .metric import Evaluator
from .utils import log, save_json, stage

MODEL_DIR = config.WORK_DIR / "models"
P1_FEATURES = ["p1", "p1_qrank", "p1_qgap", "p1_qsum", "p1_qn50", "p1_qmax_other",
               "sib_p1", "sib_n_tset", "sib_a_tset", "sib_same_src", "sib2_n_tset", "sib2_a_tset"]


# ---------------------------------------------------------------- data access
def load_pairs(split, folds=None, columns=None):
    lf = pl.scan_parquet(config.WORK_DIR / "feat" / split / "*.parquet")
    if folds is not None:
        lf = lf.filter(pl.col("fold").is_in(list(folds)))
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


def _xy(df, feats):
    return df.select(feats).to_numpy().astype(np.float32), df.get_column("y").to_numpy()


def train_lgb(tr, va, feats, name):
    """Fit LightGBM with early stopping on the tuning fold; save the model."""
    with stage(f"train_{name}", {"rows": tr.height, "features": len(feats)}):
        Xtr, ytr = _xy(tr, feats)
        Xva, yva = _xy(va, feats)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(100)])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        m.save_model(str(MODEL_DIR / f"{name}.txt"))
        log(f"{name}: best iteration {m.best_iteration}")
    return m


def score(m, df, feats):
    return m.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=config.N_THREADS)


# ---------------------------------------------------------------- stage-2 features
def add_stage2_features(df: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """p1 context within each S1's candidate list + sibling-support features.

    Sibling = the S1's highest-p1 candidate other than this target. Noisy copies of one
    business share a corrupted base, so a true copy tends to resemble its siblings;
    a planted distractor tends not to.
    """
    df = df.with_columns(
        p1_qrank=pl.col("p1").rank("ordinal", descending=True).over("rid").cast(pl.Float32),
        p1_qgap=pl.col("p1").max().over("rid") - pl.col("p1"),
        p1_qsum=pl.col("p1").sum().over("rid"),
        p1_qn50=(pl.col("p1") > 0.5).sum().over("rid").cast(pl.Float32),
    )
    top = (df.select("rid", "gid", "p1").sort(["rid", "p1"], descending=[False, True])
             .group_by("rid", maintain_order=True)
             .agg(g1=pl.col("gid").first(), p_1=pl.col("p1").first(),
                  g2=pl.col("gid").get(1, null_on_oob=True), p_2=pl.col("p1").get(1, null_on_oob=True),
                  g3=pl.col("gid").get(2, null_on_oob=True)))
    df = df.join(top, on="rid", how="left").with_columns(
        sib=pl.when(pl.col("gid") == pl.col("g1")).then(pl.col("g2")).otherwise(pl.col("g1")),
        sib2=pl.when((pl.col("gid") == pl.col("g1")) | (pl.col("gid") == pl.col("g2")))
               .then(pl.col("g3")).otherwise(pl.col("g2")),
        p1_qmax_other=pl.when(pl.col("gid") == pl.col("g1")).then(pl.col("p_2").fill_null(0)).otherwise(pl.col("p_1")),
    )
    t = tg.select("gid", "src", "nm", "ad")
    df = (df.join(t.select(gid="gid", _src="src", _nm="nm", _ad="ad"), on="gid", how="left")
            .join(t.select(sib="gid", _ssrc="src", _snm="nm", _sad="ad"), on="sib", how="left")
            .join(t.select(sib2="gid", _s2nm="nm", _s2ad="ad"), on="sib2", how="left")
            .join(df.select(sib="gid", rid="rid", sib_p1="p1"), on=["rid", "sib"], how="left"))

    def fz(a, b):
        x = df.get_column(a).fill_null("").to_list()
        y = df.get_column(b).fill_null("").to_list()
        return process.cpdist(x, y, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / np.float32(100)

    has1, has2 = df.get_column("sib").is_not_null(), df.get_column("sib2").is_not_null()
    df = df.with_columns(
        sib_n_tset=pl.Series(fz("_nm", "_snm")), sib_a_tset=pl.Series(fz("_ad", "_sad")),
        sib2_n_tset=pl.Series(fz("_nm", "_s2nm")), sib2_a_tset=pl.Series(fz("_ad", "_s2ad")),
        sib_same_src=(pl.col("_src") == pl.col("_ssrc")).cast(pl.Float32),
    ).with_columns(
        [pl.when(has1).then(pl.col(c)).otherwise(None).alias(c) for c in ("sib_n_tset", "sib_a_tset", "sib_same_src", "sib_p1")]
        + [pl.when(has2).then(pl.col(c)).otherwise(None).alias(c) for c in ("sib2_n_tset", "sib2_a_tset")]
    )
    return df.drop([c for c in df.columns if c.startswith("_")] + ["g1", "g2", "g3", "p_1", "p_2", "sib", "sib2"])


# ---------------------------------------------------------------- decision + scoring
def exclusive_mask(gid, p):
    """True where a pair has the highest score among all pairs sharing its target."""
    order = np.lexsort((-p, gid))
    first = np.ones(len(gid), bool)
    first[1:] = gid[order][1:] != gid[order][:-1]
    m = np.zeros(len(gid), bool)
    m[order[first]] = True
    return m


def make_evaluator(df_fold: pl.DataFrame, fold: int):
    """Population = every S1 of the fold (including those with no candidates)."""
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_train.parquet", columns=["rid", "fold"]).filter(pl.col("fold") == fold)
    gt = pl.read_parquet(pdir / "gt_train.parquet")
    pop = s1.select("rid").join(gt.group_by("rid").len("n_true"), on="rid", how="left") \
        .with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx")
    idx = df_fold.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), df_fold.get_column("y").to_numpy(),
                     pop.get_column("n_true").to_numpy())


def choose_decision(ev: Evaluator, p, gid):
    """Grid-search tau for plain and exclusive decisions on the tuning fold."""
    excl = exclusive_mask(gid, p)
    best = {}
    for variant, base in (("threshold", np.ones_like(excl)), ("threshold+excl", excl)):
        scores = [(ev.macro_f05(base & (p >= t)), t) for t in config.THRESHOLD_GRID]
        f, t = max(scores)
        best[variant] = {"tau": t, "tune_f05": round(f, 5), "curve": {str(tt): round(ff, 5) for ff, tt in scores[::5]}}
    return best


def decide(p, gid, variant, tau):
    keep = p >= tau
    if variant == "threshold+excl":
        keep &= exclusive_mask(gid, p)
    return keep


# ---------------------------------------------------------------- evaluate
def evaluate(approach: int):
    """Fit on TRAIN_FOLDS, choose decision on TUNE_FOLD, report on EVAL_FOLD."""
    tf, uf, ef = config.TRAIN_FOLDS, config.TUNE_FOLD, config.EVAL_FOLD
    base_cols = ["rid", "gid", "fold", "y"]
    with stage(f"evaluate_A{approach}") as _:
        if approach == 1:
            feats = PAIR_FEATURES
            df = load_pairs("train", tf + (uf, ef), base_cols + feats)
            tr, tu, ev_df = (df.filter(pl.col("fold").is_in(list(tf))), df.filter(pl.col("fold") == uf),
                             df.filter(pl.col("fold") == ef))
            del df
            m = train_lgb(tr, tu, feats, "A1")
            del tr
            p_tu, p_ev = score(m, tu, feats), score(m, ev_df, feats)
            importance = dict(zip(feats, m.feature_importance("gain").round().astype(int).tolist()))
        else:
            feats1 = PAIR_FEATURES + CONTEXT_FEATURES
            df = load_pairs("train", tf + (uf, ef), base_cols + feats1)
            f_a, f_b = tf
            da, db = df.filter(pl.col("fold") == f_a), df.filter(pl.col("fold") == f_b)
            tu = df.filter(pl.col("fold") == uf)
            m_a = train_lgb(da, tu, feats1, "A2_stage1_foldA")
            m_b = train_lgb(db, tu, feats1, "A2_stage1_foldB")
            # out-of-fold p1 for the training folds; averaged models for tune/eval folds
            df = df.with_columns(p1=pl.lit(0.0, pl.Float32))
            p1 = np.zeros(df.height, np.float32)
            fold = df.get_column("fold").to_numpy()
            X = df.select(feats1).to_numpy().astype(np.float32)
            p1[fold == f_a] = m_b.predict(X[fold == f_a], num_threads=config.N_THREADS)
            p1[fold == f_b] = m_a.predict(X[fold == f_b], num_threads=config.N_THREADS)
            other = (fold != f_a) & (fold != f_b)
            p1[other] = 0.5 * (m_a.predict(X[other], num_threads=config.N_THREADS)
                               + m_b.predict(X[other], num_threads=config.N_THREADS))
            del X, da, db, tu
            df = df.with_columns(p1=pl.Series(p1))
            tg = pl.read_parquet(config.WORK_DIR / "prep" / "tg_train.parquet", columns=["gid", "src", "nm", "ad"])
            with stage("A2_stage2_features"):
                df = add_stage2_features(df, tg)
            del tg
            feats = feats1 + P1_FEATURES
            tr, tu, ev_df = (df.filter(pl.col("fold").is_in(list(tf))), df.filter(pl.col("fold") == uf),
                             df.filter(pl.col("fold") == ef))
            del df
            m = train_lgb(tr, tu, feats, "A2_stage2")
            del tr
            p_tu, p_ev = score(m, tu, feats), score(m, ev_df, feats)
            importance = dict(zip(feats, m.feature_importance("gain").round().astype(int).tolist()))

        ev_tu = make_evaluator(tu, uf)
        choice = choose_decision(ev_tu, p_tu, tu.get_column("gid").to_numpy())
        variant = max(choice, key=lambda v: choice[v]["tune_f05"])
        tau = choice[variant]["tau"]
        ev_ev = make_evaluator(ev_df, ef)
        g_ev = ev_df.get_column("gid").to_numpy()
        res = {
            "approach": approach,
            "features": len(feats),
            "decision_selected_on_tune_fold": {"variant": variant, "tau": tau, "all": choice},
            "eval_fold": ef,
            "eval": ev_ev.report(decide(p_ev, g_ev, variant, tau)),
            "eval_other_variant": ev_ev.report(decide(
                p_ev, g_ev, "threshold" if variant != "threshold" else "threshold+excl",
                choice["threshold" if variant != "threshold" else "threshold+excl"]["tau"])),
            "eval_oracle_all_candidates": ev_ev.report(ev_df.get_column("y").to_numpy() == 1),
            "candidate_pairs_eval": ev_df.height,
            "candidates_per_s1_eval": round(ev_df.height / ev_ev.n, 2),
            "feature_gain_top15": dict(sorted(importance.items(), key=lambda kv: -kv[1])[:15]),
        }
    save_json(res, config.WORK_DIR / "reports" / f"eval_A{approach}.json")
    log(f"A{approach} eval: {res['eval']}")
    return res


# ---------------------------------------------------------------- test prediction
def predict(approach: int):
    import json
    rep = json.load(open(config.WORK_DIR / "reports" / f"eval_A{approach}.json"))
    variant, tau = rep["decision_selected_on_tune_fold"]["variant"], rep["decision_selected_on_tune_fold"]["tau"]
    base_cols = ["rid", "gid"]
    with stage(f"predict_test_A{approach}"):
        if approach == 1:
            feats = PAIR_FEATURES
            df = load_pairs("test", None, base_cols + feats)
            m = lgb.Booster(model_file=str(MODEL_DIR / "A1.txt"))
            p = m.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=config.N_THREADS)
        else:
            feats1 = PAIR_FEATURES + CONTEXT_FEATURES
            df = load_pairs("test", None, base_cols + feats1)
            X = df.select(feats1).to_numpy().astype(np.float32)
            p1 = 0.5 * (lgb.Booster(model_file=str(MODEL_DIR / "A2_stage1_foldA.txt")).predict(X, num_threads=config.N_THREADS)
                        + lgb.Booster(model_file=str(MODEL_DIR / "A2_stage1_foldB.txt")).predict(X, num_threads=config.N_THREADS))
            del X
            df = df.with_columns(p1=pl.Series(p1.astype(np.float32)))
            tg = pl.read_parquet(config.WORK_DIR / "prep" / "tg_test.parquet", columns=["gid", "src", "nm", "ad"])
            df = add_stage2_features(df, tg)
            del tg
            feats = feats1 + P1_FEATURES
            m = lgb.Booster(model_file=str(MODEL_DIR / "A2_stage2.txt"))
            p = m.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=config.N_THREADS)
        keep = decide(p, df.get_column("gid").to_numpy(), variant, tau)
        write_submission(df.select("rid", "gid"), keep)
    run_validator()


def _write_lists(path, s1, pairs, col):
    lists = pairs.join(pairs.select("gid").unique().join(
        pl.read_parquet(config.WORK_DIR / "prep" / "tg_test.parquet", columns=["gid", "entity_id"]),
        on="gid"), on="gid").group_by("rid").agg(pl.col("entity_id").unique().sort().str.join(","))
    out = s1.join(lists, on="rid", how="left").sort("rid").select(
        source1_entity_id="s1_id", **{col: pl.col("entity_id").fill_null("")})
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def write_submission(pairs: pl.DataFrame, keep: np.ndarray):
    s1 = pl.read_parquet(config.WORK_DIR / "prep" / "s1_test.parquet", columns=["rid", "entity_id"]).rename({"entity_id": "s1_id"})
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_lists(config.OUT_DIR / "candidate_pairs.tsv", s1, pairs, "candidate_entity_ids")
    m = _write_lists(config.OUT_DIR / "matching_results.tsv", s1, pairs.filter(pl.Series(keep)), "matched_entity_ids")
    n_pred = m.get_column("matched_entity_ids").str.len_chars().gt(0).sum()
    log(f"submission: {s1.height:,} S1 rows, {int(keep.sum()):,} matched pairs "
        f"({keep.sum() / s1.height:.2f}/S1), {n_pred:,} S1 with >=1 match, {pairs.height:,} candidate pairs")


def run_validator():
    cmd = [sys.executable, str(config.VALIDATOR), "--matching", str(config.OUT_DIR / "matching_results.tsv"),
           "--candidate", str(config.OUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log("validator:\n" + r.stdout[-3000:] + r.stderr[-2000:])
    return r.returncode
