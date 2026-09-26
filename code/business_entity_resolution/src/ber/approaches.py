"""The compared matchers. All use the same candidates (blocking.py), the same
S1-grouped split, the same metric and the same decision procedure; only the matcher
differs.

A1  independent pair classifier: LightGBM on PAIR_FEATURES (the two records only).
A2  relational / collective classifier:
      stage 1: LightGBM on PAIR_FEATURES + CONTEXT_FEATURES, cross-fitted
               (fold-A model scores fold B and vice versa -> out-of-fold p1);
      stage 2: LightGBM on stage-1 inputs + p1 context within the S1's candidate list
               + sibling support (does this target resemble the S1's other strongest
               target?).

Decision (both): keep pairs with p >= tau; variant "threshold+excl" additionally keeps
a target only for its highest-scoring S1 (every train target has <= 1 S1). tau and the
variant are chosen on TUNE_FOLD and reported on EVAL_FOLD.

Memory: training folds are materialized as float32 arrays; everything else streams
over the per-part feature files (each part contains complete S1 candidate lists).
"""
import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config
from .features import CONTEXT_FEATURES, PAIR_FEATURES
from .metric import Evaluator
from .utils import done, log, mark_done, save_json, stage

MODEL_DIR = config.WORK_DIR / "models"
P1_FEATURES = ["p1", "p1_qrank", "p1_qgap", "p1_qsum", "p1_qn50", "p1_qmax_other",
               "sib_p1", "sib_n_tset", "sib_a_tset", "sib_same_src", "sib2_n_tset", "sib2_a_tset"]
S1_FEATS = PAIR_FEATURES + CONTEXT_FEATURES
S2_FEATS = S1_FEATS + P1_FEATURES
TAG = config.RUN_TAG
META = ["rid", "gid", "fold"]


# ---------------------------------------------------------------- data access
def parts(split, sub="feat"):
    return sorted((config.WORK_DIR / sub / split).glob("*.parquet"))


def extra_features():
    """Names of the optional extra feature columns (config.EXTRA_SETS), in load order."""
    names = []
    for s in config.EXTRA_SETS:
        f = next((config.WORK_DIR / s / "train").glob("*.parquet"))
        names += [c for c in pl.read_parquet_schema(f) if c not in ("rid", "gid", "fold")]
    return names


def _with_extras(df: pl.DataFrame, split, folds=None, part_name=None) -> pl.DataFrame:
    """Append row-aligned extra feature sets; keys are verified, never assumed."""
    for s in config.EXTRA_SETS:
        src = config.WORK_DIR / s / split / (part_name or "*.parquet")
        lf = pl.scan_parquet(src)
        if folds is not None:
            lf = lf.filter(pl.col("fold").is_in(list(folds)))
        x = lf.collect()
        if not (x.get_column("rid").equals(df.get_column("rid")) and x.get_column("gid").equals(df.get_column("gid"))):
            raise ValueError(f"extra feature set {s} is not row-aligned with the base features")
        df = df.hstack(x.drop("rid", "gid", "fold"))
    return df


def load_xy(split, folds, feats, sub="feat"):
    """(meta, X, y) for the requested folds; X is a C-contiguous float32 array.

    Filled part by part into a preallocated array, so peak memory is X plus one part
    (a whole-frame load followed by to_numpy would transiently double it).
    """
    extras = extra_features()
    base = [f for f in feats if f not in extras] if sub == "feat" else feats
    fl = pl.col("fold").is_in(list(folds))
    ps = parts(split, sub)
    sizes = [pl.scan_parquet(p).filter(fl).select(pl.len()).collect().item() for p in ps]
    X = np.empty((sum(sizes), len(feats)), dtype=np.float32)
    metas, ys, o = [], [], 0
    for p, n in zip(ps, sizes):
        if n == 0:
            continue
        df = pl.read_parquet(p, columns=META + ["y"] + base)
        if sub == "feat" and extras:
            df = _with_extras(df, split, part_name=p.name)
        df = df.filter(fl)
        X[o:o + n] = df.select(feats).to_numpy()
        metas.append(df.select(META))
        ys.append(df.get_column("y").to_numpy())
        o += n
    return pl.concat(metas), X, np.concatenate(ys)


def train_lgb(Xtr, ytr, Xva, yva, feats, name):
    """Fit LightGBM with early stopping on the tuning fold; save the model."""
    with stage(f"train_{name}", {"rows": len(ytr), "features": len(feats)}):
        dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(100)])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        m.save_model(str(MODEL_DIR / f"{name}.txt"))
        log(f"{name}: best iteration {m.best_iteration}, valid logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    return m


def _pred(m, X):
    return m.predict(X, num_threads=config.N_THREADS).astype(np.float32)


def load_model(name):
    return lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))


# ---------------------------------------------------------------- stage-2 features
def add_stage2_features(df: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Append P1_FEATURES to df (needs rid, gid, p1). Row order is preserved.

    Sibling = the S1's highest-p1 candidate other than this target. Noisy copies of one
    business share a corrupted base, so a true copy tends to resemble its siblings; a
    planted distractor (same name, nearby house number) tends not to.
    """
    df = df.with_row_index("_row").with_columns(
        p1_qrank=pl.col("p1").rank("ordinal", descending=True).over("rid").cast(pl.Float32),
        p1_qgap=pl.col("p1").max().over("rid") - pl.col("p1"),
        p1_qsum=pl.col("p1").sum().over("rid"),
        p1_qn50=(pl.col("p1") > 0.5).sum().over("rid").cast(pl.Float32),
    )
    top = (df.select("rid", "gid", "p1").sort(["rid", "p1"], descending=[False, True])
             .group_by("rid", maintain_order=True)
             .agg(_g1=pl.col("gid").first(), _p1a=pl.col("p1").first(),
                  _g2=pl.col("gid").get(1, null_on_oob=True), _p1b=pl.col("p1").get(1, null_on_oob=True),
                  _g3=pl.col("gid").get(2, null_on_oob=True)))
    df = df.join(top, on="rid", how="left").with_columns(
        _sib=pl.when(pl.col("gid") == pl.col("_g1")).then(pl.col("_g2")).otherwise(pl.col("_g1")),
        _sib2=pl.when((pl.col("gid") == pl.col("_g1")) | (pl.col("gid") == pl.col("_g2")))
                .then(pl.col("_g3")).otherwise(pl.col("_g2")),
        p1_qmax_other=pl.when(pl.col("gid") == pl.col("_g1")).then(pl.col("_p1b").fill_null(0)).otherwise(pl.col("_p1a")),
    )
    t = tg.select("gid", "src", "nm", "ad")
    df = (df.join(t.select(gid="gid", _src="src", _nm="nm", _ad="ad"), on="gid", how="left")
            .join(t.select(_sib="gid", _ssrc="src", _snm="nm", _sad="ad"), on="_sib", how="left")
            .join(t.select(_sib2="gid", _s2nm="nm", _s2ad="ad"), on="_sib2", how="left")
            .join(df.select(_sib="gid", rid="rid", sib_p1="p1"), on=["rid", "_sib"], how="left")
            .sort("_row"))

    def fz(a, b):
        x = df.get_column(a).fill_null("").to_list()
        y = df.get_column(b).fill_null("").to_list()
        return pl.Series(process.cpdist(x, y, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / np.float32(100))

    has1, has2 = pl.col("_sib").is_not_null(), pl.col("_sib2").is_not_null()
    df = df.with_columns(
        sib_n_tset=fz("_nm", "_snm"), sib_a_tset=fz("_ad", "_sad"),
        sib2_n_tset=fz("_nm", "_s2nm"), sib2_a_tset=fz("_ad", "_s2ad"),
        sib_same_src=(pl.col("_src") == pl.col("_ssrc")).cast(pl.Float32),
    ).with_columns(
        [pl.when(has1).then(pl.col(c)).otherwise(None).alias(c) for c in ("sib_n_tset", "sib_a_tset", "sib_same_src")]
        + [pl.when(has2).then(pl.col(c)).otherwise(None).alias(c) for c in ("sib2_n_tset", "sib2_a_tset")]
    )
    return df.drop([c for c in df.columns if c.startswith("_")]).with_columns(
        [pl.col(c).cast(pl.Float32) for c in P1_FEATURES])


def a2_stage1_models():
    """Two cross-fitted stage-1 models (fold A -> scores fold B and vice versa)."""
    fa, fb = config.TRAIN_FOLDS
    feats = S1_FEATS + extra_features()
    _, Xtu, ytu = load_xy("train", [config.TUNE_FOLD], feats)
    ms = {}
    for tag, f in (("A", fa), ("B", fb)):
        name = f"A2{TAG}_stage1_fold{tag}"
        if (MODEL_DIR / f"{name}.txt").exists():
            ms[tag] = load_model(name)
            continue
        _, X, y = load_xy("train", [f], feats)
        ms[tag] = train_lgb(X, y, Xtu, ytu, feats, name)
        del X, y
    return ms["A"], ms["B"]


def build_stage2_parts(split, m_a, m_b):
    """Write work/feat2/<split>/*.parquet = stage-1 features + P1_FEATURES (resumable)."""
    fa, fb = config.TRAIN_FOLDS
    out = config.WORK_DIR / f"feat2{TAG}" / split
    feats = S1_FEATS + extra_features()
    out.mkdir(parents=True, exist_ok=True)
    tg = pl.read_parquet(config.WORK_DIR / "prep" / f"tg_{split}.parquet", columns=["gid", "src", "nm", "ad"])
    for p in parts(split):
        dst = out / p.name
        if done(dst):
            continue
        df = _with_extras(pl.read_parquet(p), split, part_name=p.name)
        X = np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)
        fold = df.get_column("fold").to_numpy()
        p1 = np.empty(len(fold), np.float32)
        # out-of-fold for the two training folds; averaged models everywhere else
        ia, ib = fold == fa, fold == fb
        io = ~(ia | ib)
        if ia.any():
            p1[ia] = _pred(m_b, X[ia])
        if ib.any():
            p1[ib] = _pred(m_a, X[ib])
        if io.any():
            p1[io] = 0.5 * (_pred(m_a, X[io]) + _pred(m_b, X[io]))
        del X
        df = add_stage2_features(df.with_columns(p1=pl.Series(p1)), tg)
        df.write_parquet(dst)
        mark_done(dst)
        log(f"stage-2 features {split}/{p.name}: {df.height:,} pairs")


# ---------------------------------------------------------------- decision + scoring
def exclusive_mask(gid, p):
    """True where a pair has the highest score among all pairs sharing its target."""
    order = np.lexsort((-p, gid))
    first = np.ones(len(gid), bool)
    first[1:] = gid[order][1:] != gid[order][:-1]
    m = np.zeros(len(gid), bool)
    m[order[first]] = True
    return m


def make_evaluator(meta: pl.DataFrame, y, fold: int):
    """Population = every S1 of the fold (including S1s with no candidates)."""
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_train.parquet", columns=["rid", "fold"]).filter(pl.col("fold") == fold)
    gt = pl.read_parquet(pdir / "gt_train.parquet")
    pop = (s1.select("rid").join(gt.group_by("rid").len("n_true"), on="rid", how="left")
             .with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), y, pop.get_column("n_true").to_numpy())


def choose_decision(ev: Evaluator, p, gid, excl=None):
    """Grid-search tau for plain and exclusive decisions on the tuning fold.

    `excl` may be precomputed over a larger scored graph than this fold (see evaluate)."""
    excl = exclusive_mask(gid, p) if excl is None else excl
    best = {}
    for variant, base in (("threshold", np.ones_like(excl)), ("threshold+excl", excl)):
        scores = [(ev.macro_f05(base & (p >= t)), t) for t in config.THRESHOLD_GRID]
        f, t = max(scores)
        best[variant] = {"tau": t, "tune_f05": round(f, 5),
                         "curve": {str(tt): round(ff, 5) for ff, tt in scores[::5]}}
    return best


def decide(p, gid, variant, tau, excl=None):
    keep = p >= tau
    if variant == "threshold+excl":
        keep &= exclusive_mask(gid, p) if excl is None else excl
    return keep


def _score_fold(approach, fold, models):
    """(meta, y, p) for one fold, streaming over parts."""
    if approach == 1:
        meta, X, y = load_xy("train", [fold], PAIR_FEATURES + extra_features())
        return meta, y, _pred(models, X)
    meta, X, y = load_xy("train", [fold], S2_FEATS + extra_features(), sub=f"feat2{TAG}")
    return meta, y, _pred(models, X)


# ---------------------------------------------------------------- evaluate
def evaluate(approach: int):
    """Fit on TRAIN_FOLDS, choose the decision on TUNE_FOLD, report on EVAL_FOLD."""
    tf, uf, ef = config.TRAIN_FOLDS, config.TUNE_FOLD, config.EVAL_FOLD
    (config.WORK_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with stage(f"evaluate_A{approach}{TAG}"):
        if approach == 1:
            feats = PAIR_FEATURES + extra_features()
            _, Xtu, ytu = load_xy("train", [uf], feats)
            _, Xtr, ytr = load_xy("train", tf, feats)
            m = train_lgb(Xtr, ytr, Xtu, ytu, feats, f"A1{TAG}")
            del Xtr, ytr, Xtu, ytu
        else:
            feats = S2_FEATS + extra_features()
            m_a, m_b = a2_stage1_models()
            with stage("A2_stage2_features_train"):
                build_stage2_parts("train", m_a, m_b)
            _, Xtu, ytu = load_xy("train", [uf], feats, sub=f"feat2{TAG}")
            _, Xtr, ytr = load_xy("train", tf, feats, sub=f"feat2{TAG}")
            m = train_lgb(Xtr, ytr, Xtu, ytu, feats, f"A2{TAG}_stage2")
            del Xtr, ytr, Xtu, ytu
        importance = dict(zip(feats, m.feature_importance("gain").round().astype(int).tolist()))

        meta_tu, y_tu, p_tu = _score_fold(approach, uf, m)
        meta_ev, y_ev, p_ev = _score_fold(approach, ef, m)
        g_tu, g_ev = meta_tu.get_column("gid").to_numpy(), meta_ev.get_column("gid").to_numpy()
        # Exclusivity over the union of both out-of-sample folds: on test a target competes
        # with every S1 of its country, so a single-fold graph under-states exclusivity.
        # (Still covers only 20% of S1; training-fold scores are in-sample and not used.)
        excl_all = exclusive_mask(np.r_[g_tu, g_ev], np.r_[p_tu, p_ev])
        excl_tu, excl_ev = excl_all[:len(g_tu)], excl_all[len(g_tu):]
        choice = choose_decision(make_evaluator(meta_tu, y_tu, uf), p_tu, g_tu, excl_tu)
        variant = max(choice, key=lambda v: choice[v]["tune_f05"])
        tau = choice[variant]["tau"]
        other = "threshold" if variant != "threshold" else "threshold+excl"

        ev = make_evaluator(meta_ev, y_ev, ef)
        keep = decide(p_ev, g_ev, variant, tau, excl_ev)
        res = {
            "approach": approach,
            "n_features": len(feats),
            "decision_selected_on_tune_fold": {"variant": variant, "tau": tau, "all": choice},
            "eval_fold": ef,
            "eval": ev.report(keep),
            f"eval_{other}": ev.report(decide(p_ev, g_ev, other, choice[other]["tau"], excl_ev)),
            "eval_blocking_oracle": ev.report(y_ev == 1),
            "eval_candidate_pairs": int(len(y_ev)),
            "feature_gain_top20": dict(sorted(importance.items(), key=lambda kv: -kv[1])[:20]),
        }
        # scored pairs of the tune and eval folds, for the decoder lab (experiments/decoder_lab.py)
        pl.concat([
            pl.DataFrame({"rid": meta_tu.get_column("rid"), "gid": meta_tu.get_column("gid"),
                          "fold": pl.Series([uf] * len(y_tu), dtype=pl.Int8), "y": y_tu, "p": p_tu}),
            pl.DataFrame({"rid": meta_ev.get_column("rid"), "gid": meta_ev.get_column("gid"),
                          "fold": pl.Series([ef] * len(y_ev), dtype=pl.Int8), "y": y_ev, "p": p_ev}),
        ]).write_parquet(config.WORK_DIR / "reports" / f"scores_A{approach}{TAG}.parquet")
    save_json(res, config.WORK_DIR / "reports" / f"eval_A{approach}{TAG}.json")
    log(f"A{approach}{TAG} eval: {json.dumps(res['eval'])}")
    return res


# ---------------------------------------------------------------- test prediction
def predict(approach: int):
    rep = json.load(open(config.WORK_DIR / "reports" / f"eval_A{approach}{TAG}.json"))
    variant, tau = rep["decision_selected_on_tune_fold"]["variant"], rep["decision_selected_on_tune_fold"]["tau"]
    with stage(f"predict_test_A{approach}{TAG}"):
        if approach == 1:
            m, sub, feats = load_model(f"A1{TAG}"), "feat", PAIR_FEATURES + extra_features()
        else:
            m_a, m_b = load_model(f"A2{TAG}_stage1_foldA"), load_model(f"A2{TAG}_stage1_foldB")
            build_stage2_parts("test", m_a, m_b)
            m, sub, feats = load_model(f"A2{TAG}_stage2"), f"feat2{TAG}", S2_FEATS + extra_features()
        metas, ps = [], []
        for p in parts("test", sub):
            df = pl.read_parquet(p)
            if sub == "feat":
                df = _with_extras(df, "test", part_name=p.name)
            ps.append(_pred(m, np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)))
            metas.append(df.select("rid", "gid"))
        meta, p = pl.concat(metas), np.concatenate(ps)
        keep = decide(p, meta.get_column("gid").to_numpy(), variant, tau)
        meta.with_columns(p=pl.Series(p), keep=pl.Series(keep)).write_parquet(
            config.WORK_DIR / "reports" / f"test_A{approach}{TAG}_pairs.parquet")
        write_submission(meta, keep)
    return run_validator()


def _write_lists(path, s1, pairs, col, tg_ids):
    lists = (pairs.join(tg_ids, on="gid").group_by("rid")
                  .agg(pl.col("entity_id").unique().sort().str.join(",")))
    out = (s1.join(lists, on="rid", how="left").sort("rid")
             .select(source1_entity_id="s1_id", **{col: pl.col("entity_id").fill_null("")}))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def write_submission(pairs: pl.DataFrame, keep: np.ndarray):
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_test.parquet", columns=["rid", "entity_id"]).rename({"entity_id": "s1_id"})
    tg_ids = pl.read_parquet(pdir / "tg_test.parquet", columns=["gid", "entity_id"])
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_lists(config.OUT_DIR / "candidate_pairs.tsv", s1, pairs, "candidate_entity_ids", tg_ids)
    m = _write_lists(config.OUT_DIR / "matching_results.tsv", s1, pairs.filter(pl.Series(keep)),
                     "matched_entity_ids", tg_ids)
    n_s1 = int((m.get_column("matched_entity_ids") != "").sum())
    log(f"submission: {s1.height:,} S1 rows, {int(keep.sum()):,} matched pairs ({keep.sum() / s1.height:.2f}/S1), "
        f"{n_s1:,} S1 with >=1 match ({n_s1 / s1.height:.3f}), {pairs.height:,} candidate pairs")


def run_validator():
    cmd = [sys.executable, str(config.VALIDATOR), "--matching", str(config.OUT_DIR / "matching_results.tsv"),
           "--candidate", str(config.OUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(config.DATA_DIR / "test"),
           "--check-ids"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"validator exit {r.returncode}:\n" + r.stdout[-3000:] + r.stderr[-2000:])
    return r.returncode
