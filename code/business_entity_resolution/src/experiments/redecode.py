"""Re-decode saved test scores without recomputing anything, plus label-free test diagnostics.

`run.py predict` saves every scored test pair in reports/test_A<n><tag>_pairs.parquet
(rid, gid, p, keep). This tool turns those scores into alternative submissions in
seconds (for leaderboard A/B tests), and compares test with validation per country to
detect shift. France has no labels, so its decision can only be checked label-free.

  diagnose  per country: candidates/S1, expected matches/S1 (sum of p), predicted
            matches/S1 and empty-prediction rate across a tau sweep; test next to the
            validation eval fold (where the true matches/S1 are known).
            -> reports/test_diagnostics_A<n><tag>.json + printed table
  write     apply a decision rule and write a complete submission to --out, then run the
            official validator. The rule is recorded in <out>/decision.json.
  pruned    same matches as the saved decision (the `keep` column), but candidate_pairs.tsv =
            only the pairs the final (stage-2) model actually scored, e.g. the stage2_lab set
            after the stage-1 p1 >= 1e-3 filter (~5.5/S1 on test instead of ~50/S1).
            The organisers rank smaller candidate sets per S1 higher; the rules define the
            candidate set as "whatever your model actually runs inference over".
  estimate  label-free macro-F0.5 estimator, calibrated on the TUNE fold only.
            `estimate validate` refits on fold 3 and checks how far fold 0 lands from its
            known score (the honest error bar of the estimator).
            `estimate test` applies the frozen fit to saved test scores, so alternative
            decisions can be ranked offline instead of by burning a leaderboard slot.

Usage:
  python src/experiments/redecode.py diagnose --approach 2
  BER_WORK_DIR=../work_xd python src/experiments/redecode.py pruned --name s2lab_all --out output_variants/xd_s2lab_all_pruned
  python src/experiments/redecode.py write --approach 2 --variant threshold+excl --tau 0.80 \
         --country-tau France=0.85 --out output_variants/a2_fr085
  BER_WORK_DIR=../work_xd python src/experiments/redecode.py estimate validate --name s2lab_all
  BER_WORK_DIR=../work_xd python src/experiments/redecode.py estimate test --name s2lab_all --tau 0.68
Cost: ~1–2 min, ~3 GB RAM for the full test set.

Read the diagnostics this way. On validation, sum(p)/S1 ≈ true matches/S1 (the model is
calibrated at train density). On test the generator is presumably the same, so a country
whose predicted matches/S1 differs strongly from the others is under- or over-matched. That
is a hypothesis to test with one leaderboard submission, never something to auto-tune.

The estimator has the same standing caveat in reverse: it sees only score counts, so a
planted look-alike that the model scores at p = 0.9998 is invisible to it by construction.
Rank two test decisions with `estimate test`, but read the fold-0 error bar first.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import TAG, exclusive_mask, run_validator, write_submission  # noqa: E402
from ber.metric import f05_per_entity  # noqa: E402
from ber.utils import log, save_json  # noqa: E402

TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
EST_EPS = 0.5        # n_true_hat below this -> singleton rule (chosen on the tune fold)
EST_TAU_SWEEP = [0.40, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]



def per_country(pairs: pl.DataFrame, pop: pl.DataFrame, excl: np.ndarray) -> dict:
    """pairs: rid, gid, p[, y]; pop: rid, country[, n_true] (every S1 of the population)."""
    pairs = pairs.with_columns(excl=pl.Series(excl))
    out = {}
    for c in sorted(pop.get_column("country").unique()):
        pc = pop.filter(pl.col("country") == c)
        x = pairs.join(pc.select("rid"), on="rid")
        n = pc.height
        per = x.group_by("rid").agg(n=pl.len(), sp=pl.col("p").sum(), mx=pl.col("p").max())
        per = pc.select("rid").join(per, on="rid", how="left").fill_null(0)
        row = {
            "n_s1": n,
            "cands_per_s1": round(x.height / n, 2),
            "cands_p95": float(per.get_column("n").quantile(0.95)),
            "expected_matches_per_s1": round(float(per.get_column("sp").mean()), 3),
            "share_s1_max_p_ge_0.9": round(float((per.get_column("mx") >= 0.9).mean()), 4),
            "share_s1_max_p_lt_0.5": round(float((per.get_column("mx") < 0.5).mean()), 4),
            "tau_sweep_excl": {},
        }
        if "n_true" in pc.columns:
            row["true_matches_per_s1"] = round(float(pc.get_column("n_true").mean()), 3)
            row["true_singleton_rate"] = round(float((pc.get_column("n_true") == 0).mean()), 4)
        for t in TAUS:
            k = x.filter(pl.col("excl") & (pl.col("p") >= t))
            kept = k.group_by("rid").len()
            row["tau_sweep_excl"][str(t)] = {
                "pred_per_s1": round(k.height / n, 3),
                "empty_rate": round(1 - kept.height / n, 4),
            }
        out[c] = row
    return out


def diagnose(approach, name=None):
    W, name = config.WORK_DIR, name or f"A{approach}{TAG}"
    test = pl.read_parquet(W / "reports" / f"test_{name}_pairs.parquet", columns=["rid", "gid", "p"])
    s1t = pl.read_parquet(W / "prep" / "s1_test.parquet", columns=["rid", "country"])
    rep = {"test": per_country(test, s1t, exclusive_mask(test.get_column("gid").to_numpy(), test.get_column("p").to_numpy()))}
    sc_path = W / "reports" / f"scores_{name}.parquet"
    if sc_path.exists():
        val = pl.read_parquet(sc_path).filter(pl.col("fold") == config.EVAL_FOLD)
        s1v = (pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
                 .filter(pl.col("fold") == config.EVAL_FOLD)
                 .join(pl.read_parquet(W / "prep" / "gt_train.parquet").group_by("rid").len("n_true"), on="rid", how="left")
                 .with_columns(pl.col("n_true").fill_null(0)))
        # exclusivity on validation is computed within the eval fold only (the scores file holds
        # tune + eval folds; test competition includes every S1, so this slightly understates it)
        rep["validation_eval_fold"] = per_country(val, s1v, exclusive_mask(val.get_column("gid").to_numpy(), val.get_column("p").to_numpy()))
    save_json(rep, W / "reports" / f"test_diagnostics_{name}.json")
    for split, d in rep.items():
        print(f"\n== {split}")
        print(f"{'country':8} {'S1':>9} {'cand/S1':>8} {'E[m]/S1':>8} {'true/S1':>8} "
              + " ".join(f"pred@{t:<4}" for t in TAUS))
        for c, r in d.items():
            print(f"{c:8} {r['n_s1']:>9,} {r['cands_per_s1']:>8} {r['expected_matches_per_s1']:>8} "
                  f"{r.get('true_matches_per_s1', '-'):>8} "
                  + " ".join(f"{r['tau_sweep_excl'][str(t)]['pred_per_s1']:>8}" for t in TAUS))
    return rep


def write(approach, variant, tau, country_tau, out, name=None):
    W, name = config.WORK_DIR, name or f"A{approach}{TAG}"
    pairs = pl.read_parquet(W / "reports" / f"test_{name}_pairs.parquet", columns=["rid", "gid", "p"])
    s1t = pl.read_parquet(W / "prep" / "s1_test.parquet", columns=["rid", "country"])
    known = set(s1t.get_column("country").unique())
    for c in country_tau:
        if c not in known:
            sys.exit(f"unknown country {c!r}; test countries are {sorted(known)}")
    taus = pl.DataFrame({"country": list(country_tau), "tau": list(country_tau.values())},
                        schema={"country": pl.String, "tau": pl.Float64})
    t = (pairs.select("rid").join(s1t, on="rid", how="left", maintain_order="left")
              .join(taus, on="country", how="left", maintain_order="left")
              .get_column("tau").fill_null(tau).to_numpy())
    p, gid = pairs.get_column("p").to_numpy(), pairs.get_column("gid").to_numpy()
    keep = p >= t
    if variant == "threshold+excl":
        keep &= exclusive_mask(gid, p)
    config.OUT_DIR = Path(out).resolve()
    write_submission(pairs.select("rid", "gid"), keep)
    dec = {"source_scores": f"test_{name}_pairs.parquet", "variant": variant, "tau": tau,
           "country_tau": country_tau, "matched_pairs": int(keep.sum())}
    dec["matched_not_in_candidates"] = subset_check(config.OUT_DIR)
    save_json(dec, config.OUT_DIR / "decision.json")
    rc = run_validator() if os.environ.get("BER_SKIP_VALIDATOR") != "1" else 0
    log(f"decision {json.dumps(dec)} -> {config.OUT_DIR} (validator exit {rc})")
    return rc


def subset_check(out):
    """Hard check (the official validator only warns): every matched id is in that S1's candidates."""
    def load(f, col):
        return (pl.read_csv(Path(out) / f, separator="\t", quote_char=None, schema_overrides={col: pl.String})
                  .with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
                  .filter(pl.col(col) != "").rename({col: "id"}))
    m = load("matching_results.tsv", "matched_entity_ids")
    c = load("candidate_pairs.tsv", "candidate_entity_ids")
    bad = m.join(c, on=["source1_entity_id", "id"], how="anti").height
    log(f"subset check {out}: {m.height:,} matched ids, {c.height:,} candidate ids, {bad} matched ids not in candidates")
    return bad


def pruned(name, out):
    """Rewrite a submission with candidate_pairs = the pairs the final model scored (keep unchanged)."""
    W = config.WORK_DIR
    pairs = pl.read_parquet(W / "reports" / f"test_{name}_pairs.parquet", columns=["rid", "gid", "keep"])
    keep = pairs.get_column("keep").to_numpy()
    config.OUT_DIR = Path(out).resolve()
    write_submission(pairs.select("rid", "gid"), keep)
    n1 = pl.read_parquet(W / "prep" / "s1_test.parquet", columns=["rid"]).height
    dec = {"source_scores": f"test_{name}_pairs.parquet", "candidates": "pairs scored by the final model",
           "candidate_pairs": pairs.height, "candidates_per_s1": round(pairs.height / n1, 3),
           "matched_pairs": int(keep.sum())}
    dec["matched_not_in_candidates"] = subset_check(config.OUT_DIR)
    assert dec["matched_not_in_candidates"] == 0
    save_json(dec, config.OUT_DIR / "decision.json")
    rc = run_validator()
    log(f"pruned candidates {json.dumps(dec)} -> {config.OUT_DIR} (validator exit {rc})")
    return rc


# ---------------------------------------------------------------- label-free macro-F0.5 estimator
# The estimator never sees a label at prediction time. Everything label-dependent (the
# P(y|p) map, the missing-mass factor k, the E[F | n_pred] table) is fit on the TUNE fold
# only; the EVAL fold and the test set are then scored exactly as they would be in
# production, so `estimate validate` measures the real out-of-sample error bar.
_NBIN, _TBINS = 8, [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]


def _nb(n_pred):
    return np.minimum(n_pred, _NBIN - 1).astype(np.int64)


def _tb(w_all):
    return np.minimum(np.digitize(w_all, _TBINS), len(_TBINS)).astype(np.int64)


def _table(key_cal, val_cal, nbins, min_count=30):
    """E[F | key] on the calibration split; sparse cells fall back to the global mean."""
    bc = np.bincount(key_cal, minlength=nbins).astype(float)
    sv = np.bincount(key_cal, weights=val_cal, minlength=nbins)
    with np.errstate(invalid="ignore"):
        m = np.where(bc >= min_count, sv / np.maximum(bc, 1), np.nan)
    if np.isnan(m).any():
        m = np.where(np.isnan(m), float(np.nanmean(m)), m)
    return m


def _fit_calibrator(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p, y)
    return iso.predict


def _f05_plugin(w_kept, n_pred, n_hat, eps=EST_EPS):
    """Plug-in F0.5 with fractional counts; the singleton rule fires below `eps`."""
    tp, pr, nt = np.asarray(w_kept, float), np.asarray(n_pred, float), np.asarray(n_hat, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ = np.where(pr > 0, tp / pr, 0.0)
        r_ = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, 1.25 * p_ * r_ / (0.25 * p_ + r_), 0.0)
    return np.where(nt < eps, (pr == 0).astype(float), f)


def _est_pop(fold):
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_train.parquet", columns=["rid", "fold", "country"]).filter(pl.col("fold") == fold)
    gt = pl.read_parquet(pdir / "gt_train.parquet")
    pop = (s1.select("rid", "country").join(gt.group_by("rid").len("n_true"), on="rid", how="left")
             .with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    return pop, pop.get_column("n_true").to_numpy().astype(float)


def est_apply(fit, p, s1, excl, tau):
    """Score a population where no label exists. Returns the three estimators."""
    return _est_on_mask(fit, p, s1, excl, excl & (p >= tau))


def _est_parts(scores, fold=None):
    """Per-fold (p, s1 index, excl, y), exclusivity computed over BOTH folds at once so a
    target competes with every S1 of its country, exactly as `evaluate` does."""
    p = scores.get_column("p").to_numpy()
    excl = exclusive_mask(scores.get_column("gid").to_numpy(), p)
    fold_arr = scores.get_column("fold").to_numpy()
    y_all = scores.get_column("y").to_numpy().astype(bool) if "y" in scores.columns else None
    out = {}
    for f in ((config.TUNE_FOLD, config.EVAL_FOLD) if fold is None else (fold,)):
        m = fold_arr == f
        pop, n_true = _est_pop(f)
        s1 = (scores.filter(pl.Series(m)).select("rid").join(pop.select("rid", "idx"), on="rid",
                                                    how="left", maintain_order="left")
                    .get_column("idx").to_numpy())
        out[f] = {"p": p[m], "s1": s1, "excl": excl[m], "n_true": n_true, "pop": pop,
                  "y": y_all[m] if y_all is not None else None}
    return out


def estimate_validate(name, taus=None):
    """Fit on TUNE (fold 3), score EVAL (fold 0): the estimator's honest error bar."""
    W = config.WORK_DIR
    sc = pl.read_parquet(W / "reports" / f"scores_{name}.parquet")
    parts = _est_parts(sc)
    tu, ev = parts[config.TUNE_FOLD], parts[config.EVAL_FOLD]
    cp = _fit_calibrator(tu["p"], tu["y"])
    taus = taus or EST_TAU_SWEEP
    rows = []
    for tau in taus:
        fit = est_fit_from_parts(tu, cp, tau, tu["n_true"])
        a_tu = est_apply_parts(fit, tu, tau)
        a_ev = est_apply_parts(fit, ev, tau)
        true_ev = a_ev["true"]
        rows.append({"tau": tau, "true_tune": round(a_tu["true"], 5), "true_eval": round(true_ev, 5),
                     "est_eval_n1d": round(a_ev["n1d"], 5), "err_n1d": round(a_ev["n1d"] - true_ev, 5),
                     "est_eval_n2d": round(a_ev["n2d"], 5), "err_n2d": round(a_ev["n2d"] - true_ev, 5),
                     "est_eval_plugin": round(a_ev["plugin"], 5), "err_plugin": round(a_ev["plugin"] - true_ev, 5)})
    worst = {k: round(max(abs(r[k]) for r in rows), 5) for k in ("err_n1d", "err_n2d", "err_plugin")}
    # how well does a *difference* between two decisions survive? (the quantity that ranks submissions)
    d_true, d_est = [], []
    for i in range(1, len(rows)):
        d_true.append(rows[i]["true_eval"] - rows[i - 1]["true_eval"])
        d_est.append(rows[i]["est_eval_n1d"] - rows[i - 1]["est_eval_n1d"])
    delta = {"worst_abs_error_of_a_delta": round(float(np.max(np.abs(np.array(d_est) - np.array(d_true)))), 5),
             "resolved_at": "differences above ~0.0005 are resolvable; <=0.0002 are not"}
    country = {"est_n1d": {}, "est_n2d": {}, "true": {}}
    tau0 = None
    repj = W / "reports" / f"{name}.json"
    if repj.exists():
        tau0 = json.load(open(repj)).get("decision", {}).get("tau")
    if tau0 in [r["tau"] for r in rows]:
        fit = est_fit_from_parts(tu, cp, tau0, tu["n_true"])
        tb = len(_TBINS) + 1
        for f, tag in ((config.TUNE_FOLD, "tune"), (config.EVAL_FOLD, "eval")):
            part = parts[f]
            keep = part["excl"] & (part["p"] >= tau0)
            n = len(part["n_true"])
            n_pred = np.bincount(part["s1"][keep], minlength=n)
            w_kept = np.bincount(part["s1"][keep], weights=cp(part["p"])[keep], minlength=n)
            w_not = np.bincount(part["s1"][~keep], weights=cp(part["p"])[~keep], minlength=n)
            w_all = w_kept + w_not
            b, t = _nb(n_pred), _tb(w_all)
            plug = _f05_plugin(w_kept, n_pred, w_kept + fit["k"] * w_not)
            tp = np.bincount(part["s1"][keep & part["y"]], minlength=n)
            f_true = f05_per_entity(tp, n_pred, part["n_true"])
            ctry = part["pop"].get_column("country").to_numpy()
            for c in sorted(set(ctry)):
                m = ctry == c
                country["est_n1d"][f"{c}_{tag}"] = round(float(fit["tab1"][b][m].mean()), 5)
                country["est_n2d"][f"{c}_{tag}"] = round(float(fit["tab2"][b * tb + t][m].mean()), 5)
                country.setdefault("est_plugin", {})[f"{c}_{tag}"] = round(float(plug[m].mean()), 5)
                country["true"][f"{c}_{tag}"] = round(float(f_true[m].mean()), 5)
    rep = {"name": name, "calib_fold": config.TUNE_FOLD, "check_fold": config.EVAL_FOLD,
           "n_s1_eval": int(len(ev["n_true"])), "worst_abs_error": worst, "delta_resolution": delta,
           "per_country_at_decision_tau": country, "rows": rows}
    save_json(rep, W / "reports" / f"estimator_validate_{name}.json")
    print(f"== estimator validation: fit fold {config.TUNE_FOLD} -> score fold {config.EVAL_FOLD} ({name})")
    print(f"{'tau':>5} {'true_t':>9} {'true_e':>9} | {'n1d':>9} {'err':>8} | {'n2d':>9} {'err':>8} | {'plugin':>9} {'err':>8}")
    for r in rows:
        print(f"{r['tau']:>5} {r['true_tune']:>9} {r['true_eval']:>9} | {r['est_eval_n1d']:>9} {r['err_n1d']:>8} | "
              f"{r['est_eval_n2d']:>9} {r['err_n2d']:>8} | {r['est_eval_plugin']:>9} {r['err_plugin']:>8}")
    print(f"worst |error| over the sweep: n1d {worst['err_n1d']}, n2d {worst['err_n2d']}, plugin {worst['err_plugin']} "
          f"(gate was +/- 0.003)")
    print(f"delta resolution: worst error of a tau-to-tau *difference* = {delta['worst_abs_error_of_a_delta']}")
    if country["true"]:
        print("per country at the decision tau:")
        print(f"{'country':>22} {'true_eval':>10} {'est_n1d':>9} {'err':>8} {'est_plugin':>11} {'err':>8}")
        for k in sorted(country["true"]):
            if not k.endswith("_eval"):
                continue
            te, e1 = country["true"][k], country["est_n1d"][k]
            ep = country.get("est_plugin", {}).get(k, float("nan"))
            print(f"{k:>22} {te:>10} {e1:>9} {round(e1 - te, 5):>8} {ep:>11} {round(ep - te, 5):>8}")
    log(f"estimator validate {name}: worst {worst}, delta {delta}")
    return rep


def est_fit_from_parts(part, cp, tau, n_true):
    """Fit k and the lookup tables on an already-split (labeled) part for one tau."""
    p, y, s1, excl = part["p"], part["y"], part["s1"], part["excl"]
    c = cp(p)
    keep = excl & (p >= tau)
    n = len(n_true)
    n_pred = np.bincount(s1[keep], minlength=n)
    tp = np.bincount(s1[keep & y], minlength=n)
    w_kept = np.bincount(s1[keep], weights=c[keep], minlength=n)
    w_not = np.bincount(s1[~keep], weights=c[~keep], minlength=n)
    f_true = f05_per_entity(tp, n_pred, n_true)
    k = float((n_true.sum() - w_kept.sum()) / max(w_not.sum(), 1e-9))
    b, t = _nb(n_pred), _tb(w_kept + w_not)
    tb = _TBINS.__len__() + 1
    return {"cp": cp, "k": k, "tau": tau, "n": n, "tab1": _table(b, f_true, _NBIN),
            "tab2": _table(b * tb + t, f_true, _NBIN * tb), "tune_true": float(f_true.mean())}


def est_apply_parts(fit, part, tau):
    r = est_apply(fit, part["p"], part["s1"], part["excl"], tau)
    if part.get("y") is not None:
        y, s1, excl, p, n = part["y"], part["s1"], part["excl"], part["p"], len(part["n_true"])
        keep = excl & (p >= tau)
        r["true"] = float(f05_per_entity(np.bincount(s1[keep & y], minlength=n),
                                         np.bincount(s1[keep], minlength=n), part["n_true"]).mean())
    return r


def estimate_test(name, tau, calib=None, country_tau=None):
    """Apply the frozen tune-fold fit to saved test scores and rank decisions offline.

    Never writes a submission: it only reports an estimated macro F0.5 per decision plus
    the fold-0 error bar, so two candidates can be compared without spending a slot.
    """
    from experiments.lb_guards import attach, frhn_mask, learn_veto, veto_mask  # local: avoids an import cycle

    W = config.WORK_DIR
    src = calib or name
    sc = pl.read_parquet(W / "reports" / f"scores_{src}.parquet")
    parts = _est_parts(sc, fold=config.TUNE_FOLD)
    tu = parts[config.TUNE_FOLD]
    cp = _fit_calibrator(tu["p"], tu["y"])
    fit = est_fit_from_parts(tu, cp, tau, tu["n_true"])

    pairs = pl.read_parquet(W / "reports" / f"test_{name}_pairs.parquet", columns=["rid", "gid", "p", "keep"])
    pop = pl.read_parquet(W / "prep" / "s1_test.parquet", columns=["rid", "country"]).with_row_index("idx")
    p, keep0 = pairs.get_column("p").to_numpy(), pairs.get_column("keep").to_numpy()
    s1 = pairs.select("rid").join(pop.select("rid", "idx"), on="rid", how="left",
                                  maintain_order="left").get_column("idx").to_numpy()
    excl = exclusive_mask(pairs.get_column("gid").to_numpy(), p)
    fit["n"] = pop.height          # tables are bin-indexed, so only the array length changes

    d = attach(pairs.select("rid", "gid"), "test")
    tune_sc = sc.filter(pl.col("fold") == config.TUNE_FOLD)
    toks = learn_veto(attach(tune_sc, "train"), 50, 0.02).get_column("tok").to_list()
    veto = veto_mask(d, toks)
    frhn = frhn_mask(d)

    decisions = {"base": keep0, "veto": keep0 & ~veto, "frhn": keep0 & ~frhn,
                 "veto+frhn": keep0 & ~(veto | frhn)}
    for t in (0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80):
        decisions[f"tau{t}"] = excl & (p >= t)
    if country_tau:
        ctry = d.get_column("country").to_numpy()
        row_tau = np.full(len(ctry), tau, float)
        for k, v in country_tau.items():
            row_tau[ctry == k] = v
        decisions["ctau:" + ",".join(f"{k}={v}" for k, v in sorted(country_tau.items()))] = excl & (p >= row_tau)
    base_est = est_apply(fit, p, s1, excl, tau)

    rows = {}
    for label, kp in decisions.items():
        est = _est_on_mask(fit, p, s1, excl, kp)
        rows[label] = {"matched_pairs": int(kp.sum()), "n1d": round(est["n1d"], 5),
                       "n2d": round(est["n2d"], 5), "plugin": round(est["plugin"], 5),
                       "d_n1d": round(est["n1d"] - base_est["n1d"], 5),
                       "d_plugin": round(est["plugin"] - base_est["plugin"], 5)}

    # label-free per-country count diagnostic (sum of calibrated p per S1)
    country = pop.get_column("country").to_numpy()
    sc_all = np.bincount(s1, weights=fit["cp"](p), minlength=fit["n"])
    by_country = {}
    for c in sorted(set(country)):
        m = country == c
        rows_c = {"n_s1": int(m.sum())}
        for label in ("base", "veto+frhn"):
            km = decisions[label]
            n_pred = np.bincount(s1[km], minlength=fit["n"])
            rows_c[f"{label}_matched_per_s1"] = round(float(n_pred[m].sum() / m.sum()), 4)
        rows_c["sum_c_per_s1"] = round(float(sc_all[m].sum() / m.sum()), 4)
        by_country[c] = rows_c

    rep = {"name": name, "calib_scores": src, "calib_fold": config.TUNE_FOLD, "tau": tau,
           "error_bar_note": "fold-0 worst |error| from `estimate validate` is the honest bar; "
                             "estimator cannot see planted look-alikes scored at high p",
           "by_country": by_country, "decisions": rows}
    save_json(rep, W / "reports" / f"estimator_test_{name}.json")
    print(f"== estimated test macro F0.5 ({name}, fit fold {config.TUNE_FOLD} of {src}, tau {tau})")
    print(f"{'decision':>12} {'pairs':>10} {'n1d':>9} {'d n1d':>8} {'n2d':>9} {'plugin':>9} {'d plugin':>9}")
    for label, r in rows.items():
        print(f"{label:>12} {r['matched_pairs']:>10,} {r['n1d']:>9} {r['d_n1d']:>8} {r['n2d']:>9} "
              f"{r['plugin']:>9} {r['d_plugin']:>9}")
    print("by country:", json.dumps(by_country))
    log(f"estimate test {name}: " + json.dumps(rows))
    return rep


def _est_on_mask(fit, p, s1, excl, keep):
    """Like est_apply but with an explicit keep mask (guard / country-tau variants)."""
    c = fit["cp"](p)
    n = fit["n"]
    n_pred = np.bincount(s1[keep], minlength=n)
    w_kept = np.bincount(s1[keep], weights=c[keep], minlength=n)
    w_not = np.bincount(s1[~keep], weights=c[~keep], minlength=n)
    n_hat = w_kept + fit["k"] * w_not
    b, t = _nb(n_pred), _tb(w_kept + w_not)
    tb = len(_TBINS) + 1
    return {"n1d": float(fit["tab1"][b].mean()), "n2d": float(fit["tab2"][b * tb + t].mean()),
            "plugin": float(_f05_plugin(w_kept, n_pred, n_hat).mean()),
            "n_pred": n_pred, "n_hat": n_hat}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["diagnose", "write", "pruned", "check", "estimate"])
    ap.add_argument("phase", nargs="?", choices=["validate", "test"], default=None,
                    help="estimate: validate (fold-3 fit -> fold-0 check) | test (apply to test scores)")
    ap.add_argument("--calib", default=None, help="estimate test: scores_<name>.parquet to fit from "
                    "(when the scored model has no saved tune fold)")
    ap.add_argument("--approach", type=int, default=2)
    ap.add_argument("--variant", choices=["threshold", "threshold+excl"], default="threshold+excl")
    ap.add_argument("--tau", type=float, default=None, help="global threshold (default: the one chosen on the tune fold)")
    ap.add_argument("--country-tau", action="append", default=[], metavar="COUNTRY=TAU")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None, help="diagnose/write/pruned/estimate: score name instead of A<n><tag>, e.g. "
                    "s2lab_all (reads reports/test_<name>_pairs.parquet and scores_<name>.parquet)")
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(a.approach, a.name)
        return
    if a.cmd == "check":
        assert a.out, "check needs --out <submission dir>"
        sys.exit(1 if subset_check(a.out) else 0)
    if a.cmd == "estimate":
        assert a.name, "estimate needs --name"
        if (a.phase or "validate") == "validate":
            estimate_validate(a.name)
        else:
            tau = a.tau if a.tau is not None else json.load(
                open(config.WORK_DIR / "reports" / f"{a.name}.json"))["decision"]["tau"]
            ct = {k: float(v) for k, v in (x.split("=", 1) for x in a.country_tau)}
            estimate_test(a.name, tau, a.calib, ct or None)
        return
    if a.cmd == "pruned":
        assert a.name and a.out, "pruned needs --name and --out"
        sys.exit(pruned(a.name, a.out))

    if not a.out:
        sys.exit("--out is required for write (never overwrite the main output/ by accident)")
    tau = a.tau
    if tau is None:
        if a.name:
            tau = json.load(open(config.WORK_DIR / "reports" / f"{a.name}.json"))["decision"]["tau"]
        else:
            rep = json.load(open(config.WORK_DIR / "reports" / f"eval_A{a.approach}{TAG}.json"))
            tau = rep["decision_selected_on_tune_fold"]["tau"]
    ct = {k: float(v) for k, v in (x.split("=", 1) for x in a.country_tau)}
    sys.exit(write(a.approach, a.variant, tau, ct, a.out, a.name))


if __name__ == "__main__":
    main()
