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

Usage:
  python src/experiments/redecode.py diagnose --approach 2
  BER_WORK_DIR=../work_xd python src/experiments/redecode.py pruned --name s2lab_all --out output_variants/xd_s2lab_all_pruned
  python src/experiments/redecode.py write --approach 2 --variant threshold+excl --tau 0.80 \
         --country-tau France=0.85 --out output_variants/a2_fr085
Cost: ~1–2 min, ~3 GB RAM for the full test set.

Read the diagnostics this way. On validation, sum(p)/S1 ≈ true matches/S1 (the model is
calibrated at train density). On test the generator is presumably the same, so a country
whose predicted matches/S1 differs strongly from the others is under- or over-matched. That
is a hypothesis to test with one leaderboard submission, never something to auto-tune.
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
from ber.utils import log, save_json  # noqa: E402

TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["diagnose", "write", "pruned", "check"])
    ap.add_argument("--approach", type=int, default=2)
    ap.add_argument("--variant", choices=["threshold", "threshold+excl"], default="threshold+excl")
    ap.add_argument("--tau", type=float, default=None, help="global threshold (default: the one chosen on the tune fold)")
    ap.add_argument("--country-tau", action="append", default=[], metavar="COUNTRY=TAU")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None, help="diagnose/write/pruned: score name instead of A<n><tag>, e.g. s2lab_all "
                    "(reads reports/test_<name>_pairs.parquet and scores_<name>.parquet)")
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(a.approach, a.name)
        return
    if a.cmd == "check":
        assert a.out, "check needs --out <submission dir>"
        sys.exit(1 if subset_check(a.out) else 0)
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
