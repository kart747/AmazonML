"""E-G8: post-hoc precision guards for planted look-alikes that the shipped stage 2 keeps on test.

Two rules are applied to the saved test scores, so no model is retrained:

  veto     drop a kept pair whose ONLY name difference from its S1 is one extra target token t
           (the S1 has no extra token), where t is an affiliate token of the generator. Veto
           tokens are learned from the TUNE fold labels: n >= --min-n single-extra-token pairs
           and true rate <= --max-true (e.g. US +group 0/6k, +holdings 0/6k; India +ventures,
           +industries). Scored on the EVAL fold, which is never used for a choice.
           Why: on test the model keeps these at 20-40x the eval rate (US +group 2.36 vs 0.06
           per 1000 S1; India +group/+ventures/+industries/+enterprises/+public).
  frhn     NOT SUPPORTED (E-G9): the "p>=0.999 noise rate" below is selection-biased (US TRUE pairs
           show the same 3x enrichment, 8.9% vs 27.6%), and a near-pure sample of true France copies
           (kept +groupe/+developpement/+france pairs) trips frhn at 2.7% vs 1.2% for all kept France
           pairs, so its 9,140 pairs show no planted enrichment. Kept for the record only.
           France only, label-free: drop a kept pair whose S1 and target name the same street
           but carry different house numbers. France true copies almost never change the house
           number (0.39% of test pairs at p >= 0.999 vs 6.9% US / 2.6% India on eval), while
           9,140 kept France pairs have a same-street different number, i.e. ~6,000 more than
           France's own noise rate predicts. This is a France-specific fact, so the rule is not
           applied to US/India, where true copies change the number 5.8-9.2% of the time.

  lfveto   REJECTED (E-G9), kept opt-in for the record; never in a candidate.
           label-free version of veto, for a country without labels. Per country, on the
           scored pairs themselves: a token t is planted if targets add it alone (>= 2 per 1000
           S1), S1s almost never carry it alone (B/A <= 0.25), and the model already rejects most
           such pairs (keep rate <= 0.3, median p <= 0.5). Kept single-extra-t pairs are dropped.
           Noise tokens (the, dr, sri, ltd) fail the keep test; legal forms fail the one-side test.
           Validated in the unseen-country proxy (US-only model -> India): discovered on one fold,
           +0.0066 F0.5 on the other, 2 true of 2,465 removed. On France test it finds
           groupe, developpement, france, snc, holding, international, participations, distribution.
           Why rejected: a token can be both planted and appended by noise (US `partners`: 11.6%
           true, but the in-country model keeps 754/761 true ones), so low keep rate does not mean
           the kept pairs are planted. House-number agreement decides it: planted look-alikes shift
           the number (US eval hn_eq 0.014), true copies keep it (US kept 0.878). France kept
           +groupe/+developpement/+france pairs have hn_eq 0.95-0.96 = true suffix-noise copies.
           On US in-country folds lfveto removes ~400 pairs, 399 true (-0.0002).

  evaluate  learn veto tokens on fold 3, report fold-0 F0.5 before/after (label-backed).
  write     apply the rules to reports/test_<name>_pairs.parquet `keep` and write a complete
            submission (candidate_pairs = the scored pairs, as in redecode.py pruned), with the
            hard subset check and the official validator.

  BER_WORK_DIR=../../work_xd python src/experiments/lb_guards.py evaluate
  BER_WORK_DIR=../../work_xd python src/experiments/lb_guards.py write --rules veto,frhn \
      --out ../../output_variants/xd_s2lab_all_guard
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402
from ber.approaches import exclusive_mask, make_evaluator, run_validator, write_submission  # noqa: E402
from ber.utils import log, save_json  # noqa: E402
from experiments.redecode import subset_check  # noqa: E402

# house number directly before a street word, then the first street-name token (France-format addresses)
STREET = r"(?:rue|r|avenue|ave|av|bd|boulevard|blvd|allee|all|place|pl|impasse|imp|chemin|ch|cours|quai|passage|chaussee|route|rte)"
HN_RX = r"(?:^|\s)0*(\d{1,4})[a-z]?\s+(?:bis\s+|ter\s+)?" + STREET + r"\s+(\S+)"


def attach(pairs, split):
    """Add name token sets and parsed house number / street of both sides."""
    W = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(W / f"s1_{split}.parquet", columns=["rid", "country", "nm", "ad"]).with_columns(pl.col("rid").cast(pl.Int32))
    tg = pl.read_parquet(W / f"tg_{split}.parquet", columns=["gid", "nm", "ad"]).with_columns(pl.col("gid").cast(pl.Int32))
    d = (pairs.with_row_index("row").join(s1.rename({"nm": "qn", "ad": "qa"}), on="rid", how="left")
              .join(tg.rename({"nm": "tn", "ad": "ta"}), on="gid", how="left").sort("row"))
    qt, tt = pl.col("qn").str.split(" "), pl.col("tn").str.split(" ")
    return d.with_columns(tx=tt.list.set_difference(qt), qx=qt.list.set_difference(tt),
                          hq=pl.col("qa").str.extract(HN_RX, 1), sq=pl.col("qa").str.extract(HN_RX, 2),
                          ht=pl.col("ta").str.extract(HN_RX, 1), st=pl.col("ta").str.extract(HN_RX, 2))


def single_extra(d):
    return (d.get_column("tx").list.len() == 1) & (d.get_column("qx").list.len() == 0)


def learn_veto(tune, min_n, max_true):
    x = tune.filter(single_extra(tune)).with_columns(tok=pl.col("tx").list.first())
    g = x.group_by("tok").agg(n=pl.len(), true_rate=pl.col("y").mean()).filter((pl.col("n") >= min_n) & (pl.col("true_rate") <= max_true))
    return g.sort("n", descending=True)


def discover_tokens(d, min_per1k=2.0, max_oneside=0.25, max_keep=0.3, max_pmed=0.5):
    """Label-free planted-token discovery on one country's scored pairs (needs tx, qx, p, keep)."""
    ns = d.get_column("rid").n_unique()
    a = d.filter(single_extra(d)).with_columns(tok=pl.col("tx").list.first())
    b = (d.filter((pl.col("qx").list.len() == 1) & (pl.col("tx").list.len() == 0))
          .group_by(tok=pl.col("qx").list.first()).agg(B=pl.len()))
    g = (a.group_by("tok").agg(A=pl.len(), keep_rate=pl.col("keep").mean(), p_med=pl.col("p").median())
          .join(b, on="tok", how="left").with_columns(pl.col("B").fill_null(0)))
    return g.filter((pl.col("A") / ns * 1000 >= min_per1k) & (pl.col("B") / pl.col("A") <= max_oneside)
                    & (pl.col("keep_rate") <= max_keep) & (pl.col("p_med") <= max_pmed)).sort("A", descending=True)


def lfveto_mask(d):
    """Per-country label-free veto; returns (mask, {country: tokens})."""
    m, toks = np.zeros(d.height, bool), {}
    ctry = d.get_column("country").to_numpy()
    for c in sorted(set(ctry[ctry != None])):  # noqa: E711
        sub = d.filter(pl.col("country") == c)
        toks[c] = discover_tokens(sub).get_column("tok").to_list()
        m |= (ctry == c) & veto_mask(d, toks[c])
    return m, toks


def veto_mask(d, tokens):
    return (single_extra(d) & d.get_column("tx").list.first().is_in(tokens)).fill_null(False).to_numpy()


def frhn_mask(d):
    return (d.get_column("country") == "France").fill_null(False).to_numpy() & (
        (d.get_column("hq") != d.get_column("ht")) & (d.get_column("sq") == d.get_column("st"))).fill_null(False).to_numpy()


def evaluate(name, min_n, max_true):
    W = config.WORK_DIR
    tau = json.load(open(W / "reports" / f"{name}.json"))["decision"]["tau"]
    sc = pl.read_parquet(W / "reports" / f"scores_{name}.parquet")
    ex = exclusive_mask(sc.get_column("gid").to_numpy(), sc.get_column("p").to_numpy())
    d = attach(sc, "train")
    veto = learn_veto(d.filter(pl.col("fold") == config.TUNE_FOLD), min_n, max_true)
    toks = veto.get_column("tok").to_list()
    log(f"veto tokens (tune fold, n>={min_n}, true rate<={max_true}): {veto.to_dicts()}")
    out = {"tau": tau, "veto_tokens": veto.to_dicts()}
    for fold in (config.TUNE_FOLD, config.EVAL_FOLD):
        m = (d.get_column("fold") == fold).to_numpy()
        dm = d.filter(pl.Series(m))
        ev = make_evaluator(dm.select("rid"), dm.get_column("y").to_numpy(), fold)
        keep = (dm.get_column("p").to_numpy() >= tau) & ex[m]
        rows = {"base": keep, "veto": keep & ~veto_mask(dm, toks),
                "hn_guard_if_applied_to_US_India": keep & ~((dm.get_column("hq") != dm.get_column("ht")) & (dm.get_column("sq") == dm.get_column("st"))).fill_null(False).to_numpy()}
        res = {}
        for k, kp in rows.items():
            r = ev.report(kp)
            res[k] = {"macro_f05": r["macro_f05"], "fp": r["false_positive_pairs"], "pred_pairs": int(kp.sum()),
                      "removed": int(keep.sum() - kp.sum()), "removed_true": int((keep & ~kp & (dm.get_column("y").to_numpy() == 1)).sum())}
        out[f"fold{fold}"] = res
        log(f"fold {fold}: " + " | ".join(f"{k} F {v['macro_f05']} removed {v['removed']} (true {v['removed_true']})" for k, v in res.items()))
    save_json(out, W / "reports" / f"lb_guards_{name}.json")


def write(name, rules, out, min_n, max_true, veto_src):
    W = config.WORK_DIR
    pairs = pl.read_parquet(W / "reports" / f"test_{name}_pairs.parquet", columns=["rid", "gid", "p", "keep"])
    keep0 = pairs.get_column("keep").to_numpy()
    d = attach(pairs, "test")
    drop = np.zeros(len(keep0), bool)
    dec = {"source_scores": f"test_{name}_pairs.parquet", "base": "saved keep of those scores", "rules": rules, "veto_src": veto_src}
    if "veto" in rules:
        sc = pl.read_parquet(W / "reports" / f"scores_{veto_src}.parquet").filter(pl.col("fold") == config.TUNE_FOLD)
        toks = learn_veto(attach(sc, "train"), min_n, max_true).get_column("tok").to_list()
        v = keep0 & veto_mask(d, toks)
        dec["veto_tokens"], dec["veto_removed"] = toks, int(v.sum())
        drop |= v
    if "lfveto" in rules:
        lm, lt = lfveto_mask(d)
        lv = keep0 & lm
        dec["lfveto_tokens"], dec["lfveto_removed"] = lt, int(lv.sum())
        drop |= lv
    if "frhn" in rules:
        f = keep0 & frhn_mask(d)
        dec["frhn_removed"] = int(f.sum())
        drop |= f
    keep = keep0 & ~drop
    by_c = d.select("country").with_columns(dropped=pl.Series(drop & keep0)).group_by("country").agg(pl.col("dropped").sum())
    dec["removed_by_country"] = {r["country"]: int(r["dropped"]) for r in by_c.to_dicts()}
    dec["matched_pairs"] = int(keep.sum())
    config.OUT_DIR = Path(out).resolve()
    write_submission(pairs.select("rid", "gid"), keep)
    dec["matched_not_in_candidates"] = subset_check(config.OUT_DIR)
    assert dec["matched_not_in_candidates"] == 0
    save_json(dec, config.OUT_DIR / "decision.json")
    rc = run_validator()
    log(f"guards {json.dumps(dec)} -> {config.OUT_DIR} (validator exit {rc})")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["evaluate", "write"])
    ap.add_argument("--name", default="s2lab_all")
    ap.add_argument("--rules", default="veto,frhn", help="comma list of veto, lfveto, frhn")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--max-true", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    ap.add_argument("--veto-src", default="s2lab_all", help="write: tune-fold scores the veto tokens are learned from "
                    "(a property of the data generator, not of the model; dense_mix has no saved tune scores)")
    a = ap.parse_args()
    if a.cmd == "evaluate":
        evaluate(a.name, a.min_n, a.max_true)
        return
    assert a.out, "write needs --out"
    sys.exit(write(a.name, a.rules.split(","), a.out, a.min_n, a.max_true, a.veto_src))


if __name__ == "__main__":
    main()
