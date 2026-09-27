"""E-G9: label-backed audit of the lb_guards rules in every labeled world we have.

For each world, model and held-out fold, apply the shipped decision (thr+excl, tau 0.68) and
then each guard, and report per country: F0.5 before/after, pairs removed, of which true, and
the S1s touched. Removed pairs are saved for inspection.

  worlds   normal  work_xd s2lab features, folds 3 and 0 (s2lab_all, s2lab_dense_mix)
           world   work_xd s2lab_world (test-world density sim: US 50% S1 deleted with their
                   copies, India 19% orphanized), folds 3 and 0, same two models
           loco    work_loco s2lab (US-only stage 1), India folds 3 and 0, scored by the
                   US-only stage 2 (unseen) and the in-country model (oracle); the France proxy
  rules    veto      one extra affiliate token in the target name (lb_guards.learn_veto).
                     Tokens are learned on the OTHER held-out fold of the normal world (3 <-> 0),
                     so no fold scores its own tokens; in loco they are learned from US rows only,
                     which is what France gets (tokens learned without seeing the country).
           veto_c    as veto, but a token is vetoed only in the country it was learned in
           lfveto    label-free token discovery (lb_guards.discover_tokens) on the fold/country's own
                     scored pairs (no labels read), then the same drop as veto
           veto+lf   union of veto and lfveto (the sub08 rule set)
           hn_all    lb_guards' France house-number rule applied to US/India: a negative control
                     (US/India true copies do change the number), NOT a candidate
  frhn is France-only and France has no labels, so it has no row here; see E-G9 in EXPERIMENTS.md
  for its label-free evidence.

  python src/experiments/guard_audit.py            (-> work_xd/reports/guard_audit.json)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ber.approaches as ap  # noqa: E402
from ber import config  # noqa: E402
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json  # noqa: E402
from lb_guards import attach, discover_tokens, learn_veto, single_extra  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
XD, LOCO = ROOT / "work_xd", ROOT / "work_loco"
TAU = 0.68
FOLDS = (config.TUNE_FOLD, config.EVAL_FOLD)


def load(wd, sub, feats, country=None):
    lf = pl.scan_parquet(wd / sub / "train" / "*.parquet").filter(pl.col("fold").is_in(FOLDS))
    df = lf.select(["rid", "gid", "fold", "y"] + feats).collect()
    if country:
        s1 = pl.read_parquet(wd / "prep" / "s1_train.parquet", columns=["rid", "country"])
        df = df.join(s1.filter(pl.col("country") == country).select("rid"), on="rid", how="semi")
    return df.select("rid", "gid", "fold", "y"), np.ascontiguousarray(df.select(feats).to_numpy(), dtype=np.float32)


def evaluator(meta, y, fold, country, dropped):
    """Population = every S1 of (fold, country) that exists in the world, incl. S1s without candidates."""
    s1 = pl.read_parquet(XD / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(XD / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter((pl.col("fold") == fold) & (pl.col("country") == country)).join(dropped, on="rid", how="anti")
             .select("rid").join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx.get_column("idx").to_numpy(), y, pop.get_column("n_true").to_numpy())


def world_dropped():
    import world_sim
    old = config.WORK_DIR
    config.WORK_DIR = XD
    _, dropped, _ = world_sim.world_drop()
    config.WORK_DIR = old
    return dropped


def veto_tokens(learn_fold, countries):
    """Affiliate tokens from the normal world's s2lab_all scores on learn_fold, per country and pooled."""
    sc = pl.read_parquet(XD / "reports" / "scores_s2lab_all.parquet").filter(pl.col("fold") == learn_fold)
    config.WORK_DIR = XD
    d = attach(sc, "train").filter(pl.col("country").is_in(countries))
    pooled = learn_veto(d, 50, 0.02).get_column("tok").to_list()
    per = {c: learn_veto(d.filter(pl.col("country") == c), 50, 0.02).get_column("tok").to_list() for c in countries}
    return pooled, per


def audit(world, wd, sub, models, countries, dropped, tok_countries, only_country=None):
    out, removed_all = {}, []
    for label, path in models.items():
        m = lgb.Booster(model_file=str(path))
        meta, X = load(wd, sub, m.feature_name(), only_country)
        p = ap._pred(m, X)
        del X
        ex = ap.exclusive_mask(meta.get_column("gid").to_numpy(), p)
        config.WORK_DIR = XD
        keep_all = (p >= TAU) & ex
        d = attach(meta.with_columns(p=pl.Series(p), keep=pl.Series(keep_all)), "train")
        y_all = d.get_column("y").to_numpy()
        hn = ((d.get_column("hq") != d.get_column("ht")) & (d.get_column("sq") == d.get_column("st"))).fill_null(False).to_numpy()
        se = single_extra(d).fill_null(False).to_numpy()
        tok = d.get_column("tx").list.first().to_numpy()
        ctry = d.get_column("country").to_numpy()
        res = {}
        for fold in FOLDS:
            other = FOLDS[0] if fold == FOLDS[1] else FOLDS[1]
            pooled, per = veto_tokens(other, tok_countries)
            veto = se & np.isin(tok, pooled)
            veto_c = se & np.zeros(len(se), bool)
            for c, ts in per.items():
                veto_c |= se & (ctry == c) & np.isin(tok, ts)
            for c in countries:
                msk = (d.get_column("fold").to_numpy() == fold) & (ctry == c)
                dm = d.filter(pl.Series(msk))
                lf_t = discover_tokens(dm).get_column("tok").to_list()
                lf = np.zeros(len(se), bool)
                lf[msk] = (single_extra(dm).fill_null(False).to_numpy() & np.isin(tok[msk], lf_t))
                rules = {"veto": veto, "veto_c": veto_c, "lfveto": lf, "veto+lf": veto | lf, "hn_all": hn}
                ev = evaluator(dm.select("rid"), y_all[msk], fold, c, dropped)
                k0 = keep_all[msk]
                r0 = ev.report(k0)
                row = {"base": {"f05": r0["macro_f05"], "fp": r0["false_positive_pairs"], "pred_pairs": int(k0.sum()),
                                "n_s1": r0.get("n_s1")}}
                for rn, rm in rules.items():
                    drop = k0 & rm[msk]
                    r = ev.report(k0 & ~drop)
                    rt = int((drop & (y_all[msk] == 1)).sum())
                    row[rn] = {"f05": r["macro_f05"], "delta": round(r["macro_f05"] - r0["macro_f05"], 6),
                               "removed": int(drop.sum()), "removed_true": rt, "removed_fp": int(drop.sum()) - rt,
                               "fp_share_of_base_fp": round((int(drop.sum()) - rt) / max(r0["false_positive_pairs"], 1), 4)}
                    if rn != "hn_all" and drop.any():
                        removed_all.append(dm.filter(pl.Series(drop)).select("rid", "gid", "fold", "country", "y", "p", "qn", "tn", "tx")
                                             .with_columns(world=pl.lit(world), model=pl.lit(label), rule=pl.lit(rn)))
                row["lfveto_tokens"] = lf_t
                res[f"fold{fold}_{c}"] = row
                log(f"{world}/{label} fold {fold} {c}: base {r0['macro_f05']} | " + " | ".join(
                    f"{rn} {row[rn]['delta']:+.5f} (-{row[rn]['removed']}, true {row[rn]['removed_true']})" for rn in rules))
            res[f"veto_tokens_learned_on_fold{other}"] = {"pooled": pooled, "per_country": per}
        out[label] = res
    return out, removed_all


def main():
    empty = pl.DataFrame({"rid": pl.Series([], dtype=pl.Int32)})
    xd_models = {"s2lab_all": XD / "models" / "s2lab_all.txt", "s2lab_dense_mix": XD / "models" / "s2lab_dense_mix.txt"}
    out, rem = {}, []
    o, r = audit("normal", XD, "s2lab", xd_models, ["US", "India"], empty, ["US", "India"])
    out["normal"], rem = o, rem + r
    wd = world_dropped().with_columns(pl.col("rid").cast(pl.Int32))
    o, r = audit("world", XD, "s2lab_world", xd_models, ["US", "India"], wd, ["US", "India"])
    out["world"], rem = o, rem + r
    loco_models = {"unseen_US_only": LOCO / "models" / "loco_s2_US.txt", "in_country": LOCO / "models" / "s2lab_all.txt"}
    o, r = audit("loco", LOCO, "s2lab", loco_models, ["India"], empty, ["US"], only_country="India")
    out["loco_US_to_India"], rem = o, rem + r
    save_json(out, XD / "reports" / "guard_audit.json")
    if rem:
        pl.concat(rem, how="vertical_relaxed").write_parquet(XD / "reports" / "guard_audit_removed.parquet")


if __name__ == "__main__":
    main()
