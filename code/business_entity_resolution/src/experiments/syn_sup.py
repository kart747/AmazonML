"""E-G12: synthetic target-country supervision, validated in the US->India unseen-country proxy.

Hypothesis: the unseen-country loss is a missing-supervision problem (E-G5 oracle 0.983 vs 0.922;
E-G11: no rule, view or pretrained similarity labels the band). So learn the target country's copy
transformations from its UNLABELED pairs, and generate labeled synthetic pairs from its own S1 records:
positives = T(q), hard negatives = planted ops on T(q).

Everything is pairwise (no pool context), so every row -- US real, synthetic, India eval -- goes through
one self-contained featurizer. Protocol: mining uses India train folds 1,2 WITHOUT labels; synthetic S1
come from India folds 1,2; US folds 1,2 = real training rows; US fold 3 = early stopping and tau;
India folds 0 and 3 = evaluation only (fold 0 is the headline).

  BER_WORK_DIR=../../work_loco python src/experiments/syn_sup.py [--n-syn 150000]
"""
import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ber.approaches as ap  # noqa: E402
from ber import config  # noqa: E402
from ber.metric import Evaluator  # noqa: E402
from ber.utils import log, save_json  # noqa: E402

W = config.WORK_DIR
LEGAL = {"pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co",
         "company", "plc", "lp", "pc", "pllc", "pa", "the", "and", "of", "sarl", "sas", "sasu", "eurl", "sa",
         "sci", "snc", "ei", "cie", "ets"}
RNG = random.Random(7)


# ------------------------------------------------------------------ data
def pairs(country, folds):
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "country", "nm", "ad"])
    tg = pl.read_parquet(W / "prep" / "tg_train.parquet", columns=["gid", "nm", "ad"])
    df = (pl.scan_parquet(W / "s2lab" / "train" / "*.parquet").filter(pl.col("fold").is_in(folds))
            .select("rid", "gid", "fold", "y").collect())
    df = df.join(s1.filter(pl.col("country") == country).select("rid", q_nm="nm", q_ad="ad"), on="rid", how="inner")
    return df.join(tg.select("gid", t_nm="nm", t_ad="ad"), on="gid", how="left").with_columns(
        pl.col("t_nm").fill_null(""), pl.col("t_ad").fill_null(""), pl.col("q_ad").fill_null(""))


# ------------------------------------------------------------------ featurizer
def idf_table(country):
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["country", "nm"]).filter(pl.col("country") == country)
    n = s1.height
    df = s1.select(tok=pl.col("nm").str.split(" ").list.unique()).explode("tok").group_by("tok").len()
    return {t: float(np.log(n / c)) for t, c in zip(df["tok"].to_list(), df["len"].to_list())}, float(np.log(n))


def featurize(df, idf, maxidf):
    qn, tn, qa, ta = (df[c].to_list() for c in ("q_nm", "t_nm", "q_ad", "t_ad"))
    cp = lambda a, b, s: process.cpdist(a, b, scorer=s, workers=-1, dtype=np.float32)
    sk = lambda s: " ".join(re.sub(r"(.)\1+", r"\1", re.sub(r"\B[aeiouy]", "", w)) for w in s.split())
    out = {"n_ratio": cp(qn, tn, fuzz.ratio), "n_tsort": cp(qn, tn, fuzz.token_sort_ratio),
           "n_tset": cp(qn, tn, fuzz.token_set_ratio), "n_partial": cp(qn, tn, fuzz.partial_ratio),
           "n_jw": cp(qn, tn, JaroWinkler.normalized_similarity),
           "n_skel": cp([sk(x) for x in qn], [sk(x) for x in tn], fuzz.token_set_ratio),
           "a_ratio": cp(qa, ta, fuzz.ratio), "a_tset": cp(qa, ta, fuzz.token_set_ratio),
           "a_partial": cp(qa, ta, fuzz.partial_ratio)}
    qs, ts = [set(x.split()) for x in qn], [set(x.split()) for x in tn]
    qas, tas = [set(x.split()) for x in qa], [set(x.split()) for x in ta]
    f = {k: [] for k in ("n_jacc", "core_jacc", "q_match_idf", "t_match_idf", "q_unm_maxidf", "t_unm_maxidf",
                          "a_jacc", "a_exact", "num_jacc", "num_first_eq", "num_logdiff", "q_num_n", "t_num_n",
                          "hn_prefix", "hn_samelen", "hn_lev", "q_ntok", "t_ntok", "t_addr_empty", "q_alen", "t_alen")}
    g = lambda t: idf.get(t, maxidf)
    for i in range(len(qn)):
        q, t = qs[i], ts[i]
        f["n_jacc"].append(len(q & t) / max(len(q | t), 1))
        qc, tc = q - LEGAL, t - LEGAL
        f["core_jacc"].append(len(qc & tc) / max(len(qc | tc), 1))
        iq, it = sum(g(x) for x in q), sum(g(x) for x in t)
        f["q_match_idf"].append(sum(g(x) for x in q & t) / max(iq, 1e-6))
        f["t_match_idf"].append(sum(g(x) for x in q & t) / max(it, 1e-6))
        f["q_unm_maxidf"].append(max([g(x) for x in q - t], default=0.0))
        f["t_unm_maxidf"].append(max([g(x) for x in t - q], default=0.0))
        a, b = qas[i], tas[i]
        f["a_jacc"].append(len(a & b) / max(len(a | b), 1))
        f["a_exact"].append(float(bool(a) and a == b))
        nq, nt = re.findall(r"\d+", qa[i]), re.findall(r"\d+", ta[i])
        f["num_jacc"].append(len(set(nq) & set(nt)) / max(len(set(nq) | set(nt)), 1))
        h1, h2 = (nq[0] if nq else None), (nt[0] if nt else None)
        both = h1 is not None and h2 is not None
        f["num_first_eq"].append(float(both and h1 == h2) if both else np.nan)
        f["num_logdiff"].append(float(np.log1p(abs(int(h1[:9]) - int(h2[:9])))) if both else np.nan)
        f["q_num_n"].append(len(nq)); f["t_num_n"].append(len(nt))
        f["hn_prefix"].append(float(both and h1 != h2 and (h1.startswith(h2) or h2.startswith(h1) or h1.endswith(h2) or h2.endswith(h1))) if both else np.nan)
        f["hn_samelen"].append(float(len(h1) == len(h2)) if both else np.nan)
        f["hn_lev"].append(float(Levenshtein.distance(h1, h2)) if both else np.nan)
        f["q_ntok"].append(len(q)); f["t_ntok"].append(len(t))
        f["t_addr_empty"].append(float(ta[i] == "")); f["q_alen"].append(len(qa[i])); f["t_alen"].append(len(ta[i]))
    out.update({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    return np.column_stack([out[k] for k in FEATS]).astype(np.float32)


FEATS = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_skel", "a_ratio", "a_tset", "a_partial",
         "n_jacc", "core_jacc", "q_match_idf", "t_match_idf", "q_unm_maxidf", "t_unm_maxidf", "a_jacc", "a_exact",
         "num_jacc", "num_first_eq", "num_logdiff", "q_num_n", "t_num_n", "hn_prefix", "hn_samelen", "hn_lev",
         "q_ntok", "t_ntok", "t_addr_empty", "q_alen", "t_alen"]


# ------------------------------------------------------------------ transformation mining (unlabeled)
def mine(unl, planted):
    """Anchor pairs = exact same non-empty address + >= 1 shared core name token (no labels used)."""
    d = unl.filter((pl.col("q_ad") != "") & (pl.col("q_ad") == pl.col("t_ad")))
    swaps, drops, adds, ops = Counter(), Counter(), Counter(), Counter()
    n_anchor = 0
    for q, t in zip(d["q_nm"].to_list(), d["t_nm"].to_list()):
        qs, ts = q.split(), t.split()
        if not (set(qs) - LEGAL) & (set(ts) - LEGAL):
            continue
        n_anchor += 1
        qx, tx = [w for w in qs if w not in ts], [w for w in ts if w not in qs]
        if len(qx) == 1 and len(tx) == 1:
            swaps[(qx[0], tx[0])] += 1
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=qx[0], b=tx[0], autojunk=False).get_opcodes():
                if tag != "equal" and (i2 - i1) <= 3 and (j2 - j1) <= 3:
                    ops[(qx[0][i1:i2], tx[0][j1:j2])] += 1
        for w in qx:
            if not tx:
                drops[w] += 1
        for w in tx:
            if not qx and w not in planted:
                adds[w] += 1
    # address reduction on name anchors (identical names) -- token keep rate, empty rate
    e = unl.filter(pl.col("q_nm") == pl.col("t_nm"))
    keep, empty = [], 0
    for qa, ta in zip(e["q_ad"].to_list(), e["t_ad"].to_list()):
        if ta == "":
            empty += 1
            continue
        a, b = qa.split(), set(ta.split())
        if a:
            keep.append(sum(w in b for w in a) / len(a))
    stats = {"n_anchor": n_anchor, "n_name_anchor": e.height, "addr_empty_rate": empty / max(e.height, 1),
             "addr_keep_mean": float(np.mean(keep)) if keep else 1.0,
             "swap_rate": sum(swaps.values()) / max(n_anchor, 1), "drop_rate": sum(drops.values()) / max(n_anchor, 1),
             "add_rate": sum(adds.values()) / max(n_anchor, 1)}
    return {"swaps": {k: v for k, v in swaps.items() if v >= 3}, "ops": [(k, v) for k, v in ops.most_common(300)],
            "drops": dict(drops.most_common(200)), "adds": dict(adds.most_common(200)), "keep": keep, "stats": stats}


def make_T(m):
    swap_by = {}
    for (a, b), c in m["swaps"].items():
        swap_by.setdefault(a, []).append((b, c))
    ops, opw = [k for k, _ in m["ops"]], [v for _, v in m["ops"]]
    drops, adds = m["drops"], list(m["adds"].items())
    st = m["stats"]
    def char_noise(w):
        for _ in range(3):
            a, b = RNG.choices(ops, opw)[0]
            if a and a in w:
                return w.replace(a, b, 1)
            if not a and len(w) > 2:
                i = RNG.randrange(1, len(w)); return w[:i] + b + w[i:]
        return w
    def T(nm, ad):
        toks = nm.split()
        out = []
        for w in toks:
            r = RNG.random()
            if w in swap_by and r < 0.5:
                bs, cs = zip(*swap_by[w]); out.append(RNG.choices(bs, cs)[0])
            elif w in drops and r < 0.15 * min(1.0, drops[w] / 50) and len(toks) > 1:
                continue
            elif r < 0.25:
                out.append(char_noise(w))
            else:
                out.append(w)
        if adds and RNG.random() < min(0.3, st["add_rate"]):
            ws, cs = zip(*adds); out.append(RNG.choices(ws, cs)[0])
        if RNG.random() < 0.05 and len(out) > 1:
            RNG.shuffle(out)
        if RNG.random() < st["addr_empty_rate"]:
            ad2 = ""
        else:
            k = RNG.choice(m["keep"]) if m["keep"] else 1.0
            at = ad.split(); ad2 = " ".join(w for w in at if RNG.random() < k) or ad
        return " ".join(out) or nm, ad2
    return T


def planted_negs(q_nm, q_ad, t_nm, t_ad, planted, other_names):
    """Hard negatives on a synthetic copy: affiliate token, house-number shift, other entity at same address."""
    k = RNG.randrange(3)
    if k == 0 and planted:
        return t_nm + " " + RNG.choice(planted), t_ad, "neg_affiliate"
    if k == 1:
        m = re.search(r"\d+", t_ad or q_ad)
        if m:
            h = m.group(0); d = RNG.randrange(1, 10)
            h2 = str(max(1, int(h) + (d if RNG.random() < 0.5 else -d)))
            src = t_ad or q_ad
            return t_nm, src[:m.start()] + h2 + src[m.end():], "neg_hnshift"
    return RNG.choice(other_names), t_ad or q_ad, "neg_otherentity"


# ------------------------------------------------------------------ evaluation
def evaluator(meta, fold, country):
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country"])
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet").group_by("rid").len("n_true")
    pop = (s1.filter((pl.col("fold") == fold) & (pl.col("country") == country)).select("rid")
             .join(gt, on="rid", how="left").with_columns(pl.col("n_true").fill_null(0)).with_row_index("idx"))
    idx = meta.select("rid").join(pop.select("rid", "idx"), on="rid", how="left", maintain_order="left")
    return Evaluator(idx["idx"].to_numpy(), meta["y"].to_numpy(), pop["n_true"].to_numpy())


def fit(X, y, Xv, yv):
    dtr = lgb.Dataset(X, y, feature_name=FEATS, free_raw_data=True)
    return lgb.train(config.LGB_PARAMS, dtr, config.LGB_ROUNDS * 2, valid_sets=[lgb.Dataset(Xv, yv, reference=dtr)],
                     callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False)])


def score(m, tu, Xtu, evs):
    p = ap._pred(m, Xtu)
    ch = ap.choose_decision(evaluator(tu, config.TUNE_FOLD, "US"), p, tu["gid"].to_numpy())
    v = max(ch, key=lambda k: ch[k]["tune_f05"]); tau = ch[v]["tau"]
    res = {"tau_from_US_tune": tau, "US_tune_f05": ch[v]["tune_f05"]}
    for fold, (ev, Xev) in evs.items():
        pe = ap._pred(m, Xev); g = ev["gid"].to_numpy(); e = evaluator(ev, fold, "India")
        ex = ap.exclusive_mask(g, pe)
        r = e.report((pe >= tau) & ex)
        curve = {t: e.report((pe >= t) & ex)["macro_f05"] for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
        res[f"India_f{fold}"] = {"f05": r["macro_f05"], "P": r["macro_precision"], "R": r["macro_recall"],
                                 "fp_pairs": r["false_positive_pairs"], "best_tau_f05": max(curve.values()), "curve": curve}
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n-syn", type=int, default=150_000)
    x = a.parse_args()
    idf_us, mx_us = idf_table("US"); idf_in, mx_in = idf_table("India")
    log("loading pairs")
    us_tr, us_tu = pairs("US", [1, 2]), pairs("US", [3])
    in_unl = pairs("India", [1, 2])                      # labels never read
    evs_df = {0: pairs("India", [0]), 3: pairs("India", [3])}
    # planted-token vocabulary: label-free discovery (E-G9) result for the India proxy
    planted = ["enterprises", "industries", "infratech", "overseas", "exports", "ventures", "group", "holdings", "trading"]
    m = mine(in_unl.drop("y"), set(planted))
    log(f"mined: {m['stats']} swaps {len(m['swaps'])} ops {len(m['ops'])} drops {len(m['drops'])} adds {len(m['adds'])}")
    T = make_T(m)
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "fold", "country", "nm", "ad"]).filter(
        (pl.col("country") == "India") & pl.col("fold").is_in([1, 2])).sample(n=x.n_syn, seed=3)
    names = s1["nm"].to_list()
    rows = []
    for q_nm, q_ad in zip(s1["nm"].to_list(), s1["ad"].fill_null("").to_list()):
        t_nm, t_ad = T(q_nm, q_ad)
        rows.append((q_nm, q_ad, t_nm, t_ad, 1, "pos"))
        n_nm, n_ad, kind = planted_negs(q_nm, q_ad, t_nm, t_ad, planted, names)
        rows.append((q_nm, q_ad, n_nm, n_ad, 0, kind))
    syn = pl.DataFrame(rows, schema=["q_nm", "q_ad", "t_nm", "t_ad", "y", "kind"], orient="row")
    log(f"synthetic rows {syn.height:,}: {syn.group_by('kind').len().to_dicts()}")
    log("featurizing")
    X_us, X_tu = featurize(us_tr, idf_us, mx_us), featurize(us_tu, idf_us, mx_us)
    X_syn = featurize(syn, idf_in, mx_in)
    evs = {f: (d, featurize(d, idf_in, mx_in)) for f, d in evs_df.items()}
    y_us, y_tu, y_syn = us_tr["y"].to_numpy(), us_tu["y"].to_numpy(), syn["y"].to_numpy()
    pos = (syn["kind"] == "pos").to_numpy()
    out = {"mining": m["stats"], "n_syn_S1": x.n_syn, "planted_vocab": planted}
    variants = {"A_US_only": (X_us, y_us),
                "B_US+syn_pos": (np.vstack([X_us, X_syn[pos]]), np.r_[y_us, y_syn[pos]]),
                "C_US+syn_pos+neg": (np.vstack([X_us, X_syn]), np.r_[y_us, y_syn])}
    for name, (X, y) in variants.items():
        mdl = fit(X, y, X_tu, y_tu)
        out[name] = score(mdl, us_tu, X_tu, evs)
        log(f"{name}: " + json.dumps({k: (v['f05'] if isinstance(v, dict) and 'f05' in v else v) for k, v in out[name].items()}))
    # ceiling: same featurizer, true India labels (folds 1,2), early stop on India fold 3 is NOT used: US tune kept
    in_tr = pairs("India", [1, 2])
    mdl = fit(np.vstack([X_us, featurize(in_tr, idf_in, mx_in)]), np.r_[y_us, in_tr["y"].to_numpy()], X_tu, y_tu)
    out["D_oracle_US+India_labels"] = score(mdl, us_tu, X_tu, evs)
    log("D oracle: " + json.dumps({k: (v['f05'] if isinstance(v, dict) and 'f05' in v else v) for k, v in out["D_oracle_US+India_labels"].items()}))
    save_json(out, W / "reports" / "syn_sup.json")


if __name__ == "__main__":
    main()
