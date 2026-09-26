"""Candidate generation shared by both approaches (RESEARCH_DESIGN.md §3.C–D).

Per country partition (labels discovered from the data, so unseen countries such
as France are handled identically; 0 of 7.64M true train pairs cross countries):

  name   char_wb 3-gram TF-IDF on the normalized name    S1 -> target top-k_name
  addr   word TF-IDF on the normalized address           S1 -> target top-k_addr
  combo  [sqrt(.5)*name | sqrt(.5)*addr], cosine = mean  S1 -> target top-k_combo
  rev    combo, target -> S1 top-k_rev over ALL S1 of the partition

The reverse view exploits the many-to-one structure (every target belongs to at
most one S1): the true S1 is usually a target's top-1 even when a generic name
floods the S1's own list.

Outputs in work/block/<split>/:
  cand_<country>.parquet  rid, gid, retrieval ranks per view, exact cos_name/cos_addr
  rev_<country>.parquet   gid, rid, rrank, rscore  (each target's full reverse list;
                          used for label-free competition features)
"""
import math
from multiprocessing import Pool

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config
from .utils import done, log, mark_done, save_json, stage

_VEC = None  # vectorizer held by each worker process (set by _init_worker)


def _init_worker(vec):
    global _VEC
    _VEC = vec


def _transform_chunk(texts):
    return _VEC.transform(texts)


def fit_transform(vec: TfidfVectorizer, fit_texts, *text_lists):
    """Fit on a sample, then transform each list in parallel.

    The fitted vectorizer is handed to workers through the pool initializer, so this
    works with both 'fork' (Linux) and 'spawn' (Windows, macOS) start methods.
    """
    vec.fit(fit_texts)
    out = []
    with Pool(config.N_THREADS, initializer=_init_worker, initargs=(vec,)) as p:
        for texts in text_lists:
            step = max(1000, math.ceil(len(texts) / (config.N_THREADS * 4)))
            parts = p.map(_transform_chunk, [texts[i:i + step] for i in range(0, len(texts), step)])
            out.append(sp.vstack(parts).tocsr().astype(np.float32))
    return out


def topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int):
    """Exact row-wise top-k cosine neighbours of A's rows among B's rows (L2-normalized).

    B is split into column blocks of `item_block` rows: the sparse top-k kernel keeps a
    dense accumulator the size of the item set, so a small block stays in CPU cache.
    Measured on India (4.1M targets): 17x faster than one call, identical results.
    Returns (query_row, item_row, score, rank) arrays; rank is 0-based within the query row.
    """
    bs = config.BLOCK["item_block"]
    blocks = [(s, B[s:s + bs].T.tocsr()) for s in range(0, B.shape[0], bs)]
    chunk = int(max(2000, min(100_000, 2e7 // (k * len(blocks)))))
    qs, its, scs, rks = [], [], [], []
    for q0 in range(0, A.shape[0], chunk):
        Ac = A[q0:q0 + chunk]
        r_, c_, d_ = [], [], []
        for s, BT in blocks:
            C = sp_matmul_topn(Ac, BT, top_n=k, sort=False, n_threads=config.N_THREADS).tocoo()
            r_.append(C.row.astype(np.int64))
            c_.append(C.col.astype(np.int64) + s)
            d_.append(C.data.astype(np.float32))
        r, c, d = np.concatenate(r_), np.concatenate(c_), np.concatenate(d_)
        o = np.lexsort((-d, r))
        r, c, d = r[o], c[o], d[o]
        starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
        rank = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))
        m = rank < k
        qs.append(r[m] + q0)
        its.append(c[m])
        scs.append(d[m])
        rks.append(rank[m].astype(np.int16))
    return np.concatenate(qs), np.concatenate(its), np.concatenate(scs), np.concatenate(rks)


def rowdot(X: sp.csr_matrix, Y: sp.csr_matrix, i: np.ndarray, j: np.ndarray, chunk=2_000_000):
    """Exact dot product X[i[n]] . Y[j[n]] for each pair n (cosine for L2-normalized rows)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(X[i[s:s + chunk]].multiply(Y[j[s:s + chunk]]).sum(1)).ravel()
    return out


def _sample(texts, n, seed=config.SEED):
    if len(texts) <= n:
        return texts
    idx = np.random.default_rng(seed).choice(len(texts), n, replace=False)
    return [texts[i] for i in idx]


def block_country(split, country, s1: pl.DataFrame, tg: pl.DataFrame, bdir):
    fc, fr = bdir / f"cand_{country}.parquet", bdir / f"rev_{country}.parquet"
    if done(fc) and done(fr):
        log(f"block {split}/{country}: cached")
        return
    B = config.BLOCK
    S_rid = s1.get_column("rid").to_numpy()
    T_gid = tg.get_column("gid").to_numpy()
    qmask = s1.get_column("fold").is_in(config.QUERY_FOLDS).to_numpy() if split == "train" \
        else np.ones(s1.height, bool)
    q_loc = np.flatnonzero(qmask)
    s_nm, t_nm = s1.get_column("nm").to_list(), tg.get_column("nm").to_list()
    s_ad, t_ad = s1.get_column("ad").to_list(), tg.get_column("ad").to_list()
    log(f"block {split}/{country}: S1 {len(S_rid):,} (queries {len(q_loc):,}), targets {len(T_gid):,}")

    with stage(f"block_{split}_{country}"):
        vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=B["name_max_df"],
                             sublinear_tf=True, lowercase=False, dtype=np.float32)
        NS, NT = fit_transform(vn, _sample(s_nm + t_nm, B["fit_sample"]), s_nm, t_nm)
        va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=B["addr_max_df"],
                             sublinear_tf=True, lowercase=False, dtype=np.float32)
        AS, AT = fit_transform(va, _sample(s_ad + t_ad, B["fit_sample"]), s_ad, t_ad)
        log(f"tfidf: name vocab {len(vn.vocabulary_):,}, addr vocab {len(va.vocabulary_):,}")
        h = np.float32(math.sqrt(0.5))
        CS = sp.hstack([NS * h, AS * h]).tocsr()
        CT = sp.hstack([NT * h, AT * h]).tocsr()

        views = {}
        for v, (A, Bm, k) in {"name": (NS[q_loc], NT, B["k_name"]), "addr": (AS[q_loc], AT, B["k_addr"]),
                              "combo": (CS[q_loc], CT, B["k_combo"])}.items():
            q, t, _, r = topk(A, Bm, k)
            views[v] = pl.DataFrame({"sl": q_loc[q].astype(np.int32), "tl": t.astype(np.int32),
                                     f"r_{v}": r})
            log(f"view {v}: {len(q):,} pairs")
        tq, sq, sc, rk = topk(CT, CS, B["k_rev"])
        rev = pl.DataFrame({"tl": tq.astype(np.int32), "sl": sq.astype(np.int32),
                            "rrank": rk, "rscore": sc})
        log(f"view rev: {rev.height:,} pairs")

        # union of all views, restricted to query S1 rows
        rev_q = rev.filter(pl.Series(qmask[rev.get_column("sl").to_numpy()])).select("sl", "tl", r_rev="rrank")
        cand = views["name"].join(views["addr"], on=["sl", "tl"], how="full", coalesce=True)
        cand = cand.join(views["combo"], on=["sl", "tl"], how="full", coalesce=True)
        cand = cand.join(rev_q, on=["sl", "tl"], how="full", coalesce=True)
        sl, tl = cand.get_column("sl").to_numpy(), cand.get_column("tl").to_numpy()
        cand = cand.with_columns(
            cos_name=rowdot(NS, NT, sl, tl), cos_addr=rowdot(AS, AT, sl, tl),
            rid=pl.Series(S_rid[sl]).cast(pl.Int32), gid=pl.Series(T_gid[tl]).cast(pl.Int32),
        ).drop("sl", "tl")
        cand.write_parquet(fc)
        rev.with_columns(rid=pl.Series(S_rid[rev.get_column("sl").to_numpy()]).cast(pl.Int32),
                         gid=pl.Series(T_gid[rev.get_column("tl").to_numpy()]).cast(pl.Int32)
                         ).drop("sl", "tl").write_parquet(fr)
        mark_done(fc)
        mark_done(fr)
        log(f"candidates {split}/{country}: {cand.height:,} ({cand.height / max(1, len(q_loc)):.1f} per query S1)")


def block(split: str):
    bdir = config.WORK_DIR / "block" / split
    bdir.mkdir(parents=True, exist_ok=True)
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / f"s1_{split}.parquet", columns=["rid", "country", "fold", "nm", "ad"])
    tg = pl.read_parquet(pdir / f"tg_{split}.parquet", columns=["gid", "country", "nm", "ad"])
    countries = sorted(set(s1.get_column("country").unique()) | set(tg.get_column("country").unique()))
    # smallest partitions first so problems surface early
    countries.sort(key=lambda c: s1.filter(pl.col("country") == c).height)
    for c in countries:
        s1c, tgc = s1.filter(pl.col("country") == c), tg.filter(pl.col("country") == c)
        if s1c.height == 0 or tgc.height == 0:
            continue
        block_country(split, c, s1c, tgc, bdir)
    if split == "train":
        blocking_report()


def blocking_report():
    """Candidate recall / volume on the query folds (train only; labels used for measurement only)."""
    bdir = config.WORK_DIR / "block" / "train"
    pdir = config.WORK_DIR / "prep"
    s1 = pl.read_parquet(pdir / "s1_train.parquet", columns=["rid", "fold", "country"]) \
        .filter(pl.col("fold").is_in(config.QUERY_FOLDS))
    gt = pl.read_parquet(pdir / "gt_train.parquet").join(s1.select("rid"), on="rid")
    cand = pl.concat([pl.read_parquet(f) for f in sorted(bdir.glob("cand_*.parquet"))], how="diagonal")
    per_s1 = cand.group_by("rid").len()
    hit = gt.join(cand, on=["rid", "gid"], how="left").with_columns(
        found=pl.col("cos_name").is_not_null(),
        **{f"in_{v}": pl.col(f"r_{v}").is_not_null() for v in ("name", "addr", "combo", "rev")})
    ent = hit.group_by("rid").agg(pl.col("found").all())
    counts = s1.select("rid").join(per_s1, on="rid", how="left").get_column("len").fill_null(0)
    rep = {
        "query_s1": s1.height, "true_pairs": gt.height, "candidate_pairs": cand.height,
        "pair_recall": round(hit.get_column("found").mean(), 5),
        "entity_full_recall": round(ent.get_column("found").mean(), 5),
        "view_recall": {v: round(hit.get_column(f"in_{v}").mean(), 5) for v in ("name", "addr", "combo", "rev")},
        "cands_per_s1": {"mean": round(counts.mean(), 2), "p50": float(counts.quantile(0.5)),
                         "p95": float(counts.quantile(0.95)), "max": int(counts.max()),
                         "zero": int((counts == 0).sum())},
    }
    save_json(rep, config.WORK_DIR / "reports" / "blocking_train.json")
    log(f"blocking report: {rep}")
    return rep
