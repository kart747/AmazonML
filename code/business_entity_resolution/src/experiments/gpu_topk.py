"""GPU exact sparse top-K (drop-in candidate for ber.blocking.topk) + CPU vs GPU benchmark.

Profiling of the train blocking run (work/log_block_train.txt) showed that ~97% of blocking
time is sparse top-K (India 33.3 of 34.5 min, US 86.9 of 88.3 min; the reverse view alone
is 53%). This module computes the same thing on the GPU:

  items B (n_items x V, L2-normalized CSR) live on the GPU as int32 CSR blocks;
  each query chunk A_c (q x V) is scattered into a dense V x q matrix;
  cuSPARSE SpMM  B_block @ A_c^T  -> (block x q) scores -> top-k over items -> merge blocks.
Only strictly positive scores are kept, exactly like sparse_dot_topn (structural nonzeros), so
outputs are comparable pair-for-pair. Ties at the k-th score may resolve differently.

Benchmark on real data (India, train prep, labels only for recall):
  .venv-dense/bin/python src/experiments/gpu_topk.py bench [--queries 20000] [--pool 1000000] [--dtype float16]
Reports time per query (CPU sparse_dot_topn with 65k column blocks vs GPU), set agreement,
and recall of true pairs for fwd name / fwd combo / rev combo.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ber import config  # noqa: E402


def _csr_to_gpu(M, device, dtype):
    import torch
    M = M.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)),
                                   torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data.astype(np.float32)).to(dtype),
                                   size=M.shape, device=device)


def topk_gpu(A: sp.csr_matrix, B: sp.csr_matrix, k: int, q_chunk=512, out_budget=512e6,
             dtype_name="float32", device="cuda"):
    """Same contract as ber.blocking.topk: (query_row, item_row, score, rank) as numpy arrays."""
    import torch
    dtype = getattr(torch, dtype_name)
    esize = torch.tensor([], dtype=dtype).element_size()
    V = B.shape[1]
    blk = int(max(65536, out_budget // (esize * q_chunk)))
    blocks = [(s, _csr_to_gpu(B[s:s + blk], device, dtype)) for s in range(0, B.shape[0], blk)]
    A = A.tocsr()
    qs, its, scs, rks = [], [], [], []
    with torch.inference_mode():
        for q0 in range(0, A.shape[0], q_chunk):
            Ac = A[q0:q0 + q_chunk].tocoo()
            nq = Ac.shape[0]
            Qt = torch.zeros((V, nq), device=device, dtype=dtype)
            Qt[torch.from_numpy(Ac.col.astype(np.int64)).to(device),
               torch.from_numpy(Ac.row.astype(np.int64)).to(device)] = torch.from_numpy(Ac.data).to(device, dtype)
            best_v = best_i = None
            for s, Bb in blocks:
                out = torch.sparse.mm(Bb, Qt)                       # block x nq
                v, i = torch.topk(out, min(k, out.shape[0]), dim=0)  # k x nq
                i = i + s
                if best_v is None:
                    best_v, best_i = v, i
                else:
                    v2, j = torch.topk(torch.cat([best_v, v], 0), k, dim=0)
                    best_i = torch.gather(torch.cat([best_i, i], 0), 0, j)
                    best_v = v2
                del out
            v = best_v.T.float().cpu().numpy()   # nq x k, sorted descending
            i = best_i.T.cpu().numpy()
            kk = v.shape[1]
            keep = v > 0
            qs.append(np.repeat(np.arange(q0, q0 + nq, dtype=np.int64), kk)[keep.ravel()])
            its.append(i.ravel()[keep.ravel()].astype(np.int64))
            scs.append(v.ravel()[keep.ravel()].astype(np.float32))
            rks.append(np.tile(np.arange(kk, dtype=np.int16), nq)[keep.ravel()])
    return np.concatenate(qs), np.concatenate(its), np.concatenate(scs), np.concatenate(rks)


def _selftest():
    """GPU and CPU top-K agree on random sparse data (no real data needed)."""
    from ber.blocking import topk
    from sklearn.preprocessing import normalize
    rng = np.random.default_rng(0)
    A = normalize(sp.random(3000, 20000, density=0.001, format="csr", dtype=np.float32, random_state=rng))
    B = normalize(sp.random(50000, 20000, density=0.001, format="csr", dtype=np.float32, random_state=rng))
    q1, i1, s1, _ = topk(A, B, 8)
    q2, i2, s2, _ = topk_gpu(A, B, 8)
    a, b = set(zip(q1.tolist(), i1.tolist())), set(zip(q2.tolist(), i2.tolist()))
    print(f"selftest: cpu {len(a)} gpu {len(b)} pairs, jaccard {len(a & b) / len(a | b):.5f}")
    assert len(a & b) / len(a | b) > 0.999


def bench(n_queries, n_rev, dtype_name, pool):
    import polars as pl
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from ber.blocking import _sample, fit_transform, topk
    W = config.WORK_DIR
    s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "country", "nm", "ad"]).filter(pl.col("country") == "India")
    tg = pl.scan_parquet(W / "prep" / "tg_train.parquet").filter(pl.col("country") == "India").select("gid", "nm", "ad").collect()
    gt = pl.read_parquet(W / "prep" / "gt_train.parquet")
    S_rid = s1.get_column("rid").to_numpy()
    rng = np.random.default_rng(0)
    qi = np.sort(rng.choice(len(S_rid), n_queries, replace=False))
    truth_f = set(map(tuple, gt.filter(pl.col("rid").is_in(S_rid[qi].tolist())).select("rid", "gid").rows()))
    # item pool: all true targets of the sampled queries + random others (keeps RAM small;
    # both implementations scale ~linearly with items, so ms/query ratios carry over)
    if pool and tg.height > pool:
        keep = set(g for _, g in truth_f)
        others = tg.get_column("gid").to_numpy()
        others = others[~np.isin(others, list(keep))]
        keep |= set(rng.choice(others, pool - len(keep), replace=False).tolist())
        tg = tg.filter(pl.col("gid").is_in(list(keep)))
    T_gid = tg.get_column("gid").to_numpy()
    print(f"bench pool: S1 {len(S_rid):,} (all India), targets {len(T_gid):,}", flush=True)
    tt = np.array(sorted({g for _, g in truth_f}))
    ti = np.searchsorted(T_gid, tt)
    extra = np.sort(rng.choice(len(T_gid), max(0, n_rev - len(ti)), replace=False))
    ti_all = np.unique(np.r_[ti, extra])
    B = config.BLOCK
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=B["name_max_df"], sublinear_tf=True, lowercase=False, dtype=np.float32)
    va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=B["addr_max_df"], sublinear_tf=True, lowercase=False, dtype=np.float32)
    s_nm, t_nm, s_ad, t_ad = (s1.get_column("nm").to_list(), tg.get_column("nm").to_list(),
                              s1.get_column("ad").to_list(), tg.get_column("ad").to_list())
    NS, NT = fit_transform(vn, _sample(s_nm + t_nm, B["fit_sample"]), s_nm, t_nm)
    AS, AT = fit_transform(va, _sample(s_ad + t_ad, B["fit_sample"]), s_ad, t_ad)
    h = np.float32(math.sqrt(0.5))
    CS, CT = sp.hstack([NS * h, AS * h]).tocsr(), sp.hstack([NT * h, AT * h]).tocsr()
    del s_nm, t_nm, s_ad, t_ad
    cases = {"fwd name k5": (NS[qi], NT, 5, "fwd"), "fwd combo k10": (CS[qi], CT, 10, "fwd"),
             "rev combo k8": (CT[ti_all], CS, 8, "rev")}
    for name, (Aq, Bi, k, d) in cases.items():
        res = {}
        for impl in ("cpu", "gpu"):
            if impl == "gpu":
                torch.cuda.synchronize()
            t0 = time.time()
            q, it, sc, _ = topk(Aq, Bi, k) if impl == "cpu" else topk_gpu(Aq, Bi, k, dtype_name=dtype_name)
            dt = time.time() - t0
            if d == "fwd":
                pairs = set(zip(S_rid[qi[q]].tolist(), T_gid[it].tolist()))
            else:
                pairs = set(zip(S_rid[it].tolist(), T_gid[ti_all[q]].tolist()))
            rec = len(pairs & truth_f) / len(truth_f)
            res[impl] = (dt, pairs, rec)
        (tc, pc, rc), (tg_, pg, rg) = res["cpu"], res["gpu"]
        jac = len(pc & pg) / max(1, len(pc | pg))
        print(f"{name:14} queries {Aq.shape[0]:>7,} items {Bi.shape[0]:>9,} | CPU {tc:7.2f}s ({tc / Aq.shape[0] * 1e3:.3f} ms/q) "
              f"| GPU {tg_:7.2f}s ({tg_ / Aq.shape[0] * 1e3:.3f} ms/q) | speed-up {tc / tg_:5.2f}x | pair-set jaccard {jac:.5f} "
              f"| recall cpu {rc:.4f} gpu {rg:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "bench"])
    ap.add_argument("--queries", type=int, default=20000)
    ap.add_argument("--rev", type=int, default=100000, help="reverse-view queries (targets)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--pool", type=int, default=1_000_000, help="max targets in the item pool (0 = all)")
    a = ap.parse_args()
    _selftest() if a.cmd == "selftest" else bench(a.queries, a.rev, a.dtype, a.pool)


if __name__ == "__main__":
    main()
