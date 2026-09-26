"""Decision rules that turn pair probabilities into per-S1 match sets.

All decoders take arrays aligned by pair (s1 index, gid, p, source) and return a keep
mask. Parameters are always chosen on the tuning fold and reported on the eval fold
(see src/experiments/decoder_lab.py).

  threshold        p >= tau
  excl             keep a target only for its highest-scoring S1 (many-to-one structure)
  relative         p >= tau and p >= alpha * (best p of this S1)
  per_source       tau_S2 / tau_S3 separately
  expected_f05     per S1 choose the prefix size k (0 allowed) of the p-sorted candidates
                   that maximizes the expected F0.5 under the ratio-of-expectations
                   approximation E[F] ~ 1.25*sum_topk(p) / (k + 0.25*(sum_all(p) + lam)),
                   with E[F(empty)] = prod(1 - p) * exp(-lam) (probability of no true
                   match); lam = expected true matches that blocking missed.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def exclusive(gid, p):
    """True where the pair has the highest score among all pairs sharing its target."""
    order = np.lexsort((-p, gid))
    first = np.ones(len(gid), bool)
    first[1:] = gid[order][1:] != gid[order][:-1]
    m = np.zeros(len(gid), bool)
    m[order[first]] = True
    return m


def group_max(s1, p, n):
    mx = np.zeros(n, dtype=np.float64)
    np.maximum.at(mx, s1, p)
    return mx[s1]


def relative(s1, p, n, tau, alpha):
    return (p >= tau) & (p >= alpha * group_max(s1, p, n))


def source_caps(s1, src, p, keep, caps=((2, 5), (3, 6))):
    """Keep at most `cap` pairs per (S1, source) among `keep`, by descending p.

    Ground truth never has more than 5 S2 or 6 S3 matches per S1 (0 violations over
    7.64M train pairs), so any extra prediction is certainly a false positive.
    """
    out = keep.copy()
    idx = np.flatnonzero(keep)
    order = idx[np.lexsort((-p[idx], src[idx], s1[idx]))]
    key = s1[order].astype(np.int64) * 8 + src[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    cap = np.zeros(len(order), dtype=np.int64)
    for s, c in caps:
        cap[src[order] == s] = c
    out[order[rank >= cap]] = False
    return out


def hn_adjust(p, first_eq, first_logdiff, b):
    """House-number guard: +b if first numbers match, -b for a small substitution (1..5)."""
    small = (first_logdiff > 0) & (first_logdiff <= np.log1p(5)) & (first_eq == 0)
    return p + b * (first_eq == 1) - b * small


def expected_f05(s1, p, n, lam=0.0, beta2=0.25):
    """Per-S1 prefix decoding maximizing approximate expected F0.5 (vectorized)."""
    order = np.lexsort((-p, s1))
    s_sorted, p_sorted = s1[order], p[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    lens = np.diff(np.r_[starts, len(s_sorted)])
    rank = np.arange(len(s_sorted)) - np.repeat(starts, lens)          # 0-based
    csum = np.cumsum(p_sorted)
    grp_start_cum = np.repeat(csum[starts] - p_sorted[starts], lens)
    cum_k = csum - grp_start_cum                                          # sum of top-(rank+1)
    total = np.zeros(n)
    np.add.at(total, s_sorted, p_sorted)
    tot = total[s_sorted]
    ef_k = (1 + beta2) * cum_k / ((rank + 1) + beta2 * (tot + lam))       # E[F] selecting top rank+1
    # probability that the S1 has no true match at all (empty set is then perfect)
    log_none = np.zeros(n)
    np.add.at(log_none, s_sorted, np.log1p(-np.clip(p_sorted, 0, 1 - 1e-9)))
    ef_empty = np.exp(log_none - lam)
    best_k = np.zeros(n)                                                  # 0 = empty set
    best_v = ef_empty.copy()
    # best prefix per S1
    grp_best = np.full(n, -1.0)
    np.maximum.at(grp_best, s_sorted, ef_k)
    take = grp_best > best_v
    is_best = ef_k >= grp_best[s_sorted] - 1e-12
    first_best_rank = np.full(n, np.inf)
    np.minimum.at(first_best_rank, s_sorted[is_best], rank[is_best])
    best_k[take] = first_best_rank[take] + 1
    keep_sorted = rank < best_k[s_sorted]
    keep = np.zeros(len(p), bool)
    keep[order] = keep_sorted
    return keep


class Calibrator:
    """Isotonic calibration p -> P(y=1) fitted on the tuning fold."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p, y):
        self.iso.fit(p, y)
        return self

    def __call__(self, p):
        return self.iso.predict(p).astype(np.float32)
