"""Exact challenge metric plus diagnostics.

Per S1 entity (every S1 of the evaluation population counts, singletons included):
  n_true == 0: F = 1 if nothing predicted else 0
  otherwise:   F = 0 if tp == 0 else 1.25*P*R / (0.25*P + R)
Macro F0.5 = mean over S1. Blocking misses count as false negatives ("realistic" score).
"""
import numpy as np


def f05_per_entity(tp, n_pred, n_true):
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def _selftest():
    # spec example: predict 3, 2 correct, 2 true -> 0.714
    assert abs(f05_per_entity([2], [3], [2])[0] - 0.7142857) < 1e-6
    assert f05_per_entity([0], [0], [0])[0] == 1.0 and f05_per_entity([0], [1], [0])[0] == 0.0
    assert f05_per_entity([0], [0], [3])[0] == 0.0


_selftest()


class Evaluator:
    """Fast repeated scoring of one population of S1 entities.

    s1_idx:  per-pair index (0..n_s1-1) of the pair's S1 within the population
    y:       per-pair label (1 = true match)
    n_true:  per-S1 number of true matches (from ground truth, includes blocking misses)
    """

    def __init__(self, s1_idx, y, n_true):
        self.s1_idx = np.asarray(s1_idx, dtype=np.int64)
        self.y = np.asarray(y, dtype=bool)
        self.n_true = np.asarray(n_true, dtype=np.int64)
        self.n = len(self.n_true)

    def counts(self, keep):
        n_pred = np.bincount(self.s1_idx[keep], minlength=self.n)
        tp = np.bincount(self.s1_idx[keep & self.y], minlength=self.n)
        return tp, n_pred

    def macro_f05(self, keep):
        tp, n_pred = self.counts(keep)
        return float(f05_per_entity(tp, n_pred, self.n_true).mean())

    def report(self, keep):
        tp, n_pred = self.counts(keep)
        f = f05_per_entity(tp, n_pred, self.n_true)
        single = self.n_true == 0
        has_pred, has_true = n_pred > 0, self.n_true > 0
        fp = int((keep & ~self.y).sum())
        tot_tp, tot_pred, tot_true = int(tp.sum()), int(n_pred.sum()), int(self.n_true.sum())
        bucket = {}
        for name, m in (("0", self.n_true == 0), ("1", self.n_true == 1),
                        ("2-3", (self.n_true >= 2) & (self.n_true <= 3)), ("4+", self.n_true >= 4)):
            bucket[name] = {"n": int(m.sum()), "f05": round(float(f[m].mean()), 5) if m.any() else None}
        return {
            "macro_f05": round(float(f.mean()), 5),
            "macro_precision": round(float((tp[has_pred] / n_pred[has_pred]).mean()), 5),
            "macro_recall": round(float((tp[has_true] / self.n_true[has_true]).mean()), 5),
            "pair_precision": round(tot_tp / max(1, tot_pred), 5),
            "pair_recall": round(tot_tp / max(1, tot_true), 5),
            "false_positive_pairs": fp,
            "false_positive_rate": round(fp / max(1, tot_pred), 5),
            "s1_with_any_fp": int(((n_pred - tp) > 0).sum()),
            "singletons": int(single.sum()),
            "singleton_accuracy": round(float((n_pred[single] == 0).mean()), 5) if single.any() else None,
            "false_empty_rate": round(float((n_pred[has_true] == 0).mean()), 5),
            "predicted_pairs_per_s1": round(tot_pred / self.n, 3),
            "f05_by_true_count": bucket,
            "n_s1": self.n,
        }
