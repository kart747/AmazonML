"""E-G10: Replay of E-X7 France pseudo-labelling in the unseen-country proxy, TRANSDUCTIVE like the real thing:
US-only stage 2 labels India fold 0 itself (p>=.95 & best -> 1, p<=.05 -> 0), retrain on US folds 1,2 +
those rows (early stop US fold 3), re-predict India fold 0. Grade added/dropped pairs with labels AND with
the label-free house-number signature used for France.

  BER_WORK_DIR=../../work_loco python src/experiments/plfr_replay.py   (~5 min, ~9 GB)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl, lightgbm as lgb
import ber.approaches as ap
from ber import config
from pseudo_label import country_rows, fit, evaluator
W = config.WORK_DIR
feats = lgb.Booster(model_file=str(W / "models" / "s2lab_all.txt")).feature_name()
tr, tu, ev = country_rows(feats, [1, 2]), country_rows(feats, [3]), country_rows(feats, [0])
src_tr, src_tu = tr.filter(pl.col("country") == "US"), tu.filter(pl.col("country") == "US")
ev = ev.filter(pl.col("country") == "India")
del tr, tu
X = lambda d: np.ascontiguousarray(d.select(feats).to_numpy(), dtype=np.float32)
base = fit(src_tr, src_tu, feats)
Xev = X(ev); g = ev.get_column("gid").to_numpy(); y = ev.get_column("y").to_numpy()
p0 = ap._pred(base, Xev); best = ap.exclusive_mask(g, p0)
lab = np.where((p0 >= 0.95) & best, 1, np.where(p0 <= 0.05, 0, -1)); k = lab >= 0
pool = ev.filter(pl.Series(k)).with_columns(y=pl.Series(lab[k]).cast(src_tr.schema["y"]))
print("pseudo rows", int(k.sum()), "pos", int((lab == 1).sum()), "pos prec", round(float(y[lab == 1].mean()), 4), "neg err", round(float(y[lab == 0].mean()), 4))
m1 = fit(pl.concat([src_tr, pool]), src_tu, feats)
p1 = ap._pred(m1, Xev)
e = evaluator(ev, 0, "India")
k0 = (p0 >= 0.68) & ap.exclusive_mask(g, p0)
out = {"base_f05": e.report(k0)["macro_f05"]}
s1 = pl.read_parquet(W / "prep" / "s1_train.parquet", columns=["rid", "ad"]).with_columns(pl.col("rid").cast(pl.Int32))
tg = pl.read_parquet(W / "prep" / "tg_train.parquet", columns=["gid", "ad"]).with_columns(pl.col("gid").cast(pl.Int32))
num = lambda c: pl.col(c).str.extract_all(r"\d+").list.first()
a = (ev.select("rid", "gid").with_row_index("row").join(s1, on="rid", how="left")
       .join(tg.rename({"ad": "ta"}), on="gid", how="left").sort("row").with_columns(hn=num("ad") == num("ta")))
hn = a.get_column("hn").fill_null(False).to_numpy(); hn_ok = a.get_column("hn").is_not_null().to_numpy()
for tau in (0.68, 0.70):
    k1 = (p1 >= tau) & ap.exclusive_mask(g, p1)
    add, drop, both = k1 & ~k0, k0 & ~k1, k0 & k1
    r = {"f05": e.report(k1)["macro_f05"]}
    for nm, m in (("added", add), ("dropped", drop), ("both", both)):
        r[nm] = {"n": int(m.sum()), "true": round(float(y[m].mean()), 4), "hn_eq": round(float(hn[m & hn_ok].mean()), 4)}
    out[f"tau{tau}"] = r
    print(tau, json.dumps(r))
print(json.dumps(out))
json.dump(out, open(str(W / "reports" / "plfr_replay_India.json"), "w"))
