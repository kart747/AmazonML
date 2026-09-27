import sys
sys.path.insert(0,"/home/kart/Projects/AmazonML/code/business_entity_resolution/src"); sys.path.insert(0,"/home/kart/Projects/AmazonML/code/business_entity_resolution/src/experiments")
import polars as pl, numpy as np, lightgbm as lgb
import ber.approaches as ap
from guard_audit import load, LOCO, XD
S="/tmp/claude-1000/-home-kart-Projects-AmazonML/2a0a7997-ff1a-4d94-85a3-e464e4de5e89/scratchpad"
m=lgb.Booster(model_file=str(LOCO/"models"/"loco_s2_US.txt"))
meta,X=load(LOCO,"s2lab",m.feature_name())
p=ap._pred(m,X); del X
s1=pl.read_parquet(XD/"prep"/"s1_train.parquet",columns=["rid","country","nm","ad"]).with_columns(pl.col("rid").cast(pl.Int32))
tg=pl.read_parquet(XD/"prep"/"tg_train.parquet",columns=["gid","nm","ad","src"]).with_columns(pl.col("gid").cast(pl.Int32))
d=meta.with_columns(p=pl.Series(p)).join(s1.rename({"nm":"qn","ad":"qa"}),on="rid",how="left").join(tg.rename({"nm":"tn","ad":"ta"}),on="gid",how="left")
# exclusivity within (country, fold) world
parts=[]
for (c,f),g in d.group_by(["country","fold"]):
    ex=ap.exclusive_mask(g["gid"].to_numpy(),g["p"].to_numpy())
    parts.append(g.with_columns(best=pl.Series(ex), keep=pl.Series(ex&(g["p"].to_numpy()>=0.68))))
pl.concat(parts).write_parquet(f"{S}/loco_all.parquet"); print(pl.concat(parts).group_by("country","fold").len())
