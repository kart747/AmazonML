import sys, json
sys.path.insert(0,"/home/kart/Projects/AmazonML/code/business_entity_resolution/src"); sys.path.insert(0,"/home/kart/Projects/AmazonML/code/business_entity_resolution/src/experiments")
import numpy as np, polars as pl, lightgbm as lgb
import ber.approaches as ap
from ber import config
from pseudo_label import country_rows, fit
S="/tmp/claude-1000/-home-kart-Projects-AmazonML/2a0a7997-ff1a-4d94-85a3-e464e4de5e89/scratchpad"
W=config.WORK_DIR
allf=lgb.Booster(model_file=str(W/"models"/"s2lab_all.txt")).feature_name()
NAME=["cos_name","n_ratio","n_tsort","n_tset","n_partial","n_jw","n_jacc","n_inter","q_ntok","t_ntok","n_nsp_ratio","n_nsp_partial","t_oov",
      "n_map_tset","n_map_ratio","n_skel_ratio","n_skel_tset","q_unm_maxidf","t_unm_maxidf","q_match_idf","t_match_idf","core_inter","core_jacc","q_core_n","t_core_n","q_nm_freq","t_nm_freq","t_nonascii","t_src"]
ADDR=["cos_addr","a_ratio","a_tset","a_partial","a_jacc","a_inter","num_jacc","num_inter","num_first_eq","num_first_logdiff","q_num_n","t_num_n","q_alen","t_alen",
      "t_addr_empty","a_exact","a_contain","num_symdiff","num_reldiff","num_q_extra","num_t_extra","hn_prefix","hn_suffix","hn_lev","hn_samelen","hn_subst_small","t_src"]
NAME=[f for f in NAME if f in allf]; ADDR=[f for f in ADDR if f in allf]
print(len(NAME),len(ADDR))
feats=sorted(set(NAME+ADDR))
tr,tu=country_rows(feats,[1,2]),country_rows(feats,[3])
us_tr,us_tu=tr.filter(pl.col("country")=="US"),tu.filter(pl.col("country")=="US"); del tr,tu
mn=fit(us_tr,us_tu,NAME); ma=fit(us_tr,us_tu,ADDR); del us_tr
L=pl.read_parquet(f"{S}/loco_all.parquet",columns=["rid","gid","fold","country","y","p","keep","best"]).filter(pl.col("country")=="India")
ev=country_rows(feats,[0,3]).filter(pl.col("country")=="India")
ev=ev.select("rid","gid","fold",*feats).join(L,on=["rid","gid","fold"],how="inner")
X=lambda fs: np.ascontiguousarray(ev.select(fs).to_numpy(),dtype=np.float32)
ev=ev.with_columns(pn=pl.Series(ap._pred(mn,X(NAME))), pa=pl.Series(ap._pred(ma,X(ADDR))))
ev.select("rid","gid","fold","y","p","keep","best","pn","pa").write_parquet(f"{S}/cotrain_india.parquet")
band=(pl.col("p")>=0.02)&(pl.col("p")<0.68)
rows=[]
for f in (3,0):
    g=ev.filter((pl.col("fold")==f)&band); T=int(g["y"].sum())
    for nm,cond in {"name>=.95":pl.col("pn")>=0.95,"addr>=.95":pl.col("pa")>=0.95,"name>=.99":pl.col("pn")>=0.99,"addr>=.99":pl.col("pa")>=0.99,
                    "both>=.9":(pl.col("pn")>=0.9)&(pl.col("pa")>=0.9),"either>=.99":(pl.col("pn")>=0.99)|(pl.col("pa")>=0.99),
                    "name>=.95&best":(pl.col("pn")>=0.95)&pl.col("best"),"addr>=.95&best":(pl.col("pa")>=0.95)&pl.col("best")}.items():
        x=g.filter(cond); tp=int(x["y"].sum())
        rows.append({"fold":f,"rule":nm,"n":x.height,"tp":tp,"prec":round(tp/max(x.height,1),3),"cov":round(tp/max(T,1),3)})
    # negatives: view-confident negatives among band
    for nm,cond in {"name<=.05":pl.col("pn")<=0.05,"addr<=.05":pl.col("pa")<=0.05,"both<=.1":(pl.col("pn")<=0.1)&(pl.col("pa")<=0.1)}.items():
        x=g.filter(cond); fp=int((x["y"]==0).sum())
        rows.append({"fold":f,"rule":"NEG "+nm,"n":x.height,"tp":fp,"prec":round(fp/max(x.height,1),3),"cov":round(fp/max(g.height-T,1),3)})
pl.Config.set_tbl_rows(40); print(pl.DataFrame(rows))
