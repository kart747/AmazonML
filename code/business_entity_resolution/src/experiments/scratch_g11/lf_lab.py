import polars as pl, numpy as np
from rapidfuzz.distance import Levenshtein
S="/tmp/claude-1000/-home-kart-Projects-AmazonML/2a0a7997-ff1a-4d94-85a3-e464e4de5e89/scratchpad"
LEGAL=["pvt","private","ltd","limited","llp","llc","inc","incorporated","corp","corporation","co","company","plc","lp","pc","pllc","pa",
       "and","the","of","sarl","sas","sasu","eurl","sa","sci","snc","ei","cie","ets","etablissements","compagnie","societe","ste","de","du","des","la","le","les"]
def rel(d):
    core=lambda c: pl.col(c).str.split(" ").list.set_difference(pl.lit(LEGAL))
    d=d.with_columns(qc=core("qn"), tc=core("tn"))
    d=d.with_columns(qx=pl.col("qc").list.set_difference("tc"), tx=pl.col("tc").list.set_difference("qc"),
                     nq=pl.col("qc").list.len(), nt=pl.col("tc").list.len())
    ini=lambda c: pl.col(c).list.eval(pl.element().str.slice(0,1)).list.join("")
    d=d.with_columns(qi=ini("qc"), ti=ini("tc"))
    acr=(((pl.col("nt")==1)&(pl.col("nq")>=2)&(pl.col("tc").list.first()==pl.col("qi")))|
         ((pl.col("nq")==1)&(pl.col("nt")>=2)&(pl.col("qc").list.first()==pl.col("ti"))))
    sw=(pl.col("qx").list.len()==1)&(pl.col("tx").list.len()==1)
    a,b=d["qx"].list.first().fill_null("").to_list(), d["tx"].list.first().fill_null("").to_list()
    sim=np.array([Levenshtein.normalized_similarity(x,y) if x and y else 0.0 for x,y in zip(a,b)])
    num=lambda c: pl.col(c).str.extract_all(r"\d+").list.first()
    qa,ta=pl.col("qa").str.split(" "),pl.col("ta").str.split(" ")
    aj=qa.list.set_intersection(ta).list.len()/qa.list.set_union(ta).list.len()
    d=d.with_columns(sim=pl.Series(sim), aj=aj, hn=(num("qa")==num("ta")).fill_null(False))
    name={"core_eq":(pl.col("qx").list.len()==0)&(pl.col("tx").list.len()==0)&(pl.col("nq")>0),
          "subset":((pl.col("qx").list.len()==0)|(pl.col("tx").list.len()==0))&(pl.col("nq")>0)&(pl.col("nt")>0)&~((pl.col("qx").list.len()==0)&(pl.col("tx").list.len()==0)),
          "acronym":acr, "swap_typo":sw&(pl.col("sim")>=0.7), "swap_word":sw&(pl.col("sim")<0.7)}
    addr={"hn+aj.5":pl.col("hn")&(pl.col("aj")>=0.5), "addr_exact":(pl.col("aj")>=0.999), "any":pl.lit(True)}
    return d, name, addr
def run(tag):
    d=pl.read_parquet(f"{S}/loco_all.parquet")
    d,name,addr=rel(d)
    band=(pl.col("p")>=0.02)&(pl.col("p")<0.68)
    out=[]
    for (c,f),g in sorted(d.group_by(["country","fold"]),key=lambda x:(x[0][0]!="US",x[0][1]!=3)):
        bt=g.filter(band&(pl.col("y")==1)).height; bn=g.filter(band).height
        for nn,ne in name.items():
            for an,ae in addr.items():
                for bst in (False,True):
                    m=ne&ae&(pl.col("best") if bst else pl.lit(True))
                    x=g.filter(band&m); tp=int(x["y"].sum())
                    allm=g.filter(m)
                    out.append({"ctry":c,"fold":f,"rule":f"{nn}&{an}"+("&best" if bst else ""),"band_n":x.height,"band_tp":tp,"band_fp":x.height-tp,
                                "prec":round(tp/max(x.height,1),3),"cov_band_true":round(tp/max(bt,1),3),"all_n":allm.height,"kept_share":round(allm["keep"].mean() or 0,3)})
        print(c,f,"band pairs",bn,"band true",bt, "true rate", round(bt/bn,3))
    r=pl.DataFrame(out); r.write_parquet(f"{S}/lf_{tag}.parquet"); return r
if __name__=="__main__":
    r=run("v1")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
    w=r.pivot(on=["ctry","fold"],index="rule",values=["prec","cov_band_true","band_n"],aggregate_function="first")
    print(w.sort("rule"))
