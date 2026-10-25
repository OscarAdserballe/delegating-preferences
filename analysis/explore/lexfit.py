exec(open("load.py").read())
import itertools
I=pd.read_parquet("implied.parquet")
# lexicographic rules: ordered pair of (attribute, direction) for primary and secondary, third random
rules=[(a,s1,b,s2) for a in range(4) for b in range(4) if a!=b for s1 in (-1,1) for s2 in (-1,1)]
def lexpick(i,a,s1,b,s2):
    v=s1*H[i,:,a]; c=np.flatnonzero(np.isclose(v,v.max()))
    if len(c)>1:
        w=s2*H[i,c,b]; c=c[np.isclose(w,w.max())]
    return set(c)
LP={(i,k):lexpick(i,*r) for i in range(100) for k,r in enumerate(rules)}
rows=[]
for (arm,u,cp),g in held[held.arm.isin(LLM+["bayes","random","oracle"])].groupby(["arm","user_id","checkpoint"]):
    p=dict(zip(g.menu_index,g.pick))
    best=max(np.mean([ (not pd.isna(p[i])) and int(p[i]) in LP[(i,k)] for i in range(100)]) for k in range(len(rules)))
    rows.append(dict(arm=arm,uid=u,cp=cp,lex=best))
L=pd.DataFrame(rows).merge(I[["arm","uid","cp","fit"]],on=["arm","uid","cp"])
print("best 2-level lexicographic rule fit vs best linear θ fit (share of 100 picks):")
print(L.groupby(["arm","cp"])[["lex","fit"]].mean().unstack(0).round(3).swaplevel(axis=1).sort_index(axis=1))
L.to_parquet("lexfit.parquet")
