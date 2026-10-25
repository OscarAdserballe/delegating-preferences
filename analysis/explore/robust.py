exec(open("load.py").read())
rng=np.random.default_rng(0)
def ci(x,n=5000):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; b=x[rng.integers(0,len(x),(n,len(x)))].mean(1); return f"{x.mean():+.1f} [{np.percentile(b,2.5):+.1f}, {np.percentile(b,97.5):+.1f}] (n={len(x)})"
def ci_diff(a,b,n=5000):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ba=a[rng.integers(0,len(a),(n,len(a)))].mean(1); bb=b[rng.integers(0,len(b),(n,len(b)))].mean(1); d=ba-bb
    return f"{a.mean()-b.mean():+.1f} [{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}]"
th=pd.read_parquet("users.parquet").set_index("uid")
acc=held.groupby(["user_id","arm","checkpoint"]).correct.mean().unstack(["arm","checkpoint"])*100
conv=[u for u in users if th.loc[u,"contra"]==0]; contr=[u for u in users if th.loc[u,"contra"]>=2]
for a in ["gemma4-31b-think","gemma4-31b","bayes"]:
    g=acc[(a,5)]-acc[(a,0)]
    print(f"{a} gain cp0→5: conventional {ci(g[conv])} | 2–3 contrarian {ci(g[contr])} | diff {ci_diff(g[contr],g[conv])}")
gap=acc[("bayes",5)]-acc[("gemma4-31b-think",5)]
print("Bayes − thinking gap at cp5: conventional",ci(gap[conv]),"| contrarian",ci(gap[contr]))
# echo per user
lastpos={(u,cp):D.users[u].choices[cp-1] for u in users for cp in range(1,6)}
h=held[(held.checkpoint>=1)&held.arm.isin(LLM)].copy(); h["echo"]=[p==lastpos[(u,c)] for p,u,c in zip(h.pick,h.user_id,h.checkpoint)]
dl=df[df.kind=="dialogue"][["arm","user_id","checkpoint","pick","label"]].copy(); dl["inc"]=dl.pick!=dl.label; dl["checkpoint"]+=1
h=h.merge(dl[["arm","user_id","checkpoint","inc"]],on=["arm","user_id","checkpoint"])
for a in LLM:
    e=h[h.arm==a].groupby(["user_id","inc"]).echo.mean().unstack()*100
    print(f"{a} echo after incorrect − after correct (per-user paired): {ci((e[True]-e[False]).dropna())}")
# departure sign recovery: per-user indicator at cp5
I=pd.read_parquet("implied.parquet"); I5=I[I.cp==5]
for a in ["gemma4-31b-think","bayes"]:
    x=I5[I5.arm==a].set_index("uid")
    r=[float(np.sign(x.loc[u,"th_departure"])==np.sign(D.users[u].theta[0]))*100 for u in users if D.users[u].theta[0]!=0]
    p=[float(np.sign(x.loc[u,"th_price"])==np.sign(D.users[u].theta[3]))*100 for u in users if D.users[u].theta[3]>0]
    print(f"{a}: departure direction right {ci(r)} | price direction right for pricier-loving users {ci(p)}")
