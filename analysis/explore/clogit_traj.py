exec(open("load.py").read())
def fit(mi,ch,l2=0.05):
    Xs=H[mi]; w=np.zeros(4)
    for it in range(1200):
        s=Xs@w; s-=s.max(1,keepdims=True); p=np.exp(s); p/=p.sum(1,keepdims=True)
        g=(Xs[np.arange(len(ch)),ch]-(p[...,None]*Xs).sum(1)).mean(0)-l2*w; w+=2.0*g
    return w
rows=[]
for (arm,u,cp),g in held[held.arm.isin(LLM+["bayes","oracle"])&held.pick.notna()].groupby(["arm","user_id","checkpoint"]):
    w=fit(g.menu_index.to_numpy(),g.pick.astype(int).to_numpy()); t=D.users[u].theta
    rows.append(dict(arm=arm,uid=u,cp=cp,**{f"w_{f}":w[j] for j,f in enumerate(FEAT)},**{f"t_{f}":t[j] for j,f in enumerate(FEAT)}))
C=pd.DataFrame(rows); C.to_parquet("clogit.parquet")
c5=C[C.cp==5]
print("mean |weight| per attribute at cp5 (scale of revealed sensitivity):")
print(c5.groupby("arm")[[f"w_{f}" for f in FEAT]].apply(lambda g:g.abs().mean()).round(2))
out=[]
for f in FEAT:
    for d,lab in [(-1,"neg"),(1,"pos")]:
        s=C[np.sign(C[f"t_{f}"])==d]
        out.append((s.assign(right=np.sign(s[f"w_{f}"])==d).groupby(["arm","cp"]).right.mean()*100).rename(f"{f}_{lab}"))
O=pd.concat(out,axis=1)
print("\ndirection recovered (%), by cp — thinking Gemma vs Bayes:")
print(O.loc[["gemma4-31b-think","bayes"]].round(0).T)
