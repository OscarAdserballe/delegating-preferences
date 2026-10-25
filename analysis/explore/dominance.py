exec(open("load.py").read())
th=pd.DataFrame([dict(uid=u,**{f:D.users[u].theta[i] for i,f in enumerate(FEAT)}) for u in users]).set_index("uid")
a=th.abs()
srt=np.sort(a.to_numpy(),1)[:,::-1]
th["dominance"]=srt[:,0]/(srt.sum(1))       # share of total |weight| on the top attribute
th["n_nonzero"]=(a>0).sum(1)
th["top_tie"]=np.isclose(srt[:,0],srt[:,1])  # two attributes tied for max
# lexicographic-consistency: share of the user's held-out choices explained by a single-attribute rule (best of 8 rules)
rules=[(j,s) for j in range(4) for s in (-1,1)]
lex={}
for u in users:
    best=0
    for j,s in rules:
        v=s*H[:,:,j]; ok=np.isclose(v.max(1),v[np.arange(100),D.users[u].heldout_choices])
        best=max(best,ok.mean())
    lex[u]=best
th["lex_fit"]=pd.Series(lex)
acc=held[held.checkpoint==5].groupby(["user_id","arm"]).correct.mean().unstack()
m=th.join(acc)
print(m[["dominance","n_nonzero","lex_fit"]].describe().round(2))
print("\ncorrelations with cp5 accuracy:")
print(m[["dominance","n_nonzero","lex_fit","bayes","gemma4-31b","gemma4-31b-think","qwen25-7b"]].corr().loc[["dominance","n_nonzero","lex_fit"],["bayes","gemma4-31b","gemma4-31b-think","qwen25-7b"]].round(2))
m["lexbin"]=pd.qcut(m.lex_fit,3,labels=["low: no single rule","mid","high: one-attribute rule fits"])
print("\ncp5 accuracy by how well a single-attribute rule explains the user's choices:")
print((m.groupby("lexbin")[["bayes","gemma4-31b","gemma4-31b-think","qwen25-7b"]].mean()*100).round(1))
print("\nlex_fit range per bin:", m.groupby("lexbin").lex_fit.agg(["min","max"]).round(2).to_dict())
th.to_parquet("users2.parquet")
