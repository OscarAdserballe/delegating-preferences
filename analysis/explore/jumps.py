exec(open("load.py").read())
h=held.sort_values(["arm","user_id","menu_index","checkpoint"]).copy()
h["prev"]=h.groupby(["arm","user_id","menu_index"]).pick.shift()
h["flip"]=(h.pick!=h.prev)&h.prev.notna()
dl=df[df.kind=="dialogue"][["arm","user_id","checkpoint","pick","label"]].copy(); dl["inc"]=dl.pick!=dl.label; dl["checkpoint"]+=1
h=h.merge(dl[["arm","user_id","checkpoint","inc"]],on=["arm","user_id","checkpoint"])
print("share of held-out picks that change after the round's feedback (%), by feedback, cp 2–5:")
print((h[(h.checkpoint>=2)&h.arm.isin(LLM+["bayes"])].groupby(["arm","inc"]).flip.mean().unstack()*100).round(1))
# Bayes uses only user choice, so 'inc' is relative to the arm's own pick — for Bayes, inc means Bayes's live pick was wrong
I=pd.read_parquet("implied.parquet").sort_values(["arm","uid","cp"])
I["dcos"]=np.nan
for (a,u),g in I.groupby(["arm","uid"]):
    v=g[[f"th_{f}" for f in FEAT]].to_numpy()
    cs=[np.nan]+[ (v[k]@v[k-1])/(np.linalg.norm(v[k])*np.linalg.norm(v[k-1])+1e-12) for k in range(1,len(v))]
    I.loc[g.index,"dcos"]=cs
print("\nmean cosine between consecutive implied-θ (1 = belief unchanged), cp 2–5:")
print(I[(I.cp>=2)&I.arm.isin(LLM+["bayes"])].pivot_table(index="arm",columns="cp",values="dcos").round(2))
