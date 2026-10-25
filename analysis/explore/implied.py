exec(open("load.py").read())
# best option per theta per menu: (624,100)
U=np.einsum("tf,mof->tmo",T,H)
bestmask=U>=U.max(2,keepdims=True)-1e-9   # (624,100,3)
rows=[]
for (arm,uid,cp),g in held.groupby(["arm","user_id","checkpoint"]):
    g=g.sort_values("menu_index"); p=g.pick.to_numpy()
    ok=~pd.isna(p); pi=np.where(ok,p,0).astype(int)
    fit=(bestmask[:,np.arange(100),pi]&ok).sum(1)/100  # share of picks rationalised by each theta
    top=fit.max(); idx=np.flatnonzero(fit>=top-1e-9)
    th=T[idx].mean(0); tt=D.users[uid].theta
    cos=th@tt/(np.linalg.norm(th)*np.linalg.norm(tt)+1e-12)
    rows.append(dict(arm=arm,uid=uid,cp=cp,fit=top,n_max=len(idx),cos=cos,**{f"th_{f}":th[i] for i,f in enumerate(FEAT)},rank_true=(fit>fit[uid]).sum()))
I=pd.DataFrame(rows); I.to_parquet("implied.parquet")
print("best-fit rationalisability (share of 100 picks explained by the single best θ):")
print(I.pivot_table(index="arm",columns="cp",values="fit").round(3))
print("\ncosine(implied θ, true θ):")
print(I.pivot_table(index="arm",columns="cp",values="cos").round(2))
print("\nrank of the TRUE θ among 624 by fit (0=best):")
print(I.pivot_table(index="arm",columns="cp",values="rank_true",aggfunc="median"))
d0=I[(I.cp==0)].groupby("arm")[[f"th_{f}" for f in FEAT]].mean().round(2)
print("\ncheckpoint-0 implied default θ (departure+ = later, duration+ = longer, stops+ = more, price+ = pricier):\n",d0)
