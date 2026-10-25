exec(open("load.py").read())
def fit(mi,ch,l2=0.05):
    Xs=H[mi]; w=np.zeros(4)
    for it in range(1500):
        s=Xs@w; s-=s.max(1,keepdims=True); p=np.exp(s); p/=p.sum(1,keepdims=True)
        g=(Xs[np.arange(len(ch)),ch]-(p[...,None]*Xs).sum(1)).mean(0)-l2*w; w+=2.0*g
    return w
rows=[]
for (arm,u),g in held[(held.checkpoint==5)&held.arm.isin(LLM+["bayes","oracle"])&held.pick.notna()].groupby(["arm","user_id"]):
    w=fit(g.menu_index.to_numpy(),g.pick.astype(int).to_numpy()); t=D.users[u].theta
    for j,f in enumerate(FEAT):
        if t[j]!=0: rows.append(dict(arm=arm,feat=f,right=np.sign(w[j])==np.sign(t[j]),contrarian=(t[j]>0 and f!="departure")))
R=pd.DataFrame(rows)
print("direction of each attribute's weight in a per-user conditional logit on cp5 picks matches the truth (%):")
print((R.pivot_table(index="arm",columns="feat",values="right")*100).round(0))
print("\n…for contrarian directions only (likes long / stops / pricier):")
print((R[R.contrarian].pivot_table(index="arm",columns="feat",values="right")*100).round(0))
