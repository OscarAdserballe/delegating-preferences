exec(open("load.py").read())
lk=paper_data.build_feature_lookup(__import__("json").loads((ROOT/"data/paper"/paper_data.HELDOUT_FILE).read_text()))
dep=sorted(lk[0].items(), key=lambda kv: kv[1]); print("departure values:", dep)
vals=sorted(set(H[:,:,0].ravel()))
def design(Hm):
    # linear duration, stops, price + departure one-hot (drop first)
    oh=np.stack([(np.isclose(Hm[...,0],v)).astype(float) for v in vals[1:]],-1)
    return np.concatenate([Hm[...,1:],oh],-1)
X=design(H)  # 100,3,k
def fit(picks):  # picks: (n,) menu idx, (n,) choice
    mi,ch=picks
    Xs=X[mi]; w=np.zeros(X.shape[-1])
    for it in range(800):
        s=Xs@w; s-=s.max(1,keepdims=True); p=np.exp(s); p/=p.sum(1,keepdims=True)
        g=(Xs[np.arange(len(ch)),ch]-(p[...,None]*Xs).sum(1)).mean(0)-0.01*w
        w+=1.0*g
    return w
names=["duration","stops","price"]+[f"dep={dict((v,k) for k,v in lk[0].items())[v]}" for v in vals[1:]]
out={}
for arm in ["gemma4-31b","gemma4-31b-think","qwen25-7b","oracle"]:
    for cp in [0,5]:
        g=held[(held.arm==arm)&(held.checkpoint==cp)&held.pick.notna()]
        if cp==0: g=g[g.user_id==g.user_id.iloc[0]]
        out[(arm,cp)]=fit((g.menu_index.to_numpy(),g.pick.astype(int).to_numpy()))
R=pd.DataFrame(out,index=names).round(2)
print("conditional-logit weights (departure relative to", dict((v,k) for k,v in lk[0].items())[vals[0]],"):")
print(R)
