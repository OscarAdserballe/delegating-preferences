exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
pe=re.compile(r"prefers? (an )?(early|earlier|morning)",re.I); pl=re.compile(r"prefers? (a )?(late|later|evening|night)",re.I)
pc=re.compile(r"prefers? (the )?(cheap|cheapest|lower price|lowest price)",re.I); px=re.compile(r"prefers? (the )?(more )?(expensive|most expensive|higher price|pricier)",re.I)
flags={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: flags[e["key"]]=(bool(pe.search(th)),bool(pl.search(th)),bool(pc.search(th)),bool(px.search(th)))
F=pd.DataFrame([(k,*v) for k,v in flags.items()],columns=["key","says_early","says_late","says_cheap","says_pricey"])
M=M.merge(F,on="key")
th_pick=held[held.arm=="gemma4-31b-think"].set_index(["user_id","checkpoint","menu_index"]).pick
M["pick"]=[th_pick.get((u,c,i)) for u,c,i in zip(M.uid,M.cp,M.menu_index)]
M=M[M.pick.notna()&(M.cp>=2)]
dep=H[M.menu_index.astype(int).to_numpy(),:,0]; pr=H[M.menu_index.astype(int).to_numpy(),:,3]; pk=M.pick.astype(int).to_numpy()
M["picked_earliest"]=np.isclose(dep[np.arange(len(M)),pk],dep.min(1)); M["picked_latest"]=np.isclose(dep[np.arange(len(M)),pk],dep.max(1))
M["picked_cheapest"]=np.isclose(pr[np.arange(len(M)),pk],pr.min(1)); M["picked_priciest"]=np.isclose(pr[np.arange(len(M)),pk],pr.max(1))
for say,act in [("says_early","picked_earliest"),("says_late","picked_latest"),("says_cheap","picked_cheapest"),("says_pricey","picked_priciest")]:
    g=M.groupby(say)[act].mean()*100
    print(f"{act}: when trace {say}: {g.get(True,np.nan):.1f}%  vs otherwise {g.get(False,np.nan):.1f}%   (n={M[say].sum()})")
