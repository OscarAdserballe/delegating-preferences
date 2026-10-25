exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
pi=re.compile(r"(crucial|key|important|critical) (insight|observation|finding|clue)",re.I)
pr=re.compile(r"priority\s*1|top priority|most important (factor|attribute|criterion)|primary (driver|factor)",re.I)
f={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: f[e["key"]]=(bool(pi.search(th)),bool(pr.search(th)))
M=M.merge(pd.DataFrame([(k,*v) for k,v in f.items()],columns=["key","insight","top_priority"]),on="key")
m=M[M.cp>=1]
print("share of traces (cp>=1) with 'key/crucial insight' and with a single 'top priority' attribute, by last feedback (%):")
print((m.groupby("prev_incorrect")[["insight","top_priority"]].mean()*100).round(1))
print("\naccuracy with/without a single top-priority attribute (cp>=1):", (m.groupby("top_priority").correct.mean()*100).round(1).to_dict())
print("…for users who care about 1–2 attributes vs 3–4:")
nz={u:(D.users[u].theta!=0).sum() for u in users}; m=m.assign(nz=m.uid.map(nz).clip(lower=2))
print((m.groupby(["nz","top_priority"]).correct.mean().unstack()*100).round(1))
