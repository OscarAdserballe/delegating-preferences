exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
pc=re.compile(r"(score|points?\b|weight(ed|ing|s)?\b|trade-?off|total|sum|net benefit|\+\d|\-\d\b)",re.I)
pl=re.compile(r"(priority\s*\d|top priority|most important|primary (factor|driver)|regardless of|above all|tie-?break|first criterion|secondary)",re.I)
f={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: f[e["key"]]=(len(pc.findall(th)),len(pl.findall(th)))
M=M.merge(pd.DataFrame([(k,*v) for k,v in f.items()],columns=["key","comp","lex"]),on="key")
M["style"]=np.select([(M.comp>=3)&(M.lex==0),(M.lex>=1)&(M.comp<3)],["compensatory","lexicographic"],"mixed/neither")
m=M[M.cp>=2]
print("reasoning style share (cp>=2, %):", (m["style"].value_counts(normalize=True)*100).round(1).to_dict())
print("accuracy by style and cp (%):"); print((m.pivot_table(index="style",columns="cp",values="correct")*100).round(1))
nz={u:(D.users[u].theta!=0).sum() for u in users}; m=m.assign(nz=m.uid.map(nz))
print("\naccuracy by style × # attributes the user cares about (cp>=2):"); print((m.pivot_table(index="style",columns="nz",values="correct")*100).round(1))
