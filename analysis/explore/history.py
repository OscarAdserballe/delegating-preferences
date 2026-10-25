exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
pr=re.compile(r"(?:round|example|case|scenario|turn|interaction|set)\s*#?\s*(\d)",re.I)
f={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: f[e["key"]]=[int(x) for x in pr.findall(th)]
M["refs"]=M.key.map(f)
m=M[(M.cp==5)&M.refs.notna()]
cnt=np.zeros(7); cov=[]
for refs,cut in zip(m.refs,m.cut):
    s=set(r for r in refs if 1<=r<=6)
    for r in s: cnt[r]+=1
    cov.append(len(s & {1,2,3,4,5}))
print("cp5 traces:",len(m),"| share that explicitly reference past round k (%):", {k:round(cnt[k]/len(m)*100,1) for k in range(1,6)})
m=m.assign(cov=cov)
print("accuracy by # of the 5 past rounds explicitly revisited:"); print((m.groupby("cov").correct.agg(["mean","size"])).assign(mean=lambda d:(d["mean"]*100).round(1)))
print("cut-off share by coverage:", (m.groupby("cov").cut.mean()*100).round(0).to_dict())
