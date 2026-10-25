exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
pa=re.compile(r"(logic puzzle|puzzle|preference test|this (is a|kind of) (test|task|game)|these types of (tasks|tests|puzzles|prompts)|trick question|benchmark|evaluation)",re.I)
pt=re.compile(r"(typical traveler|most (people|travel(l)?ers)|average traveler|common(ly)?,? people|usually,? people|standard preference)",re.I)
f={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: f[e["key"]]=(bool(pa.search(th)),bool(pt.search(th)))
M=M.merge(pd.DataFrame([(k,*v) for k,v in f.items()],columns=["key","test_aware","typical_traveler"]),on="key")
print("share of traces framing the task as a puzzle/test, and appealing to a 'typical traveller', by cp (%):")
print((M.groupby("cp")[["test_aware","typical_traveler"]].mean()*100).round(1).T)
print("accuracy at cp0 with/without 'typical traveller' appeal:", (M[M.cp==0].groupby("typical_traveler").correct.mean()*100).round(1).to_dict())
