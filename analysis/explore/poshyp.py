exec(open("load.py").read())
pat=re.compile(r"(\"flight [123]\" pattern|flight [123] pattern|always (picks?|chooses?|prefers?) flight [123]|position|always the (first|second|third|last) (option|flight)|(first|last|third) option every|flight number)",re.I)
M=pd.read_parquet("traces_joined.parquet")
hits=[]
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: hits.append((e["key"],bool(pat.search(th))))
Hh=pd.DataFrame(hits,columns=["key","pos_hyp"])
M=M.merge(Hh,on="key",how="left")
print("share of thinking traces entertaining a position/label hypothesis, by cp:", (M.groupby("cp").pos_hyp.mean()*100).round(2).to_dict())
print("accuracy with vs without:", (M[M.cp>=1].groupby("pos_hyp").correct.mean()*100).round(1).to_dict())
