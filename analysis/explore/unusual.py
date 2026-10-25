exec(open("load.py").read())
pat=re.compile(r"\b(unusual|weird|strange|counter-?intuitive|odd|bizarre|irrational|surprising|atypical|uncommon|rare)\b",re.I)
reg=re.compile(r"regardless of",re.I)
M=pd.read_parquet("traces_joined.parquet")
rows=[]
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if th: rows.append((e["key"],len(pat.findall(th))>0,len(reg.findall(th))>0))
M=M.merge(pd.DataFrame(rows,columns=["key","unusual","regardless"]),on="key")
M["contra_grp"]=M.contra.clip(upper=2)
print("share of traces calling something unusual/weird/odd..., by cp and user contrarianism (%):")
print((M.pivot_table(index="contra_grp",columns="cp",values="unusual")*100).round(1))
print("\nshare of traces with 'regardless of' (absolute single-attribute rules), by cp (%):", (M.groupby("cp").regardless.mean()*100).round(1).to_dict())
print("accuracy with/without 'regardless of', cp>=2:", (M[M.cp>=2].groupby("regardless").correct.mean()*100).round(1).to_dict())
print("accuracy with/without 'unusual', cp>=2:", (M[M.cp>=2].groupby("unusual").correct.mean()*100).round(1).to_dict())
