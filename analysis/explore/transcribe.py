exec(open("load.py").read())
line_pat=re.compile(r"(\d{1,2}:\d{2}\s*[AP]M|\d+\s*h(r)?\s*\d*\s*m|\$\d+)",re.I)
rows=[]
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if not th: continue
        lines=th.split("\n"); tot=len(th)
        tr=sum(len(x) for x in lines if len(line_pat.findall(x))>=2)   # lines restating a flight's attributes
        rows.append(dict(cp=(e["request"]["n_messages"]-1)//2,share=tr/max(tot,1),cut=e["usage"]["finish_reason"]=="length"))
R=pd.DataFrame(rows)
print("share of thinking characters spent restating flights (lines with >=2 of time/duration/price), by cp (%):")
print((R.groupby("cp").share.mean()*100).round(1).to_dict())
print("…cut-off vs finished traces (cp>=3):", (R[R.cp>=3].groupby("cut").share.mean()*100).round(1).to_dict())
