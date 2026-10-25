exec(open("load.py").read())
ask_h={prompts.ask(m):i for i,m in enumerate(D.heldout)}
rows=[]
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    uid=int(Path(p).name.split(".")[0])
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking")
        if not th: continue
        lu=e["request"]["last_user"]; lu2=lu.removesuffix("\n\n"+prompts.FORMAT_REMINDER)
        idx=None
        for q,i in ask_h.items():
            if lu2.endswith(q): idx=i;break
        cp=(e["request"]["n_messages"]-1)//2
        low=th.lower()
        rows.append(dict(uid=uid,cp=cp,menu_index=idx,chars=len(th),cut=e["usage"]["finish_reason"]=="length",
          prev_incorrect="is incorrect" in lu, key=e["key"],
          norm=len(re.findall(r"\b(objectively|usually|typically|generally|most people|most travel(l)?ers|standard|common sense|obviously|clearly the best)\b",low)),
          wait=len(re.findall(r"\bwait\b",low)),
          contradict=bool(re.search(r"contradict|inconsistent|conflict",low)),
          dep=len(re.findall(r"departure|morning|evening|afternoon|early|late|red-?eye|\bam\b|\bpm\b",low)),
          price=len(re.findall(r"price|cheap|expensive|cost|\$",low)),
          stops=len(re.findall(r"stop",low)),
          dur=len(re.findall(r"duration|long|short|fast|hours?\b|\bhr\b",low)),
          hyp=len(re.findall(r"hypothes|maybe the user|perhaps the user|the user (prefers|likes|values|wants)",low)),
          rank=bool(re.search(r"(rank|priorit|order of importance|weights?)",low)),
          cheaper_better=bool(re.search(r"(lower|cheaper)( price)? is better|prefers? (the )?cheap|price.{0,20}lower is better",low)),
          expensive_ok=bool(re.search(r"(prefers?|likes?|values?) (the )?(more )?expensive|higher price|doesn'?t care about (the )?price|not price[- ]sensitive|price (is )?(not|less) important",low)),
        ))
X=pd.DataFrame(rows); X.to_parquet("traces.parquet"); print(len(X), X.menu_index.notna().mean())
