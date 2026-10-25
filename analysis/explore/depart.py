exec(open("load.py").read())
pats={"dismiss":r"departure(?: time)?s?\b[^.\n]{0,40}(doesn'?t|does not|don'?t|do not|isn'?t|is not|seems? not)[^.\n]{0,25}(matter|important|factor|relevant|priority|consistent|pattern|driver)",
      "no_pattern":r"(no (clear |consistent |obvious )?pattern)[^.\n]{0,30}departure|departure[^.\n]{0,30}(no (clear |consistent |obvious )?pattern|random|varies|inconsistent)",
      "time_of_day_norm":r"(red-?eye|too early|inconvenient|reasonable (departure )?time|civilized|convenient (departure )?time|mid-?day|morning flights? (are|is) (better|preferred))",
      "prefers_early":r"prefers? (an )?(early|earlier|morning)",
      "prefers_late":r"prefers? (a )?(late|later|evening|night)"}
cnt={k:0 for k in pats}; n=0; byu=[]
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    u=int(Path(p).name.split(".")[0])
    for l in gzip.open(p,"rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        if not th or e["request"]["n_messages"]<9: continue  # cp>=4
        n+=1; d={k:bool(re.search(v,th,re.I)) for k,v in pats.items()}
        for k in pats: cnt[k]+=d[k]
        byu.append(dict(uid=u,dep=D.users[u].theta[0],**d))
print("cp>=4 traces:",n); print({k:round(v/n*100,1) for k,v in cnt.items()})
B=pd.DataFrame(byu); B["dep_sign"]=np.sign(B.dep)
print((B.groupby("dep_sign")[list(pats)].mean()*100).round(1))
