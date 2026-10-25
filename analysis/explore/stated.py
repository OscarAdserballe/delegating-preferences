exec(open("load.py").read())
P={"price":{-1:r"(prefers?|favou?rs?|values?|likes?|wants?|chooses?|picks?)[^.\n]{0,25}(cheap|lower[- ]price|lowest[- ]price|less expensive|budget)|(cheaper|lower price) is (better|preferred)",
             1:r"(prefers?|favou?rs?|values?|likes?|wants?|chooses?|picks?)[^.\n]{0,25}(more expensive|most expensive|higher[- ]price|highest[- ]price|pricier|premium)|(higher price|more expensive) is (better|preferred)"},
   "duration":{-1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(short|shorter|shortest|faster|fastest|quick)",
                1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(long|longer|longest|slower)"},
   "stops":{-1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(fewer stops|non-?stop|direct|0 stops|fewest stops|no stops)",
             1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(more stops|most stops|layovers?|2 stops|multiple stops)"},
   "departure":{-1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(early|earlier|earliest|morning)",
                 1:r"(prefers?|favou?rs?|values?|likes?|wants?)[^.\n]{0,25}(late|later|latest|evening|night)"}}
rows=[]
for u in users:
    us=D.users[u]
    for l in gzip.open(ROOT/f"results/main/calls/gemma4-31b-think/{u}.jsonl.gz","rt"):
        e=json.loads(l); th=e.get("thinking") or ""
        nm=e["request"]["n_messages"]; r=(nm-1)//2
        if not th or r==0 or r>4: continue
        if not e["request"]["last_user"].endswith(prompts.ask(us.rounds[r])): continue  # live dialogue round r (r rounds of history)
        tail=th[-1500:]  # final conclusions
        row=dict(uid=u,round=r)
        for f,d in P.items():
            neg=len(re.findall(d[-1],tail,re.I)); pos=len(re.findall(d[1],tail,re.I))
            row[f]=0 if neg==pos else (1 if pos>neg else -1)
        rows.append(row)
S=pd.DataFrame(rows).drop_duplicates(["uid","round"],keep="last")
print("live-round traces parsed:",len(S))
for f,j in zip(FEAT,range(4)):
    S[f"true_{f}"]=[np.sign(D.users[u].theta[j]) for u in S.uid]
out={}
for f in FEAT:
    s=S[S[f"true_{f}"]!=0]
    out[f]={"states a direction %":(s[f]!=0).mean()*100,"…and it is right %":(s[s[f]!=0][f]==s[s[f]!=0][f"true_{f}"]).mean()*100,
            "right for contrarian-direction users %":(lambda z:(z[f]==z[f"true_{f}"]).mean()*100)(s[(s[f]!=0)&(s[f"true_{f}"]>0)])}
print(pd.DataFrame(out).round(1))
print("\nby round: share of stated directions that are right (all attributes pooled):")
L=[]
for f in FEAT:
    s=S[(S[f"true_{f}"]!=0)&(S[f]!=0)]; L.append(pd.DataFrame({"round":s["round"],"right":s[f]==s[f"true_{f}"],"feat":f}))
L=pd.concat(L); print((L.pivot_table(index="feat",columns="round",values="right")*100).round(0))
S.to_parquet("stated.parquet")
