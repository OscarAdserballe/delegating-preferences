exec(open("load.py").read())
u=int(sys.argv[1]); us=D.users[u]
dl=df[(df.arm=="gemma4-31b-think")&(df.kind=="dialogue")&(df.user_id==u)].set_index("checkpoint")
E=[json.loads(l) for l in gzip.open(ROOT/f"results/main/calls/gemma4-31b-think/{u}.jsonl.gz","rt")]
print(f"user {u} θ(dep,dur,stops,price)={us.theta.tolist()}")
for r in range(5):
    c=[e for e in E if e["request"]["n_messages"]==2*r+1 and e["request"]["last_user"].endswith(prompts.ask(us.rounds[r])) and e.get("thinking")]
    th=c[-1]["thinking"] if c else ""
    concl=[s for s in re.split(r"\n",th) if re.search(r"(prefer|priorit|hypothes|insight|pattern|rule|driver|consistent)",s,re.I)]
    print(f"\n-- round {r+1}: picked F{int(dl.loc[r].pick)+1}, user wanted F{us.choices[r]+1} ({'✓' if dl.loc[r].pick==us.choices[r] else '✗'}), forced={dl.loc[r].forced}")
    for s in concl[-6:]: print("   ", s.strip()[:200])
