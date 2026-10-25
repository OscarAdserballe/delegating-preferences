exec(open("load.py").read())
M=pd.read_parquet("traces_joined.parquet")
B=BayesianAssistant(T); unc={}
for u in users:
    us=D.users[u]; s=DialogueState()
    for cp in range(6):
        unc[(u,cp)]=np.log2((B.posterior(s)>0).sum())
        if cp<5: s=s.after(PastRound(us.rounds[cp],None,us.choices[cp]))
U=M.groupby(["uid","cp"]).agg(chars=("chars","mean"),cut=("cut","mean"),acc=("correct","mean"),contra=("contra","first")).reset_index()
U["bayes_bits_left"]=[unc[(u,c)] for u,c in zip(U.uid,U.cp)]
dl=df[(df.arm=="gemma4-31b-think")&(df.kind=="dialogue")][["user_id","checkpoint","pick","label"]]
n_incorrect={}
for u in users:
    g=dl[dl.user_id==u].sort_values("checkpoint")
    for cp in range(6): n_incorrect[(u,cp)]=int(((g.checkpoint<cp)&(g.pick!=g.label)).sum())
U["n_incorrect_so_far"]=[n_incorrect[(u,c)] for u,c in zip(U.uid,U.cp)]
u=U[U.cp>=1]
print("per (user, checkpoint>=1): correlations with mean thinking length and cut-off share")
print(u[["chars","cut","bayes_bits_left","contra","n_incorrect_so_far","acc"]].corr()[["chars","cut"]].round(2))
print("\nthinking length by # 'incorrect' received so far (chars, cut-off share):")
print(u.groupby("n_incorrect_so_far")[["chars","cut","acc"]].mean().round(2).join(u.groupby("n_incorrect_so_far").size().rename("n")))
U.to_parquet("effort.parquet")
