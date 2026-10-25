exec(open("load.py").read())
B=BayesianAssistant(T)
rows=[]
for u in users:
    us=D.users[u]; s=DialogueState()
    for cp in range(6):
        if cp>0:
            chosen=np.stack([us.rounds[k].phi[us.choices[k]] for k in range(cp)])
            label=np.array(us.heldout_choices)
            # distance from the user's best held-out flight to the nearest previously chosen flight
            dist=np.linalg.norm(H[np.arange(100),label][:,None,:]-chosen[None],axis=2).min(1)
            mt=B.posterior(s)@T; ut=H@mt; srt=np.sort(ut,1)
            conf=srt[:,-1]-srt[:,-2]
            for i in range(100): rows.append(dict(uid=u,cp=cp,menu_index=i,dist=dist[i],bayes_conf=conf[i]))
        if cp<5: s=s.after(PastRound(us.rounds[cp],None,us.choices[cp]))
S=pd.DataFrame(rows)
h=held[held.checkpoint>=1].merge(S,left_on=["user_id","checkpoint","menu_index"],right_on=["uid","cp","menu_index"])
h["dbin"]=pd.qcut(h.dist,4,labels=["near a past choice","q2","q3","far from past choices"])
print("accuracy by distance of the user's best held-out flight to the nearest flight they chose before (cp>=1):")
print((h.groupby(["dbin","arm"]).correct.mean().unstack()[LLM+["bayes","copy_last"]]*100).round(1))
w=h.pivot_table(index=["user_id","checkpoint","menu_index","bayes_conf"],columns="arm",values="pick").reset_index()
w["cbin"]=pd.qcut(w.bayes_conf.rank(method="first"),4,labels=["Bayes unsure","q2","q3","Bayes confident"])
print("\nagreement with Bayes's pick by Bayes's confidence (%):")
print((w.groupby("cbin")[LLM].apply(lambda g: pd.Series({a:(g[a]==w.loc[g.index,"bayes"]).mean() for a in LLM}))*100).round(1))
