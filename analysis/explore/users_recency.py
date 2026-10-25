exec(open("load.py").read())
acc=held[held.checkpoint==5].groupby(["user_id","arm"]).correct.mean().unstack()
print("cross-user correlation of cp5 accuracy:")
print(acc[["bayes","gemma4-31b","gemma4-31b-think","qwen25-7b","copy_last"]].corr().round(2))
# Bayes posterior size after 5 rounds
B=BayesianAssistant(T)
post_n={}
for u in users:
    s=DialogueState()
    for r in range(5): s=s.after(PastRound(D.users[u].rounds[r],None,D.users[u].choices[r]))
    post_n[u]=(B.posterior(s)>0).sum()
acc["bayes_consistent_thetas"]=pd.Series(post_n)
print("\ncorr with # θ still consistent after 5 rounds (Bayes uncertainty):",acc.corr()["bayes_consistent_thetas"].round(2).to_dict())
# recency: for cp>=2, share of held-out picks equal to nearest-neighbour of the user's choice in round k
rec=[]
for (arm,u,cp),g in held[(held.checkpoint>=2)&held.arm.isin(LLM+["bayes"])].groupby(["arm","user_id","checkpoint"]):
    us=D.users[u]; g=g.sort_values("menu_index"); p=g.pick.to_numpy()
    for k in range(cp):
        c=us.rounds[k].phi[us.choices[k]]
        nn=np.argmin(np.linalg.norm(H-c,axis=2),axis=1)
        rec.append(dict(arm=arm,cp=cp,lag=cp-1-k,match=np.mean(nn==p)))
R=pd.DataFrame(rec)
print("\nshare of held-out picks = nearest neighbour of the user's choice k rounds ago (lag 0 = most recent):")
print((R.groupby(["arm","lag"]).match.mean().unstack()*100).round(1))
