exec(open("load.py").read())
B=BayesianAssistant(T)
acc=held.groupby(["arm","user_id","checkpoint"]).correct.mean()
rows=[]
for u in users:
    us=D.users[u]; s=DialogueState()
    for r in range(5):
        n0=(B.posterior(s)>0).sum(); s=s.after(PastRound(us.rounds[r],None,us.choices[r])); n1=(B.posterior(s)>0).sum()
        phi=us.rounds[r].phi; c=us.choices[r]
        extreme=sum(int(np.isclose(phi[c,j],phi[:,j].max()) and not np.isclose(phi[:,j].max(),phi[:,j].min()) and (phi[:,j]==phi[c,j]).sum()==1)+
                    int(np.isclose(phi[c,j],phi[:,j].min()) and not np.isclose(phi[:,j].max(),phi[:,j].min()) and (phi[:,j]==phi[c,j]).sum()==1) for j in range(4))
        conventional=all(phi[c,j]<=phi[:,j].min()+1e-9 for j in (1,2,3))   # cheapest, shortest, fewest stops at once
        row=dict(uid=u,r=r,bits=np.log2(n0/n1),extreme=extreme,conventional_choice=conventional)
        for a in LLM+["bayes"]:
            row[a]=(acc[(a,u,r+1)]-acc[(a,u,r)])*100
        rows.append(row)
S=pd.DataFrame(rows)
print("Bayes information per round (bits of the 624-θ hypothesis space eliminated):", S.groupby("r").bits.mean().round(2).to_dict())
print("\ncorrelation across (user,round) of accuracy change with Bayes bits:", S[["bits"]+LLM+["bayes"]].corr()["bits"].round(2).to_dict())
print("\naccuracy change (pts) by # attributes on which the user's choice was the unique extreme:")
print(S.groupby("extreme")[LLM+["bayes"]].agg("mean").round(1).join(S.groupby("extreme").size().rename("n")))
print("\naccuracy change when the user chose the 'conventional' flight (cheapest+shortest+fewest stops) vs not:")
print(S.groupby("conventional_choice")[LLM+["bayes"]].mean().round(1).join(S.groupby("conventional_choice").size().rename("n")))
S.to_parquet("rounds.parquet")
