exec(open("load.py").read())
from flight_prefs.user_sampling import stratum
lastpos={(u,cp):D.users[u].choices[cp-1] for u in users for cp in range(1,6)}
h=held[held.checkpoint>=1].copy()
h["echo"]=[p==lastpos[(u,c)] for p,u,c in zip(h.pick,h.user_id,h.checkpoint)]
dlg=df[df.kind=="dialogue"][["arm","user_id","checkpoint","pick","label"]].copy()
dlg["prev_incorrect"]=dlg.pick!=dlg.label; dlg["checkpoint"]+=1
h=h.merge(dlg[["arm","user_id","checkpoint","prev_incorrect"]],on=["arm","user_id","checkpoint"])
x=h[h.prev_incorrect&h.arm.isin(LLM)]
print("after an 'incorrect': accuracy of picks that echo the named position vs not (%), and the excess echo over the 33% base rate:")
t=x.groupby(["arm","echo"]).correct.mean().unstack()*100
t["echo_rate"]=x.groupby("arm").echo.mean()*100
print(t.round(1))
# counterfactual: if excess echo picks (beyond base rate) had been at the non-echo accuracy
st={u:stratum(D.users[u].theta) for u in users}
held["stratum"]=held.user_id.map(st)
print("\ncp5 accuracy by ||θ|| stratum (0 = weakest preferences, 4 = strongest):")
print((held[held.checkpoint==5].groupby(["stratum","arm"]).correct.mean().unstack()[LLM+["bayes"]]*100).round(1).join(held[(held.checkpoint==5)&(held.arm=="bayes")].groupby("stratum").user_id.nunique().rename("users")))
