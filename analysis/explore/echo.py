exec(open("load.py").read())
lastpos={(u,cp):D.users[u].choices[cp-1] for u in users for cp in range(1,6)}
h=held[held.checkpoint>=1].copy()
h["last_pref_pos"]=[lastpos[(u,c)] for u,c in zip(h.user_id,h.checkpoint)]
h["echo"]=h.pick==h.last_pref_pos
h["label_is_echo"]=h.label==h.last_pref_pos
print("P(held-out pick is in the same POSITION as the flight the user preferred last round) (%):")
print((h.groupby(["arm","checkpoint"]).echo.mean().unstack()*100).round(1).loc[LLM+["bayes","random","copy_last"]])
print("\nbaseline: P(user's own held-out choice is in that position):", round(h[h.arm=="bayes"].label_is_echo.mean()*100,1))
# does echo depend on whether last feedback was 'incorrect' (user stated "I prefer Flight N")
dlg=df[df.kind=="dialogue"][["arm","user_id","checkpoint","pick","label"]]
dlg["prev_incorrect"]=dlg.pick!=dlg.label; dlg["checkpoint"]+=1
h=h.merge(dlg[["arm","user_id","checkpoint","prev_incorrect"]],on=["arm","user_id","checkpoint"],how="left")
print("\necho rate by whether the last feedback was 'incorrect. I prefer Flight N' (%):")
print((h[h.arm.isin(LLM+["bayes"])].groupby(["arm","prev_incorrect"]).echo.mean().unstack()*100).round(1))
print("\nposition shares of picks by checkpoint (Flight 1/2/3 %):")
for a in LLM: print(a, (held[held.arm==a].groupby("checkpoint").pick.value_counts(normalize=True).unstack()*100).round(0).to_dict("index"))
