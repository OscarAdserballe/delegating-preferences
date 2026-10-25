exec(open("load.py").read())
c0=held[held.checkpoint==0].groupby(["arm","menu_index"]).pick.first()
h=held[(held.checkpoint==5)&held.arm.isin(LLM+["bayes"])].copy()
h["default_right"]=[c0.loc[("gemma4-31b-think" if a=="bayes" else a,i)]==l for a,i,l in zip(h.arm,h.menu_index,h.label)]
h["kept_default"]=[c0.loc[("gemma4-31b-think" if a=="bayes" else a,i)]==p for a,i,p in zip(h.arm,h.menu_index,h.pick)]
t=h.groupby(["arm","default_right"]).correct.mean().unstack()*100
t["share of menus where own default is right"]=h.groupby("arm").default_right.mean()*100
t["share of cp5 correct picks that are just the kept default"]=h[h.correct].groupby("arm").kept_default.mean()*100
print("cp5 accuracy split by whether the arm's own round-0 pick was right for this user (Bayes uses thinking-Gemma's default as reference):")
print(t.round(1))
dl=df[df.kind=="dialogue"].groupby(["arm","checkpoint"]).correct.mean().unstack()*100
ho=held[held.checkpoint<5].groupby(["arm","checkpoint"]).correct.mean().unstack()*100
print("\nlive dialogue accuracy minus held-out accuracy at the same checkpoint (pts):"); print((dl-ho).loc[LLM+["bayes"]].round(1))
