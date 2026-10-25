exec(open("load.py").read())
th=pd.read_parquet("users.parquet").set_index("uid")
c0=held[held.checkpoint==0].groupby(["arm","menu_index"]).pick.first()
h=held[(held.checkpoint==5)&held.arm.isin(LLM+["bayes","random"])].copy()
h["default"]=[c0.get((a,i)) for a,i in zip(h.arm,h.menu_index)]
# use Gemma-think's default as a common reference for Bayes/random too
g0=c0.loc["gemma4-31b-think"]
h["gemma_default"]=h.menu_index.map(g0)
h["contra"]=h.user_id.map(th.contra).clip(upper=2)
w=h[(~h.correct)&(h.label!=h.gemma_default)]  # only errors where the default is itself wrong
print("cp5 errors on menus where the default pick is wrong for this user: share of errors that ARE Gemma-thinking's round-0 default (%)")
print("(chance among the two wrong options = 50%)")
print((w.groupby(["arm","contra"]).apply(lambda g:(g.pick==g.gemma_default).mean())*100).unstack().round(1))
print("n errors:", w.groupby("arm").size().to_dict())
