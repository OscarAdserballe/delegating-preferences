exec(open("load.py").read())
E=pd.read_parquet("held_extra.parquet")
h=held.merge(E[["arm","user_id","checkpoint","menu_index","maxreg"]],on=["arm","user_id","checkpoint","menu_index"])
h=h[h.maxreg>0]
h["welfare"]=1-h.regret/h.maxreg   # 1 = best option, 0 = worst option, random ≈ 0.5
W=h.groupby(["arm","user_id","checkpoint"]).welfare.mean().reset_index()
t=W.groupby(["arm","checkpoint"]).welfare.mean().unstack()*100
print("share of achievable welfare delivered (0 = always worst option, 100 = always best), by rounds of feedback:")
print(t.loc[LLM+["bayes","copy_last","always_first","random","oracle"]].round(1))
# value of information: welfare gain per round
print("\nwelfare gain from round 0 to 5 (pts):", (t[5]-t[0]).loc[LLM+["bayes","copy_last"]].round(1).to_dict())
# How many rounds does Bayes need to reach each LLM's final welfare?
for a in LLM:
    f=t.loc[a,5]; r=next((c for c in range(6) if t.loc["bayes",c]>=f),None)
    print(f"{a}: final welfare {f:.1f} — Bayes reaches this after {r} round(s)")
W.to_parquet("welfare.parquet")
