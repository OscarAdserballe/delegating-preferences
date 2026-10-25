exec(open("load.py").read())
h=held.sort_values(["arm","user_id","menu_index","checkpoint"]).copy()
h["prev_c"]=h.groupby(["arm","user_id","menu_index"]).correct.shift()
h["w2r"]=(h.prev_c==False)&h.correct; h["r2w"]=(h.prev_c==True)&~h.correct
print("wrong→right minus right→wrong per checkpoint (pts):")
print(((h[h.checkpoint>0].groupby(["checkpoint","arm"])[["w2r","r2w"]].mean().eval("w2r-r2w"))*100).unstack()[LLM+["bayes"]].round(1))
print("\nright→wrong rate (unlearning) per checkpoint (%):")
print((h[h.checkpoint>0].groupby(["checkpoint","arm"]).r2w.mean()*100).unstack()[LLM+["bayes"]].round(1))
# stakes: utility margin best - second best for the user
def margin(u,i):
    v=np.sort(H[i]@D.users[u].theta)[::-1]; return v[0]-v[1]
mm={(u,i):margin(u,i) for u in users for i in range(100)}
held["margin"]=[mm[(u,i)] for u,i in zip(held.user_id,held.menu_index)]
held["mbin"]=pd.qcut(held.margin.rank(method="first"),4,labels=["q1 near-tie","q2","q3","q4 clear"])
print("\ncp5 accuracy by stakes (utility gap between best and 2nd-best option) quartile:")
print((held[held.checkpoint==5].groupby(["mbin","arm"]).correct.mean().unstack()[LLM+["bayes","random"]]*100).round(1))
print("\nmean regret of WRONG picks (how bad mistakes are), cp5, and as share of the max possible regret:")
held["maxreg"]=[ (H[i]@D.users[u].theta).max()-(H[i]@D.users[u].theta).min() for u,i in zip(held.user_id,held.menu_index)]
w=held[(held.checkpoint==5)&(~held.correct)&(held.maxreg>0)]
print(w.groupby("arm").apply(lambda g: pd.Series({"regret_when_wrong":g.regret.mean(),"share_of_max":(g.regret/g.maxreg).mean(),"worst_option_share":(np.isclose(g.regret,g.maxreg)).mean()})).loc[LLM+["bayes","random","copy_last"]].round(3))
held[["arm","user_id","checkpoint","menu_index","margin","maxreg"]].to_parquet("held_extra.parquet")
