exec(open("load.py").read())
acc=held.groupby(["user_id","arm","checkpoint"]).correct.mean().unstack(["arm","checkpoint"])
for a in ["gemma4-31b","gemma4-31b-think","qwen25-7b"]:
    gain=(acc[(a,5)]-acc[(a,0)])*100; gap=(acc[(a,5)]-acc[("bayes",5)])*100
    print(f"{a}: users with gain cp0→5 >10pts: {(gain>10).mean():.0%}, <0: {(gain<0).mean():.0%}; users beating Bayes at cp5: {(gap>0).sum()}/50; gain quantiles {np.percentile(gain,[10,50,90]).round(0)}")
# shared mistakes: when Bayes is wrong, does the LLM pick what Bayes picked?
w=held.pivot_table(index=["user_id","checkpoint","menu_index"],columns="arm",values="pick").dropna()
lab=held[held.arm=="bayes"].set_index(["user_id","checkpoint","menu_index"]).label
w=w.join(lab)
bw=w[(w.bayes!=w.label)&(w.index.get_level_values(1)>=3)]
print("\nwhen Bayes is wrong (cp>=3): P(arm picks same as Bayes):", {a: round((bw[a]==bw.bayes).mean()*100,1) for a in LLM})
br=w[(w.bayes==w.label)&(w.index.get_level_values(1)>=3)]
print("when Bayes is right (cp>=3): P(arm right):", {a: round((br[a]==br.label).mean()*100,1) for a in LLM})
print("when Bayes is wrong (cp>=3): P(arm right):", {a: round((bw[a]==bw.label).mean()*100,1) for a in LLM})
# think vs no-think disagreement
d=w[w["gemma4-31b"]!=w["gemma4-31b-think"]]
print("\nthink vs no-think disagree on", round(len(d)/len(w)*100,1),"% of items; by cp:", (w.assign(dis=w["gemma4-31b"]!=w["gemma4-31b-think"]).groupby(level=1).dis.mean()*100).round(1).to_dict())
print("when they disagree: think right", round((d["gemma4-31b-think"]==d.label).mean()*100,1),"%, no-think right", round((d["gemma4-31b"]==d.label).mean()*100,1),"%, think agrees with Bayes", round((d["gemma4-31b-think"]==d.bayes).mean()*100,1),"%, no-think agrees with Bayes", round((d["gemma4-31b"]==d.bayes).mean()*100,1),"%")
