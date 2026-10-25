exec(open("load.py").read())
W=pd.read_parquet("welfare.parquet")
w5=W[W.checkpoint==5].pivot(index="user_id",columns="arm",values="welfare")*100
print("distribution of per-user welfare at cp5 (% of achievable):")
print(w5[LLM+["bayes","copy_last"]].describe(percentiles=[.1,.25,.5,.9]).round(1).T[["min","10%","25%","50%","90%"]])
for thr in [60,70]:
    print(f"share of users served below {thr}%:", {a:round((w5[a]<thr).mean()*100) for a in LLM+["bayes"]})
th=pd.read_parquet("users.parquet").set_index("uid")
worst=w5["gemma4-31b-think"].nsmallest(8).index
print("\n8 worst-served users by thinking Gemma: contrarian counts", th.loc[worst,"contra"].tolist(), "vs population mean", round(th.contra.mean(),2))
