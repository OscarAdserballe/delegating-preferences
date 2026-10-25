exec(open("load.py").read())
dl=df[df.kind=="dialogue"]
first=dl[dl.checkpoint==0].set_index(["arm","user_id"]).apply(lambda r:r.pick==r.label,axis=1).rename("first_right")
acc=held.groupby(["arm","user_id","checkpoint"]).correct.mean().unstack()
x=acc.join(first,on=["arm","user_id"])
th=pd.read_parquet("users.parquet").set_index("uid")
x=x.reset_index().merge(th[["contra"]],left_on="user_id",right_index=True)
print("accuracy at cp1 and cp5 by whether the arm's own round-1 live pick was right (%), with # users:")
t=x[x.arm.isin(LLM+["bayes"])].groupby(["arm","first_right"])[[1,5]].mean()*100
t["n"]=x[x.arm.isin(LLM+["bayes"])].groupby(["arm","first_right"]).size()
print(t.round(1))
print("\n…restricted to conventional users (0 contrarian attributes):")
y=x[(x.contra==0)&x.arm.isin(LLM+["bayes"])]
t=y.groupby(["arm","first_right"])[[0,1,5]].mean()*100; t["n"]=y.groupby(["arm","first_right"]).size(); print(t.round(1))
