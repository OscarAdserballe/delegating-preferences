exec(open("load.py").read())
acc=held.groupby(["user_id","arm","checkpoint"]).correct.mean().unstack(["arm","checkpoint"])*100
th=pd.read_parquet("users.parquet").set_index("uid"); u2=pd.read_parquet("users2.parquet")
T5=pd.DataFrame({"think0":acc[("gemma4-31b-think",0)],"think5":acc[("gemma4-31b-think",5)],"nothink5":acc[("gemma4-31b",5)],"qwen5":acc[("qwen25-7b",5)],"bayes5":acc[("bayes",5)]})
T5=T5.join(th[["contra"]]).join(u2[["lex_fit","n_nonzero"]])
T5["theta"]=[str(D.users[u].theta.tolist()) for u in T5.index]
dl=df[(df.arm=="gemma4-31b-think")&(df.kind=="dialogue")].sort_values("checkpoint")
T5["live"]=dl.groupby("user_id").apply(lambda g:"".join("✓" if p==l else "✗" for p,l in zip(g.pick,g.label)))
print(T5.sort_values("think5").round(0).to_string())
T5.to_parquet("usertable.parquet")
