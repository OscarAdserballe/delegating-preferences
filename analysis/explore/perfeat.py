exec(open("load.py").read())
I=pd.read_parquet("implied.parquet")
tru=pd.DataFrame([dict(uid=u,**{f"t_{f}":D.users[u].theta[i] for i,f in enumerate(FEAT)}) for u in users])
J=I.merge(tru,on="uid")
for f in FEAT:
    J[f"sgn_{f}"]=np.sign(J[f"t_{f}"])
    J[f"agree_{f}"]=np.sign(J[f"th_{f}"].round(6))==J[f"sgn_{f}"]
print("Share of (user) where implied θ_j has the TRUE sign, cp5, by the true sign of θ_j")
for f in FEAT:
    t=J[J.cp==5].groupby(["arm",f"sgn_{f}"])[f"agree_{f}"].mean().unstack().round(2)
    print(f"\n--- {f} (cols: true sign −1/0/+1) ---\n", t.loc[["bayes","gemma4-31b","gemma4-31b-think","qwen25-7b","copy_last"]])
print("\nusers per sign:", {f:tru[f"t_{f}"].apply(np.sign).value_counts().to_dict() for f in FEAT})
