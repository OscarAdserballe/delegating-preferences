exec(open("load.py").read())
I=pd.read_parquet("implied.parquet")
tru={u:D.users[u].theta for u in users}
rows=[]
for _,r in I[I.arm.isin(LLM+["bayes"])].iterrows():
    t=tru[r.uid]
    for j,f in enumerate(FEAT):
        if t[j]==0: continue
        rows.append(dict(arm=r.arm,cp=r.cp,feat=f,direction="conventional" if (t[j]<0 and f!="departure") else ("contrarian" if f!="departure" else ("early" if t[j]<0 else "late")),
                         right=np.sign(r[f"th_{f}"])==np.sign(t[j])))
A=pd.DataFrame(rows)
for f in FEAT:
    print(f"\n{f}: share of users whose implied preference has the right direction (%)")
    print((A[A.feat==f].pivot_table(index=["direction","arm"],columns="cp",values="right")*100).round(0))
A.to_parquet("attrcurves.parquet")
