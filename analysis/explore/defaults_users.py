exec(open("load.py").read())
c0=held[held.checkpoint==0].groupby(["arm","menu_index"]).pick.first().unstack(0)
print("round-0 default agreement between models (%):")
print((pd.DataFrame({a:{b:(c0[a]==c0[b]).mean()*100 for b in LLM} for a in LLM})).round(0))
# what does the shared default look like? share of default picks that are cheapest/shortest/fewest stops
for a in LLM:
    p=c0[a].astype(int).to_numpy()
    print(a,"default pick is cheapest %.0f%%, shortest %.0f%%, fewest stops %.0f%%, Pareto-dominated %.0f%%"%(
      np.mean(np.isclose(H[np.arange(100),p,3],H[:,:,3].min(1)))*100,np.mean(np.isclose(H[np.arange(100),p,1],H[:,:,1].min(1)))*100,
      np.mean(np.isclose(H[np.arange(100),p,2],H[:,:,2].min(1)))*100,
      np.mean([any(all(H[i,o,1:]<=H[i,p[i],1:]) and any(H[i,o,1:]<H[i,p[i],1:]) for o in range(3) if o!=p[i]) for i in range(100)])*100))
acc=held[held.checkpoint==5].groupby(["user_id","arm"]).correct.mean().unstack()
u2=pd.read_parquet("users2.parquet"); th=pd.read_parquet("users.parquet").set_index("uid")
x=acc.join(u2[["dominance","lex_fit","n_nonzero"]]).join(th[["contra"]])
x["think_minus_bayes"]=(x["gemma4-31b-think"]-x.bayes)*100
print("\nusers where thinking Gemma >= Bayes at cp5:")
print(x[x.think_minus_bayes>=0][["bayes","gemma4-31b-think","gemma4-31b","lex_fit","dominance","contra"]].round(2).join(pd.DataFrame({u:D.users[u].theta for u in users}).T.rename(columns=dict(enumerate(FEAT)))))
print("\nall-users means for comparison:", x[["bayes","lex_fit","dominance","contra"]].mean().round(2).to_dict())
