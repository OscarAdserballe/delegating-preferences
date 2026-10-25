exec(open("load.py").read())
m=held[held.checkpoint>=3].groupby(["menu_index","arm"]).correct.mean().unstack()*100
print("menu-level accuracy correlation across menus (cp>=3):")
print(m[LLM+["bayes","copy_last"]].corr().round(2))
# menu features: spread of prices etc.
spread={i:dict(price_spread=np.ptp(H[i,:,3]),dur_spread=np.ptp(H[i,:,1]),stops_spread=np.ptp(H[i,:,2]),dep_spread=np.ptp(H[i,:,0]),
               dominated=any(all(H[i,o,1:]<=H[i,p,1:]) and any(H[i,o,1:]<H[i,p,1:]) for o in range(3) for p in range(3) if o!=p)) for i in range(100)}
S=pd.DataFrame(spread).T.astype(float); x=m.join(S)
x["gap"]=x.bayes-x["gemma4-31b-think"]
print("\ncorrelation of the Bayes − thinking gap per menu with menu properties:")
print(x[["gap","price_spread","dur_spread","stops_spread","dep_spread","dominated"]].corr()["gap"].round(2).to_dict())
f=held[(held.arm=="gemma4-31b-think")&(held.checkpoint>=1)]
print("\nposition of pick, forced vs unforced (thinking, cp>=1, %):")
print((f.groupby("forced").pick.value_counts(normalize=True).unstack()*100).round(1))
print("label positions:", (f.label.value_counts(normalize=True).sort_index()*100).round(1).to_dict())
