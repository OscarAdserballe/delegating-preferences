exec(open("load.py").read())
kind={}
for u in users:
    t=D.users[u].theta
    for i in range(100):
        b=D.users[u].heldout_choices[i]
        wins=0
        for j in range(4):
            if t[j]==0: continue
            v=np.sign(t[j])*H[i,:,j]
            if np.isclose(v[b],v.max()) and (np.isclose(v,v.max()).sum()==1): wins+=1
        kind[(u,i)]=wins
h=held.copy(); h["wins"]=[kind[(u,i)] for u,i in zip(h.user_id,h.menu_index)]
h["opt"]=np.where(h.wins==0,"compromise (best on no attribute)",np.where(h.wins==1,"best on 1 attribute","best on 2+ attributes"))
print("share of menus:", (h[(h.arm=="bayes")&(h.checkpoint==5)].opt.value_counts(normalize=True)*100).round(1).to_dict())
print("\ncp5 accuracy by the kind of option the user prefers:")
print((h[h.checkpoint==5].groupby(["opt","arm"]).correct.mean().unstack()[LLM+["bayes","copy_last","random"]]*100).round(1))
# n nonzero attributes
nz={u:(D.users[u].theta!=0).sum() for u in users}
h["nz"]=h.user_id.map(nz)
print("\ncp5 accuracy by number of attributes the user cares about:")
print((h[h.checkpoint==5].groupby(["nz","arm"]).correct.mean().unstack()[LLM+["bayes"]]*100).round(1).join(h[(h.checkpoint==5)&(h.arm=="bayes")].groupby("nz").user_id.nunique().rename("users")))
