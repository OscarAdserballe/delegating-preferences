exec(open("load.py").read())
th=pd.read_parquet("users.parquet").set_index("uid")
grp=np.array([min(th.loc[u,"contra"],2) for u in users])
target=held[held.arm=="gemma4-31b-think"].assign(g=lambda d:d.user_id.map(dict(zip(users,grp)))).groupby(["g","checkpoint"]).correct.mean().unstack().to_numpy()
convn=((T[:,1]>0).astype(int)+(T[:,2]>0)+(T[:,3]>0))
cons=[[ (T@D.users[u].rounds[r].phi.T)[:,D.users[u].choices[r]]>= (T@D.users[u].rounds[r].phi.T).max(1)-1e-9 for r in range(5)] for u in users]
def sim(s,eps,temp):
    prior=np.exp(-s*convn); acc=np.zeros((len(users),6))
    for k,u in enumerate(users):
        lp=np.log(prior)
        for cp in range(6):
            w=np.exp(lp-lp.max()); w/=w.sum(); mt=w@T; ut=H@mt
            # softmax choice with temperature -> expected accuracy
            z=ut/temp; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
            acc[k,cp]=p[np.arange(100),D.users[u].heldout_choices].mean()
            if cp<5: lp=lp+np.where(cons[k][cp],0,np.log(eps))
    return np.stack([acc[grp==g].mean(0) for g in range(3)])
best=None
for s in [0,0.5,1,1.5,2,3]:
    for eps in [1e-9,0.05,0.1,0.2,0.35,0.5,0.7]:
        for temp in [0.02,0.05,0.1,0.2]:
            m=sim(s,eps,temp); err=np.sqrt(((m-target)**2).mean())
            if best is None or err<best[0]: best=(err,s,eps,temp,m)
err,s,eps,temp,m=best
print(f"best fit: conventional-prior strength {s}, likelihood for inconsistent θ {eps} (1e-9 = exact Bayes), choice temperature {temp}; RMSE {err*100:.1f} pts")
print("target (thinking Gemma) by group 0/1/2 × cp:\n",(target*100).round(0))
print("model:\n",(m*100).round(0))
m2=sim(s,1e-9,temp); print("same prior & noise but exact updating (eps→0):\n",(m2*100).round(0))
rows=[]
for s in [0,1,2,3,4]:
    for eps in [1e-9,0.1,0.3,0.5]:
        for temp in [0.01,0.05,0.1,0.2,0.3]:
            m=sim(s,eps,temp); rows.append(dict(s=s,eps=eps,temp=temp,rmse=np.sqrt(((m-target)**2).mean())*100))
R=pd.DataFrame(rows)
print("\nbest RMSE (pts) for each (prior strength, eps):"); print(R.groupby(["s","eps"]).rmse.min().unstack().round(1))
print("\nbest RMSE for each temperature:", R.groupby("temp").rmse.min().round(1).to_dict())
