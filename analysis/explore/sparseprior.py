exec(open("load.py").read())
nz=(T!=0).sum(1)
U=np.einsum("tf,mof->tmo",T,H)
cons_cache={}
def run(lam,prior_def=None):
    prior=np.exp(-lam*nz); 
    if prior_def is not None: prior=prior*prior_def
    prior/=prior.sum()
    acc=np.zeros((len(users),6))
    for k,u in enumerate(users):
        us=D.users[u]; post=prior.copy()
        for cp in range(6):
            mt=(post/post.sum())@T; picks=np.argmax(H@mt,1)
            acc[k,cp]=np.mean(picks==np.array(us.heldout_choices))
            if cp<5:
                uu=T@us.rounds[cp].phi.T; post=post*(uu[:,us.choices[cp]]>=uu.max(1)-1e-9)
    return acc
for lam in [0,0.5,1,2]:
    a=run(lam)
    print(f"Bayes with sparsity prior λ={lam}: accuracy by cp", (a.mean(0)*100).round(1), " | by #attrs cp5:", {n:round(a[np.array([(D.users[u].theta!=0).sum()==n for u in users]),5].mean()*100,1) for n in [1,2,3,4]})
# 'conventional default' prior: favour θ with negative duration/stops/price
conv=np.exp(-0.7*((T[:,1]>0).astype(int)+(T[:,2]>0)+(T[:,3]>0)))
a=run(0,conv); print("Bayes with a 'conventional' prior (penalise liking long/stops/expensive):", (a.mean(0)*100).round(1))
th=pd.read_parquet("users.parquet").set_index("uid")
grp=np.array([th.loc[u,"contra"] for u in users])
for s in [1.5,3,6]:
    conv=np.exp(-s*((T[:,1]>0).astype(int)+(T[:,2]>0)+(T[:,3]>0)))
    a=run(0,conv)
    print(f"conventional prior strength {s}: overall", (a.mean(0)*100).round(1), "| cp0/cp5 for 0-contra users", (a[grp==0][:,[0,5]].mean(0)*100).round(1), "| 2-3 contra", (a[grp>=2][:,[0,5]].mean(0)*100).round(1))
