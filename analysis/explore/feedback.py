exec(open("load.py").read())
dlg=df[df.kind=="dialogue"][["arm","user_id","checkpoint","pick","label"]].copy()
dlg["fb"]=np.where(dlg.pick==dlg.label,"correct","incorrect")
acc=held.groupby(["arm","user_id","checkpoint"]).correct.mean().rename("acc").reset_index()
nxt=acc.copy(); nxt["checkpoint"]-=1; nxt=nxt.rename(columns={"acc":"acc_next"})
m=dlg.merge(acc,on=["arm","user_id","checkpoint"]).merge(nxt,on=["arm","user_id","checkpoint"])
m["delta"]=(m.acc_next-m.acc)*100
print("Change in held-out accuracy from checkpoint r to r+1 after the round-r feedback (pts), mean [n]:")
print(m[m.arm.isin(LLM+["bayes","copy_last"])].groupby(["arm","fb"]).delta.agg(["mean","count"]).round(1).unstack())
print("\nDialogue-round accuracy (share of the 5 live rounds answered correctly):")
print((dlg.assign(c=dlg.fb=="correct").groupby("arm").c.mean()*100).round(1))
# flip rate: pick on same menu changes between consecutive checkpoints
h=held.sort_values(["arm","user_id","menu_index","checkpoint"])
h["prev"]=h.groupby(["arm","user_id","menu_index"]).pick.shift()
h["flip"]=(h.pick!=h.prev)&h.prev.notna()
h["flip_to_right"]=h.flip&h.correct; h["flip_to_wrong"]=h.flip&~h.correct
f=h[h.checkpoint>0].groupby(["arm","checkpoint"])[["flip","flip_to_right","flip_to_wrong"]].mean()*100
print("\nShare of held-out picks that change from previous checkpoint (%), and whether into right or wrong:")
print(f.loc[LLM+["bayes"]].round(1).unstack(0)["flip"])
print("\nnet flips to right minus to wrong (pts):")
print((f["flip_to_right"]-f["flip_to_wrong"]).unstack(0)[LLM+["bayes"]].round(1))
