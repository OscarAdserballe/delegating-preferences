exec(open("load.py").read())
from dataclasses import replace
from flight_prefs.answer_parsing import parse_pick
from flight_prefs.call_log import call_key
from flight_prefs.experiment import RunConfig
cfg=RunConfig.from_toml(ROOT/"configs/main.toml"); spec=[a for a in cfg.arms if a.arm=="gemma4-31b-think"][0]
fspec=replace(spec,thinking=False,thinking_budget=None,max_tokens=16,force_answer=False)
E={}
for p in glob.glob(str(ROOT/"results/main/calls/gemma4-31b-think/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); E[e["key"]]=e
dl=df[(df.arm=="gemma4-31b-think")&(df.kind=="dialogue")].sort_values("checkpoint")
dt=df[(df.arm=="gemma4-31b-think")&(df.kind=="heldout")].set_index(["user_id","checkpoint","menu_index"])
rows=[]
for u in users:
    us=D.users[u]; s=DialogueState(); rr=dl[dl.user_id==u].set_index("checkpoint")
    for cp in range(6):
        for i,m in enumerate(D.heldout):
            msgs=prompts.build_messages(s,m); e=E.get(call_key(spec,msgs))
            if e is None or e["usage"]["finish_reason"]!="length": continue
            th=(e.get("thinking") or "")
            f=E.get(call_key(fspec,prompts.with_forced_answer(msgs,th or e["text"])))
            ms=re.findall(r"Flight\s*\**\s*([123])",th[-500:])
            rows.append(dict(uid=u,cp=cp,menu=i,lean=int(ms[-1])-1 if ms else None,forced=parse_pick(f["text"]) if f else None,found=f is not None,label=us.heldout_choices[i]))
        if cp<5:
            r=rr.loc[cp]; s=s.after(PastRound(us.rounds[cp],int(r.pick),us.choices[cp]))
R=pd.DataFrame(rows); R.to_parquet("forced_pairs.parquet")
print(len(R),"cut-off held-out calls; follow-up found:",R.found.mean().round(3))
r=R[R.lean.notna()&R.forced.notna()]
print("forced pick == last flight named in the final 500 chars of the cut-off thinking: %.1f%% (n=%d)"%((r.lean==r.forced).mean()*100,len(r)))
print("accuracy: forced pick %.1f%%, 'lean' %.1f%%"%((r.forced==r.label).mean()*100,(r.lean==r.label).mean()*100))
