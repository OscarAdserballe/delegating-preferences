exec(open("load.py").read())
dl=df[(df.arm=="gemma4-31b-think")&(df.kind=="dialogue")]
first=dl[(dl.checkpoint==0)&(dl.pick!=dl.label)].user_id.tolist()[:20]
E={}
import random
random.seed(4)
for u in random.sample(first,3):
    us=D.users[u]
    for l in gzip.open(ROOT/f"results/main/calls/gemma4-31b-think/{u}.jsonl.gz","rt"):
        e=json.loads(l)
        if e["request"]["n_messages"]==3 and e["request"]["last_user"].endswith(prompts.ask(us.rounds[1])):
            th=e.get("thinking") or ""
            print(f"\n===== user {u} θ={us.theta.tolist()} — dialogue round 2, after round-1 'incorrect'; chars={len(th)} finish={e['usage']['finish_reason']}")
            print(e["request"]["last_user"][:130].replace("\n"," | "))
            # print the part of the thinking that reasons about round 1
            print(th[:1700])
            break
