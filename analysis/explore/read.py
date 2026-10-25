exec(open("load.py").read())
import random
M=pd.read_parquet("traces_joined.parquet")
keys=set()
def get(key,uid):
    for l in gzip.open(ROOT/f"results/main/calls/gemma4-31b-think/{uid}.jsonl.gz","rt"):
        e=json.loads(l)
        if e["key"]==key: return e
sel=sys.argv[1]
q={"contra_right":(M.cp==5)&(M.contra>=2)&M.correct&~M.cut,
   "contra_wrong":(M.cp==5)&(M.contra>=2)&~M.correct,
   "pricey_says_ok_wrong":(M.cp>=3)&(M.t_price>0)&M.expensive_ok&~M.correct,
   "contradict":(M.cp>=3)&M.contradict,
   "conv_wrong":(M.cp==5)&(M.contra==0)&~M.correct&~M.cut}[sel]
S=M[q].sample(int(sys.argv[2]),random_state=int(sys.argv[3]))
for _,r in S.iterrows():
    e=get(r.key,r.uid); th=e["thinking"]
    print(f"\n===== user {r.uid} θ={D.users[r.uid].theta.tolist()} cp={r.cp} menu={r.menu_index} correct={r.correct} forced={r.forced} chars={len(th)}")
    print("MENU:", e["request"]["last_user"][-330:].replace("\n"," | "))
    print("...TAIL OF THINKING:\n", th[-1400:])
