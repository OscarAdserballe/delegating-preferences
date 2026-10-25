exec(open("load.py").read())
rows=[]
for p in glob.glob(str(ROOT/"results/main/calls/qwen25-7b/*.gz")):
    for l in gzip.open(p,"rt"):
        e=json.loads(l); lu=e["request"]["last_user"]
        if lu.endswith(prompts.FORMAT_REMINDER): continue
        m=re.match(r"Your option Flight (\d) is (in)?correct\.(?: I prefer Flight (\d)\.)?",lu)
        stated=int(m.group(3)) if m and m.group(3) else (int(m.group(1)) if m else None)
        t=e["text"]
        g=re.match(r"(Given|Based on) your preference for Flight (\d)",t)
        rows.append(dict(cp=(e["request"]["n_messages"]-1)//2, incorrect=bool(m and m.group(2)), stated=stated,
             opener=t[:45], given_pref=bool(g), given_n=int(g.group(2)) if g else None))
Q=pd.DataFrame(rows)
q=Q[Q.cp>=1]
print("Qwen first-attempt replies, cp>=1:",len(q))
print("share opening with 'Given/Based on your preference for Flight N':", round(q.given_pref.mean()*100,1),"% ; after 'incorrect':",round(q[q.incorrect].given_pref.mean()*100,1),"% ; after 'correct':",round(q[~q.incorrect].given_pref.mean()*100,1),"%")
gp=q[q.given_pref]
print("…and N equals the flight the user just named:", round((gp.given_n==gp.stated).mean()*100,1),"%")
print("\nmost common openers:"); print(q.opener.str.slice(0,40).value_counts().head(8).to_string())
