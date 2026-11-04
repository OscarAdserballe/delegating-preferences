import base64, json
from pathlib import Path
D = Path(__file__).resolve().parent
F = D.parent / "delegation"
def img(p): return "data:image/png;base64," + base64.b64encode((F / p).read_bytes()).decode()
S = json.loads((F / "summary.json").read_text())

P = []  # (id, kind, html) kind: "yours" = trimmed from your draft, "new" = written by Claude
def p(i, kind, html): P.append((i, kind, html))

BODY = f'''
<header data-strata-id="title">
  <div class="eyebrow">Draft · v1 · for annotation</div>
  <h1>Delegating preferences</h1>
  <p class="sub">Replicating Qiu et al. (2026) with newer LLMs, and measuring the welfare cost of delegating choices to AI agents.</p>
  <div class="key"><span class="tag yours">yours</span> trimmed from your draft <span class="tag new">new</span> written by Claude <span class="count" id="wc"></span></div>
</header>

<section data-strata-id="s1">
<h2>Why this matters</h2>
<p class="yours" data-strata-id="s1p1">The biggest delegation problem we're going to face is gradually handing more and more decision-making power to AI agents. Partly for convenience: agents are yet another way of removing the small, thoughtless frictions that slow us down. Partly to extend our reach: until now, the only way to cope with capacity was an extra pair of hands, and agents change that drastically.</p>
<p class="yours" data-strata-id="s1p2">There's something promising, and frankly a little scary, about extending one's exact preferences across a swarm of AI agents. Even the strictest, most authoritarian people have so far had to align many people's interests and tell a captivating story. With agents, that constraint starts to disappear.</p>
<p class="yours" data-strata-id="s1p3">Which makes one question central: how efficiently do these agents translate our preferences? <span class="newin">And where we haven't told them, whose preferences fill the gaps?</span></p>
</section>

<section data-strata-id="s2">
<h2>The experiment</h2>
<p class="yours" data-strata-id="s2p1">Qiu et al. (2026) set out to show how Bayesian learning can be used as a training objective, a novel form of teaching. Dually, the same setup can be read through an economics angle: how efficiently can we translate our preferences into LLMs?</p>
<p class="yours" data-strata-id="s2p2">The experiment is quite elegant. It simulates a user whose utility function we know, defined over four features of a flight: departure time, duration, number of stops and price. The model is shown a menu of three flights and picks one; the user says whether it was right, and which flight they'd have picked. This runs for five rounds. <span class="newin">After each round, the conversation is frozen and the model is tested on 100 new menus without feedback, so we measure what it has learned rather than what it was told.</span></p>
<p class="yours" data-strata-id="s2p3">The benchmark is a Bayesian learner. It's a little bit of cheating: it knows the functional form of the user's utility, and the exact set of possible users, which the LLM doesn't. It is the information optimum: how much can be learned from these examples at all.</p>

<div class="walk" data-strata-id="walk">
  <div class="wk-head"><span class="tag new">new</span><b>One user, round by round.</b> User 343 doesn't care when they fly, mildly prefers longer flights with more stops, and strongly prefers pricier ones.</div>
  <div class="rounds">
    <div class="rd" data-strata-id="walk-r1">
      <div class="rd-n">Round 1</div>
      <div class="menu-wrap"><table class="menu"><tr><td>Flight 1</td><td>5:12 PM</td><td>16 h 6 m</td><td>2 stops</td><td>$190</td></tr>
      <tr class="agent"><td>Flight 2</td><td>3:36 PM</td><td>12 h 12 m</td><td>2 stops</td><td>$370</td></tr>
      <tr class="user"><td>Flight 3</td><td>10:48 AM</td><td>16 h 6 m</td><td>2 stops</td><td>$640</td></tr></table></div>
      <div class="think">“Since I have no previous history with this user, I have to make a ‘best judgment’ guess about what a typical traveler prefers.”</div>
      <div class="fb">Gemma picks Flight 2 → <i>“Your option Flight 2 is incorrect. I prefer Flight 3.”</i></div>
    </div>
    <div class="rd" data-strata-id="walk-r2">
      <div class="rd-n">Round 2</div>
      <div class="think">“Hypothesis: The user prefers morning departures, regardless of price or duration.”</div>
      <div class="fb">Wrong again. The user wanted the $730 flight.</div>
    </div>
    <div class="rd" data-strata-id="walk-r35">
      <div class="rd-n">Rounds 3–5</div>
      <div class="think">“The user prefers the most expensive flight.”</div>
      <div class="fb">Right three times in a row.</div>
    </div>
  </div>
  <div class="wk-foot">
    <img alt="User 343: held-out accuracy by round for each agent" src="{img('figures/4_user343.png')}">
    <p>After each round, all agents are scored on the same 100 held-out menus. Gemma (thinking) goes from 6% to 77%; the Bayesian learner from 27% to 90%; Qwen never gets above chance. Note that round 0 is not random for the LLMs: they start from a default traveller, cheap, short and direct, which is exactly wrong for this user.</p>
  </div>
</div>
</section>

<section data-strata-id="s3">
<h2>Replication and newer models</h2>
<p class="yours" data-strata-id="s3p1">The paper's basic finding was that, when they tested this, LLMs were poor at learning from past interactions to infer something about future choices. <span class="newin">We replicate it with the paper's own data and model (Qwen 2.5 7B), then run Gemma 4 31B with and without thinking: 50 users, 605 decisions per user and model, temperature 0.</span></p>
<div class="tables">
<table class="bt" data-strata-id="tab-design"><caption>Design</caption>
<tr><th>Users</th><td>50 simulated, stratified by preference strength</td></tr>
<tr><th>Rounds</th><td>5, with feedback after each pick</td></tr>
<tr><th>Test</th><td>100 held-out menus after each round, no feedback</td></tr>
<tr><th>Agents</th><td>Qwen 2.5 7B · Gemma 4 31B · Gemma 4 31B thinking · Bayesian learner</td></tr>
<tr><th>Outcomes</th><td>Accuracy; share of achievable welfare</td></tr></table>
<table class="bt num" data-strata-id="tab-repl"><caption>Final-round accuracy (%)</caption>
<tr><th></th><th>Paper</th><th>Ours</th></tr>
<tr><td>Bayesian learner</td><td>≈80</td><td>80.6</td></tr>
<tr><td>Qwen 2.5 7B</td><td>37</td><td>38.0</td></tr>
<tr><td>Gemma 2 9B → Gemma 4 31B</td><td>37</td><td>59.4</td></tr>
<tr><td>Gemma 4 31B, thinking</td><td>–</td><td>62.7</td></tr></table>
</div>
<figure data-strata-id="fig1"><img alt="Held-out accuracy by round" src="{img('figures/1_replication.png')}"></figure>
<p class="new" data-strata-id="s3p2">The replication holds: Qwen lands at 38% against the paper's 37%, flat after round 1. Gemma 4, an open model rather than the frontier, reaches 59%, and 63% with thinking. Measured above chance (33%), the learning signal went from about 4 points to 26–30. Thinking adds a little (+3.4 points, 95% CI 1.8–5.0) without changing how fast the model learns. The Bayesian learner still reaches 81%.</p>
</section>

<section data-strata-id="s4">
<h2>The economics</h2>
<p class="yours" data-strata-id="s4p1">From an economic angle this is very close to a standard setup. A consumer has a utility function and has delegated their choices to a personal assistant, which can only learn from past examples. The relevant metric is how good this assistant is at understanding my preferences, expressed as welfare: the share of achievable utility it delivers on each menu.</p>
<figure class="eq" data-strata-id="eq"><img alt="Welfare definitions" src="{img('latex/welfare_formulas.png')}"></figure>
<p class="yours" data-strata-id="s4p2">Each example I provide is a cost, and the welfare the agent delivers is the benefit. With a linear cost <i>c</i> per example, net welfare is W(n) − c·n, and we get the classic trade-off: there's a benefit to specifying, but also a cost. <span class="newin">Gemma's five rounds are worth about 1.6 rounds of the Bayesian learner (95% CI 1.2–1.9).</span></p>
<figure data-strata-id="fig3"><img alt="Net welfare by number of examples" src="{img('figures/3_net_welfare.png')}"></figure>
</section>

<section data-strata-id="s5">
<h2>A principal–agent reading</h2>
<p class="yours" data-strata-id="s5p1">When a principal delegates a task, two things are needed for good delegation. <b>Interest transmission:</b> to what extent do the principal's interests transmit to the agent? <b>Capacity and capability:</b> can the agent actually do the task, as an apprentice can't yet do what the master craftsman does?</p>
<p class="yours" data-strata-id="s5p2">Unlike the biased agent in Alonso &amp; Matouschek (2008), there's no conflict of interest here, no moral hazard. <span class="newin">But calling these agents completely faithful delegates goes too far. There's a third component: the <b>default rule</b>, what the agent does where the principal hasn't said anything. With no information, every model serves the same default traveller. Users who don't care about price get a preference for cheap flights filled in anyway. The agent isn't strategically biased; it's biased by its prior.</span></p>
</section>

<section data-strata-id="s6">
<h2>Unusual principals</h2>
<p class="yours" data-strata-id="s6p1">Some preferences are unusual by definition: preferring longer flights, pricier flights, more stops. <span class="newin">We call a principal unusual if they prefer the odd direction on at least two of the three (18 of 50 users), and conventional if on none (14).</span></p>
<figure data-strata-id="fig2"><img alt="Welfare for conventional vs unusual principals" src="{img('figures/2_unusual_vs_conventional.png')}"></figure>
<p class="new" data-strata-id="s6p2">With no information, Gemma already serves conventional principals better than the Bayesian learner does after one round (77% vs 72% of achievable welfare), and then barely improves. Unusual principals start below random (33%) and have to teach the agent; they end at 72%, against 90% for the Bayesian learner. At any cost above about 1 point per example, a conventional principal is best off telling the agent nothing, while an unusual one should give 2–3 examples. The default serves the typical principal for free and taxes the unusual one.</p>
</section>

<section data-strata-id="s7">
<h2>Limits and next steps</h2>
<ul class="new" data-strata-id="s7l">
<li>Only five rounds: where the optimum is 5, it may lie further out.</li>
<li>The unusual/conventional rule was set after seeing the data; a pre-registered set of about 20 new unusual principals would confirm it.</li>
<li>One menu seed, 50 users; open models only.</li>
</ul>
</section>
'''

HTML = '''<title>Delegating Preferences Draft</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
/* A single essay column; figures sit on white plates; paragraph margins mark whose words they are. */
:root{--bg:#f6f6f3;--ink:#1b1f24;--ink-2:#4d5560;--ink-3:#808894;--rule:#dcdfe3;--plate:#ffffff;--tint:#eceee9;
--yours:#2f7a5b;--new:#b5651d;--serif:"Source Serif 4",Georgia,serif;--sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181b;--ink:#e8eaec;--ink-2:#b3bac2;--ink-3:#808894;--rule:#2c3238;--plate:#ffffff;--tint:#1f2428;--yours:#5fbf94;--new:#e09a55;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15181b;--ink:#e8eaec;--ink-2:#b3bac2;--ink-3:#808894;--rule:#2c3238;--plate:#ffffff;--tint:#1f2428;--yours:#5fbf94;--new:#e09a55;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.65}
main{max-width:760px;margin:0 auto;padding-block:36px 90px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(34px,5vw,48px);line-height:1.08;margin:6px 0 8px}
.sub{font-size:19px;color:var(--ink-2);margin:0 0 14px}
h2{font-family:var(--sans);font-weight:600;font-size:22px;margin:46px 0 12px}
p{margin:0 0 16px}
p.yours,p.new,ul.new{padding-left:14px;border-left:3px solid var(--yours)}
p.new,ul.new{border-left-color:var(--new)}
.newin{background:linear-gradient(transparent 62%,color-mix(in srgb,var(--new) 22%,transparent) 62%)}
.key{font-family:var(--sans);font-size:13px;color:var(--ink-2);display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}
.tag{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:4px;color:#fff}
.tag.yours{background:var(--yours)} .tag.new{background:var(--new)}
.count{margin-left:auto;font-family:var(--mono);color:var(--ink-3)}
figure{margin:20px 0;background:var(--plate);border:1px solid var(--rule);border-radius:8px;padding:10px}
figure img,.wk-foot img{display:block;width:100%;height:auto}
figure.eq{padding:14px 18px}
.walk{margin:24px 0;border:1px solid var(--rule);border-radius:10px;padding:16px;background:var(--tint);font-family:var(--sans);font-size:15px}
.wk-head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}
.rounds{display:flex;flex-direction:column;gap:10px}
.rd{background:var(--bg);border-radius:8px;padding:12px 14px;min-width:0}
.rd-n{font-family:var(--mono);font-size:12px;color:var(--ink-3);margin-bottom:6px}
.menu{border-collapse:collapse;font-family:var(--mono);font-size:13px;width:100%;margin-bottom:8px}
.menu td{padding:4px 8px;border-bottom:1px solid var(--rule);white-space:nowrap}
.menu tr.agent td{background:color-mix(in srgb,var(--new) 15%,transparent)}
.menu tr.user td{background:color-mix(in srgb,var(--yours) 18%,transparent)}
.menu tr.agent td:first-child::after{content:"  ← Gemma";color:var(--new)} .menu tr.user td:first-child::after{content:"  ← user";color:var(--yours)}
.think{font-family:var(--serif);font-style:italic;color:var(--ink-2);margin:4px 0}
.fb{font-size:14px}
.wk-foot{margin-top:12px;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:14px;align-items:center}
.wk-foot img{background:#fff;border-radius:6px;padding:6px}
.wk-foot p{font-size:14.5px;margin:0}
.tables{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:18px;margin:18px 0;font-family:var(--sans);font-size:14px}
table.bt{border-collapse:collapse;width:100%;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink)}
table.bt caption{text-align:left;font-weight:600;padding-bottom:6px}
table.bt th,table.bt td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
table.bt th{color:var(--ink-2);font-weight:500;white-space:nowrap}
table.bt.num td+td,table.bt.num th+th{text-align:right;font-variant-numeric:tabular-nums}
ul.new{margin:0;padding-left:34px}
.menu-wrap{overflow-x:auto}
@media (max-width:640px){.tables,.wk-foot{grid-template-columns:minmax(0,1fr)} body{font-size:17px}}
</style>
<main>''' + BODY + '''</main>
<script>
const t=[...document.querySelectorAll("main p, main li")].map(e=>e.textContent).join(" ");
document.getElementById("wc").textContent=t.trim().split(/\\s+/).length+" words";
</script>'''
(D / "delegating_preferences_draft.html").write_text(HTML)
print(len(HTML))
