"""Builds the annotatable HTML draft of the 'Delegating preferences' write-up (v2)."""

import base64
import json
from pathlib import Path

D = Path(__file__).resolve().parent
F = D.parent / "delegation"


def img(p):
    return "data:image/png;base64," + base64.b64encode((F / p).read_bytes()).decode()


S = json.loads((F / "summary.json").read_text())


def m(body, display=True):
    """MathML wrapper: display equations get their own centred line."""
    attr = ' display="block"' if display else ""
    for ch in "()[]{}":  # fences never stretch; otherwise they grow to the whole line
        body = body.replace(f"<mo>{ch}</mo>", f'<mo stretchy="false">{ch}</mo>')
    return f"<math{attr}>{body}</math>"


th_i = "<msub><mi>θ</mi><mi>i</mi></msub>"
EQ_U = m(
    f"<msub><mi>U</mi><mi>i</mi></msub><mo>(</mo><mi>f</mi><mo>)</mo><mo>=</mo>"
    f"<msubsup><mi>θ</mi><mi>i</mi><mo>⊤</mo></msubsup><mi>φ</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>,</mo><mspace width='1.2em'/>"
    f"<mi>φ</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>∈</mo><msup><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mn>4</mn></msup><mo>,</mo><mspace width='1.2em'/>"
    f"{th_i}<mo>∈</mo><msup><mrow><mo>{{</mo><mo>−</mo><mn>1</mn><mo>,</mo><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>,</mo><mn>0</mn><mo>,</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>,</mo><mn>1</mn><mo>}}</mo></mrow><mn>4</mn></msup>"
)
EQ_B = m(
    "<msub><mi>H</mi><mi>t</mi></msub><mo>=</mo><mo>{</mo><mi>θ</mi><mo>:</mo><mtext>θ consistent with every pick so far</mtext><mo>}</mo>"
) + m(
    "<msub><mover><mi>f</mi><mo>^</mo></mover><mi>t</mi></msub><mo>=</mo><munder><mrow><mi>arg</mi><mo>&#x2061;</mo><mi>max</mi></mrow><mi>f</mi></munder>"
    "<mspace width='0.2em'/><mi>𝔼</mi><msup><mrow><mo>[</mo><mi>θ</mi><mo>∣</mo><msub><mi>H</mi><mi>t</mi></msub><mo>]</mo></mrow><mo>⊤</mo></msup><mi>φ</mi><mo>(</mo><mi>f</mi><mo>)</mo>"
)
EQ_W = m(
    "<msub><mi>w</mi><mi>i</mi></msub><mo>(</mo><mi>menu</mi><mo>)</mo><mo>=</mo><mfrac>"
    "<mrow><msub><mi>U</mi><mi>i</mi></msub><mo>(</mo><mtext>agent’s pick</mtext><mo>)</mo><mo>−</mo><msub><mi>min</mi><mi>f</mi></msub><msub><mi>U</mi><mi>i</mi></msub><mo>(</mo><mi>f</mi><mo>)</mo></mrow>"
    "<mrow><msub><mi>max</mi><mi>f</mi></msub><msub><mi>U</mi><mi>i</mi></msub><mo>(</mo><mi>f</mi><mo>)</mo><mo>−</mo><msub><mi>min</mi><mi>f</mi></msub><msub><mi>U</mi><mi>i</mi></msub><mo>(</mo><mi>f</mi><mo>)</mo></mrow></mfrac>"
    "<mo>,</mo><mspace width='1.2em'/><mi>W</mi><mo>(</mo><mi>n</mi><mo>)</mo><mo>=</mo><mn>100</mn><mo>·</mo><mover><msub><mi>w</mi><mi>i</mi></msub><mo>‾</mo></mover>"
)
EQ_NW = m(
    "<mi>NW</mi><mo>(</mo><mi>n</mi><mo>;</mo><mi>c</mi><mo>)</mo><mo>=</mo><mi>W</mi><mo>(</mo><mi>n</mi><mo>)</mo><mo>−</mo><mi>c</mi><mo>·</mo><mi>n</mi><mo>,</mo><mspace width='1.2em'/>"
    "<msup><mi>n</mi><mo>*</mo></msup><mo>(</mo><mi>c</mi><mo>)</mo><mo>=</mo><munder><mrow><mi>arg</mi><mo>&#x2061;</mo><mi>max</mi></mrow><mrow><mi>n</mi><mo>∈</mo><mo>{</mo><mn>0</mn><mo>,</mo><mo>…</mo><mo>,</mo><mn>5</mn><mo>}</mo></mrow></munder><mspace width='0.2em'/><mi>NW</mi><mo>(</mo><mi>n</mi><mo>;</mo><mi>c</mi><mo>)</mo>"
)
EQ_D = m(
    "<msub><mi>D</mi><mi>i</mi></msub><mo>=</mo><mn mathvariant='bold'>1</mn><mo>[</mo><mspace width='0.15em'/><mo>#</mo><mo>{</mo><mi>k</mi><mo>∈</mo><mo>{</mo><mtext>duration, stops, price</mtext><mo>}</mo><mo>:</mo>"
    "<msub><mi>θ</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>&gt;</mo><mn>0</mn><mo>}</mo><mo>≥</mo><mn>2</mn><mspace width='0.15em'/><mo>]</mo>"
)

BN = S["best_n"]
COST = BN["costs_pct_per_example"]


def bn_row(label, g, a):
    return f"<tr><td>{label}</td>" + "".join(f"<td>{v}</td>" for v in BN[g][a]) + "</tr>"


BN_TABLE = (
    '<table class="bt num" data-strata-id="tab-bestn"><caption>Best number of examples, n*, by cost per example (points of welfare)</caption>'
    "<tr><th></th>" + "".join(f"<th>c = {c}</th>" for c in COST) + "</tr>"
    '<tr class="grp"><td colspan="6">Conventional principals (n = 14)</td></tr>'
    + bn_row("Gemma 4 31B, thinking", "conventional", "gemma4-31b-think")
    + bn_row("Gemma 4 31B", "conventional", "gemma4-31b")
    + bn_row("Qwen 2.5 7B", "conventional", "qwen25-7b")
    + '<tr class="grp"><td colspan="6">Unusual principals (n = 18)</td></tr>'
    + bn_row("Gemma 4 31B, thinking", "unusual", "gemma4-31b-think")
    + bn_row("Gemma 4 31B", "unusual", "gemma4-31b")
    + bn_row("Qwen 2.5 7B", "unusual", "qwen25-7b")
    + '<tr class="grp"><td colspan="6">Either type</td></tr>'
    + bn_row("Bayesian learner", "unusual", "bayes")
    + "</table>"
)

BODY = f'''
<header data-strata-id="title">
  <div class="eyebrow">Draft · v2 · for annotation</div>
  <h1>Delegating preferences</h1>
  <p class="sub">Replicating Qiu et al. (2026) with newer LLMs, and pricing what it costs to teach an AI agent what you want.</p>
  <div class="key"><span class="tag yours">yours</span> from your raw material, trimmed <span class="tag new">new</span> written by Claude <span class="count" id="wc"></span></div>
</header>

<section data-strata-id="s1">
<h2>Why this matters</h2>
<p class="yours" data-strata-id="s1p1">The biggest delegation problem we're going to face is gradually handing more and more decision-making power to AI agents. Partly for convenience: “friction” is the metaphor of choice for the small, thoughtless activities that slow us down, and agents are yet another way of removing it. This is the advent of the personal agent. Partly to extend our reach: until now, the only way to cope with capacity was an extra pair of hands. Software eased that, and it was with software that we began to see millions of dollars of revenue per person. Agents are very much the culmination of that.</p>
<p class="yours" data-strata-id="s1p2">There's something promising, and frankly a little scary, about extending one's exact preferences across a swarm of AI agents. Delegating has always meant aligning many people's interests and telling a captivating story (even the strictest, most authoritarian people have had to). With agents, that constraint starts to disappear.</p>
<p class="yours" data-strata-id="s1p3">But nonetheless, the question of: how efficiently do these agents translate our preferences? Adjacently, whose preferences are embedded in the models themselves, likely from pre-training and RLHF emphases? <span class="newin">Kirk et al. (2024) find that RLHF generalises better than supervised fine-tuning but significantly reduces output diversity: post-training pulls a model away from the spread of what people do and towards one preferred answer.</span> And which types of preferences take more care to communicate to them?</p>
</section>

<section data-strata-id="s2">
<h2>The experiment</h2>
<p class="yours" data-strata-id="s2p1">Qiu et al. (2026) set out to show how Bayesian learning can be used as a training objective, a novel form of teaching that overcomes LLMs' failure to update their beliefs across interactions. Dually, and the contribution of this extension here, it can be read through an economics angle: how efficiently can we translate our preferences into LLMs?</p>
<p class="yours" data-strata-id="s2p2">The experiment simulates a user whose utility function we know. Flights are described by four features, departure time, duration, number of stops and price, each scaled to between 0 and 1, and the user's utility is linear in them: <span class="newin">a weight from −1 to 1 on each, so a user can like or dislike any feature, or not care.</span></p>
<div class="eq" data-strata-id="eq-u">{EQ_U}</div>
<p class="yours" data-strata-id="s2p2b">The model is shown a menu of three flights and picks one; the user says whether it was right, and which flight they'd have picked. In the first round there is no history, so it is basically a guess. The clever part is that this runs for five rounds, so we can see how much better the model gets as it sees more of the user's past choices. <span class="newin">After each round the conversation is frozen and the model is tested on 100 new menus without feedback, so we measure what it has learned rather than what it was told.</span></p>
<p class="yours" data-strata-id="s2p3">The benchmark is a Bayesian learner. It's a little bit of cheating: it knows the functional form of the user's utility, which the LLM doesn't. <span class="newin">It starts with every weight vector equally likely, throws out those that contradict a choice the user has made, and recommends the flight that is best on average over what remains.</span> It is the information optimum: how much can be learned from these examples at all.</p>
<div class="eq" data-strata-id="eq-b">{EQ_B}</div>
<p class="new" data-strata-id="s2p4">To make this concrete, here is one user, round by round, with Gemma 4 31B (thinking) as the agent.</p>

<div class="walk" data-strata-id="walk">
  <div class="wk-head"><span class="tag new">new</span><b>User 343</b> doesn't care when they fly, mildly prefers longer flights with more stops, and strongly prefers pricier ones: θ = (0, ½, ½, 1).</div>
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
    <p>On the 100 held-out menus, Gemma (thinking) goes from 6% to 77% and the Bayesian learner from 27% to 90%; Qwen never gets above chance. Gemma's round 0 is not a coin flip: it starts from a typical traveller, cheap, short and direct, which is exactly wrong for this user. We come back to that.</p>
  </div>
</div>
</section>

<section data-strata-id="s3">
<h2>Replication and newer models</h2>
<p class="yours" data-strata-id="s3p1">The paper's basic finding was that, back when they tested this, LLMs were very poor at learning from past interactions to infer something about future choices. <span class="newin">We replicate it with the paper's own users and menus and its weakest model, Qwen 2.5 7B, then run Gemma 4 31B with and without thinking.</span></p>
<table class="bt solo" data-strata-id="tab-design"><caption>Design</caption>
<tr><th>Users</th><td>50 simulated, stratified by preference strength</td></tr>
<tr><th>Rounds</th><td>5, with feedback after each pick</td></tr>
<tr><th>Test</th><td>100 held-out menus after each round, no feedback; 605 decisions per user and agent</td></tr>
<tr><th>Agents</th><td>Qwen 2.5 7B · Gemma 4 31B · Gemma 4 31B thinking · Bayesian learner</td></tr>
<tr><th>Outcomes</th><td>Accuracy (picks the user's flight); share of achievable welfare</td></tr></table>
<figure data-strata-id="fig1"><img alt="Held-out accuracy by round" src="{img('figures/1_replication.png')}"></figure>
<p class="new" data-strata-id="s3p2">The replication holds: Qwen lands at 38% against the paper's 37%, flat after the first round. The picture for newer models is different.</p>
<table class="bt num solo narrow" data-strata-id="tab-repl"><caption>Final-round accuracy (%)</caption>
<tr><th></th><th>Paper</th><th>Ours</th></tr>
<tr><td>Bayesian learner</td><td>≈80</td><td>80.6</td></tr>
<tr><td>Qwen 2.5 7B</td><td>37</td><td>38.0</td></tr>
<tr><td>Gemma 2 9B → Gemma 4 31B</td><td>37</td><td>59.4</td></tr>
<tr><td>Gemma 4 31B, thinking</td><td>–</td><td>62.7</td></tr></table>
<p class="yours" data-strata-id="s3p3">Compared to the original experiment, the models have gotten substantially better. Gemma 4 is an open model, not the frontier, and it's roughly a doubling of the score. <span class="newin">Measured above chance (33%), the learning signal went from about 4 points to 26–30. Thinking adds a little (+3.4 points, 95% CI 1.8–5.0) without changing how fast the model learns.</span> The gap to the Bayesian learner (which, remember, has a strictly greater informational basis) is still pretty surprising.</p>
</section>

<section data-strata-id="s4">
<h2>Trying to price the cost of instruction</h2>
<p class="yours" data-strata-id="s4p1">From an economic angle this is very close to a standard setup. A consumer has a utility function, and has delegated their choices to a personal assistant who can only learn from past examples. The relevant metric is how good this assistant is at understanding my preferences, and we can express that as welfare: <span class="newin">on each menu, the share of the achievable utility range the agent's pick delivers, where 0 is the worst flight on the menu and 100 the best. Picking at random gives about 50.</span></p>
<div class="eq" data-strata-id="eq">{EQ_W}</div>
<p class="yours" data-strata-id="s4p2">Each example I provide is a cost, and the welfare the agent delivers is the benefit. <span class="newin">The cost is the principal's time and attention: choosing, correcting, explaining.</span> We assume for now that it is linear, a cost <i>c</i> per example, measured in points of welfare. Net welfare and the best number of examples are then:</p>
<div class="eq" data-strata-id="eq-nw">{EQ_NW}</div>
<p class="yours" data-strata-id="s4p3">And we get the classic trade-off curve: there's a benefit to specifying, as the agent gets better, but also a cost. <span class="newin">With a cost of 2 points per example, net welfare rises, peaks and flattens: everyone except Qwen should give about three examples.</span></p>
<figure data-strata-id="fig3"><img alt="Welfare and net welfare by number of examples, all users" src="{img('figures/6_hump_all.png')}"></figure>
<p class="new" data-strata-id="s4p4">The curves also give an exchange rate between agents. Five rounds with Gemma (thinking) buy the same welfare as about 1.6 rounds with the Bayesian learner (95% CI 1.2–1.9); for Qwen it is 0.2. Put differently, the principal of the best LLM has to give roughly three times as many examples to get the same welfare.</p>
</section>

<section data-strata-id="s5">
<h2>A principal–agent reading</h2>
<p class="yours" data-strata-id="s5p1">This is a little dissimilar to the usual setups. When a principal delegates a task, one can decompose it along three components, all necessary for good delegation:</p>
<ul class="yours" data-strata-id="s5l">
<li><b>Interest alignment:</b> to what extent do the principal's interests transmit to the agent? Does the agent want what the principal wants?</li>
<li><b>Learning efficiency:</b> how much does each example teach it?</li>
<li><b>Default rule:</b> what is the baseline when the principal has said nothing?</li>
</ul>
<p class="yours" data-strata-id="s5p2">The last two are the interesting ones. When we hire someone with law experience, their baseline is quite high: they know what to act on, what to do. Compare that to a junior, who may have a poor default rule but a higher learning efficiency. <span class="newin">In the curves above, learning efficiency is the slope and the default rule is the intercept at zero examples.</span></p>
<p class="yours" data-strata-id="s5p3">Orthogonal to both is interest alignment: is there an antagonistic relationship between the principal and the agent? For language models, training addresses this head-on. The RLHF step has it as the explicit intent, aligning the model with the preferences of humans rating a whole bunch of answers. The more modern flavour, reinforcement learning from verifiable rewards (RLVR), does it indirectly: the model is given an explicit task and judged on whether it reaches the goal. So in this kind of delegation, the interest problem is essentially short-circuited.</p>
<p class="yours" data-strata-id="s5p4">Unlike the classic case of optimal delegation to a biased agent (Alonso &amp; Matouschek, 2008), there's no real bias in the agent we're delegating to here. There are fringe cases: the AI labs also run strong research programmes to make sure models can't be used for harm, and in those cases the agent is deliberately biased against some principals. But on the whole, we can think of these agents as faithful delegates. The real problem is capability: for now they are simply unable to take in the signal.</p>
<p class="yours" data-strata-id="s5p5">That's true for the Bayesian learner, but not for the LLMs, and round 0 is where the agent's own preferences show.</p>
</section>

<section data-strata-id="s6">
<h2>Unusual principals and strong defaults</h2>
<p class="yours" data-strata-id="s6p0">What I said about no bias isn't quite true. If we analyse the round-0 traces, there's a direct preference for cheap, short, direct flights. The model doesn't act as an agnostic agent with no information; it presumes some baked-in things, very likely from its training data.</p>
<figure data-strata-id="fig5"><img alt="Round-0 accuracy by number of contrarian tastes" src="{img('figures/5_round0_default.png')}"></figure>
<p class="yours" data-strata-id="s6p0b">With no information, all three models serve the same default traveller. They're right about 60% of the time for users who share that taste and 7–8% for users who want the opposite on everything. In 60% of round-0 traces, the reasoning appeals to what “a typical traveller” wants. <span class="newin">The Bayesian learner, with a flat prior, sits at chance for everyone. And the default isn't only a starting point: for users who don't care about price, Gemma keeps filling in a preference for cheap flights even after five rounds of feedback.</span></p>
<p class="yours" data-strata-id="s6p1">This is where unusual principals come in. Some preferences are unusual by definition: preferring longer flights, pricier flights, more stops. <span class="newin">We call a principal unusual if they prefer the odd direction on at least two of the three (18 of 50 users), and conventional if on none (14). The 18 in between are left out.</span></p>
<div class="eq" data-strata-id="eq-d">{EQ_D}</div>
<p class="new" data-strata-id="s6p2">Splitting the net-welfare curves by principal type, and comparing only the LLMs, the default rule turns out to decide how much instruction pays. A conventional principal starts at 73–77% of achievable welfare with no examples, already better than the Bayesian learner after one round (72%), and the models barely improve from there. For them, every example is mostly cost. An unusual principal starts at 33%, below random, and has to teach the agent; Gemma (thinking) gets them to 72%, still well short of the Bayesian learner's 90%.</p>
<figure data-strata-id="fig2"><img alt="Net welfare for conventional vs unusual principals, LLMs only" src="{img('figures/7_llm_split.png')}"></figure>
<p class="new" data-strata-id="s6p3">The table reads the same curves across costs. Once an example costs more than about a point of welfare, a conventional principal is best off telling the agent nothing, while an unusual one should still give two or three. For the Bayesian learner, the two types are treated identically. The default serves the typical principal for free and taxes the unusual one.</p>
{BN_TABLE}
</section>

<section data-strata-id="s7">
<h2>Limits and next steps</h2>
<ul class="new" data-strata-id="s7l">
<li>Only five rounds: where the best number of examples is 5, it may lie further out.</li>
<li>The cost per example is assumed, not measured. A realistic cost (time to give an example, or a price from a survey) would pin down where the hump sits.</li>
<li>The unusual/conventional rule was set after seeing the data. A pre-registered run on about 20 new unusual principals would confirm it.</li>
<li>One menu seed, 50 users, open models only; Gemma ran on a single provider whose quantisation is unstated.</li>
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
h2{font-family:var(--sans);font-weight:600;font-size:22px;margin:52px 0 14px}
p{margin:0 0 16px}
p.yours,p.new,ul.new,ul.yours{padding-left:14px;border-left:3px solid var(--yours)}
p.new,ul.new{border-left-color:var(--new)}
ul.yours,ul.new{margin:0 0 16px;padding-left:34px}
ul li{margin-bottom:6px}
.newin{background:linear-gradient(transparent 62%,color-mix(in srgb,var(--new) 22%,transparent) 62%)}
.key{font-family:var(--sans);font-size:13px;color:var(--ink-2);display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}
.tag{font-family:var(--mono);font-size:11px;padding:2px 7px;border-radius:4px;color:#fff}
.tag.yours{background:var(--yours)} .tag.new{background:var(--new)}
.count{margin-left:auto;font-family:var(--mono);color:var(--ink-3)}
figure{margin:28px 0;background:var(--plate);border:1px solid var(--rule);border-radius:8px;padding:10px}
figure img,.wk-foot img{display:block;width:100%;height:auto}
.eq{margin:4px 0 22px;padding:10px 12px;overflow-x:auto;font-size:19px;color:var(--ink)}
math{font-family:"STIX Two Math","Latin Modern Math","Cambria Math",var(--serif)}
.walk{margin:28px 0;border:1px solid var(--rule);border-radius:10px;padding:16px;background:var(--tint);font-family:var(--sans);font-size:15px}
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
table.bt{border-collapse:collapse;width:100%;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);font-family:var(--sans);font-size:14px;margin:30px 0 30px}
table.bt.narrow{max-width:460px;margin-left:auto;margin-right:auto}
table.bt caption{text-align:left;font-weight:600;padding-bottom:8px}
table.bt th,table.bt td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
table.bt th{color:var(--ink-2);font-weight:500;white-space:nowrap}
table.bt.num td+td,table.bt.num th+th{text-align:right;font-variant-numeric:tabular-nums}
table.bt tr.grp td{font-weight:600;color:var(--ink-2);padding-top:12px;text-align:left}
.menu-wrap{overflow-x:auto}
@media (max-width:640px){.wk-foot{grid-template-columns:minmax(0,1fr)} body{font-size:17px} .eq{font-size:16px} table.bt{font-size:13px}}
</style>
<main>''' + BODY + '''</main>
<script>
const t=[...document.querySelectorAll("main p, main li")].map(e=>e.textContent).join(" ");
document.getElementById("wc").textContent=t.trim().split(/\\s+/).length+" words";
</script>'''
(D / "delegating_preferences_draft.html").write_text(HTML)
print(len(HTML))
