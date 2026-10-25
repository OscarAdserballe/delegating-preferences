# Exploration notes (main run, 50 users) — candidates, with robustness flags

## Defaults (round 0)
- All three LLMs share a default user: cheap, short, few stops, slightly early (logit weights ≈ −1, −1, −1, −0.2/−0.35). Default picks agree 69–84% across models; almost never Pareto-dominated.
- 62% of round-0 thinking traces appeal to a "typical traveller / most people"; 38% frame it as a puzzle/test. Fades to ~0 by round 3.
- Round-0 accuracy: 59–62% for conventional users (0 contrarian attrs, n=14) vs 7–8% for 3-contrarian (n=3). corr(alignment with default, acc0) ≈ 0.98.

## Learning depends on the user's distance from the default  [ROBUST, bootstrap]
- Thinking gain cp0→5: conventional +7 [+2,+14] vs 2–3 contrarian +38 [+29,+47]; Bayes +46 vs +47.
- Default dependence corr(alignment, acc) by cp: Qwen 0.97→0.81 (never lets go); Gemma 0.95→0.36; thinking 0.98→0.28–0.36; Bayes ~0.2.
- Bayes with a strongly "conventional" prior starts like Gemma (66% / 17%) but reaches 86% / 71% — prior alone does not explain the gap; updating does.
- Best simple model of thinking-Gemma's curves: conventional prior (s≈2–3) + exact constraint updating + noisy choice (softmax T≈0.2), RMSE 3.4 pts. Under-reaction trades off with prior strength → suggestive only.
- Conventional users: Gemma plateaus (59→65); welfare: Gemma with 0 info (74–77%) beats Bayes after 1 round (72%) but ends 80–83 vs 91.

## Attribute-level  [ROBUST via per-user conditional logit; grid method too harsh]
- Departure time ignored: revealed |weight| 0.30 vs 1.0 (oracle) and 0.90 (Bayes); direction right 60% (≈chance) vs Bayes 92%.
- Contrarian price hardest: pricier-loving users' price direction right 61% at cp5 (oscillates 44–61 after cp2) vs Bayes 94%; contrarian duration 88%, stops 83%.
- Wrong picks are cheaper than the user's best 59–64% (Bayes 44%); Gemma loses most utility on departure (0.19 per wrong pick, > random 0.12).
- Traces: when stated, directions mostly right (duration 96, stops 80, price 77) but stated only 18–43% of the time (regex, rough).
- Stated belief → action: "prefers pricier" → picks priciest 73% vs 30%; "prefers early" → earliest 53% vs 39%.

## Surface heuristics  [ROBUST]
- Position echo: after "incorrect. I prefer Flight N", picks position N on unrelated menus: Qwen 86% (+48.5 pts paired), Gemma 50% (+13.8), thinking 45% (+8.2); chance 33.5%. Echo picks less accurate (Gemma 45 vs 55).
- Qwen first replies after "incorrect": 28% open "Given your preference for Flight N", N = named flight 99.8%.
- Explicit position hypotheses in thinking are rare (≤0.2%) → Gemma's echo is implicit.
- Qwen: 57% of its correct cp5 picks are its unchanged default; on menus where its default is wrong: 25.8% (below chance).

## How thinking reasons
- Lexicographic, not compensatory: 41% of traces rank attributes ("priority 1", "regardless of"); compensatory scoring 0.1%.
- 2-level lexicographic rules fit thinking picks 80% vs users' own choices 77% — more rule-like than the users.
- Reads trade-offs as contradictions: 12% of cp≥2 traces call the (always-consistent) user contradictory/inconsistent; price-lovers 20% vs 7%.
- One-shot rule induction / over-correction after a correction (qualitative: "Crucial insight… Priority 1: stops" → 20 h non-stop).
- Explicit hypothesising peaks at round 1 (2.0/trace) and fades to 0.36 by round 5.
- 47% of thinking characters at round 5 restate past flights; 98% of cp5 traces revisit all five rounds; budget binds on transcription.
- Thinking effort tracks surprise not uncertainty: cut-off share 44% (0 incorrect) → 92% (2) → 100% (4+); within-cp corr with #incorrect 0.38–0.61, with Bayes remaining bits −0.15…−0.51.
- Forced share ↑ → coherence ↓ (r −0.39…−0.80 within cp).
- Forced answer ≠ simple continuation: matches last-named flight 46%; more accurate (51%) than that "lean" (37%).

## Dynamics
- After "incorrect": Bayes +18.9 pts next checkpoint; Gemma +6.6, thinking +9.5, Qwen +0.2.
- Unlearning: right→wrong ~10%/round for Gemma, never declines; Bayes 15→2%.
- Flips after incorrect: Gemma 43–44% of held-out picks change vs Bayes 30%; implied-θ consecutive cosine ~0.70 at cp5 vs Bayes 0.95.
- Per-round learning uncorrelated with round informativeness (r 0.06–0.12 vs Bayes 0.59).
- Recency: copies most recent choice slightly more (52.8 → 46%); Bayes flat.

## Economics / welfare
- Share of achievable welfare cp5: Bayes 91, thinking 77.6, Gemma 75.0, copy_last 68.7, Qwen 54.5, random 49.5.
- Exchange rate: Gemma after 5 rounds ≈ Bayes after 2 (overall); by type: conventional 2.3, contrarian 1.4.
- Tail: Bayes worst user 71%; thinking 8% of users <60%, 20% <70%; worst-served mostly contrarian.

## Menus / users
- Difficulty for Bayes uncorrelated with difficulty for Gemma across users (r 0.05), partly shared across menus (r 0.52).
- When Bayes is wrong (ambiguous), LLMs at chance (~32%).
- Stakes: clear menus 74–77% vs Bayes 96%.
- Compromise options (6.5% of menus): thinking 38% < no-thinking 43%; Bayes 63%.
- Single-attribute users: Gemma ≥ Bayes (user 310 price-only: 94 vs 67) — sparse-prior Bayes would help those but hurts overall.
- Near vs far from past choices: all arms drop; Gemma far 35–38% > copy_last 24%.

## Did NOT hold / weak
- "Thinking helps the tail most": regression to the mean; by contrarianism no difference (+3.0/+1.5/+3.4).
- "Unusual/odd" labelling of contrarian preferences: small (10 vs 4% at round 1).
- ‖θ‖ strata: no pattern. Key-insight phrasing: no accuracy link.
- Forced-vs-unforced accuracy gap is selection (previous analysis).

## Late additions
- Matched-start check: conventional-prior Bayes (s=3) vs thinking Gemma: cp0 diff 0–4 pts; cp5 +16 [+10,+22] conv, +10.5 [+4,+18] 1-contra, +15.6 [+6,+25] 2–3 contra.
- Exchange rate (interpolated, bootstrap): thinking 1.56 [1.19,1.92] Bayes rounds; no-think 1.32 [0.97,1.64]; Qwen 0.22 [0.07,0.39].
- Archetypes (thinking): 9 start-right-stay-put (all conventional), 17 start-wrong-learn (mean contra 1.9), 6 failing; Qwen 12 stay-put / 38 failing; Bayes 42 learners.
- Attribute updates after a correction: shift toward the named flight's extremes +20 (thinking) vs +21 (Bayes), persists a round later (58 vs 61); Qwen +2.
- Evidence integration: after two corrections pointing to different extremes, thinking honours the latest +7.9 [+2.9,+12.6] more than the earlier; Bayes −5.4 (n.s.).
- Last-position penalty: no-think −6.1 [−9.1,−3.1], thinking −3.5 [−6.6,−0.3], Bayes −1.4 n.s.
- Thinking helps 29 users (> +2 welfare), ~same 10, hurts 11.
- Round-0 thinking is short when one option is cheapest+shortest+fewest stops (0% cut, 1.4k chars) vs 26% cut when options trade off.
- Volatility of accuracy across rounds: not distinctive vs Bayes (drops >10 pts: 6% vs 4%).
- Confident end-of-trace wording → 79% vs 69% accuracy; hedging absent.
- Vignettes: 343 (pricier/longer/stops, 6→77), 23 (mixed, oscillates 34→58→36→52→63→27), 156 (all −0.5; 5/5 live right, held-out 79→72), 126 (one correction → 'Priority 1: stops' → 20 h non-stop), 310 (price only, 56→96 in one round; Bayes 67).
- Plateau mechanism: net learning per round (wrong→right − right→wrong, cp2–5): thinking conventional +1.8 [+0.6,+3.1] vs contrarian +5.9; no-think conventional +0.8 [−0.4,+2.0]; Bayes +6.2 / +6.9. Unlearn rate higher for contrarians (12% vs 8%).
- Default fills silence: users indifferent to price (θ=0, n=11) get revealed price weight −0.56 (thinking) vs Bayes +0.04; thinking − Bayes −0.60 [−0.96,−0.25]; persists cp1–5; 73% of these users get a clearly negative price weight (Bayes 36%).
- Nuance: a conventional-prior Bayesian (s=3) also gives price-indifferent users a −0.47 price weight → fill-in is what any default prior does when evidence is silent; the issue is whose prior (the agent's), not irrationality.
