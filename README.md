# Bayesian teaching: replication and economic extension (Spike 2)

Replicates the off-the-shelf result of Qiu et al. (2026), *Bayesian teaching enables probabilistic
reasoning in large language models* (Nature Communications 17:1238), on the flight recommendation
task, and extends it with (a) Gemma 4 31B with thinking off vs on and (b) utility regret
θᵀφ(best) − θᵀφ(picked).

## Setup

```bash
uv sync
uv run python scripts/download_paper_data.py   # Zenodo 17677329 → data/paper/ (git-ignored)
```

## Gates

| Slice | Command | Gate |
|---|---|---|
| S0 | `uv run python scripts/check_bayesian_assistant.py` | Bayes 33 → 57 → 67 → 73 → 77 → 80% on all 624 users |
| S1 | `uv run python scripts/check_simulated_agents.py` | random ≈ 33%, oracle 100% with regret 0, 605 decisions per user |
| S2 | `uv run python scripts/run_experiment.py --config configs/s2_one_llm.toml` then `scripts/check_one_llm.py` | 605 parsed replies, 0 duplicate calls after kill + resume, transcript as Table H3 |
| S3 | `run_experiment.py --config configs/pilot.toml`, `analyse_results.py results/pilot --add results/probe-friendli`, `estimate_cost.py --pilot results/pilot --add results/probe-friendli`, `check_pilot.py` | 0 unparsed, providers stable, thinking cap binds < 5%, main run projected < $20, every table and figure |

S3 status: **gate fails** on Gemma 4 thinking (runaway thinking; no provider enforces the 1,024 budget),
cost ($39 projected) and Crusoe throughput (shared-pool rate limits). See `results/pilot/analysis/`.
`results/probe-friendli` holds both Gemma arms on Friendli for the same 2 users (Crusoe could not finish them).

## Data notes

- The interaction file's menus are mapped to features with a per-feature text → value lookup built
  from the held-out file; it covers every flight and agrees with the file's own `rounds_numpy`.
- The Bayesian Assistant keeps every θ under which each user choice is a best option (uniform prior
  over the 624 non-zero θ), picks argmax of E[θ]ᵀφ, and breaks ties at random (seeded by state+menu).
- Accuracy is scored against the paper's labelled choice. 5.3% of held-out (user, menu) pairs are
  ties under θ, and the paper's labels break those at random, so we also report tie-aware accuracy. The oracle knows the user, including how they break ties.
- `data/user_sample.json`: all 624 users in a seeded order stratified by ‖θ‖ into the plan's five
  groups (88/144/88/96/208); any prefix keeps the mix. The main run uses the first 50, the pilot the first 2.

## Prompts and parsing

- Wording from Supp. Table H3; chat layout (`\n\n` between parts, `Flight N:\n<attributes>`) and the
  "Your option Flight N is correct." line from the paper's own Zenodo training dialogues.
- Past assistant turns are re-sent as exactly "The best option is Flight N."; raw replies and thinking
  are only logged.
- A pick is read only from an explicit declaration ("The best option is Flight N", "Flight N is the best
  option", …) or a bare "Flight N". Anything else gets one retry with the reminder
  "Answer with exactly one sentence of the form 'The best option is Flight N.'" appended to the last
  user message; if that fails too, the pick is scored wrong (and worst-option regret) and reported.
  Qwen 2.5 7B needs that reminder for every round-0 prompt: unprompted it starts an analysis
  ("Based on your preferences, …") that 16 tokens cut off.

## Running

`scripts/run_experiment.py --config configs/<x>.toml [--max-cost USD] [--push-every SECONDS]`. Every call is
logged under `results/<run_id>/calls/<arm>/<user>.jsonl.gz`, keyed by sha256(model, provider, messages,
params); rerunning the same config reuses the log and only pays for missing calls. With `--push-every`
the logs are committed and pushed during the run. Each run writes `manifest.json` and
`decisions.parquet`.
