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

## Data notes

- The interaction file's menus are mapped to features with a per-feature text → value lookup built
  from the held-out file; it covers every flight and agrees with the file's own `rounds_numpy`.
- The Bayesian Assistant keeps every θ under which each user choice is a best option (uniform prior
  over the 624 non-zero θ), picks argmax of E[θ]ᵀφ, and breaks ties at random (seeded by state+menu).
- Accuracy is scored against the paper's labelled choice. 5.3% of held-out (user, menu) pairs are
  ties under θ, and the paper's labels break those at random, so we also report tie-aware accuracy. The oracle knows the user, including how they break ties.
- `data/user_sample.json`: all 624 users in a seeded order stratified by ‖θ‖ into the plan's five
  groups (88/144/88/96/208); any prefix keeps the mix. The main run uses the first 50, the pilot the first 2.
