# Delegating preferences

*Replicating Qiu et al. (2026) with newer LLMs, and pricing what it costs to teach an AI agent what you want.*

The write-up is on the blog: **[Delegating preferences](https://blog.oscaradserballe.com/delegating-preferences/)**. In short, Gemma 4 31B roughly doubles the paper's LLM accuracy at learning a user's flight preferences from feedback, but still trails a Bayesian learner, and its built-in default traveller means instruction pays off mostly for principals with unusual tastes.

This repository holds the code, data and every logged LLM call behind it.

## Reproducing it

Every figure and table in the write-up comes from the data and code here. The LLM calls are logged, so every figure and table regenerates offline without an API key.

### Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/download_paper_data.py   # the paper's users and menus, Zenodo 17677329 → data/paper/ (git-ignored)
uv run pytest                                  # 44 tests
```

### Regenerate the figures and tables

```bash
uv run python analysis/delegation/make_figures.py   # figures 1–3, 5–7 and summary.json
uv run python analysis/delegation/user343.py        # figure 4 (the walkthrough user)
uv run python scripts/analyse_results.py results/main
```

LaTeX versions of the tables and welfare definitions are in `analysis/delegation/latex/` (`tables.tex`, `welfare.tex`; rendered to `render.pdf` and PNGs).

### Rerun the experiment

```bash
export OPENROUTER_API_KEY=...
uv run python scripts/run_experiment.py --config configs/main.toml [--max-cost USD] [--push-every SECONDS]
```

Every call is logged under `results/<run_id>/calls/<arm>/<user>.jsonl.gz`, keyed by sha256(model, provider, messages, params). Rerunning the same config reuses the log and only pays for missing calls, so with the committed logs the main run replays for free. Each run writes `manifest.json` and `decisions.parquet` (242,000 decisions for the main run). The main run cost about $23 on OpenRouter.

### Repository layout

| Path | What it holds |
|---|---|
| `src/flight_prefs/` | Paper data loader, utility, Bayesian learner, simulated and LLM agents, prompts, answer parsing, rate-limited OpenRouter client, call log, interaction loop |
| `configs/` | Run configs: `main.toml` (the run reported here), `pilot.toml`, `s2_one_llm.toml`, `probe_friendli.toml` |
| `results/main/` | The main run: decisions, call logs, manifest, analysis tables |
| `results/{simulated,s2-one-llm,pilot,pilot2,probe-friendli}/` | Earlier gates and pilots |
| `analysis/delegation/` | Figures, tables and LaTeX for this write-up |
| `analysis/explore/` | Exploratory data mining of the main run (notes in `NOTES.md`; one-off scripts, kept as run) |
| `analysis/draft/` | The HTML draft this README was written from |
| `scripts/` | Experiment runner, analysis, cost estimate, and the S0–S3 gate checks |

### Build gates

The experiment was built in slices, each with a gate that had to pass before the next.

| Slice | Command | Gate |
|---|---|---|
| S0 | `scripts/check_bayesian_assistant.py` | Bayes 33 → 57 → 67 → 73 → 77 → 80% on all 624 users |
| S1 | `scripts/check_simulated_agents.py` | random ≈ 33%, oracle 100% with regret 0, 605 decisions per user |
| S2 | `run_experiment.py --config configs/s2_one_llm.toml`, then `scripts/check_one_llm.py` | 605 parsed replies, 0 duplicate calls after kill + resume, transcript as the paper's Table H3 |
| S3 | `run_experiment.py --config configs/pilot.toml`, then `analyse_results.py`, `estimate_cost.py`, `check_pilot.py` | 0 unparsed, providers stable, thinking cap binds < 5%, main run projected < $20 |
| S4 | `run_experiment.py --config configs/main.toml` | 50 users × 3 LLM agents + Bayesian learner and simulated agents |

### Method notes

**Models and providers.** All models run through OpenRouter at temperature 0. Qwen 2.5 7B runs on Phala, its only provider. Both Gemma 4 31B arms run on Friendli, whose quantisation is not stated: Crusoe (bf16) was throttled by OpenRouter's shared pool, and Friendli's picks match Crusoe's on 96% of identical prompts in the pilot.

**Thinking cap.** Providers ignore `reasoning.max_tokens`, so the cap is the output limit: 1,056 tokens (1,024 thinking + 32 answer). A call cut off there gets one follow-up with thinking off: the partial thinking as the assistant's turn, then "Stop thinking and answer now. Answer with exactly one sentence of the form 'The best option is Flight N.'". Those picks have `forced = True` in `decisions.parquet`. `scripts/forced_selection.py` checks that the gap between forced and unforced picks is selection (harder items get cut off), not damage from forcing.

**Prompts and parsing.** Wording is from the paper's Supplementary Table H3; the chat layout and the "Your option Flight N is correct." line come from the paper's own Zenodo training dialogues. Past assistant turns are re-sent as exactly "The best option is Flight N.". A pick is read only from an explicit declaration ("The best option is Flight N", "Flight N is the best option", …) or a bare "Flight N". Anything else gets one retry with a format reminder; if that fails too, the pick is scored wrong (with worst-option regret) and reported, never dropped. Qwen needs the reminder for every round-0 prompt: unprompted, it starts an analysis that its 16-token limit cuts off.

**Data.** The interaction file's menus are mapped to features with a per-feature text → value lookup built from the held-out file; it covers every flight and agrees with the file's own `rounds_numpy`. `data/user_sample.json` holds all 624 users in a seeded order, stratified by $`\lVert\theta\rVert`$ into five groups (88/144/88/96/208), so any prefix keeps the mix; the main run uses the first 50.

**Scoring.** Accuracy is scored against the paper's labelled choice. 5.3% of held-out (user, menu) pairs are ties under $`\theta`$ and the paper's labels break those at random, so tie-aware accuracy is reported too. The Bayesian learner keeps every $`\theta`$ under which each user choice is a best option (uniform prior over the 624 non-zero $`\theta`$), picks $`\arg\max\, \mathbb{E}[\theta]^{\top}\varphi`$, and breaks ties at random, seeded by state and menu.

**Principal types.** The unusual/conventional split was defined after seeing the data. A pre-registered run on about 20 new unusual principals would confirm it.
