# RL Group Project — ICU Sepsis Treatment

NOVA IMS Reinforcement Learning course group project. Trains and compares
tabular and deep RL agents on the `ICU-Sepsis-v2` benchmark from Komorowski
et al. (2018), then extends the standard expected-return objective with a
risk-sensitive (CVaR-style) variant.

The single deliverable is [`rl_sepsis_project.ipynb`](rl_sepsis_project.ipynb).
Generated figures land in [`plots/`](plots/). The assignment brief is in
[`project_description/RL Project.pdf`](project_description/RL%20Project.pdf).

## Setup

Python 3.11 or 3.12. The deps are pure pip (no system packages required).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

End-to-end notebook execution (writes outputs back in place):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 rl_sepsis_project.ipynb
```

Interactive (browser):

```bash
.venv/bin/jupyter notebook rl_sepsis_project.ipynb
```

Quick environment smoke test (Config A + Config B base, ~20 s):

```bash
.venv/bin/python -m envs.continuous_sepsis_env
```

## Notebook structure

| Section | What it covers |
|---|---|
| 0. Setup | Imports, seeds, plot config |
| 1. Explore environment | Discrete + clinical envs, random baselines, failure-mode breakdown |
| 2. Config A (tabular) | Value Iteration, Q-Learning, SARSA, optimality-gap, policy interpretation, exploration-vs-exploitation sweep |
| 3. Config B (function approximation) | Linear-Q, DQN, results table, action heatmaps, SOFA-stratified actions |
| 4. Comparison A vs B | VI-policy-on-clinical-env oracle bound, observation-cost decomposition |
| 5. Creative Extension | Risk-sensitive Q-Learning (Lagrangian death penalty, β-sweep) |

## Reproducibility

All seeds are set in the first cell (`SEED = 42`). Per-algorithm training
seeds are explicit constants (`SEEDS = [0, 1, 2]` for Config A,
`SEEDS_B = [0, 1, 2]` for Config B). Evaluation uses the fixed
`evaluate_policy` and `evaluate_agent` helpers in
[`rl_utils.py`](rl_utils.py) with 1,000 episodes per agent.

Difficulty (`SOFA_BIAS = 5.0`, `LAM = 0.02`) is applied inside
`make_sepsis_env()`; do not reach into the raw icu-sepsis env's
`_r_mat` / `_d_0` directly.

## Compute budget

Full notebook execution takes ~30 min on a modern laptop (mostly DQN
training in Config B). Set `CONFIGB_TOTAL_STEPS=10000` to run a fast smoke
version.

## Layout

```
envs/                            # discrete + continuous envs, clinical wrappers
project_description/             # assignment brief PDF
plots/                           # generated figures
rl_sepsis_project.ipynb  # the deliverable
rl_utils.py                      # shared evaluation + plotting helpers
requirements.txt                 # pinned deps
.claude/                         # project style guides + run skill
```
