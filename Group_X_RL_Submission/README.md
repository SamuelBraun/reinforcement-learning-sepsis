# Reinforcement Learning Project — Group X

**Course:** Reinforcement Learning 2025/2026 — NOVA Information Management School
**Topic:** RL for Sepsis Treatment in the ICU — Tabular and Deep Methods on ICU-Sepsis-v2

## Group members

| Name | Student ID |
|---|---|
| Carolina Luz | 20250409 |
| Lukas Belser | 20250338 |
| Margarida Quintino | 20250411 |
| Pal Harnos | 20250487 |
| Samuel Braun | 20250355 |

## What is in this folder

```
Group_X_RL_Submission/
├── README.md                       # this file
├── Group_X_RL_Report.pdf           # main deliverable: 15-page report
├── rl_sepsis_project.ipynb         # main notebook (all results reproducible)
├── requirements.txt                # Python dependencies
├── rl_utils.py                     # shared evaluation + plotting helpers
├── envs/                           # custom ICU-Sepsis environments
│   ├── __init__.py
│   ├── env_setup.py                # make_sepsis_env factory (Config A)
│   ├── continuous_sepsis_env.py    # 47-dim continuous wrapper (Config B base)
│   └── wrappers.py                 # three clinical-reality wrappers
├── plots/                          # all figures (PNG, embedded in notebook + report)
├── RL_DEFENSE_CHEATSHEET.pdf       # supplementary 45-page study guide
└── project_description/            # original assignment brief (for reference)
    └── RL Project.pdf
```

## How to reproduce the results

Python 3.11 (or 3.12). No system packages required.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then either run interactively:

```bash
.venv/bin/jupyter notebook rl_sepsis_project.ipynb
```

Or end-to-end (headless, writes outputs back in place):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=4800 rl_sepsis_project.ipynb
```

Full execution takes approximately 30–40 minutes on a modern laptop. All
seeds are fixed in the first cell (`SEED = 42`, per-algorithm seeds
explicit), so the numbers in `Group_X_RL_Report.pdf` are reproducible.

## Notebook structure

| Section | What it covers |
|---|---|
| 0. Setup | Imports, seeds, plot config |
| 1. Explore the environment | Discrete + clinical envs, random baselines, failure-mode breakdown |
| 2. Configuration A (tabular) | Value Iteration, Q-Learning, SARSA, optimality-gap, policy interpretation, exploration-vs-exploitation sweep |
| 3. Configuration B (function approximation) | Linear-Q (raw + RFF), tuned DQN, results table, action heatmaps, SOFA-stratified actions |
| 4. Comparison Config A vs B | Bridge experiment with three-way decomposition of the Configuration B gap |
| 5. Creative extension | Risk-sensitive Q-Learning (Lagrangian death penalty, β-sweep across 5 seeds) |

## Headline results

| Agent | Survival | Notes |
|---|---|---|
| Random (Config A clean) | 67.5% | floor |
| Q-Learning (200k × 3 seeds) | 76.1% | pooled |
| SARSA (200k × 3 seeds) | 74.3% | indistinguishable from QL |
| Value Iteration | **78.8%** | model-based ceiling |
| Linear-Q (raw) | 66.6% | collapses to action 0 |
| Linear-Q (RFF) | 66.7% | still collapses (96% on action 0) |
| Tuned DQN | 68.1% | SOFA-responsive |
| **Risk-sensitive QL (β=0.5)** | **75.5%** | Pareto sweet spot |

Bridge experiment three-way decomposition (10.7 pp total gap from VI clean to tuned DQN):
- 0.3 pp wrapper cost (acute events on VI)
- 6.2 pp algorithm-quality gap (VI tabular vs Q-Learning tabular on clinical)
- 4.2 pp observation-regime cost (Q-Learning tabular vs DQN function approximation)

## Notes for the grader

- The fixed difficulty (`SOFA_BIAS = 5.0`, `LAM = 0.02`) is applied inside
  `envs.env_setup.make_sepsis_env()` and matches the project specification.
- All evaluation uses the fixed protocol from `rl_utils.evaluate_policy`
  and `rl_utils.evaluate_agent` (1000 episodes per agent, seed 42).
- The cheatsheet (`RL_DEFENSE_CHEATSHEET.pdf`) is supplementary
  documentation that walks every concept from first principles. It is
  not required for grading but is included as "additional documentation"
  per the submission guidelines.
