"""
rl_utils.py - Shared helpers for the RL Sepsis project
======================================================
True helper functions extracted from rl_sepsis_project.ipynb: seeding,
evaluation, statistics, learning-curve smoothing, plotting, and policy
interpretability. Algorithm bodies (Q-learning, SARSA, value/policy
iteration, linear-Q, SB3 training) stay in the notebook.

Imports the environment factories from envs/ but never modifies them.
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from envs.env_setup import N_ACTIONS, make_sepsis_env
from envs.wrappers import make_clinical_env

# Mirrors the notebook's first-cell SEED; defined here so the seed=SEED
# default arguments below bind to 42 when this module is imported.
SEED = 42


# Seeding

def set_global_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


# Environment construction

def make_clinical_env_quiet():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        env = make_clinical_env()
    return env


# Evaluation

def evaluate_policy(pi, n_episodes=1000, seed=SEED):
    """Monte-Carlo evaluation of a deterministic policy pi (shape (N_STATES,))
    on the fixed 1000-episode seed set. Returns (returns, lengths) arrays."""
    env_eval = make_sepsis_env(quiet=True)
    np.random.seed(seed)
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs, _ = env_eval.reset(seed=np.random.randint(100_000))
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action = int(pi[obs])
            obs, r, te, tr, _ = env_eval.step(action)
            total_r += r
            steps += 1
            done = te or tr
        returns.append(total_r)
        lengths.append(steps)
    env_eval.close()
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    return returns_arr, lengths_arr


def evaluate_agent(predict_fn, n_episodes=1000, seed=SEED):
    'Fixed-seed evaluation of any agent (predict_fn maps obs -> action) on the clinical env.'
    env_eval = make_clinical_env_quiet()
    np.random.seed(seed)
    returns = []
    lengths = []
    noisy_flags = []
    missing_flags = []
    acute_flags = []
    sofa_starts = []
    for _ in range(n_episodes):
        obs, info = env_eval.reset(seed=np.random.randint(100_000))
        ep_noisy = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_sofa = float(info.get('sofa_score', np.nan))
        ep_acute = False
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action = predict_fn(obs)
            obs, r, te, tr, info = env_eval.step(action)
            total_r += r
            steps += 1
            done = te or tr
            if info.get('acute_event', False):
                ep_acute = True
        returns.append(total_r)
        lengths.append(steps)
        noisy_flags.append(ep_noisy)
        missing_flags.append(ep_missing)
        acute_flags.append(ep_acute)
        sofa_starts.append(ep_sofa)
    env_eval.close()
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    eval_info = {
        'noisy': np.array(noisy_flags),
        'missing': np.array(missing_flags),
        'acute': np.array(acute_flags),
        'sofa_start': np.array(sofa_starts),
    }
    return returns_arr, lengths_arr, eval_info


def evaluate_tabular_on_clinical(pi, n_episodes=1000, seed=SEED):
    env_eval = make_clinical_env_quiet()
    base = env_eval.unwrapped
    np.random.seed(seed)
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs, info = env_eval.reset(seed=np.random.randint(100_000))
        state = int(base._raw._current_state)
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action = int(pi[state])
            obs, r, te, tr, info = env_eval.step(action)
            state = int(base._raw._current_state)
            total_r += r
            steps += 1
            done = te or tr
        returns.append(total_r)
        lengths.append(steps)
    env_eval.close()
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    return returns_arr, lengths_arr


def run_random_baseline(n_episodes=1000, seed=SEED):
    np.random.seed(seed)
    env_eval = make_sepsis_env(quiet=True)
    env_eval.action_space.seed(seed)
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs, _ = env_eval.reset(seed=np.random.randint(100_000))
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action = env_eval.action_space.sample()
            obs, r, te, tr, _ = env_eval.step(action)
            total_r += r
            steps += 1
            done = te or tr
        returns.append(total_r)
        lengths.append(steps)
    env_eval.close()
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    return returns_arr, lengths_arr


def random_predict(obs):
    action = int(np.random.randint(N_ACTIONS))
    return action


# Statistics

def subset_survival(survived, condition):
    if condition.sum() == 0:
        rate = np.nan
    else:
        rate = float(survived[condition].mean()) * 100
    return rate


def bootstrap_survival_ci(returns_arr, n_resample=1000, ci=0.95, seed=SEED):
    rng_bs = np.random.default_rng(seed)
    n = len(returns_arr)
    survived_mask = returns_arr > 0
    boot_means = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng_bs.integers(0, n, size=n)
        boot_means[i] = survived_mask[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha)) * 100
    hi = float(np.quantile(boot_means, 1.0 - alpha)) * 100
    return lo, hi


# Learning-curve utilities

def rolling_mean(arr, window):
    if len(arr) < window:
        return arr.copy()
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    result = (cumsum[window:] - cumsum[:-window]) / window
    return result


def mean_curve(runs):
    curve_list = []
    for run in runs:
        curve_list.append(run['snapshot_returns'])
    stack = np.vstack(curve_list)
    out = stack.mean(axis=0)
    return out


def steps_to_threshold(steps, curve, threshold):
    reached = np.where(curve >= threshold)[0]
    if len(reached) == 0:
        result = np.nan
    else:
        first_pos = int(reached[0])
        result = int(steps[first_pos])
    return result


def fmt_conv(steps_value):
    if np.isnan(steps_value):
        text = 'not reached'
    else:
        text = f'{int(steps_value):,} steps'
    return text


def best_run(runs):
    final_list = []
    for run in runs:
        final_list.append(float(np.max(run['snapshot_returns'])))
    best_idx = int(np.argmax(np.array(final_list)))
    chosen = runs[best_idx]
    return chosen


# Plotting

def plot_learning_curve(runs, color, label, fname, baseline, plots_dir, show_eps=True):
    steps = runs[0]['snapshot_steps']
    curve_list = []
    for run in runs:
        curve_list.append(run['snapshot_returns'])
    curve_stack = np.vstack(curve_list)
    curve_mean = curve_stack.mean(axis=0)
    curve_std = curve_stack.std(axis=0)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(steps, curve_mean, color=color, linewidth=2.0,
             label=f'{label} (mean of {len(runs)} seeds)')
    band_lo = curve_mean - curve_std
    band_hi = curve_mean + curve_std
    ax1.fill_between(steps, band_lo, band_hi, color=color, alpha=0.2, label='+/-1 std across seeds')
    ax1.axhline(baseline, color='gray', linestyle=':', linewidth=1.5,
                label=f'Random baseline ({baseline:.3f})')
    ax1.set_xlabel('Environment steps')
    ax1.set_ylabel('Eval return (200 episodes)')
    ax1.set_title(f'{label} learning curve - Config B', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    eps_trace = runs[0]['eps_trace']
    eps_valid = np.isfinite(eps_trace).any()
    if show_eps and eps_valid:
        ax2 = ax1.twinx()
        ax2.plot(steps, eps_trace, color='tomato', linewidth=1.2, alpha=0.7, label='epsilon')
        ax2.set_ylabel('epsilon', color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{fname}', bbox_inches='tight')
    plt.show()


# Policy interpretability

def policy_action_counts(pi, n_episodes=500, seed=SEED):
    env_p = make_sepsis_env(quiet=True)
    counts = np.zeros((5, 5), dtype=float)
    rng_local = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = env_p.reset(seed=int(rng_local.integers(0, 100_000)))
        done = False
        while not done:
            action = int(pi[obs])
            vaso = action // 5
            fluid = action % 5
            counts[vaso, fluid] += 1
            obs, _, te, tr, _ = env_p.step(action)
            done = te or tr
    env_p.close()
    return counts


def policy_action_grid(predict_fn, n_episodes=500, seed=SEED):
    env_eval = make_clinical_env_quiet()
    np.random.seed(seed)
    counts = np.zeros(N_ACTIONS)
    for _ in range(n_episodes):
        obs, info = env_eval.reset(seed=np.random.randint(100_000))
        done = False
        while not done:
            action = predict_fn(obs)
            counts[action] += 1
            obs, r, te, tr, info = env_eval.step(action)
            done = te or tr
    env_eval.close()
    total = counts.sum()
    grid = (counts / total).reshape(5, 5)
    return grid


def action_by_sofa(predict_fn, n_episodes=800, seed=SEED):
    env_eval = make_clinical_env_quiet()
    np.random.seed(seed)
    sofa_starts = []
    vaso_means = []
    fluid_means = []
    for _ in range(n_episodes):
        obs, info = env_eval.reset(seed=np.random.randint(100_000))
        sofa0 = float(info.get('sofa_score', np.nan))
        ep_vaso = []
        ep_fluid = []
        done = False
        while not done:
            action = predict_fn(obs)
            ep_vaso.append(action // 5)
            ep_fluid.append(action % 5)
            obs, r, te, tr, info = env_eval.step(action)
            done = te or tr
        sofa_starts.append(sofa0)
        vaso_means.append(float(np.mean(ep_vaso)))
        fluid_means.append(float(np.mean(ep_fluid)))
    out = {
        'sofa': np.array(sofa_starts),
        'vaso': np.array(vaso_means),
        'fluid': np.array(fluid_means),
    }
    env_eval.close()
    return out
