"""Shared policy-evaluation helpers for the deep-RL drivers.

Both the Double DQN driver and the prioritized-replay driver roll the learned
greedy policy out from each hidden start bucket and compare it against a
uniform-random legal policy and the exact tabular optimum.  Keeping these here
means the DQN and PER reports are produced by identical evaluation code.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from rl_ids import config as C
from rl_ids.dqn import legal_indices

# per-driver defaults (kept in sync with the drivers' own constants)
EVAL_EPISODES = 1_000
RANDOM_EPISODES = 1_000


def rollout(agent, env, start: str, seed: int, episodes: int) -> tuple:
    """Greedy-policy returns from a fixed start state; also counts 1st actions."""
    rets = np.empty(episodes)
    first_actions: Counter[int] = Counter()
    for k in range(episodes):
        obs, info = env.reset(seed=seed + k, options={"state": start})
        mask = info["action_mask"]
        first_actions[agent.greedy(obs, mask)] += 1
        g, disc, done = 0.0, 1.0, False
        while not done:
            a = agent.greedy(obs, mask)
            obs, r, term, trunc, info = env.step(a)
            mask = info["action_mask"]
            g += disc * float(r)
            disc *= C.GAMMA
            done = term or trunc
        rets[k] = g
    dominant = C.ACTIONS[first_actions.most_common(1)[0][0]]
    return float(rets.mean()), float(rets.std()), dominant, first_actions


def evaluate_greedy(agent, env, seed: int, episodes: int = EVAL_EPISODES) -> dict:
    """``{state: (mean, std, dominant_action, first_action_counts)}``."""
    return {s: rollout(agent, env, s, seed, episodes) for s in C.STATES}


def evaluate_random(env, seed: int, episodes: int = RANDOM_EPISODES) -> dict:
    """``{state: (mean, std)}`` for a uniform-random legal policy."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in C.STATES:
        rets = np.empty(episodes)
        for k in range(episodes):
            obs, info = env.reset(seed=seed + k, options={"state": s})
            mask = info["action_mask"]
            g, disc, done = 0.0, 1.0, False
            while not done:
                a = int(rng.choice(legal_indices(mask)))
                obs, r, term, trunc, info = env.step(a)
                mask = info["action_mask"]
                g += disc * float(r)
                disc *= C.GAMMA
                done = term or trunc
            rets[k] = g
        out[s] = (float(rets.mean()), float(rets.std()))
    return out
