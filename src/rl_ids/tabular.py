"""Day 4: First-Visit Constant-α GLIE Monte Carlo control (tabular world).

Validates the Day-2 environment / Day-3 reward on the 3-state tabular version
of the problem BEFORE any neural net is involved.  The agent's "state" is the
true hidden bucket (info["mdp_state"]) - the tabular sanity check assumes
perfect state perception so that any discrepancy is blamed on the env/reward
model, exactly as the plan's pitfall intends.

    Q(s,a) <- Q(s,a) + alpha * (G_t - Q(s,a))      [first visit of (s,a)]
    eps_t  <- 1 / (1 + t / d)                       [GLIE, eps -> 0]

Also provides an exact Bellman back-up (value iteration) over the configured
transition kernel, so the MC result can be compared against the closed-form
solution instead of only the slide targets.
"""
from __future__ import annotations

import numpy as np

from rl_ids import config as C
from rl_ids.reward import reward_function

SIDX: dict[str, int] = {s: i for i, s in enumerate(C.STATES)}
NACT: int = len(C.ACTIONS)


def legal_mask(state: str) -> np.ndarray:
    mask = np.zeros(NACT, dtype=bool)
    mask[list(C.VALID_ACTIONS[state])] = True
    return mask


def epsilon_glie(episode: int, divisor: int = 1000) -> float:
    """GLIE schedule: eps -> 0 as episodes -> inf.

    ``divisor`` stretches the decay so a finite run actually explores enough to
    visit every action many times (1/(1+episode) alone only yields ~log(n)
    exploratory steps in n episodes - the harmonic series - which is too slow
    for a finite sanity check).
    """
    return 1.0 / (1.0 + episode / float(divisor))


def egreedy_action(Q: np.ndarray, state: str, eps: float, rng) -> int:
    mask = legal_mask(state)
    if rng.random() < eps:
        return int(rng.choice(np.flatnonzero(mask)))
    return int(greedy_actions(Q, state)[0])


def greedy_actions(Q: np.ndarray, state: str) -> list[int]:
    mask = legal_mask(state)
    q = Q[SIDX[state]].copy()
    q[~mask] = -np.inf
    best = float(np.max(q))
    return [int(i) for i in np.flatnonzero(q == best)]


def first_visit_returns(
    trace: list[tuple[str, int, float]], gamma: float = C.GAMMA
) -> dict[tuple[str, int], float]:
    """Return-at-first-visit G for every (state, action) of an episode trace.

    First-visit means the *first* occurrence of (s, a) counting from the start
    of the episode.  Computing this by walking the trace backwards while marking
    `seen` and updating on the first miss silently targets the *last* occurrence
    instead - the one whose return is least discounted - and biases every
    estimate upward.  So all returns are computed up-front and the forward pass
    only records first visits.
    """
    returns = np.empty(len(trace))
    g = 0.0
    for t in range(len(trace) - 1, -1, -1):
        g = trace[t][2] + gamma * g
        returns[t] = g
    out: dict[tuple[str, int], float] = {}
    seen: set[tuple[str, int]] = set()
    for t, (ss, aa, _) in enumerate(trace):
        if (ss, aa) in seen:
            continue
        seen.add((ss, aa))
        out[(ss, aa)] = float(returns[t])
    return out


def mc_control(
    env,
    episodes: int,
    alpha: float = 0.05,
    gamma: float = C.GAMMA,
    eps_schedule=epsilon_glie,
    seed: int | None = None,
    starts: list[str] | None = None,
    track_every: int = 1000,
    q0: float = 0.0,
):
    """First-visit constant-α GLIE MC control with exploring starts.

    ``q0`` applies *optimistic initialization* to every legal (state, action)
    pair (standard MC control practice: it forces early greedy visits to every
    action so that slow-to-improve arms such as Block are still explored before
    ε has decayed).  Illegal entries stay at 0 and are never selected.

    Returns (Q, track) where track holds pristine Q snapshots at milestones and
    the per-episode epsilon used, for learning-curve reporting.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((len(C.STATES), NACT))
    if q0 > 0.0:
        for i, s in enumerate(C.STATES):
            Q[i, C.VALID_ACTIONS[s]] = q0
    track = {"episode": [], "Q": [], "eps": []}

    for ep in range(episodes):
        if starts is not None:
            start = starts[ep % len(starts)]
        else:
            start = C.STATES[int(rng.integers(len(C.STATES)))]
        _, info = env.reset(seed=(seed or 0) + ep, options={"state": start})
        eps = float(eps_schedule(ep))

        trace: list[tuple[str, int, float]] = []
        done = False
        s = info["mdp_state"]
        while not done:
            a = egreedy_action(Q, s, eps, rng)
            _, r, term, trunc, info = env.step(a)
            trace.append((s, a, float(r)))
            s = info["mdp_state"]
            done = term or trunc

        # First-visit constant-α update (see first_visit_returns for why the
        # naive reversed-trace `seen` approach is wrong).
        for (ss, aa), ret in first_visit_returns(trace, gamma).items():
            Q[SIDX[ss], aa] += alpha * (ret - Q[SIDX[ss], aa])

        if ep % track_every == 0 or ep == episodes - 1:
            track["episode"].append(ep)
            track["Q"].append(Q.copy())
            track["eps"].append(eps)
    return Q, track


def bellman_solution(gamma: float = C.GAMMA, tol: float = 1e-9,
                     max_iter: int = 50_000) -> np.ndarray:
    """Exact q* by value iteration over the configured transition kernel.

    Termination matches the env: a step is terminal when it lands on Normal
    from a non-Normal state ("resolved"), or on the Isolate marker.
    """
    Q = np.zeros((len(C.STATES), NACT))
    for _ in range(max_iter):
        V = np.array([
            float(np.max(Q[i, legal_mask(s)])) for i, s in enumerate(C.STATES)
        ])
        delta = 0.0
        for i, s in enumerate(C.STATES):
            for a in range(NACT):
                if a not in C.VALID_ACTIONS[s]:
                    continue
                q = 0.0
                for nxt, p in C.TRANSITIONS[s][C.ACTIONS[a]].items():
                    if nxt == C.TERMINAL_MARKER:
                        q += p * reward_function(s, a, None)
                    else:
                        j = SIDX[nxt]
                        # resolved transition ends the episode (no V(Normal))
                        cont = 0.0 if (nxt == "Normal" and s != "Normal") else V[j]
                        q += p * (reward_function(s, a, nxt) + gamma * cont)
                delta = max(delta, abs(Q[i, a] - q))
                Q[i, a] = q
        if delta < tol:
            break
    return Q


def evaluate_greedy(
    env,
    Q: np.ndarray,
    start: str,
    episodes: int = 2000,
    seed: int | None = None,
    gamma: float = C.GAMMA,
) -> tuple[float, float]:
    """Mean +/- std discounted return of the greedy (eps=0) policy from `start`."""
    rng = np.random.default_rng(seed)
    returns = np.empty(int(episodes))
    for ep in range(int(episodes)):
        _, info = env.reset(seed=(seed or 0) + ep, options={"state": start})
        G, disc, done = 0.0, 1.0, False
        while not done:
            a = egreedy_action(Q, info["mdp_state"], 0.0, rng)
            _, r, term, trunc, info = env.step(a)
            G += disc * float(r)
            disc *= gamma
            done = term or trunc
        returns[ep] = G
    return float(returns.mean()), float(returns.std())


def greedy_policy_table(Q: np.ndarray) -> dict[str, list[str]]:
    return {s: [C.ACTIONS[a] for a in greedy_actions(Q, s)] for s in C.STATES}