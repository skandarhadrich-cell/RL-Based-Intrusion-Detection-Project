"""Day 2: Gymnasium environment for RL-IDS.

The agent sees a real flow from the CSE-CIC-IDS2018 train table - one of the
17 normalized telemetry features, corrupted by Gaussian noise and random
"missing telemetry" masking (partial observability).  The *true* MDP state
(bucket of the hidden attack_label) is NOT part of the observation; it drives
the stochastic transition model and the reward, and is exposed only in ``info``
for evaluation/bootstrapping.

Transitions: (state, action) -> next state sampled from the configured slide
model (config.TRANSITIONS).  Observations are always drawn from real flows
whose hidden state equals the sampled next state, so the agent never sees
synthetic vectors: the telemetry it observes is physically consistent with the
condition it is really in.

Episode semantics:
    - reset(): fresh episode, starts in Normal (hidden), first obs sampled
      from Normal flows.
    - step(): sample next hidden state, compute reward(state, action, next),
      subtract any latency penalty for DPI/Throttle overuse, then observe a
      real flow from the new state.
    - Done when: an incident is *resolved* (S/C -> Normal), the agent
      *Isolates*, or max_steps is reached.

``info["action_mask"]`` marks legal actions per the playbook:
    Normal {Monitor} | Suspicious {Monitor, DPI, Throttle, Honeypot}
    | Compromised {Monitor, Block, Isolate}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from rl_ids import config as C
from rl_ids.reward import reward_function, latency_penalty


def empirical_state_transitions(
    parquet_path: str,
    max_rows_per_day: int | None = 2_000_000,
) -> pd.DataFrame:
    """Consecutive-flow transition matrix (state_t -> state_{t+1}) from real data.

    Grounds the configured slide model in the actual dataset: for each source
    day, sort flows by time and tabulate how often state at row i is followed by
    state at row i+1, aggregated over all train days.  Boundary flow pairs
    between days are excluded (only within-day adjacency counts).
    """
    df = pd.read_parquet(
        parquet_path, columns=["source_day", "TimestampEpoch", "mdp_state"]
    )
    acc: dict[tuple[str, str], int] = {}
    for day, grp in df.groupby("source_day", sort=False):
        grp = grp.sort_values("TimestampEpoch")
        if max_rows_per_day is not None:
            # head + tail keep contiguous real flow sequences (attacks usually
            # mid-day, so take both ends of the day's timeline).
            half = max_rows_per_day // 2
            grp = pd.concat([grp.head(half), grp.tail(max_rows_per_day - half)])
        states = grp["mdp_state"].to_numpy()
        for i in range(len(states) - 1):
            acc[(states[i], states[i + 1])] = acc.get((states[i], states[i + 1]), 0) + 1

    table = pd.DataFrame(0.0, index=list(C.STATES), columns=list(C.STATES))
    rc = {st: sum(v for (a, b), v in acc.items() if a == st) for st in C.STATES}
    for (a, b), n in acc.items():
        if rc[a] > 0:
            table.loc[a, b] = n / rc[a]
        else:
            table.loc[a, b] = 0.0
    return table


class IDSEnv2018(gym.Env):
    """Partially-observable MDP over real IDS2018 flows (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        max_steps: int = C.MAX_STEPS_DEFAULT,
        obs_noise_sigma: float = C.OBS_NOISE_SIGMA,
        obs_mask_prob: float = C.OBS_MASK_PROB,
        max_rows_per_state: int | None = C.DEFAULT_MAX_ROWS_PER_STATE,
        parquet_path: str | None = None,
        seed: int | None = None,
        truncate_obs: int | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "holdout"):
            raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")
        self.split = split
        self.parquet_path = parquet_path or str(
            C.TRAIN_PARQUET if split == "train" else C.HOLDOUT_PARQUET
        )
        self.max_steps = int(max_steps)
        self.obs_noise_sigma = float(obs_noise_sigma)
        self.obs_mask_prob = float(obs_mask_prob)

        # Real-flow pools per hidden state (feature matrix + matching labels).
        self.pools, self.pool_labels = self._build_pools(max_rows_per_state)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(C.OBS_FEATURES),), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(C.ACTIONS))

        self._steps = 0
        self._latency_run = 0
        self._state = "Normal"
        self._last_label: str | None = None
        self._last_row_idx: int | None = None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_pools(self, max_rows: int | None) -> tuple[dict, dict]:
        df = pd.read_parquet(
            self.parquet_path,
            columns=list(C.OBS_FEATURES) + ["mdp_state", "attack_label"],
        )
        pools, labels = {}, {}
        for st in C.STATES:
            sub = df[df["mdp_state"] == st]
            if max_rows is not None:
                sub = sub.head(max_rows)
            pools[st] = sub[C.OBS_FEATURES].to_numpy(dtype=np.float32)
            labels[st] = sub["attack_label"].to_numpy(dtype=object)
        del df
        return pools, labels

    def _sample_obs(self, state: str) -> np.ndarray:
        """Pick a real flow from `state`'s pool and corrupt it (noise+mask)."""
        n = len(self.pools[state])
        idx = int(self._rng.integers(n))
        row = self.pools[state][idx].copy().astype(np.float64)
        row += self._rng.normal(0.0, self.obs_noise_sigma, size=row.shape)
        mask = self._rng.random(row.shape) < self.obs_mask_prob
        row[mask] = 0.0  # missing telemetry observed as neutral (z=0)
        self._last_label = str(self.pool_labels[state][idx])
        self._last_row_idx = idx
        return row.astype(np.float32)

    def _action_mask(self, state: str | None = None) -> np.ndarray:
        st = state if state is not None else self._state
        m = np.zeros(len(C.ACTIONS), dtype=bool)
        for i in C.VALID_ACTIONS[st]:
            m[i] = True
        return m

    def _sample_next_state(self, action_name: str) -> str:
        dist = C.TRANSITIONS[self._state][action_name]
        keys = list(dist.keys())
        probs = np.asarray([dist[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()  # guard against float drift
        return str(self._rng.choice(keys, p=probs))

    def _seed_rng(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_rng(seed)
        start = (options or {}).get("state", "Normal")
        if start not in C.STATES:
            raise ValueError(f"start state must be one of {C.STATES}, got {start!r}")
        self._state = start
        self._steps = 0
        self._latency_run = 0
        obs = self._sample_obs(self._state)
        info = {
            "mdp_state": self._state,
            "attack_label": self._last_label,
            "action_mask": self._action_mask(),
            "latency_run": 0,
            "step": 0,
            "source": ("real_flow", self._last_row_idx),
        }
        return obs, info

    def step(self, action: int):
        a = int(action)
        if not self.action_space.contains(a):
            raise ValueError(f"action {a!r} out of range for {self.action_space}")
        self._steps += 1
        name = C.ACTIONS[a]

        # Latency overuse: consecutive DPI/Throttle actions build a queue.
        self._latency_run = self._latency_run + 1 if name in ("DPI", "Throttle") else 0
        latency_pen = latency_penalty(self._latency_run)

        term_reason = None
        if C.ACTION_INDEX[name] not in C.VALID_ACTIONS[self._state]:
            # Guard rail: masked actions are heavily penalized, no state change.
            reward = float(C.INVALID_ACTION_PENALTY - latency_pen)
            next_state = self._state
            terminated = self._steps >= self.max_steps
            term_reason = "timeout" if terminated else "invalid_action"
        else:
            next_state = self._sample_next_state(name)
            reward = float(reward_function(self._state, name, next_state) - latency_pen)

            resolved = next_state == "Normal" and self._state != "Normal"
            isolated = next_state == C.TERMINAL_MARKER
            terminated = resolved or isolated or self._steps >= self.max_steps
            if isolated:
                term_reason = "isolated"
            elif resolved:
                term_reason = "resolved"
            elif self._steps >= self.max_steps:
                term_reason = "timeout"
            # observation always from the sampled (non-terminal) state
            if next_state == C.TERMINAL_MARKER:
                next_state = self._state

        self._state = next_state
        obs = self._sample_obs(self._state)
        info = {
            "mdp_state": self._state,
            "attack_label": self._last_label,
            "action_mask": self._action_mask(),
            "latency_run": self._latency_run,
            "step": self._steps,
            "terminal_reason": term_reason,
            "source": ("real_flow", self._last_row_idx),
        }
        return obs, reward, terminated, False, info

    def action_mask(self) -> np.ndarray:
        """Expose the playbook mask (used by tabular/DQN policies)."""
        return self._action_mask()

    @property
    def state(self) -> str:
        """True hidden MDP state - for evaluation only, never an observation."""
        return self._state

    def render(self):
        return None