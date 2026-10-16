"""Prioritized Experience Replay (proportional) for the RL-IDS agent.

Uniform replay revisits every transition equally often, so a rare but
informative transition (e.g. a Suspicious flow that resolves) is drowned out by
millions of easy Normal ones.  **Prioritized Experience Replay** (Schaul et al.
2015) instead samples transitions in proportion to their last TD error:

    P(i) = p_i^alpha / sum_k p_k^alpha,        p_i = |delta_i| + eps   (alpha=0.6)

Sampling by ``P(i)`` biases the update; the bias is corrected with
importance-sampling (IS) weights that are annealed in over training:

    w_i = ( N * P(i) )^(-beta) / max_j ( N * P(j) )^(-beta),   beta: 0.4 -> 1.0

``PrioritizedReplayBuffer`` reuses the storage of :class:`rl_ids.dqn.ReplayBuffer`
and adds a :class:`SumTree` so sampling and priority updates are both O(log N).
New transitions enter at the current maximum priority so nothing is starved.
"""
from __future__ import annotations

import numpy as np

from rl_ids import config as C
from rl_ids.dqn import ReplayBuffer


class SumTree:
    """Binary tree of partial sums over leaf priorities.

    Leaves live at ``[tree_size : tree_size + capacity]``; internal node ``i``
    holds the sum of its two children, so ``tree[1]`` is the total.  Sampling a
    uniform value in ``(0, total]`` walks down the tree to the owning leaf in
    O(log N), keeping proportional sampling cheap regardless of capacity.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        size = 1
        while size < self.capacity:
            size <<= 1
        self.tree_size = size
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, index: int) -> float:
        return float(self.tree[index + self.tree_size])

    def update(self, index: int, priority: float) -> None:
        pos = index + self.tree_size
        delta = float(priority) - self.tree[pos]
        self.tree[pos] = float(priority)
        pos //= 2
        while pos >= 1:
            self.tree[pos] += delta
            pos //= 2

    def sample(self, value: float) -> int:
        """Return the data index of the leaf that ``value`` falls into."""
        pos = 1
        while pos < self.tree_size:
            left = 2 * pos
            if value <= self.tree[left]:
                pos = left
            else:
                value -= self.tree[left]
                pos = left + 1
        return pos - self.tree_size


def per_beta(step: int, start: float = C.PER_BETA_START,
             end: float = C.PER_BETA_END,
             steps: int = C.PER_BETA_STEPS) -> float:
    """Linear beta-annealing schedule (0.4 -> 1.0 by ``steps``), clipped at 1."""
    frac = min(1.0, max(0.0, step / float(steps)))
    return float(start + frac * (end - start))


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay on top of the uniform buffer's storage."""

    def __init__(self, capacity: int | None = None, obs_dim: int | None = None,
                 n_actions: int | None = None, alpha: float = C.PER_ALPHA,
                 seed: int | None = None) -> None:
        super().__init__(capacity, obs_dim, n_actions, seed)
        self.alpha = float(alpha)
        self._tree = SumTree(self.capacity)
        self._max_priority = 1.0

    def push(self, obs, action: int, reward: float, next_obs, done: bool,
             next_mask) -> None:
        super().push(obs, action, reward, next_obs, done, next_mask)
        # the slot just written (wrap-safe), seeded with the running max priority
        slot = (self._ptr - 1) % self.capacity
        self._tree.update(slot, self._max_priority ** self.alpha)

    def sample(self, batch_size: int, beta: float = C.PER_BETA_START) -> dict[str, np.ndarray]:
        """Stratified proportional sample; adds ``weights`` and ``indices``."""
        n = int(batch_size)
        idx = np.empty(n, dtype=np.int64)
        priorities = np.empty(n, dtype=np.float64)
        total = self._tree.total
        segment = total / n
        for k in range(n):
            value = self._rng.uniform(segment * k, segment * (k + 1))
            if value <= 0.0:
                value = np.finfo(np.float64).tiny
            i = int(self._tree.sample(value))
            idx[k] = i
            priorities[k] = self._tree.get(i)

        probs = priorities / total
        weights = (self._size * probs) ** (-float(beta))
        weights /= weights.max()  # normalise so the largest weight is 1
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "next_mask": self.next_mask[idx],
            "weights": weights.astype(np.float32),
            "indices": idx,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Refresh priorities to ``(|td_error| + eps)^alpha`` after an update."""
        prios = np.abs(np.asarray(td_errors, dtype=np.float64)).reshape(-1) + C.PER_EPS
        for i, p in zip(np.asarray(indices).reshape(-1), prios):
            self._tree.update(int(i), float(p) ** self.alpha)
            if p > self._max_priority:
                self._max_priority = float(p)
