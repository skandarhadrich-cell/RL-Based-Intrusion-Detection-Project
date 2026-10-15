"""Double DQN agent for the RL-IDS environment.

Vanilla DQN uses the *target* network both to select and to evaluate the greedy
next action, so a single over-optimistic estimate can inflate the TD target
(maximisation bias).  **Double DQN** decouples the two roles:

    a* = argmax_a Q_online(s', a)          # online net *selects* the action
    y  = r + gamma * Q_target(s', a*)      # target net *evaluates* it

The playbook **action mask** is applied at every argmax (both when acting and
when building the bootstrap target), so the agent never selects — and never
bootstraps through — an illegal action.  Masked entries are set to ``-inf``.

The Q-network consumes only the 17-feature observation, so the agent must infer
the hidden severity bucket from telemetry; the mask only constrains legality.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rl_ids import config as C

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# masking helpers
# --------------------------------------------------------------------------- #
def masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    """argmax of ``q`` restricted to legal actions (``mask`` is a bool array)."""
    q = np.asarray(q, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return int(np.argmax(np.where(mask, q, NEG_INF)))


def legal_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def epsilon_schedule(episode: int, start: float = C.DQN_EPS_START,
                     end: float = C.DQN_EPS_END,
                     decay: float = C.DQN_EPS_DECAY) -> float:
    """Per-episode multiplicative ε decay, floored at ``end``."""
    return float(max(end, start * (decay ** int(episode))))


def double_dqn_target(online_next: torch.Tensor, target_next: torch.Tensor,
                      reward: torch.Tensor, done: torch.Tensor,
                      next_mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Double-DQN TD target (pure function, unit-testable).

    ``online_next``/``target_next``: ``[B, A]`` Q-values; ``next_mask``: ``[B, A]``
    bool; ``reward``/``done``: ``[B]``.  Returns ``[B]``.
    """
    masked_online = online_next.masked_fill(~next_mask, NEG_INF)
    a_star = masked_online.argmax(dim=1, keepdim=True)
    boot = target_next.gather(1, a_star).squeeze(1)
    return reward + gamma * boot * (1.0 - done)


# --------------------------------------------------------------------------- #
# network / buffer / agent
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """MLP action-value approximator: obs -> Q(s, .) for every action."""

    def __init__(self, obs_dim: int | None = None, n_actions: int | None = None,
                 hidden: tuple[int, ...] | None = None) -> None:
        super().__init__()
        obs_dim = int(obs_dim or len(C.OBS_FEATURES))
        n_actions = int(n_actions or len(C.ACTIONS))
        hidden = tuple(hidden or C.DQN_HIDDEN)

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer.

    The *next-state* action mask is stored with every transition because the
    Double-DQN target masks the online network's argmax at ``s'``.
    """

    def __init__(self, capacity: int | None = None, obs_dim: int | None = None,
                 n_actions: int | None = None, seed: int | None = None) -> None:
        self.capacity = int(capacity or C.DQN_REPLAY_CAPACITY)
        self.obs_dim = int(obs_dim or len(C.OBS_FEATURES))
        self.n_actions = int(n_actions or len(C.ACTIONS))
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.next_mask = np.zeros((self.capacity, self.n_actions), dtype=bool)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def push(self, obs, action: int, reward: float, next_obs, done: bool,
             next_mask) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = int(action)
        self.reward[i] = float(reward)
        self.done[i] = float(bool(done))
        self.next_mask[i] = next_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "next_mask": self.next_mask[idx],
        }

    def __len__(self) -> int:
        return self._size


class DoubleDQNAgent:
    """Online/target Q-networks, ε-greedy masked acting, and the Double-DQN update."""

    def __init__(self, obs_dim: int | None = None, n_actions: int | None = None,
                 hidden: tuple[int, ...] | None = None, lr: float = C.DQN_LR,
                 gamma: float = C.GAMMA, target_sync: int = C.DQN_TARGET_SYNC,
                 grad_clip: float | None = C.DQN_GRAD_CLIP, device: str = "cpu",
                 seed: int | None = None) -> None:
        self.obs_dim = int(obs_dim or len(C.OBS_FEATURES))
        self.n_actions = int(n_actions or len(C.ACTIONS))
        self.hidden = tuple(hidden or C.DQN_HIDDEN)
        self.gamma = float(gamma)
        self.target_sync = int(target_sync)
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
        self.online = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.updates = 0
        self._rng = np.random.default_rng(seed)

    # ---- inference -------------------------------------------------------- #
    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        """Q-values for a single observation (or batch of them)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.online(x).cpu().numpy()

    def greedy(self, obs, mask) -> int:
        """Masked argmax action (ε = 0)."""
        return masked_argmax(self.q_values(obs)[0], mask)

    def act(self, obs, mask, epsilon: float, rng: np.random.Generator | None = None) -> int:
        """ε-greedy action restricted to the legal set."""
        rng = rng if rng is not None else self._rng
        mask = np.asarray(mask, dtype=bool)
        if float(epsilon) > 0.0 and rng.random() < float(epsilon):
            return int(rng.choice(legal_indices(mask)))
        return self.greedy(obs, mask)

    # ---- learning --------------------------------------------------------- #
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch: dict[str, np.ndarray]) -> float:
        """One gradient step on a sampled batch; returns the loss."""
        obs = torch.as_tensor(batch["obs"], device=self.device)
        action = torch.as_tensor(batch["action"], device=self.device).long()
        reward = torch.as_tensor(batch["reward"], device=self.device).float()
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        done = torch.as_tensor(batch["done"], device=self.device).float()
        next_mask = torch.as_tensor(batch["next_mask"], device=self.device).bool()

        with torch.no_grad():
            target = double_dqn_target(
                self.online(next_obs), self.target(next_obs),
                reward, done, next_mask, self.gamma,
            )

        q_pred = self.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.sync_target()
        return float(loss.item())

    # ---- persistence ------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "gamma": self.gamma,
            "updates": self.updates,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DoubleDQNAgent":
        blob = torch.load(path, map_location=device)
        agent = cls(obs_dim=blob["obs_dim"], n_actions=blob["n_actions"],
                    hidden=tuple(blob["hidden"]), gamma=blob["gamma"], device=device)
        agent.online.load_state_dict(blob["online"])
        agent.sync_target()
        agent.updates = int(blob.get("updates", 0))
        return agent
