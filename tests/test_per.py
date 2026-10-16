"""Prioritized Experience Replay tests (sum-tree, buffer, IS weights, schedule).

All synthetic and tiny - no dataset is loaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_ids import config as C
from rl_ids.dqn import DoubleDQNAgent, ReplayBuffer
from rl_ids.per import PrioritizedReplayBuffer, SumTree, per_beta

NACT = len(C.ACTIONS)
OBS = len(C.OBS_FEATURES)


# --------------------------------------------------------------------------- #
# SumTree
# --------------------------------------------------------------------------- #
def test_sumtree_update_get_total():
    tree = SumTree(capacity=4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(i, p)
    assert tree.total == pytest.approx(10.0)
    assert [tree.get(i) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]


def test_sumtree_update_replaces_and_propagates():
    tree = SumTree(capacity=4)
    for i in range(4):
        tree.update(i, 1.0)
    assert tree.total == pytest.approx(4.0)
    tree.update(2, 5.0)                       # 1,1,5,1 -> total 8
    assert tree.total == pytest.approx(8.0)
    assert tree.get(2) == pytest.approx(5.0)


def test_sumtree_sample_maps_value_to_leaf():
    tree = SumTree(capacity=4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(i, p)
    # cumulative: [1, 3, 6, 10]
    assert tree.sample(0.5) == 0
    assert tree.sample(1.5) == 1
    assert tree.sample(4.0) == 2
    assert tree.sample(7.0) == 3


def test_sumtree_non_power_of_two_capacity():
    tree = SumTree(capacity=5)                # rounds up to 8 leaves
    for i in range(5):
        tree.update(i, float(i + 1))
    assert tree.total == pytest.approx(15.0)
    assert tree.sample(0.5) == 0
    assert tree.sample(14.5) == 4


# --------------------------------------------------------------------------- #
# beta schedule
# --------------------------------------------------------------------------- #
def test_per_beta_anneals_and_clips():
    assert per_beta(0) == pytest.approx(C.PER_BETA_START)
    assert per_beta(C.PER_BETA_STEPS) == pytest.approx(C.PER_BETA_END)
    assert per_beta(10 * C.PER_BETA_STEPS) == pytest.approx(C.PER_BETA_END)
    betas = [per_beta(s) for s in range(0, C.PER_BETA_STEPS, 1000)]
    assert all(a <= b for a, b in zip(betas, betas[1:]))   # non-decreasing


# --------------------------------------------------------------------------- #
# PrioritizedReplayBuffer
# --------------------------------------------------------------------------- #
def _transition(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=OBS).astype(np.float32)
    nxt = rng.normal(size=OBS).astype(np.float32)
    mask = rng.random(NACT) < 0.5
    mask[0] = True
    return obs, 1, float(rng.normal()), nxt, False, mask


def test_per_buffer_is_replay_buffer_subclass_and_grows():
    buf = PrioritizedReplayBuffer(capacity=100, obs_dim=OBS, n_actions=NACT, seed=0)
    assert isinstance(buf, ReplayBuffer)
    for i in range(20):
        buf.push(*_transition(i))
    assert len(buf) == 20


def test_per_buffer_sample_shapes_and_weight_bounds():
    buf = PrioritizedReplayBuffer(capacity=100, obs_dim=OBS, n_actions=NACT, seed=0)
    for i in range(64):
        buf.push(*_transition(i))
    batch = buf.sample(16, beta=0.4)
    assert batch["obs"].shape == (16, OBS)
    assert batch["next_mask"].shape == (16, NACT)
    assert batch["indices"].shape == (16,)
    assert batch["weights"].shape == (16,)
    assert batch["weights"].max() == pytest.approx(1.0)
    assert np.all(batch["weights"] > 0.0)
    assert np.all(batch["indices"] >= 0) and np.all(batch["indices"] < 64)


def test_per_uniform_priorities_give_unit_is_weights():
    buf = PrioritizedReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=0)
    for i in range(64):
        buf.push(*_transition(i))
    # every transition has the same (max) priority -> uniform P -> weights all 1
    batch = buf.sample(32, beta=1.0)
    assert np.allclose(batch["weights"], 1.0)


def test_new_transitions_get_max_priority():
    buf = PrioritizedReplayBuffer(capacity=100, obs_dim=OBS, n_actions=NACT,
                                  alpha=0.6, seed=0)
    for i in range(5):
        buf.push(*_transition(i))
    buf.update_priorities(np.array([2]), np.array([100.0]))
    max_p = abs(100.0) + C.PER_EPS
    assert buf._tree.get(2) == pytest.approx(max_p ** 0.6)
    buf.push(*_transition(99))                # lands at slot 5
    assert buf._tree.get(5) == pytest.approx(max_p ** 0.6)


def test_per_update_priorities_steers_sampling():
    buf = PrioritizedReplayBuffer(capacity=200, obs_dim=OBS, n_actions=NACT,
                                  alpha=0.6, seed=0)
    for i in range(200):
        buf.push(*_transition(i))
    # starve everything except index 7
    idx = np.arange(200)
    td = np.full(200, 1e-4)
    td[7] = 1000.0
    buf.update_priorities(idx, td)

    counts = np.zeros(200, dtype=int)
    for _ in range(400):
        counts[buf.sample(1)["indices"][0]] += 1
    assert counts[7] > counts.sum() * 0.5      # the boosted transition dominates


def test_per_wraparound_keeps_sampling_valid():
    buf = PrioritizedReplayBuffer(capacity=8, obs_dim=OBS, n_actions=NACT, seed=0)
    for i in range(23):                        # wrap several times
        buf.push(*_transition(i))
    assert len(buf) == 8
    batch = buf.sample(8, beta=0.6)
    assert np.all(batch["indices"] >= 0) and np.all(batch["indices"] < 8)


# --------------------------------------------------------------------------- #
# agent learning with IS weights
# --------------------------------------------------------------------------- #
def _agent():
    return DoubleDQNAgent(obs_dim=OBS, n_actions=NACT, hidden=(32,), target_sync=10_000, seed=0)


def _batch(buf, size=16, beta=0.4):
    return buf.sample(size, beta=beta)


def test_learn_returns_loss_and_td_errors_of_right_shape():
    agent = _agent()
    buf = PrioritizedReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=1)
    for i in range(64):
        buf.push(*_transition(i))
    batch = _batch(buf)
    loss, td = agent.learn(batch, weights=batch["weights"])
    assert np.isfinite(loss)
    assert td.shape == (16,)
    assert np.all(np.isfinite(td))


def test_update_still_returns_float_for_backward_compat():
    agent = _agent()
    buf = ReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=2)
    for i in range(64):
        buf.push(*_transition(i))
    loss = agent.update(buf.sample(16))
    assert isinstance(loss, float) and np.isfinite(loss)


def test_weights_change_the_loss():
    agent = _agent()
    buf = PrioritizedReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=3)
    for i in range(64):
        buf.push(*_transition(i))
    batch = _batch(buf)
    # two learners with identical weights so we can isolate the weight effect
    a1, a2 = _agent(), _agent()
    a2.online.load_state_dict(a1.online.state_dict())
    loss_uniform, _ = a1.learn(batch, weights=np.ones(16))
    skewed = np.zeros(16)
    skewed[0] = 1.0
    loss_skewed, _ = a2.learn(batch, weights=skewed)
    assert not np.isclose(loss_uniform, loss_skewed)
