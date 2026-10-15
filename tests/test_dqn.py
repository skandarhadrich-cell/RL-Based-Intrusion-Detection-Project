"""Double DQN agent tests.

Everything here is synthetic and tiny: no Parquet/dataset is loaded, so the
suite stays fast and RAM-light (the online training run is the heavy part).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_ids import config as C
from rl_ids.dqn import (
    DoubleDQNAgent,
    QNetwork,
    ReplayBuffer,
    double_dqn_target,
    epsilon_schedule,
    legal_indices,
    masked_argmax,
)

NACT = len(C.ACTIONS)
OBS = len(C.OBS_FEATURES)


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #
def test_masked_argmax_respects_mask():
    q = np.array([0.0, 5.0, 9.0, 1.0, 3.0, 2.0])
    mask = np.array([True, False, False, True, False, True])  # legal {0, 3, 5}
    assert masked_argmax(q, mask) == 5          # max legal value is 2.0 at idx 5
    # without the mask the global argmax (index 2, illegal) would be chosen
    assert int(np.argmax(q)) == 2
    assert masked_argmax(q, mask) != 2


def test_masked_argmax_single_legal():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = np.zeros(NACT, dtype=bool)
    mask[1] = True
    assert masked_argmax(q, mask) == 1


def test_legal_indices():
    mask = np.array([True, False, True, False, False, True])
    assert legal_indices(mask).tolist() == [0, 2, 5]


# --------------------------------------------------------------------------- #
# epsilon schedule
# --------------------------------------------------------------------------- #
def test_epsilon_schedule_monotonic_and_floored():
    eps = [epsilon_schedule(e) for e in range(0, 4000)]
    assert eps[0] == pytest.approx(C.DQN_EPS_START)
    assert all(a >= b for a, b in zip(eps, eps[1:]))       # non-increasing
    assert eps[-1] == pytest.approx(C.DQN_EPS_END, abs=1e-9)
    assert min(eps) >= C.DQN_EPS_END


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def test_network_output_shape():
    net = QNetwork(obs_dim=OBS, n_actions=NACT, hidden=(32, 16))
    out = net(torch.zeros(5, OBS))
    assert out.shape == (5, NACT)


# --------------------------------------------------------------------------- #
# replay buffer
# --------------------------------------------------------------------------- #
def _fake_transition(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=OBS).astype(np.float32)
    nxt = rng.normal(size=OBS).astype(np.float32)
    mask = rng.random(NACT) < 0.5
    mask[0] = True  # always at least one legal action
    return obs, 1, 0.5, nxt, False, mask


def test_replay_buffer_push_len_and_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_dim=OBS, n_actions=NACT, seed=0)
    for i in range(10):
        buf.push(*_fake_transition(i))
    assert len(buf) == 10
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, OBS)
    assert batch["next_obs"].shape == (8, OBS)
    assert batch["action"].shape == (8,)
    assert batch["reward"].shape == (8,)
    assert batch["done"].shape == (8,)
    assert batch["next_mask"].shape == (8, NACT)
    assert batch["next_mask"].dtype == bool


def test_replay_buffer_wraparound_caps_size():
    buf = ReplayBuffer(capacity=5, obs_dim=OBS, n_actions=NACT, seed=0)
    for i in range(13):
        buf.push(*_fake_transition(i))
    assert len(buf) == 5
    # sampling stays in range and shapes hold after wraparound
    assert buf.sample(4)["obs"].shape == (4, OBS)


# --------------------------------------------------------------------------- #
# Double DQN target (the correctness core)
# --------------------------------------------------------------------------- #
def test_double_dqn_target_selects_online_evaluates_target():
    # batch of 2
    online_next = torch.tensor([[1.0, 5.0, 0.0], [4.0, 1.0, 2.0]])
    target_next = torch.tensor([[1.0, 2.0, 3.0], [9.0, 0.5, 0.5]])
    reward = torch.tensor([1.0, -2.0])
    done = torch.tensor([0.0, 0.0])
    mask = torch.ones(2, 3, dtype=bool)
    gamma = 0.9

    y = double_dqn_target(online_next, target_next, reward, done, mask, gamma)

    # row 0: online argmax = 1 -> target[.,1]=2.0 -> 1 + 0.9*2 = 2.8
    # row 1: online argmax = 0 -> target[.,0]=9.0 -> -2 + 0.9*9 = 6.1
    assert y[0].item() == pytest.approx(2.8)
    assert y[1].item() == pytest.approx(6.1)


def test_double_dqn_target_masks_illegal_online_argmax():
    # global online argmax is action 2, but it is illegal -> pick action 1
    online_next = torch.tensor([[0.0, 1.0, 100.0]])
    target_next = torch.tensor([[0.0, 7.0, 999.0]])
    mask = torch.tensor([[True, True, False]])
    y = double_dqn_target(online_next, target_next, torch.tensor([0.0]),
                          torch.tensor([0.0]), mask, 0.9)
    assert y[0].item() == pytest.approx(0.9 * 7.0)


def test_double_dqn_target_terminal_has_no_bootstrap():
    online_next = torch.tensor([[1.0, 2.0, 3.0]])
    target_next = torch.tensor([[10.0, 20.0, 30.0]])
    y = double_dqn_target(online_next, target_next, torch.tensor([5.0]),
                          torch.tensor([1.0]), torch.ones(1, 3, dtype=bool), 0.9)
    assert y[0].item() == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# agent acting
# --------------------------------------------------------------------------- #
def _agent():
    return DoubleDQNAgent(obs_dim=OBS, n_actions=NACT, hidden=(16,), seed=0)


def test_agent_greedy_is_masked_and_deterministic():
    agent = _agent()
    obs = np.zeros(OBS, dtype=np.float32)
    mask = np.zeros(NACT, dtype=bool)
    mask[3] = True  # only Honeypot legal
    assert agent.greedy(obs, mask) == 3
    assert agent.greedy(obs, mask) == 3  # no randomness


def test_agent_act_epsilon_one_is_legal_and_explores():
    agent = _agent()
    obs = np.zeros(OBS, dtype=np.float32)
    mask = np.array([False, True, False, True, False, False])  # legal {1,3}
    rng = np.random.default_rng(0)
    seen = {agent.act(obs, mask, 1.0, rng) for _ in range(50)}
    assert seen <= {1, 3}
    assert len(seen) >= 1


def test_agent_q_values_shape():
    agent = _agent()
    assert agent.q_values(np.zeros(OBS, dtype=np.float32)).shape == (1, NACT)


# --------------------------------------------------------------------------- #
# learning step / target sync
# --------------------------------------------------------------------------- #
def test_update_returns_finite_loss_and_changes_online_not_target():
    agent = _agent()
    buf = ReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=1)
    for i in range(64):
        buf.push(*_fake_transition(i))
    before = [p.detach().clone() for p in agent.online.parameters()]
    target_before = [p.detach().clone() for p in agent.target.parameters()]

    loss = agent.update(buf.sample(16))

    assert np.isfinite(loss)
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters()))
    assert changed
    # the target net is frozen between syncs
    assert all(torch.equal(a, b) for a, b in zip(target_before, agent.target.parameters()))


def test_sync_target_copies_online_weights():
    agent = _agent()
    buf = ReplayBuffer(capacity=16, obs_dim=OBS, n_actions=NACT, seed=2)
    for i in range(16):
        buf.push(*_fake_transition(i))
    agent.update(buf.sample(8))
    agent.sync_target()
    assert all(torch.equal(o, t) for o, t in
               zip(agent.online.parameters(), agent.target.parameters()))


def test_update_reduces_loss_on_fixed_batch():
    """Overfitting a single batch must drive the Huber loss down (sanity)."""
    agent = DoubleDQNAgent(obs_dim=OBS, n_actions=NACT, hidden=(64, 64),
                           lr=1e-3, target_sync=10_000, seed=3)
    buf = ReplayBuffer(capacity=64, obs_dim=OBS, n_actions=NACT, seed=4)
    for i in range(64):
        buf.push(*_fake_transition(i))
    batch = buf.sample(32)

    first = agent.update(batch)
    for _ in range(300):
        last = agent.update(batch)
    assert last < first


# --------------------------------------------------------------------------- #
# save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    agent = DoubleDQNAgent(obs_dim=OBS, n_actions=NACT, hidden=(16,), seed=5)
    obs = np.zeros(OBS, dtype=np.float32)
    mask = np.ones(NACT, dtype=bool)
    before = agent.q_values(obs)

    path = tmp_path / "agent.pt"
    agent.save(path)
    loaded = DoubleDQNAgent.load(path)

    np.testing.assert_allclose(before, loaded.q_values(obs))
    assert loaded.greedy(obs, mask) == agent.greedy(obs, mask)
