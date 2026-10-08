"""Day 2 checkpoint tests: the Gymnasium environment.

Checkpoint (from the plan):
  a working gym.Env with the MDP state/action structure, obs in Box(n_obs),
  action masks per playbook, stochastic transitions, terminal conditions,
  partial observability (noise+masking) and data-driven observations sampled
  from real flows conditioned on the hidden state.
"""
import numpy as np
import pytest

from rl_ids import config as C
from rl_ids.env import IDSEnv2018, empirical_state_transitions


@pytest.fixture(scope="session")
def env() -> IDSEnv2018:
    return IDSEnv2018(
        split="train",
        max_rows_per_state=500,   # small pools: tests stay fast + RAM-safe
        max_steps=20,
        seed=7,
    )


# --------------------------------------------------------------------------- #
# 1. Space structure: Box(n_obs) observation, Discrete(6) action
# --------------------------------------------------------------------------- #
def test_space_structure(env):
    assert env.action_space.n == len(C.ACTIONS)
    assert env.observation_space.shape == (len(C.OBS_FEATURES),)
    assert env.observation_space.contains(env.observation_space.sample())


def test_action_discrete_size():
    env = IDSEnv2018(max_rows_per_state=1, max_steps=2, seed=0)
    assert env.action_space.n == len(C.ACTIONS) == 6


# --------------------------------------------------------------------------- #
# 2. Playbook masks per state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state,expected",
    [
        ("Normal", {"Monitor"}),
        ("Suspicious", {"Monitor", "DPI", "Throttle", "Honeypot"}),
        ("Compromised", {"Monitor", "Block", "Isolate"}),
    ],
)
def test_action_masks(state, expected, env):
    mask = env._action_mask(state)
    legal = {C.ACTIONS[i] for i in np.flatnonzero(mask)}
    assert legal == expected
    # and matches the config table's per-state playbook (indices)
    assert set(np.flatnonzero(mask)) == set(C.VALID_ACTIONS[state])


# --------------------------------------------------------------------------- #
# 3. reset() starts a Normal episode with a valid Box observation
# --------------------------------------------------------------------------- #
def test_reset_returns_normal_obs_and_mask(env):
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert info["mdp_state"] == "Normal"
    assert info["step"] == 0
    assert info["latency_run"] == 0
    assert set(np.flatnonzero(info["action_mask"])) == set(C.VALID_ACTIONS["Normal"])


def test_reset_can_start_from_any_state(env):
    for st in C.STATES:
        obs, info = env.reset(seed=0, options={"state": st})
        assert info["mdp_state"] == st
        assert env.observation_space.contains(obs)


# --------------------------------------------------------------------------- #
# 4. Observations are REAL dataset flows (noised/masked), sampled per hidden state
# --------------------------------------------------------------------------- #
def test_obs_corruption_noise_and_mask():
    """With a strong corruptor the observation diverges from its raw pool row;
    with it disabled it equals the raw row exactly."""
    noisy = IDSEnv2018(max_rows_per_state=50, max_steps=3,
                       obs_noise_sigma=0.5, obs_mask_prob=0.9, seed=3)
    obs, info = noisy.reset()
    idx = info["source"][1]
    raw = noisy.pools["Normal"][idx]
    assert not np.allclose(obs, raw), "expected noise/masking to alter the obs"

    clean = IDSEnv2018(max_rows_per_state=50, max_steps=3,
                       obs_noise_sigma=0.0, obs_mask_prob=0.0, seed=3)
    obs2, info2 = clean.reset()
    idx2 = info2["source"][1]
    assert np.array_equal(obs2, clean.pools["Normal"][idx2])


def test_observations_drawn_from_target_state_pool():
    """After a transition, the next obs row must live in the pool of the hidden
    state recorded in info (real data, no synthetic vectors)."""
    env = IDSEnv2018(max_rows_per_state=200, max_steps=5,
                     obs_noise_sigma=0.0, obs_mask_prob=0.0, seed=11)
    obs, info = env.reset(seed=11, options={"state": "Suspicious"})
    acts = [C.ACTION_INDEX["Honeypot"], C.ACTION_INDEX["Monitor"],
            C.ACTION_INDEX["DPI"], C.ACTION_INDEX["Throttle"]]
    for a in acts:
        obs, r, term, trunc, info = env.step(a)
        assert info["source"][0] == "real_flow"
        ridx = info["source"][1]
        pool = env.pools[info["mdp_state"]]
        assert 0 <= ridx < len(pool), "obs index outside state pool"
        assert np.array_equal(obs, pool[ridx]), "obs must be a real pool row"


# --------------------------------------------------------------------------- #
# 5. Transitions: stochastic, legal, and empirically grounded
# --------------------------------------------------------------------------- #
def test_transition_probs_are_normalized():
    for state, actions in C.TRANSITIONS.items():
        for action, dist in actions.items():
            assert abs(sum(dist.values()) - 1.0) < 1e-6, (state, action)


def test_empirical_transitions_rows_sum_to_one():
    table = empirical_state_transitions(C.TRAIN_PARQUET, max_rows_per_day=20_000)
    assert list(table.index) == list(C.STATES)
    assert list(table.columns) == list(C.STATES)
    entries = table.to_numpy()
    assert ((entries >= 0.0) & (entries <= 1.0)).all(), "probabilities out of range"
    for st in C.STATES:
        rowsum = float(table.loc[st].sum())
        if rowsum > 0.0:  # a state absent from the small sample sums to 0
            assert abs(rowsum - 1.0) < 1e-6, f"{st} row sums to {rowsum}"
    print("\nEmpirical flow-sequence transitions (train sample):")
    print(table.round(4).to_string())


def test_monitor_episode_terminates_at_max_steps():
    env = IDSEnv2018(max_rows_per_state=50, max_steps=3,
                     obs_noise_sigma=0.0, obs_mask_prob=0.0, seed=5)
    env.reset(seed=5)
    term = False
    steps = 0
    while not term:
        obs, r, term, trunc, info = env.step(C.ACTION_INDEX["Monitor"])
        steps += 1
        assert env.observation_space.contains(obs)
    assert steps == 3
    assert info["terminal_reason"] == "timeout"


# --------------------------------------------------------------------------- #
# 6. Terminal semantics: Isolate ends the episode; resolve ends it
# --------------------------------------------------------------------------- #
def test_isolate_is_terminal():
    env = IDSEnv2018(max_rows_per_state=50, max_steps=20,
                     obs_noise_sigma=0.0, obs_mask_prob=0.0, seed=9)
    env.reset(seed=9, options={"state": "Compromised"})
    obs, r, term, trunc, info = env.step(C.ACTION_INDEX["Isolate"])
    assert term is True
    assert trunc is False
    assert info["terminal_reason"] == "isolated"
    assert r == pytest.approx(3.0)  # TP (+18) - Isolate cost (15)


# --------------------------------------------------------------------------- #
# 7. Latency overuse penalty (consecutive DPI/Throttle)
# --------------------------------------------------------------------------- #
def test_latency_penalty_on_throttle_overuse(env, monkeypatch):
    # Force Throttle to always stay Suspicious so we can stack two throttles.
    monkeypatch.setitem(
        C.TRANSITIONS, "Suspicious",
        {"Monitor": {"Suspicious": 1.0},
         "DPI": {"Suspicious": 1.0},
         "Throttle": {"Suspicious": 1.0},
         "Honeypot": {"Suspicious": 1.0}},
    )
    e = IDSEnv2018(max_rows_per_state=20, max_steps=10,
                   obs_noise_sigma=0.0, obs_mask_prob=0.0, seed=2)
    e.reset(seed=2, options={"state": "Suspicious"})
    a = C.ACTION_INDEX["Throttle"]
    _, r1, t1, _, i1 = e.step(a)
    assert r1 == pytest.approx(-1.5)       # cost only, no latency (queue=1)
    assert i1["latency_run"] == 1
    _, r2, t2, _, i2 = e.step(a)
    assert r2 == pytest.approx(-1.5 - 0.25)  # cost + alpha * (queue-1)
    assert i2["latency_run"] == 2
    # A different action resets the queue.
    _, _, _, _, i3 = e.step(C.ACTION_INDEX["Monitor"])
    assert i3["latency_run"] == 0


# --------------------------------------------------------------------------- #
# 8. Guard rail: masked (illegal) actions are heavily penalized
# --------------------------------------------------------------------------- #
def test_invalid_action_penalized(env):
    env.reset(seed=1)
    # DPI is not in the Normal playbook.
    obs, r, term, trunc, info = env.step(C.ACTION_INDEX["DPI"])
    assert r == pytest.approx(C.INVALID_ACTION_PENALTY)
    assert term is False
    assert info["mdp_state"] == "Normal"


# --------------------------------------------------------------------------- #
# 9. Reproducibility with a fixed seed
# --------------------------------------------------------------------------- #
def test_seeded_reset_is_reproducible():
    a = IDSEnv2018(max_rows_per_state=50, max_steps=3, seed=0)
    b = IDSEnv2018(max_rows_per_state=50, max_steps=3, seed=0)
    oa, ia = a.reset(seed=123)
    ob, ib = b.reset(seed=123)
    assert np.array_equal(oa, ob)
    assert ia["source"] == ib["source"]


# --------------------------------------------------------------------------- #
# 10. All actions sampled once from each legal state stay finite
# --------------------------------------------------------------------------- #
def test_all_legal_actions_finite(env):
    from itertools import product
    for st, a in product(C.STATES, C.ACTIONS):
        if a in C.VALID_ACTIONS[st]:
            e = IDSEnv2018(max_rows_per_state=20, max_steps=2, seed=1)
            e.reset(seed=1, options={"state": st})
            obs, r, term, trunc, info = e.step(C.ACTION_INDEX[a])
            assert np.isfinite(obs).all()
            assert np.isfinite(r)
            if a in ("Isolate",):
                assert term is True