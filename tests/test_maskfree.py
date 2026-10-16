"""Mask-free mode (Path 2) tests.

Coverage:
    - the env offers every action in every state when ``mask_actions=False``
      and never charges the illegal-action penalty
    - the reward is total over all 3x6 (state, action) pairs
    - the mask-free exact Bellman optimum is *identical* to the masked one
      (the playbook policy and values are preserved - the core design gate)
    - the mask-free tabular helpers expose the full action set
"""
import numpy as np
import pytest

from rl_ids import config as C
from rl_ids.env import IDSEnv2018
from rl_ids.reward import reward_function, _NET
from rl_ids.tabular import (
    SIDX, bellman_solution, greedy_actions, legal_mask,
)

STATES = list(C.STATES)
EXPECTED_POLICY = {"Normal": "Monitor", "Suspicious": "Honeypot",
                   "Compromised": "Block"}

# Analytic masked optimum (matches tests/test_tabular_mc_control.py).
MASKED_OPTIMUM = {"Normal": 8.7215, "Suspicious": 23.163, "Compromised": 14.508}


@pytest.fixture(scope="module")
def env_mf() -> IDSEnv2018:
    return IDSEnv2018(split="train", max_rows_per_state=500, max_steps=20,
                      seed=7, mask_actions=False)


# --------------------------------------------------------------------------- #
# 1. Environment offers all actions, never penalises
# --------------------------------------------------------------------------- #
def test_mask_free_env_exposes_all_actions(env_mf):
    _, info = env_mf.reset(seed=0)
    assert info["action_mask"].all()
    assert info["action_mask"].sum() == len(C.ACTIONS)


@pytest.mark.parametrize("state", STATES)
def test_mask_free_env_never_charges_invalid_penalty(env_mf, state):
    """Every action in every state is legal - the guard-rail never fires."""
    for a in C.ALL_ACTION_INDICES:
        env_mf.reset(seed=a, options={"state": state})
        _, r, _, _, _ = env_mf.step(a)
        assert r != pytest.approx(C.INVALID_ACTION_PENALTY)
        assert r != pytest.approx(C.INVALID_ACTION_PENALTY - C.LATENCY_ALPHA)


def test_masked_env_still_guards():
    """The default (masked) env keeps the illegal-action guard rail."""
    env = IDSEnv2018(max_rows_per_state=20, max_steps=5, seed=0)
    env.reset(seed=1, options={"state": "Normal"})
    _, r, _, _, _ = env.step(C.ACTION_INDEX["DPI"])   # DPI not in Normal mask
    assert r == pytest.approx(C.INVALID_ACTION_PENALTY)


# --------------------------------------------------------------------------- #
# 2. Reward is total over all 18 (state, action) pairs
# --------------------------------------------------------------------------- #
def test_net_table_covers_all_pairs():
    covered = {(st, a) for (st, a, _sp) in _NET}
    assert covered == {(st, act) for st in STATES for act in C.ACTIONS}


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("action", C.ACTIONS)
def test_reward_total_matches_net(state, action):
    """enforce_legality=False returns the real value for every (state, action)."""
    a = C.ACTION_INDEX[action]
    sub = {sp: net for (st, act, sp), net in _NET.items()
           if st == state and act == action}
    for sp, net in sub.items():
        assert reward_function(state, a, sp, enforce_legality=False) == pytest.approx(net)


def test_legality_guard_default_unchanged():
    """Default enforce_legality=True still penalises off-playbook actions."""
    assert reward_function("Normal", "Block", "Normal") == C.INVALID_ACTION_PENALTY
    assert reward_function("Normal", "Block", "Normal",
                           enforce_legality=False) == pytest.approx(-14.0)


# --------------------------------------------------------------------------- #
# 3. The design gate: mask-free optimum == masked optimum == playbook
# --------------------------------------------------------------------------- #
def test_mask_free_bellman_preserves_playbook():
    Qf = bellman_solution(all_actions=True)
    for st in STATES:
        acts = [C.ACTIONS[a] for a in greedy_actions(Qf, st, all_actions=True)]
        assert acts == [EXPECTED_POLICY[st]], (st, acts)


def test_mask_free_optimum_equals_masked_optimum():
    Qm = bellman_solution(all_actions=False)
    Qf = bellman_solution(all_actions=True)
    for st, expected in MASKED_OPTIMUM.items():
        i = SIDX[st]
        v_masked = float(np.max(Qm[i, list(C.VALID_ACTIONS[st])]))
        v_free = float(np.max(Qf[i, :]))
        assert v_masked == pytest.approx(expected, abs=1e-3)
        assert v_free == pytest.approx(expected, abs=1e-3)


def test_off_playbook_actions_are_dominated():
    """In every state the playbook q* strictly beats every other action."""
    Qf = bellman_solution(all_actions=True)
    for st in STATES:
        i = SIDX[st]
        best = C.ACTION_INDEX[EXPECTED_POLICY[st]]
        for a in C.ALL_ACTION_INDICES:
            if a == best:
                continue
            assert Qf[i, best] > Qf[i, a], (st, C.ACTIONS[a])


# --------------------------------------------------------------------------- #
# 4. Tabular helpers expose the full action set
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", STATES)
def test_all_actions_mask_is_full(state):
    assert legal_mask(state, all_actions=True).all()


def test_greedy_actions_consider_all_when_free():
    Qf = bellman_solution(all_actions=True)
    # the playbook action is the unique argmax everywhere
    for st in STATES:
        assert len(greedy_actions(Qf, st, all_actions=True)) == 1
