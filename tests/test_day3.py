"""Day 3 checkpoint tests: reward model correctness.

The plan's worked example (from the MDP slides):

    t=0 Normal·Monitor    R1=0      -> Suspicious
    t=1 Suspicious·Throttle R2=-41.5 -> Compromised
    t=2 Compromised·Block  R3=-2     -> Compromised
    t=3 Compromised·Block  R4=+18    -> Normal (terminal)

with gamma=0.9 picks the return sequence down to G0=-25.85.  The example's own
arithmetic uses R1=0, R2=-41.5, R3=-2, R4=+18, which gives exactly
G0 = -25.848; the slides print -25.85 (rounded).  We assert the exact value
AND agreement with the printed checkpoint within 1e-2.
"""
import pytest

from rl_ids import config as C
from rl_ids.reward import (
    action_cost,
    classification,
    detection_payoff,
    discounted_returns,
    latency_penalty,
    reward_function,
    scripted_detection_events,
    step_breakdown,
    step_reward,
)

GAMMA = C.GAMMA

# (state, action, next_state) trace from the worked example.
WORKED_EPISODE = [
    ("Normal", "Monitor", "Suspicious"),
    ("Suspicious", "Throttle", "Compromised"),
    ("Compromised", "Block", "Compromised"),
    ("Compromised", "Block", "Normal"),
]
EXPECTED_REWARDS = [0.0, -41.5, -2.0, 18.0]


# --------------------------------------------------------------------------- #
# 1. Pure reward = function of (state, action, next_state)
# --------------------------------------------------------------------------- #
def test_worked_episode_rewards():
    rewards = [reward_function(s, a, sp) for s, a, sp in WORKED_EPISODE]
    assert rewards == pytest.approx(EXPECTED_REWARDS, abs=1e-9)


def test_worked_episode_returns():
    rewards = [reward_function(s, a, sp) for s, a, sp in WORKED_EPISODE]
    g = discounted_returns(rewards, GAMMA)
    assert g[3] == pytest.approx(18.0)
    assert g[2] == pytest.approx(-2.0 + 0.9 * 18.0)                 # 14.2
    assert g[1] == pytest.approx(-41.5 + 0.9 * 14.2)                # -28.72
    assert g[0] == pytest.approx(-25.848, abs=1e-3)                 # exact
    assert g[0] == pytest.approx(-25.85, abs=1e-2)                  # printed plan


def test_reward_is_stateless():
    """Same (s,a,s') always yields the same reward - no internal counter."""
    r1 = reward_function("Compromised", "Block", "Normal")
    r2 = reward_function("Compromised", "Block", "Normal")
    assert r1 == r2 == 18.0


# --------------------------------------------------------------------------- #
# 2. Detection event classification
# --------------------------------------------------------------------------- #
def test_classification_events():
    assert classification("Normal", "Monitor", "Suspicious") == "NONE"
    assert classification("Suspicious", "Throttle", "Compromised") == "FN"
    assert classification("Suspicious", "Throttle", "Normal") == "TN"
    assert classification("Suspicious", "DPI", "Compromised") == "TP"
    assert classification("Suspicious", "Honeypot", "Suspicious") == "TP"
    assert classification("Suspicious", "Honeypot", "Normal") == "TN"
    assert classification("Compromised", "Block", "Compromised") == "NONE"
    assert classification("Compromised", "Block", "Normal") == "TP"
    assert classification("Compromised", "Isolate", None) == "TP"


def test_worked_episode_events():
    events = scripted_detection_events(WORKED_EPISODE)
    assert events == ["NONE", "FN", "NONE", "TP"]


# --------------------------------------------------------------------------- #
# 3. Action costs and terminal payoffs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action_name,expected",
    [("Monitor", 0.0), ("DPI", 1.0), ("Throttle", 1.5),
     ("Honeypot", 3.0), ("Block", 2.0), ("Isolate", 15.0)],
)
def test_action_costs(action_name, expected):
    assert C.ACTION_COSTS[C.ACTION_INDEX[action_name]] == expected


def test_isolate_terminal_payoff():
    assert reward_function("Compromised", "Isolate", None) == pytest.approx(3.0)
    assert reward_function("Compromised", "Isolate", C.TERMINAL_MARKER) == pytest.approx(3.0)


def test_invalid_action_penalty():
    # Block is not in the Normal playbook -> guard-rail penalty.
    assert reward_function("Normal", "Block", "Normal") == C.INVALID_ACTION_PENALTY
    assert reward_function("Normal", "DPI", "Normal") == C.INVALID_ACTION_PENALTY


# --------------------------------------------------------------------------- #
# 4. Cost-free event values are the slide constants
# --------------------------------------------------------------------------- #
def test_detection_payoffs_used():
    assert C.DETECTION_REWARDS["TP"] == 18.0
    assert C.DETECTION_REWARDS["FN"] == -40.0
    assert C.DETECTION_REWARDS["FP"] == -12.0
    assert C.DETECTION_REWARDS["TN"] == 1.0


# --------------------------------------------------------------------------- #
# 5. Full-distribution reward closure (every legal (s,a,s') table entry)
# --------------------------------------------------------------------------- #
def test_reward_closed_over_transition_table():
    for st, actions in C.TRANSITIONS.items():
        for a, dist in actions.items():
            for nxt in dist:
                r = reward_function(st, a, nxt if nxt != C.TERMINAL_MARKER else None)
                assert np_isfinite(r), (st, a, nxt, r)


# --------------------------------------------------------------------------- #
# Day 3: explicit R(s,a) = R_detection - C_action - P_latency components
# --------------------------------------------------------------------------- #
def test_detection_payoff_none_is_zero():
    assert detection_payoff("NONE") == 0.0
    assert detection_payoff("TP") == 18.0
    assert detection_payoff("FN") == -40.0


def test_action_cost_accepts_name_or_index():
    for i, name in enumerate(C.ACTIONS):
        assert action_cost(i) == C.ACTION_COSTS[i] == action_cost(name)


@pytest.mark.parametrize("queue,expected", [(0, 0.0), (1, 0.0), (2, 0.25),
                                            (3, 0.5), (5, 1.0)])
def test_latency_penalty(queue, expected):
    assert latency_penalty(queue, alpha=C.LATENCY_ALPHA) == pytest.approx(expected)


def test_latency_formula_matches_config():
    assert latency_penalty(2) == C.LATENCY_ALPHA * max(0, 2 - 1)


def test_step_breakdown_decomposes_net():
    """For every net table entry, net == detection - cost (documented exceptions
    separate): the terminal Block-resolve (+18) and Monitor steps."""
    from rl_ids.reward import _NET
    for (st, a, sp), net in _NET.items():
        bd = step_breakdown(st, a, sp)
        assert bd["net"] == net
        if (st, a) in {("Compromised", "Block")} and sp == "Normal":
            assert net == 18.0                      # slide convention: cost folded in
            assert bd["detection"] == 18.0
        elif bd["event"] == "NONE":
            assert net == -bd["cost"] or net == 0.0  # monitor or cost-only step
        else:
            assert net == pytest.approx(bd["detection"] - bd["cost"])


def test_step_reward_applies_latency():
    # a lone Throttle is latency-free; the 2nd consecutive one is penalized.
    assert step_reward("Suspicious", "Throttle", "Suspicious", latency_run=1) \
        == pytest.approx(-1.5)
    assert step_reward("Suspicious", "Throttle", "Suspicious", latency_run=2) \
        == pytest.approx(-1.5 - C.LATENCY_ALPHA)


def test_returns_from_trace_matches_worked_example():
    from rl_ids.reward import returns_from_trace
    g = returns_from_trace(WORKED_EPISODE, gamma=GAMMA)
    assert g[0] == pytest.approx(-25.848, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. Checkpoint: G0 == -25.85 for the scripted episode (slides).
# --------------------------------------------------------------------------- #
def test_checkpoint_g0_equals_slides():
    """The plan's checkpoint. Note: with the printed rewards (0,-41.5,-2,+18)
    the recurrence gives exactly -25.848 (= 0.9 * -28.72); the slide prints
    -25.85 rounded. We enforce the EXACT recurrence at 1e-6 AND the printed
    value at 2 decimal places."""
    g = discounted_returns(EXPECTED_REWARDS, GAMMA)
    assert g[0] == pytest.approx(-25.848, abs=1e-6)
    assert round(g[0], 2) == -25.85
    assert g[0] == pytest.approx(-25.85, abs=1e-2)  # printed checkpoint value


# --------------------------------------------------------------------------- #
# 7. Replay consistency: online env.step reward == pure re-computation
# --------------------------------------------------------------------------- #
def test_replay_consistency_online_vs_offline():
    import numpy as np
    from rl_ids.env import IDSEnv2018
    env = IDSEnv2018(max_rows_per_state=500, max_steps=40, seed=11)
    rng = np.random.default_rng(3)
    worst, n = 0.0, 0
    for ep in range(12):
        _, info = env.reset(seed=100 + ep)
        prev = info["mdp_state"]
        while True:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            _, r, term, trunc, info = env.step(a)
            nxt = C.TERMINAL_MARKER if info.get("terminal_reason") == "isolated" \
                else info["mdp_state"]
            expected = step_reward(prev, a, nxt, info["latency_run"])
            assert r == pytest.approx(expected, abs=1e-9)
            worst = max(worst, abs(r - expected))
            n += 1
            prev = info["mdp_state"]
            if term or trunc:
                break
    assert worst < 1e-9
    assert n > 0


def np_isfinite(x) -> bool:
    import math
    return math.isfinite(x)