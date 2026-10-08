"""Day 4: tabular sanity check - first-visit constant-α GLIE MC control.

Coverage:
    - first-visit return semantics (regression for the reversed-trace bug)
    - exact Bellman solution vs analytic closed-form values
    - unbiased MC evaluation of the greedy policy reproduces the Bellman q*
    - mc_control converges to the expected policy and Q stabilizes
"""
import numpy as np
import pytest

from rl_ids import config as C
from rl_ids.env import IDSEnv2018
from rl_ids.tabular import (
    SIDX,
    bellman_solution,
    epsilon_glie,
    evaluate_greedy,
    first_visit_returns,
    greedy_policy_table,
    mc_control,
)

GAMMA = C.GAMMA
M, B, H = C.ACTION_INDEX["Monitor"], C.ACTION_INDEX["Block"], C.ACTION_INDEX["Honeypot"]

EXPECTED_POLICY = {"Normal": ["Monitor"], "Suspicious": ["Honeypot"],
                   "Compromised": ["Block"]}

# Analytic values on the locked Day-3 reward model (computed by hand, verified
# against the env's transition kernel and the Bellman solver).
BELLMAN = {
    ("Normal", "Monitor"): 8.7215,
    ("Suspicious", "Monitor"): 17.204,
    ("Suspicious", "DPI"): 15.935,
    ("Suspicious", "Throttle"): 4.827,
    ("Suspicious", "Honeypot"): 23.163,
    ("Compromised", "Monitor"): 11.751,
    ("Compromised", "Block"): 14.508,
    ("Compromised", "Isolate"): 3.0,
}


@pytest.fixture(scope="module")
def env():
    return IDSEnv2018(max_rows_per_state=3000, max_steps=100, seed=0)


# --------------------------------------------------------------------------- #
# first-visit semantics (regression)
# --------------------------------------------------------------------------- #
def test_first_visit_returns_takes_first_occurrence():
    # N, Monitor appears 3x; its first-visit return must be the FULL episode
    # return from position 0 (= 0.9^3 * 18), NOT the last occurrence's 16.2.
    trace = [
        ("Normal", M, 0.0),
        ("Normal", M, 0.0),
        ("Normal", M, 0.0),
        ("Compromised", B, 18.0),
    ]
    got = first_visit_returns(trace, gamma=GAMMA)
    assert got[("Normal", M)] == pytest.approx(18.0 * GAMMA ** 3)          # 13.122
    assert got[("Compromised", B)] == pytest.approx(18.0)
    # naive reversed `seen`-update would give 16.2 for (Normal, Monitor)
    assert got[("Normal", M)] != pytest.approx(0.9 * 18.0)


def test_first_visit_returns_handles_repeats_and_multi_step():
    # (S, Honeypot) persists twice then resolves, then a (C, Block) persist+resolve
    trace = [
        ("Suspicious", H, 15.0),
        ("Suspicious", H, 15.0),
        ("Compromised", B, -2.0),
        ("Compromised", B, 18.0),
    ]
    got = first_visit_returns(trace, gamma=GAMMA)
    g_sh = 15 + 0.9 * 15 + 0.9 ** 2 * (-2) + 0.9 ** 3 * 18
    g_cb = -2 + 0.9 * 18
    assert got[("Suspicious", H)] == pytest.approx(g_sh)
    assert got[("Compromised", B)] == pytest.approx(g_cb)


def test_epsilon_glie_decays_to_zero():
    assert epsilon_glie(0) == 1.0
    eps = [epsilon_glie(ep) for ep in range(0, 10_000, 100)]
    assert eps == sorted(eps, reverse=True)
    assert epsilon_glie(10 ** 7, 1) < 1e-3
    # stretching divisor keeps exploration alive longer (finite-run GLIE)
    assert epsilon_glie(10_000, 1000) > epsilon_glie(10_000, 1)


# --------------------------------------------------------------------------- #
# exact Bellman solution
# --------------------------------------------------------------------------- #
def test_bellman_solution_matches_analytic_values(env):
    Qb = bellman_solution()
    for (s, name), expected in BELLMAN.items():
        a = C.ACTION_INDEX[name]
        assert float(Qb[SIDX[s], a]) == pytest.approx(expected, abs=1e-3)


def test_bellman_policy_is_expected(env):
    Qb = bellman_solution()
    assert greedy_policy_table(Qb) == EXPECTED_POLICY


def test_evaluate_greedy_reproduces_bellman_on_optimal_policy(env):
    Qb = bellman_solution()
    opt_action = {"Normal": "Monitor", "Suspicious": "Honeypot",
                  "Compromised": "Block"}
    # n small for speed; tolerances reflect MC sampling noise only
    for start, tol in (("Normal", 0.6), ("Suspicious", 2.0), ("Compromised", 1.0)):
        mean, std = evaluate_greedy(env, Qb, start, episodes=3000, seed=5)
        qstar = float(Qb[SIDX[start], C.ACTION_INDEX[opt_action[start]]])
        assert mean == pytest.approx(qstar, abs=tol)
        assert std >= 0.0


# --------------------------------------------------------------------------- #
# MC control convergence
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained():
    env = IDSEnv2018(max_rows_per_state=3000, max_steps=100, seed=0)
    schedule = lambda ep: epsilon_glie(ep, 1000.0)
    Q, track = mc_control(env, episodes=10000, alpha=0.05, seed=7,
                          eps_schedule=schedule, track_every=1000)
    return Q, track


def test_mc_control_converges_to_expected_policy(trained):
    Q, _ = trained
    assert greedy_policy_table(Q) == EXPECTED_POLICY


def test_mc_control_q_values_within_band(trained):
    Q, _ = trained
    # bands around the exact values; widened for the intrinsic constant-alpha
    # MC noise on the high-variance Suspicious returns
    assert 6.0 <= float(Q[SIDX["Normal"], M]) <= 12.0
    assert 14.0 <= float(Q[SIDX["Suspicious"], H]) <= 30.0
    assert 10.0 <= float(Q[SIDX["Compromised"], B]) <= 19.0


def test_mc_control_q_stabilizes(trained):
    Q, track = trained
    snaps = track["Q"]
    assert len(snaps) >= 6
    drifts = []
    for i in range(1, len(snaps)):
        drifts.append(max(
            abs(float(snaps[i][SIDX[s], a]) - float(snaps[i - 1][SIDX[s], a]))
            for s in C.STATES for a in C.VALID_ACTIONS[s]))
    # per-snapshot noise shrinks as epsilon -> 0 and estimates settle
    assert drifts[-1] < drifts[0]
    assert drifts[-1] < 3.0


def test_illegal_action_entries_never_used(trained):
    Q, _ = trained
    for s in C.STATES:
        for a in range(len(C.ACTIONS)):
            if a in C.VALID_ACTIONS[s]:
                continue
            # never selected by the mask, never backed up: must stay 0
            assert float(Q[SIDX[s], a]) == 0.0
    # legal entries must have been backed up into the right regime
    assert float(Q[SIDX["Normal"], M]) > 0.0
    assert float(Q[SIDX["Suspicious"], H]) > 0.0
    assert float(Q[SIDX["Compromised"], B]) >= 10.0  # well above Isolate (3)