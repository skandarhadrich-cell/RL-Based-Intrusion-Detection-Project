"""Day 3: reward-function verification.

Emits ``data/processed/day3_report.md`` and re-checks the three claims:

  1. R(s,a) = R_detection - C_action(a) - P_latency(a)   (component API)
  2. The slides' worked episode reproduces exactly
        rewards [0, -41.5, -2, +18]  ->  G0 = -25.848  (slides print -25.85)
  3. Online env rewards == pure-function replay, step for step, over many
     seeded episodes (latency branch included).

Run:  ~/venvs/rl-ids/bin/python -m rl_ids.validate_day3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tabulate import tabulate

from rl_ids import config as C
from rl_ids.env import IDSEnv2018
from rl_ids.reward import (
    classification,
    discounted_returns,
    latency_penalty,
    reward_function,
    step_reward,
)

# The slides' worked episode.
WORKED = [
    ("Normal", "Monitor", "Suspicious"),
    ("Suspicious", "Throttle", "Compromised"),
    ("Compromised", "Block", "Compromised"),
    ("Compromised", "Block", "Normal"),
]
EXPECTED_REWARDS = [0.0, -41.5, -2.0, 18.0]


def verify_worked_example() -> dict:
    rewards = [reward_function(s, a, sp) for s, a, sp in WORKED]
    assert rewards == EXPECTED_REWARDS, f"rewards mismatch: {rewards}"
    g = discounted_returns(rewards, C.GAMMA)
    g0_exact = g[0]
    assert abs(g0_exact - (-25.848)) < 1e-6, g0_exact
    assert round(g0_exact, 2) == -25.85
    assert abs(g0_exact - (-25.85)) < 0.01
    events = [classification(s, a, sp) for s, a, sp in WORKED]
    return {"rewards": rewards, "returns": g, "events": events, "g0_exact": g0_exact}


def replay_consistency(seed: int = 2026, episodes: int = 60, max_steps: int = 60) -> dict:
    """Roll seeded episodes; every online reward must equal step_reward(...)."""
    rng = np.random.default_rng(seed)
    env = IDSEnv2018(max_rows_per_state=3000, max_steps=max_steps, seed=seed)
    n_steps = 0
    worst = 0.0
    mismatches = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        prev_state = info["mdp_state"]
        while True:
            legal = np.flatnonzero(info["action_mask"])
            a = int(rng.choice(legal))
            _, r, term, trunc, info2 = env.step(a)
            # reconstruct the true next_state exactly as env sampled it
            if info2.get("terminal_reason") == "isolated":
                nxt = C.TERMINAL_MARKER
            else:
                nxt = info2["mdp_state"]
            expected = step_reward(prev_state, a, nxt, info2["latency_run"])
            diff = abs(r - expected)
            worst = max(worst, diff)
            if diff > 1e-6:
                mismatches += 1
            n_steps += 1
            prev_state = info2["mdp_state"]
            if term or trunc:
                break
    return {"episodes": episodes, "steps": n_steps,
            "worst_abs_diff": worst, "mismatches": mismatches}


def render() -> str:
    w = verify_worked_example()
    rc = replay_consistency()

    det_rows = [(e, C.DETECTION_REWARDS[e]) for e in ("TN", "TP", "FP", "FN")]
    cost_rows = [(name, C.ACTION_COSTS[i]) for name, i in C.ACTION_INDEX.items()]
    step_rows = [
        (s, a, sp or "terminal", reward_function(s, a, sp))
        for s, acts in C.TRANSITIONS.items()
        for a, dist in acts.items()
        for sp in dist
    ]
    worked_rows = [
        (t, *step, C.GAMMA, w["rewards"][t], w["returns"][t])
        for t, step in enumerate(WORKED)
    ]

    out = []
    out.append("# Day 3 report - reward function\n")
    out.append(f"Discount factor `gamma = {C.GAMMA}`.\n")

    out.append("## Formula\n")
    out.append("```\nR(s,a) = R_detection(state, action, next) - C_action(a) - P_latency(run)\n```\n")

    out.append("### R_detection (confusion-matrix payoff)\n")
    out.append(tabulate(det_rows, headers=["event", "value"], tablefmt="github"))
    out.append("\n### C_action (operating cost)\n")
    out.append(tabulate(cost_rows, headers=["action", "cost"], tablefmt="github"))
    out.append(
        "\n### P_latency\n\n"
        f"`alpha * max(0, run - 1)` with `alpha = {C.LATENCY_ALPHA}`, where `run` is the "
        "consecutive DPI/Throttle count. Billed from the 2nd consecutive use:\n"
    )
    out.append(tabulate(
        [(q, latency_penalty(q)) for q in range(5)],
        headers=["queue length", "P_latency"], tablefmt="github"))

    out.append("\n## Net reward table (state, action -> next state)\n")
    out.append(tabulate(
        step_rows, headers=["state", "action", "next", "net R(s,a)"], tablefmt="github"))

    out.append("\n## Worked episode (slides) verification\n")
    out.append("| t | state·action | next | rewards | G_t  |")
    out.append("|---|--------------|------|--------:|-----:|")
    for t, (s, a, sp), r, g in zip(
            range(len(WORKED)), WORKED, w["rewards"], w["returns"]):
        out.append(f"| {t} | {s}·{a} | {sp} | {r:>7.4g} | {g:>6.4g} |")
    out.append("")
    out.append("Backward returns checked against the slides:")
    out.append("```")
    out.append(f"G3 = 18")
    out.append(f"G2 = -2 + 0.9*18    = {w['returns'][2]:.4g}")
    out.append(f"G1 = -41.5 + 0.9*14.2 = {w['returns'][1]:.4g}")
    out.append(f"G0 = 0.9*(-28.72)  = {w['g0_exact']:.6f}   (slides print -25.85)")
    out.append("```")
    out.append(
        "> The slides print `G0 = -25.85`; their own arithmetic "
        "(`0.9 * -28.72`) yields **-25.848**. Tests assert the exact recurrence "
        "to 1e-6 **and** that it rounds to the printed -25.85.")
    out.append("\nEvent classification of the worked episode: "
               + " → ".join(w["events"]))

    out.append("\n## Latency demo\n")
    out.append("| queue run | step reward formula | P_latency | net |")
    out.append("|-----------|---------------------|----------:|----:|")
    out.append("| 1 | Throttle (S→S): net -1.5 | 0.00 | -1.50 |")
    out.append("| 2 | Throttle (S→S) after 1 DPI/Throttle | 0.25 | -1.75 |")
    out.append("| 3 | third consecutive DPI/Throttle | 0.50 | -2.00 |")

    out.append("\n## Replay consistency (online env == pure re-computation)\n")
    out.append(f"- Episodes rolled: {rc['episodes']} (seeded), steps: {rc['steps']}")
    out.append(f"- Mismatching steps: {rc['mismatches']}")
    out.append(f"- Worst |online − offline| over {rc['steps']} steps: "
               f"{rc['worst_abs_diff']:.3g}\n")
    if rc["mismatches"] == 0:
        out.append("**PASS**: every online `env.step` reward equals `step_reward(state, action, next, latency_run)`.")
    else:
        out.append("**FAIL**: replay diverges from the pure function - inspect!")
    return "\n".join(out)


def main() -> None:
    text = render()
    path = C.PROCESSED_DIR / "day3_report.md"
    path.write_text(text)
    print(f"Wrote {path}")
    print(f"G0 exact = {verify_worked_example()['g0_exact']:.6f} (slides print -25.85)")


if __name__ == "__main__":
    main()