# Day 3 — Reward function

**Goal:** `R(s,a) = R_detection − C_action(a) − P_latency(a)`, implemented as a pure function of
`(true_label, action)` and **verified against the slides' worked example**.

**Status: DONE** — 28 reward tests pass; every online `env.step` reward is reproduced
exactly by offline pure-function replay (0 mismatches over ~1,000 steps).

---

## The formula

```
R(s,a) = R_detection(state, action, next) − C_action(a) − P_latency(latency_run)
```

### R_detection — confusion-matrix payoff

| event | value | meaning |
|-------|------:|---------|
| TP    | +18   | confirmed attack detected & handled |
| TN    | +1    | benign flow, light touch |
| FP    | −12   | heavy action wasted on benign traffic |
| FN    | −40   | attack slipped through / escalated |
| NONE  | 0     | pure-monitor step (no claim) |

### C_action — operating cost

| action | cost | | action | cost |
|--------|-----:|-|--------|-----:|
| Monitor | 0.0 | | Honeypot | 3.0 |
| DPI | 1.0 | | Block | 2.0 |
| Throttle | 1.5 | | Isolate | 15.0 |

### P_latency — overuse penalty

`P_latency = α · max(0, latency_run − 1)` with α = 0.25, where `latency_run` is the number of
**consecutive** DPI/Throttle actions. Billing starts at the 2nd consecutive use — a lone
Throttle costs nothing extra (required so the worked example's `R2 = −41.5` stays exact) while
sustained DPI/Throttle usage accrues α per extra queued step.

`reward.py` exposes the components (`detection_payoff`, `action_cost`, `latency_penalty`,
`step_breakdown`, `step_reward`) plus the pure net table `reward_function(state, action, next)`
that the environment consumes. The env bills latency through the same `latency_penalty()`
function — one source of truth.

## Worked example (the Day-3 checkpoint)

| t | state·action | next | reward |
|---|--------------|------|-------:|
| 0 | Normal·Monitor | Suspicious | 0 |
| 1 | Suspicious·Throttle | Compromised | **−41.5** |
| 2 | Compromised·Block | Compromised | **−2** |
| 3 | Compromised·Block | Normal (terminal) | **+18** |

Backward returns at γ = 0.9: G₃ = 18, G₂ = 14.2, G₁ = −28.72, **G₀ = −25.848**.

> **Precision note.** The slides print `G0 = -25.85`, but their own arithmetic
> (`0.9 × −28.72`) is **−25.848** — the printed value is rounded. The checkpoint test
> asserts the *exact* recurrence at ±1e-6 **and** that it rounds to the printed −25.85.
> (Fixing this rounding isn't possible without breaking the printed per-step rewards
> 0/−41.5/−2/+18.)

Event trace: `NONE → FN → NONE → TP` (Throttle lets the attack escalate = FN; the resolving
Block scores the TP).

## Net reward table

Monitor is 0 everywhere. On Suspicious: DPI → TN 0 / persist −1 / TP 17 ·
Throttle → TN −0.5 / persist −1.5 / **escalate FN −41.5** · Honeypot → TN −2 / TP 15.
On Compromised: **Block persist −2** / **Block resolve +18** (slide convention: TP payoff with
the 2.0 cost folded in) · **Isolate +3** (TP − 15). Full table in
`data/processed/day3_report.md`.

## Verification performed

- **28 pytest** (`tests/test_day3.py`): worked-episode rewards & returns, statelessness,
  event classification for every rule, action costs, Isolate payoff, invalid-action guard,
  component decomposition (`net == detection − cost` for every table entry, with the two
  documented slide conventions), latency curve, and the G0 checkpoint.
- **Replay consistency**: 60 seeded episodes (984 steps) → online `env.step` reward equals
  `step_reward(state, action, next, latency_run)` on every step, worst diff 0.
- `python -m rl_ids.validate_day3` regenerates `data/processed/day3_report.md`.

## How to run

```bash
PYTHONPATH=src ~/venvs/rl-ids/bin/python -m rl_ids.validate_day3   # report
~/venvs/rl-ids/bin/python -m pytest tests/test_day3.py -v
```

## Day-4 heads-up (already flagged in `docs/day2.md`)

The stationary model with these rewards satisfies the Day-3 checkpoint exactly but caps
`q(Compromised, Block) ≤ 18` (persist pays no detection), so the slide's q*-target of 31.06
can't coexist with persist-reward −2. Day 4's tabular run will report converged q-values vs.
the targets and quantify the gap.

## Artifacts

- `src/rl_ids/reward.py` — components + pure net table (used by the env)
- `src/rl_ids/validate_day3.py` — report generator + replay-consistency check
- `data/processed/day3_report.md` — generated verification report
- `tests/test_day3.py` (28 tests) · `docs/day3.md`

---
*Next: Day 4 — tabular Monte-Carlo control sanity check (`docs/day4.md`).*