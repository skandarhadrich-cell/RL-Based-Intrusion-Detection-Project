# Day 3 report - reward function

Discount factor `gamma = 0.9`.

## Formula

```
R(s,a) = R_detection(state, action, next) - C_action(a) - P_latency(run)
```

### R_detection (confusion-matrix payoff)

| event   |   value |
|---------|---------|
| TN      |       1 |
| TP      |      18 |
| FP      |     -12 |
| FN      |     -40 |

### C_action (operating cost)

| action   |   cost |
|----------|--------|
| Monitor  |    0   |
| DPI      |    1   |
| Throttle |    1.5 |
| Honeypot |    3   |
| Block    |    2   |
| Isolate  |   15   |

### P_latency

`alpha * max(0, run - 1)` with `alpha = 0.25`, where `run` is the consecutive DPI/Throttle count. Billed from the 2nd consecutive use:

|   queue length |   P_latency |
|----------------|-------------|
|              0 |        0    |
|              1 |        0    |
|              2 |        0.25 |
|              3 |        0.5  |
|              4 |        0.75 |

## Net reward table (state, action -> next state)

| state       | action   | next         |   net R(s,a) |
|-------------|----------|--------------|--------------|
| Normal      | Monitor  | Normal       |          0   |
| Normal      | Monitor  | Suspicious   |          0   |
| Normal      | Monitor  | Compromised  |          0   |
| Suspicious  | Monitor  | Suspicious   |          0   |
| Suspicious  | Monitor  | Compromised  |          0   |
| Suspicious  | Monitor  | Normal       |          0   |
| Suspicious  | DPI      | Suspicious   |         -1   |
| Suspicious  | DPI      | Compromised  |         17   |
| Suspicious  | DPI      | Normal       |          0   |
| Suspicious  | Throttle | Suspicious   |         -1.5 |
| Suspicious  | Throttle | Compromised  |        -41.5 |
| Suspicious  | Throttle | Normal       |         -0.5 |
| Suspicious  | Honeypot | Suspicious   |         15   |
| Suspicious  | Honeypot | Compromised  |         15   |
| Suspicious  | Honeypot | Normal       |         -2   |
| Compromised | Monitor  | Compromised  |          0   |
| Compromised | Monitor  | Normal       |          0   |
| Compromised | Block    | Compromised  |         -2   |
| Compromised | Block    | Normal       |         18   |
| Compromised | Isolate  | __terminal__ |          3   |

## Worked episode (slides) verification

| t | state·action | next | rewards | G_t  |
|---|--------------|------|--------:|-----:|
| 0 | Normal·Monitor | Suspicious |       0 | -25.85 |
| 1 | Suspicious·Throttle | Compromised |   -41.5 | -28.72 |
| 2 | Compromised·Block | Compromised |      -2 |   14.2 |
| 3 | Compromised·Block | Normal |      18 |     18 |

Backward returns checked against the slides:
```
G3 = 18
G2 = -2 + 0.9*18    = 14.2
G1 = -41.5 + 0.9*14.2 = -28.72
G0 = 0.9*(-28.72)  = -25.848000   (slides print -25.85)
```
> The slides print `G0 = -25.85`; their own arithmetic (`0.9 * -28.72`) yields **-25.848**. Tests assert the exact recurrence to 1e-6 **and** that it rounds to the printed -25.85.

Event classification of the worked episode: NONE → FN → NONE → TP

## Latency demo

| queue run | step reward formula | P_latency | net |
|-----------|---------------------|----------:|----:|
| 1 | Throttle (S→S): net -1.5 | 0.00 | -1.50 |
| 2 | Throttle (S→S) after 1 DPI/Throttle | 0.25 | -1.75 |
| 3 | third consecutive DPI/Throttle | 0.50 | -2.00 |

## Replay consistency (online env == pure re-computation)

- Episodes rolled: 60 (seeded), steps: 984
- Mismatching steps: 0
- Worst |online − offline| over 984 steps: 0

**PASS**: every online `env.step` reward equals `step_reward(state, action, next, latency_run)`.