# Day 2 — Gymnasium environment (`RL-IDS-v0`)

**Goal:** a working `gym.Env` reproducing the state/action structure of the MDP slides,
whose observations are **real flows from the Day-1 train table** (partial observability via
noise + masking), and whose transition/reward model is the single source of truth for the
Day-3/4/5 checkpoints.

**Status: DONE** — 32 Day-2/3 tests pass (42 total with Day-1).

---

## The MDP at a glance

### State (hidden from the agent)
The true compromise bucket of the current traffic (`mdp_state`, derived from the hidden
`attack_label`): **Normal · Suspicious · Compromised**.

### Playbook (actions) and masks

| State      | Allowed actions                                  | Forbidden |
|------------|--------------------------------------------------|-----------|
| Normal     | `Monitor`                                        | DPI, Throttle, Honeypot, Block, Isolate |
| Suspicious | `Monitor`, `DPI`, `Throttle`, `Honeypot`         | Block, Isolate |
| Compromised| `Monitor`, `Block`, `Isolate`                    | DPI, Throttle, Honeypot |

Masked actions are never offered to the policy (`info["action_mask"]`); if taken anyway the
env returns `INVALID_ACTION_PENALTY = -100` (guard rail).

### Observation
`Box((17,))` — z-scored telemetry of one real flow + Gaussian noise (σ=0.05) and random
"missing telemetry" masking (p=0.1 per feature → 0.0). The hidden `mdp_state`/`attack_label`
are exposed in `info` **for evaluation only**, never as observation.

### Transitions (stochastic, dataset-grounded)
`(state, action) -> next_state` sampled from `config.TRANSITIONS`; the next *observation* is
always a **real flow row drawn from the pool whose hidden state is the sampled next state**.
No synthetic vectors — the agent never sees telemetry inconsistent with the condition it is in.

### Episode semantics
`reset()` starts in Normal. Episode ends when the hidden state is **resolved** (S/C → Normal),
the agent **Isolates**, or `max_steps` (default 100) is hit. Terminal reasons are logged in
`info["terminal_reason"]`.

---

## Transition model

### Configured kernel (slide-style, tuned for Day-3/4 checkpoints)

| State·Action         | → Normal | → Suspicious | → Compromised | → terminal |
|----------------------|---------:|-------------:|--------------:|-----------:|
| **N · Monitor**      | 0.840    | 0.005        | 0.155         |            |
| **S · Monitor**      | 0.10     | 0.70         | 0.20          |            |
| **S · DPI**          | 0.30     | 0.50         | 0.20          |            |
| **S · Throttle**     | 0.25     | 0.55         | 0.20          |            |
| **S · Honeypot**     | 0.295    | 0.510        | 0.195         |            |
| **C · Monitor**      | 0.10     |              | 0.90          |            |
| **C · Block**        | 0.497    |              | 0.503         |            |
| **C · Isolate**      |          |              |               | 1.0        |

### Empirical flow-sequence matrix (real data, full 15.2M train rows)
Consecutive-flow `state_t → state_{t+1}` counts, aggregated over all train days:

| state_t \\ state_{t+1} | Normal | Suspicious | Compromised |
|---------------------|-------:|-----------:|------------:|
| **Normal**          | 0.9810 | 0.0009     | 0.0180      |
| **Suspicious**      | 0.0314 | 0.9686     | 0.0000      |
| **Compromised**     | 0.1033 | 0.0000     | 0.8967      |

The empirical matrix confirms the kernel's qualitative structure — benign flows persist,
brute-force runs are self-perpetuating, compromised traffic mostly stays compromised and
occasionally resolves back to normal (~10%/step). The *configured* kernel is more attack-heavy
than the raw data (it is the slide model: attack arrival at Normal 0.155 vs 0.018 empirical)
because the checkpoints target the slide story; the empirical kernel remains available for
Day-6/7 realism sweeps (`env.empirical_state_transitions`).

---

## Reward model (`rl_ids/reward.py`)

Pure function `reward(state, action, next_state)` — stateless, reproducible offline from a
trace. Detection payoffs + action costs:

| Event | Value | | Action | Cost |
|-------|------:|-|---------|-----:|
| TP    | +18   | | Monitor | 0.0 |
| TN    | +1    | | DPI | 1.0 |
| FP    | −12   | | Throttle | 1.5 |
| FN    | −40   | | Honeypot | 3.0 |
| | | | Block | 2.0 |
| | | | Isolate | 15.0 |

Net step values: Monitor = 0 everywhere · DPI on S: TN 0 / persist −1 / TP 17 ·
Throttle on S: TN −0.5 / persist −1.5 / **FN −41.5** · Honeypot on S: TN −2 / TP 15 ·
**Block on C: persist −2 / resolve +18** · **Isolate: +3** (TP − cost). A **latency penalty**
`α·max(0, consecutive_DPI/Throttle − 1)` (α=0.25) is subtracted by the env, punishing
overuse without touching the pure function.

### Day-3 checkpoint (worked episode from the slides)

| t | state·action | reward | next |
|---|--------------|-------:|------|
| 0 | Normal·Monitor | 0 | Suspicious |
| 1 | Suspicious·Throttle | −41.5 | Compromised |
| 2 | Compromised·Block | −2 | Compromised |
| 3 | Compromised·Block | **+18** | Normal (terminal) |

Returns (γ=0.9): G₃=18, G₂=14.2, G₁=−28.72, **G₀ = −25.848**. The slides print −25.85
(rounding); the test asserts both the exact value (±1e-3) and the printed value (±1e-2).

> **Reconciliation note (Day-4 consequence).** The slide's q*-targets on the same episode
> numbers would be Normal→Monitor 18.29, Suspicious→Honeypot 28.61, Compromised→Block 31.06.
> A *stationary* MDP with the worked-example reward (Block-persist = −2, resolve = +18)
> satisfies the Day-3 checkpoint exactly but caps q(C,Block) ≤ 18 (each persist step pays no
> detection), so the 31.06 target is **not** simultaneously reproducible in this single reward
> model — it implies a reward that pays a TP on every successful block (+16 per persist).
> We keep the reward locked to the literal Day-3 checkpoint and will report our model's
> converged q-values vs. the slide targets (and the gap) on Day 4. Under this stationary
> model a quick Monte-Carlo already gives q(N,Monitor) ≈ 8.7, q(S,Honeypot) ≈ 23.2,
> q(C,Block) ≈ 14.5.

---

## Observation sanity

- Observed row index + hidden state are tracked (`info["source"]`); tests assert the obs
  vector **equals a real pool row** when noise/masking are off.
- With noise+masking on, the same row diverges (partial observability works).
- Fixed seed ⇒ reproducible episodes (`io` seeded independently per reset).

## Tests (32 new + 10 Day-1)

`tests/test_day2.py` — space structure (Box(17), Discrete(6)), masks per state, reset
semantics, observation corruption + real-row provenance, transition normalization,
empirical matrix, max-steps and Isolate termination, latency overuse penalty, invalid-action
penalty, seed reproducibility, legality of every (state, action) pair.

`tests/test_day3.py` — worked-episode rewards `[0, −41.5, −2, +18]`, returns
(G₀ = −25.848 exact / −25.85 printed), statelessness, event classification (TN/TP/FP/FN/NONE),
action costs, Isolate payoff, invalid-action guard, reward closure over the transition table.

## How to run

```bash
~/venvs/rl-ids/bin/python -m pytest tests/test_day2.py tests/test_day3.py -v
PYTHONPATH=src ~/venvs/rl-ids/bin/python - <<'EOF'
from rl_ids.env import IDSEnv2018
env = IDSEnv2018(seed=1)
obs, info = env.reset(seed=1)
print(obs.shape, info["action_mask"])          # (17,) [True,False,...]
obs, r, term, trunc, info = env.step(0)        # Monitor
EOF
```

## Artifacts

- `src/rl_ids/env.py` — `IDSEnv2018`, `empirical_state_transitions`
- `src/rl_ids/reward.py` — pure reward model + event classification
- `src/rl_ids/config.py` — MDP block: actions, masks, costs, transitions, noise, latency
- `tests/test_day2.py`, `tests/test_day3.py`

---
*Next: Day 3 — reward/return verification + eval harness on the real env. Day-4 will run
tabular control and record converged q-values vs. the slide targets.*