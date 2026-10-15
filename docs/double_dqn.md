# Double DQN agent

> **Goal:** train a neural action-value agent on the **partially-observable**
> `IDSEnv2018` — real-flow observations (17 features, Gaussian noise, telemetry
> masking) plus the playbook action mask — and confirm it recovers the optimal
> response policy that the tabular check already validated.
>
> Artifacts: `src/rl_ids/dqn.py`, `src/rl_ids/train_dqn.py`,
> `tests/test_dqn.py`, `data/processed/double_dqn_report.md`,
> `data/processed/double_dqn_learning_curve.png`, `models/double_dqn.pt`.

---

## 1. Why Double DQN

Vanilla DQN uses the **target** network both to *select* and to *evaluate* the
greedy next action:

```
y = r + γ · max_a Q_target(s', a)
```

The same network that picks the maximiser also scores it, so any transient
over-estimate inflates the target and gets reinforced — the classic
**maximisation bias**. **Double DQN** decouples the two roles:

```
a* = argmax_a Q_online(s', a)        # online net SELECTS (masked to legal)
y  = r + γ · Q_target(s', a*)        # target net EVALUATES that action
```

Selection now uses the fast-changing online net, while the value is scored by
the slowly-updated target net, cutting the upward bias.

## 2. Observation model & the playbook mask

- The Q-network consumes **only the 17 normalized features** (`config.OBS_FEATURES`).
  The hidden severity bucket (`Normal`/`Suspicious`/`Compromised`) is **never**
  an input — the agent must infer it from telemetry, exactly as documented in
  `docs/environment.md`.
- The **action mask** (`info["action_mask"]`, from `config.VALID_ACTIONS`) is
  applied at *every* argmax — both when acting and when building the bootstrap
  target — so the agent never selects, and never bootstraps through, an illegal
  action. Masked Q entries are set to `-inf`. The mask only constrains
  **legality**; it is not concatenated onto the observation.

## 3. Algorithm

```
for episode:
    s ← reset (exploring start, uniform over the 3 buckets)
    repeat:
        a ← ε-greedy over legal actions (masked argmax of Q_online(s,·))
        s', r, done ← env.step(a)
        store (s, a, r, s', done, mask(s'))        # next mask needed for the target
        if |buffer| ≥ warm-up:
            batch ← uniform sample
            a* ← argmax Q_online(s',·) masked by mask(s')
            y  ← r + γ · Q_target(s', a*) · (1 − done)
            gradient step on Huber(Q_online(s,a), y)
        every `target_sync` updates: Q_target ← Q_online
```

- **Exploring starts** (uniform over buckets) mirror the tabular checkpoint: a
  Normal-only start reaches Suspicious in <1% of episodes and would starve its
  action-values.
- **Uniform replay** for now; **prioritized** replay is the next phase and will
  swap in behind the same `ReplayBuffer.sample` interface.

## 4. Hyper-parameters

| Setting | Value |
|---|---|
| Network | MLP `(128, 128)`, ReLU |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Smooth L1 (Huber) |
| Discount γ | 0.9 |
| Replay capacity | 200 000 |
| Batch size | 64 |
| Warm-up | 2 000 transitions |
| Target sync | every 1 000 updates (hard copy) |
| Gradient clip | max-norm 10.0 |
| ε | 1.0 → floor 0.05, ×0.995 per episode |
| Episodes | 5 000 |

All values live in `config.py` (`DQN_*`) so the run is reproducible from a single
place.

## 5. Evaluation

The greedy policy (ε = 0) is rolled out from **each start bucket** and reported
against two references:

1. a **uniform-random** legal policy (same episodes), and
2. the **exact tabular optimum** — `max_a q*(s,a)` over legal `a`, from
   `rl_ids.tabular.bellman_solution`.

`train_dqn.py` writes the dominant first action per bucket and flags whether it
matches the expected `Normal→Monitor`, `Suspicious→Honeypot`, `Compromised→Block`.
It raises (rather than writing a report) if the greedy policy fails to beat the
random baseline outright — a wiring bug, not under-training.

## 6. Results

_Pending the RAM upgrade: the online run loads real-flow observation pools into
memory, so run it after raising VM RAM (or lower `MAX_ROWS_PER_STATE`)._

```
PYTHONPATH=src python -m rl_ids.train_dqn
```

This writes `data/processed/double_dqn_report.md`,
`data/processed/double_dqn_learning_curve.png` and `models/double_dqn.pt`.

## 7. Files

| File | Purpose |
|---|---|
| `src/rl_ids/dqn.py` | `QNetwork`, `ReplayBuffer`, `DoubleDQNAgent`, masked argmax, ε schedule, Double-DQN target |
| `src/rl_ids/train_dqn.py` | online training driver, evaluation, report + learning curve |
| `tests/test_dqn.py` | 17 unit tests (masking, target correctness, buffer, target sync, save/load) |
| `data/processed/double_dqn_report.md` | generated report (policy, returns, dynamics) |
| `data/processed/double_dqn_learning_curve.png` | return / loss / ε trajectories |
| `models/double_dqn.pt` | trained online-network weights (git-ignored) |
