# Prioritized Experience Replay

> **Goal:** replace uniform replay in the Double DQN agent with **proportional
> prioritized replay** (sum-tree sampling + importance-sampling correction) and
> measure whether it changes the learned policy or closes the Suspicious gap to
> the exact optimum.
>
> Artifacts: `src/rl_ids/per.py`, `src/rl_ids/evaluate.py`,
> `src/rl_ids/train_per.py`, `tests/test_per.py`,
> `data/processed/prioritized_replay_report.md`,
> `data/processed/prioritized_replay_learning_curve.png`,
> `models/double_dqn_per.pt`.

---

## 1. Why prioritized replay

Uniform replay draws every stored transition with equal probability, so a *rare
but informative* transition — e.g. the step where a Suspicious flow resolves, or
a large-surprise TD error — is drowned out by the millions of easy Normal ones.
**Prioritized Experience Replay** (Schaul et al., 2015) samples transitions in
proportion to how surprising they currently are:

```
p_i = |δ_i| + ε                     δ_i = TD error of transition i
P(i) = p_i^α / Σ_k p_k^α           α = 0.6  (0 = uniform, 1 = fully greedy)
```

Priority is refreshed after each update from the freshly computed TD error, so
the buffer continuously re-focuses on what the agent currently gets wrong.

## 2. Sum-tree sampling

Drawing `n` samples from `P(i)` and updating `p_i` run in **O(log N)** with a
binary **sum-tree**. Leaves hold the `p_i^α`; each internal node holds the sum of
its children, so `tree[1]` is the total. A sample is a uniform value in
`(0, total]` walked down to its leaf; a priority update walks back up adjusting
partial sums.

`PrioritizedReplayBuffer` subclasses the uniform `ReplayBuffer` and reuses its
array storage — only the sampler changes. Sampling is **stratified**: the unit
interval is split into `batch` equal segments and one value is drawn from each,
which keeps the mini-batch diverse while staying proportional.

## 3. Importance-sampling correction

Sampling by `P(i)` biases the gradient toward high-priority (often high-variance)
transitions, so the update is un-biased with per-sample importance weights:

```
w_i = ( N · P(i) )^(−β) / max_j ( N · P(j) )^(−β)
```

`β` is annealed **linearly 0.4 → 1.0** over `PER_BETA_STEPS` updates: early on
the bias is tolerated (faster learning), later on it is fully corrected for
convergence. The element-wise Huber loss is multiplied by `w_i` before reduction.

New transitions enter the tree at the **current maximum priority**, so nothing
starves before its first TD error is known.

## 4. Algorithm

```
buffer ← PrioritizedReplayBuffer(α)              # sum-tree over p^α
for episode:
    s ← reset (exploring start, uniform over the 3 buckets)
    repeat:
        a ← ε-greedy over legal actions (masked argmax of Q_online(s,·))
        s', r, done ← env.step(a)
        buffer.push(..., priority = running max)
        if |buffer| ≥ warm-up:
            β  ← anneal(agent.updates)           # 0.4 → 1.0
            batch ← buffer.sample(B, β)          # adds weights + indices
            loss, δ ← agent.learn(batch, weights=batch["weights"])
            buffer.update_priorities(batch["indices"], δ)
```

The only changes to the agent are: the Huber loss is computed element-wise so it
can be weighted, and `learn()` additionally returns the detached TD errors.
`DoubleDQNAgent.update(batch)` is unchanged for the uniform path.

## 5. Hyper-parameters

| Setting | Value |
|---|---|
| Priority exponent α | 0.6 |
| Priority floor ε | 1e-5 |
| IS exponent β | 0.4 → 1.0 over 20 000 updates |
| Replay capacity | 200 000 |
| Batch size | 64 |
| Warm-up | 2 000 transitions |
| Network / optimizer | MLP `(128, 128)` ReLU · Adam lr 1e-3 |
| Loss | Smooth L1 (Huber), IS-weighted |
| Discount γ | 0.9 |
| Target sync | every 1 000 updates (hard copy) |
| Gradient clip | max-norm 10.0 |
| ε | 1.0 → floor 0.05, ×0.995 per episode |
| Episodes | 5 000 |

Agent, environment, episodes, pool and seed (**7**) are **identical** to the
Double DQN run — only the buffer changes — so the two reports are directly
comparable. All values live in `config.py` (`PER_*`).

## 6. Results (5 000 episodes, seed 7)

22 147 steps · 20 148 gradient updates · final ε = 0.050 · final β = 1.000.

**Learned policy — matches the expected optimum in every bucket:**

| Start state | Dominant greedy action | Expected | Match |
|---|---|---|---|
| Normal | Monitor | Monitor | ✅ |
| Suspicious | Honeypot | Honeypot | ✅ |
| Compromised | Block | Block | ✅ |

**PER vs uniform replay** (greedy returns over 1 000 episodes/state; the random
baseline and Bellman optimum are shared because they depend only on the
environment and seed):

| Start state | Uniform greedy | PER greedy | Bellman optimum |
|---|---:|---:|---:|
| Normal | +8.58 | **+8.47** | +8.72 |
| Suspicious | +20.17 | **+20.65** | +23.16 |
| Compromised | +14.39 | **+14.30** | +14.51 |

- Normal and Compromised move by less than 0.11 — both were already ~0.15 from
  the optimum, so there is almost nothing to recover.
- **Suspicious improves by +0.48**, shrinking the gap to the exact optimum from
  **2.99 → 2.51**. This is the one bucket where the hidden state must be inferred
  from a high-variance, resolution-biased reward (σ ≈ 21), so re-weighting toward
  the transitions the agent currently mis-reads is exactly the right lever.

![PER learning curve](../data/processed/prioritized_replay_learning_curve.png)

```
PYTHONPATH=src python -m rl_ids.train_per
```

This writes `data/processed/prioritized_replay_report.md`,
`data/processed/prioritized_replay_learning_curve.png` and
`models/double_dqn_per.pt`.

## 7. Files

| File | Purpose |
|---|---|
| `src/rl_ids/per.py` | `SumTree`, `PrioritizedReplayBuffer`, β schedule |
| `src/rl_ids/evaluate.py` | shared greedy/random rollout evaluation |
| `src/rl_ids/train_per.py` | PER training driver, report + learning curve |
| `tests/test_per.py` | 14 unit tests (sum-tree, sampling, IS weights, β, weighted loss) |
| `data/processed/prioritized_replay_report.md` | generated report |
| `data/processed/prioritized_replay_learning_curve.png` | return / loss / ε / β trajectories |
| `models/double_dqn_per.pt` | trained online-network weights (git-ignored) |
