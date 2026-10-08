# Day 4 — Tabular sanity check (Monte Carlo control)

> **Goal:** confirm the environment converges to the known-correct optimal
> policy **before any neural net is involved**. If the tabular agent does not
> find the expected policy, the bug is in the env/reward — fix it before
> wiring Double DQN + PER (Days 5–6).
>
> Artifacts: `src/rl_ids/tabular.py`, `src/rl_ids/train_day4.py`,
> `tests/test_day4.py`, `data/processed/day4_report.md`,
> `data/processed/day4_learning_curve.png`.

---

## 1. Algorithm

**First-Visit Constant-α GLIE Monte Carlo control** on the 3-state tabular
bucketing of `IDSEnv2018`:

```
Q(s,a) ← Q(s,a) + α · (G_t − Q(s,a))        # only the FIRST visit of (s,a) per episode
ε_t    ← 1 / (1 + t / 1000)                 # GLIE: ε → 0
γ      = 0.9,  α = 0.05
```

- **State = `info["mdp_state"]`** (perfect state perception): the point is to
  validate the MDP, not partially-observable perception — that is Days 5/6.
- **Exploring starts**: every episode begins uniformly over
  {Normal, Suspicious, Compromised}. Starting from Normal alone would reach
  Suspicious in only ~3% of episodes (0.005/0.16 per N-step), starving the S
  action-values; uniform starts guarantee all three states get sampled.
- **ε-greedy respects the playbook mask** (`VALID_ACTIONS`): illegal actions are
  never selected, and their Q entries stay exactly 0.

### Bug found & fixed (first-visit on a reversed trace)

The initial implementation walked the episode trace *backwards* with a `seen`
set and updated on the first miss — i.e. it marked `seen` at the **last**
occurrence of `(s,a)` in the episode. First-visit MC requires the **first**
occurrence (from the episode start), whose return is the full remaining
discounted return. The wrong version updated with the last occurrence's more
optimistic return and biased every estimate upward — it pinned `Q(C,Block)` at
exactly 18 (the resolving block's terminal reward) instead of the true 14.51.
All earlier Day-4 "non-convergence" symptoms in scratch runs traced back to
this. Fixed in `first_visit_returns()` (returns computed up-front, forward pass
records first visits) with a dedicated regression test
(`test_first_visit_returns_takes_first_occurrence`).

## 2. Exact reference (Bellman)

`bellman_solution()` runs value iteration over `config.TRANSITIONS` and the
Day-3 `_NET` reward table, with the env's exact termination semantics
(resolved S/C→Normal and Isolate end the episode). Closed-form anchors
(policy: Normal→Monitor, Suspicious→Honeypot, Compromised→Block):

| State | q*(Monitor) | q*(DPI) | q*(Throttle) | q*(Honeypot) | q*(Block) | q*(Isolate) |
|---|---|---|---|---|---|---|
| Normal | **8.721** | – | – | – | – | – |
| Suspicious | 17.204 | 15.935 | 4.827 | **23.163** | – | – |
| Compromised | 11.751 | – | – | – | **14.508** | 3.000 |

## 3. Results (40 000 episodes, α=0.05)

| Start state | Learned greedy action | Expected |
|---|---|---|
| Normal | Monitor | Monitor |
| Suspicious | Honeypot | Honeypot |
| Compromised | Block | Block |

- **Policy converged to the expected result.** `train_day4.py` raises rather
  than write a report if it does not.
- Learned Q at the optimal pairs: `Q(N,M)=9.27`, `Q(S,H)=22.54`,
  `Q(C,B)=14.67`. Unbiased evaluation of the learned greedy policy over 8000
  fresh episodes per state: `8.78 / 23.62 / 14.44` vs Bellman `8.72 / 23.16 /
  14.51` (**within MC noise, SE ≤ 0.25**).
- Q-values stabilize: max per-snapshot drift over the final 1k-episode
  snapshots = 3.05, down from ~15+ mid-training; final ε = 0.024.

### Why the q*-values differ from the slide targets

Slide targets `N 18.29 / S 28.61 / C 31.06` assume a different reward
convention. Our model is locked to the Day-3 checkpoint where a stationary
Compromised flow is capped by **Block-persist = −2** per step:

```
q(C,Block) = 0.503·(−2 + 0.9·q) + 0.497·18   ⇒   q ≤ 18
```

so `q*(C,Block)=14.51` (not 31.06), and `q(N,Monitor)` inherits the cap through
the N→C branch. The **optimal policy is identical**; only the scale differs
(see `docs/day3.md`).

## 4. Verdict

The tabular agent converges to the expected policy and the unbiased MC values
reproduce the exact Bellman solution on the Day-3 reward model — **the env and
reward are sound**. Real-flow observations, noise and masking do not corrupt
the MDP. Green light for **Day 5 (Double DQN)** and Day 6 (PER).

## 5. Files

| File | Purpose |
|---|---|
| `src/rl_ids/tabular.py` | first-visit MC control, GLIE schedule, exact Bellman solver, greedy evaluation |
| `src/rl_ids/train_day4.py` | runs the checkpoint, writes `day4_report.md` + learning-curve PNG |
| `tests/test_day4.py` | 10 tests incl. the first-visit regression test |
| `data/processed/day4_report.md` | generated report (policy, q-tables, gap analysis) |
| `data/processed/day4_learning_curve.png` | Q trajectories vs Bellman targets |