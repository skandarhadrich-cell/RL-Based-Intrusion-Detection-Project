# Coverage Ablation — Adding the Missing Attack Concepts

> **Goal:** the mask-free agents fail on 2018 `Infilteration` (recall ≈ 0.02) and
> only half-catch 2017 `PortScan` (recall ≈ 0.41). The diagnosis across the probe
> (`docs/holdout_evaluation.md`) and the policy detector
> (`docs/maskfree_agent.md`) is **missing training coverage**. This phase turns
> that diagnosis into an experiment: add the missing concepts, retrain, and
> measure the detection delta — **additively**, leaving every earlier artifact in
> place as the pure out-of-distribution baseline.
>
> Artifacts: `src/rl_ids/coverage_split.py`, `src/rl_ids/train_coverage.py`,
> `src/rl_ids/coverage_eval.py`, `tests/test_coverage_split.py`,
> `data/processed/coverage_split_report.md`,
> `data/processed/coverage_dqn_report.md`, `coverage_per_report.md`,
> `data/processed/coverage_ablation_report.md`,
> `models/double_dqn_coverage.pt`, `models/double_dqn_per_coverage.pt`.

---

## 1. The coverage gap, measured

From `data/processed/dataset_preprocessing_report.md` and the mask-free reports:

| Bucket | Train-split attack labels | Eval counterpart |
|---|---|---|
| Suspicious | 100% FTP/SSH brute force — **0 PortScan** | 2017 PortScan test |
| Compromised | DDoS/DoS-dominated — **36 Infiltration rows** | 2018 `Infilteration`, 137,720 rows |

So the two families that are missed in evaluation are exactly the two families
with **no (or negligible) training representation** — while `DDoS`, which *is*
covered (≈1.24M 2018 `HOIC`/`LOIC` rows share the `Compromised` bucket), transfers
and is recalled at ≈0.36. Coverage, not tuning, sets the ceiling.

This is a **missing-concept** problem, not class imbalance: the environment draws
observations from a per-state pool with a `head(200 000)` cap and uniform
exploring starts, so the agent's *exposure* to each bucket is already balanced
(≈200k/168k/200k). Re-sampling cannot invent a concept that is absent; only adding
it can.

## 2. Where the missing rows come from

The missing concepts live in the **currently held-out days** — which is why they
were never learned. The ablation carves them out and splits each so a fair,
still-unseen remainder stays for testing. `train.parquet` and the original
`holdout_*.parquet` are never touched.

- **PortScan (2017).** The scans are *clustered late* in the day: only ≈3.4k of
  the day's rows sit in the first 80% of the timestamp range. A naive cut would
  strand them on one side, so the split is an **80/20 time cut** (earlier → train,
  later → test), which keeps scans on both sides — train 3,372 / test 1,813
  Suspicious.
- **Infiltration (2018).** Both days have long attack-free gaps mid-day, so a
  within-day cut is fragile; the split is **day vs day** (Wed-28-02 → train,
  Thu-01-03 → test), mirroring the project's existing day-based holdout
  philosophy.

Full counts and the exact cut epoch are in
`data/processed/coverage_split_report.md`.

**Scaling note.** `preprocess.py` writes the `day_*.parquet` files *before* it
fits the scaler, so they hold **raw** feature values, while `train.parquet` holds
z-scores. The coverage corpus is put back on the train scale by re-applying the
*saved* `scaler.json` — fit on the original train split only, never refit — so the
networks are scored on the same scale they were trained on. (Feeding raw
magnitudes silently destroys detection: an early build had baseline PortScan
recall collapse from 0.41 to 0.006.) `tests/test_coverage_split.py` now guards
this.

## 3. What the agent actually sees (representation)

The coverage rows are merged into the training corpus and **RNG-shuffled** (seed
7) so the per-state `head()` cap samples them in proportion instead of truncating
them away. The driver asserts the ablated concepts survive the cap before
training — otherwise the ablation would silently compare against *no* added data:

| State | Added | In pool (after cap) | New concept in pool |
|---|---:|---:|---|
| Normal | 429,414 | 200,000 | — |
| Suspicious | 3,372 | 171,373 | 3,372 `PortScan` (2.0%) |
| Compromised | 63,106 | 200,000 | 5,492 `Infilteration` (2.7%) |

**Honest caveat:** because the dominant families are far larger, each new concept
is only ≈2% of its pool. So this experiment tests whether *coverage at natural
proportion* is sufficient. If detection barely moves, the binding constraint is
**representation**, not existence — a follow-up (class-balanced pools or a larger
cap) would be the next lever, not more of the same tuning.

## 4. Retraining: only the data changed

The coverage agents are retrained with the **same** network, episodes (5,000),
seed (7), observation pipeline and pool cap as the mask-free baselines. The single
difference is the augmented corpus — so any change in behaviour is attributable to
*coverage*. Both agents recover the same playbook policy
(`Monitor` / `Honeypot` / `Block`); details in
`data/processed/coverage_dqn_report.md` and `coverage_per_report.md`.

## 5. The ablation

Baseline (no coverage) and coverage agents are scored on the **same** unseen test
sets, using the agent's own greedy action as the decision (mask-free, no probe,
no label leak). The headline is the attack-label **recall** — read together with
the benign false-alarm rate. Corrupted observations:

| Test set | Attack label | Architecture | Baseline recall | Coverage recall | Δ recall | Baseline FPR | Coverage FPR |
|---|---|---|---:|---:|---:|---:|---:|
| PortScan (2017) | `PortScan` | DQN | 0.226 | **0.564** | +0.338 | 0.068 | 0.227 |
| PortScan (2017) | `PortScan` | PER | 0.108 | **0.312** | +0.204 | 0.086 | **0.072** |
| Infiltration (2018) | `Infilteration` | DQN | 0.017 | 0.080 | +0.062 | 0.025 | 0.072 |
| Infiltration (2018) | `Infilteration` | PER | 0.089 | 0.091 | +0.001 | 0.107 | 0.079 |

Each row is a **within-architecture** ablation: same network, seed, episodes and
pool cap; only the training data changed.

**Caveat.** The PortScan test is the last 20% of the PortScan day, i.e. a
**within-day temporal split** time-adjacent to the PortScan training portion — so
its gain is an optimistic bound. The Infiltration test is a **separate day**
(Thu-01 vs Wed-28) and is clean, so its smaller movement is the conservative
reading.

![Coverage ablation confusion matrices](../data/processed/coverage_ablation_confusion.png)

## 6. Reading the result

**PortScan (a truly missing concept): coverage is the fix.** Recall jumps in both
architectures — DQN 0.226 → 0.564, PER 0.108 → 0.312 — on a family that had
**zero** training rows before. The diagnosis is confirmed: for an absent family,
adding it is what moves detection.

**Read the false-alarm rate with the recall.** The DQN's PortScan gain costs a
false-alarm rise (0.068 → 0.227), while PER buys a comparable recall jump at a
*flat* FPR (0.086 → 0.072) — PER is the clean win. *Which* agent consumes the new
coverage matters: PER turns it into recall without turning it into false alarms.

**Infiltration (added but diluted): coverage helps less.** DQN recall rises
0.017 → 0.080, PER is flat (0.089 → 0.091). Existence was necessary but not
sufficient: at natural proportion the concept stays ≈2.7% of the Compromised pool,
too diluted to dominate the DDoS/DoS mass. The binding constraint has **shifted
from coverage to representation** — the next lever is class-balanced pools or a
larger cap, not more of the same tuning.

**Two failures, two different fixes.** PortScan was *absent* → coverage fixed it.
Infiltration was *rare* → coverage alone is not enough. The mask-free results in
`maskfree_holdout_detection_report.md` remain the honest **out-of-distribution**
baseline; the coverage agents are the **in-coverage** counterpart.

## 7. Run it

```
PYTHONPATH=src python -m rl_ids.coverage_split    # build the additive corpus
PYTHONPATH=src python -m rl_ids.train_coverage     # retrain DQN + PER on it
PYTHONPATH=src python -m rl_ids.coverage_eval      # baseline-vs-coverage ablation
```

## 8. Files

| File | Purpose |
|---|---|
| `src/rl_ids/coverage_split.py` | carve + split the additive coverage corpus |
| `src/rl_ids/train_coverage.py` | retrain mask-free DQN + PER on the augmented corpus |
| `src/rl_ids/coverage_eval.py` | baseline-vs-coverage ablation + figure |
| `src/rl_ids/config.py` | coverage paths, split fraction, pool-representation floors |
| `tests/test_coverage_split.py` | split contract (additive, disjoint, deterministic) |
| `data/processed/coverage_split_report.md` | split counts + cut epoch |
| `data/processed/coverage_dqn_report.md` / `coverage_per_report.md` | training reports |
| `data/processed/coverage_ablation_report.md` | the ablation |
| `docs/coverage_ablation.md` | this document |
