# 🛡️ RL-Based Intrusion Detection — DQN Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Gymnasium-1.x-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-CSE--CIC--IDS2018%20%2B%20CIC--IDS2017-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/tests-174%20passed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

> An autonomous network intrusion-detection and response agent trained with
> value-based reinforcement learning — a **tabular Monte-Carlo control**
> sanity check first, then **Double DQN + Prioritized Experience Replay** —
> inside a custom **Gymnasium** environment. Telemetry comes from
> **CSE-CIC-IDS2018** and **CIC-IDS2017**; the attack ground-truth label is used
> **only for the reward and for evaluation**, never as an observation. Detection
> is finally scored **without** the action mask, so the agent's own action *is*
> the decision.

---

## 📑 Table of Contents

- [Results at a Glance](#-results-at-a-glance)
- [Overview](#-overview)
- [MDP Formulation](#-mdp-formulation)
- [State Space](#-state-space-s)
- [Action Space](#-action-space-a)
- [Reward Function](#-reward-function-r)
- [Environment](#-custom-gymnasium-environment)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Phase Status & Docs](#-phase-status--docs)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Course Objectives](#-course-objectives)

---

## ✨ Results at a Glance

A value-based RL agent that learns **when and how to respond** to network
intrusions from raw flow telemetry — validated against an exact reference at every
stage, then challenged on data it has never seen.

| Stage | Headline result |
|-------|-----------------|
| **Tabular MC control** | Converges to the exact Bellman optimal policy (`8.72 / 23.16 / 14.51`) — the env and reward are sound *before* any neural net. |
| **Double DQN** | Recovers the full IR playbook (Monitor / Honeypot / Block) and beats a random policy in every bucket. |
| **+ Prioritized Replay** | Sharpens the hardest bucket (Suspicious 20.17 → 20.65) with the same agent, env and seed. |
| **Mask-free (Path 2)** | Same optimal policy **without** the action mask — detection read straight off the agent's own actions: 2017 `DDoS` recall **0.36**, `PortScan` **0.41** at a ~**0.10** benign false-alarm rate. |
| **Coverage ablation** | Adding the missing attack families lifts `PortScan` recall **0.11 → 0.31** (PER), proving the ceiling is **data coverage, not tuning**. |

> **The one-line finding.** The agent reliably detects the attack families it was
> trained on; once measured honestly (mask removed), its failures are a
> *data-coverage* limit — demonstrated by adding the missing concepts and watching
> recall rise.

Every number is backed by a committed report under
[`data/processed/`](data/processed/) and a phase doc under [`docs/`](docs/).

---

## 🔍 Overview

Signature-based IDS rely on static rules that fail against novel or adaptive
threats. This project replaces the rule set with a **self-learning agent** that
observes real network-flow telemetry, infers the likely severity of the
current traffic, and selects a defensive action from a fixed incident-response
playbook.

The agent is built incrementally and each stage is verified against an exact
reference before the next one is added:

1. **Preprocessing** — turn raw CICFlowMeter CSVs into a clean, z-scored,
   labeled feature table (zero NaN/Inf), with explicit train/holdout splits.
2. **Environment** — a custom `gym.Env` with partial observability (noise +
   telemetry masking) whose transitions are grounded in the real data.
3. **Reward function** — a pure, stateless `R(s, a, s')` with confusion-matrix
   payoffs and operating costs.
4. **Tabular sanity check** — first-visit Monte-Carlo control on a 3-state
   abstraction, checked against the exact Bellman solution.
5. **Deep RL** — **Double DQN** with a target network and **Prioritized
   Experience Replay**.
6. **Honest detection** — remove the action mask so the agent's own action *is*
   the detection decision, then add the missing attack concepts to show the
   remaining failures are a **data-coverage** limit, not a tuning one.

Everything is organized so a single label→state mapping, feature list, and
transition/reward model are the only sources of truth.

---

## 🧮 MDP Formulation

The defense problem is cast as a finite **Markov Decision Process**:

$$\mathcal{M} = (S,\ A,\ P,\ R,\ \gamma)$$

| Symbol | Description |
|--------|-------------|
| $S$ | 17-dimensional continuous observation vector (partially observable telemetry) |
| $A$ | 6 discrete incident-response actions |
| $P$ | Stochastic transition function $P(s' \mid s, a)$ over the hidden 3-state MDP |
| $R$ | Scalar reward $R(s, a, s')$ = detection payoff − action cost − latency penalty |
| $\gamma$ | Discount factor, $\gamma = 0.9$ |

The agent maximises the expected discounted return
$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.

The **true** MDP state (the severity bucket) is hidden; the agent only sees a
noisy, incomplete observation of one real flow. The hidden state is exposed in
`info` **for reward computation and evaluation only**.

---

## 📡 State Space $S$

Each observation is a `Box(17,)` vector: 17 normalized CICFlowMeter features of
one real flow, corrupted by Gaussian noise ($\sigma = 0.05$) and random
"missing telemetry" masking (per-feature probability $0.1$).

| Group | Features |
|-------|----------|
| Packet rate | `Flow Pkts/s`, `Fwd Pkts/s`, `Bwd Pkts/s` |
| Byte count / rate | `Flow Byts/s`, `TotLen Fwd Pkts`, `TotLen Bwd Pkts` |
| Concurrency (derived) | `Active Conn Cnt` |
| Flow context | `Flow Duration`, `Pkt Len Mean` |
| TCP flags | `FIN`, `SYN`, `RST`, `PSH`, `ACK`, `URG` flag counts |
| Context | `Dst Port Bucket`, `Protocol` |

> `Active Conn Cnt` is **derived** (a sweep-line count of flows concurrent at
> each flow's start), not a shipped column — the same derivation is used for
> both datasets.

The hidden columns — `attack_label`, `mdp_state`, `TimestampEpoch`, `source_day`,
`source_dataset` — are kept for reward/evaluation and **never** fed to the agent.

---

## 🎮 Action Space $A$

The agent selects from **6 discrete defensive actions** of the IR playbook, each
with an operating cost:

| ID | Action | Description | Cost |
|----|--------|-------------|-----:|
| `0` | **Monitor** | Observe only; no intervention | 0.0 |
| `1` | **DPI** | Deep-packet inspection | 1.0 |
| `2` | **Throttle** | Rate-limit the flow | 1.5 |
| `3` | **Honeypot** | Lure the flow into a decoy | 3.0 |
| `4` | **Block** | Drop the flow | 2.0 |
| `5` | **Isolate** | Hard quarantine (terminal) | 15.0 |

A **playbook mask** restricts which actions are legal per hidden state; illegal
actions are never offered to the policy and are penalised (`−100`) if taken
anyway.

| Hidden state | Allowed actions |
|--------------|-----------------|
| Normal | Monitor |
| Suspicious | Monitor, DPI, Throttle, Honeypot |
| Compromised | Monitor, Block, Isolate |

---

## 🏆 Reward Function $R$

The reward is a pure, stateless function:

$$R(s,a,s') = R_{\text{detection}}(\text{event}) - C_{\text{action}}(a) - P_{\text{latency}}(a)$$

**Detection payoff** (confusion-matrix event):

| Event | Value | Meaning |
|-------|------:|---------|
| TP | +18 | attack detected and resolved |
| TN | +1 | benign flow, light touch |
| FP | −12 | heavy action wasted on benign traffic |
| FN | −40 | attack slipped through / escalated |
| NONE | 0 | pure-monitor step |

**Latency penalty** — sustained DPI/Throttle use is billed
$P_{\text{latency}} = \alpha \cdot \max(0, \text{run} - 1)$ with
$\alpha = 0.25$, where `run` is the number of consecutive DPI/Throttle steps.

The env bills latency through the same pure-function component, so online
training rewards and offline replay agree step-for-step.

---

## 🌐 Custom Gymnasium Environment

`src/rl_ids/env.py` implements `IDSEnv2018`. `reset()` starts in the Normal
state; an episode ends when the hidden state resolves, when the agent
**Isolates**, or at `max_steps` (default 100). Next observations are always
**real flow rows** drawn from the pool matching the sampled next state — never
synthetic vectors.

```python
from rl_ids.env import IDSEnv2018

env = IDSEnv2018(seed=1)
obs, info = env.reset(seed=1)
print(obs.shape, info["action_mask"])   # (17,)  [True, False, ...]
obs, r, terminated, truncated, info = env.step(0)   # Monitor
```

Transitions are stochastic and dataset-grounded (`config.TRANSITIONS`); an
empirical flow-sequence matrix (`env.empirical_state_transitions`) is also
available for realism sweeps.

---

## 📦 Datasets

Training and offline evaluation use two Canadian Institute for Cybersecurity
datasets, merged into **one 17-feature schema** (identical meaning and units).

| Dataset | Source | Role | Merged train rows |
|---------|--------|------|------------------:|
| **CSE-CIC-IDS2018** | AWS S3 CICFlowMeter export (10 CSVs, ~6.9 GB) | primary training source | 14,650,066 |
| **CIC-IDS2017** | `GeneratedLabelledFlows` / *TrafficLabelling* archive | additional training data | 2,098,059 |
| **Merged train** | — | used to fit the scaler and (later) the agent | **16,748,125** |

**Eval-only holdouts** (never trained on, never scaler-fit):

| Split | Rows | Days |
|-------|-----:|------|
| `holdout_infiltration` (2018) | 701,039 | Wed-28-02, Thu-01-03 (Infiltration) |
| `holdout_ids2017_portscan_ddos` (2017) | 346,574 | Fri Afternoon PortScan + DDoS |

Full methodology, feature-parity checks and caveats:
[`docs/dataset_preprocessing.md`](docs/dataset_preprocessing.md) and the
generated [`data/processed/dataset_preprocessing_report.md`](data/processed/dataset_preprocessing_report.md).

---

## 📁 Project Structure

```
RL-Based-Intrusion-Detection-Project/
│
├── src/rl_ids/
│   ├── config.py            # single source of truth: paths, features, labels, MDP
│   ├── preprocess.py        # raw CSVs -> labeled, normalized Parquet + report
│   ├── inspect_source.py    # pre-integration raw-CSV inspection tool
│   ├── env.py               # IDSEnv2018 (Gymnasium environment)
│   ├── reward.py            # pure reward model + event classification
│   ├── tabular.py           # first-visit MC control + exact Bellman solver
│   ├── train_tabular.py     # tabular sanity-check driver (report + curve)
│   ├── dqn.py               # Double DQN agent: network, replay, masked update
│   ├── train_dqn.py         # Double DQN training driver (report + curve)
│   ├── per.py               # sum-tree prioritized replay buffer + β schedule
│   ├── evaluate.py          # shared greedy/random rollout evaluation
│   ├── train_per.py         # PER training driver (report + curve)
│   ├── metrics.py           # pure per-class precision/recall/F1/FPR functions
│   ├── holdout_eval.py      # probe-based holdout detection evaluation driver
│   ├── train_tabular_maskfree.py  # mask-free tabular gate (perfect perception)
│   ├── train_maskfree.py    # mask-free Double DQN + PER training driver
│   ├── maskfree_detect.py   # policy-based holdout detection (non-circular)
│   ├── coverage_split.py    # additive coverage corpus (missing attack concepts)
│   ├── train_coverage.py    # retrain DQN + PER on the augmented corpus
│   ├── coverage_eval.py     # baseline-vs-coverage ablation
│   └── validate_reward.py   # reward verification report + replay check
│
├── tests/                   # pytest suite (174 tests)
│   ├── test_dataset_preprocessing.py
│   ├── test_ids2017_integration.py
│   ├── test_environment.py
│   ├── test_reward_function.py
│   ├── test_tabular_mc_control.py
│   ├── test_dqn.py
│   ├── test_per.py
│   ├── test_holdout_metrics.py
│   ├── test_maskfree.py
│   └── test_coverage_split.py
│
├── docs/
│   ├── dataset_preprocessing.md   # acquisition, cleaning, merge, splits
│   ├── environment.md             # MDP, transitions, observation model
│   ├── reward_function.md         # reward design + worked example
│   ├── tabular_mc_control.md      # MC control vs exact Bellman solution
│   ├── double_dqn.md              # Double DQN agent design + training
│   ├── prioritized_replay.md      # prioritized replay: sum-tree, IS, results
│   ├── holdout_evaluation.md      # per-class P/R/F1/FPR on the untouched holdouts
│   ├── maskfree_agent.md          # Path 2: mask removed, retrain, detect from actions
│   └── coverage_ablation.md       # add the missing concepts, measure the delta
│
├── data/
│   ├── raw/                 # downloaded CSVs (git-ignored)
│   └── processed/           # Parquet splits, scaler.json, generated reports
│
├── models/                  # trained weights (git-ignored)
│
├── download_ids2018.sh      # fetch CSE-CIC-IDS2018 CSVs
├── download_ids2017.sh      # fetch + unpack CIC-IDS2017 archives
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU (recommended for the DQN phase) or CPU

### Setup

```bash
# 1. Clone the repository
git clone git@github.com:skandarhadrich-cell/RL-Based-Intrusion-Detection-Project.git
cd RL-Based-Intrusion-Detection-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` (core)

```
torch>=2.0.0
gymnasium>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tabulate>=0.9.0
pytest>=7.4.0
```

---

## 🚀 Usage

### 1. Download the raw data

```bash
./download_ids2018.sh     # 10 CSE-CIC-IDS2018 CSVs  -> data/raw/
./download_ids2017.sh     # CIC-IDS2017 archives       -> data/raw/ids2017/
```

### 2. Preprocess (build the merged train/holdout sets + scaler)

```bash
PYTHONPATH=src python -m rl_ids.preprocess
```

This writes `data/processed/{train,holdout_infiltration,holdout_ids2017_portscan_ddos}.parquet`,
`scaler.json`, and regenerates `data/processed/dataset_preprocessing_report.md`.

### 3. Verify the reward model

```bash
PYTHONPATH=src python -m rl_ids.validate_reward   # -> reward_function_report.md
```

### 4. Run the tabular sanity check

```bash
PYTHONPATH=src python -m rl_ids.train_tabular     # -> tabular_mc_control_report.md + curve
```

### 5. Train the Double DQN agent

```bash
PYTHONPATH=src python -m rl_ids.train_dqn         # -> double_dqn_report.md + curve + models/double_dqn.pt
```

This is the heavy online step: it loads real-flow observation pools into memory,
so give the machine enough RAM (or lower `MAX_ROWS_PER_STATE` in
`src/rl_ids/train_dqn.py`).

### 6. Train with Prioritized Experience Replay

```bash
PYTHONPATH=src python -m rl_ids.train_per         # -> prioritized_replay_report.md + curve + models/double_dqn_per.pt
```

Same agent and settings as step 5, but with proportional prioritized replay
(sum-tree sampling + importance-sampling correction). It is the other heavy
online step.

### 7. Per-class detection evaluation on the untouched holdouts

```bash
PYTHONPATH=src python -m rl_ids.holdout_eval    # -> holdout_classification_report.md + holdout_confusion_matrix.png
```

Loads both trained checkpoints and measures **per-class precision / recall /
F1 / FPR** on the two eval-only holdout days. No retraining. The action mask is
*not* handed to the evaluation — it is a one-to-one function of the hidden
severity bucket, so using it would make detection trivially perfect. Instead a
logistic-regression probe is fitted on the frozen 128-d layer of the Q-network
(train split only) and tested on the holdouts, with the raw-17-feature probe as
a no-representation baseline.

### 8. Mask-free agent (Path 2): drop the mask and retrain

```bash
PYTHONPATH=src python -m rl_ids.train_tabular_maskfree   # tabular gate (perfect perception)
PYTHONPATH=src python -m rl_ids.train_maskfree           # mask-free Double DQN + PER
PYTHONPATH=src python -m rl_ids.maskfree_detect          # policy-based holdout detection
```

Removes the playbook action mask — every action is offered in every state — and
retrains both agents so they must **infer** the hidden severity bucket from
telemetry. The mask-free MDP has the *same* optimal policy and values as the
masked one (verified via the exact Bellman solution), so any policy the agent
finds it found by perceiving the state. With no mask, the agent's own action *is*
the detection decision, which is what makes the holdout detection non-circular.

### 9. Coverage ablation: add the missing attack concepts

```bash
PYTHONPATH=src python -m rl_ids.coverage_split    # build the additive coverage corpus
PYTHONPATH=src python -m rl_ids.train_coverage     # retrain DQN + PER on it
PYTHONPATH=src python -m rl_ids.coverage_eval      # baseline-vs-coverage ablation
```

Carves the missing `PortScan` (2017) and `Infiltration` (2018) rows out of the
held-out days, splits each so an unseen remainder stays for testing (80/20 time
cut for PortScan, day-vs-day for Infiltration), retrains both agents on the
augmented corpus with **everything else held fixed**, and scores baseline vs
coverage on identical unseen test sets. Purely additive — the original
`train.parquet`, the holdouts and their reports are untouched.

### 10. Run the test suite

```bash
python -m pytest tests/ -q
```

*(In this working copy the project uses a dedicated venv at `~/venvs/rl-ids`;
replace `python` with its interpreter if needed.)*

---

## 📚 Phase Status & Docs

| Phase | Status | Doc |
|-------|--------|-----|
| Dataset acquisition & preprocessing | ✅ done | [`docs/dataset_preprocessing.md`](docs/dataset_preprocessing.md) |
| Gymnasium environment | ✅ done | [`docs/environment.md`](docs/environment.md) |
| Reward function | ✅ done | [`docs/reward_function.md`](docs/reward_function.md) |
| Tabular MC sanity check | ✅ done | [`docs/tabular_mc_control.md`](docs/tabular_mc_control.md) |
| Double DQN | ✅ done | [`docs/double_dqn.md`](docs/double_dqn.md) |
| Prioritized Experience Replay | ✅ done | [`docs/prioritized_replay.md`](docs/prioritized_replay.md) |
| Holdout detection evaluation | ✅ done | [`docs/holdout_evaluation.md`](docs/holdout_evaluation.md) |
| Mask-free agent (Path 2) | ✅ done | [`docs/maskfree_agent.md`](docs/maskfree_agent.md) |
| Coverage ablation (add missing concepts) | ✅ done | [`docs/coverage_ablation.md`](docs/coverage_ablation.md) |

---

## 📊 Results

The tabular **first-visit Monte-Carlo control** run (40,000 episodes, α = 0.05)
converges to the expected optimal policy on a 3-state abstraction, and its
unbiased evaluation reproduces the exact Bellman solution:

| Start state | Converged action | Bellman q* | MC evaluation |
|-------------|------------------|-----------:|--------------:|
| Normal | Monitor | 8.72 | 8.78 |
| Suspicious | Honeypot | 23.16 | 23.62 |
| Compromised | Block | 14.51 | 14.44 |

This confirms the environment and reward model are sound **before any neural
network is involved** — real-flow observations, noise and masking do not corrupt
the underlying MDP.

![Q-value trajectories vs Bellman targets](data/processed/tabular_mc_control_learning_curve.png)

The **Double DQN** agent (5,000 episodes, seed 7) then recovers the *same*
optimal response policy from the partially-observable telemetry alone, beating a
uniform-random policy in every bucket:

| Start state | Greedy action | Greedy return | Random | Bellman optimum |
|-------------|---------------|--------------:|-------:|----------------:|
| Normal | Monitor | 8.58 | 3.72 | 8.72 |
| Suspicious | Honeypot | 20.17 | 5.18 | 23.16 |
| Compromised | Block | 14.39 | 6.36 | 14.51 |

![Double DQN learning curve](data/processed/double_dqn_learning_curve.png)

Adding **Prioritized Experience Replay** (same agent, environment, episodes and
seed — only the buffer changes) keeps the optimal policy in every bucket and
sharpens the one bucket that had room to improve:

| Start state | Uniform greedy | PER greedy | Random | Bellman optimum |
|-------------|---------------:|-----------:|-------:|----------------:|
| Normal | 8.58 | 8.47 | 3.72 | 8.72 |
| Suspicious | 20.17 | **20.65** | 5.18 | 23.16 |
| Compromised | 14.39 | 14.30 | 6.36 | 14.51 |

The Suspicious gap to the exact optimum shrinks **2.99 → 2.51**; Normal and
Compromised were already within ≈0.15 and barely move.

![PER learning curve](data/processed/prioritized_replay_learning_curve.png)

### Holdout detection (per-class precision / recall / F1 / FPR)

The two trained agents are then scored on the **untouched holdout days** —
`holdout_infiltration` (CSE-CIC-IDS2018) and
`holdout_ids2017_portscan_ddos` (CIC-IDS2017) — using a logistic-regression
probe on the frozen 128-d layer of `Q_online` fitted on the train split only.
Corrupted observations (noise + telemetry masking, as at training time), 3,000
flows per bucket:

| Model | Holdout | Accuracy | Macro F1 | Mean attack recall | Benign FPR |
|-------|---------|---------:|---------:|-------------------:|-----------:|
| uniform DQN | Infiltration (2018) | 0.468 | 0.267 | 0.166 | 0.173 |
| PER | Infiltration (2018) | 0.473 | 0.271 | 0.170 | 0.168 |
| raw-17 baseline | Infiltration (2018) | 0.456 | 0.284 | 0.264 | 0.256 |
| uniform DQN | PortScan + DDoS (2017) | 0.620 | 0.533 | 0.746 | 0.141 |
| PER | PortScan + DDoS (2017) | **0.641** | **0.556** | 0.724 | **0.138** |
| raw-17 baseline | PortScan + DDoS (2017) | 0.605 | 0.524 | 0.617 | 0.159 |

The pattern is a **coverage** story, not a tuning one: the 2017 holdout's DDoS
family is represented in training by ~1.24M rows of 2018 `HOIC`/`LOIC` traffic,
and PER recalls DDoS at **0.967** (F1 0.919). `PortScan` (0.480) and
`Infilteration` (0.170) have no counterpart in training and are missed. The
per-class tables also make the **mask leak** explicit: supplying the true action
mask scores **0.999** accuracy, while mask-free Q-argmax scores **0.199** —
so a probe, not the agent's actions, is the only honest detection signal here.

![Holdout confusion matrices](data/processed/holdout_confusion_matrix.png)

### Mask-free agent (Path 2): detect from the agent's own actions

The probe above works around the mask, but it needs an extra classifier. **Path 2
removes the mask instead**: every action is offered in every state, both agents
are retrained, and they must infer the hidden bucket from telemetry. The
mask-free MDP has the *same* exact Bellman optimum (`8.721 / 23.163 / 14.508`), so
the target policy is unchanged — only the mask is gone. Both agents recover the
**full playbook policy** (Monitor / Honeypot / Block):

| Start state | Masked uniform | Mask-free uniform | Masked PER | Mask-free PER |
|-------------|---------------:|------------------:|-----------:|--------------:|
| Normal | 8.58 | 6.54 | 8.47 | 5.18 |
| Suspicious | 20.17 | 20.10 | 20.65 | 20.67 |
| Compromised | 14.39 | 13.00 | 14.30 | 12.78 |

With no mask the greedy action *is* the detection decision, so it can be scored
directly on the holdouts — no probe, no circularity:

| Model | Holdout | Accuracy | Macro F1 | Benign FPR |
|-------|---------|---------:|---------:|-----------:|
| mask-free uniform | Infiltration (2018) | 0.488 | 0.220 | **0.025** |
| mask-free PER | Infiltration (2018) | 0.451 | 0.214 | 0.099 |
| mask-free uniform | PortScan + DDoS (2017) | 0.457 | 0.394 | 0.097 |
| mask-free PER | PortScan + DDoS (2017) | 0.437 | 0.362 | 0.099 |
| masked uniform / PER | any | ≈0.00 | ≈0.00 | **1.000** |

The masked agents flag **every** benign flow as an attack (FPR = 1.000) — their
mask-free argmax collapses, because nothing ever forced their Q-outputs to encode
the state. The mask-free agents are the only ones that separate benign from
attack: `DDoS` recall 0.36 and `PortScan` recall 0.41 at a ~0.10 false-alarm rate
on 2017, while 2018 `Infilteration` (no training counterpart) stays hard — the
same coverage limit the probe found, now measured end-to-end.

![Mask-free confusion matrices](data/processed/maskfree_holdout_confusion_matrix.png)

### Coverage ablation: prove the failure is a data-coverage limit

The probe and the policy detector agree that the missed families are exactly the
ones with **no training representation** (`PortScan`: 0 rows; `Infilteration`: 36
rows vs 137,720 in the holdout). The coverage ablation adds them — carved from the
held-out days, split so an unseen remainder stays for testing — and retrains both
agents with **only the data changed** (same net, seed, episodes, pool cap). Each
concept is added at its natural proportion (≈2% of its bucket's pool), so this
tests whether coverage *alone* is sufficient:

| Test set | Attack label | Architecture | Baseline recall | Coverage recall | Δ recall | Baseline FPR | Coverage FPR |
|---|---|---|---:|---:|---:|---:|---:|
| PortScan (2017) | `PortScan` | DQN | 0.226 | **0.564** | +0.338 | 0.068 | 0.227 |
| PortScan (2017) | `PortScan` | PER | 0.108 | **0.312** | +0.204 | 0.086 | **0.072** |
| Infiltration (2018) | `Infilteration` | DQN | 0.017 | 0.080 | +0.062 | 0.025 | 0.072 |
| Infiltration (2018) | `Infilteration` | PER | 0.089 | 0.091 | +0.001 | 0.107 | 0.079 |

Two different stories fall out of the same experiment:

- **PortScan — a truly missing concept — is the fix.** Recall jumps in both
  architectures on a family with **zero** prior training rows. PER is the clean
  win: recall **0.108 → 0.312** at a *flat* false-alarm rate (0.086 → 0.072).
  (The DQN's larger jump costs a false-alarm rise, 0.068 → 0.227.)
- **Infiltration — added but rare — barely moves.** DQN 0.017 → 0.080, PER flat.
  At ≈2.7% of the Compromised pool the concept is too diluted to dominate the
  DDoS/DoS mass, so the binding constraint **shifts from coverage to
  representation** (class-balanced pools or a larger cap would be the next lever).

The original mask-free numbers remain the honest **out-of-distribution** baseline;
these are the **in-coverage** counterpart.

![Coverage ablation confusion matrices](data/processed/coverage_ablation_confusion.png)

---

## 🗺️ Roadmap

- [x] Multi-source preprocessing (CSE-CIC-IDS2018 + CIC-IDS2017), splits + scaler
- [x] Partial-observability Gymnasium environment
- [x] Pure reward model + worked-example verification
- [x] Tabular Monte-Carlo control vs exact Bellman solution
- [x] Double DQN with target network (recovers the optimal response policy)
- [x] Prioritized Experience Replay (sum-tree + IS correction; sharpens Suspicious)
- [x] Per-class precision / recall / F1 / FPR on the untouched holdouts
- [x] Mask-free agent (Path 2): drop the mask, retrain, detect from actions
- [x] Coverage ablation: add the missing PortScan/Infiltration concepts and measure the delta

---

## 🎓 Course Objectives

This project exercises the core **Temporal Difference learning** toolkit:

| Objective | Implementation |
|-----------|----------------|
| **TD(0) Q-learning** | off-policy Bellman update with a bootstrapped target |
| **Function approximation** | a Double DQN network replaces the tabular $Q$-table |
| **Experience Replay** | uniform replay + **prioritized** mini-batch sampling (sum-tree, IS-weighted) |
| **Target network** | frozen $\theta^-$ decouples the TD target (hard-copied every 1k updates) |
| **MDP formalisation** | full $(S, A, P, R, \gamma)$ with stochastic transitions |
| **ε-greedy exploration** | balances exploitation vs. exploration (GLIE in the tabular stage) |
| **Reward shaping** | asymmetric confusion-matrix payoffs drive containment behaviour |

---

<p align="center">
  Built for the <strong>Temporal Difference Learning</strong> course · ENIT Cybersecurity Track<br/>
  <em>Autonomous defense. No signatures. No waiting.</em>
</p>
