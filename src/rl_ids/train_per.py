"""Prioritized Experience Replay training driver (PER phase).

Same Double DQN agent and environment as ``train_dqn.py``, but the uniform
replay buffer is replaced by the proportional **prioritized** buffer
(``rl_ids.per``).  Everything else is held fixed (5,000 episodes, seed 7, 200k
rows/state) so the two reports are directly comparable.

Writes:
  * ``data/processed/prioritized_replay_report.md``
  * ``data/processed/prioritized_replay_learning_curve.png``
  * ``models/double_dqn_per.pt`` (trained weights, git-ignored)

Run:  PYTHONPATH=src python -m rl_ids.train_per
NOTE: the online run loads real-flow observation pools into memory.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ids import config as C
from rl_ids.dqn import DoubleDQNAgent, epsilon_schedule
from rl_ids.env import IDSEnv2018
from rl_ids.evaluate import evaluate_greedy, evaluate_random
from rl_ids.per import PrioritizedReplayBuffer, per_beta
from rl_ids.tabular import SIDX, bellman_solution

# --------------------------------------------------------------------------- #
# Hyper-parameters - identical to train_dqn.py except for the buffer
# --------------------------------------------------------------------------- #
TRAIN_EPISODES = 5_000
MAX_ROWS_PER_STATE = C.DQN_MAX_ROWS_PER_STATE
EVAL_EPISODES = 1_000
RANDOM_EPISODES = 1_000
TRACK_EVERY = 50
SEED = 7

EXPECTED_POLICY = {"Normal": "Monitor", "Suspicious": "Honeypot",
                   "Compromised": "Block"}

# Greedy returns of the *uniform* Double DQN run, for the PER-vs-uniform table.
# Source: data/processed/double_dqn_report.md (identical env/seeds).
UNIFORM_GREEDY = {"Normal": 8.578, "Suspicious": 20.173, "Compromised": 14.387}


def main() -> None:
    env = IDSEnv2018(split="train", max_rows_per_state=MAX_ROWS_PER_STATE, seed=SEED)
    agent = DoubleDQNAgent(seed=SEED)
    buffer = PrioritizedReplayBuffer(seed=SEED)
    rng = np.random.default_rng(SEED)

    history = {"episode": [], "return": [], "loss": [], "eps": [], "beta": []}
    recent_returns: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=500)
    total_steps = 0
    last_beta = C.PER_BETA_START

    for ep in range(TRAIN_EPISODES):
        eps = epsilon_schedule(ep)
        start = C.STATES[int(rng.integers(len(C.STATES)))]
        obs, info = env.reset(seed=SEED + ep, options={"state": start})
        mask = info["action_mask"]

        g, disc, done = 0.0, 1.0, False
        while not done:
            a = agent.act(obs, mask, eps, rng)
            next_obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            next_mask = info["action_mask"]
            buffer.push(obs, a, r, next_obs, done, next_mask)
            obs, mask = next_obs, next_mask
            g += disc * float(r)
            disc *= C.GAMMA
            total_steps += 1
            if len(buffer) >= C.DQN_REPLAY_WARMUP:
                last_beta = per_beta(agent.updates)
                batch = buffer.sample(C.DQN_BATCH_SIZE, beta=last_beta)
                loss, td = agent.learn(batch, weights=batch["weights"])
                buffer.update_priorities(batch["indices"], td)
                recent_losses.append(loss)
        recent_returns.append(g)

        if ep % TRACK_EVERY == 0 or ep == TRAIN_EPISODES - 1:
            history["episode"].append(ep)
            history["return"].append(float(np.mean(recent_returns)))
            history["loss"].append(float(np.mean(recent_losses)) if recent_losses else float("nan"))
            history["eps"].append(eps)
            history["beta"].append(last_beta)

    # ---------------------------------------------------------------------- #
    # evaluation
    # ---------------------------------------------------------------------- #
    greedy = evaluate_greedy(agent, env, seed=SEED + 90_000, episodes=EVAL_EPISODES)
    random_base = evaluate_random(env, seed=SEED + 95_000, episodes=RANDOM_EPISODES)

    Qb = bellman_solution()
    optimum = {s: float(np.max(Qb[SIDX[s], list(C.VALID_ACTIONS[s])])) for s in C.STATES}

    policy = {s: greedy[s][2] for s in C.STATES}
    policy_match = {s: policy[s] == EXPECTED_POLICY[s] for s in C.STATES}
    beats_random = {s: greedy[s][0] > random_base[s][0] for s in C.STATES}

    if sum(greedy[s][0] for s in C.STATES) <= sum(random_base[s][0] for s in C.STATES):
        raise RuntimeError(
            "PER agent failed to beat a uniform-random policy - inspect the "
            "agent/buffer wiring before trusting the run."
        )

    agent.save(C.PER_MODEL_PATH)

    # ---------------------------------------------------------------------- #
    # report
    # ---------------------------------------------------------------------- #
    lines: list[str] = []
    add = lines.append
    add("# Prioritized Experience Replay - training report\n")
    add("`data/processed/prioritized_replay_report.md` - generated by "
        "`PYTHONPATH=src python -m rl_ids.train_per`\n")

    add("## 1. Setup\n")
    add("- Environment: `IDSEnv2018(split=\"train\")`, 17-feature **real-flow** "
        "observations with Gaussian noise and telemetry masking (partial "
        "observability). The hidden severity bucket drives transitions/rewards "
        "and is never an observation.")
    add("- Action mask: the playbook mask (`info[\"action_mask\"]`) is applied at "
        "every argmax (acting and bootstrapping); the Q-network sees only the "
        "17 features.")
    add("- Algorithm: **Double DQN + PER** - online net selects "
        "`a* = argmax_a Q_online(s',a)`, target net evaluates "
        "`y = r + γ·Q_target(s',a*)`; transitions are sampled by priority with "
        "importance-sampling correction.")
    add(f"- Network: MLP {list(C.DQN_HIDDEN)} (ReLU), Adam lr={C.DQN_LR:g}, "
        f"Huber loss, grad-clip={C.DQN_GRAD_CLIP}.")
    add(f"- Replay: **prioritized (proportional)**, capacity {C.DQN_REPLAY_CAPACITY:,}, "
        f"batch {C.DQN_BATCH_SIZE}, α = {C.PER_ALPHA}, ε = {C.PER_EPS}, "
        f"β {C.PER_BETA_START} → {C.PER_BETA_END} over {C.PER_BETA_STEPS:,} updates, "
        f"Warm-up {C.DQN_REPLAY_WARMUP:,} transitions, target sync every "
        f"{C.DQN_TARGET_SYNC:,} updates.")
    add(f"- Exploration: ε {C.DQN_EPS_START:g} → floor {C.DQN_EPS_END:g}, "
        f"×{C.DQN_EPS_DECAY:g} per episode (final ε = {history['eps'][-1]:.4f}).")
    add(f"- Episodes = {TRAIN_EPISODES:,} (exploring starts, uniform over "
        f"{{Normal, Suspicious, Compromised}}), steps = {total_steps:,}, "
        f"updates = {agent.updates:,}.")
    add(f"- Obs pool ≤ {MAX_ROWS_PER_STATE:,} rows/state; γ = {C.GAMMA}; "
        f"seed = {SEED}.\n")

    add("## 2. PER mechanics\n")
    add(f"- Priority `p_i = |TD_i| + ε`, stored as `p_i^α` in a binary **sum-tree** "
        f"(O(log N) sampling and priority updates).")
    add(f"- Sampling `P(i) = p_i^α / Σ p_k^α`; new transitions enter at the current "
        f"**max** priority so nothing starves before its first update.")
    add(f"- Importance-sampling weights `w_i = (N·P(i))^(−β)` (max-normalised) "
        f"multiply the element-wise Huber loss, un-biasing the prioritized sample.")
    add(f"- Final β = {last_beta:.3f} (β reaches 1.0 after {C.PER_BETA_STEPS:,} "
        f"updates, i.e. full correction).\n")

    add("## 3. Learned policy\n")
    add("| Start state | Dominant greedy action | Expected optimal | Match |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {policy[s]} | {EXPECTED_POLICY[s]} | "
            f"{'yes' if policy_match[s] else '**no**'} |")
    add("")
    add("Dominant action = the first action the greedy policy takes most often "
        f"over {EVAL_EPISODES:,} episodes started in that bucket.\n")

    add("## 4. Returns: PER vs random vs exact optimum\n")
    add("| Start state | PER greedy | Random | Exact optimum (Bellman) |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {greedy[s][0]:+.3f} ± {greedy[s][1]:.2f} | "
            f"{random_base[s][0]:+.3f} | {optimum[s]:+.3f} |")
    add("")
    beats_detail = ", ".join(f"{s}: {'yes' if beats_random[s] else 'no'}" for s in C.STATES)
    add(f"- PER beats a uniform-random policy: "
        f"**{'yes' if all(beats_random.values()) else 'no'}** ({beats_detail}).\n")

    add("## 5. PER vs uniform replay\n")
    add("Identical agent, environment, episodes and seeds; only the buffer differs. "
        "The random baseline and the Bellman optimum are shared (env/seed-determined), "
        "so only the greedy column changes.")
    add("")
    add("| Start state | Uniform greedy | PER greedy | Bellman optimum |")
    add("|---|---:|---:|---:|")
    for s in C.STATES:
        add(f"| {s} | {UNIFORM_GREEDY[s]:+.3f} | {greedy[s][0]:+.3f} | {optimum[s]:+.3f} |")
    add("")
    for s in C.STATES:
        d = greedy[s][0] - UNIFORM_GREEDY[s]
        add(f"- **{s}**: PER {d:+.3f} vs uniform.")
    sus_gap_per = optimum["Suspicious"] - greedy["Suspicious"][0]
    sus_gap_uni = optimum["Suspicious"] - UNIFORM_GREEDY["Suspicious"]
    add(f"- Suspicious gap to optimum: uniform {sus_gap_uni:.3f} → PER "
        f"{sus_gap_per:.3f} ({sus_gap_per - sus_gap_uni:+.3f}).\n")

    add("## 6. Training dynamics\n")
    tail_loss = [l for l in history["loss"][-10:] if not np.isnan(l)]
    add(f"- Final ε = {history['eps'][-1]:.5f}; final β = {last_beta:.4f}.")
    loss_line = (f"- Mean Huber loss (weighted, last tracked points) = {np.mean(tail_loss):.4f}"
                 if tail_loss else "- Mean Huber loss: n/a (warm-up not reached).")
    add(loss_line)
    add(f"- Mean episode return (last point) = {history['return'][-1]:+.3f}.")
    add(f"- Learning curve: `{C.PER_CURVE_PNG.name}`\n")
    add(f"![PER learning curve]({C.PER_CURVE_PNG.name})\n")

    add("## 7. Verdict\n")
    if all(policy_match.values()):
        add("PER recovers the **expected optimal policy** in every bucket and beats "
            "the random baseline. ")
    else:
        missed = [s for s in C.STATES if not policy_match[s]]
        add(f"PER beats the random baseline but has **not** recovered the expected "
            f"action in: {', '.join(missed)}. ")
    if sus_gap_per < sus_gap_uni - 0.25:
        add(f"On the Suspicious gap it **helped**: {sus_gap_uni:.2f} → "
            f"{sus_gap_per:.2f} toward the optimum.")
    elif sus_gap_per > sus_gap_uni + 0.25:
        add(f"On the Suspicious gap it **hurt** slightly: {sus_gap_uni:.2f} → "
            f"{sus_gap_per:.2f}.")
    else:
        add(f"On the Suspicious gap it was **neutral** ({sus_gap_uni:.2f} → "
            f"{sus_gap_per:.2f}): prioritisation does not move a gap driven by "
            f"reward variance and telemetry perception rather than under-training.")
    add("\n")

    add("## 8. Files\n")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `src/rl_ids/per.py` | `SumTree`, `PrioritizedReplayBuffer`, β schedule |")
    add("| `src/rl_ids/evaluate.py` | shared greedy/random rollout evaluation |")
    add("| `src/rl_ids/train_per.py` | this driver |")
    add("| `models/double_dqn_per.pt` | trained online-network weights (git-ignored) |")
    add(f"| `data/processed/{C.PER_REPORT_PATH.name}` | this report |")
    add(f"| `data/processed/{C.PER_CURVE_PNG.name}` | learning curve |")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.PER_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------------------- #
    # learning curve
    # ---------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["episode"], history["return"], lw=1.2, color="tab:blue")
    axes[0].set_title("Episode return (100-ep mean)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("discounted return")
    axes[1].plot(history["episode"], history["loss"], lw=1.2, color="tab:orange")
    axes[1].set_title("Huber loss (500-update mean)")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("loss")
    axes[2].plot(history["episode"], history["eps"], lw=1.2, color="tab:green")
    axes[2].set_title("ε (exploration)")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("ε")
    axes[3].plot(history["episode"], history["beta"], lw=1.2, color="tab:red")
    axes[3].set_title("β (IS correction)")
    axes[3].set_xlabel("episode")
    axes[3].set_ylabel("β")
    fig.tight_layout()
    C.PER_CURVE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.PER_CURVE_PNG, dpi=110)
    plt.close(fig)

    print(f"wrote {C.PER_REPORT_PATH}")
    print(f"wrote {C.PER_CURVE_PNG}")
    print(f"wrote {C.PER_MODEL_PATH}")
    print(f"policy={policy}  match={policy_match}")
    print(f"greedy={ {s: round(greedy[s][0], 3) for s in C.STATES} }")
    print(f"random={ {s: round(random_base[s][0], 3) for s in C.STATES} }")
    print(f"updates={agent.updates}  steps={total_steps}  "
          f"final_eps={history['eps'][-1]:.5f}  final_beta={last_beta:.4f}")


if __name__ == "__main__":
    main()
