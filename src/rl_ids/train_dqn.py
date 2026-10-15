"""Double DQN training driver (DQN phase).

Trains a Double DQN agent *online* on ``IDSEnv2018`` — partially-observable
real-flow observations plus the playbook action mask — then evaluates the greedy
policy against a uniform-random baseline and the exact tabular optimum.

Writes:
  * ``data/processed/double_dqn_report.md``
  * ``data/processed/double_dqn_learning_curve.png``
  * ``models/double_dqn.pt`` (trained weights, git-ignored)

Run:  PYTHONPATH=src python -m rl_ids.train_dqn

Prioritized experience replay and the 2018-only vs merged-data A/B come next.
NOTE: the online run loads real-flow observation pools into memory, so raise the
VM RAM (or lower ``MAX_ROWS_PER_STATE``) before running.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ids import config as C
from rl_ids.dqn import DoubleDQNAgent, ReplayBuffer, epsilon_schedule, legal_indices
from rl_ids.env import IDSEnv2018
from rl_ids.tabular import SIDX, bellman_solution

# --------------------------------------------------------------------------- #
# Hyper-parameters (single place to tune before running)
# --------------------------------------------------------------------------- #
TRAIN_EPISODES = 5_000
MAX_ROWS_PER_STATE = C.DQN_MAX_ROWS_PER_STATE
EVAL_EPISODES = 1_000
RANDOM_EPISODES = 1_000
TRACK_EVERY = 50
SEED = 7

EXPECTED_POLICY = {"Normal": "Monitor", "Suspicious": "Honeypot",
                   "Compromised": "Block"}


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def _rollout(agent, env, start, seed, episodes):
    """Greedy-policy returns from a fixed start state; also counts 1st action."""
    rets = np.empty(episodes)
    first_actions: Counter[int] = Counter()
    for k in range(episodes):
        obs, info = env.reset(seed=seed + k, options={"state": start})
        mask = info["action_mask"]
        first_actions[agent.greedy(obs, mask)] += 1
        g, disc, done = 0.0, 1.0, False
        while not done:
            a = agent.greedy(obs, mask)
            obs, r, term, trunc, info = env.step(a)
            mask = info["action_mask"]
            g += disc * float(r)
            disc *= C.GAMMA
            done = term or trunc
        rets[k] = g
    dominant = C.ACTIONS[first_actions.most_common(1)[0][0]]
    return float(rets.mean()), float(rets.std()), dominant, first_actions


def evaluate_greedy(agent, env, seed, episodes=EVAL_EPISODES):
    return {s: _rollout(agent, env, s, seed, episodes) for s in C.STATES}


def evaluate_random(env, seed, episodes=RANDOM_EPISODES):
    rng = np.random.default_rng(seed)
    out = {}
    for s in C.STATES:
        rets = np.empty(episodes)
        for k in range(episodes):
            obs, info = env.reset(seed=seed + k, options={"state": s})
            mask = info["action_mask"]
            g, disc, done = 0.0, 1.0, False
            while not done:
                a = int(rng.choice(legal_indices(mask)))
                obs, r, term, trunc, info = env.step(a)
                mask = info["action_mask"]
                g += disc * float(r)
                disc *= C.GAMMA
                done = term or trunc
            rets[k] = g
        out[s] = (float(rets.mean()), float(rets.std()))
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    env = IDSEnv2018(split="train", max_rows_per_state=MAX_ROWS_PER_STATE, seed=SEED)
    agent = DoubleDQNAgent(seed=SEED)
    buffer = ReplayBuffer(seed=SEED)
    rng = np.random.default_rng(SEED)

    history = {"episode": [], "return": [], "loss": [], "eps": []}
    recent_returns: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=500)
    total_steps = 0

    for ep in range(TRAIN_EPISODES):
        eps = epsilon_schedule(ep)
        # Exploring starts (uniform over the hidden buckets), matching the
        # tabular checkpoint: a Normal-only start reaches Suspicious in <1% of
        # episodes and would starve its action-values.
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
                recent_losses.append(agent.update(buffer.sample(C.DQN_BATCH_SIZE)))
        recent_returns.append(g)

        if ep % TRACK_EVERY == 0 or ep == TRAIN_EPISODES - 1:
            history["episode"].append(ep)
            history["return"].append(float(np.mean(recent_returns)))
            history["loss"].append(float(np.mean(recent_losses)) if recent_losses else float("nan"))
            history["eps"].append(eps)

    # ---------------------------------------------------------------------- #
    # evaluation
    # ---------------------------------------------------------------------- #
    greedy = evaluate_greedy(agent, env, seed=SEED + 90_000)
    random_base = evaluate_random(env, seed=SEED + 95_000)

    Qb = bellman_solution()
    optimum = {s: float(np.max(Qb[SIDX[s], list(C.VALID_ACTIONS[s])])) for s in C.STATES}

    policy = {s: greedy[s][2] for s in C.STATES}
    policy_match = {s: policy[s] == EXPECTED_POLICY[s] for s in C.STATES}

    beats_random = {s: greedy[s][0] > random_base[s][0] for s in C.STATES}
    if sum(greedy[s][0] for s in C.STATES) <= sum(random_base[s][0] for s in C.STATES):
        raise RuntimeError(
            "Double DQN failed to beat a uniform-random policy - inspect the "
            "agent/env wiring before trusting the run."
        )

    agent.save(C.DQN_MODEL_PATH)

    # ---------------------------------------------------------------------- #
    # report
    # ---------------------------------------------------------------------- #
    lines: list[str] = []
    add = lines.append
    add("# Double DQN - training report\n")
    add("`data/processed/double_dqn_report.md` - generated by "
        "`PYTHONPATH=src python -m rl_ids.train_dqn`\n")

    add("## 1. Setup\n")
    add("- Environment: `IDSEnv2018(split=\"train\")`, 17-feature **real-flow** "
        "observations with Gaussian noise and telemetry masking (partial "
        "observability). The hidden severity bucket drives transitions/rewards "
        "and is never an observation.")
    add("- Action mask: the playbook mask (`info[\"action_mask\"]`) is applied at "
        "every argmax (acting and bootstrapping); the Q-network sees only the "
        "17 features.")
    add(f"- Algorithm: **Double DQN** - online net selects "
        f"`a* = argmax_a Q_online(s',a)`, target net evaluates "
        f"`y = r + γ·Q_target(s',a*)`.")
    add(f"- Network: MLP {list(C.DQN_HIDDEN)} (ReLU), Adam lr={C.DQN_LR:g}, "
        f"Huber loss, grad-clip={C.DQN_GRAD_CLIP}.")
    add(f"- Replay: uniform, capacity {C.DQN_REPLAY_CAPACITY:,}, batch "
        f"{C.DQN_BATCH_SIZE}, warm-up {C.DQN_REPLAY_WARMUP:,} transitions, "
        f"target sync every {C.DQN_TARGET_SYNC:,} updates.")
    add(f"- Exploration: ε {C.DQN_EPS_START:g} → floor {C.DQN_EPS_END:g}, "
        f"×{C.DQN_EPS_DECAY:g} per episode (final ε = {history['eps'][-1]:.4f}).")
    add(f"- Episodes = {TRAIN_EPISODES:,} (exploring starts, uniform over "
        f"{{Normal, Suspicious, Compromised}}), steps = {total_steps:,}, "
        f"updates = {agent.updates:,}.")
    add(f"- Obs pool ≤ {MAX_ROWS_PER_STATE:,} rows/state; γ = {C.GAMMA}; "
        f"seed = {SEED}.\n")

    add("## 2. Learned policy\n")
    add("| Start state | Dominant greedy action | Expected optimal | Match |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {policy[s]} | {EXPECTED_POLICY[s]} | "
            f"{'yes' if policy_match[s] else '**no**'} |")
    add("")
    add("Dominant action = the first action the greedy policy takes most often "
        "over " + f"{EVAL_EPISODES:,} episodes started in that bucket.\n")

    add("## 3. Returns: greedy vs random vs exact optimum\n")
    add("| Start state | Greedy return | Random return | Exact optimum (Bellman) |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {greedy[s][0]:+.3f} ± {greedy[s][1]:.2f} | "
            f"{random_base[s][0]:+.3f} | {optimum[s]:+.3f} |")
    add("")
    beats_detail = ", ".join(f"{s}: {'yes' if beats_random[s] else 'no'}" for s in C.STATES)
    add(f"- Greedy beats a uniform-random policy: "
        f"**{'yes' if all(beats_random.values()) else 'no'}** ({beats_detail}).")
    add("- The exact optimum is the max Bellman q* over each state's legal "
        "actions (from `rl_ids.tabular.bellman_solution`). A gap here reflects "
        "both approximation error and the finite run.\n")

    add("## 4. Training dynamics\n")
    tail_loss = [l for l in history["loss"][-10:] if not np.isnan(l)]
    add(f"- Final ε = {history['eps'][-1]:.5f}.")
    loss_line = (f"- Mean Huber loss (last tracked points) = {np.mean(tail_loss):.4f}"
                 if tail_loss else "- Mean Huber loss: n/a (warm-up not reached).")
    add(loss_line)
    add(f"- Mean episode return (last point) = {history['return'][-1]:+.3f}.")
    add(f"- Learning curve: `{C.DQN_CURVE_PNG.name}`\n")
    add(f"![Double DQN learning curve]({C.DQN_CURVE_PNG.name})\n")

    add("## 5. Verdict\n")
    if all(policy_match.values()):
        add("The Double DQN agent recovers the **expected optimal policy** in "
            "every bucket and beats the random baseline, learning from the "
            "partially-observable telemetry. Ready for prioritized replay and "
            "the holdout A/B.\n")
    else:
        missed = [s for s in C.STATES if not policy_match[s]]
        add(f"The agent beats the random baseline but has **not** recovered the "
            f"expected action in: {', '.join(missed)}. Inspect the learning "
            f"curve / raise episodes before the A/B comparison.\n")

    add("## 6. Files\n")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `src/rl_ids/dqn.py` | `QNetwork`, `ReplayBuffer`, `DoubleDQNAgent` |")
    add("| `src/rl_ids/train_dqn.py` | this driver |")
    add("| `models/double_dqn.pt` | trained online-network weights (git-ignored) |")
    add(f"| `data/processed/{C.DQN_REPORT_PATH.name}` | this report |")
    add(f"| `data/processed/{C.DQN_CURVE_PNG.name}` | learning curve |")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.DQN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------------------- #
    # learning curve
    # ---------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
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
    fig.tight_layout()
    C.DQN_CURVE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.DQN_CURVE_PNG, dpi=110)
    plt.close(fig)

    print(f"wrote {C.DQN_REPORT_PATH}")
    print(f"wrote {C.DQN_CURVE_PNG}")
    print(f"wrote {C.DQN_MODEL_PATH}")
    print(f"policy={policy}  match={policy_match}")
    print(f"greedy={ {s: round(greedy[s][0], 3) for s in C.STATES} }")
    print(f"random={ {s: round(random_base[s][0], 3) for s in C.STATES} }")
    print(f"updates={agent.updates}  steps={total_steps}  final_eps={history['eps'][-1]:.5f}")


if __name__ == "__main__":
    main()
