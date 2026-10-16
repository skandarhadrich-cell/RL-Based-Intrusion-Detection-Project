"""Mask-free (Path 2) training driver.

Same Double DQN / prioritized-replay agents and the same environment as the
masked phase, with one change: the playbook action mask is **removed**.  Every
action is offered in every state (``IDSEnv2018(mask_actions=False)``), so nothing
tells the agent which actions are "correct".  To find the optimal policy it must
infer the hidden severity bucket from the 17 noisy telemetry features alone.

The underlying MDP is unchanged: the mask-free exact Bellman solution has the
*same* optimal policy and value as the masked one (Normal 8.721, Suspicious
23.163, Compromised 14.508), so the playbook is still the target - it is simply
no longer handed over (see ``docs/maskfree_agent.md`` for the verification).

Writes:
  * ``data/processed/maskfree_dqn_report.md``
  * ``data/processed/maskfree_dqn_learning_curve.png``
  * ``models/double_dqn_maskfree.pt``   (git-ignored)
  * ``data/processed/maskfree_per_report.md``
  * ``data/processed/maskfree_per_learning_curve.png``
  * ``models/double_dqn_per_maskfree.pt`` (git-ignored)

Run:  PYTHONPATH=src python -m rl_ids.train_maskfree
NOTE: the online run loads real-flow observation pools into memory.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ids import config as C
from rl_ids.dqn import DoubleDQNAgent, ReplayBuffer, epsilon_schedule
from rl_ids.env import IDSEnv2018
from rl_ids.evaluate import evaluate_greedy, evaluate_random
from rl_ids.per import PrioritizedReplayBuffer, per_beta
from rl_ids.tabular import SIDX, bellman_solution

# --------------------------------------------------------------------------- #
# Hyper-parameters - identical to the masked drivers (only the mask differs)
# --------------------------------------------------------------------------- #
TRAIN_EPISODES = 5_000
MAX_ROWS_PER_STATE = C.DQN_MAX_ROWS_PER_STATE
EVAL_EPISODES = 1_000
RANDOM_EPISODES = 1_000
TRACK_EVERY = 50
SEED = 7

EXPECTED_POLICY = {"Normal": "Monitor", "Suspicious": "Honeypot",
                   "Compromised": "Block"}

# Greedy returns of the *masked* agents, for the mask-free-vs-masked comparison.
# Source: data/processed/double_dqn_report.md / prioritized_replay_report.md
# (identical env, episodes and seeds).
MASKED_DQN_GREEDY = {"Normal": 8.578, "Suspicious": 20.173, "Compromised": 14.387}
MASKED_PER_GREEDY = {"Normal": 8.471, "Suspicious": 20.651, "Compromised": 14.298}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_agent(kind: str, episodes: int = TRAIN_EPISODES, seed: int = SEED,
                parquet_path=None, max_rows_per_state: int | None = None) -> dict:
    """Train one mask-free agent; ``kind`` is ``"uniform"`` or ``"per"``.

    Returns a dict with the environment, trained agent, training history, step
    count and (PER only) the final beta.

    ``parquet_path`` / ``max_rows_per_state`` default to the mask-free phase's
    train split and pool cap; the coverage ablation overrides them to train on
    the augmented corpus while keeping everything else identical.
    """
    if kind not in ("uniform", "per"):
        raise ValueError(f"kind must be 'uniform' or 'per', got {kind!r}")
    use_per = kind == "per"

    env = IDSEnv2018(
        split="train",
        max_rows_per_state=(MAX_ROWS_PER_STATE if max_rows_per_state is None
                            else max_rows_per_state),
        seed=seed, mask_actions=False, parquet_path=parquet_path,
    )
    agent = DoubleDQNAgent(seed=seed)
    buffer = PrioritizedReplayBuffer(seed=seed) if use_per else ReplayBuffer(seed=seed)
    rng = np.random.default_rng(seed)

    history = {"episode": [], "return": [], "loss": [], "eps": [], "beta": []}
    recent_returns: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=500)
    total_steps = 0
    last_beta = C.PER_BETA_START

    for ep in range(episodes):
        eps = epsilon_schedule(ep)
        # Exploring starts over the hidden buckets (same as the masked phase).
        start = C.STATES[int(rng.integers(len(C.STATES)))]
        obs, info = env.reset(seed=seed + ep, options={"state": start})
        mask = info["action_mask"]  # all-ones in mask-free mode

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
                if use_per:
                    last_beta = per_beta(agent.updates)
                    batch = buffer.sample(C.DQN_BATCH_SIZE, beta=last_beta)
                    loss, td = agent.learn(batch, weights=batch["weights"])
                    buffer.update_priorities(batch["indices"], td)
                    recent_losses.append(loss)
                else:
                    recent_losses.append(agent.update(buffer.sample(C.DQN_BATCH_SIZE)))
        recent_returns.append(g)

        if ep % TRACK_EVERY == 0 or ep == episodes - 1:
            history["episode"].append(ep)
            history["return"].append(float(np.mean(recent_returns)))
            history["loss"].append(
                float(np.mean(recent_losses)) if recent_losses else float("nan"))
            history["eps"].append(eps)
            history["beta"].append(last_beta)

    return {"env": env, "agent": agent, "history": history,
            "total_steps": total_steps, "last_beta": last_beta}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _write_report(kind: str, res: dict, masked_greedy: dict) -> dict:
    verb = "PER" if kind == "per" else "Double DQN"
    env, agent, history = res["env"], res["agent"], res["history"]
    total_steps, last_beta = res["total_steps"], res["last_beta"]
    report_path = (C.MASKFREE_PER_REPORT_PATH if kind == "per"
                   else C.MASKFREE_DQN_REPORT_PATH)
    curve_path = C.MASKFREE_PER_CURVE_PNG if kind == "per" else C.MASKFREE_DQN_CURVE_PNG
    model_path = C.MASKFREE_PER_MODEL_PATH if kind == "per" else C.MASKFREE_DQN_MODEL_PATH

    greedy = evaluate_greedy(agent, env, seed=SEED + 90_000, episodes=EVAL_EPISODES)
    random_base = evaluate_random(env, seed=SEED + 95_000, episodes=RANDOM_EPISODES)

    Qb = bellman_solution(all_actions=True)
    optimum = {s: float(np.max(Qb[SIDX[s], :])) for s in C.STATES}

    policy = {s: greedy[s][2] for s in C.STATES}
    policy_match = {s: policy[s] == EXPECTED_POLICY[s] for s in C.STATES}
    beats_random = {s: greedy[s][0] > random_base[s][0] for s in C.STATES}

    if sum(greedy[s][0] for s in C.STATES) <= sum(random_base[s][0] for s in C.STATES):
        raise RuntimeError(
            f"mask-free {verb} failed to beat a uniform-random policy - inspect "
            "the agent/env wiring before trusting the run."
        )

    agent.save(model_path)

    lines: list[str] = []
    add = lines.append
    add(f"# Mask-free {verb} - training report\n")
    add(f"`data/processed/{report_path.name}` - generated by "
        "`PYTHONPATH=src python -m rl_ids.train_maskfree`\n")

    add("## 1. Setup\n")
    add("- Environment: `IDSEnv2018(split=\"train\", mask_actions=False)`, "
        "17-feature **real-flow** observations with Gaussian noise and telemetry "
        "masking (partial observability).")
    add("- **No action mask.** Every one of the 6 actions is offered in every "
        "state; the agent gets no hint of which actions are playbook-legal. It "
        "must infer the hidden severity bucket from the 17 features alone.")
    add("- The MDP is otherwise unchanged: the mask-free exact Bellman solution "
        "has the *same* optimal policy and value as the masked one "
        "(Normal 8.721, Suspicious 23.163, Compromised 14.508).")
    algo = ("**Double DQN + PER** - priority sampling with importance-sampling "
            "correction" if kind == "per" else "**Double DQN** - online net "
            "selects `a* = argmax_a Q_online(s',a)`, target net evaluates it")
    add(f"- Algorithm: {algo}.")
    add(f"- Network: MLP {list(C.DQN_HIDDEN)} (ReLU), Adam lr={C.DQN_LR:g}, "
        f"Huber loss, grad-clip={C.DQN_GRAD_CLIP}.")
    add(f"- Episodes = {TRAIN_EPISODES:,} (exploring starts), steps = "
        f"{total_steps:,}, updates = {agent.updates:,}; obs pool ≤ "
        f"{MAX_ROWS_PER_STATE:,} rows/state; γ = {C.GAMMA}; seed = {SEED}.\n")

    add("## 2. Learned policy\n")
    add("| Start state | Dominant greedy action | Expected optimal | Match |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {policy[s]} | {EXPECTED_POLICY[s]} | "
            f"{'yes' if policy_match[s] else '**no**'} |")
    add("")
    add(f"Dominant action = the first action the greedy policy takes most often "
        f"over {EVAL_EPISODES:,} episodes started in that bucket.\n")

    add(f"## 3. Returns: {verb} vs random vs exact optimum\n")
    add("| Start state | Mask-free greedy | Random | Exact optimum (Bellman) | Masked baseline |")
    add("|---|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {greedy[s][0]:+.3f} ± {greedy[s][1]:.2f} | "
            f"{random_base[s][0]:+.3f} | {optimum[s]:+.3f} | "
            f"{masked_greedy[s]:+.3f} |")
    add("")
    beats_detail = ", ".join(f"{s}: {'yes' if beats_random[s] else 'no'}"
                             for s in C.STATES)
    add(f"- Beats a uniform-random policy (over **all** actions): "
        f"**{'yes' if all(beats_random.values()) else 'no'}** ({beats_detail}).")
    add("- The exact optimum is the max mask-free Bellman q* over each state's "
        "actions (from `rl_ids.tabular.bellman_solution(all_actions=True)`); it "
        "equals the masked optimum, confirming the task is unchanged.\n")

    add("## 4. Mask-free vs masked\n")
    add("Same agent, environment, episodes and seeds; only the mask differs. "
        "A gap here is the cost of *perceiving* the bucket instead of being told "
        "it.\n")
    add(f"| Start state | Masked {verb} greedy | Mask-free {verb} greedy | Δ (mask-free − masked) |")
    add("|---|---:|---:|---:|")
    for s in C.STATES:
        d = greedy[s][0] - masked_greedy[s]
        add(f"| {s} | {masked_greedy[s]:+.3f} | {greedy[s][0]:+.3f} | {d:+.3f} |")
    add("")

    add("## 5. Training dynamics\n")
    tail_loss = [l for l in history["loss"][-10:] if not np.isnan(l)]
    add(f"- Final ε = {history['eps'][-1]:.5f}.")
    if kind == "per":
        add(f"- Final β = {last_beta:.4f}.")
    if tail_loss:
        add(f"- Mean Huber loss (last tracked points) = {np.mean(tail_loss):.4f}.")
    else:
        add("- Mean Huber loss: n/a (warm-up not reached).")
    add(f"- Mean episode return (last point) = {history['return'][-1]:+.3f}.")
    add(f"- Learning curve: `{curve_path.name}`\n")
    add(f"![Mask-free {verb} learning curve]({curve_path.name})\n")

    add("## 6. Verdict\n")
    if all(policy_match.values()):
        add(f"Mask-free {verb} recovers the **expected optimal policy** in every "
            f"bucket and beats the random baseline - it learned to infer the "
            f"hidden bucket from the telemetry without the mask.\n")
    else:
        missed = [s for s in C.STATES if not policy_match[s]]
        add(f"Mask-free {verb} beats the random baseline but has **not** "
            f"recovered the expected action in: {', '.join(missed)}. Without the "
            f"mask the agent must perceive the hidden bucket, which is the hard "
            f"part (see `docs/maskfree_agent.md`).\n")

    add("## 7. Files\n")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `src/rl_ids/train_maskfree.py` | this driver |")
    add(f"| `models/{model_path.name}` | trained online-network weights (git-ignored) |")
    add(f"| `data/processed/{report_path.name}` | this report |")
    add(f"| `data/processed/{curve_path.name}` | learning curve |")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- learning curve ----
    n_panels = 4 if kind == "per" else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    axes[0].plot(history["episode"], history["return"], lw=1.2, color="tab:blue")
    axes[0].set_title("Episode return (100-ep mean)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("discounted return")
    axes[1].plot(history["episode"], history["loss"], lw=1.2, color="tab:orange")
    axes[1].set_title("Huber loss (500-update mean)")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("loss")
    axes[2].plot(history["episode"], history["eps"], lw=1.2, color="tab:green")
    axes[2].set_title("ε (exploration)")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("ε")
    if kind == "per":
        axes[3].plot(history["episode"], history["beta"], lw=1.2, color="tab:red")
        axes[3].set_title("β (IS correction)")
        axes[3].set_xlabel("episode"); axes[3].set_ylabel("β")
    fig.tight_layout()
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve_path, dpi=110)
    plt.close(fig)

    return {"policy": policy, "policy_match": policy_match,
            "greedy": {s: greedy[s][0] for s in C.STATES},
            "random": {s: random_base[s][0] for s in C.STATES},
            "optimum": optimum, "updates": agent.updates,
            "final_eps": history["eps"][-1], "report": str(report_path)}


def main() -> None:
    print("=== mask-free Double DQN (uniform replay) ===")
    res_u = train_agent("uniform")
    out_u = _write_report("uniform", res_u, MASKED_DQN_GREEDY)
    print(f"policy={out_u['policy']}  match={out_u['policy_match']}")
    print(f"greedy={ {s: round(v, 3) for s, v in out_u['greedy'].items()} }")
    print(f"optimum={ {s: round(v, 3) for s, v in out_u['optimum'].items()} }")

    print("=== mask-free Double DQN + PER ===")
    res_p = train_agent("per")
    out_p = _write_report("per", res_p, MASKED_PER_GREEDY)
    print(f"policy={out_p['policy']}  match={out_p['policy_match']}")
    print(f"greedy={ {s: round(v, 3) for s, v in out_p['greedy'].items()} }")
    print(f"optimum={ {s: round(v, 3) for s, v in out_p['optimum'].items()} }")

    print(f"wrote {C.MASKFREE_DQN_REPORT_PATH}")
    print(f"wrote {C.MASKFREE_PER_REPORT_PATH}")


if __name__ == "__main__":
    main()
