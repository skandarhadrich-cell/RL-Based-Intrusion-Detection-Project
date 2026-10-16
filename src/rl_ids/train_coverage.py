"""Coverage-ablation training driver.

Retrains the mask-free Double DQN / PER on an **augmented corpus** that adds the
missing concepts (PortScan -> Suspicious, Infilteration -> Compromised) on top of
the original train split.  Everything else is held fixed (episodes, seed,
network, pool cap), so any change in behaviour is attributable to the added
*coverage*.

The augmented corpus is ``train.parquet`` (+) ``coverage_train.parquet``,
RNG-shuffled so the sampler sees the new concepts in proportion instead of them
being truncated away by the per-state ``head()`` cap.  The driver asserts the
training pools actually contain the new concepts before training.

Writes:
  * ``data/processed/coverage_dqn_report.md`` + ``coverage_dqn_learning_curve.png``
  * ``data/processed/coverage_per_report.md`` + ``coverage_per_learning_curve.png``
  * ``models/double_dqn_coverage.pt`` / ``models/double_dqn_per_coverage.pt`` (git-ignored)

Run:  PYTHONPATH=src python -m rl_ids.train_coverage
NOTE: reads the full train split into memory; raise VM RAM if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ids import config as C
from rl_ids.evaluate import evaluate_greedy, evaluate_random
from rl_ids.tabular import SIDX, bellman_solution
from rl_ids.train_maskfree import (
    EVAL_EPISODES, EXPECTED_POLICY, RANDOM_EPISODES, SEED, TRAIN_EPISODES,
    train_agent,
)

# Greedy returns of the *no-coverage* mask-free agents (same env/episodes/seed);
# source: data/processed/maskfree_dqn_report.md / maskfree_per_report.md.
NOCOV_DQN_GREEDY = {"Normal": 6.542, "Suspicious": 20.095, "Compromised": 12.996}
NOCOV_PER_GREEDY = {"Normal": 5.181, "Suspicious": 20.670, "Compromised": 12.776}

NEW_CONCEPTS = {
    "Suspicious": ("PortScan", C.COVERAGE_MIN_PORTSCAN_IN_POOL),
    "Compromised": ("Infilteration", C.COVERAGE_MIN_INFILTRATION_IN_POOL),
}


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def build_combined_corpus(seed: int = SEED) -> None:
    """train (+) coverage_train, RNG-shuffled and written for the env to read.

    Done in Arrow rather than pandas: the train split carries four object
    columns whose pandas copies alone need ~2.5 GB, and we only need a shuffled
    *file* here (the env loads it itself later).  A fixed `seed` makes the row
    order - and therefore the per-state `head()` sample - reproducible.
    """
    t_orig = pq.read_table(C.TRAIN_PARQUET)
    t_cov = pq.read_table(C.COVERAGE_TRAIN_PARQUET)
    table = pa.concat_tables([t_orig, t_cov])
    del t_orig, t_cov
    perm = np.random.default_rng(seed).permutation(table.num_rows)
    table = table.take(pa.array(perm))
    pq.write_table(table, C.COVERAGE_COMBINED_PARQUET)
    del table


def _pool_frame(combined: pd.DataFrame, state: str,
                cap: int = C.DQN_MAX_ROWS_PER_STATE) -> pd.DataFrame:
    """Exactly what ``IDSEnv2018._build_pools`` keeps: first `cap` rows of state."""
    return combined[combined["mdp_state"] == state].head(cap)


def check_pools(seed: int = SEED) -> dict:
    """Assert the ablated concepts survive the per-state cap; return composition.

    The combined parquet is already shuffled, so reading only the two label
    columns is cheap and faithful to the order the environment will see.
    """
    frame = pd.read_parquet(C.COVERAGE_COMBINED_PARQUET,
                            columns=["mdp_state", "attack_label"])
    pool = {st: _pool_frame(frame, st) for st in C.STATES}
    comp = {}
    for st, df in pool.items():
        comp[st] = {k: int(v) for k, v in df["attack_label"].value_counts().items()}
        comp[st]["_n"] = int(len(df))
    for st, (label, need) in NEW_CONCEPTS.items():
        have = int((pool[st]["attack_label"] == label).sum())
        if have < need:
            raise RuntimeError(
                f"training pool for {st} has only {have} {label} rows "
                f"(< {need}); raise the pool cap or curate the corpus")
    del frame
    return comp


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _write_report(kind: str, res: dict, pool_comp: dict, corpus_comp: dict,
                  nocov_greedy: dict) -> dict:
    verb = "PER" if kind == "per" else "Double DQN"
    env, agent, history = res["env"], res["agent"], res["history"]
    report_path = (C.COVERAGE_PER_REPORT_PATH if kind == "per"
                   else C.COVERAGE_DQN_REPORT_PATH)
    curve_path = C.COVERAGE_PER_CURVE_PNG if kind == "per" else C.COVERAGE_DQN_CURVE_PNG
    model_path = C.COVERAGE_PER_MODEL_PATH if kind == "per" else C.COVERAGE_DQN_MODEL_PATH

    greedy = evaluate_greedy(agent, env, seed=SEED + 90_000, episodes=EVAL_EPISODES)
    random_base = evaluate_random(env, seed=SEED + 95_000, episodes=RANDOM_EPISODES)
    Qb = bellman_solution(all_actions=True)
    optimum = {s: float(np.max(Qb[SIDX[s], :])) for s in C.STATES}
    policy = {s: greedy[s][2] for s in C.STATES}
    policy_match = {s: policy[s] == EXPECTED_POLICY[s] for s in C.STATES}

    agent.save(model_path)

    lines: list[str] = []
    add = lines.append
    add(f"# Coverage-ablation {verb} - training report\n")
    add(f"`data/processed/{report_path.name}` - generated by "
        "`PYTHONPATH=src python -m rl_ids.train_coverage`\n")

    add("## 1. Setup\n")
    add("- Train corpus = `train.parquet` (+) `coverage_train.parquet` "
        "(PortScan-day 80% + Wed-28 Infilteration), RNG-shuffled, seed "
        f"{SEED}.")
    add("- Environment: `IDSEnv2018(mask_actions=False)`, 17-feature real-flow "
        "observations with noise + telemetry masking (same as the mask-free phase).")
    add("- **Only the data changed**: same network, episodes, seed and pool cap "
        f"as the no-coverage mask-free run (all actions offered, no mask).")
    add(f"- Episodes = {history['episode'][-1] + 1:,}, steps = {res['total_steps']:,}, "
        f"updates = {agent.updates:,}; pool cap = {C.DQN_MAX_ROWS_PER_STATE:,}/state; "
        f"γ = {C.GAMMA}.\n")

    add("## 2. Added corpus & what the agent actually sees\n")
    add("| State | Added rows | Corpus total | In pool (after cap) | New concept in pool |")
    add("|---|---:|---:|---:|---|")
    for st in C.STATES:
        added = int(corpus_comp[st].get("added", 0))
        total = int(corpus_comp[st]["total"])
        seen = pool_comp[st]["_n"]
        label = NEW_CONCEPTS.get(st, (None,))[0]
        marker = f"{pool_comp[st].get(label, 0):,} {label}" if label else "—"
        add(f"| {st} | {added:,} | {total:,} | {seen:,} | {marker} |")
    add("")
    add("Added rows are the coverage concepts; the new concept must survive the "
        "per-state `head()` cap (asserted at train time) or the ablation would be "
        "silently comparing against no added data at all.\n")

    add("## 3. Learned policy\n")
    add("| Start state | Dominant greedy action | Expected optimal | Match |")
    add("|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {policy[s]} | {EXPECTED_POLICY[s]} | "
            f"{'yes' if policy_match[s] else '**no**'} |")
    add("")

    add(f"## 4. Returns: coverage {verb} vs random vs optimum\n")
    add("| Start state | Coverage greedy | Random | Exact optimum | No-coverage mask-free |")
    add("|---|---|---|---|---|")
    for s in C.STATES:
        add(f"| {s} | {greedy[s][0]:+.3f} ± {greedy[s][1]:.2f} | "
            f"{random_base[s][0]:+.3f} | {optimum[s]:+.3f} | {nocov_greedy[s]:+.3f} |")
    add("")
    add("The coverage return is scored on the **augmented** pool, the "
        "no-coverage figure on the original pool, so the gap mixes a harder pool "
        "with any change in competence - both agents still recover the same "
        "playbook policy (section 3). Detection (the point of the ablation) is "
        "measured on identical test sets in `coverage_ablation_report.md`.\n")

    add("## 5. Training dynamics\n")
    tail_loss = [l for l in history["loss"][-10:] if not np.isnan(l)]
    add(f"- Final ε = {history['eps'][-1]:.5f}.")
    if kind == "per":
        add(f"- Final β = {res['last_beta']:.4f}.")
    if tail_loss:
        add(f"- Mean Huber loss (last tracked points) = {np.mean(tail_loss):.4f}.")
    add(f"- Mean episode return (last point) = {history['return'][-1]:+.3f}.")
    add(f"- Learning curve: `{curve_path.name}`\n")
    add(f"![Coverage {verb} learning curve]({curve_path.name})\n")

    add("## 6. Files\n")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `src/rl_ids/coverage_split.py` | builds the coverage corpus |")
    add("| `src/rl_ids/train_coverage.py` | this driver |")
    add(f"| `models/{model_path.name}` | trained online-network weights (git-ignored) |")
    add(f"| `data/processed/{report_path.name}` | this report |")
    add(f"| `data/processed/{curve_path.name}` | learning curve |")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- learning curve ----
    n_panels = 4 if kind == "per" else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    axes[0].plot(history["episode"], history["return"], lw=1.2, color="tab:blue")
    axes[0].set_title("Episode return (100-ep mean)"); axes[0].set_xlabel("episode")
    axes[1].plot(history["episode"], history["loss"], lw=1.2, color="tab:orange")
    axes[1].set_title("Huber loss (500-update mean)"); axes[1].set_xlabel("episode")
    axes[2].plot(history["episode"], history["eps"], lw=1.2, color="tab:green")
    axes[2].set_title("ε (exploration)"); axes[2].set_xlabel("episode")
    if kind == "per":
        axes[3].plot(history["episode"], history["beta"], lw=1.2, color="tab:red")
        axes[3].set_title("β (IS correction)"); axes[3].set_xlabel("episode")
    fig.tight_layout()
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve_path, dpi=110)
    plt.close(fig)

    return {"policy": policy, "policy_match": policy_match,
            "greedy": {s: greedy[s][0] for s in C.STATES},
            "optimum": optimum, "updates": agent.updates,
            "report": str(report_path)}


def main() -> None:
    print("=== building augmented corpus ===")
    build_combined_corpus()
    pool_comp = check_pools()

    # Corpus composition (per state: total after merge, rows the coverage split
    # added).  Read only the state column of each file to stay light.
    orig_states = pd.read_parquet(C.TRAIN_PARQUET, columns=["mdp_state"])["mdp_state"]
    comb_states = pd.read_parquet(C.COVERAGE_COMBINED_PARQUET,
                                  columns=["mdp_state"])["mdp_state"]
    corpus_comp = {}
    for st in C.STATES:
        total = int((comb_states == st).sum())
        added = total - int((orig_states == st).sum())
        corpus_comp[st] = {"total": total, "added": added}
    del orig_states, comb_states
    print("pool composition:", {k: v for k, v in pool_comp.items()})

    for kind, nocov in (("uniform", NOCOV_DQN_GREEDY), ("per", NOCOV_PER_GREEDY)):
        print(f"=== coverage {kind} ===")
        res = train_agent(kind, parquet_path=str(C.COVERAGE_COMBINED_PARQUET))
        out = _write_report(kind, res, pool_comp, corpus_comp, nocov)
        print(f"policy={out['policy']}  match={out['policy_match']}")
        print(f"greedy={ {s: round(v, 3) for s, v in out['greedy'].items()} }")

    print(f"wrote {C.COVERAGE_DQN_REPORT_PATH}")
    print(f"wrote {C.COVERAGE_PER_REPORT_PATH}")


if __name__ == "__main__":
    main()
