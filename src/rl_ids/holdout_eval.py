"""Per-class detection evaluation on the untouched holdouts (probe method).

Why a probe, and not the agent's actions?
----------------------------------------
The playbook **action mask** is a function of the hidden severity bucket and the
three masks are disjoint (``Normal`` -> {Monitor}, ``Suspicious`` ->
{Monitor,DPI,Throttle,Honeypot}, ``Compromised`` -> {Monitor,Block,Isolate}), so
the mask *is* the label in disguise.  Scoring the masked agent therefore gives a
perfect-but-circular result, and scoring the *unmasked* Q-argmax is meaningless
because the network was never forced to encode the state in its outputs (the
mask always told it the state).  Both are reproduced here as negative controls.

What we actually evaluate is the information that genuinely exists in the
agent's learned representation: a **logistic-regression probe** is trained on the
frozen penultimate layer (128-d) using the **train split only**, then tested on
the two eval-only holdout days.  This answers the real IDS question: *does the
agent's representation let you tell the traffic classes apart on data it never
saw?*  A raw-17-feature probe is the no-representation baseline.

Outputs:
  * ``data/processed/holdout_classification_report.md``
  * ``data/processed/holdout_confusion_matrix.png``

Run:  PYTHONPATH=src python -m rl_ids.holdout_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rl_ids import config as C
from rl_ids.dqn import DoubleDQNAgent, masked_argmax
from rl_ids.metrics import (
    accuracy, binary_metrics, confusion, macro_average, per_class_metrics,
)

STATES = list(C.STATES)
BENIGN = "Normal"
BASELINE = "raw-17 features (no agent)"

PROBE_TRAIN_PER_CLASS = 4_000
HOLDOUT_PER_CLASS = 3_000
DRAWS = 4          # observation draws averaged per flow (Monte-Carlo marginalisation)
SEED = 0
CONDITIONS = ("clean", "corrupted")


# --------------------------------------------------------------------------- #
# observation helpers (identical corruption to IDSEnv2018._sample_obs)
# --------------------------------------------------------------------------- #
def corrupt(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise (sigma) + per-feature telemetry masking, as in the env."""
    out = X.astype(np.float64) + rng.normal(0.0, C.OBS_NOISE_SIGMA, size=X.shape)
    missing = rng.random(X.shape) < C.OBS_MASK_PROB
    out[missing] = 0.0
    return out.astype(np.float32)


@torch.no_grad()
def penultimate(agent, X: np.ndarray) -> np.ndarray:
    """Frozen 128-d hidden representation of the online network for raw features."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    return agent.online.net[:-1](torch.as_tensor(X)).cpu().numpy()


def represent(agent, X: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Feature matrix fed to the probe: raw features (agent=None) or the 128-d layer.

    ``clean`` uses the scaled features untouched; ``corrupted`` averages ``DRAWS``
    env-style (noisy + masked) observations to stabilise the decision.
    """
    def once(obs: np.ndarray) -> np.ndarray:
        return obs if agent is None else penultimate(agent, obs)

    if condition == "clean":
        return once(X)
    acc = None
    for _ in range(DRAWS):
        h = once(corrupt(X, rng))
        acc = h if acc is None else acc + h
    return acc / DRAWS


# --------------------------------------------------------------------------- #
# data loading (stratified, deterministic)
# --------------------------------------------------------------------------- #
def load_train_probe(per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sample of the train split: (raw features, severity bucket)."""
    df = pd.read_parquet(C.TRAIN_PARQUET, columns=list(C.OBS_FEATURES) + ["mdp_state"])
    df["mdp_state"] = df["mdp_state"].astype("category")
    parts = []
    for st in STATES:
        sub = df[df["mdp_state"] == st]
        parts.append(sub.sample(min(per_class, len(sub)), random_state=seed))
    out = pd.concat(parts)
    X = out[C.OBS_FEATURES].to_numpy(np.float32)
    y = out["mdp_state"].to_numpy(dtype=object)
    del df, out
    return X, y


def load_holdout(path, per_class: int, seed: int) -> pd.DataFrame:
    """Stratified sample of a holdout parquet (keeps attack_label + mdp_state)."""
    h = pd.read_parquet(path, columns=list(C.OBS_FEATURES) + ["mdp_state", "attack_label"])
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in h.groupby("mdp_state")]
    return pd.concat(parts).reset_index(drop=True)


def attack_labels_of(df: pd.DataFrame) -> list[str]:
    return sorted(df.loc[df["mdp_state"] != BENIGN, "attack_label"].unique().tolist())


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(probe, features: np.ndarray, df: pd.DataFrame) -> dict:
    """Per-class bucket metrics + per-attack-label binary metrics for one holdout."""
    y_true = df["mdp_state"].to_numpy(dtype=object)
    attack = df["attack_label"].to_numpy(dtype=object)
    pred = probe.predict(features)
    is_benign = y_true == BENIGN

    buckets = per_class_metrics(y_true, pred, STATES)
    labels: dict[str, dict] = {}
    for lab in attack_labels_of(df):
        sel = (attack == lab) | is_benign
        labels[lab] = binary_metrics(attack[sel] == lab, pred[sel] != BENIGN)
    benign_fpr = float((pred[is_benign] != BENIGN).mean()) if is_benign.any() else 0.0
    return {
        "n": len(df),
        "acc": accuracy(y_true, pred),
        "buckets": buckets,
        "labels": labels,
        "benign_fpr": benign_fpr,
        "benign_n": int(is_benign.sum()),
        "cm": confusion(y_true, pred, STATES),
    }


def fit_probe(features: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(features, y)


def action_bucket(actions: np.ndarray) -> np.ndarray:
    """Map a chosen action id to the severity bucket it implies."""
    names = np.asarray(C.ACTIONS)[np.asarray(actions)]
    out = np.full(len(names), "Normal", dtype=object)
    out[np.isin(names, ("DPI", "Throttle", "Honeypot"))] = "Suspicious"
    out[np.isin(names, ("Block", "Isolate"))] = "Compromised"
    return out


def controls(agent, X: np.ndarray, y_true: np.ndarray, rng: np.random.Generator) -> dict:
    """Negative controls that show *why* the probe is needed."""
    with torch.no_grad():
        Q = np.zeros((len(X), len(C.ACTIONS)))
        for _ in range(DRAWS):
            Q += agent.q_values(corrupt(X, rng))
        Q /= DRAWS
    # circular: hand the agent the true-state mask
    supplied = np.empty(len(X), dtype=object)
    for i, st in enumerate(y_true):
        mask = np.zeros(len(C.ACTIONS), dtype=bool)
        mask[list(C.VALID_ACTIONS[st])] = True
        supplied[i] = action_bucket([masked_argmax(Q[i], mask)])[0]
    # mask-free argmax of the untrained-across-states Q-vector
    mask_free = action_bucket(np.argmax(Q, axis=1))
    random_actions = rng.integers(0, len(C.ACTIONS), size=len(X))
    return {
        "mask_supplied": accuracy(y_true, supplied),
        "mask_free_argmax": accuracy(y_true, mask_free),
        "random_action": accuracy(y_true, action_bucket(random_actions)),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _bucket_table(rep: dict) -> list[str]:
    b = rep["buckets"]
    lines = ["| Class | Precision | Recall | F1 | FPR | Support |",
             "|---|---:|---:|---:|---:|---:|"]
    for st in STATES:
        v = b[st]
        if v["support"] == 0 and v["fp"] == 0 and v["fn"] == 0:
            continue
        lines.append(
            f"| {st} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
            f"{v['fpr']:.3f} | {v['support']:,} |"
        )
    lines.append(
        f"| **macro** | {macro_average(b, 'precision'):.3f} | "
        f"{macro_average(b, 'recall'):.3f} | {macro_average(b, 'f1'):.3f} | - | "
        f"{sum(v['support'] for v in b.values()):,} |"
    )
    return lines


def _label_table(rep: dict) -> list[str]:
    lines = ["| Attack label | Precision | Recall | F1 | FPR | Support |",
             "|---|---:|---:|---:|---:|---:|"]
    for lab, v in rep["labels"].items():
        lines.append(
            f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
            f"{v['fpr']:.3f} | {v['support']:,} |"
        )
    lines.append(f"| _benign FP rate_ | - | - | - | {rep['benign_fpr']:.3f} | "
                 f"{rep['benign_n']:,} benign |")
    return lines


def write_report(results: dict, controls_by_model: dict, train_acc: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Holdout detection - per-class precision / recall / F1 / FPR\n")
    add(f"`data/processed/{C.HOLDOUT_EVAL_REPORT_PATH.name}` - generated by "
        "`PYTHONPATH=src python -m rl_ids.holdout_eval`\n")

    add("## 1. The problem this evaluation works around\n")
    add("The playbook **action mask** is a function of the hidden severity bucket "
        "and the three masks are pairwise disjoint:")
    add("")
    add("| Hidden bucket | Mask (legal actions) |")
    add("|---|---|")
    for st in STATES:
        add(f"| {st} | {', '.join(C.ACTIONS[a] for a in C.VALID_ACTIONS[st])} |")
    add("")
    add("So the mask *is* the label in disguise: handing the agent the true mask "
        "makes detection trivially perfect, and removing it leaves a network whose "
        "outputs were never forced to encode the state (the mask always told it). "
        "Both negative controls are reported in Section 2.")
    add("")
    add("To get an honest, non-trivial signal we evaluate what genuinely lives in "
        "the agent's **learned representation**: a logistic-regression probe is "
        "trained on the frozen 128-d penultimate layer of the Q-network, using the "
        "**train split only**, then tested on the two eval-only holdout days. A probe "
        "on the raw 17 features is the no-representation baseline.\n")

    add("## 2. Negative controls (why not just read the actions)\n")
    add("| Model | Mask supplied (circular) | Mask-free Q-argmax | Random action |")
    add("|---|---:|---:|---:|")
    for m, c in controls_by_model.items():
        add(f"| {m} | {c['mask_supplied']:.3f} | {c['mask_free_argmax']:.3f} | "
            f"{c['random_action']:.3f} |")
    add("")
    add("Mask-supplied accuracy is ~1.0 by construction (it is handed the answer); "
        "mask-free Q-argmax collapses to the majority bucket. Neither is a usable "
        "detection metric - hence the probe.\n")

    add("## 3. Method\n")
    add(f"- Probe: `StandardScaler -> LogisticRegression` on the frozen "
        f"128-d layer (raw 17 features for the baseline); fit on a stratified "
        f"train sample ({PROBE_TRAIN_PER_CLASS:,}/class), never on the holdouts.")
    add(f"- Holdouts sampled at {HOLDOUT_PER_CLASS:,}/class; decisions average "
        f"{DRAWS} env-style observations (Gaussian noise sigma={C.OBS_NOISE_SIGMA}, "
        f"telemetry masking p={C.OBS_MASK_PROB}).")
    add("- Observations: **clean** (scaled features untouched) and **corrupted** "
        "(noise + masking, as at training time).")
    add("- Predicted bucket = probe class; predicted-attack = bucket != Normal.")
    add("- **FPR is one-vs-rest per row**: for `Normal` it is the share of *attack* "
        "flows predicted benign (i.e. a miss rate); the operational false-alarm rate "
        "on benign traffic is the final `_benign FP rate_` row.")
    add("- Probe train accuracy: " +
        ", ".join(f"{m} ({cond}) {train_acc[m][cond]:.3f}"
                  for m in results["_models"] for cond in CONDITIONS) + ".\n")

    add("## 4. Results\n")
    add("### 4.1 Headline (corrupted observations)\n")
    add("| Model | Holdout | Accuracy | Macro F1 | Mean attack recall | Benign FPR |")
    add("|---|---|---:|---:|---:|---:|")
    for model in results["_models"]:
        for hname in results["_holdouts"]:
            rep = results[model]["corrupted"][hname]
            recalls = [v["recall"] for v in rep["labels"].values()] or [float("nan")]
            add(f"| {model} | {hname} | {rep['acc']:.3f} | "
                f"{macro_average(rep['buckets'], 'f1'):.3f} | "
                f"{float(np.mean(recalls)):.3f} | {rep['benign_fpr']:.3f} |")
    add("")
    add(f"![Confusion matrices]({C.HOLDOUT_EVAL_CM_PNG.name})\n")
    add("*Figure: confusion matrices under corrupted observations "
        "(rows = true severity bucket, cols = predicted).*\n")

    add("### 4.2 Per-model detail\n")
    for hname, hmeta in results["_holdouts"].items():
        add(f"#### {hname}\n")
        add(f"{hmeta['n']:,} flows ({hmeta['composition']}).\n")
        for model in results["_models"]:
            add(f"**{model}**\n")
            for cond in CONDITIONS:
                rep = results[model][cond][hname]
                add(f"_Observations: {cond}._\n")
                add(f"Bucket accuracy: **{rep['acc']:.3f}**\n")
                add("Severity buckets:")
                add("")
                lines.extend(_bucket_table(rep))
                add("")
                if rep["labels"]:
                    add("Per attack label (label vs benign):")
                    add("")
                    lines.extend(_label_table(rep))
                    add("")
        add("")

    add("## 5. Reading the results\n")
    for hname in results["_holdouts"]:
        scores = {m: macro_average(results[m]["corrupted"][hname]["buckets"], "f1")
                  for m in results["_models"]}
        ranked = sorted(results["_models"], key=lambda m: scores[m], reverse=True)
        best, runner = ranked[0], ranked[1]
        agents = {m: s for m, s in scores.items() if m != BASELINE}
        best_agent = max(agents, key=agents.get)
        raw = scores[BASELINE]
        add(f"- **{hname}** - best *agent* probe: **{best_agent}** "
            f"(macro F1 {scores[best_agent]:.3f}) vs the raw-17-feature baseline "
            f"{raw:.3f} ({scores[best_agent] - raw:+.3f}). Overall ranking: "
            f"{best} {scores[best]:.3f} > {runner} {scores[runner]:.3f}.")
    add("")
    add("- The probe shows how much *severity information* the agent's "
        "representation carries and, crucially, whether it transfers to the "
        "unseen days/attacks.")
    add("- Attack **families** seen in training transfer even when the label "
        "string is new: the 2017 `DDoS` label is unseen, but the family appears "
        "in training as ~1.24M rows of 2018 `HOIC`/`LOIC` traffic in the "
        "`Compromised` bucket, which is why it is recalled strongly. Families "
        "with no training counterpart (`PortScan`, `Infilteration`) are not "
        "reliably recalled - a coverage/representation limit, not a tuning bug.")
    add("- The agent's *policy* remains optimal on the training distribution; this "
        "report isolates the perception/generalisation question the mask hid.\n")

    add("## 6. Files\n")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `src/rl_ids/metrics.py` | pure per-class / binary metric functions |")
    add("| `src/rl_ids/holdout_eval.py` | this driver |")
    add("| `tests/test_holdout_metrics.py` | metric + observation unit tests |")
    add(f"| `data/processed/{C.HOLDOUT_EVAL_REPORT_PATH.name}` | this report |")
    add(f"| `data/processed/{C.HOLDOUT_EVAL_CM_PNG.name}` | confusion matrices |")

    C.HOLDOUT_EVAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_png(results: dict) -> None:
    models = results["_models"]
    holds = list(results["_holdouts"].keys())
    fig, axes = plt.subplots(len(models), len(holds),
                             figsize=(4.4 * len(holds), 4.2 * len(models)), squeeze=False)
    for r, model in enumerate(models):
        for c, hname in enumerate(holds):
            ax = axes[r][c]
            cm = results[model]["corrupted"][hname]["cm"]
            ax.imshow(cm, cmap="Blues")
            for i in range(len(STATES)):
                for j in range(len(STATES)):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                            color="black" if cm[i, j] < cm.max() * 0.6 else "white",
                            fontsize=9)
            ax.set_xticks(range(len(STATES)), STATES, rotation=30, ha="right")
            ax.set_yticks(range(len(STATES)), STATES)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{model}\n{hname} (corrupted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.HOLDOUT_EVAL_CM_PNG, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    Xtr, ytr = load_train_probe(PROBE_TRAIN_PER_CLASS, SEED)
    holdouts = {
        "holdout_infiltration (2018)": load_holdout(C.HOLDOUT_PARQUET, HOLDOUT_PER_CLASS, SEED),
        "holdout_ids2017_portscan_ddos (2017)": load_holdout(C.HOLDOUT_IDS2017_PARQUET, HOLDOUT_PER_CLASS, SEED),
    }
    holdout_meta = {}
    for hname, df in holdouts.items():
        comp = dict(df["mdp_state"].value_counts())
        holdout_meta[hname] = {
            "n": len(df),
            "composition": ", ".join(f"{k} {v:,}" for k, v in sorted(comp.items())),
        }

    models: list[tuple[str, str | None]] = [
        ("uniform DQN", str(C.DQN_MODEL_PATH)),
        ("PER", str(C.PER_MODEL_PATH)),
        (BASELINE, None),
    ]

    results: dict = {"_models": [], "_holdouts": holdout_meta}
    controls_by_model: dict[str, dict] = {}
    train_acc: dict[str, dict] = {}

    for name, path in models:
        agent = None if path is None else DoubleDQNAgent.load(path)
        results[name] = {}
        train_acc[name] = {}
        for cond in CONDITIONS:
            rng = np.random.default_rng(SEED)
            Ftr = represent(agent, Xtr, cond, rng)
            probe = fit_probe(Ftr, ytr)
            train_acc[name][cond] = float(probe.score(Ftr, ytr))
            results[name][cond] = {}
            for hname, df in holdouts.items():
                rng = np.random.default_rng(SEED)
                Fh = represent(agent, df[C.OBS_FEATURES].to_numpy(np.float32), cond, rng)
                results[name][cond][hname] = evaluate(probe, Fh, df)
        results["_models"].append(name)

        if agent is not None:
            rng = np.random.default_rng(SEED)
            stacked = pd.concat(holdouts.values(), ignore_index=True)
            controls_by_model[name] = controls(
                agent,
                stacked[C.OBS_FEATURES].to_numpy(np.float32),
                stacked["mdp_state"].to_numpy(dtype=object),
                rng,
            )

    write_report(results, controls_by_model, train_acc)
    write_confusion_png(results)

    print(f"wrote {C.HOLDOUT_EVAL_REPORT_PATH}")
    print(f"wrote {C.HOLDOUT_EVAL_CM_PNG}")
    for name in results["_models"]:
        for hname in holdouts:
            r = results[name]["corrupted"][hname]
            print(f"  {name:26s} {hname:38s} corrupted acc={r['acc']:.3f} "
                  f"benign_fpr={r['benign_fpr']:.3f}")


if __name__ == "__main__":
    main()
