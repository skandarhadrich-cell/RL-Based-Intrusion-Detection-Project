"""Build the additive coverage corpus for the Suspicious/Infiltration ablation.

The baseline agents were trained with **no PortScan** (the whole Suspicious
bucket is FTP/SSH brute force) and only **36 Infiltration rows** (vs 137,720 in
the 2018 holdout).  This driver carves the missing concepts out of the
*currently held-out* days and splits them so a fair, unseen remainder stays for
testing - without touching the original ``train.parquet`` / ``holdout_*.parquet``.

Splits (see ``docs/coverage_ablation.md``):

* **PortScan (2017)** - the attack rows are clustered in the last ~30% of the
  day, so a naive time cut would strand them all on one side.  An **80/20 time
  cut** keeps scans on both sides (train ≈3.4k / test ≈1.8k Suspicious).
* **Infiltration (2018)** - split **day vs day** (Wed-28 train, Thu-01 test);
  both days have long attack-free gaps, so a within-day cut is fragile.

Writes (git-ignored parquet + a committed report):
  * ``data/processed/coverage_train.parquet``           (added to training)
  * ``data/processed/coverage_test_portscan.parquet``    (unseen Suspicious test)
  * ``data/processed/coverage_test_infiltration.parquet``(unseen Infiltration test)
  * ``data/processed/coverage_split_report.md``

Run:  PYTHONPATH=src python -m rl_ids.coverage_split
"""
from __future__ import annotations

import json

import pandas as pd

from rl_ids import config as C

MIN_TEST_ATTACK_ROWS = 1_500

# (parquet path, the attack label expected in it) for the two coverage tests.
_PORTSCAN_ATTACK_LABEL = "PortScan"
_INFILTRATION_ATTACK_LABEL = "Infilteration"


def _scaler() -> dict:
    """The scaler fitted on the **original train split** (preprocess.py).

    The ``day_*.parquet`` files are written *before* scaling, so they hold raw
    feature values while ``train.parquet`` / ``holdout_*.parquet`` hold z-scores.
    The coverage corpus must be put on the same scale as the models were trained
    on, using the *same* scaler (never a refit).
    """
    if not C.SCALER_JSON.exists():
        raise FileNotFoundError(f"missing {C.SCALER_JSON} - run preprocess")
    payload = json.loads(C.SCALER_JSON.read_text(encoding="utf-8"))
    missing = [f for f in C.OBS_FEATURES if f not in payload["mean"]]
    if missing:
        raise AssertionError(f"scaler.json is missing features: {missing}")
    return payload


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=C.OBS_FEATURES + C.HIDDEN_COLUMNS)
    payload = _scaler()
    for f in C.OBS_FEATURES:
        # scale in float64, then store as float32 so the schema matches
        # train.parquet / holdout_*.parquet exactly (avoids an Arrow concat error).
        df[f] = ((df[f].astype("float64") - payload["mean"][f])
                 / payload["scale"][f]).astype("float32")
    return df.reset_index(drop=True)


def _counts(df: pd.DataFrame) -> dict:
    """State + attack-label breakdown for the report/validation."""
    return {
        "n": int(len(df)),
        "states": {k: int(v) for k, v in df["mdp_state"].value_counts().items()},
        "labels": {k: int(v) for k, v in df["attack_label"].value_counts().items()},
    }


def build_split(frac: float = C.COVERAGE_TRAIN_FRAC) -> dict:
    """Return the coverage train/test frames (no files written)."""
    ps = _load(C.COVERAGE_DAY_PORTSCAN)
    lo = int(ps["TimestampEpoch"].min())
    hi = int(ps["TimestampEpoch"].max())
    cut = int(lo + frac * (hi - lo))

    portscan_train = ps[ps["TimestampEpoch"] <= cut].reset_index(drop=True)
    portscan_test = ps[ps["TimestampEpoch"] > cut].reset_index(drop=True)

    inf_train = _load(C.COVERAGE_DAY_INFILTRATION_TRAIN)
    inf_test = _load(C.COVERAGE_DAY_INFILTRATION_TEST)

    train = pd.concat([portscan_train, inf_train], ignore_index=True)
    return {
        "frac": frac, "lo": lo, "hi": hi, "cut": cut,
        "train": train,
        "portscan_train": portscan_train, "portscan_test": portscan_test,
        "infiltration_train": inf_train, "infiltration_test": inf_test,
    }


def validate(split: dict) -> None:
    """Fail loudly if the split would make the ablation unfair or impossible."""
    ps_tr, ps_te = split["portscan_train"], split["portscan_test"]
    inf_te = split["infiltration_test"]

    # 1. Train/test never share a PortScan row (disjoint by the time cut).
    if ps_tr["TimestampEpoch"].max() > ps_te["TimestampEpoch"].min():
        raise AssertionError("PortScan train/test time ranges overlap")

    # 2. Both coverage test sets carry enough attack rows to measure recall.
    n_ps = int((ps_te["attack_label"] == _PORTSCAN_ATTACK_LABEL).sum())
    n_inf = int((inf_te["attack_label"] == _INFILTRATION_ATTACK_LABEL).sum())
    if n_ps < MIN_TEST_ATTACK_ROWS:
        raise AssertionError(
            f"PortScan test has only {n_ps} attack rows (<{MIN_TEST_ATTACK_ROWS}); "
            f"lower COVERAGE_TRAIN_FRAC to enlarge it")
    if n_inf < MIN_TEST_ATTACK_ROWS:
        raise AssertionError(
            f"Infiltration test has only {n_inf} attack rows (<{MIN_TEST_ATTACK_ROWS})")

    # 3. The added training corpus actually contains the missing concepts.
    if int((ps_tr["attack_label"] == _PORTSCAN_ATTACK_LABEL).sum()) == 0:
        raise AssertionError("PortScan train portion has no PortScan rows")
    if int((split["infiltration_train"]["attack_label"]
            == _INFILTRATION_ATTACK_LABEL).sum()) == 0:
        raise AssertionError("Infiltration train day has no Infilteration rows")


def _write_report(split: dict) -> None:
    ts = pd.to_datetime(split["cut"], unit="s", utc=True)
    lines: list[str] = []
    add = lines.append
    add("# Coverage split report\n")
    add(f"`data/processed/{C.COVERAGE_SPLIT_REPORT_PATH.name}` - generated by "
        "`PYTHONPATH=src python -m rl_ids.coverage_split`\n")
    add("Additive corpus carved from the currently held-out days, split so an "
        "unseen remainder stays for testing. The original `train.parquet` and "
        "`holdout_*.parquet` are untouched.\n")
    add("**Feature scaling.** The `day_*.parquet` files are written *before* "
        "preprocessing scales the data, so they hold raw values. Every row here is "
        "z-scored with the **train-split scaler** (`scaler.json`, fit on the "
        "original train split only) so the coverage corpus matches the scale the "
        "models were trained on - no refit.\n")

    add("## Split rule\n")
    pct_train = int(split["frac"] * 100)
    pct_test = 100 - pct_train
    add(f"- PortScan day: **{pct_train}/{pct_test} time cut** at epoch {split['cut']} "
        f"({ts:%Y-%m-%d %H:%M} UTC); earlier -> train, later -> test.")
    add("- Infiltration: **day vs day** - Wed-28-02-2018 -> train, "
        "Thu-01-03-2018 -> test (no within-day leakage).\n")

    add("## Artifacts\n")
    add("| Artifact | Rows | Class breakdown | Attack-label breakdown |")
    add("|---|---:|---|---|")
    for name, frame in [
        ("coverage_train.parquet (PortScan-day 80% + Wed-28)", split["train"]),
        ("coverage_test_portscan.parquet (PortScan-day 20%)", split["portscan_test"]),
        ("coverage_test_infiltration.parquet (Thu-01)", split["infiltration_test"]),
    ]:
        c = _counts(frame)
        st = ", ".join(f"{k} {v:,}" for k, v in sorted(c["states"].items()))
        lb = ", ".join(f"{k} {v:,}" for k, v in sorted(c["labels"].items()))
        add(f"| `{name}` | {c['n']:,} | {st} | {lb} |")
    add("")

    add("## Why this split\n")
    n_ps_train = int((split["portscan_train"]["attack_label"] == _PORTSCAN_ATTACK_LABEL).sum())
    add(f"- PortScan attacks are **clustered late**: only {n_ps_train:,} of the "
        f"day's scans sit before the {pct_train}% cut, so a 60/40 or 70/30 cut "
        "would leave almost none in training. 80/20 is the balance point.")
    add("- Infiltration has attack-free gaps mid-day on both days, so whole-day "
        "separation is safer than a within-day cut. This mirrors the project's "
        "existing day-based holdout philosophy.\n")

    C.COVERAGE_SPLIT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    split = build_split()
    validate(split)

    C.COVERAGE_TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    split["train"].to_parquet(C.COVERAGE_TRAIN_PARQUET, index=False)
    split["portscan_test"].to_parquet(C.COVERAGE_TEST_PORTSCAN_PARQUET, index=False)
    split["infiltration_test"].to_parquet(C.COVERAGE_TEST_INFILTRATION_PARQUET, index=False)
    _write_report(split)

    for name, frame in [
        ("coverage_train", split["train"]),
        ("coverage_test_portscan", split["portscan_test"]),
        ("coverage_test_infiltration", split["infiltration_test"]),
    ]:
        c = _counts(frame)
        print(f"{name}: {c['n']:,} rows  states={c['states']}")
        print(f"    labels={c['labels']}")
    print(f"wrote {C.COVERAGE_TRAIN_PARQUET}")
    print(f"wrote {C.COVERAGE_TEST_PORTSCAN_PARQUET}")
    print(f"wrote {C.COVERAGE_TEST_INFILTRATION_PARQUET}")
    print(f"wrote {C.COVERAGE_SPLIT_REPORT_PATH}")


if __name__ == "__main__":
    main()
