"""Day 1 checkpoint tests.

Checkpoint (from the plan):
  df.isna().sum().sum() == 0, class balance printed, and a 5-row sample
  manually traced to the correct MDP state.
"""
import json

import numpy as np
import pandas as pd
import pytest

from rl_ids import config as C


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def train() -> pd.DataFrame:
    assert C.TRAIN_PARQUET.exists(), f"missing {C.TRAIN_PARQUET} - run preprocess"
    return pd.read_parquet(C.TRAIN_PARQUET)


@pytest.fixture(scope="session")
def holdout() -> pd.DataFrame:
    assert C.HOLDOUT_PARQUET.exists(), f"missing {C.HOLDOUT_PARQUET} - run preprocess"
    return pd.read_parquet(C.HOLDOUT_PARQUET)


@pytest.fixture(scope="session")
def scaler() -> dict:
    assert C.SCALER_JSON.exists(), f"missing {C.SCALER_JSON} - run preprocess"
    return json.loads(C.SCALER_JSON.read_text())


# --------------------------------------------------------------------------- #
# 1. Checkpoint: zero NaN / zero Inf
# --------------------------------------------------------------------------- #
def test_no_nan_no_inf(train, holdout):
    for name, df in (("train", train), ("holdout", holdout)):
        n_nan = int(df.isna().sum().sum())
        assert n_nan == 0, f"{name} has {n_nan} NaN cells"
        numeric = df.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all(), f"{name} has +/-Inf values"


# --------------------------------------------------------------------------- #
# 2. Holdout isolation: Infiltration days never appear in train
# --------------------------------------------------------------------------- #
def test_holdout_isolation(train, holdout):
    train_days = set(train["source_day"].unique())
    holdout_days = set(holdout["source_day"].unique())
    assert holdout_days == set(C.HOLDOUT_DAYS), f"unexpected holdout days {holdout_days}"
    assert not (train_days & holdout_days), "holdout days leaked into training"
    expected_raw = {p.name.split("_")[0] for p in C.RAW_DIR.glob("*.csv")}
    assert expected_raw == (train_days | holdout_days), (
        f"day mismatch: raw={sorted(expected_raw)} "
        f"processed={sorted(train_days | holdout_days)}"
    )


# --------------------------------------------------------------------------- #
# 3. Label -> state mapping consistency (every row, both splits)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", ["train", "holdout"])
def test_state_mapping_consistent(split, train, holdout):
    df = train if split == "train" else holdout
    expected = df["attack_label"].map(C.state_for_label)
    bad = df.loc[expected != df["mdp_state"], ["attack_label", "mdp_state"]]
    assert bad.empty, f"state mismatch rows:\n{bad.head()}"


def test_all_three_states_present(train):
    present = set(train["mdp_state"].unique())
    assert present == set(C.STATES), f"missing states: {set(C.STATES) - present}"


# --------------------------------------------------------------------------- #
# 4. No label leakage into observations
# --------------------------------------------------------------------------- #
def test_no_label_leakage():
    overlap = set(C.OBS_FEATURES) & set(C.HIDDEN_COLUMNS)
    assert not overlap, f"hidden columns leaked into OBS_FEATURES: {overlap}"
    for forbidden in ("attack_label", "mdp_state", "Label", "source_day",
                      "TimestampEpoch"):
        assert forbidden not in C.OBS_FEATURES


# --------------------------------------------------------------------------- #
# 5. Scaler was fit on train only and covers every observation feature
# --------------------------------------------------------------------------- #
def test_scaler_complete(scaler):
    assert scaler["features"] == C.OBS_FEATURES
    for feat in C.OBS_FEATURES:
        assert np.isfinite(scaler["mean"][feat])
        assert np.isfinite(scaler["scale"][feat]) and scaler["scale"][feat] > 0


def test_train_obs_are_standardized(train):
    """Train split must be z-scored (mean ~0, std ~1) - scaler fit on train."""
    stats = train[C.OBS_FEATURES].mean().abs()
    assert float(stats.max()) < 1e-3, f"train obs not centered: {stats.idxmax()}"


# --------------------------------------------------------------------------- #
# 6. Checkpoint: class balance printed + 5-row manual trace
# --------------------------------------------------------------------------- #
def test_class_balance_printed(train, holdout):
    counts = train["mdp_state"].value_counts().reindex(C.STATES, fill_value=0)
    total = int(counts.sum())
    print("\nClass balance (train):")
    for state, n in counts.items():
        print(f"  {state:<12} {n:>10,}  ({100 * n / total:5.2f}%)")
    print(f"  {'TOTAL':<12} {total:>10,}")
    print(f"Holdout rows:   {len(holdout):,} "
          f"({', '.join(sorted(holdout['source_day'].unique()))})")
    for state in C.STATES:
        assert counts[state] > 0, f"state {state} has zero training samples"


def test_five_row_sample_trace(train, holdout):
    """Manual trace: 1 exemplar per state + 2 random rows, label -> state."""
    rng = np.random.default_rng(42)
    parts = [train[train["mdp_state"] == s].head(1) for s in C.STATES]
    extra = train.iloc[rng.choice(len(train), size=2, replace=False)]
    sample = pd.concat(parts + [extra]).head(5)

    print("\n5-row sample trace:")
    rows = []
    for _, r in sample.iterrows():
        expected = C.state_for_label(r["attack_label"])
        assert expected == r["mdp_state"], (
            f"row trace failed: {r['attack_label']!r} -> {expected} != {r['mdp_state']}"
        )
        rows.append((r["source_day"], r["attack_label"], r["mdp_state"], expected))
    print(f"  {'source_day':<22} {'attack_label':<20} "
          f"{'mdp_state':<13} expected")
    for day, label, state, expected in rows:
        print(f"  {day:<22} {label:<20} {state:<13} {expected}")

    # Holdout exemplar must trace too (eval data sanity).
    h = holdout.iloc[0]
    assert C.state_for_label(h["attack_label"]) == h["mdp_state"]
