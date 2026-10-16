"""Coverage-ablation split tests.

The coverage corpus must be (a) additive - carved from the held-out days, never
from the original train split, (b) split so an *unseen* remainder stays for
testing, and (c) deterministic.  These pin the split contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_ids import config as C
from rl_ids.coverage_split import (
    MIN_TEST_ATTACK_ROWS, build_split, validate, _PORTSCAN_ATTACK_LABEL,
    _INFILTRATION_ATTACK_LABEL,
)


@pytest.fixture(scope="module")
def split() -> dict:
    for path in (C.COVERAGE_DAY_PORTSCAN, C.COVERAGE_DAY_INFILTRATION_TRAIN,
                 C.COVERAGE_DAY_INFILTRATION_TEST):
        assert path.exists(), f"missing {path} - run preprocess"
    return build_split()


def test_validate_passes(split):
    validate(split)  # must not raise


def test_split_is_deterministic():
    a = build_split()
    b = build_split()
    assert a["cut"] == b["cut"]
    assert len(a["train"]) == len(b["train"])
    assert len(a["portscan_test"]) == len(b["portscan_test"])
    assert len(a["infiltration_test"]) == len(b["infiltration_test"])
    assert a["train"].reset_index(drop=True).equals(b["train"].reset_index(drop=True))


def test_portscan_train_test_are_time_disjoint(split):
    tr, te = split["portscan_train"], split["portscan_test"]
    assert tr["TimestampEpoch"].max() <= split["cut"] < te["TimestampEpoch"].min()
    assert len(tr) > 0 and len(te) > 0


def test_portscan_scans_on_both_sides(split):
    n_tr = int((split["portscan_train"]["attack_label"] == _PORTSCAN_ATTACK_LABEL).sum())
    n_te = int((split["portscan_test"]["attack_label"] == _PORTSCAN_ATTACK_LABEL).sum())
    assert n_tr > 0, "no PortScan rows in the training portion"
    assert n_te >= MIN_TEST_ATTACK_ROWS


def test_infiltration_is_a_clean_day_split(split):
    tr, te = split["infiltration_train"], split["infiltration_test"]
    assert set(tr["source_day"].unique()) == {"Wednesday-28-02-2018"}
    assert set(te["source_day"].unique()) == {"Thursday-01-03-2018"}
    n_tr = int((tr["attack_label"] == _INFILTRATION_ATTACK_LABEL).sum())
    n_te = int((te["attack_label"] == _INFILTRATION_ATTACK_LABEL).sum())
    assert n_tr > 0 and n_te >= MIN_TEST_ATTACK_ROWS
    # whole days never overlap
    assert not (set(tr["source_day"]) & set(te["source_day"]))


def test_coverage_train_contains_both_new_concepts(split):
    tr = split["train"]
    assert (tr["attack_label"] == _PORTSCAN_ATTACK_LABEL).any()
    assert (tr["attack_label"] == _INFILTRATION_ATTACK_LABEL).any()


def test_coverage_train_is_a_subset_of_heldout_days(split):
    # additive only: the coverage train draw comes from the held-out days
    days = set(split["train"]["source_day"].unique())
    assert days == {"Friday-WorkingHours-Afternoon-PortScan", "Wednesday-28-02-2018"}


def test_written_artifacts_match_split(split):
    """If the parquet artifacts exist, they must equal a fresh build."""
    if not C.COVERAGE_TRAIN_PARQUET.exists():
        pytest.skip("coverage parquets not built")
    for path, frame in (
        (C.COVERAGE_TRAIN_PARQUET, split["train"]),
        (C.COVERAGE_TEST_PORTSCAN_PARQUET, split["portscan_test"]),
        (C.COVERAGE_TEST_INFILTRATION_PARQUET, split["infiltration_test"]),
    ):
        on_disk = pd.read_parquet(path)
        assert len(on_disk) == len(frame)
        assert list(on_disk["attack_label"]) == list(frame["attack_label"])


# The ``day_*.parquet`` files are written *before* preprocess.py scales the data,
# so they hold RAW feature values; ``train.parquet`` holds z-scores.  Forgetting
# to re-apply the train scaler silently feeds the networks huge raw magnitudes and
# destroys detection.  These features have large raw values (>=1e3); if they are
# scaled their magnitudes are O(1).
_BIG_RAW_FEATURES = (
    "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Duration",
)


@pytest.mark.parametrize("which", ["train", "portscan_test", "infiltration_test"])
def test_coverage_frames_are_scaled_like_train(split, which):
    frame = split[which]
    for f in _BIG_RAW_FEATURES:
        med = float(np.median(np.abs(frame[f].to_numpy(dtype=float))))
        assert med < 50.0, (
            f"{which}:{f} median|x|={med:.1f} looks unscaled - the day-parquet "
            "raw values leaked through; apply the train-split scaler")

