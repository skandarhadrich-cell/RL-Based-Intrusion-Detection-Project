"""CIC-IDS2017 integration tests: the 2017 source in the merged pipeline.

These assert the requirements of the controlled integration experiment:

  1. the 17-feature observation schema is preserved and each 2017 feature has
     the same meaning/units as its 2018 counterpart (rate cross-check against
     the rate columns CIC-IDS2017 ships itself);
  2. the absent feature (Active Conn Cnt) is *derived* from 2017 timestamps,
     never zero-filled;
  3. label mapping + provenance are explicit (attack_label + source_dataset);
  4. the 2017 eval-only split (PortScan + DDoS) never touches training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_ids import config as C


@pytest.fixture(scope="session")
def train() -> pd.DataFrame:
    assert C.TRAIN_PARQUET.exists(), f"missing {C.TRAIN_PARQUET} - run preprocess"
    return pd.read_parquet(C.TRAIN_PARQUET)


@pytest.fixture(scope="session")
def holdout2017() -> pd.DataFrame:
    assert C.HOLDOUT_IDS2017_PARQUET.exists(), "run preprocess"
    return pd.read_parquet(C.HOLDOUT_IDS2017_PARQUET)


# --------------------------------------------------------------------------- #
# 4. Eval-only split is correct and never trained on
# --------------------------------------------------------------------------- #
def test_2017_holdout_is_portscan_and_ddos(holdout2017):
    days = set(holdout2017["source_day"].unique())
    assert days == {
        "Friday-WorkingHours-Afternoon-PortScan",
        "Friday-WorkingHours-Afternoon-DDos",
    }
    assert set(holdout2017[C.SOURCE_DATASET_COLUMN].unique()) == {C.SOURCE_2017}
    # Both attack files also carry BENIGN background traffic, so Normal appears
    # alongside the two attack states; what matters is that the attack states
    # are represented and mapped correctly.
    states = set(holdout2017["mdp_state"].unique())
    assert {"Suspicious", "Compromised"} <= states, states
    assert states <= {"Normal", "Suspicious", "Compromised"}, states

    portscan = holdout2017[holdout2017["source_day"].str.contains("PortScan")]
    assert set(portscan["mdp_state"].unique()) <= {"Normal", "Suspicious"}
    assert (portscan["mdp_state"] == "Suspicious").any()
    assert (portscan["attack_label"] == "PortScan").any()

    ddos = holdout2017[holdout2017["source_day"].str.contains("DDos")]
    assert set(ddos["mdp_state"].unique()) <= {"Normal", "Compromised"}
    assert (ddos["mdp_state"] == "Compromised").any()
    assert (ddos["attack_label"] == "DDoS").any()


def test_2017_holdout_absent_from_train(train):
    train_days = set(train["source_day"].unique())
    assert not (train_days & set(C.HOLDOUT_DAYS_2017))


# --------------------------------------------------------------------------- #
# Observation schema preserved
# --------------------------------------------------------------------------- #
def test_observation_schema_unchanged(train, holdout2017):
    assert len(C.OBS_FEATURES) == 17
    expected = set(C.OBS_FEATURES + C.HIDDEN_COLUMNS)
    assert set(train.columns) == expected
    assert set(holdout2017.columns) == expected


# --------------------------------------------------------------------------- #
# 2. Active Conn Cnt is DERIVED for 2017 (not zero-filled to match the schema)
# --------------------------------------------------------------------------- #
def test_active_conn_cnt_derived_not_zero_filled(train):
    ids2017 = train[train[C.SOURCE_DATASET_COLUMN] == C.SOURCE_2017]
    # NOTE: train.parquet holds *standardized* observations, so negative values
    # are expected here even though raw connection counts are non-negative.
    # A zero-filled feature would collapse to a single constant after z-scoring
    # (std == 0); real sweep-line concurrency does not.
    acc = ids2017["Active Conn Cnt"].to_numpy()
    assert acc.size > 0
    assert np.isfinite(acc).all()
    assert float(acc.std()) > 0.01, "Active Conn Cnt is ~constant -> looks zero-filled"
    assert np.unique(acc).size > 1000, "Active Conn Cnt has too few distinct values"
    assert float(acc.max() - acc.min()) > 1.0, "Active Conn Cnt range too narrow"


# --------------------------------------------------------------------------- #
# 1. Feature parity: 2017 columns map to the canonical schema, and our
#    recomputed rates reproduce the rates CIC-IDS2017 ships (unit check).
# --------------------------------------------------------------------------- #
def test_all_canonical_columns_present_in_2017():
    spec = C.IDS2017_SPEC
    path = spec.raw_dir / spec.files[0]
    raw = pd.read_csv(path, nrows=0, encoding=spec.encoding)
    canonical = {spec.canonical(col) for col in raw.columns}
    missing = set(C.RAW_COLUMNS) - canonical
    assert not missing, f"2017 header cannot produce canonical {missing}"


def test_recomputed_rates_match_2017_shipped_rates():
    """Flow Pkts/s and Flow Byts/s recomputed from 2017 totals+duration must
    match the rates CIC-IDS2017 ships itself -> same units (packets/s, bytes/s,
    microsecond duration) across both datasets."""
    spec = C.IDS2017_SPEC
    path = spec.raw_dir / "Wednesday-workingHours.pcap_ISCX.csv"
    df = pd.read_csv(path, nrows=50_000, encoding=spec.encoding, low_memory=False)
    df = df.rename(columns={c: spec.canonical(c) for c in df.columns})

    dur_s = pd.to_numeric(df["Flow Duration"], errors="coerce") / 1e6
    ok = dur_s > 0
    fwd = pd.to_numeric(df["Tot Fwd Pkts"], errors="coerce")
    bwd = pd.to_numeric(df["Tot Bwd Pkts"], errors="coerce")
    fwd_len = pd.to_numeric(df["TotLen Fwd Pkts"], errors="coerce")
    bwd_len = pd.to_numeric(df["TotLen Bwd Pkts"], errors="coerce")

    our_pkts = ((fwd + bwd) / dur_s)[ok].to_numpy(dtype=float)
    our_byts = ((fwd_len + bwd_len) / dur_s)[ok].to_numpy(dtype=float)
    shipped_pkts = pd.to_numeric(df["Flow Packets/s"], errors="coerce")[ok].to_numpy(dtype=float)
    shipped_byts = pd.to_numeric(df["Flow Bytes/s"], errors="coerce")[ok].to_numpy(dtype=float)

    good_p = np.isfinite(our_pkts) & np.isfinite(shipped_pkts) & (shipped_pkts > 0)
    good_b = np.isfinite(our_byts) & np.isfinite(shipped_byts) & (shipped_byts > 0)
    assert good_p.sum() > 100 and good_b.sum() > 100

    med_rel_p = float(np.median(np.abs(our_pkts[good_p] - shipped_pkts[good_p])
                                / shipped_pkts[good_p]))
    med_rel_b = float(np.median(np.abs(our_byts[good_b] - shipped_byts[good_b])
                                / shipped_byts[good_b]))
    assert med_rel_p < 0.01, f"Flow Pkts/s parity off (median rel err {med_rel_p:.4f})"
    assert med_rel_b < 0.01, f"Flow Byts/s parity off (median rel err {med_rel_b:.4f})"
    corr = np.corrcoef(our_byts[good_b], shipped_byts[good_b])[0, 1]
    assert corr > 0.999, f"Flow Byts/s correlation {corr:.5f}"


# --------------------------------------------------------------------------- #
# 3. Explicit label policy for the CIC-IDS2017 label spellings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,state", [
    ("BENIGN", "Normal"),
    ("DoS Hulk", "Compromised"),
    ("DoS GoldenEye", "Compromised"),
    ("DoS slowloris", "Compromised"),
    ("DoS Slowhttptest", "Compromised"),
    ("DDoS", "Compromised"),
    ("FTP-Patator", "Suspicious"),
    ("SSH-Patator", "Suspicious"),
    ("Bot", "Compromised"),
    ("Heartbleed", "Compromised"),
    ("Infiltration", "Compromised"),
    ("PortScan", "Suspicious"),
    ("Web Attack \u0096 Brute Force", "Compromised"),   # cp1252 en-dash byte
    ("Web Attack \u2013 XSS", "Compromised"),            # real en-dash
    ("Web Attack \ufffd Sql Injection", "Compromised"),  # utf-8 replacement
])
def test_2017_label_policy(label, state):
    assert C.state_for_label(label) == state


def test_2017_attack_labels_preserved(train):
    """Original attack strings + dataset source are preserved (not collapsed)."""
    ids2017 = train[train[C.SOURCE_DATASET_COLUMN] == C.SOURCE_2017]
    labels = set(ids2017["attack_label"].unique())
    assert "BENIGN" in labels
    assert {"FTP-Patator", "SSH-Patator"} & labels, labels
