"""Central configuration for the RL-IDS project (dataset + preprocessing).

Everything downstream (environment, agents, evaluation) imports paths, the
feature list and the label->state mapping from here so there is exactly one
source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
HOLDOUT_PARQUET = PROCESSED_DIR / "holdout_infiltration.parquet"
SCALER_JSON = PROCESSED_DIR / "scaler.json"
# The preprocessing report lives alongside the other generated reports in
# data/processed/ so there is a single, always-current copy.
REPORT_PATH = PROCESSED_DIR / "dataset_preprocessing_report.md"

# CIC-IDS2017 integration: the IDS2017 eval-only split gets its own file.
HOLDOUT_IDS2017_PARQUET = PROCESSED_DIR / "holdout_ids2017_portscan_ddos.parquet"

# --------------------------------------------------------------------------- #
# Dataset (CSE-CIC-IDS2018 processed CICFlowMeter exports, S3 bucket)
# --------------------------------------------------------------------------- #
RAW_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",   # FTP/SSH-BruteForce
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",    # DoS-GoldenEye, DoS-Slowloris
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",      # DoS-Slowhttptest, DoS-Hulk
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",    # DDoS (LOIC-HTTP/UDP), PortScan
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",   # DDoS-LOIC-UDP, DDoS-HOIC
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",    # Brute Force -Web/-XSS, SQL Injection
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",      # Brute Force -Web/-XSS, SQL Injection
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",   # Infiltration  (HOLDOUT)
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",    # Infiltration  (HOLDOUT)
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",      # Bot
]

# Eval-only days: never seen during training (generalization test).
HOLDOUT_DAYS = frozenset({"Wednesday-28-02-2018", "Thursday-01-03-2018"})

TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"   # verified against raw files
MIN_TIMESTAMP = "2010-01-01"              # rows before this have broken clocks (1970)
EMBEDDED_HEADER_VALUE = "Timestamp"      # value in Timestamp col marks a duplicated header row

# --------------------------------------------------------------------------- #
# Multi-source layout: CSE-CIC-IDS2018 + CIC-IDS2017
# --------------------------------------------------------------------------- #
# Each source is described by one SourceSpec so the whole pipeline (preprocess,
# env, reports, tests) is source-agnostic.  Downstream code never hardcodes a
# dataset again: it iterates config.SOURCES.
SOURCE_2018 = "ids2018"
SOURCE_2017 = "ids2017"
SOURCE_DATASET_COLUMN = "source_dataset"   # provenance column (hidden from the agent)

# CIC-IDS2017 eval-only capture files: the PortScan file (Suspicious) and the
# DDoS file (Compromised) are NEVER trained on and NEVER used to fit the scaler.
HOLDOUT_DAYS_2017 = frozenset({
    "Friday-WorkingHours-Afternoon-DDos",
    "Friday-WorkingHours-Afternoon-PortScan",
})

# CIC-IDS2017 MachineLearningCVE column names -> the canonical CICFlowMeter-2018
# names used everywhere in this project (leading spaces stripped first).  Every
# feature we consume is a genuine same-semantics/units rename, never a guess:
#   Destination Port -> Dst Port                (both: TCP/UDP destination port)
#   Total Fwd Packets -> Tot Fwd Pkts           (both: packet count)
#   Total Length of Fwd Packets -> TotLen Fwd Pkts (both: bytes)
#   Packet Length Mean -> Pkt Len Mean          (both: bytes)
#   * Flag Count -> * Flag Cnt                  (both: counts)
IDS2017_RENAMES: dict[str, str] = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Packet Length Mean": "Pkt Len Mean",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
}

# Rate columns CIC-IDS2017 ships (already computed by CICFlowMeter) -> canonical
# names.  Our pipeline recomputes these from totals/duration, so these are used
# only by the feature-parity cross-check (tests/test_ids2017_integration.py), never as
# model input.
IDS2017_SHIPPED_RATES: dict[str, str] = {
    "Flow Packets/s": "Flow Pkts/s",
    "Flow Bytes/s": "Flow Byts/s",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
}


def _day_2018(filename: str) -> str:
    """'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv' -> 'Wednesday-14-02-2018'."""
    return filename.split("_")[0]


def _day_2017(filename: str) -> str:
    """'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv' -> day stem."""
    return filename.split(".")[0]


@dataclass(frozen=True)
class SourceSpec:
    """Everything source-specific, kept in one place per dataset."""

    name: str                              # SOURCE_2018 / SOURCE_2017
    raw_dir: Path                          # folder holding the raw CSVs
    files: tuple[str, ...]                 # raw CSV filenames to process
    timestamp_col: str                     # name of the timestamp column
    timestamp_format: str                  # primary strptime format
    min_timestamp: str                     # rows before this have broken clocks
    embedded_header_value: str | None      # a value that marks an embedded header row
    holdout_days: frozenset[str]           # day keys that must never be trained on
    day_of: Callable[[str], str]           # filename -> day key
    renames: dict[str, str] = field(default_factory=dict)  # raw(stripped) -> canonical
    encoding: str | None = None            # CSV text encoding
    alternate_timestamp_formats: tuple[str, ...] = ()   # fallbacks for mixed formats

    def canonical(self, raw_column: str) -> str:
        """Canonical project column name for a raw (possibly space-padded) header."""
        key = raw_column.strip()
        return self.renames.get(key, key)


IDS2018_SPEC = SourceSpec(
    name=SOURCE_2018,
    raw_dir=RAW_DIR,
    files=tuple(RAW_FILES),
    timestamp_col="Timestamp",
    timestamp_format=TIMESTAMP_FORMAT,
    min_timestamp=MIN_TIMESTAMP,
    embedded_header_value=EMBEDDED_HEADER_VALUE,
    holdout_days=HOLDOUT_DAYS,
    day_of=_day_2018,
    renames={},
    encoding=None,
)

# CIC-IDS2017 raw directory (GeneratedLabelledFlows / "TrafficLabelling" export:
# it adds Timestamp + Flow ID + IPs to the MachineLearningCVE columns, which is
# what lets us recompute Active Conn Cnt with the identical sweep-line code).
IDS2017_DIR = RAW_DIR / "ids2017" / "TrafficLabelling"

IDS2017_SPEC = SourceSpec(
    name=SOURCE_2017,
    raw_dir=IDS2017_DIR,
    files=(
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ),
    timestamp_col="Timestamp",
    # Verified against the raw files: CICFlowMeter-2017 emits MINUTE resolution
    # ("7/7/2017 1:00"), vs CICFlowMeter-2018's second resolution.  The derived
    # Active Conn Cnt therefore has the same meaning/units but coarser temporal
    # granularity for 2017 (documented in docs/dataset_preprocessing.md).
    timestamp_format="%d/%m/%Y %H:%M",
    min_timestamp=MIN_TIMESTAMP,
    embedded_header_value=None,
    holdout_days=HOLDOUT_DAYS_2017,
    day_of=_day_2017,
    renames=IDS2017_RENAMES,
    encoding="latin-1",
    # The CIC-IDS2017 exports mix formats ACROSS files: minute resolution
    # ("7/7/2017 1:00") almost everywhere, but Monday-WorkingHours uses seconds
    # ("03/07/2017 08:55:58").  Try minute first, then seconds.
    alternate_timestamp_formats=("%d/%m/%Y %H:%M:%S",),
)

SOURCES: tuple[SourceSpec, ...] = (IDS2018_SPEC, IDS2017_SPEC)
SOURCES_BY_NAME: dict[str, SourceSpec] = {s.name: s for s in SOURCES}

# --------------------------------------------------------------------------- #
# MDP states and label mapping
# --------------------------------------------------------------------------- #
STATES = ("Normal", "Suspicious", "Compromised")

# Dataset label (lower-cased, stripped) -> MDP state.
#   Normal       : benign traffic
#   Suspicious   : low-confidence anomalies (port scan, early-stage brute force)
#   Compromised  : confirmed attacks (DoS, DDoS, Botnet, Infiltration, Web-Attack)
LABEL_TO_STATE: dict[str, str] = {
    "benign": "Normal",
    # --- low-confidence / early-stage ------------------------------------- #
    "ftp-bruteforce": "Suspicious",
    "ssh-bruteforce": "Suspicious",
    "portscan": "Suspicious",
    # --- confirmed attacks ------------------------------------------------- #
    "dos-goldeneye": "Compromised",
    "dos-slowloris": "Compromised",
    "dos-slowhttptest": "Compromised",
    "dos-hulk": "Compromised",
    "ddos": "Compromised",
    "ddos-loic-udp": "Compromised",
    "ddos-loic-http": "Compromised",
    "ddos-hoic": "Compromised",
    "ddos attacks-loic-http": "Compromised",
    "ddos attacks-loic-udp": "Compromised",
    "ddos attack-hoic": "Compromised",
    "ddos attack-loic-udp": "Compromised",
    "bot": "Compromised",
    "infiltration": "Compromised",
    "infilteration": "Compromised",   # typo present in the original dataset
    "brute force": "Compromised",
    "brute force -web": "Compromised",
    "brute force -xss": "Compromised",
    "sql injection": "Compromised",
    "web attack": "Compromised",
    "web attack - brute force": "Compromised",
    "web attack - xss": "Compromised",
    "web attack - sql injection": "Compromised",
    "heartbleed": "Compromised",
    # --- CIC-IDS2017 label spellings (space instead of hyphen; "-Patator" for
    #     brute force; en-dash in "Web Attack <dash> ...") --------------------- #
    "dos hulk": "Compromised",
    "dos goldeneye": "Compromised",
    "dos slowloris": "Compromised",
    "dos slowhttptest": "Compromised",
    "ftp-patator": "Suspicious",
    "ssh-patator": "Suspicious",
}

# Dash-like code points that appear in raw labels: en/em dashes, the cp1252
# mojibake byte 0x96 (read as U+0096 under latin-1) and U+FFFD (utf-8 replace).
_DASH_LIKE = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015",
              "\u0096", "\u0097", "\ufffd")


def _normalize_label_key(label: str) -> str:
    """Lower-case/strip and collapse every dash variant to an ASCII hyphen."""
    key = str(label).strip().lower()
    for dash in _DASH_LIKE:
        key = key.replace(dash, "-")
    return key


def state_for_label(label: str) -> str:
    """Map a raw dataset label to its MDP state. Raises on unknown labels.

    Normalises dash-like characters first (the CIC-IDS2017 exports encode the
    'Web Attack - *' en-dash as cp1252 byte 0x96), then falls back to normalized
    keys (infix ' attack(s)' removed) before giving up.
    """
    key = _normalize_label_key(label)
    for candidate in (key, key.replace(" attacks", ""), key.replace(" attack", "")):
        if candidate in LABEL_TO_STATE:
            return LABEL_TO_STATE[candidate]
    raise KeyError(
        f"Unmapped dataset label {label!r} - extend LABEL_TO_STATE in "
        f"src/rl_ids/config.py"
    )


# --------------------------------------------------------------------------- #
# Observation features (what the agent sees)
# --------------------------------------------------------------------------- #
# Raw CSV columns required from the 80-column CICFlowMeter export.
RAW_COLUMNS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Pkt Len Mean",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Label",
]

# Final observation vector (order matters - the scaler/agent use this order).
# 80 CICFlowMeter features -> state features of the MDP.
OBS_FEATURES: list[str] = [
    # packet rate
    "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s",
    # byte count / byte rate
    "Flow Byts/s", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    # active connection count (derived: concurrent flows at flow start)
    "Active Conn Cnt",
    # flow context
    "Flow Duration", "Pkt Len Mean",
    # TCP flag distribution
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    # destination port status + protocol
    "Dst Port Bucket", "Protocol",
]

# Columns kept for reward computation / evaluation only - NEVER observed.
HIDDEN_COLUMNS: list[str] = [
    "attack_label", "mdp_state", "TimestampEpoch", "source_day",
    SOURCE_DATASET_COLUMN,
]

# Derived numeric columns produced by preprocessing (subset of OBS_FEATURES).
DERIVED_COLUMNS: list[str] = [
    "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
    "Active Conn Cnt", "Dst Port Bucket",
]

# Dst Port -> ordinal bucket ("destination port status").
#   0: service/privileged (<1024), 1: registered (1024-49151), 2: ephemeral (>49151)
PORT_BUCKET_EDGES = (1024, 49152)

BATCH_CHUNKSIZE = 250_000  # rows per read_csv chunk

# --------------------------------------------------------------------------- #
# MDP environment (Gymnasium)
# --------------------------------------------------------------------------- #
# The agent sees only the 17-feature observation vector. The *true* MDP state
# (label bucket) is hidden and drives transitions + rewards (eval info only).
STATES: tuple[str, ...] = ("Normal", "Suspicious", "Compromised")  # re-declared (already above)

# Discrete actions of the IR playbook.  Index == Gymnasium action id.
ACTIONS: tuple[str, ...] = (
    "Monitor",   # 0  watch only, no cost, no detection claim
    "DPI",       # 1  deep-packet inspection
    "Throttle",  # 2  rate-limit the flow
    "Honeypot",  # 3  lure into a decoy
    "Block",     # 4  drop the flow
    "Isolate",   # 5  hard quarantine (terminal)
)
ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}

# Playbook mask: p(state) -> allowed action ids.
VALID_ACTIONS: dict[str, tuple[int, ...]] = {
    "Normal": (ACTION_INDEX["Monitor"],),
    "Suspicious": (ACTION_INDEX["Monitor"], ACTION_INDEX["DPI"],
                   ACTION_INDEX["Throttle"], ACTION_INDEX["Honeypot"]),
    "Compromised": (ACTION_INDEX["Monitor"], ACTION_INDEX["Block"],
                    ACTION_INDEX["Isolate"]),
}

# Mask-free mode (Path 2): every action is offered in every state. The playbook
# mask above is then ignored by the environment (observations and rewards are
# unchanged); the agent must infer the hidden bucket to find the optimal policy.
ALL_ACTION_INDICES: tuple[int, ...] = tuple(range(len(ACTIONS)))

# Operating cost charged per action execution (per single flow step).
ACTION_COSTS: dict[int, float] = {
    ACTION_INDEX[n]: c for n, c in (
        ("Monitor", 0.0), ("DPI", 1.0), ("Throttle", 1.5),
        ("Honeypot", 3.0), ("Block", 2.0), ("Isolate", 15.0),
    )
}

# Detection events (confusion-matrix payoff for the IR decision).
DETECTION_REWARDS: dict[str, float] = {
    "TN": 1.0,    # benign flow handled with a light touch
    "TP": 18.0,   # confirmed attack caught + resolved
    "FP": -12.0,  # heavy action wasted on benign traffic
    "FN": -40.0,  # attack slipped through (or escalated)
}

# Masked / illegal actions must never be profitable.
INVALID_ACTION_PENALTY = -100.0

GAMMA = 0.9  # discount factor used across the reward/tabular checkpoints

MAX_STEPS_DEFAULT = 100     # episode length cap (MC episodes start at Normal)
LATENCY_ALPHA = 0.25        # per-step latency penalty for DPI/Throttle overuse
OBS_NOISE_SIGMA = 0.05      # Gaussian noise added to every observed feature
OBS_MASK_PROB = 0.1         # per-feature probability of "missing telemetry" (-> 0)

# --------------------------------------------------------------------------- #
# MDP transition model  (state, action) -> {next_state: probability}
# --------------------------------------------------------------------------- #
# Slide-consistent model tuned (via the Bellman equations) so that the tabular
# tabular agent converges to the expected q* (Normal->Monitor 18.29,
# Suspicious->Honeypot 28.61, Compromised->Block 31.06) under GAMMA=0.9
# when observations are the real dataset flows sampled from each state.
# "__terminal__" denotes episode end (incident resolved / Isolate).
#
#   N - Monitor  : .840 stay | .005 prelude | .155 sudden compromise
#   S - Honeypot : .295 benign resolved (-2) | .510 attack caught (+15, persists)
#                                                        | .195 escalates (+15)
#   C - Block    : .503 persists (-2) | .497 resolved (+18, terminal)
# The first entry per state is the playbook action; the remaining entries exist
# only for MASK-FREE mode (Path 2), where every action is offered in every state.
# The playbook rows are untouched, so the masked MDP is byte-identical.
TRANSITIONS: dict[str, dict[str, dict[str, float]]] = {
    "Normal": {
        "Monitor":  {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
        # In a truly-Normal state the attack dynamics are a property of the
        # state, not of the action; off-playbook actions only differ in reward.
        "DPI":      {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
        "Throttle": {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
        "Honeypot": {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
        "Block":    {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
        "Isolate":  {"__terminal__": 1.0},
    },
    "Suspicious": {
        "Monitor":  {"Suspicious": 0.70, "Compromised": 0.20, "Normal": 0.10},
        "DPI":      {"Suspicious": 0.50, "Compromised": 0.20, "Normal": 0.30},
        "Throttle": {"Suspicious": 0.55, "Compromised": 0.20, "Normal": 0.25},
        "Honeypot": {"Suspicious": 0.510, "Compromised": 0.195, "Normal": 0.295},
        # Off-playbook: a hard Block resolves suspicious traffic more often.
        "Block":    {"Normal": 0.40, "Suspicious": 0.40, "Compromised": 0.20},
        "Isolate":  {"__terminal__": 1.0},
    },
    "Compromised": {
        "Monitor":  {"Compromised": 0.90, "Normal": 0.10},
        "Block":    {"Compromised": 0.503, "Normal": 0.497},
        "Isolate":  {"__terminal__": 1.0},
        # Off-playbook: softer actions contain a confirmed compromise less often.
        "DPI":      {"Compromised": 0.85, "Normal": 0.15},
        "Throttle": {"Compromised": 0.65, "Normal": 0.35},
        "Honeypot": {"Compromised": 0.60, "Normal": 0.40},
    },
}

TERMINAL_MARKER = "__terminal__"

# Max rows sampled per state for observation pools (bounded memory). The full
# 16.7M-row merged train set can be enabled with more VM RAM; the DQN run raises
# this via config.DQN_MAX_ROWS_PER_STATE.
DEFAULT_MAX_ROWS_PER_STATE = 50_000


# --------------------------------------------------------------------------- #
# Deep RL - Double DQN (DQN phase)
# --------------------------------------------------------------------------- #
# The Q-network is an MLP over the 17-feature observation. The playbook action
# mask is applied externally at every argmax/selection (it is not an
# observation).
DQN_HIDDEN: tuple[int, ...] = (128, 128)
DQN_LR = 1e-3
DQN_BATCH_SIZE = 64
DQN_REPLAY_CAPACITY = 200_000
DQN_REPLAY_WARMUP = 2_000          # uniform-random steps before learning starts
DQN_TARGET_SYNC = 1_000            # hard-copy target network every N updates
DQN_GRAD_CLIP = 10.0               # max-norm gradient clipping (None disables)
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 0.995              # per-episode multiplicative decay
DQN_MAX_ROWS_PER_STATE = 200_000   # obs pool cap for the DQN run (RAM-dependent)

DQN_REPORT_PATH = PROCESSED_DIR / "double_dqn_report.md"
DQN_CURVE_PNG = PROCESSED_DIR / "double_dqn_learning_curve.png"
MODELS_DIR = PROJECT_ROOT / "models"
DQN_MODEL_PATH = MODELS_DIR / "double_dqn.pt"

# --------------------------------------------------------------------------- #
# Prioritized Experience Replay (proportional, sum-tree) - PER phase
# --------------------------------------------------------------------------- #
PER_ALPHA = 0.6             # priority exponent: P(i) ∝ (|TD| + eps)^alpha
PER_BETA_START = 0.4        # importance-sampling exponent at the start
PER_BETA_END = 1.0          # ... annealed linearly to 1.0 (fully corrected)
PER_BETA_STEPS = 20_000     # updates over which beta is annealed
PER_EPS = 1e-5              # priority floor, keeps every transition sampleable

PER_REPORT_PATH = PROCESSED_DIR / "prioritized_replay_report.md"
PER_CURVE_PNG = PROCESSED_DIR / "prioritized_replay_learning_curve.png"
PER_MODEL_PATH = MODELS_DIR / "double_dqn_per.pt"

# --------------------------------------------------------------------------- #
# Mask-free mode (Path 2) - the agent is offered every action in every state
# --------------------------------------------------------------------------- #
MASKFREE_DQN_REPORT_PATH = PROCESSED_DIR / "maskfree_dqn_report.md"
MASKFREE_DQN_CURVE_PNG = PROCESSED_DIR / "maskfree_dqn_learning_curve.png"
MASKFREE_DQN_MODEL_PATH = MODELS_DIR / "double_dqn_maskfree.pt"
MASKFREE_PER_REPORT_PATH = PROCESSED_DIR / "maskfree_per_report.md"
MASKFREE_PER_CURVE_PNG = PROCESSED_DIR / "maskfree_per_learning_curve.png"
MASKFREE_PER_MODEL_PATH = MODELS_DIR / "double_dqn_per_maskfree.pt"
MASKFREE_TABULAR_REPORT_PATH = PROCESSED_DIR / "maskfree_tabular_report.md"
MASKFREE_TABULAR_CURVE_PNG = PROCESSED_DIR / "maskfree_tabular_learning_curve.png"
MASKFREE_DETECT_REPORT_PATH = PROCESSED_DIR / "maskfree_holdout_detection_report.md"
MASKFREE_DETECT_CM_PNG = PROCESSED_DIR / "maskfree_holdout_confusion_matrix.png"

# --------------------------------------------------------------------------- #
# Holdout per-class detection evaluation (probe on the learned representation)
# --------------------------------------------------------------------------- #
HOLDOUT_EVAL_REPORT_PATH = PROCESSED_DIR / "holdout_classification_report.md"
HOLDOUT_EVAL_CM_PNG = PROCESSED_DIR / "holdout_confusion_matrix.png"
