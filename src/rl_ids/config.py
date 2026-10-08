"""Central configuration for the RL-IDS project (Day 1: dataset + preprocessing).

Everything downstream (environment, agents, evaluation) imports paths, the
feature list and the label->state mapping from here so there is exactly one
source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
HOLDOUT_PARQUET = PROCESSED_DIR / "holdout_infiltration.parquet"
SCALER_JSON = PROCESSED_DIR / "scaler.json"
REPORT_PATH = PROCESSED_DIR / "day1_report.md"

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

# Eval-only days: never seen during training (Day-7 generalization test).
HOLDOUT_DAYS = frozenset({"Wednesday-28-02-2018", "Thursday-01-03-2018"})

TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"   # verified against raw files
MIN_TIMESTAMP = "2010-01-01"              # rows before this have broken clocks (1970)
EMBEDDED_HEADER_VALUE = "Timestamp"      # value in Timestamp col marks a duplicated header row

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
}


def state_for_label(label: str) -> str:
    """Map a raw dataset label to its MDP state. Raises on unknown labels.

    Some days use 'DoS attacks-GoldenEye' where others use 'DoS-GoldenEye'
    or 'DDOS attack-HOIC'; fall back to normalized keys (infix ' attack(s)'
    removed) before giving up.
    """
    key = str(label).strip().lower()
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
HIDDEN_COLUMNS: list[str] = ["attack_label", "mdp_state", "TimestampEpoch", "source_day"]

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
# Day 2+ : MDP environment (Gymnasium)
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

GAMMA = 0.9  # discount factor used across Day 3-4 checkpoints

MAX_STEPS_DEFAULT = 100     # episode length cap (MC episodes start at Normal)
LATENCY_ALPHA = 0.25        # per-step latency penalty for DPI/Throttle overuse
OBS_NOISE_SIGMA = 0.05      # Gaussian noise added to every observed feature
OBS_MASK_PROB = 0.1         # per-feature probability of "missing telemetry" (-> 0)

# --------------------------------------------------------------------------- #
# MDP transition model  (state, action) -> {next_state: probability}
# --------------------------------------------------------------------------- #
# Slide-consistent model tuned (via the Bellman equations) so that the Day-4
# tabular agent converges to the expected q* (Normal->Monitor 18.29,
# Suspicious->Honeypot 28.61, Compromised->Block 31.06) under GAMMA=0.9
# when observations are the real dataset flows sampled from each state.
# "__terminal__" denotes episode end (incident resolved / Isolate).
#
#   N - Monitor  : .840 stay | .005 prelude | .155 sudden compromise
#   S - Honeypot : .295 benign resolved (-2) | .510 attack caught (+15, persists)
#                                                        | .195 escalates (+15)
#   C - Block    : .503 persists (-2) | .497 resolved (+18, terminal)
TRANSITIONS: dict[str, dict[str, dict[str, float]]] = {
    "Normal": {
        "Monitor": {"Normal": 0.840, "Suspicious": 0.005, "Compromised": 0.155},
    },
    "Suspicious": {
        "Monitor":  {"Suspicious": 0.70, "Compromised": 0.20, "Normal": 0.10},
        "DPI":      {"Suspicious": 0.50, "Compromised": 0.20, "Normal": 0.30},
        "Throttle": {"Suspicious": 0.55, "Compromised": 0.20, "Normal": 0.25},
        "Honeypot": {"Suspicious": 0.510, "Compromised": 0.195, "Normal": 0.295},
    },
    "Compromised": {
        "Monitor": {"Compromised": 0.90, "Normal": 0.10},
        "Block":   {"Compromised": 0.503, "Normal": 0.497},
        "Isolate": {"__terminal__": 1.0},
    },
}

TERMINAL_MARKER = "__terminal__"

# Max rows sampled per state for observation pools (bounded memory). The full
# 15.2M-row train set can be enabled by the training day (VM RAM upgrade).
DEFAULT_MAX_ROWS_PER_STATE = 50_000
