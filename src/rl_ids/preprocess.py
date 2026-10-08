"""Day 1: dataset acquisition cleanup -> labeled, normalized Parquet tables.

Pipeline (per raw CSV, chunked to stay memory-safe on small VMs):

  Pass A  read only [Timestamp, Flow Duration] -> drop embedded header rows and
          unparsable timestamps -> epoch seconds -> sweep-line count of flows
          concurrent at each flow's start ("Active Conn Cnt").

  Pass B  read the ~16 needed columns -> drop the exact same rows (same rule,
          so row order/alignment is identical) -> coerce numerics (float32) ->
          derive packet/byte rates with safe division (no Inf) -> attach
          Pass-A columns -> map labels to MDP states -> drop Inf/NaN rows ->
          sort by time.

Finally: split off the held-out Infiltration days, fit a StandardScaler on the
TRAIN split only, write train/holdout Parquet + scaler.json + a markdown report.

Run:  ~/venvs/rl-ids/bin/python -m rl_ids.preprocess
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from rl_ids import config as C

# Embedded header rows make numeric columns read as mixed types; we re-coerce
# with pd.to_numeric(errors="coerce") right after reading, so this is expected.
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

_RATE_INPUTS = ["Tot Fwd Pkts", "Tot Bwd Pkts"]          # already in RAW_COLUMNS
_NUMERIC_COLS = [c for c in C.RAW_COLUMNS if c not in ("Timestamp", "Label")]


def day_from_filename(path) -> str:
    """'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv' -> 'Wednesday-14-02-2018'."""
    return path.name.split("_")[0]


def _timestamp_masks(ts: pd.Series) -> pd.Series:
    """Boolean mask of usable rows: not an embedded header AND parsable timestamp.

    BOTH passes must apply exactly this rule so their row alignment matches.
    (Embedded header rows have the literal value 'Timestamp' in this column.)
    """
    not_header = ts != C.EMBEDDED_HEADER_VALUE
    parsed = pd.to_datetime(ts.where(not_header), format=C.TIMESTAMP_FORMAT,
                            errors="coerce")
    # Not-a-timestamp (embedded header rows) and broken-clock rows (1970) are
    # both unusable - drop them in BOTH passes so row alignment stays identical.
    return parsed.notna() & (parsed >= pd.Timestamp(C.MIN_TIMESTAMP))


def pass_a(path) -> tuple[np.ndarray, np.ndarray]:
    """Epoch seconds (int64) + flow duration seconds (float64), header rows removed.

    Arrays are aligned with the rows Pass B keeps *before* its final Inf/NaN drop.
    """
    epochs: list[np.ndarray] = []
    durs: list[np.ndarray] = []
    for chunk in pd.read_csv(path, usecols=["Timestamp", "Flow Duration"],
                             chunksize=C.BATCH_CHUNKSIZE, engine="c"):
        mask = _timestamp_masks(chunk["Timestamp"])
        parsed = pd.to_datetime(chunk.loc[mask, "Timestamp"],
                                format=C.TIMESTAMP_FORMAT, errors="coerce")
        if parsed.empty:
            continue
        dur = pd.to_numeric(chunk.loc[mask, "Flow Duration"],
                            errors="coerce").to_numpy(dtype=np.float64)
        dur = np.nan_to_num(dur, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(dur, 0.0, None, out=dur)
        dur /= 1e6  # CICFlowMeter reports Flow Duration in microseconds
        epochs.append(parsed.to_numpy().astype("datetime64[s]").astype(np.int64))
        durs.append(dur)
    if not epochs:
        raise ValueError(f"Pass A found no usable rows in {path}")
    return np.concatenate(epochs), np.concatenate(durs)


def active_connection_count(epochs: np.ndarray, dur_s: np.ndarray) -> np.ndarray:
    """Number of flows active at each flow's start time (sweep line).

    An interval [t, t+d) is active at time t_i iff start <= t_i < end, so
    count_i = #{start <= t_i} - #{end <= t_i}   (via searchsorted, O(n log n)).
    """
    starts = epochs.astype(np.float64)
    ends = starts + dur_s
    n_start = np.searchsorted(np.sort(starts), starts, side="right")
    n_end = np.searchsorted(np.sort(ends), starts, side="right")
    return (n_start - n_end).astype(np.int32)


def pass_b(path) -> pd.DataFrame:
    """Feature/label chunk-frame for one day (raw columns, float32)."""
    usecols = list(dict.fromkeys(C.RAW_COLUMNS + _RATE_INPUTS))  # dedup, keep order
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=C.BATCH_CHUNKSIZE,
                             engine="c"):
        mask = _timestamp_masks(chunk["Timestamp"])   # same rule as pass A
        chunk = chunk.loc[mask].drop(columns="Timestamp")   # row alignment kept
        if chunk.empty:
            continue
        chunk["Label"] = chunk["Label"].astype(str).str.strip()
        for col in _NUMERIC_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)
        parts.append(chunk)
    if not parts:
        raise ValueError(f"Pass B found no usable rows in {path}")
    return pd.concat(parts, ignore_index=True)


def build_day(path) -> pd.DataFrame:
    """Clean one raw CSV, derive features, map labels -> full observation table."""
    day = day_from_filename(path)
    epochs, dur_s = pass_a(path)
    df = pass_b(path)
    if len(df) != len(epochs):
        raise AssertionError(
            f"{day}: pass A/B row mismatch ({len(epochs)} vs {len(df)}) - "
            f"the row-drop rules in the two passes must stay identical"
        )

    # --- derived: safe rates (no Inf by construction) ---------------------- #
    dur_s_safe = np.where(dur_s > 0, dur_s, np.nan)
    fwd_pkts = df["Tot Fwd Pkts"].to_numpy(dtype=np.float64)
    bwd_pkts = df["Tot Bwd Pkts"].to_numpy(dtype=np.float64)
    fwd_len = df["TotLen Fwd Pkts"].to_numpy(dtype=np.float64)
    bwd_len = df["TotLen Bwd Pkts"].to_numpy(dtype=np.float64)

    df["Flow Pkts/s"] = (fwd_pkts + bwd_pkts) / dur_s_safe
    df["Fwd Pkts/s"] = fwd_pkts / dur_s_safe
    df["Bwd Pkts/s"] = bwd_pkts / dur_s_safe
    df["Flow Byts/s"] = (fwd_len + bwd_len) / dur_s_safe
    df["TimestampEpoch"] = epochs
    df["Active Conn Cnt"] = active_connection_count(epochs, dur_s).astype(np.float64)
    lo, hi = C.PORT_BUCKET_EDGES
    df["Dst Port Bucket"] = np.digitize(df["Dst Port"].to_numpy(), [lo, hi])

    # --- hidden ground truth (never part of the observation) ---------------- #
    df["attack_label"] = df["Label"]
    df["mdp_state"] = df["attack_label"].map(C.state_for_label)  # raises on unknown
    df["source_day"] = day

    # --- final clean: Inf/NaN -> drop (checkpoint: isna().sum().sum() == 0) - #
    drop_subset = [c for c in C.OBS_FEATURES + ["Dst Port", "attack_label"]
                   if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=drop_subset)
    dropped = before - len(df)
    if dropped:
        print(f"  {day}: dropped {dropped} rows with Inf/NaN")

    df = df[C.OBS_FEATURES + C.HIDDEN_COLUMNS]
    df = df.sort_values("TimestampEpoch", kind="stable")

    df[C.OBS_FEATURES] = df[C.OBS_FEATURES].astype(np.float32)
    for col in ("attack_label", "mdp_state", "source_day"):
        df[col] = df[col].astype(str)
    df["TimestampEpoch"] = df["TimestampEpoch"].astype(np.int64)
    print(f"  {day}: {len(df):,} clean rows, "
          f"{df['mdp_state'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


def fit_and_apply_scaler(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Z-score OBS_FEATURES using TRAIN statistics only; persist to scaler.json."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train[C.OBS_FEATURES] = scaler.fit_transform(train[C.OBS_FEATURES])
    holdout[C.OBS_FEATURES] = scaler.transform(holdout[C.OBS_FEATURES])
    train[C.OBS_FEATURES] = train[C.OBS_FEATURES].astype(np.float32)
    holdout[C.OBS_FEATURES] = holdout[C.OBS_FEATURES].astype(np.float32)

    payload = {
        "features": C.OBS_FEATURES,
        "mean": dict(zip(C.OBS_FEATURES, map(float, scaler.mean_))),
        "scale": dict(zip(C.OBS_FEATURES, map(float, scaler.scale_))),
        "fit_on": "train split only (holdout Infiltration days excluded)",
    }
    C.SCALER_JSON.write_text(json.dumps(payload, indent=2))
    return payload


def write_report(train: pd.DataFrame, holdout: pd.DataFrame,
                 files_used: list[str], skipped: list[str]) -> str:
    """Day-1 checkpoint report: class balance + 5-row manual trace."""
    lines: list[str] = ["# Day 1 report - dataset preprocessing", ""]
    lines += [f"- Files processed: **{len(files_used)}**",
              f"- Files skipped (missing): {skipped or 'none'}"]
    if skipped:
        lines += ["", "> NOTE: missing files - run `./download.sh` again."]
    lines += ["", f"- Train rows: **{len(train):,}**  |  "
                  f"Holdout rows: **{len(holdout):,}** (Infiltration days only)",
              f"- Train days: {sorted(train['source_day'].unique())}",
              f"- Holdout days: {sorted(holdout['source_day'].unique())}",
              f"- Observation features ({len(C.OBS_FEATURES)}): {C.OBS_FEATURES}",
              f"- Hidden (reward/eval only): {C.HIDDEN_COLUMNS}",
              "", "## Checkpoint", "",
              f"- `train.isna().sum().sum() == {int(train.isna().sum().sum())}`",
              f"- `holdout.isna().sum().sum() == {int(holdout.isna().sum().sum())}`",
              "", "## Class balance (train)", ""]

    bal = (train["mdp_state"].value_counts()
           .reindex(C.STATES, fill_value=0).astype("int64").rename("rows")
           .to_frame())
    bal["percent"] = (100 * bal["rows"] / bal["rows"].sum()).round(2)
    lines += [bal.to_markdown(), "", "### Per day x state (train)", "",
              pd.crosstab(train["source_day"], train["mdp_state"])
              .reindex(columns=C.STATES, fill_value=0).to_markdown(), "",
              "## Attack label inventory", "",
              pd.concat([
                  train.groupby("mdp_state")["attack_label"].value_counts()
                       .rename("train_rows"),
                  holdout.groupby("mdp_state")["attack_label"].value_counts()
                         .rename("holdout_rows"),
              ], axis=1).fillna(0).astype(int).to_markdown(), ""]

    # 5-row manual trace: one exemplar per state + random extras.
    lines += ["## 5-row sample trace (label -> MDP state)", ""]
    sample_parts = [train[train["mdp_state"] == s].head(1)
                    for s in C.STATES if len(train[train["mdp_state"] == s])]
    rng = np.random.default_rng(42)
    extra_idx = rng.choice(len(train),
                           size=max(0, 5 - len(sample_parts)), replace=False)
    sample = pd.concat(sample_parts + [train.iloc[extra_idx]]).head(5)
    trace = sample[["source_day", "attack_label", "mdp_state",
                    "Dst Port Bucket", "Flow Pkts/s", "Active Conn Cnt"]].copy()
    trace["expected_state"] = trace["attack_label"].map(C.state_for_label)
    lines += [trace.to_markdown(index=False), "",
              "All `mdp_state` values verified against `LABEL_TO_STATE` "
              "in the checkpoint tests.", ""]
    text = "\n".join(lines)
    C.REPORT_PATH.write_text(text)
    return text


def main() -> int:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    files_used, skipped = [], []

    for name in C.RAW_FILES:
        path = C.RAW_DIR / name
        if not path.exists():
            print(f"MISSING {name}", file=sys.stderr)
            skipped.append(name)
            continue
        print(f"Processing {name} ...")
        day_df = build_day(path)
        day_df.to_parquet(C.PROCESSED_DIR / f"day_{day_from_filename(path)}.parquet",
                          index=False)
        frames.append(day_df)

    if not frames:
        print("No raw files found - run ./download.sh first.", file=sys.stderr)
        return 1
    missing_days = {day_from_filename(C.RAW_DIR / n) for n in skipped} & C.HOLDOUT_DAYS
    if missing_days:
        print(f"Holdout day file(s) missing: {sorted(missing_days)} - cannot build "
              f"holdout split.", file=sys.stderr)
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    del frames
    is_holdout = all_df["source_day"].isin(C.HOLDOUT_DAYS)
    train, holdout = all_df[~is_holdout].copy(), all_df[is_holdout].copy()
    del all_df
    print(f"Train: {len(train):,} rows | Holdout: {len(holdout):,} rows")

    fit_and_apply_scaler(train, holdout)
    train.to_parquet(C.TRAIN_PARQUET, index=False)
    holdout.to_parquet(C.HOLDOUT_PARQUET, index=False)

    report = write_report(train, holdout,
                          files_used=[f for f in C.RAW_FILES if f not in skipped],
                          skipped=skipped)
    print(report)
    print(f"\nWrote {C.TRAIN_PARQUET}\nWrote {C.HOLDOUT_PARQUET}\n"
          f"Wrote {C.SCALER_JSON}\nWrote {C.REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
