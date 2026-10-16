"""Dataset acquisition cleanup -> labeled, normalized Parquet tables.

Now multi-source: CSE-CIC-IDS2018 **and** CIC-IDS2017 are
processed through the *same* code path, described entirely by
``config.SourceSpec`` objects.  Nothing downstream needs to know a feature name
differs between the two datasets.

Pipeline (per raw CSV, chunked to stay memory-safe on small VMs):

  Pass A  read only [Timestamp, Flow Duration] -> drop embedded header rows and
          unparsable timestamps -> epoch seconds -> sweep-line count of flows
          concurrent at each flow's start ("Active Conn Cnt").

  Pass B  read the ~16 needed columns -> drop the exact same rows (same rule,
          so row order/alignment is identical) -> coerce numerics (float32) ->
          derive packet/byte rates with safe division (no Inf) -> attach
          Pass-A columns -> map labels to MDP states -> drop exact duplicate
          rows (quality control) -> drop Inf/NaN rows -> sort by time.

Finally: split off each source's held-out day(s) (IDS2018 Infiltration days;
IDS2017 PortScan + DDoS capture files), fit ONE StandardScaler on the merged
TRAIN split only, write train/holdout Parquet + scaler.json + a markdown report.

Run:  PYTHONPATH=src ~/venvs/rl-ids/bin/python -m rl_ids.preprocess
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


# --------------------------------------------------------------------------- #
# Source adapter: read any raw CSV through its SourceSpec
# --------------------------------------------------------------------------- #
def _read_chunks(path, spec: C.SourceSpec, wanted_canonical: list[str]):
    """Yield chunks with canonical column names, reading only wanted columns.

    ``usecols`` is a predicate over the *raw* headers, so leading-space /
    differently-named columns (e.g. ' Destination Port', 'Total Fwd Packets')
    are matched via the spec's canonical mapping instead of assumed identical.
    """
    wanted = set(wanted_canonical)
    reader = pd.read_csv(
        path,
        usecols=lambda col: spec.canonical(col) in wanted,
        chunksize=C.BATCH_CHUNKSIZE,
        engine="c",
        encoding=spec.encoding,
    )
    for chunk in reader:
        chunk = chunk.rename(columns={c: spec.canonical(c) for c in chunk.columns})
        yield chunk


def _parse_timestamps(ts: pd.Series, spec: C.SourceSpec) -> pd.Series:
    """Parse a timestamp column, trying the spec's format list in order.

    CIC-IDS2017 mixes minute and second resolution across files, so rows that
    fail the primary format are retried with each alternate format.
    """
    parsed = pd.to_datetime(ts, format=spec.timestamp_format, errors="coerce")
    for fmt in spec.alternate_timestamp_formats:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed = parsed.where(
            ~missing,
            pd.to_datetime(ts.where(missing), format=fmt, errors="coerce"),
        )
    return parsed


def _timestamp_masks(ts: pd.Series, spec: C.SourceSpec) -> pd.Series:
    """Boolean mask of usable rows: not an embedded header AND parsable timestamp.

    BOTH passes must apply exactly this rule so their row alignment matches.
    """
    if spec.embedded_header_value is None:
        not_header = pd.Series(True, index=ts.index)
    else:
        not_header = ts != spec.embedded_header_value
    parsed = _parse_timestamps(ts.where(not_header), spec)
    # Not-a-timestamp (embedded header rows, empty/trailing junk rows) and
    # broken-clock rows (1970) are both unusable - drop them in BOTH passes so
    # row alignment stays identical.
    return parsed.notna() & (parsed >= pd.Timestamp(spec.min_timestamp))


def pass_a(path, spec: C.SourceSpec) -> tuple[np.ndarray, np.ndarray]:
    """Epoch seconds (int64) + flow duration seconds (float64), header rows removed.

    Arrays are aligned with the rows Pass B keeps *before* its final Inf/NaN drop.
    """
    epochs: list[np.ndarray] = []
    durs: list[np.ndarray] = []
    for chunk in _read_chunks(path, spec, ["Timestamp", "Flow Duration"]):
        mask = _timestamp_masks(chunk["Timestamp"], spec)
        parsed = _parse_timestamps(chunk.loc[mask, "Timestamp"], spec)
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


def pass_b(path, spec: C.SourceSpec) -> pd.DataFrame:
    """Feature/label chunk-frame for one day (raw columns, float32)."""
    usecols = list(dict.fromkeys(C.RAW_COLUMNS + _RATE_INPUTS))  # dedup, keep order
    parts: list[pd.DataFrame] = []
    for chunk in _read_chunks(path, spec, usecols):
        mask = _timestamp_masks(chunk["Timestamp"], spec)   # same rule as pass A
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


def build_day(path, spec: C.SourceSpec) -> tuple[pd.DataFrame, dict]:
    """Clean one raw CSV, derive features, map labels -> full observation table."""
    day = spec.day_of(path.name)
    epochs, dur_s = pass_a(path, spec)
    df = pass_b(path, spec)
    if len(df) != len(epochs):
        raise AssertionError(
            f"{spec.name}/{day}: pass A/B row mismatch ({len(epochs)} vs {len(df)})"
            f" - the row-drop rules in the two passes must stay identical"
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
    df[C.SOURCE_DATASET_COLUMN] = spec.name

    # --- final clean: Inf/NaN -> drop (checkpoint: isna().sum().sum() == 0) - #
    drop_subset = [c for c in C.OBS_FEATURES + ["Dst Port", "attack_label"]
                   if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=drop_subset)
    infnan_dropped = before - len(df)
    if infnan_dropped:
        print(f"  {spec.name}/{day}: dropped {infnan_dropped} rows with Inf/NaN")

    df = df[C.OBS_FEATURES + C.HIDDEN_COLUMNS]

    # --- exact duplicate rows within the file (quality control) ------------- #
    # Identical raw flows produce identical derived rows, so this removes true
    # duplicate records (a known ~9% issue in the CIC-IDS2017 exports).
    before = len(df)
    df = df.drop_duplicates()
    dup_dropped = before - len(df)

    df = df.sort_values("TimestampEpoch", kind="stable")
    df[C.OBS_FEATURES] = df[C.OBS_FEATURES].astype(np.float32)
    for col in ("attack_label", "mdp_state", "source_day", C.SOURCE_DATASET_COLUMN):
        df[col] = df[col].astype(str)
    df["TimestampEpoch"] = df["TimestampEpoch"].astype(np.int64)

    info = {
        "source": spec.name,
        "day": day,
        "rows": len(df),
        "dup_dropped": dup_dropped,
        "infnan_dropped": infnan_dropped,
        "is_holdout": day in spec.holdout_days,
        "states": df["mdp_state"].value_counts().to_dict(),
    }
    print(f"  {spec.name}/{day}: {len(df):,} clean rows "
          f"({dup_dropped:,} duplicate rows dropped), {info['states']}")
    return df.reset_index(drop=True), info


# --------------------------------------------------------------------------- #
# Scaling + reporting
# --------------------------------------------------------------------------- #
def fit_and_apply_scaler(train: pd.DataFrame, holdouts: list[pd.DataFrame]) -> dict:
    """Z-score OBS_FEATURES using the merged TRAIN statistics only; persist."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train[C.OBS_FEATURES] = scaler.fit_transform(train[C.OBS_FEATURES])
    for holdout in holdouts:
        holdout[C.OBS_FEATURES] = scaler.transform(holdout[C.OBS_FEATURES])
    train[C.OBS_FEATURES] = train[C.OBS_FEATURES].astype(np.float32)
    for holdout in holdouts:
        holdout[C.OBS_FEATURES] = holdout[C.OBS_FEATURES].astype(np.float32)

    payload = {
        "features": C.OBS_FEATURES,
        "mean": dict(zip(C.OBS_FEATURES, map(float, scaler.mean_))),
        "scale": dict(zip(C.OBS_FEATURES, map(float, scaler.scale_))),
        "fit_on": "merged train split only (ids2018 + ids2017; all holdout days "
                  "excluded)",
        "sources": sorted(train[C.SOURCE_DATASET_COLUMN].unique().tolist()),
    }
    C.SCALER_JSON.write_text(json.dumps(payload, indent=2))
    return payload


def _per_source_balance(train: pd.DataFrame) -> str:
    ct = (pd.crosstab(train[C.SOURCE_DATASET_COLUMN], train["mdp_state"])
          .reindex(columns=list(C.STATES), fill_value=0))
    ct["TOTAL"] = ct.sum(axis=1)
    pct = (100 * ct[list(C.STATES)].div(ct["TOTAL"], axis=0)).round(2).add_suffix(" %")
    return pd.concat([ct, pct], axis=1).to_markdown()


def write_report(train: pd.DataFrame, holdouts: dict[str, pd.DataFrame],
                 run_stats: list[dict], scaler_payload: dict) -> str:
    """Preprocessing report: class balance, provenance, duplicates, trace."""
    lines: list[str] = [
        "# Dataset preprocessing Report",
        "",
        f"- Files processed: **{len(run_stats)}**",
        "",
        "## Sources",
        "",
        f"- Merged train rows: **{len(train):,}**",
        f"- Train sources: {sorted(train[C.SOURCE_DATASET_COLUMN].unique())}",
        f"- Observation features ({len(C.OBS_FEATURES)}): {C.OBS_FEATURES}",
        f"- Hidden (reward/eval only): {C.HIDDEN_COLUMNS}",
        f"- Scaler fit on: {scaler_payload['fit_on']}",
        "",
        "## Holdout splits (eval only, never trained on)",
        "",
    ]
    for name, ho in holdouts.items():
        lines.append(f"- `{name}`: **{len(ho):,}** rows, days "
                     f"{sorted(ho['source_day'].unique())}")
    lines += ["", "## Per-file processing (rows kept / duplicate rows dropped)", "",
              "| source | file/day | rows kept | dup dropped | Inf/NaN dropped | holdout |",
              "|---|---|---:|---:|---:|:--:|"]
    for s in run_stats:
        lines.append(f"| {s['source']} | {s['day']} | {s['rows']:,} | "
                     f"{s['dup_dropped']:,} | {s['infnan_dropped']:,} | "
                     f"{'yes' if s['is_holdout'] else ''} |")
    lines += ["",
              f"- Total duplicate rows dropped: "
              f"**{sum(s['dup_dropped'] for s in run_stats):,}**", ""]

    lines += ["", "## Checkpoint", "",
              f"- `train.isna().sum().sum() == {int(train.isna().sum().sum())}`"]
    for name, ho in holdouts.items():
        lines.append(f"- `{name}.isna().sum().sum() == {int(ho.isna().sum().sum())}`")
    lines += ["", "## Class balance (merged train)", ""]
    bal = (train["mdp_state"].value_counts()
           .reindex(C.STATES, fill_value=0).astype("int64").rename("rows")
           .to_frame())
    bal["percent"] = (100 * bal["rows"] / bal["rows"].sum()).round(2)
    lines += [bal.to_markdown(), "", "### Per-source class balance (train)", "",
              _per_source_balance(train), "",
              "### Per day x state (train)", "",
              pd.crosstab(train["source_day"], train["mdp_state"])
              .reindex(columns=C.STATES, fill_value=0).to_markdown(), "",
              "## Attack label inventory (train)", "",
              train.groupby([C.SOURCE_DATASET_COLUMN, "mdp_state"])["attack_label"]
                   .value_counts().rename("rows").to_markdown(), ""]

    # 5-row manual trace: one exemplar per state + random extras.
    lines += ["## 5-row sample trace (label -> MDP state)", ""]
    sample_parts = [train[train["mdp_state"] == s].head(1)
                    for s in C.STATES if len(train[train["mdp_state"] == s])]
    rng = np.random.default_rng(42)
    extra_idx = rng.choice(len(train),
                           size=max(0, 5 - len(sample_parts)), replace=False)
    sample = pd.concat(sample_parts + [train.iloc[extra_idx]]).head(5)
    trace = sample[[C.SOURCE_DATASET_COLUMN, "source_day", "attack_label", "mdp_state",
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
    run_stats: list[dict] = []

    for spec in C.SOURCES:
        if not spec.raw_dir.exists():
            print(f"MISSING source dir {spec.raw_dir} - run ./download_ids2017.sh"
                  if spec.name == C.SOURCE_2017 else
                  f"MISSING source dir {spec.raw_dir} - run ./download_ids2018.sh",
                  file=sys.stderr)
            return 1
        day_done: set[str] = set()
        for name in spec.files:
            path = spec.raw_dir / name
            if not path.exists():
                print(f"MISSING [{spec.name}] {name}", file=sys.stderr)
                continue
            print(f"Processing [{spec.name}] {name} ...")
            day_df, info = build_day(path, spec)
            day_df.to_parquet(
                C.PROCESSED_DIR / f"day_{spec.name}_{info['day']}.parquet",
                index=False,
            )
            frames.append(day_df)
            run_stats.append(info)
            day_done.add(info["day"])
        missing_holdout = set(spec.holdout_days) - day_done
        if missing_holdout:
            print(f"[{spec.name}] holdout day file(s) missing: "
                  f"{sorted(missing_holdout)} - cannot build holdout split.",
                  file=sys.stderr)
            return 1

    if not frames:
        print("No raw files found.", file=sys.stderr)
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    del frames
    is_holdout = pd.Series(False, index=all_df.index)
    for spec in C.SOURCES:
        is_holdout |= ((all_df[C.SOURCE_DATASET_COLUMN] == spec.name)
                       & all_df["source_day"].isin(spec.holdout_days))
    train = all_df[~is_holdout].copy()
    holdout_2018 = all_df[is_holdout & (all_df[C.SOURCE_DATASET_COLUMN] == C.SOURCE_2018)].copy()
    holdout_2017 = all_df[is_holdout & (all_df[C.SOURCE_DATASET_COLUMN] == C.SOURCE_2017)].copy()
    del all_df
    print(f"Train: {len(train):,} rows | "
          f"IDS2018 holdout: {len(holdout_2018):,} rows | "
          f"IDS2017 holdout: {len(holdout_2017):,} rows")

    scaler_payload = fit_and_apply_scaler(train, [holdout_2018, holdout_2017])
    train.to_parquet(C.TRAIN_PARQUET, index=False)
    holdout_2018.to_parquet(C.HOLDOUT_PARQUET, index=False)
    holdout_2017.to_parquet(C.HOLDOUT_IDS2017_PARQUET, index=False)

    report = write_report(
        train,
        {"holdout_infiltration (ids2018)": holdout_2018,
         "holdout_ids2017_portscan_ddos (ids2017)": holdout_2017},
        run_stats, scaler_payload,
    )
    print(report)
    print(f"\nWrote {C.TRAIN_PARQUET}\nWrote {C.HOLDOUT_PARQUET}\n"
          f"Wrote {C.HOLDOUT_IDS2017_PARQUET}\nWrote {C.SCALER_JSON}\n"
          f"Wrote {C.REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
