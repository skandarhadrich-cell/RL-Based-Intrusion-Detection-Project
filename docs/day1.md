# Day 1 — Dataset acquisition & preprocessing

**Goal:** a clean, labeled feature table (zero NaNs/Infs) ready to drive the Gymnasium environment (Day 2+).

**Status: DONE** — all 10 checkpoint tests pass on the full 10-day dataset.

---

## What was done

1. **Acquisition** — downloaded all 10 CSE-CIC-IDS2018 CICFlowMeter CSVs (~6.9 GB) from AWS S3 with
   a resume-safe script (`download.sh`). Tuesday-20-02 is the 84-column variant (extra
   `Flow ID / Src IP / Dst IP / Src Port`); Wednesday-14-02 etc. are the 80-column variant.
   Reading by column *name* (`usecols`) makes the schema difference a non-issue.

2. **Cleaning** — per file, chunked (250k rows) …
   - dropped **embedded mid-file header rows** (`Timestamp == 'Timestamp'` sentinel),
   - dropped **broken-clock 1970 timestamps** (`Timestamp < 2010-01-01`),
   - dropped rows with **Inf/NaN** after numeric coercion,
   - µs → s conversion for `Flow Duration` (in a first pass this was off by 10⁶, which
     wrecked byte/packet rates and `Active Conn Cnt` — fixed and locked by tests),
   - **recomputed** `Flow Byts/s`, `Flow Pkts/s`, `Fwd/Bwd Pkts/s` with safe division
     (no `Inf` from duration=0), and rebuilt `Active Conn Cnt` with a sweep-line count of
     concurrent flows at each flow's start.

3. **Label → MDP state mapping** (`config.LABEL_TO_STATE`):
   | Dataset label                                        | MDP state |
   |------------------------------------------------------|-----------|
   | `Benign`                                             | Normal    |
   | FTP/SSH-Bruteforce, PortScan*                        | Suspicious |
   | DoS\*, DDoS\*, Bot, Infiltration, Web/Brute-Force/XSS/SQLi, Heartbleed | Compromised |

   \* No `PortScan` rows exist in this export (verified 0 matches in the raw files);
   the key is kept as a defensive guard. Normalization falls back on a `" attack(s)"`
   infix strip, so `DoS attacks-GoldenEye` == `DoS-GoldenEye`; the dataset typo
   `Infilteration` is mapped too. Unmapped labels raise `KeyError` (fail fast).

4. **Feature reduction** — 80 CICFlowMeter columns → **17 observation features**
   (packet rates, byte counts, active connection count, TCP flag histogram,
   destination-port bucket, protocol; see `config.OBS_FEATURES`).
   The hidden columns `attack_label`, `mdp_state`, `TimestampEpoch`, `source_day`
   are kept **only** for reward/eval — never fed to the agent.

5. **Split + scaler** — both Infiltration days (Wed-28-02, Thu-01-03) are held out as
   **eval-only**; `StandardScaler` is fit on the *train split only* and saved to
   `scaler.json` (verified: train obs are exactly z-scored).

## Results

| Split | Days | Rows |
|---|---|---|
| Train | 8 (all attack classes except Infiltration) | **15,202,086** |
| Holdout | 2 (Infiltration only) | **935,083** |

**Class balance (train):**

| mdp_state   | rows       | %     |
|-------------|-----------:|------:|
| Normal      | 12,615,791 | 82.99 |
| Suspicious  |    380,943 |  2.51 |
| Compromised |  2,205,352 | 14.51 |
| **TOTAL**   | 15,202,086 |       |

Per-day × state (train): Wed-14-02 `N 663,803 / S 380,943` · Thu-15-02
`N 988,050 / C 52,498` · Fri-16-02 `N 446,772 / C 601,802` · Tue-20-02
`N 7,313,104 / C 576,191` · Wed-21-02 `N 360,833 / C 687,742` · Thu-22-02
`N 1,042,594 / C 362` · Fri-23-02 `N 1,042,301 / C 566` · Fri-02-03
`N 758,334 / C 286,191`.

Suspicious is purely FTP/SSH-Bruteforce (Wed-14-02). Compromised is dominated by
DDoS-HOIC (686 k), DDoS-LOIC-HTTP (576 k) and DoS-Hulk (462 k); Brute-Force-Web/XSS
and SQLi rows are few (611/230/87) but present.

## Checkpoints (all passing)

- `train.isna().sum().sum() == 0`, `holdout.isna().sum().sum() == 0`, no Inf anywhere.
- Class balance printed (above).
- 5-row sample trace: one exemplar per state + 2 random rows all map
  label → expected MDP state (`mdp_state == state_for_label(attack_label)`).

## Bugs found & fixed

| Bug | Impact | Fix |
|---|---|---|
| `Flow Duration` treated as µs | rates & `Active Conn Cnt` off by up to 10⁶ | convert to seconds, tests lock it |
| Pass A / Pass B row alignment | features attached to wrong flows | identical filter rule + chunk ordering + assert row counts equal |
| Break/epoch mixing | 1970 clock rows | `MIN_TIMESTAMP = 2010-01-01` lower bound |
| `Inf` from `duration == 0` | NaN/Inf in rates | safe division (`np.divide(..., where=...)`) + drop remainder |
| Label variants & typo | unmapped rows / crashes | normalization fallback + `KeyError` guard rail |

## Artifacts

- `data/raw/*.csv` (10 files, ~6.9 GB) — raw CICFlowMeter exports
- `data/processed/train.parquet` (15.2 M rows), `holdout_infiltration.parquet` (935 k)
- `data/processed/scaler.json` — z-score params fit on train only
- `data/processed/day1_report.md` — auto-generated pipeline report
- `src/rl_ids/{config.py, preprocess.py}` · `tests/test_day1.py` · `download.sh`

## How to run

```bash
~/venvs/rl-ids/bin/python -m rl_ids.preprocess     # rebuild everything
~/venvs/rl-ids/bin/python -m pytest tests/test_day1.py -v
```

---
*Next: Day 2 — Gymnasium environment (`docs/day2.md`).*