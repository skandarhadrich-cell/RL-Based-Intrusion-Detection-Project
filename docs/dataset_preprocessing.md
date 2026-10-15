# Dataset acquisition & preprocessing

**Goal:** a clean, labeled feature table (zero NaNs/Infs) ready to drive the Gymnasium environment.

**Status: DONE** — all 10 checkpoint tests pass on the full 10-day dataset.

> **Extension (this revision):** CIC-IDS2017 is integrated as a **second training
> source** (controlled data-source experiment). See
> [CIC-IDS2017 integration](#cic-ids2017-integration).
> The 2018-only pipeline above is preserved byte-for-byte as the A/B baseline
> (`train_ids2018_only.parquet`, `scaler_ids2018_baseline.json`).

---

## What was done

1. **Acquisition** — downloaded all 10 CSE-CIC-IDS2018 CICFlowMeter CSVs (~6.9 GB) from AWS S3 with
   a resume-safe script (`download_ids2018.sh`). Tuesday-20-02 is the 84-column variant (extra
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

## Results (original 2018-only pipeline)

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
- `data/processed/train.parquet` (16.7 M merged rows), `holdout_infiltration.parquet` (701 k)
- `data/processed/scaler.json` — z-score params fit on the merged train split only
- `Dataset preprocessing Report.md` (repo root) — auto-generated pipeline report
- `src/rl_ids/{config.py, preprocess.py}` · `tests/test_dataset_preprocessing.py` · `download_ids2018.sh`

## How to run

```bash
~/venvs/rl-ids/bin/python -m rl_ids.preprocess     # rebuild everything
~/venvs/rl-ids/bin/python -m pytest tests/test_dataset_preprocessing.py -v
```

---

## CIC-IDS2017 integration

**Goal:** add CIC-IDS2017 as a *second training source* without changing the
17-feature observation schema, then A/B the original 2018-only model
against the merged-data model on identical, untouched eval sets.

### 1. Why the `TrafficLabelling` archive (not `MachineLearningCVE`)

Both CIC-IDS2017 downloads were fetched (see `download_ids2017.sh`) and inspected
(`inspect_source.py --encoding latin-1` → `data/processed/source_inspection_ids2017*.md`):

| Archive | `Timestamp`? | Verdict |
|---|---|---|
| `MachineLearningCVE/` | **no** | rejected — cannot derive `Active Conn Cnt` |
| `TrafficLabelling/` | **yes** | **used** |

`Active Conn Cnt` is a *derived* feature (sweep-line count of concurrent flows at
each flow start). The MachineLearningCVE export has no timestamps, so that feature
could not be computed without fabricating it. The timestamped TrafficLabelling
export lets us run the **identical derivation code** for both datasets. Per-file
valid row counts in TrafficLabelling match MachineLearningCVE for all 8 files
(the WebAttacks file carries 288,602 trailing junk rows, auto-dropped by the
timestamp mask: valid 170,366 = MachineLearningCVE count).

### 2. Feature parity (same meaning & units, not just names)

2017 column names are mapped onto the canonical schema (`config.IDS2017_RENAMES`);
every canonical raw column is producible from a 2017 header. Parity is verified on
raw data by `tests/test_ids2017_integration.py::test_recomputed_rates_match_2017_shipped_rates`:
our recomputed `Flow Pkts/s` / `Flow Byts/s` (from `Total Fwd+Bwd Packets` /
`TotLen Fwd+Bwd` ÷ `Flow Duration` in seconds) reproduce the rate columns
CIC-IDS2017 **ships itself** with median relative error < 1% and correlation > 0.999
— confirming identical units (packets/s, bytes/s, microsecond duration) across both
datasets.

`Active Conn Cnt` is *derived* for 2017, never zero-filled: the integration test
asserts it is non-constant with a wide spread after standardization (a zero-fill
would collapse to a single constant).

### 3. Timestamp granularity caveat (known limitation)

2017 timestamps come in two formats, both handled (`alternate_timestamp_formats`):

- minute resolution `%d/%m/%Y %H:%M` (`7/7/2017 1:00`) in almost all files;
- second resolution `%d/%m/%Y %H:%M:%S` (`03/07/2017 08:55:58`) on the Monday file.

Consequence: 2017 `Active Conn Cnt` has the **same meaning and units** but a
**coarser (minute) resolution** and an ambiguous 12-hour clock. It is therefore a
noisier concurrency signal than the 2018 (second-resolution) values — documented,
not corrected.

### 4. Cleaning & de-duplication

The 2018 pipeline above originally kept duplicate rows. This revision drops
**exact duplicate rows** (uniformly across both sources) and reports counts
(`Dataset preprocessing Report.md`): **1,169,192** duplicates removed
(786,067 in 2018, 383,125 in 2017). Inf/NaN handling, the 1970-clock guard and the
rate recomputation are unchanged.

> Note: the drop applies to **every** file, holdout files included, so the 2018
> Infiltration holdout shrank 935,083 → **701,039** rows (same two days, never
> trained on, never scaler-fit). Both A/B arms are evaluated on this *same*
> holdout, so the A/B remains fair.

### 5. Splits (eval-only, never trained on, never scaler-fit)

| Split | Rows | Days |
|---|---:|---|
| `holdout_infiltration` (ids2018) | 701,039 | `Wednesday-28-02-2018`, `Thursday-01-03-2018` |
| `holdout_ids2017_portscan_ddos` (ids2017) | 346,574 | `…Afternoon-DDos` (Compromised), `…Afternoon-PortScan` (Suspicious) |

### 6. Results — merged train

| Split | Rows |
|---|---:|
| **Merged train** | **16,748,125** |
| — ids2018 | 14,650,066 |
| — ids2017 | 2,098,059 |

**Class balance (merged train):** Normal 85.68% (14.35 M) · Suspicious 1.00%
(168,001) · Compromised 13.32% (2.23 M).

**Per source:**

| source | Normal | Suspicious | Compromised | Normal % | Susp % | Comp % |
|---|---:|---:|---:|---:|---:|---:|
| ids2018 | 12,460,503 | 156,643 | 2,032,920 | 85.05 | 1.07 | 13.88 |
| ids2017 | 1,889,519 | 11,358 | 197,185 | 90.06 | 0.54 | 9.40 |

The 2017 Suspicious rows (FTP-Patator 6,330 + SSH-Patator 5,028) are the new
signal the merge adds; 2017 also contributes DoS (Hulk/GoldenEye/Slowloris/Slowhttptest),
Bot, Web-Attack, Heartbleed and 36 Infiltration rows (see the label inventory in
`Dataset preprocessing Report.md`).

### 7. Label policy & provenance

`config.LABEL_TO_STATE` now resolves the 2017 spellings (`DoS Hulk`, `DoS GoldenEye`,
`DoS slowloris`, `DoS Slowhttptest`, `FTP-Patator`, `SSH-Patator`), including labels
carrying a cp1252 en-dash byte (`Web Attack \x96 Brute Force`). The original attack
string is preserved (`attack_label`) alongside a new hidden provenance column
**`source_dataset`** (`ids2018` / `ids2017`) and `source_day`. All three are
reward/eval-only and excluded from `OBS_FEATURES`.

### 8. Scalers

- `scaler.json` — refit on the **merged train only** (both sources; all holdout days
  excluded); applied to train and both holdouts.
- `scaler_ids2018_baseline.json` — archived copy of the original 2018-only scaler,
  used by the baseline arm.

### 9. Tests

- `tests/test_dataset_preprocessing.py` — refactored `test_holdout_isolation` for two
  holdouts + `test_merged_provenance` + provenance in the leakage guard.
- `tests/test_ids2017_integration.py` — 2017 holdout content/isolation, observation-schema
  invariance, derived-vs-zero-filled `Active Conn Cnt`, canonical-column coverage,
  shipped-rate parity, and the full 2017 label→state policy.

Full suite: **88 passed**.

### How to rebuild

```bash
./download_ids2017.sh                                  # fetch + unpack both 2017 archives
~/venvs/rl-ids/bin/python -m rl_ids.inspect_source --help
PYTHONPATH=src ~/venvs/rl-ids/bin/python -m rl_ids.preprocess
~/venvs/rl-ids/bin/python -m pytest tests/ -q
```

---
*Next: the Gymnasium environment (`docs/environment.md`).*