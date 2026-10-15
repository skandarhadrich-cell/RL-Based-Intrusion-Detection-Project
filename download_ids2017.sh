#!/usr/bin/env bash
# Fetch the CIC-IDS2017 archives and unpack them under
# data/raw/ids2017/.
#
# Two archives are needed:
#   * MachineLearningCSV.zip    - the feature-only CICFlowMeter export (79 cols,
#                                 NO Timestamp). This is the "clean" reference.
#   * GeneratedLabelledFlows.zip- the full labelled-flow export (a.k.a.
#                                 "TrafficLabelling"): adds Flow ID / IPs / ports
#                                 AND **Timestamp**, so the derived concurrency
#                                 feature (Active Conn Cnt) can be recomputed
#                                 with the exact same code path as 2018.
#
# Provenance: the official UNB distribution (http://cicresearch.ca/CICDataset/
# CIC-IDS-2017/) no longer serves files directly (it 302s to the UNB index).
# We use the Hugging Face mirror bencorn/CICIDS2017 - its README points back to
# the UNB original + paper, and x-linked-size matches the API listing exactly.
#
# Resume-safe: per-file size check, byte-range resume, retries, sha256 record.
set -euo pipefail

BASE="https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs"
DIR="/home/lali/Rl-ids-Agent/data/raw/ids2017"
mkdir -p "$DIR"
cd "$DIR"

# name:expected_bytes  (byte sizes reported by the Hugging Face API listing)
ARCHIVES=(
  "MachineLearningCSV.zip:235102953"
  "GeneratedLabelledFlows.zip:283876488"
)

for spec in "${ARCHIVES[@]}"; do
  zip="${spec%%:*}"
  expected="${spec##*:}"

  # ---- 1. download (skip if complete, resume if partial) ------------------ #
  if [[ -f "$zip" && "$(stat -c%s "$zip")" == "$expected" ]]; then
    echo "SKIP  $zip (complete: $expected bytes)"
  else
    for attempt in $(seq 1 300); do
      have=0
      [[ -f "$zip" ]] && have="$(stat -c%s "$zip")"
      if (( have > expected )); then
        echo "TRUNCATING oversize partial ($have > $expected)"
        truncate -s "$expected" "$zip"
        have=$expected
      fi
      echo "GET   $zip (attempt $attempt, have $have/$expected)"
      curl -L --fail --retry 5 --connect-timeout 20 -C "$have" \
           -o "$zip" "$BASE/$zip" || true
      [[ -f "$zip" && "$(stat -c%s "$zip")" == "$expected" ]] && break
      sleep 3
    done
  fi

  if [[ ! -f "$zip" || "$(stat -c%s "$zip")" != "$expected" ]]; then
    echo "FAIL  $zip size is $(stat -c%s "$zip" 2>/dev/null || echo 0), expected $expected" >&2
    exit 1
  fi

  # ---- 2. integrity record (re-verified on every subsequent run) ---------- #
  if [[ -f "$zip.sha256" ]]; then
    echo "CHECK $(sha256sum -c "$zip.sha256")"
  else
    sha256sum "$zip" > "$zip.sha256"
    echo "WROTE $zip.sha256 ($(cut -d' ' -f1 "$zip.sha256"))"
  fi

  # ---- 3. unpack (idempotent via marker) ---------------------------------- #
  marker="$DIR/.extracted_${zip%.zip}"
  if [[ ! -f "$marker" ]]; then
    echo "UNZIP $zip"
    unzip -o -q "$zip" -d "$DIR"
    touch "$marker"
  fi
done

echo "--- unpacked CSVs ---"
find "$DIR" -name '*.csv' -printf '%10s  %p\n' | sort -k2
