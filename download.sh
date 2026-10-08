#!/usr/bin/env bash
# Day 1: fetch all 10 processed CSE-CIC-IDS2018 CSVs.
# Resume-safe: per-file size check, byte-range resume, outer retry loop.
set -u
BASE="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
DIR="/home/lali/Rl-ids-Agent/data/raw"
FILES=(
  "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
  "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
  "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
  "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
)
mkdir -p "$DIR"
for f in "${FILES[@]}"; do
  out="$DIR/$f"
  expected=$(curl -sI "$BASE/$f" | tr -d '\r' | awk '/^[Cc]ontent-[Ll]ength:/{print $2}')
  if [[ -f "$out" && -n "$expected" && "$(stat -c%s "$out")" == "$expected" ]]; then
    echo "SKIP  $f (complete)"
    continue
  fi
  echo "GET   $f (expected $expected bytes)"
  ok=0
  for attempt in $(seq 1 100); do
    if [[ -n "$expected" && -f "$out" ]] && [[ "$(stat -c%s "$out")" -gt "$expected" ]]; then
      rm -f "$out"   # corrupt/oversized partial - start over
    fi
    args=(-sS --retry 10 --retry-delay 5 --retry-all-errors)
    if [[ -f "$out" && "$(stat -c%s "$out")" -gt 0 ]]; then
      args+=(-C -)
    fi
    curl "${args[@]}" -o "$out" "$BASE/$f"
    rc=$?
    actual=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ -n "$expected" && "$actual" == "$expected" ]]; then
      ok=1
      break
    fi
    echo "  RETRY $f attempt=$attempt rc=$rc size=$actual/$expected"
    sleep 5
  done
  if [[ $ok == 1 ]]; then
    echo "OK    $f ($actual bytes)"
  else
    echo "FAIL  $f after 100 attempts"
    exit 1
  fi
done
echo "ALL_DOWNLOADS_COMPLETE"
