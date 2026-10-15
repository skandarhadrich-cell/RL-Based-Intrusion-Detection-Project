"""Inspect an arbitrary raw-CSV source before integrating it.

Answers the questions that must precede any hardcoded config: what files exist,
what the raw headers really look like (leading spaces, casing), which labels
occur and how often, what the timestamp format is, how many duplicate rows /
Inf / NaN values there are.  Memory-safe (chunked) for multi-hundred-MB CSVs.

Run:
    PYTHONPATH=src ~/venvs/rl-ids/bin/python -m rl_ids.inspect_source \
        data/raw/ids2017 --name ids2017
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from rl_ids import config as C


def _find(cols: list[str], *names: str) -> str | None:
    """Case/space-insensitive column lookup (' Label' -> 'Label')."""
    norm = {c.strip().lower(): c for c in cols}
    for n in names:
        if n.strip().lower() in norm:
            return norm[n.strip().lower()]
    return None


def inspect_file(path: Path, max_rows: int | None = None,
                 encoding: str | None = None) -> dict:
    header = list(pd.read_csv(path, nrows=0, engine="c",
                              encoding=encoding).columns)
    label_col = _find(header, "Label")
    ts_col = _find(header, "Timestamp")

    rows = 0
    labels: Counter[str] = Counter()
    ts_samples: list[str] = []
    seen_hashes: set[int] = set()
    dup = 0
    nan_or_inf: Counter[str] = Counter()
    checked_numeric = 0

    for chunk in pd.read_csv(path, chunksize=C.BATCH_CHUNKSIZE, engine="c",
                             low_memory=False, encoding=encoding):
        rows += len(chunk)
        if label_col is not None:
            labels.update(chunk[label_col].astype(str).str.strip().tolist())
        if ts_col is not None and len(ts_samples) < 5:
            for v in chunk[ts_col].dropna().astype(str).head(5):
                if v not in ts_samples:
                    ts_samples.append(v)
        # exact-duplicate detection via vectorised row hashing
        for h in pd.util.hash_pandas_object(chunk, index=False).tolist():
            if h in seen_hashes:
                dup += 1
            else:
                seen_hashes.add(h)
        # Inf/NaN on a bounded sample of rows
        if checked_numeric < 200_000:
            num = chunk.select_dtypes(include=[np.number])
            for c in num.columns:
                bad = int((~np.isfinite(num[c].to_numpy(dtype="float64"))).sum())
                if bad:
                    nan_or_inf[c] += bad
            checked_numeric += len(chunk)
        if max_rows is not None and rows >= max_rows:
            break

    return {
        "file": path.name,
        "rows": rows,
        "dup_rows": dup,
        "labels": labels,
        "ts_samples": ts_samples,
        "nan_or_inf": nan_or_inf,
        "checked_numeric": checked_numeric,
        "header": header,
    }


def render(dir_path: Path, name: str, infos: list[dict]) -> str:
    lines = [f"# Source inspection - {name}", "",
             f"Directory: `{dir_path}`", f"Files: **{len(infos)}**", ""]
    total = sum(i["rows"] for i in infos)
    total_dup = sum(i["dup_rows"] for i in infos)
    lines += [f"- Total rows: **{total:,}**",
              f"- Exact duplicate rows: **{total_dup:,}** "
              f"({100 * total_dup / max(total, 1):.2f}%)", ""]

    lines += ["## Files", "", "| file | rows | dup rows | labels (top 8) |", "|---|---:|---:|---|"]
    for i in infos:
        top = ", ".join(f"{k}={v:,}" for k, v in i["labels"].most_common(8))
        lines.append(f"| {i['file']} | {i['rows']:,} | {i['dup_rows']:,} | {top} |")
    lines.append("")

    lines += ["## Header (first file, shown raw)", ""]
    lines.append("```")
    for c in infos[0]["header"]:
        lines.append(repr(c))
    lines.append("```")
    lines.append("")

    lines += ["## Timestamp samples", ""]
    for i in infos:
        lines.append(f"- `{i['file']}`: {i['ts_samples'][:3]}")
    lines.append("")

    lines += ["## Non-finite numeric cells (sampled <=200k rows/file)", ""]
    for i in infos:
        if i["nan_or_inf"]:
            lines.append(f"- `{i['file']}`: "
                         + ", ".join(f"{k}={v:,}" for k, v in i["nan_or_inf"].most_common()))
        else:
            lines.append(f"- `{i['file']}`: none")
    lines.append("")

    lines += ["## Combined label inventory", ""]
    combined: Counter[str] = Counter()
    for i in infos:
        combined.update(i["labels"])
    lines += ["| label | rows |", "|---|---:|"]
    for k, v in combined.most_common():
        lines.append(f"| {k!r} | {v:,} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m rl_ids.inspect_source <dir> [--name NAME] [--out FILE] "
              "[--encoding ENC]", file=sys.stderr)
        return 2
    dir_path = Path(argv[1])
    name = "source"
    out: Path | None = None
    encoding: str | None = None
    if "--name" in argv:
        name = argv[argv.index("--name") + 1]
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    if "--encoding" in argv:
        encoding = argv[argv.index("--encoding") + 1]

    csvs = sorted(dir_path.rglob("*.csv"))
    if not csvs:
        print(f"no CSVs under {dir_path}", file=sys.stderr)
        return 1
    infos = [inspect_file(p, encoding=encoding) for p in csvs]
    text = render(dir_path, name, infos)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
