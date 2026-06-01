#!/usr/bin/env python3
"""
Read per-PDE run CSVs and write a long-form summary CSV:

- Column ``pde name``: stem of each source ``*.csv`` (one row per PDE).
- Column ``mse``: ``mean (std)`` over runs in scientific notation (e.g. ``1.23e-02``),
  where each run's value is domain MSE + boundary MSE (see below).

Per source row, total = domain MSE + boundary MSE. Domain MSE is taken from column
``mse``, or ``rmse**2`` if there is no ``mse`` column (or ``mse`` is non-finite and
``rmse`` exists). Boundary MSE uses ``boundary_mse`` or ``b_rmse**2`` under the
same rules.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


def _float_cell(row: dict[str, str], key: str) -> float:
    if key not in row:
        return float("nan")
    s = row[key]
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return float("nan")
    return float(s)


def _domain_mse(row: dict[str, str], fields: frozenset[str]) -> float:
    if "mse" in fields:
        m = _float_cell(row, "mse")
        if math.isfinite(m):
            return m
        if "rmse" in fields:
            r = _float_cell(row, "rmse")
            return r * r if math.isfinite(r) else float("nan")
        return float("nan")
    if "rmse" in fields:
        r = _float_cell(row, "rmse")
        return r * r if math.isfinite(r) else float("nan")
    raise ValueError("need 'mse' or 'rmse'")


def _boundary_mse(row: dict[str, str], fields: frozenset[str]) -> float:
    if "boundary_mse" in fields:
        b = _float_cell(row, "boundary_mse")
        if math.isfinite(b):
            return b
        if "b_rmse" in fields:
            br = _float_cell(row, "b_rmse")
            return br * br if math.isfinite(br) else float("nan")
        return float("nan")
    if "b_rmse" in fields:
        br = _float_cell(row, "b_rmse")
        return br * br if math.isfinite(br) else float("nan")
    raise ValueError("need 'boundary_mse' or 'b_rmse'")


def _fmt_sci_e(x: float, decimals: int = 2) -> str:
    """Lowercase scientific notation (e.g. 1.23e-03); non-finite → 'nan'."""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{decimals}e}"


def _row_total_mse(row: dict[str, str], fields: frozenset[str]) -> float:
    """mse + boundary_mse, using rmse^2 / b_rmse^2 when mse columns are absent."""
    m = _domain_mse(row, fields)
    b = _boundary_mse(row, fields)
    return m + b


def summarize_file(path: Path) -> tuple[str, float, float]:
    stem = path.stem
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row")
        fields = frozenset(reader.fieldnames)
        if not (("mse" in fields or "rmse" in fields) and ("boundary_mse" in fields or "b_rmse" in fields)):
            raise ValueError(
                f"{path}: need ('mse' or 'rmse') and ('boundary_mse' or 'b_rmse'), got {sorted(fields)}"
            )
        sums: list[float] = []
        for row in reader:
            total = _row_total_mse(row, fields)
            if math.isfinite(total):
                sums.append(total)
    if not sums:
        print(
            f"Warning: {path}: no finite mse+boundary_mse rows; using nan",
            file=sys.stderr,
        )
        return stem, float("nan"), float("nan")
    mean = statistics.fmean(sums)
    # std of the per-run totals (sample std when len>1)
    std = statistics.stdev(sums) if len(sums) > 1 else 0.0
    return stem, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate mse+boundary_mse per PDE CSV.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default='./',
        help="Directory containing per-PDE *.csv files (default: ./runs_all_pdes next to this script).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input-dir>/pde_summary.csv).",
    )
    parser.add_argument(
        "--sci-digits",
        type=int,
        default=2,
        metavar="N",
        help="Decimal digits in mantissa after the leading digit (default: 2).",
    )
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    out_path = args.output or (input_dir / "pde_summary.csv")

    csv_files = sorted(input_dir.glob("*.csv"))
    # exclude our own output if re-run in same dir
    csv_files = [p for p in csv_files if p.name != "pde_summary.csv"]
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    d = args.sci_digits
    rows: list[tuple[str, str]] = []
    for p in csv_files:
        name, mean, std = summarize_file(p)
        mse_cell = f"{_fmt_sci_e(mean, d)} ({_fmt_sci_e(std, d)})"
        rows.append((name, mse_cell))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pde name", "mse"])
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
