#!/usr/bin/env python
"""
Merge-upload local chain_eval CSV files into the HF dataset.

Recovery path for runs whose live upload failed (bad/expired token, network):
download the kernel output (launch_kaggle.py output) or take a server's
runs_chain_eval/csv/ dir, then re-upload once a valid write token is at hand.

    export HF_TOKEN_WRITE=hf_...
    python experiments/chain_eval/upload_csv.py path/to/burgers_1d.csv [more.csv ...]
    python experiments/chain_eval/upload_csv.py --dir runs_chain_eval/csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def main():
    sys.path.insert(0, REPO_ROOT)
    import pandas as pd

    from experiments.chain_eval import hf_results

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="*", help="Local per-PDE CSV files (named {pde_name}.csv)")
    parser.add_argument("--dir", default=None, help="Upload every *.csv in this directory")
    parser.add_argument("--hf-repo", default="danil-e/pinnacle-optuna-db")
    parser.add_argument("--hf-dir", default="csv_chain")
    parser.add_argument("--hf-token-write", default=None)
    args = parser.parse_args()

    files = list(args.csvs)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        parser.error("No CSV files given.")

    write_token = args.hf_token_write or os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not write_token:
        sys.exit("Set HF_TOKEN_WRITE (or pass --hf-token-write).")
    read_token = os.environ.get("HF_TOKEN_READ") or write_token

    n_fail = 0
    for path in files:
        pde_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        rows = df.reindex(columns=hf_results.CSV_COLUMNS).to_dict("records")
        print(f"{path}: {len(rows)} rows -> {args.hf_dir}/{pde_name}.csv")
        ok = hf_results.upload_rows(
            args.hf_repo, args.hf_dir, pde_name, rows, write_token, read_token,
            local_dir=os.path.join(os.path.dirname(path) or ".", "merged"),
        )
        n_fail += 0 if ok else 1
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
