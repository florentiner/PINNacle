"""
Per-seed result rows -> per-PDE CSV on the Hugging Face Hub.

One CSV per PDE: {hf_dir}/{pde_name}.csv in the dataset repo (same column
layout as the old csv_seed/ files). The upload path is merge-safe: before
every upload the current remote CSV is re-downloaded and our rows for
(pde_name, chain_key, smoke_test, seed) replace any stale ones, so retries
and concurrent kernels working on *different* PDEs never lose data.
"""
from __future__ import annotations

import os
import time

import pandas as pd

CSV_COLUMNS = [
    "run_timestamp",
    "pde_name",
    "value_type",
    "smoke_test",
    "chain_key",
    "seed",
    "mse_op",
    "mse_bnd",
    "mse_total",
    "l2re_op",
    "l2re_bnd",
    "l2re_total",
    "elapsed_s",
    "chain_json",
]

_KEY_COLS = ["pde_name", "chain_key", "smoke_test", "seed"]


def csv_path_in_repo(hf_dir: str, csv_name: str) -> str:
    """csv_name is the file stem: the pde name, or e.g. '{pde}_{value_type}'."""
    return f"{hf_dir}/{csv_name}.csv"


def download_csv(repo_id: str, path_in_repo: str, token: str | None = None):
    """Return the remote CSV as a DataFrame, or None if it does not exist.

    An invalid/expired token falls back to an anonymous download (the dataset
    repo is public), so a bad read token never breaks seed-resume logic.
    """
    from huggingface_hub import hf_hub_download

    for tok in ([token, None] if token else [None]):
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                token=tok,
                force_download=True,
            )
            return pd.read_csv(local)
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "EntryNotFound" in name or "RepositoryNotFound" in name:
                return None
            if "401" in msg or "invalid user token" in msg or "unauthorized" in msg:
                print(f"WARNING: HF token rejected for download; retrying anonymously.")
                continue
            print(f"WARNING: could not download {path_in_repo}: {e}")
            return None
    return None


def existing_seeds(df, pde_name: str, chain_key: str, smoke_test: bool) -> set:
    """Seeds already recorded for this (pde, chain, smoke) combination."""
    if df is None or df.empty:
        return set()
    mask = (
        (df["pde_name"] == pde_name)
        & (df["chain_key"] == chain_key)
        & (df["smoke_test"].astype(str).str.lower() == str(smoke_test).lower())
    )
    return set(int(s) for s in df.loc[mask, "seed"].tolist())


def merge_rows(remote_df, rows: list[dict]) -> pd.DataFrame:
    """Replace remote rows matching our keys with ours, keep everything else."""
    ours = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if remote_df is None or remote_df.empty:
        return ours
    remote_df = remote_df.reindex(columns=CSV_COLUMNS)
    keys = set(
        tuple(r) for r in ours[_KEY_COLS].astype(str).itertuples(index=False, name=None)
    )
    keep = ~remote_df[_KEY_COLS].astype(str).apply(tuple, axis=1).isin(keys)
    return pd.concat([remote_df[keep], ours], ignore_index=True)


def upload_rows(
    repo_id: str,
    hf_dir: str,
    csv_name: str,
    rows: list[dict],
    write_token: str | None,
    read_token: str | None = None,
    local_dir: str = "csv_chain_local",
    max_attempts: int = 4,
) -> bool:
    """Merge `rows` into the remote CSV and upload. Always keeps a local copy."""
    path_in_repo = csv_path_in_repo(hf_dir, csv_name)
    local_path = os.path.join(local_dir, f"{csv_name}.csv")
    os.makedirs(local_dir, exist_ok=True)

    if not write_token:
        merged = merge_rows(download_csv(repo_id, path_in_repo, read_token), rows)
        merged.to_csv(local_path, index=False)
        print(f"No HF write token — saved locally only: {local_path} ({len(merged)} rows)")
        return False

    from huggingface_hub import upload_file

    for attempt in range(1, max_attempts + 1):
        merged = merge_rows(download_csv(repo_id, path_in_repo, read_token), rows)
        merged.to_csv(local_path, index=False)
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=write_token,
                commit_message=f"chain_eval: {csv_name} +{len(rows)} seed rows",
            )
            print(
                f"Uploaded {path_in_repo} ({len(merged)} rows) -> "
                f"https://huggingface.co/datasets/{repo_id}"
            )
            return True
        except Exception as e:
            wait = 5 * attempt
            print(f"WARNING: HF upload attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(wait)
    print(f"ERROR: giving up on upload; results kept locally at {local_path}")
    return False
