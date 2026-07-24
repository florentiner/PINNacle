#!/usr/bin/env python
"""
Push PINNacle chain-eval GPU kernels to one or more Kaggle accounts.

Reads accounts + tokens from a local (gitignored!) accounts.json — see
accounts.example.json. Each account gets one private script kernel that clones
this repo's branch and runs run_all.py for its assigned PDEs (10 seeds each,
per-seed CSV upload to the HF dataset). PDEs without an explicit assignment
are split round-robin across accounts.

Commands:
    python experiments/chain_eval/launch_kaggle.py launch [--smoke] [--accounts a,b]
    python experiments/chain_eval/launch_kaggle.py status
    python experiments/chain_eval/launch_kaggle.py output [--accounts a,b]

Requires the `kaggle` CLI >= 1.7 (KGAT access-token auth via KAGGLE_API_TOKEN).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BUILD_DIR = os.path.join(SCRIPT_DIR, "kaggle_build")
STATE_FILE = os.path.join(BUILD_DIR, "pushed.json")
TEMPLATE = os.path.join(SCRIPT_DIR, "kernel_template.py")

DEFAULT_REPO_URL = "https://github.com/florentiner/PINNacle.git"
DEFAULT_BRANCH = "chain_eval"
SMOKE_PDES = ["burgers_1d"]


def load_accounts(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(
            f"Accounts file not found: {path}\n"
            f"Copy {os.path.join(SCRIPT_DIR, 'accounts.example.json')} to it and fill in tokens."
        )
    with open(path) as f:
        cfg = json.load(f)
    if not cfg.get("accounts"):
        sys.exit("accounts.json has no 'accounts' entries.")
    return cfg


def kaggle_env(token: str) -> dict:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    # Make sure a leftover ~/.kaggle/kaggle.json can't override the account token.
    env["KAGGLE_CONFIG_DIR"] = os.path.join(BUILD_DIR, ".kaggle_cfg")
    return env


def discover_username(token: str) -> str:
    out = subprocess.run(
        ["kaggle", "config", "view"], env=kaggle_env(token),
        capture_output=True, text=True,
    )
    m = re.search(r"username:\s*(\S+)", out.stdout)
    if not m:
        sys.exit(f"Could not resolve Kaggle username from token (output: {out.stdout!r} {out.stderr!r})")
    return m.group(1)


def assign_pdes(cfg: dict, pdes_override: list[str] | None) -> dict[str, list[str]]:
    sys.path.insert(0, REPO_ROOT)
    from experiments.chain_eval.pde_names import ALL_PDE_NAMES

    accounts = cfg["accounts"]
    if pdes_override is not None:
        pool = list(pdes_override)
        assignment = {a["name"]: [] for a in accounts}
        rr_names = [a["name"] for a in accounts]
    else:
        # "pdes" missing -> account takes part in round-robin of unassigned PDEs;
        # "pdes": [] -> account explicitly runs no main-chain PDEs.
        assignment = {}
        assigned = set()
        for a in accounts:
            listed = a.get("pdes", None) or []
            unknown = [p for p in listed if p not in ALL_PDE_NAMES]
            if unknown:
                sys.exit(f"Account {a['name']}: unknown PDEs {unknown}")
            assignment[a["name"]] = list(listed)
            assigned.update(listed)
        rr_names = [a["name"] for a in accounts if "pdes" not in a]
        pool = [p for p in ALL_PDE_NAMES if p not in assigned] if rr_names else []
        leftovers = [p for p in ALL_PDE_NAMES if p not in assigned]
        if leftovers and not rr_names:
            print(f"NOTE: not assigned to any account: {', '.join(leftovers)}")
    for i, pde in enumerate(pool):
        assignment[rr_names[i % len(rr_names)]].append(pde)
    return assignment


def build_jobs(account: dict, cfg: dict, pdes: list[str]) -> list[dict]:
    """Best-chain evals (csv_seed, continuous+fixed) first, then main-chain PDEs."""
    main_chain_key = cfg.get("chain_key") or "chain_adam_lbfgs"
    jobs = []
    for pde in account.get("best_pdes", []):
        for vt in ("continuous", "fixed"):
            chain_json = f"experiments/chain_eval/best_chains/{pde}_{vt}.json"
            if not os.path.exists(os.path.join(SCRIPT_DIR, "best_chains", f"{pde}_{vt}.json")):
                sys.exit(f"Account {account['name']}: no best chain file {chain_json}")
            jobs.append({
                "pde": pde,
                "chain_json": chain_json,
                "value_type": vt,
                "hf_dir": cfg.get("best_hf_dir", "csv_seed"),
                "csv_name": f"{pde}_{vt}",
                "chain_key": f"{pde}_{vt}",
            })
    for pde in pdes:
        jobs.append({
            "pde": pde,
            "chain_json": None,
            "value_type": "chain",
            "hf_dir": cfg.get("hf_dir", "csv_chain"),
            "csv_name": pde,
            "chain_key": main_chain_key,
        })
    return jobs


def build_kernel_dir(account: dict, cfg: dict, jobs: list[dict], args) -> str:
    slug = f"pinnacle-chain-{re.sub(r'[^a-z0-9-]', '-', account['name'].lower())}"
    if args.smoke:
        slug += "-smoke"
    username = account["username"]
    kdir = os.path.join(BUILD_DIR, slug)
    os.makedirs(kdir, exist_ok=True)

    config = {
        "repo": cfg.get("repo_url", DEFAULT_REPO_URL),
        "branch": cfg.get("branch", DEFAULT_BRANCH),
        "jobs": jobs,
        "n_seeds": 2 if args.smoke else cfg.get("n_seeds", 10),
        "seed_base": cfg.get("seed_base", 42),
        "test_epochs": 3 if args.smoke else None,
        "display_every": 1 if args.smoke else cfg.get("display_every", 100),
        # Benchmarked on T4x2: 2 workers/GPU is >=1.0x everywhere, 1.67x on light PDEs.
        "workers_per_gpu": cfg.get("workers_per_gpu", 2),
        "hf_repo": cfg.get("hf_repo", "danil-e/pinnacle-optuna-db"),
        "hf_token_write": cfg.get("hf_token_write", ""),
        "hf_token_read": cfg.get("hf_token_read", ""),
        "force": bool(args.force),
    }
    with open(TEMPLATE) as f:
        body = f.read().replace("__CONFIG_JSON__", json.dumps(config))
    with open(os.path.join(kdir, "kernel_body.py"), "w") as f:
        f.write(body)

    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "kernel_body.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return kdir


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def cmd_launch(cfg: dict, args):
    accounts = cfg["accounts"]
    if args.accounts:
        wanted = {a.strip() for a in args.accounts.split(",")}
        accounts = [a for a in accounts if a["name"] in wanted]
        if not accounts:
            sys.exit(f"No accounts match {args.accounts}")
        cfg = dict(cfg, accounts=accounts)

    pdes_override = None
    if args.smoke:
        pdes_override = SMOKE_PDES * len(accounts)  # 1 smoke PDE per account
    elif args.pdes:
        pdes_override = [p.strip() for p in args.pdes.split(",") if p.strip()]
    assignment = assign_pdes(cfg, pdes_override)

    state = load_state()
    for account in accounts:
        pdes = assignment.get(account["name"], [])
        jobs = build_jobs(account, cfg, pdes) if not args.smoke else build_jobs(
            dict(account, best_pdes=[]), cfg, pdes
        )
        if not jobs:
            print(f"[{account['name']}] no jobs assigned — skipping.")
            continue
        token = account["kaggle_token"]
        account = dict(account, username=account.get("username") or discover_username(token))
        kdir = build_kernel_dir(account, cfg, jobs, args)
        with open(os.path.join(kdir, "kernel-metadata.json")) as f:
            ref = json.load(f)["id"]
        shape = cfg.get("machine_shape", "NvidiaTeslaT4")
        job_names = ", ".join(j["csv_name"] for j in jobs)
        print(f"[{account['name']}] pushing {ref}  (shape {shape})  jobs: {job_names}")
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "_push_with_shape.py"), kdir, shape],
            env=kaggle_env(token),
        )
        if r.returncode != 0:
            print(f"[{account['name']}] shaped push failed — falling back to plain kaggle CLI push")
            r = subprocess.run(["kaggle", "kernels", "push", "-p", kdir], env=kaggle_env(token))
        if r.returncode == 0:
            state[account["name"]] = {
                "ref": ref,
                "jobs": [j["csv_name"] for j in jobs],
                "smoke": bool(args.smoke),
            }
            save_state(state)
            print(f"[{account['name']}] pushed: https://www.kaggle.com/code/{ref}")
        else:
            print(f"[{account['name']}] PUSH FAILED (exit {r.returncode})")


def _accounts_by_name(cfg):
    return {a["name"]: a for a in cfg["accounts"]}

def cmd_status(cfg: dict, args):
    state = load_state()
    if not state:
        sys.exit("No pushed kernels recorded (kaggle_build/pushed.json missing).")
    by_name = _accounts_by_name(cfg)
    for name, info in state.items():
        if args.accounts and name not in args.accounts.split(","):
            continue
        acc = by_name.get(name)
        if not acc:
            print(f"[{name}] not in accounts.json — skipping")
            continue
        r = subprocess.run(
            ["kaggle", "kernels", "status", info["ref"]],
            env=kaggle_env(acc["kaggle_token"]), capture_output=True, text=True,
        )
        print(f"[{name}] {info['ref']}: {(r.stdout or r.stderr).strip()}")


def cmd_output(cfg: dict, args):
    state = load_state()
    by_name = _accounts_by_name(cfg)
    for name, info in state.items():
        if args.accounts and name not in args.accounts.split(","):
            continue
        acc = by_name.get(name)
        if not acc:
            continue
        out_dir = os.path.join(BUILD_DIR, "logs", name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{name}] downloading output of {info['ref']} -> {out_dir}")
        subprocess.run(
            ["kaggle", "kernels", "output", info["ref"], "-p", out_dir],
            env=kaggle_env(acc["kaggle_token"]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["launch", "status", "output"], nargs="?", default="launch")
    parser.add_argument("--accounts-json", default=os.path.join(SCRIPT_DIR, "accounts.json"))
    parser.add_argument("--accounts", default=None, help="Comma-separated account names to act on")
    parser.add_argument("--pdes", default=None, help="Override PDE list (comma-separated), split round-robin")
    parser.add_argument("--smoke", action="store_true",
                        help="Push a tiny end-to-end test kernel: 1 PDE, 2 seeds, 3 epochs/stage")
    parser.add_argument("--force", action="store_true", help="Ignore seeds already recorded on HF")
    args = parser.parse_args()

    cfg = load_accounts(args.accounts_json)
    if args.command == "launch":
        cmd_launch(cfg, args)
    elif args.command == "status":
        cmd_status(cfg, args)
    else:
        cmd_output(cfg, args)


if __name__ == "__main__":
    main()
