#!/usr/bin/env python3
"""Push the ns2d_liddriven Optuna kernel to N Kaggle accounts.

Each account runs its OWN DB file (`optuna_studies/ns2d_liddriven_<type>__<user>.db`)
so concurrent accounts never clobber each other's HF-synced DB. Trials per
account = ceil(--trials / N_ACCOUNTS), so the fleet totals ~--trials for the
value type. Merge the per-account DBs afterwards with merge_dbs.py.

Usage:
    HF_TOKEN_WRITE=hf_xxx python launch.py --value-type fixed --accounts saintsantiago1991 elenablairm corbinvillalobosq
    HF_TOKEN_WRITE=hf_xxx python launch.py --value-type continuous --accounts briannarodgers06 suttonldennis00
"""
import argparse, json, math, os, subprocess, sys

HERE      = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS  = os.path.expanduser("~/.kaggle/ns2d_liddriven_accounts.tsv")
BUILD     = os.path.join(HERE, "_build")        # contains tokens — keep out of git
TEMPLATE  = os.path.join(HERE, "kernel_template.py")
HF_REPO   = "danil-e/pinnacle-optuna-db"
GITHUB    = "https://github.com/florentiner/PINNacle.git"
BRANCH    = "optuna_exp"


def load_accounts(path=ACCOUNTS):
    accts = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    accts[parts[0].strip()] = parts[1].strip()
    return accts


def kaggle_env(token, cfg_dir):
    env = {k: v for k, v in os.environ.items()
           if k not in ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN")}
    env["KAGGLE_CONFIG_DIR"] = cfg_dir
    env["KAGGLE_API_TOKEN"] = token
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-type", required=True, choices=["fixed", "continuous"])
    ap.add_argument("--accounts", nargs="+", required=True,
                    help="account usernames from ~/.kaggle/ns2d_liddriven_accounts.tsv")
    ap.add_argument("--trials", type=int, default=50, help="TOTAL trials across accounts")
    ap.add_argument("--processes", type=int, default=2, help="workers per account session")
    ap.add_argument("--timeout-hours", type=float, default=11.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN_WRITE")
    if not hf_token and not args.dry_run:
        sys.exit("ERROR: set HF_TOKEN_WRITE (needed for DB upload to HF).")

    accounts = load_accounts()
    missing = [a for a in args.accounts if a not in accounts]
    if missing:
        sys.exit(f"ERROR: accounts not in {ACCOUNTS}: {missing}")

    per = math.ceil(args.trials / len(args.accounts))
    # Each of the N_PROCESSES workers runs its own optimize(n_trials=...) against the
    # shared local DB, so the account total is processes * per_worker.
    per_worker = max(1, math.ceil(per / args.processes))
    tmpl = open(TEMPLATE).read()
    os.makedirs(BUILD, exist_ok=True)

    for acct in args.accounts:
        token = accounts[acct]
        slug = f"ns2d-liddriven-{args.value_type}"
        kdir = os.path.join(BUILD, f"{acct}__{slug}")
        os.makedirs(kdir, exist_ok=True)

        config = {
            "pde": "ns2d_liddriven",
            "value_type": args.value_type,
            "n_trials": per_worker,
            "n_processes": args.processes,
            "timeout_hours": args.timeout_hours,
            "db_filename": f"ns2d_liddriven_{args.value_type}__{acct}.db",
            "study_name": f"ns2d_liddriven_{args.value_type}",
            "hf_repo": HF_REPO,
            "hf_token": hf_token or "PLACEHOLDER",
            "github_repo": GITHUB,
            "branch": BRANCH,
        }
        src = tmpl.replace("__CONFIG_JSON__", json.dumps(config))
        with open(os.path.join(kdir, "run.py"), "w") as fh:
            fh.write(src)

        meta = {
            "id": f"{acct}/{slug}",
            "title": slug,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": [],
        }
        with open(os.path.join(kdir, "kernel-metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"[{acct}] {slug}: {per} trials x {args.processes} workers -> {config['db_filename']}", flush=True)
        if args.dry_run:
            print(f"   built (dry-run): {kdir}", flush=True)
            continue

        cfg_dir = os.path.join(BUILD, ".kgcfg", acct)
        os.makedirs(cfg_dir, exist_ok=True)
        proc = subprocess.run(["kaggle", "kernels", "push", "-p", kdir],
                              env=kaggle_env(token, cfg_dir),
                              capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        print("   " + out.replace("\n", "\n   "), flush=True)


if __name__ == "__main__":
    main()
