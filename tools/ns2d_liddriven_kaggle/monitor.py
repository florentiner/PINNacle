#!/usr/bin/env python3
"""Monitor the ns2d_liddriven Kaggle fleet and auto-resume finished sessions.

For each account: report kernel status + trial count in its HF DB. If the kernel
is no longer running but the account's study is below target, re-push it (which
resumes from the HF DB). Intended to run from cron every ~30 min.

    python monitor.py            # report only
    python monitor.py --resume   # report + re-push incomplete accounts
"""
import argparse, math, os, re, subprocess, sys, tempfile

HERE     = os.path.dirname(os.path.abspath(__file__))
BUILD    = os.path.join(HERE, "_build")
ACCOUNTS = os.path.expanduser("~/.kaggle/ns2d_liddriven_accounts.tsv")
HF_FILE  = os.path.expanduser("~/.kaggle/hf_write_token")
HF_REPO  = "danil-e/pinnacle-optuna-db"

# value_type -> fleet plan
PLAN = {
    "fixed":      {"slug": "ns2d-liddriven-fixed",
                   "accounts": ["saintsantiago1991", "elenablairm", "corbinvillalobosq"],
                   "trials_per_worker": 9,  "processes": 2, "timeout_hours": 11},
    "continuous": {"slug": "ns2d-liddriven-continuous",
                   "accounts": ["briannarodgers06", "suttonldennis00"],
                   "trials_per_worker": 13, "processes": 2, "timeout_hours": 11},
}
RUNNING = ("RUNNING", "QUEUED")


def load_accounts():
    a = {}
    for line in open(ACCOUNTS):
        p = line.strip().split("\t")
        if len(p) >= 2:
            a[p[0]] = p[1]
    return a


def kernel_status(acct, slug, token):
    cfg = tempfile.mkdtemp()
    env = dict(os.environ, KAGGLE_CONFIG_DIR=cfg, KAGGLE_API_TOKEN=token)
    r = subprocess.run(["kaggle", "kernels", "status", f"{acct}/{slug}"],
                       env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    m = re.search(r"status \"KernelWorkerStatus\.(\w+)\"", out)
    return m.group(1) if m else (out.strip().splitlines()[-1] if out.strip() else "?")


def hf_trial_count(db_filename, hf_token):
    """Download the per-account DB from HF and count completed trials."""
    try:
        import sqlite3
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename=f"optuna_studies/{db_filename}",
                                repo_type="dataset", token=hf_token, local_dir=tempfile.mkdtemp())
        except EntryNotFoundError:
            return 0
        con = sqlite3.connect(p)
        n = con.execute(
            "SELECT COUNT(*) FROM trials WHERE state IN ('COMPLETE','FAIL','PRUNED')"
        ).fetchone()[0]
        con.close()
        return n
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="re-push accounts below target")
    ap.add_argument("--quiet", action="store_true",
                    help="print only actionable lines (resumes / completions / errors)")
    args = ap.parse_args()

    accounts = load_accounts()
    hf_token = open(HF_FILE).read().strip() if os.path.exists(HF_FILE) else ""
    printed_header = set()

    for vt, plan in PLAN.items():
        if not args.quiet:
            print(f"\n=== {vt} ===")
        for acct in plan["accounts"]:
            db = f"ns2d_liddriven_{vt}__{acct}.db"
            target = plan["trials_per_worker"] * plan["processes"]
            status = kernel_status(acct, plan["slug"], accounts[acct])
            n = hf_trial_count(db, hf_token)
            done = isinstance(n, int) and n >= target
            actionable = done or status not in RUNNING or not isinstance(n, int)
            if args.quiet and not actionable:
                continue
            if args.quiet and vt not in printed_header:
                print(f"=== {vt} ===")
                printed_header.add(vt)
            print(f"  {acct:22s} status={status:8s} trials={n}/{target}"
                  f"{'  DONE' if done else ''}")
            if args.resume and status not in RUNNING and not done:
                print(f"    -> resuming {acct} ({plan['slug']})")
                env = dict(os.environ, HF_TOKEN_WRITE=hf_token)
                subprocess.run([sys.executable, os.path.join(HERE, "launch.py"),
                                "--value-type", vt, "--accounts", acct,
                                "--trials", str(target), "--processes", str(plan["processes"]),
                                "--timeout-hours", str(plan["timeout_hours"])], env=env)


if __name__ == "__main__":
    main()
