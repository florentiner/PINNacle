"""Fetch a completed Kaggle kernel's output and archive it under runs/, then
(optionally) bridge JAX params into torch artifacts and regenerate comparison
figures — so every session's error landscapes, models, and histories live
locally.

Usage:
  python analysis/fetch_and_archive.py --kernel danilezhov/pinnacle-causal-ks-full \
      --dest runs/kaggle-causal-ks-session3 [--token-file ~/.kaggle/alt_token_antisanct] \
      [--bridge ks|gs] [--compare-baseline runs/07.18-13.19.39-baseline-chaotic/0-0]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)         # owner/slug
    p.add_argument("--dest", required=True)           # runs/<name>
    p.add_argument("--token-file", default=None)      # for alt-account kernels
    p.add_argument("--bridge", choices=["ks", "gs"], default=None)
    p.add_argument("--compare-baseline", default=None)
    args = p.parse_args()

    env = dict(os.environ)
    if args.token_file:
        env["KAGGLE_API_TOKEN"] = open(os.path.expanduser(args.token_file)).read().strip()

    tmp = tempfile.mkdtemp(prefix="kaggle-fetch-")
    subprocess.check_call(["kaggle", "kernels", "output", args.kernel, "-p", tmp],
                          env=env)
    os.makedirs(args.dest, exist_ok=True)
    for name in os.listdir(tmp):
        if name == "pinnacle":       # code copy — skip, it's in the repo
            continue
        src, dst = os.path.join(tmp, name), os.path.join(args.dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    shutil.rmtree(tmp)
    print(f"[ARCHIVED] {args.kernel} -> {args.dest}")

    if args.bridge:
        # find the run dir containing trajectory/ under dest
        run_dir = None
        for root, dirs, files in os.walk(args.dest):
            if os.path.basename(root) == "trajectory" and any(
                    f.endswith("_final_params.npz") for f in files):
                run_dir = os.path.dirname(root)
                break
        if run_dir is None:
            print("[BRIDGE] no trained params found — skipping")
            return
        subprocess.check_call([sys.executable, "-m", "causalpinn.jax_bridge",
                               "--outdir", run_dir, "--case", args.bridge,
                               "--device", "cpu"])
        if args.compare_baseline:
            out = f"analysis/out/{args.bridge}-{os.path.basename(args.dest)}"
            subprocess.check_call([sys.executable, "analysis/compare_chaotic.py",
                                   "--case", args.bridge,
                                   "--baseline", args.compare_baseline,
                                   "--causal", run_dir, "--out", out])
            print(f"[FIGURES] {out}")


if __name__ == "__main__":
    main()
