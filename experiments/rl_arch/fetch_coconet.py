#!/usr/bin/env python
"""Клон кода CoCoNet (github.com/AI4OPT/CoCoNet, MIT) на зафиксированном коммите — для conformal_check.py в кернеле.

    python experiments/rl_arch/fetch_coconet.py --dir /kaggle/working/CoCoNet
"""
import argparse
import os
import subprocess

COMMIT = "fa42c36"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.dir, "coconet")):
        subprocess.run(["git", "clone", "-q", "https://github.com/AI4OPT/CoCoNet.git", a.dir], check=True)
    subprocess.run(["git", "-C", a.dir, "checkout", "-q", COMMIT], check=True)
    print("CoCoNet:", subprocess.run(["git", "-C", a.dir, "log", "--oneline", "-1"], capture_output=True,
                                     text=True).stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
