"""Causal PINN benchmark on PINNacle's chaotic cases (KS, Gray-Scott).

Faithful PyTorch port of Wang, Sankaran & Perdikaris (CMAME 2024) causal training
(PredictiveIntelligenceLab/CausalPINNs), emitting PINNacle-parity artifacts.

Examples:
  python benchmark_causal.py --case ks --device cuda:0 --name causal-ks
  python benchmark_causal.py --case ks --no-causal --name causal-ks-ablation
  python benchmark_causal.py --case gs --encoding fourier --name causal-gs
  python benchmark_causal.py --case ks --resume-dir runs/07.18-.../0-0   # resume
"""
import argparse
import os
import time

os.environ.setdefault("DDEBACKEND", "pytorch")


def main():
    p = argparse.ArgumentParser(description="Causal PINN (SOTA) on chaotic PDEs")
    p.add_argument("--case", choices=["ks", "gs"], default="ks")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")  # cpu | cuda:0
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--encoding", choices=["fourier", "plain"], default="fourier")
    p.add_argument("--no-causal", action="store_true", help="ablation: W=1")
    p.add_argument("--windows", type=int, default=None)
    p.add_argument("--n-t", type=int, default=32)
    p.add_argument("--n-s", type=int, default=None)
    p.add_argument("--iter-cap", type=int, default=200000, help="per tol stage")
    p.add_argument("--tol-list", type=str, default="1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--snapshot-every", type=int, default=25000)
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--resume-dir", type=str, default="")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--compile", action="store_true", help="torch.compile the net")
    args = p.parse_args()

    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig, run

    name = args.name or f"causal-{args.case}" + ("-ablation" if args.no_causal else "")
    if args.outdir:
        outdir = args.outdir
    elif args.resume_dir:
        outdir = args.resume_dir if os.path.basename(args.resume_dir) == "0-0" \
            else os.path.join(args.resume_dir, "0-0")
    else:
        stamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
        outdir = f"runs/{stamp}-{name}/0-0"
    os.makedirs(outdir, exist_ok=True)

    defaults = {"ks": dict(windows=10, n_s=256), "gs": dict(windows=20, n_s=256)}[args.case]
    cfg = CausalConfig(
        case=args.case,
        device=args.device,
        seed=args.seed,
        encoding=args.encoding,
        causal=not args.no_causal,
        windows=args.windows or defaults["windows"],
        n_t=args.n_t,
        n_s=args.n_s or defaults["n_s"],
        tol_list=tuple(float(x) for x in args.tol_list.split(",")),
        iter_cap=args.iter_cap,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        snapshot_every=args.snapshot_every,
        max_hours=args.max_hours,
        outdir=outdir,
        resume_dir=args.resume_dir,
        compile=args.compile,
    )

    import torch
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to cpu")
        cfg.device = "cpu"

    case = get_case(args.case, cfg)
    print(f"[RUN] case={args.case} causal={cfg.causal} encoding={cfg.encoding} "
          f"windows={cfg.windows} device={cfg.device} outdir={outdir}")
    run(case, cfg)

    from causalpinn.check_artifacts import check
    ok = check(outdir, args.case)
    print(f"[ARTIFACT CHECK] {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
