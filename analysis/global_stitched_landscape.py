"""A REAL global error space for the chaotic causal run — not a mosaic, not a stitch.

The mosaic paints true local slices side by side; it is honest but it is still N separate
objectives. There IS a single global objective, and the per-window data is exactly what
it needs: the causal method's output is the STITCHED solution, whose parameters are the
TUPLE of all window networks, Theta = (theta_0, ..., theta_{N-1}). On that product space
one well-defined global loss exists:

    L_global(Theta) = mean_k <PDE residual^2 of window k on its own slice>          (physics)
                    + ||u_0(x,0) - u_IC(x)||^2                                      (initial condition)
                    + mean_k ||u_k(x,T_w) - u_{k+1}(x,0)||^2                        (interface continuity)

Every term is a real evaluation of real networks; nothing is projected or averaged away.
This is precisely what "the whole marching solution is correct" means, and the causal run
is a trajectory in this space: while window k trains, coordinates of block k move and the
others stand still (windows behind the front are frozen at their solutions, windows ahead
are at their earliest recorded state). The plane is centred on the FINAL full solution
Theta*, so the surface is a true 2D cut of the true global objective through the real
optimum — unlike a joint plane through a cloud centroid.

Writes analysis/out/global_stitched_{case}.npz; plotted by plot_global_stitched.py.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "analysis")
os.environ.setdefault("DDEBACKEND", "pytorch")
from landscape_full_training import CFG, flat, unflat, gather_snaps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--grid", type=int, default=17)
    ap.add_argument("--npts", type=int, default=200)
    ap.add_argument("--nspace", type=int, default=256)
    ap.add_argument("--zoom", type=float, default=0.10)
    args = ap.parse_args()
    cfg = CFG[args.case]
    N = cfg["n_win"]
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)

    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict

    ccfg = CausalConfig(case=args.case, device="cpu", windows=N)
    case = get_case(args.case, ccfg)
    net = case.build_net(cfg["encoding"], ccfg.seed, torch.device("cpu"))
    ref_sd = {k: v.clone() for k, v in net.state_dict().items()}
    D = flat(ref_sd).numel()
    print(f"[{args.case}] {N} windows x {D} params = {N*D} global dimensions", flush=True)

    def load(path):
        net.load_state_dict(jax_npz_to_state_dict(np.load(path)), strict=False)
        return flat(net.state_dict()).clone()

    finals, firsts = {}, {}
    for k in range(N):
        p = os.path.join(cfg["final_dir"], "trajectory", f"w{k}_final_params.npz")
        if os.path.exists(p):
            finals[k] = load(p)
        sn = gather_snaps(cfg, k)
        if sn:
            firsts[k] = load(sn[0])
    for k in range(N):                       # windows without snapshots: use their final
        firsts.setdefault(k, finals[k])
    Theta_star = torch.cat([finals[k] for k in range(N)])

    # ---------------- the global objective ----------------
    if args.case == "ks":
        tw = torch.tensor(rng.uniform(0, case.T_w * 1.01, size=(args.npts, 1)),
                          dtype=torch.float32)
        xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(args.npts, 1)),
                          dtype=torch.float32)
        xs = torch.tensor(case.x_star[:, None], dtype=torch.float32)
        u_ic = torch.tensor(case.ref[:, 0, :], dtype=torch.float32)
        t_end = torch.full_like(xs, float(case.T_w))
        t_zero = torch.zeros_like(xs)

        def wpred(vec, t, x):
            net.load_state_dict(unflat(vec, ref_sd))
            return net(t, x)

        def wres(vec):
            net.load_state_dict(unflat(vec, ref_sd))
            with torch.enable_grad():
                r = case.residual(net, tw, xw, case.T_w)
            return float((r ** 2).mean())
    else:
        tw = torch.tensor(rng.uniform(0, 1.01, size=(args.npts, 1)), dtype=torch.float32)
        xw = torch.tensor(rng.uniform(-1, 1, size=(args.npts, 2)), dtype=torch.float32)
        ic_c = np.asarray(case.ic_arrays()[0])
        sub = rng.choice(len(ic_c), min(args.nspace, len(ic_c)), replace=False)
        xs = torch.tensor(ic_c[sub], dtype=torch.float32)
        u_ic = torch.tensor(np.asarray(case.ic_arrays()[1])[sub], dtype=torch.float32)
        t_end = torch.ones(len(xs), 1)
        t_zero = torch.zeros(len(xs), 1)

        def wpred(vec, t, x):
            net.load_state_dict(unflat(vec, ref_sd))
            return net(t, x[:, 0:1], x[:, 1:2])

        def wres(vec):
            net.load_state_dict(unflat(vec, ref_sd))
            with torch.enable_grad():
                r = case.residual(net, tw, xw, case.T_w)
            return float((r ** 2).mean())

    def L_global(Theta, parts=False):
        blocks = [Theta[k * D:(k + 1) * D] for k in range(N)]
        res = np.mean([wres(b) for b in blocks])
        ic = float(((wpred(blocks[0], t_zero, xs) - u_ic) ** 2).mean())
        cont = []
        prev_end = wpred(blocks[0], t_end, xs)
        for k in range(1, N):
            start = wpred(blocks[k], t_zero, xs)
            cont.append(float(((prev_end - start) ** 2).mean()))
            prev_end = wpred(blocks[k], t_end, xs)
        c = float(np.mean(cont))
        return (res, ic, c) if parts else res + ic + c

    r0, i0, c0 = L_global(Theta_star, parts=True)
    print(f"  L_global at the trained solution: residual {r0:.3e} + IC {i0:.3e} + "
          f"continuity {c0:.3e} = {r0+i0+c0:.3e}", flush=True)

    # ---------------- the run as a trajectory in this space ----------------
    traj, labels = [], []
    for k in range(N):
        snaps = gather_snaps(cfg, k)
        states = [load(s) for s in snaps] + [finals[k]]
        steps = [int(re.search(r"snap_(\d+)", s).group(1)) for s in snaps] + [-1]
        for st, vec in zip(steps, states):
            Th = torch.cat([finals[j] if j < k else (vec if j == k else firsts[j])
                            for j in range(N)])
            traj.append(Th)
            labels.append((k, st))
    Tr = torch.stack(traj)
    print(f"  global trajectory: {len(Tr)} states", flush=True)

    Xc = Tr - Theta_star
    _, S, V = torch.pca_lowrank(Xc, q=6)
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    ev2 = float((S[:2] ** 2).sum() / (Xc ** 2).sum())
    a_tr = (Xc @ d1).numpy()
    b_tr = (Xc @ d2).numpy()
    print(f"  plane holds {ev2*100:.1f}% of the global trajectory's variance", flush=True)

    def grid_through_zero(lo, hi, n):
        g = np.linspace(lo, hi, n)
        g[np.argmin(np.abs(g))] = 0.0          # make sure Theta* is sampled
        return np.unique(g)

    pa = 0.22 * (a_tr.max() - a_tr.min() + 1e-9)
    pb = 0.22 * (b_tr.max() - b_tr.min() + 1e-9)
    A = grid_through_zero(min(a_tr.min() - pa, -pa), a_tr.max() + pa, args.grid)
    B = grid_through_zero(min(b_tr.min() - pb, -pb), b_tr.max() + pb, args.grid)
    Z = np.zeros((len(B), len(A)))
    for i, b in enumerate(B):
        for j, a in enumerate(A):
            Z[i, j] = L_global(Theta_star + a * d1 + b * d2)
        print(f"  row {i+1}/{len(B)} (wide)", flush=True)

    zf = args.zoom
    AZ = grid_through_zero(-zf * pa, zf * pa, args.grid)
    BZ = grid_through_zero(-zf * pb, zf * pb, args.grid)
    ZZ = np.zeros((len(BZ), len(AZ)))
    for i, b in enumerate(BZ):
        for j, a in enumerate(AZ):
            ZZ[i, j] = L_global(Theta_star + a * d1 + b * d2)
        print(f"  row {i+1}/{len(BZ)} (zoom)", flush=True)

    # loss along the trajectory (subsampled: every window end + every k-th state)
    keep = [i for i, (k, st) in enumerate(labels) if st == -1]
    keep += list(range(0, len(Tr), max(1, len(Tr) // 90)))
    keep = sorted(set(keep))
    lt = np.full(len(Tr), np.nan)
    for i in keep:
        lt[i] = L_global(Tr[i])
        if len(keep) > 20 and keep.index(i) % 20 == 0:
            print(f"  traj {keep.index(i)}/{len(keep)}", flush=True)
    np.savez(f"analysis/out/global_stitched_{args.case}.npz", A=A, B=B, Z=Z,
             AZ=AZ, BZ=BZ, ZZ=ZZ,
             a_tr=a_tr, b_tr=b_tr, labels=np.array(labels), loss_traj=lt,
             ev2=ev2, parts=np.array([r0, i0, c0]))
    print(f"saved global_stitched_{args.case}.npz  (Z {Z.min():.3e} … {Z.max():.3e})")


if __name__ == "__main__":
    main()
