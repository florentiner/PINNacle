"""Bridge: import JAX-engine trained params into the PyTorch port and generate
the full PINNacle-parity artifact set (predictions, error landscapes, residual
fields, stitched grids) via the existing RunLogger machinery.

Usage:
  python -m causalpinn.jax_bridge --outdir runs/...-causal-ks/0-0 [--device cuda:0]
(expects trajectory/w{k}_final_params.npz written by causalpinn/jax_runner.py)
"""
import argparse
import glob
import os
import re

import numpy as np
import torch


def jax_npz_to_state_dict(npz):
    """Map reference modified_MLP params to causalpinn.model.ModifiedMLP.
    JAX uses  y = x @ W + b  (W: in x out); torch Linear stores weight (out, in)."""
    sd = {}
    sd["gate_u.weight"] = torch.tensor(npz["U1"].T.copy())
    sd["gate_u.bias"] = torch.tensor(npz["b1"].copy())
    sd["gate_v.weight"] = torch.tensor(npz["U2"].T.copy())
    sd["gate_v.bias"] = torch.tensor(npz["b2"].copy())
    n = int(npz["n_layers"])
    for i in range(n - 1):  # hidden layers (last JAX layer is the output layer)
        sd[f"layers.{i}.weight"] = torch.tensor(npz[f"W{i}"].T.copy())
        sd[f"layers.{i}.bias"] = torch.tensor(npz[f"bb{i}"].copy())
    sd["out.weight"] = torch.tensor(npz[f"W{n - 1}"].T.copy())
    sd["out.bias"] = torch.tensor(npz[f"bb{n - 1}"].copy())
    return sd


def self_test(case, net, npz, device):
    """Verify torch net(params from JAX) == JAX apply on random points."""
    rng = np.random.default_rng(0)
    if case.name == "ks":
        t = rng.uniform(0, 0.101, size=(64, 1)).astype(np.float32)
        coords = [rng.uniform(0, 2 * np.pi, size=(64, 1)).astype(np.float32)]
        # integer-power semantics of the JAX reference: negative exponents -> 0
        M_t, M_x = 6, 5
        e = np.arange(-(M_t // 2), M_t // 2)
        k_t = np.where(e >= 0, 10.0 ** np.maximum(e, 0), 0.0)
        k_x = np.arange(1, M_x + 1)
        enc = np.concatenate([k_t * t, np.ones_like(t), np.cos(k_x * coords[0]),
                              np.sin(k_x * coords[0])], axis=1)
    else:  # gs, plain encoding: [1, k_t*tau (k_t=10^{0..2}), x, y]
        t = rng.uniform(0, 1.01, size=(64, 1)).astype(np.float32)
        coords = [rng.uniform(-1, 1, size=(64, 1)).astype(np.float32),
                  rng.uniform(-1, 1, size=(64, 1)).astype(np.float32)]
        k_t = 10.0 ** np.arange(0, 3)
        enc = np.concatenate([np.ones_like(t), k_t * t] + coords, axis=1)
    with torch.no_grad():
        y_t = net(torch.tensor(t, device=device),
                  *[torch.tensor(c, device=device) for c in coords]).cpu().numpy()
    U = np.tanh(enc @ npz["U1"] + npz["b1"])
    V = np.tanh(enc @ npz["U2"] + npz["b2"])
    h = enc
    n = int(npz["n_layers"])
    for i in range(n - 1):
        z = np.tanh(h @ npz[f"W{i}"] + npz[f"bb{i}"])
        h = z * U + (1 - z) * V
    y_j = h @ npz[f"W{n - 1}"] + npz[f"bb{n - 1}"]
    err = np.abs(y_t - y_j).max()
    assert err < 1e-4, f"bridge self-test failed: max diff {err}"
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--case", choices=["ks", "gs"], default="ks")
    p.add_argument("--windows", type=int, default=None,
                   help="true window count of the RUN (not the number of "
                        "param files present — partial runs bridge partially)")
    args = p.parse_args()
    if args.windows is None:
        args.windows = {"ks": 10, "gs": 20}[args.case]

    import sys
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("DDEBACKEND", "pytorch")
    from causalpinn.cases import get_case
    from causalpinn.hypothesis_log import RunLogger
    from causalpinn.train import CausalConfig, predict_grid, residual_grid

    param_files = sorted(glob.glob(os.path.join(args.outdir, "trajectory",
                                                "w*_final_params.npz")),
                         key=lambda s: int(re.search(r"w(\d+)_", s).group(1)))
    assert param_files, f"no w*_final_params.npz under {args.outdir}/trajectory"

    cfg = CausalConfig(case=args.case, device=args.device, windows=args.windows,
                       outdir=args.outdir)
    case = get_case(args.case, cfg)
    device = torch.device(args.device)
    logger = RunLogger(case, cfg, resume=True)
    stitched = np.full_like(case.ref, np.nan, dtype=np.float32)

    for pf in param_files:
        k = int(re.search(r"w(\d+)_final_params", pf).group(1))
        npz = np.load(pf)
        net = case.build_net("fourier" if args.case == "ks" else "plain", cfg.seed, device)
        res = net.load_state_dict(jax_npz_to_state_dict(npz), strict=False)
        assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys}"
        assert all(k.startswith("encoding.") for k in res.missing_keys), \
            f"missing non-buffer keys: {res.missing_keys}"  # encoding buffers are constants
        err = self_test(case, net, npz, device)
        print(f"[BRIDGE w{k}] params loaded, forward self-test max diff {err:.2e}")
        t_loc, coords = case.eval_points_local(k)
        pred = predict_grid(net, t_loc, coords, device, n_comp=case.n_comp)
        opt = torch.optim.Adam(net.parameters())  # placeholder for ckpt format
        logger.window_done(k, -1, net, opt, pred, stitched,
                           predict_grid, residual_grid)
        ic_coords, _ = case.ic_arrays()
        t1_local = case.T_w * case.t_scale
        handoff = predict_grid(net, [t1_local], ic_coords, device,
                               n_comp=case.n_comp)[0]
        logger.save_handoff(k, handoff.astype(np.float32))

    logger.finalize(stitched)
    # convert the JAX-side history into the standard causal/history.npz format
    # (must run AFTER finalize: the logger's flush would overwrite it)
    hj = os.path.join(args.outdir, "causal", "history_jax.npz")
    if os.path.exists(hj):
        h = np.load(hj)
        keep = ["step", "window", "stage", "tol", "W", "L_t", "t_r",
                "w_min", "loss_ic"]
        np.savez_compressed(os.path.join(args.outdir, "causal", "history.npz"),
                            **{k: h[k] for k in keep if k in h.files})
        # mirror the scalar series into metrics.csv-compatible columns
        with open(os.path.join(args.outdir, "metrics.csv"), "a") as f:
            for i in range(len(h["step"])):
                f.write(f"{h['step'][i]},{h['window'][i]},{h['stage'][i]},"
                        f"{h['tol'][i]},{h['step'][i]},{h['walltime'][i]},nan,"
                        f"{h['loss'][i]},{h['loss_ic'][i]},{h['loss_res'][i]},"
                        f"{h['w_min'][i]},{h['l2_window'][i]},nan,"
                        + ",".join(["nan"] * 6) + "\n")
    print("[BRIDGE] artifacts complete.")


if __name__ == "__main__":
    main()
