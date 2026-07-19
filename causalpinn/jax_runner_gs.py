"""JAX engine for the Gray-Scott causal adaptation (winner config from pass 1:
PLAIN encoding). Mirrors causalpinn/jax_runner.py (KS) structurally: modified MLP,
causal weights, tol annealing, 20 time-marching windows over t in [0,200] with
normalized window-local tau in [0,1] and residual scaled by T_w.

Matches the torch port's PlainEncoding exactly: enc = [1, k_t*tau (k_t=10^{0..2}),
x, y] with coords already in [-1,1]. Bridge back to torch via jax_bridge (--case gs).
Run:  python -m causalpinn.jax_runner_gs --outdir ... --ref ref/grayscott.dat
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as onp

import jax
import jax.numpy as np
from jax import grad, jit, lax, random, vmap
from jax.example_libraries import optimizers


B, D, EPS1, EPS2 = 0.04, 0.1, 1e-5, 5e-6   # from src/pde/chaotic.py GrayScott defaults


def modified_MLP_plain(layers, M_t=2, activation=np.tanh):
    k_t = np.power(10.0, np.arange(0, M_t + 1))   # float base -> [1,10,100]

    def input_encoding(t, x, y):
        return np.hstack([1.0, k_t * t, x, y])

    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        return glorot_stddev * random.normal(key, (d_in, d_out)), np.zeros(d_out)

    def init(rng_key):
        U1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            return xavier_init(k1, d_in, d_out)

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)

    def apply(params, t, x, y):
        params, U1, b1, U2, b2 = params
        h = input_encoding(t, x, y)
        U = activation(np.dot(h, U1) + b1)
        V = activation(np.dot(h, U2) + b2)
        for W, b in params[:-1]:
            z = activation(np.dot(h, W) + b)
            h = z * U + (1 - z) * V
        W, b = params[-1]
        return np.dot(h, W) + b            # (2,)

    return init, apply


class GSCausalJax:

    def __init__(self, args, ref_grid, xs, ys, t_star):
        self.args = args
        self.ref = ref_grid                  # (nx, ny, nt, 2)
        self.xs, self.ys, self.t_star = xs, ys, t_star
        self.M_t = 2
        layers = [1 + (self.M_t + 1) + 2] + [128] * 8 + [2]
        self.init, self.apply = modified_MLP_plain(layers, M_t=self.M_t)
        self.n_t, self.n_s = args.n_t, args.n_s
        self.M = np.triu(np.ones((self.n_t, self.n_t)), k=1).T
        self.steps_per_win = (len(t_star) - 1) // args.windows   # 1
        self.T_w = float(t_star[self.steps_per_win] - t_star[0])  # 10.0
        xx, yy = onp.meshgrid(xs, ys, indexing="ij")
        self.ic_x = np.asarray(xx.ravel())
        self.ic_y = np.asarray(yy.ravel())

    def u_fn(self, params, t, x, y):
        return self.apply(params, t, x, y)   # (2,)

    def residual(self, params, t, x, y):
        out = self.u_fn(params, t, x, y)
        u, v = out[0], out[1]
        du_t = grad(lambda t_: self.u_fn(params, t_, x, y)[0])(t)
        dv_t = grad(lambda t_: self.u_fn(params, t_, x, y)[1])(t)
        u_xx = grad(grad(lambda x_: self.u_fn(params, t, x_, y)[0]))(x)
        u_yy = grad(grad(lambda y_: self.u_fn(params, t, x, y_)[0]))(y)
        v_xx = grad(grad(lambda x_: self.u_fn(params, t, x_, y)[1]))(x)
        v_yy = grad(grad(lambda y_: self.u_fn(params, t, x, y_)[1]))(y)
        r_u = du_t - self.T_w * (EPS1 * (u_xx + u_yy) + B * (1 - u) - u * v ** 2)
        r_v = dv_t - self.T_w * (EPS2 * (v_xx + v_yy) - D * v + u * v ** 2)
        return np.stack([r_u, r_v])

    def make_step(self, state0):
        Y_ic = np.asarray(state0)            # (n_ic, 2)
        r_batch = vmap(vmap(self.residual, (None, None, 0, 0)), (None, 0, None, None))
        u_ic_fn = vmap(self.u_fn, (None, None, 0, 0))

        def loss_ics(params):
            pred = u_ic_fn(params, 0.0, self.ic_x, self.ic_y)
            return np.mean((pred - Y_ic) ** 2)

        def loss_fn(params, t_r, x_r, y_r, tol):
            L_0 = self.args.w_ic * loss_ics(params)
            r = r_batch(params, t_r, x_r, y_r)      # (n_t, n_s, 2)
            L_t = np.mean(r ** 2, axis=(1, 2))
            W = lax.stop_gradient(np.exp(-tol * (self.M @ L_t + L_0)))
            return np.mean(W * L_t + L_0), (L_0, L_t, W)

        opt_init, opt_update, get_params = optimizers.adam(
            optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9))

        @jit
        def step(i, opt_state, t_r, x_r, y_r, tol):
            params = get_params(opt_state)
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
                params, t_r, x_r, y_r, tol)
            return opt_update(i, g, opt_state), l, aux

        return step, opt_init, get_params

    def window_l2(self, params, k):
        cols = [k * self.steps_per_win, (k + 1) * self.steps_per_win]
        taus = [0.0, 1.0]
        preds = []
        for tau in taus:
            p = vmap(self.u_fn, (None, None, 0, 0))(params, tau, self.ic_x, self.ic_y)
            preds.append(onp.asarray(p).reshape(len(self.xs), len(self.ys), 2))
        pred = onp.stack(preds, axis=2)                    # (nx, ny, 2cols, 2)
        ref = onp.stack([self.ref[:, :, c, :] for c in cols], axis=2)
        return float(onp.linalg.norm(pred - ref) / onp.linalg.norm(ref))


def params_to_npz(params):
    p, U1, b1, U2, b2 = params
    out = {"U1": U1, "b1": b1, "U2": U2, "b2": b2, "n_layers": onp.array(len(p))}
    for i, (W, b) in enumerate(p):
        out[f"W{i}"] = W
        out[f"bb{i}"] = b
    return {k: onp.asarray(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=str, default="ref/grayscott.dat")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--n-t", type=int, default=32)
    p.add_argument("--n-s", type=int, default=256)
    p.add_argument("--iter-cap", type=int, default=100000)
    p.add_argument("--tol-list", type=str, default="1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--w-ic", type=float, default=1e4)
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--param-snap-every", type=int, default=0,
                   help="save full param snapshots every N iters (0=off)")
    args = p.parse_args()
    tol_list = [float(x) for x in args.tol_list.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    traj = os.path.join(args.outdir, "trajectory")
    causal_dir = os.path.join(args.outdir, "causal")
    os.makedirs(traj, exist_ok=True)
    os.makedirs(causal_dir, exist_ok=True)

    d = onp.loadtxt(args.ref)
    xs, ys, ts = onp.unique(d[:, 0]), onp.unique(d[:, 1]), onp.unique(d[:, 2])
    nx, ny, nt = len(xs), len(ys), len(ts)
    # rows: x fastest, then y, then t -> (nt, ny, nx, comp) -> (nx, ny, nt, comp)
    uv = onp.stack([d[:, 3].reshape(nt, ny, nx), d[:, 4].reshape(nt, ny, nx)], axis=-1)
    ref_grid = onp.moveaxis(uv, 0, 2)
    ref_grid = onp.swapaxes(ref_grid, 0, 1)   # (nx, ny, nt, 2)

    model = GSCausalJax(args, ref_grid, xs, ys, ts)
    n_win = args.windows

    ck_path = os.path.join(args.outdir, "jax_ckpt.pkl")
    if os.path.exists(ck_path):
        with open(ck_path, "rb") as f:
            ck = pickle.load(f)
        print(f"[RESUME] window {ck['window']} stage {ck['stage']}")
    else:
        ic0 = ref_grid[:, :, 0, :].reshape(-1, 2)
        ck = {"window": 0, "stage": 0, "state0": ic0,
              "key": random.PRNGKey(args.seed),
              "opt_packed": None, "win_iter": 0, "walltime_v2": 0.0}

    t0_wall = time.time()
    base_wall = ck.get("walltime_v2", 0.0)

    def elapsed():
        return base_wall + (time.time() - t0_wall)

    def over_budget():
        return (time.time() - t0_wall) > args.max_hours * 3600 - 240

    hist_path = os.path.join(causal_dir, "history_jax.npz")
    hist = {k: [] for k in ["step", "window", "stage", "tol", "w_min", "loss",
                            "loss_ic", "loss_res", "l2_window", "walltime"]}
    hist_vec = {"W": [], "L_t": [], "t_r": []}
    if os.path.exists(hist_path):
        h = onp.load(hist_path)
        for k in hist:
            hist[k] = list(h[k])
        for k in hist_vec:
            hist_vec[k] = list(h[k])

    def flush_hist():
        onp.savez_compressed(hist_path,
                             **{k: onp.asarray(v) for k, v in hist.items()},
                             **{k: onp.asarray(v) for k, v in hist_vec.items()})

    key = ck["key"]
    for k_win in range(ck["window"], n_win):
        step, opt_init, get_params = model.make_step(ck["state0"])
        params = model.init(rng_key=random.PRNGKey(args.seed))
        opt_state = opt_init(params)
        win_iter = 0
        if ck["opt_packed"] is not None and k_win == ck["window"]:
            opt_state = optimizers.pack_optimizer_state(ck["opt_packed"])
            win_iter = ck["win_iter"]

        stage0 = ck["stage"] if k_win == ck["window"] else 0
        for stage in range(stage0, len(tol_list)):
            tol = tol_list[stage]
            print(f"[W{k_win} S{stage}] tol={tol}", flush=True)
            for it in range(args.iter_cap):
                key, k1, k2, k3 = random.split(key, 4)
                t_r = random.uniform(k1, (args.n_t,), minval=0.0, maxval=1.01).sort()
                x_r = random.uniform(k2, (args.n_s,), minval=-1.0, maxval=1.0)
                y_r = random.uniform(k3, (args.n_s,), minval=-1.0, maxval=1.0)
                opt_state, lval, (L0, L_t, W) = step(win_iter, opt_state,
                                                     t_r, x_r, y_r, tol)
                win_iter += 1
                if args.param_snap_every and win_iter % args.param_snap_every == 0:
                    onp.savez(os.path.join(traj, f"w{k_win}_snap_{win_iter}.npz"),
                              **params_to_npz(get_params(opt_state)))
                if (it + 1) % args.log_every == 0 or it == 0:
                    params_now = get_params(opt_state)
                    l2 = model.window_l2(params_now, k_win)
                    w_min = float(W.min())
                    hist["step"].append(win_iter); hist["window"].append(k_win)
                    hist["stage"].append(stage); hist["tol"].append(tol)
                    hist["w_min"].append(w_min); hist["loss"].append(float(lval))
                    hist["loss_ic"].append(float(L0) / args.w_ic)
                    hist["loss_res"].append(float(L_t.mean()))
                    hist["l2_window"].append(l2); hist["walltime"].append(elapsed())
                    print(f"  it {win_iter}: loss {float(lval):.3e} l2_win {l2:.3e} "
                          f"W_min {w_min:.3f}", flush=True)
                    hist_vec["W"].append(onp.asarray(W, dtype=onp.float32))
                    hist_vec["L_t"].append(onp.asarray(L_t, dtype=onp.float32))
                    hist_vec["t_r"].append(onp.asarray(t_r, dtype=onp.float32))
                    if w_min > 0.99:
                        break
                    # periodic in-stage checkpoint: survives hard kills
                    if (it + 1) % 10000 < args.log_every:
                        ck.update(window=k_win, stage=stage, key=key,
                                  opt_packed=optimizers.unpack_optimizer_state(opt_state),
                                  win_iter=win_iter, walltime_v2=elapsed())
                        with open(ck_path + ".tmp", "wb") as f:
                            pickle.dump(ck, f)
                        os.replace(ck_path + ".tmp", ck_path)
                        flush_hist()
                    if over_budget():
                        ck.update(window=k_win, stage=stage, key=key,
                                  opt_packed=optimizers.unpack_optimizer_state(opt_state),
                                  win_iter=win_iter, walltime_v2=elapsed())
                        with open(ck_path, "wb") as f:
                            pickle.dump(ck, f)
                        flush_hist()
                        print("[TIME GUARD] saved; exiting for resume.")
                        sys.exit(0)
            ck.update(window=k_win, stage=stage + 1, key=key,
                      opt_packed=optimizers.unpack_optimizer_state(opt_state),
                      win_iter=win_iter, walltime_v2=elapsed())
            with open(ck_path, "wb") as f:
                pickle.dump(ck, f)
            flush_hist()

        params = get_params(opt_state)
        onp.savez(os.path.join(traj, f"w{k_win}_final_params.npz"),
                  **params_to_npz(params))
        state0 = onp.asarray(vmap(model.u_fn, (None, None, 0, 0))(
            params, 1.0, model.ic_x, model.ic_y))
        l2 = model.window_l2(params, k_win)
        print(f"[WINDOW {k_win}] done at iter {win_iter}, window l2 = {l2:.3e}",
              flush=True)
        ck.update(window=k_win + 1, stage=0, state0=state0, key=key,
                  opt_packed=None, win_iter=0, walltime_v2=elapsed())
        with open(ck_path, "wb") as f:
            pickle.dump(ck, f)
        flush_hist()

    with open(os.path.join(args.outdir, "jax_run_meta.json"), "w") as f:
        json.dump({"engine": "jax-gs-plain", "args": vars(args),
                   "jax": jax.__version__, "windows_done": n_win}, f, indent=2)
    print("[DONE] all GS windows trained (jax engine).")


if __name__ == "__main__":
    main()
