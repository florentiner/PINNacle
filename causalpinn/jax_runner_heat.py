"""JAX engine for the Heat2D_LongTime causal adaptation (trivial-attractor study,
analysis/TRIVIAL_HYPOTHESIS.md). Mirrors jax_runner_gs.py structurally: modified MLP,
causal weights, tol annealing, time-marching windows over t in [0,100] with
window-local tau in [0,1] and residual scaled by T_w.

PDE (src/pde/heat.py Heat2D_LongTime, k=1):
    u_t = 0.001*(u_xx + u_yy) + 5*sin(u^2)*(1 + 2*sin(pi*t/4))*sin(4*pi*x)*sin(2*pi*y)
Trivial exact solution: u ≡ 0 (the source self-gates: sin(0)=0). IC = sin(4pi x)sin(3pi y).
Dirichlet-0 BC is enforced EXACTLY by the ansatz u = 16*x(1-x)*y(1-y)*net(t,x,y)
(representation matched to structure; no BC loss term needed, IC vanishes on the
boundary consistently).

Init modes: --init-mode random (control) | trivial (distill net output to 0 first,
then causal-train from inside the trivial attractor).
Run:  python -m causalpinn.jax_runner_heat --outdir ... --ref ref/heat_longtime.dat
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
from jax import jit, lax, random, vmap
from jax.example_libraries import optimizers


KAPPA = 0.001      # diffusion
AMP = 5.0          # source amplitude
T_TOTAL = 100.0


def modified_MLP_plain(layers, M_t=2, activation=np.tanh):
    k_t = np.power(10.0, np.arange(0, M_t + 1))   # [1,10,100]

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
        return np.dot(h, W) + b            # (1,)

    return init, apply


class HeatCausalJax:

    def __init__(self, args, ref_grid, xs, ys, t_star):
        self.args = args
        self.ref = ref_grid                  # (nx, ny, nt, 1)
        self.xs, self.ys, self.t_star = xs, ys, t_star
        self.M_t = 2
        layers = [1 + (self.M_t + 1) + 2] + [128] * 8 + [1]
        self.init, self.apply = modified_MLP_plain(layers, M_t=self.M_t)
        self.n_t, self.n_s = args.n_t, args.n_s
        self.M = np.triu(np.ones((self.n_t, self.n_t)), k=1).T
        self.steps_per_win = (len(t_star) - 1) // args.windows   # 25 for 20 windows
        self.T_w = float(t_star[self.steps_per_win] - t_star[0])  # 5.0
        # dense IC/handoff grid (ref spatial grid is only 16x12 - too coarse to pin
        # the sin(4pi x)sin(3pi y) mode); l2 eval still uses the ref grid
        gx = onp.linspace(0.0, 1.0, args.ic_grid)
        gy = onp.linspace(0.0, 1.0, args.ic_grid)
        xx, yy = onp.meshgrid(gx, gy, indexing="ij")
        self.ic_x = np.asarray(xx.ravel())
        self.ic_y = np.asarray(yy.ravel())
        # ref-grid coords for evaluation
        rxx, ryy = onp.meshgrid(xs, ys, indexing="ij")
        self.ref_x = np.asarray(rxx.ravel())
        self.ref_y = np.asarray(ryy.ravel())

    def u_fn(self, params, t, x, y):
        # hard Dirichlet-0 ansatz: 16*x(1-x)*y(1-y) in [0,1] on the unit square
        mult = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        return mult * self.apply(params, t, x, y)   # (1,)

    def residual(self, params, tau, x, y, t0):
        from jax.experimental.jet import jet
        out, (_dx1, d2x) = jet(lambda x_: self.u_fn(params, tau, x_, y),
                               (x,), [[1.0, 0.0]])
        _, (_dy1, d2y) = jet(lambda y_: self.u_fn(params, tau, x, y_),
                             (y,), [[1.0, 0.0]])
        _, (d1t,) = jet(lambda t_: self.u_fn(params, t_, x, y),
                        (tau,), [[1.0]])
        u = out[0]
        u_t = d1t[0]
        u_xx, u_yy = d2x[0], d2y[0]
        t_g = t0 + tau * self.T_w
        source = (AMP * np.sin(u ** 2) * (1.0 + 2.0 * np.sin(np.pi * t_g / 4.0))
                  * np.sin(4.0 * np.pi * x) * np.sin(2.0 * np.pi * y))
        r = u_t - self.T_w * (KAPPA * (u_xx + u_yy) + source)
        return np.reshape(r, (1,))

    def make_step(self, state0, t0_win):
        Y_ic = np.asarray(state0)            # (n_ic, 1)
        r_batch = vmap(vmap(self.residual, (None, None, 0, 0, None)),
                       (None, 0, None, None, None))
        u_ic_fn = vmap(self.u_fn, (None, None, 0, 0))

        def loss_ics(params):
            pred = u_ic_fn(params, 0.0, self.ic_x, self.ic_y)
            return np.mean((pred - Y_ic) ** 2)

        def loss_fn(params, t_r, x_r, y_r, tol):
            L_0 = self.args.w_ic * loss_ics(params)
            r = r_batch(params, t_r, x_r, y_r, t0_win)   # (n_t, n_s, 1)
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
        # evaluate at 6 tau-slices aligned with ref columns (steps_per_win divisible by 5)
        sub = self.steps_per_win // 5
        cols = [k * self.steps_per_win + j * sub for j in range(6)]
        taus = [j / 5.0 for j in range(6)]
        preds = []
        for tau in taus:
            p = vmap(self.u_fn, (None, None, 0, 0))(params, tau, self.ref_x, self.ref_y)
            preds.append(onp.asarray(p).reshape(len(self.xs), len(self.ys)))
        pred = onp.stack(preds, axis=2)                    # (nx, ny, 6)
        ref = onp.stack([self.ref[:, :, c, 0] for c in cols], axis=2)
        return float(onp.linalg.norm(pred - ref) / max(onp.linalg.norm(ref), 1e-12))

    def eval_window_on_ref(self, params, k):
        """Full prediction of window k on the ref grid: (nx, ny, steps_per_win+1, 1)."""
        cols = range(k * self.steps_per_win, (k + 1) * self.steps_per_win + 1)
        taus = [(c - k * self.steps_per_win) / self.steps_per_win for c in cols]
        preds = []
        for tau in taus:
            p = vmap(self.u_fn, (None, None, 0, 0))(params, float(tau),
                                                    self.ref_x, self.ref_y)
            preds.append(onp.asarray(p).reshape(len(self.xs), len(self.ys)))
        return onp.stack(preds, axis=2)[..., None]

    def distill_trivial(self, key, iters=8000):
        """Fit the net so u(t,x,y) ~= 0 over the window domain: the trivial solution."""
        params = self.init(rng_key=random.PRNGKey(self.args.seed))
        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_state = opt_init(params)
        u_batch = vmap(self.u_fn, (None, 0, 0, 0))

        def dloss(params, t_r, x_r, y_r):
            return np.mean(u_batch(params, t_r, x_r, y_r) ** 2)

        @jit
        def dstep(i, opt_state, t_r, x_r, y_r):
            l, g = jax.value_and_grad(dloss)(get_params(opt_state), t_r, x_r, y_r)
            return opt_update(i, g, opt_state), l

        for i in range(iters):
            key, k1, k2, k3 = random.split(key, 4)
            t_r = random.uniform(k1, (1024,))
            x_r = random.uniform(k2, (1024,))
            y_r = random.uniform(k3, (1024,))
            opt_state, l = dstep(i, opt_state, t_r, x_r, y_r)
            if (i + 1) % 1000 == 0:
                print(f"  [distill] it {i+1}: mean u^2 = {float(l):.3e}", flush=True)
            if float(l) < 1e-12:
                break
        return get_params(opt_state), key


def params_to_npz(params):
    p, U1, b1, U2, b2 = params
    out = {"U1": U1, "b1": b1, "U2": U2, "b2": b2, "n_layers": onp.array(len(p))}
    for i, (W, b) in enumerate(p):
        out[f"W{i}"] = W
        out[f"bb{i}"] = b
    return {k: onp.asarray(v) for k, v in out.items()}


def load_ref_wide(path):
    """heat_longtime.dat: 192 rows x [x, y, u(t_0..t_500)]; -> (nx,ny,nt,1), sorted axes."""
    d = onp.loadtxt(path, comments="%")
    xs, ys = onp.unique(d[:, 0]), onp.unique(d[:, 1])
    nx, ny, nt = len(xs), len(ys), d.shape[1] - 2
    grid = onp.full((nx, ny, nt), onp.nan)
    ix = onp.searchsorted(xs, d[:, 0])
    iy = onp.searchsorted(ys, d[:, 1])
    assert onp.allclose(xs[ix], d[:, 0]) and onp.allclose(ys[iy], d[:, 1])
    grid[ix, iy, :] = d[:, 2:]
    assert not onp.isnan(grid).any(), "ref rows do not tile the grid"
    t_star = onp.linspace(0.0, T_TOTAL, nt)
    return grid[..., None], xs, ys, t_star


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=str, default="ref/heat_longtime.dat")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=0,
                   help="stop after training this many windows (0 = all)")
    p.add_argument("--n-t", type=int, default=32)
    p.add_argument("--n-s", type=int, default=256)
    p.add_argument("--ic-grid", type=int, default=64)
    p.add_argument("--iter-cap", type=int, default=100000)
    p.add_argument("--tol-list", type=str, default="1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--w-ic", type=float, default=1e4)
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--init-mode", type=str, default="random",
                   choices=["random", "trivial"],
                   help="trivial = distill net to u==0 first, then causal-train")
    p.add_argument("--param-snap-every", type=int, default=0)
    args = p.parse_args()
    tol_list = [float(x) for x in args.tol_list.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    traj = os.path.join(args.outdir, "trajectory")
    causal_dir = os.path.join(args.outdir, "causal")
    arrays_dir = os.path.join(args.outdir, "arrays")
    os.makedirs(traj, exist_ok=True)
    os.makedirs(causal_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

    ref_grid, xs, ys, t_star = load_ref_wide(args.ref)
    model = HeatCausalJax(args, ref_grid, xs, ys, t_star)
    n_win = args.windows if args.max_windows == 0 else min(args.windows,
                                                           args.max_windows)
    onp.save(os.path.join(arrays_dir, "ref.npy"), ref_grid)
    with open(os.path.join(arrays_dir, "grid_meta.json"), "w") as f:
        json.dump({"axes": [xs.tolist(), ys.tolist(), t_star.tolist()],
                   "axis_sizes": [len(xs), len(ys), len(t_star)],
                   "grid_array_shape": [len(xs), len(ys), len(t_star), 1],
                   "pde": "Heat2D_LongTime", "windows": args.windows,
                   "steps_per_win": model.steps_per_win, "T_w": model.T_w}, f)

    ck_path = os.path.join(args.outdir, "jax_ckpt.pkl")
    if os.path.exists(ck_path):
        with open(ck_path, "rb") as f:
            ck = pickle.load(f)
        print(f"[RESUME] window {ck['window']} stage {ck['stage']}")
        key = ck["key"]
        init_params = None
    else:
        key = random.PRNGKey(args.seed)
        init_params = None
        if args.init_mode == "trivial":
            print("[INIT] distilling net to the trivial solution u==0 ...", flush=True)
            init_params, key = model.distill_trivial(key)
            onp.savez(os.path.join(traj, "trivial_init_params.npz"),
                      **params_to_npz(init_params))
            # record how trivial the init is on the ref grid
            l2_triv = model.window_l2(init_params, 0)
            print(f"[INIT] distilled; window-0 l2 of init = {l2_triv:.4f} "
                  f"(1.0 = exactly trivial)", flush=True)
        # analytic first-window IC on the dense grid: sin(4pi x) sin(3pi y)
        icx = onp.asarray(model.ic_x)
        icy = onp.asarray(model.ic_y)
        ic0 = (onp.sin(4 * onp.pi * icx) * onp.sin(3 * onp.pi * icy)).reshape(-1, 1)
        ck = {"window": 0, "stage": 0, "state0": ic0,
              "key": key, "opt_packed": None, "win_iter": 0, "walltime_v2": 0.0,
              "init_mode": args.init_mode}

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
        t0_win = k_win * model.T_w
        step, opt_init, get_params = model.make_step(ck["state0"], t0_win)
        if k_win == 0 and args.init_mode == "trivial" and init_params is not None:
            params = init_params
        else:
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
                x_r = random.uniform(k2, (args.n_s,), minval=0.0, maxval=1.0)
                y_r = random.uniform(k3, (args.n_s,), minval=0.0, maxval=1.0)
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
        pred_w = model.eval_window_on_ref(params, k_win)
        onp.save(os.path.join(arrays_dir, f"pred_w{k_win}_final.npy"), pred_w)
        cols = slice(k_win * model.steps_per_win, (k_win + 1) * model.steps_per_win + 1)
        onp.save(os.path.join(arrays_dir, f"err_w{k_win}_final.npy"),
                 pred_w - ref_grid[:, :, cols, :])
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
        json.dump({"engine": "jax-heat-lt", "args": vars(args),
                   "jax": jax.__version__, "windows_done": n_win}, f, indent=2)
    print(f"[DONE] {n_win} heat windows trained (jax engine).")


if __name__ == "__main__":
    main()
