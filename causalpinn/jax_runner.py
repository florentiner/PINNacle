"""Original-code JAX engine for chaotic KS causal training (fast path via XLA+jet).

This is the authors' CausalPINNs/KS/chaotic_KS.py adapted minimally:
  - jax>=0.4.3x API (no jax.config import, no torch dependency)
  - forensic logging: full W / L_t vectors, per-window structured params (npz),
    stage-boundary checkpointing + --max-hours time guard for Kaggle chaining
Artifacts (predictions/error landscapes/residual fields) are generated afterwards
by importing the saved params into the PyTorch port: causalpinn/jax_bridge.py.
Run standalone:  python -m causalpinn.jax_runner --outdir ... --ref ref/Kuramoto_Sivashinsky.dat
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
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree


# ---------------- architecture (verbatim reference) ----------------
def modified_MLP(layers, L=1.0, M_t=6, M_x=5, activation=np.tanh):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        return glorot_stddev * random.normal(key, (d_in, d_out)), np.zeros(d_out)

    def input_encoding(t, x):
        w = 2 * np.pi / L
        k_t = np.power(10, np.arange(-M_t // 2, M_t // 2))
        k_x = np.arange(1, M_x + 1)
        return np.hstack([k_t * t, 1, np.cos(k_x * w * x), np.sin(k_x * w * x)])

    def init(rng_key):
        U1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            return xavier_init(k1, d_in, d_out)

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        t, x = inputs[0], inputs[1]
        inputs = input_encoding(t, x)
        U = activation(np.dot(inputs, U1) + b1)
        V = activation(np.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(np.dot(inputs, W) + b)
            inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
        W, b = params[-1]
        return np.dot(inputs, W) + b

    return init, apply


def params_to_npz(params):
    p, U1, b1, U2, b2 = params
    out = {"U1": U1, "b1": b1, "U2": U2, "b2": b2, "n_layers": onp.array(len(p))}
    for i, (W, b) in enumerate(p):
        out[f"W{i}"] = W
        out[f"bb{i}"] = b
    return {k: onp.asarray(v) for k, v in out.items()}


class KSCausalJax:

    def __init__(self, args, usol, t_star, x_star):
        self.args = args
        self.usol, self.t_star, self.x_star = usol, t_star, x_star
        self.M_t, self.M_x = 6, 5
        layers = [2 * self.M_x + self.M_t + 1] + [128] * 8 + [1]
        self.init, self.apply = modified_MLP(layers, L=2 * np.pi,
                                             M_t=self.M_t, M_x=self.M_x)
        self.n_t, self.n_x = args.n_t, args.n_s
        self.M = np.triu(np.ones((self.n_t, self.n_t)), k=1).T
        self.num_step = (len(t_star) - 1) // args.windows      # 25
        self.t1 = t_star[self.num_step] - t_star[0]            # 0.1

        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0)), (None, 0, None))

    def neural_net(self, params, t, x):
        return self.apply(params, np.stack([t, x]))[0]

    def residual_net(self, params, t, x):
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_fn = lambda x_: self.neural_net(params, t, x_)
        _, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x,), [[1.0, 0.0, 0.0, 0.0]])
        return u_t + 100.0 / 16.0 * u * u_x + 100.0 / 16.0 ** 2 * u_xx \
            + 100.0 / 16.0 ** 4 * u_xxxx

    def make_step(self, state0):
        X_ic = self.x_star
        Y_ic = state0.flatten()

        def loss_ics(params):
            u_pred = vmap(self.neural_net, (None, None, 0))(params, 0.0, X_ic)
            return np.mean((Y_ic - u_pred.flatten()) ** 2)

        def residuals_and_weights(params, t_r, x_r, tol):
            L_0 = 1e4 * loss_ics(params)
            r_pred = self.r_pred_fn(params, t_r, x_r)
            L_t = np.mean(r_pred ** 2, axis=1)
            W = lax.stop_gradient(np.exp(-tol * (self.M @ L_t + L_0)))
            return L_0, L_t, W

        def loss_fn(params, t_r, x_r, tol):
            L_0, L_t, W = residuals_and_weights(params, t_r, x_r, tol)
            return np.mean(W * L_t + L_0), (L_0, L_t, W)

        opt_init, opt_update, get_params = optimizers.adam(
            optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9))

        @jit
        def step(i, opt_state, t_r, x_r, tol):
            params = get_params(opt_state)
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, t_r, x_r, tol)
            return opt_update(i, g, opt_state), l, aux

        return step, opt_init, get_params, residuals_and_weights, loss_ics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=str, default="ref/Kuramoto_Sivashinsky.dat")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--n-t", type=int, default=32)
    p.add_argument("--n-s", type=int, default=256)
    p.add_argument("--iter-cap", type=int, default=200000)
    p.add_argument("--tol-list", type=str, default="1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--param-snap-every", type=int, default=0,
                   help="save full param snapshots every N iters (0=off) — "
                        "enables loss-landscape trajectory plots")
    args = p.parse_args()
    tol_list = [float(x) for x in args.tol_list.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    traj = os.path.join(args.outdir, "trajectory")
    causal_dir = os.path.join(args.outdir, "causal")
    os.makedirs(traj, exist_ok=True)
    os.makedirs(causal_dir, exist_ok=True)

    # ref data (x, t, u) pointwise, t fastest -> (512, 251)
    d = onp.loadtxt(args.ref)
    x_star = onp.unique(d[:, 0])
    t_star = onp.unique(d[:, 1])
    usol = d[:, 2].reshape(len(x_star), len(t_star))

    model = KSCausalJax(args, usol, t_star, x_star)
    n_win = args.windows
    num_step = model.num_step

    # resume?
    ck_path = os.path.join(args.outdir, "jax_ckpt.pkl")
    if os.path.exists(ck_path):
        with open(ck_path, "rb") as f:
            ck = pickle.load(f)
        print(f"[RESUME] window {ck['window']} stage {ck['stage']}")
    else:
        ck = {"window": 0, "stage": 0, "state0": usol[:, 0:1],
              "walltime": 0.0, "key": random.PRNGKey(args.seed),
              "opt_packed": None, "win_iter": 0}

    t0_wall = time.time()
    # cumulative walltime across sessions; ck["walltime"] from old versions
    # double-counted within a session, so a dedicated v2 key starts clean
    base_wall = ck.get("walltime_v2", 0.0)

    def elapsed():
        return base_wall + (time.time() - t0_wall)

    def over_budget():
        # the 12h limit is per Kaggle session -> guard on session time only
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
        u_exact = usol[:, k_win * num_step:(k_win + 1) * num_step + 1]
        step, opt_init, get_params, r_and_w, loss_ics = model.make_step(ck["state0"])
        params = model.init(rng_key=random.PRNGKey(args.seed))
        opt_state = opt_init(params)
        win_iter = 0
        if ck["opt_packed"] is not None and k_win == ck["window"]:
            opt_state = optimizers.pack_optimizer_state(ck["opt_packed"])
            win_iter = ck["win_iter"]

        t_eval = np.asarray(t_star[k_win * num_step:(k_win + 1) * num_step + 1]
                            - t_star[k_win * num_step])
        x_eval = np.asarray(x_star)

        stage0 = ck["stage"] if k_win == ck["window"] else 0
        for stage in range(stage0, len(tol_list)):
            tol = tol_list[stage]
            print(f"[W{k_win} S{stage}] tol={tol}")
            for it in range(args.iter_cap):
                key, k1, k2 = random.split(key, 3)
                t_r = random.uniform(k1, (args.n_t,), minval=0.0,
                                     maxval=1.01 * model.t1).sort()
                x_r = random.uniform(k2, (args.n_s,), minval=0.0, maxval=2 * np.pi)
                opt_state, lval, (L0, L_t, W) = step(win_iter, opt_state, t_r, x_r, tol)
                win_iter += 1
                if args.param_snap_every and win_iter % args.param_snap_every == 0:
                    onp.savez(os.path.join(traj, f"w{k_win}_snap_{win_iter}.npz"),
                              **params_to_npz(get_params(opt_state)))
                if (it + 1) % args.log_every == 0 or it == 0:
                    params_now = get_params(opt_state)
                    u_pred = onp.asarray(model.u_pred_fn(params_now, t_eval, x_eval))
                    # u_pred_fn: outer vmap over x, inner over t -> (512, n_t_eval)
                    l2 = float(onp.linalg.norm(u_pred - u_exact)
                               / onp.linalg.norm(u_exact))
                    w_min = float(W.min())
                    hist["step"].append(win_iter); hist["window"].append(k_win)
                    hist["stage"].append(stage); hist["tol"].append(tol)
                    hist["w_min"].append(w_min); hist["loss"].append(float(lval))
                    hist["loss_ic"].append(float(L0) / 1e4)
                    hist["loss_res"].append(float(L_t.mean()))
                    hist["l2_window"].append(l2); hist["walltime"].append(elapsed())
                    hist_vec["W"].append(onp.asarray(W, dtype=onp.float32))
                    hist_vec["L_t"].append(onp.asarray(L_t, dtype=onp.float32))
                    hist_vec["t_r"].append(onp.asarray(t_r, dtype=onp.float32))
                    print(f"  it {win_iter}: loss {float(lval):.3e} "
                          f"l2_win {l2:.3e} W_min {w_min:.3f}", flush=True)
                    if w_min > 0.99:
                        break
                    # periodic in-stage checkpoint: survives hard kills
                    # (quota exhaustion) between graceful time-guard exits
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
            # stage done -> checkpoint at stage boundary
            ck.update(window=k_win, stage=stage + 1, key=key,
                      opt_packed=optimizers.unpack_optimizer_state(opt_state),
                      win_iter=win_iter, walltime_v2=elapsed())
            with open(ck_path, "wb") as f:
                pickle.dump(ck, f)
            flush_hist()

        # window done
        params = get_params(opt_state)
        onp.savez(os.path.join(traj, f"w{k_win}_final_params.npz"),
                  **params_to_npz(params))
        u_pred = onp.asarray(model.u_pred_fn(params, t_eval, x_eval))  # (512, n_t_eval)
        onp.save(os.path.join(traj, f"w{k_win}_u_pred.npy"), u_pred)
        state0 = onp.asarray(
            vmap(model.neural_net, (None, None, 0))(params, model.t1, x_eval))[:, None]
        l2 = float(onp.linalg.norm(u_pred - u_exact) / onp.linalg.norm(u_exact))
        print(f"[WINDOW {k_win}] done at iter {win_iter}, window l2 = {l2:.3e}")
        ck.update(window=k_win + 1, stage=0, state0=state0, key=key,
                  opt_packed=None, win_iter=0, walltime_v2=elapsed())
        with open(ck_path, "wb") as f:
            pickle.dump(ck, f)
        flush_hist()

    with open(os.path.join(args.outdir, "jax_run_meta.json"), "w") as f:
        json.dump({"engine": "jax-original", "args": vars(args),
                   "jax": jax.__version__, "windows_done": n_win}, f, indent=2)
    print("[DONE] all windows trained (jax engine). "
          "Now run causalpinn/jax_bridge.py to generate artifacts.")


if __name__ == "__main__":
    main()
