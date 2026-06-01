"""
benchmark_pde_gpu.py
Benchmark GPU memory and speed for each PDE.

Supports CUDA (Kaggle/server) and MPS (Apple Silicon Mac).
On Mac, MPS uses unified memory shared with system RAM — the reported MB is
equivalent to what a discrete GPU would need on Kaggle.

Runs 3 measurement rounds per PDE (warmup + 3×measure), takes the max peak.

Usage (from PINNacle root):
    python experiments/optuna_multi_pde/benchmark_pde_gpu.py [--pdes all|name1 name2] [--steps 200] [--out pde_gpu_benchmark.csv]
"""

import os, sys, csv, time, threading, argparse, traceback
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import numpy as np
import torch
import deepxde as dde

dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

# ── Device setup ──────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

if DEVICE == "mps" and hasattr(torch, "set_default_device"):
    torch.set_default_device("mps")

# ── PDE imports ───────────────────────────────────────────────────────────────
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.inverse import PoissonInv, HeatInv
from src.pde.ns import NS2D_Classic, NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson1D, Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime

# ── PDE registry ──────────────────────────────────────────────────────────────
HIDDEN = [100] * 5

ALL_PDES = [
    # (name,                        factory,                                use_recommend_net)
    ("burgers_1d",                  lambda: Burgers1D(),                    False),
    ("burgers_2d",                  lambda: Burgers2D(),                    False),
    ("heat2d_varyingcoef",          lambda: Heat2D_VaryingCoef(),           False),
    ("heat2d_multiscale",           lambda: Heat2D_Multiscale(),            False),
    ("heat2d_complexgeometry",      lambda: Heat2D_ComplexGeometry(),       False),
    ("heat2d_longtime",             lambda: Heat2D_LongTime(),              False),
    ("heatnd",                      lambda: HeatND(),                       False),
    ("grayscott",                   lambda: GrayScottEquation(),            False),
    ("kuramoto_sivashinsky",        lambda: KuramotoSivashinskyEquation(),  False),
    ("poissoninv",                  lambda: PoissonInv(),                   True),
    ("heatinv",                     lambda: HeatInv(),                      True),
    ("ns2d_classic",                lambda: NS2D_Classic(),                 False),
    ("ns2d_backstep",               lambda: NS2D_BackStep(),                False),
    ("ns2d_longtime",               lambda: NS2D_LongTime(),                False),
    ("poisson2d_classic",           lambda: Poisson2D_Classic(),            False),
    ("poissonboltzmann2d",          lambda: PoissonBoltzmann2D(),           False),
    ("poisson2d_manyarea",          lambda: Poisson2D_ManyArea(),           False),
    ("poisson3d_complexgeometry",   lambda: Poisson3D_ComplexGeometry(),    False),
    ("poissonnd",                   lambda: PoissonND(),                    False),
    ("wave1d",                      lambda: Wave1D(),                       False),
    ("wave2d_heterogeneous",        lambda: Wave2D_Heterogeneous(),         False),
    ("wave2d_longtime",             lambda: Wave2D_LongTime(),              False),
]

# Only the 17 new ones (not the 5 with existing results)
NEW_17 = [
    "burgers_2d", "heat2d_varyingcoef", "heat2d_multiscale", "heat2d_complexgeometry",
    "heat2d_longtime", "heatnd", "grayscott", "poissoninv", "ns2d_classic",
    "ns2d_backstep", "poisson2d_classic", "poissonboltzmann2d", "poisson2d_manyarea",
    "poissonnd", "wave1d", "wave2d_heterogeneous", "wave2d_longtime",
]

# ── Memory helpers ─────────────────────────────────────────────────────────────

class PeakMemoryMonitor:
    """Poll GPU/MPS memory in a background thread; call .start() / .stop() / .peak_mb."""
    def __init__(self, device, interval=0.05):
        self.device = device
        self.interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            mem = self._sample()
            if mem > self._peak:
                self._peak = mem
            time.sleep(self.interval)

    def _sample(self):
        if self.device == "cuda":
            return torch.cuda.memory_allocated()
        if self.device == "mps":
            return torch.mps.current_allocated_memory()
        return 0

    def start(self):
        self._peak = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self):
        return self._peak / (1024 ** 2)

    def reset(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        self._peak = self._sample()


def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ── Model helpers ──────────────────────────────────────────────────────────────

def build_model(pde_factory, use_recommend_net):
    pde = pde_factory()
    if use_recommend_net:
        net = pde.recommend_net
    else:
        net = dde.nn.FNN([pde.input_dim] + HIDDEN + [pde.output_dim], "tanh", "Glorot normal")
    net = net.float()
    loss_weights = np.ones(pde.num_loss, dtype=np.float32)
    model = pde.create_model(net)
    return model, loss_weights


def run_adam_steps(model, loss_weights, n_steps):
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    model.compile(opt, loss_weights=loss_weights)
    model.optimizer = opt
    model.train(iterations=n_steps, display_every=n_steps + 1)


# ── Benchmark one PDE ──────────────────────────────────────────────────────────

def clear_device_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def set_default_device_safe(dev):
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(dev)


def benchmark_pde(name, pde_factory, use_recommend_net, n_steps, n_rounds, device):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}  (device={device})")
    model = None
    try:
        monitor = PeakMemoryMonitor(device)

        # Build model once; reuse across rounds
        model, loss_weights = build_model(pde_factory, use_recommend_net)

        # Warmup (not measured)
        run_adam_steps(model, loss_weights, n_steps)
        sync_device(device)

        times, peak_mbs = [], []
        for rnd in range(n_rounds):
            monitor.reset()
            monitor.start()
            t0 = time.perf_counter()
            run_adam_steps(model, loss_weights, n_steps)
            sync_device(device)
            elapsed = time.perf_counter() - t0
            monitor.stop()
            times.append(elapsed)
            peak_mbs.append(monitor.peak_mb)
            print(f"  Round {rnd+1}/{n_rounds}: {elapsed:.2f}s  peak={monitor.peak_mb:.1f} MB")

        elapsed_mean = float(np.mean(times))
        peak_mb_max  = float(np.max(peak_mbs))
        steps_per_sec = n_steps / elapsed_mean
        est_steps_5min = int(steps_per_sec * 300)

        print(f"  → Mean time: {elapsed_mean:.2f}s  Max peak GPU mem: {peak_mb_max:.1f} MB")
        print(f"  → {steps_per_sec:.1f} steps/s  |  est ~{est_steps_5min} steps in 5 min")

        return {
            "pde": name, "status": "ok",
            "device": device, "steps": n_steps, "rounds": n_rounds,
            "elapsed_mean_s": round(elapsed_mean, 3),
            "steps_per_sec": round(steps_per_sec, 2),
            "peak_gpu_mb_max": round(peak_mb_max, 1),
            "est_steps_5min": est_steps_5min,
            "error": "",
        }
    except RuntimeError as e:
        err_str = str(e)
        tb = traceback.format_exc()
        # MPS device-mismatch: scipy/cache_tensor creates CPU tensors, can't run on MPS.
        # Switch default device to CPU, retry, then restore.
        if device == "mps" and ("Expected all tensors to be on the same device" in err_str
                                 or "Placeholder storage has not been allocated" in err_str):
            print(f"  Device mismatch on MPS (scipy CPU tensors) — retrying on CPU for timing.")
            if model is not None:
                del model
                model = None
            clear_device_cache(device)
            set_default_device_safe("cpu")
            try:
                model_cpu, lw_cpu = build_model(pde_factory, use_recommend_net)
                run_adam_steps(model_cpu, lw_cpu, n_steps)
                times_cpu = []
                for rnd in range(n_rounds):
                    t0 = time.perf_counter()
                    run_adam_steps(model_cpu, lw_cpu, n_steps)
                    times_cpu.append(time.perf_counter() - t0)
                    print(f"  [CPU] Round {rnd+1}/{n_rounds}: {times_cpu[-1]:.2f}s")
                elapsed_mean = float(np.mean(times_cpu))
                steps_per_sec = n_steps / elapsed_mean
                est_steps_5min = int(steps_per_sec * 300)
                print(f"  → CPU mean: {elapsed_mean:.2f}s  |  {steps_per_sec:.1f} steps/s")
                del model_cpu
                return {
                    "pde": name, "status": "cpu_only",
                    "device": "cpu", "steps": n_steps, "rounds": n_rounds,
                    "elapsed_mean_s": round(elapsed_mean, 3),
                    "steps_per_sec": round(steps_per_sec, 2),
                    "peak_gpu_mb_max": 0.0,
                    "est_steps_5min": est_steps_5min,
                    "error": "MPS device mismatch (scipy CPU tensors); ran on CPU",
                }
            except Exception as e2:
                print(f"  CPU retry also failed: {e2}")
                print(traceback.format_exc())
                return {
                    "pde": name, "status": "error",
                    "device": device, "steps": n_steps, "rounds": n_rounds,
                    "elapsed_mean_s": "", "steps_per_sec": "",
                    "peak_gpu_mb_max": "", "est_steps_5min": "",
                    "error": f"MPS fail: {err_str[:80]}; CPU retry: {str(e2)[:80]}",
                }
            finally:
                set_default_device_safe(device)
        print(f"  ERROR: {e}")
        print(tb)
        return {
            "pde": name, "status": "error",
            "device": device, "steps": n_steps, "rounds": n_rounds,
            "elapsed_mean_s": "", "steps_per_sec": "",
            "peak_gpu_mb_max": "", "est_steps_5min": "",
            "error": err_str,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ERROR: {e}")
        print(tb)
        return {
            "pde": name, "status": "error",
            "device": device, "steps": n_steps, "rounds": n_rounds,
            "elapsed_mean_s": "", "steps_per_sec": "",
            "peak_gpu_mb_max": "", "est_steps_5min": "",
            "error": str(e),
        }
    finally:
        if model is not None:
            del model
        clear_device_cache(device)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU memory and speed per PDE (CUDA or MPS).")
    parser.add_argument("--pdes", nargs="+", default=["new17"],
                        help="PDE names, 'all' for all 22, 'new17' for the 17 new ones (default).")
    parser.add_argument("--steps", type=int, default=200,
                        help="Adam steps per measurement round (default 200).")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Measurement rounds per PDE after warmup (default 3); peak = max across rounds.")
    parser.add_argument("--out", type=str, default="pde_gpu_benchmark.csv")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Total GPU memory: {total_mb:.0f} MB")
    elif DEVICE == "mps":
        print("Apple MPS — unified memory (shared with system RAM).")
        print("Memory reported = amount used by ML workload; equivalent to discrete GPU VRAM on Kaggle.")
    else:
        print("No GPU found — running on CPU (memory stats will be 0).")

    # Resolve PDE list
    selected = args.pdes
    if "all" in selected:
        pde_list = ALL_PDES
    elif "new17" in selected:
        name_set = set(NEW_17)
        pde_list = [p for p in ALL_PDES if p[0] in name_set]
    else:
        name_set = set(selected)
        pde_list = [p for p in ALL_PDES if p[0] in name_set]
        missing = name_set - {p[0] for p in pde_list}
        if missing:
            print(f"WARNING: Unknown PDE names: {missing}")

    print(f"\nRunning benchmark on {len(pde_list)} PDE(s): {args.steps} steps × {args.rounds} rounds + 1 warmup\n")

    results = []
    for name, factory, recommend in pde_list:
        row = benchmark_pde(name, factory, recommend, args.steps, args.rounds, DEVICE)
        results.append(row)

    # Write CSV
    fieldnames = ["pde", "status", "device", "steps", "rounds",
                  "elapsed_mean_s", "steps_per_sec", "peak_gpu_mb_max", "est_steps_5min", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'PDE':<32} {'Status':>7} {'Peak MB':>8} {'steps/s':>9} {'est 5min':>10}")
    print("-" * 70)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['pde']:<32} {'OK':>7} {r['peak_gpu_mb_max']:>8} {r['steps_per_sec']:>9} {r['est_steps_5min']:>10}")
        elif r["status"] == "cpu_only":
            print(f"{r['pde']:<32} {'CPU':>7} {'N/A':>8} {r['steps_per_sec']:>9} {r['est_steps_5min']:>10}  (MPS incompat)")
        else:
            short_err = str(r["error"])[:40]
            print(f"{r['pde']:<32} {'ERROR':>7}  {short_err}")

    ok = sum(1 for r in results if r["status"] in ("ok", "cpu_only"))
    print(f"\n{ok}/{len(results)} PDEs ran  |  Results saved to: {args.out}")
    if DEVICE == "mps":
        print("Note: MPS peak_gpu_mb_max = unified memory used; use this as VRAM estimate for Kaggle GPU sizing.")


if __name__ == "__main__":
    main()
