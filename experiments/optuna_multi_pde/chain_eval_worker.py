"""
Single-seed chain evaluation worker.
Called as a subprocess by kaggle_chain_eval.ipynb for each seed.
Writes one JSON result file per seed.
"""
from __future__ import annotations
import os, sys, json, argparse, time

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PINNACLE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PINNACLE_ROOT)
os.chdir(_PINNACLE_ROOT)

import torch
if (
    not torch.cuda.is_available()
    and getattr(torch.backends, "mps", None)
    and torch.backends.mps.is_available()
    and hasattr(torch, "set_default_device")
):
    torch.set_default_device("mps")

import deepxde as dde


def _load_get_model(pde_name: str, hidden_layers: str = "100*5"):
    """Dynamically import build_get_model from the PDE's optuna script."""
    import importlib.util
    script = os.path.join(_SCRIPT_DIR, f"{pde_name}_optuna.py")
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"No optuna script found for PDE '{pde_name}' at {script}"
        )
    spec = importlib.util.spec_from_file_location("_pde_optuna_mod", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_get_model"):
        raise AttributeError(f"{script} has no build_get_model()")
    return mod.build_get_model(hidden_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde-name",      required=True)
    parser.add_argument("--chain-json",    required=True, help="Path to chain JSON file")
    parser.add_argument("--seed",          type=int, required=True)
    parser.add_argument("--result-json",   required=True, help="Output JSON path for this seed")
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--save-dir",      default="runs_eval")
    args = parser.parse_args()

    with open(args.chain_json) as f:
        chain = json.load(f)

    get_model = _load_get_model(args.pde_name, args.hidden_layers)

    from optuna_trainer import train_chain

    save_path = os.path.join(args.save_dir, f"{args.pde_name}_seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    t0 = time.time()
    mse, rmse, brmse, l2re, bc_l2re = train_chain(
        get_model=get_model,
        chain_config=chain,
        display_every=args.display_every,
        save_path=save_path,
        experiment=None,
        trial_number=args.seed,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    result = {
        "pde_name":  args.pde_name,
        "seed":      args.seed,
        "mse":       float(mse),
        "rmse":      float(rmse),
        "brmse":     float(brmse),
        "l2re":      float(l2re),
        "bc_l2re":   float(bc_l2re),
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
    with open(args.result_json, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"[seed {args.seed}] MSE={mse:.4e}  RMSE={rmse:.4e}  "
        f"L2RE={l2re:.4e}  BC_L2RE={bc_l2re:.4e}  ({elapsed:.0f}s)"
    )


if __name__ == "__main__":
    main()
