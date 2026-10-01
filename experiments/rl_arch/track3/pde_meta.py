"""Таблица задач для переноса между УрЧП (трек 3).

Для каждого УрЧП из реестра считает без обучения, на CPU:
  * размерности входа и выхода, число слагаемых потерь (оператор и границы),
    признаки «зависит от времени» и «обратная задача» — дешёвый описатель задачи;
  * ошибку необученной сети E0 = rmse + brmse и l2re по сидам оценки — масштаб,
    которым нормируются награды при обучении на нескольких УрЧП;
  * обучающий лосс необученной сети (наблюдаемая величина).

Результат: pde_meta.json рядом со скриптом. Его читают offline_rl.py (--subdirs),
online_train_env.py и online_eval_env.py.

    DDEBACKEND=pytorch python experiments/rl_arch/track3/pde_meta.py --pdes all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "pde_meta.json")

# папка буфера в danil-e/rlpinn-ablation-buffers -> имя в реестре УрЧП
SUBDIR_TO_PDE = {
    "burgers1d": "burgers_1d", "burgers2d": "burgers_2d", "grayscott": "grayscott",
    "heat2d_complexgeometry": "heat2d_complexgeometry", "heat2d_longtime": "heat2d_longtime",
    "heat2d_multiscale": "heat2d_multiscale", "heat2d_varyingcoef": "heat2d_varyingcoef",
    "heatinv": "heatinv", "heatnd": "heatnd", "kuramoto_sivashinsky": "kuramoto_sivashinsky",
    "ns2d_backstep": "ns2d_backstep", "ns2d_liddriven": "ns2d_liddriven",
    "ns2d_longtime": "ns2d_longtime", "poisson2d_classic": "poisson2d_classic",
    "poisson2d_manyarea": "poisson2d_manyarea",
    "poisson3d_complexgeometry": "poisson3d_complexgeometry",
    "poisson_boltzmann_2d": "poissonboltzmann2d", "poissoninv": "poissoninv",
    "poissonnd": "poissonnd", "wave1d": "wave1d",
    "wave2d_heterogeneous": "wave2d_heterogeneous", "wave2d_longtime": "wave2d_longtime",
}
PDE_TO_SUBDIR = {v: k for k, v in SUBDIR_TO_PDE.items()}
# волновые задачи заданы на прямоугольнике (x, t) без отдельной временной области,
# поэтому признак времени по типу геометрии у них не срабатывает — задаём явно
TIME_OVERRIDE = {"wave1d": 1, "wave2d_heterogeneous": 1}


def measure(pde_name: str, seeds, hidden="100*5"):
    import numpy as np
    import torch
    import deepxde as dde
    from experiments.chain_eval.pde_registry import build_get_model
    from experiments.chain_eval.pde_names import INVERSE_PDE_NAMES
    from src.utils.callbacks import TesterCallback

    inverse = pde_name in INVERSE_PDE_NAMES
    rows = []
    desc = None
    for seed in seeds:
        dde.config.set_default_float("float32")
        torch.set_default_dtype(torch.float32)
        dde.config.set_random_seed(seed)
        get_model = build_get_model(pde_name, hidden, inverse_plain_fnn=True)
        model, loss_weights = get_model()

        def reinit(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.net.apply(reinit)
        pde = model.pde
        if desc is None:
            geom = model.data.geom
            has_time = isinstance(geom, (dde.geometry.GeometryXTime, dde.geometry.TimeDomain))
            n_loss = int(pde.num_loss)
            n_pde = int(pde.num_pde)
            desc = dict(in_dim=int(pde.input_dim), out_dim=int(pde.output_dim), n_pde=n_pde,
                        n_bnd=n_loss - n_pde,
                        time=int(TIME_OVERRIDE.get(pde_name, bool(has_time))),
                        inverse=int(inverse))
        # шаг Adam с нулевым шагом обучения: веса не меняются, метрики — как в среде
        opt = torch.optim.Adam(model.net.parameters(), lr=0.0)
        model.compile(opt, loss_weights=loss_weights)
        tester = TesterCallback(log_every=1)
        with tempfile.TemporaryDirectory() as td:
            model.train(iterations=1, display_every=1, callbacks=[tester],
                        model_save_path=td, save_model=False)
        rmse = float(getattr(tester, "rmse", float("nan")))
        brmse = float(getattr(tester, "brmse", float("nan")))
        l2 = math.hypot(float(getattr(tester, "l2re", float("nan"))),
                        float(getattr(tester, "bc_l2re", float("nan"))))
        lv = np.asarray(model.train_state.loss_train, dtype=float)
        rows.append(dict(seed=seed, err=rmse + brmse, l2re=l2, loss=float(lv.sum()),
                         loss_oper=float(lv[:desc["n_pde"]].sum()),
                         loss_bnd=float(lv[desc["n_pde"]:].sum())))
        print(f"{pde_name} seed {seed}: err={rmse + brmse:.4g} l2re={l2:.4g} "
              f"loss={lv.sum():.4g}", flush=True)
    med = lambda k: float(np.median([r[k] for r in rows]))
    return dict(desc, subdir=PDE_TO_SUBDIR.get(pde_name), init_err=med("err"),
                init_l2re=med("l2re"), init_loss=med("loss"),
                init_loss_oper=med("loss_oper"), init_loss_bnd=med("loss_bnd"),
                n_seeds=len(rows), per_seed=rows)


def map_stats(table, max_files: int):
    """Статистики карт потерь по задачам (среднее и разброс трёх каналов по всем состояниям
    буфера, кроме нулевых стартовых) — для режима состояния tasknorm. Буферы берутся из
    $RL_BUFFER_DIR или из HF-датасета."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "experiments", "rl_arch"))
    import offline_rl as O
    for name, row in table.items():
        sd = row.get("subdir")
        if not sd:
            continue
        try:
            eps = O.load_episodes(None, sd, max_files=max_files)
            d = O.episodes_to_arrays(eps, chain_fix=True, reward_form="delta", init_err=1.0,
                                     budget=0, verbose=False)
        except Exception as e:
            print(f"{name}: буфер не прочитан ({type(e).__name__}: {e})", flush=True)
            continue
        S = d["S"][d["FIRST"] == 0][:, :3]
        row["map_mean"] = [float(x) for x in S.mean(axis=(0, 2, 3))]
        row["map_std"] = [float(x) for x in S.std(axis=(0, 2, 3))]
        row["map_states"], row["map_files"] = int(len(S)), int(len(eps))
        print(f"{name}: состояний {len(S)}, среднее {np.round(row['map_mean'], 3)}, "
              f"разброс {np.round(row['map_std'], 3)}", flush=True)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdes", default="all")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--map-stats", action="store_true",
                    help="не считать E0, а дописать статистики карт потерь по буферам (режим tasknorm)")
    ap.add_argument("--max-files", type=int, default=12, help="файлов буфера на УрЧП для статистик карт")
    args = ap.parse_args()
    from experiments.chain_eval.pde_names import ALL_PDE_NAMES
    names = ALL_PDE_NAMES if args.pdes == "all" else [x.strip() for x in args.pdes.split(",")]
    table = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if args.map_stats:
        table = map_stats(table, args.max_files)
        json.dump(table, open(args.out, "w"), indent=1, sort_keys=True)
        print(f"записано: {args.out}")
        return
    for name in names:
        try:
            table[name] = measure(name, [int(s) for s in args.seeds.split(",")])
        except Exception as e:                       # одна задача не должна ронять таблицу
            print(f"{name}: не удалось ({type(e).__name__}: {e})", flush=True)
            continue
        json.dump(table, open(args.out, "w"), indent=1, sort_keys=True)
    print(f"записано: {args.out} ({len(table)} УрЧП)")


if __name__ == "__main__":
    main()
