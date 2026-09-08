"""Смок-тест реестра уравнений: каждая спецификация действительно строится.

Проверяет для всех (или выбранных) уравнений реестра:
  * класс задачи импортируется, конструктор принимает kwargs из реестра;
  * опорные данные (ref/*.dat) на месте;
  * build_get_model даёт модель и веса лоссов согласованной длины;
  * сетка действий и порог успеха заполнены осмысленно.

Ничего не обучает и в сеть не ходит: только сборка задачи на CPU.

Запуск из корня репозитория:
    python experiments/agent_ablation/smoke_test_registry.py
    python experiments/agent_ablation/smoke_test_registry.py --tier solvable
    # плюс один чанк Adam на 100 итераций с колбэками RL-цикла (несколько минут на CPU)
    python experiments/agent_ablation/smoke_test_registry.py --tier solvable --train-iters 100
"""
import argparse
import os
import sys
import traceback

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import (  # noqa: E402
    ABLATION_MODES, PDE_SPECS, build_get_model, select,
)


def train_one_chunk(model, loss_weights, iters, out_dir):
    """Один чанк Adam ровно тем путём, каким его гоняет rl_trainer.

    Те же колбэки (Tester/Plot/Loss + ModelSaver), тот же compile с экземпляром
    torch-оптимизатора, та же выжимка взвешенного train loss. Ловит всё, что
    не ловит голая сборка задачи: валидацию на опорных данных (граничные
    точки, IC-маска), рисование, сохранение снимков весов для ландшафта.
    """
    import numpy as np
    import torch
    from src.utils.callbacks import (
        TesterCallback, PlotCallback, LossCallback, ModelSaverCallback,
    )
    from rl_trainer import _extract_weighted_train_loss

    tester = TesterCallback(log_every=iters, verbose=False)
    callbacks = [
        tester,
        PlotCallback(log_every=iters, fast=True),
        LossCallback(verbose=False),
        ModelSaverCallback(total_iterations=iters, n_save_models=2),
    ]
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    model.compile(opt, loss_weights=loss_weights)
    model.optimizer = opt
    tester.reset_trajectory_tracking()
    model.train(iterations=iters, display_every=iters, callbacks=callbacks,
                model_save_path=out_dir, save_model=False)

    train_loss = _extract_weighted_train_loss(model)
    if not np.isfinite(train_loss):
        raise AssertionError(f"взвешенный train loss не конечен: {train_loss}")
    for attr in ("rmse", "brmse", "mse", "bc_mse", "l2re", "bc_l2re", "traj_l2re_min"):
        if not hasattr(tester, attr):
            raise AssertionError(f"у TesterCallback нет атрибута {attr}, который читает rl_trainer")
    if not np.isfinite(tester.l2re):
        raise AssertionError(f"l2re по оператору не конечен: {tester.l2re}")
    saved = len(callbacks[-1].saved_models)
    if saved < 1:
        raise AssertionError("ModelSaverCallback не сохранил ни одного снимка весов")
    bnd = "nan" if not np.isfinite(tester.bc_l2re) else f"{tester.bc_l2re:.3g}"
    return f"loss={train_loss:.3g}, l2re={tester.l2re:.3g}, l2re_bnd={bnd}, снимков={saved}"


def check_spec(spec, train_iters=0):
    get_model = build_get_model(spec)
    model, loss_weights = get_model()

    n_loss = len(loss_weights)
    if n_loss < 1:
        raise AssertionError("пустой вектор весов лоссов")

    grid = spec.optimizers
    for name, cfg in grid.items():
        if len(cfg["lr"]) != 3 or len(cfg["epochs"]) != 3:
            raise AssertionError(f"сетка действий {name} не 3x3: {cfg}")
    if spec.tolerance is not None and not (spec.tolerance > 0):
        raise AssertionError(f"неположительный порог успеха {spec.tolerance}")
    detail = f"лоссов={n_loss}, действий={len(grid) * 9}"

    if train_iters > 0:
        import tempfile
        out_dir = tempfile.mkdtemp(prefix=f"smoke_{spec.key}_")
        detail += ", " + train_one_chunk(model, loss_weights, train_iters, out_dir)
    return detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", action="append", default=None)
    parser.add_argument("--tier", action="append", default=None,
                        choices=["solvable", "borderline", "unsolvable"])
    parser.add_argument("--train-iters", type=int, default=0,
                        help="Прогнать на каждом уравнении один чанк Adam на столько "
                             "итераций с колбэками RL-цикла (0 = только сборка).")
    args = parser.parse_args()

    keys = list(args.pde or [])
    for tier in args.tier or []:
        keys += [k for k in select(tiers=(tier,)) if k not in keys]
    if not keys:
        keys = list(PDE_SPECS)

    print(f"Режимы абляции: {', '.join(ABLATION_MODES)}")
    print(f"Проверяем уравнений: {len(keys)}\n")

    failures = []
    skipped = 0
    for key in keys:
        spec = PDE_SPECS[key]
        if not spec.available:
            skipped += 1
            print(f"  SKIP {key:26s} {spec.note}")
            continue
        try:
            detail = check_spec(spec, args.train_iters)
            tol = "порог не откалиброван" if spec.tolerance is None else f"tol={spec.tolerance:.6g}"
            print(f"  OK   {key:26s} {spec.cls:26s} {detail}, {tol}")
        except Exception as exc:
            print(f"  FAIL {key:26s} {type(exc).__name__}: {exc}")
            failures.append((key, traceback.format_exc()))

    print(f"\nСобралось: {len(keys) - len(failures) - skipped}/{len(keys) - skipped}"
          f"{f' (пропущено нереализованных: {skipped})' if skipped else ''}")
    for key, tb in failures:
        print(f"\n--- {key} ---\n{tb}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
