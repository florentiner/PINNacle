"""Абляция DQN-стека агента на Poisson3D_ComplexGeometry.

Режимы (--ablation):
  none            — полный агент (PER + soft-Watkins + trust-region);
  no_per          — без prioritized replay (равномерная выборка, старый буфер);
  no_soft_watkins — без soft-Watkins Q(λ) (старый 1-step Double DQN таргет);
  no_trust_region — без trust-region маски в лоссе.

Источник буфера (--buffer-src):
  hf    (дефолт) — открытый HF-датасет (--hf-repo/--hf-subdir), COMET_API_KEY
                   для чтения не нужен; датасет наполняется скриптом
                   export_buffer_transitions.py;
  local          — локальная папка --buffer-dir (формат экспорта);
  comet          — как раньше: comet-проект --buffer-proj из workspace
                   saitama32 (нужен ключ с доступом к нему).

Логирование результатов (--hf-results / --no-comet / по умолчанию):
  --hf-results <репо> — метрики, параметры, лог запуска и снапшоты агента
                        уезжают в HF-датасет (нужен HF_TOKEN с правом записи);
                        Comet не используется вообще. Рекомендуемый режим для
                        запуска на удалённом GPU-сервере;
  --no-comet          — то же самое, но без выгрузки: всё только локально;
  по умолчанию        — свой comet-проект <--comet-project>-<--ablation>
                        (ключ/воркспейс из .env).
"""
import os
import sys
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import time
import argparse

import dill
import numpy as np
import torch

PDE_NAME = "poisson3d_complexgeometry"


def build_get_model_poisson3d_complexgeometry(hidden_layers: str, **pde_kwargs):
    def get_model():
        import deepxde as dde
        from src.pde.poisson import Poisson3D_ComplexGeometry
        from src.utils.args import parse_hidden_layers

        pde = Poisson3D_ComplexGeometry(**pde_kwargs)

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            if t in ("boundary", "initial"):
                loss_weights[i] = 100.0
            elif t == "pde":
                loss_weights[i] = 1.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="poisson3d_complexgeometry_rl_ablation")
    parser.add_argument("--datapath", type=str, default="ref/poisson_3d.dat")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs_single")

    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_per", "no_soft_watkins", "no_trust_region"],
        help="Какой компонент DQN-стека выключить (none = полный агент).",
    )
    parser.add_argument(
        "--comet-project",
        type=str,
        default="rlpinn-poisson3d-complexgeometry-ablation",
        help="Префикс таргетного comet-проекта; итоговое имя <prefix>-<ablation>.",
    )
    parser.add_argument(
        "--buffer-src",
        type=str,
        default="hf",
        choices=["hf", "local", "comet"],
        help="Откуда грузить буфер: hf (открытый датасет), local (--buffer-dir), comet.",
    )
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default=None,
        help="Папка с экспортированным буфером (для --buffer-src local).",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="danil-e/rlpinn-ablation-buffers",
        help="HF-датасет с буфером (для --buffer-src hf).",
    )
    parser.add_argument(
        "--hf-subdir",
        type=str,
        default="poisson3d_complexgeometry",
        help="Подпапка PDE внутри HF-датасета.",
    )
    parser.add_argument(
        "--buffer-proj",
        type=str,
        default="rlpinn-poisson3d-complexgeometry-tolerance",
        help="Comet-проект-источник транзишенов (для --buffer-src comet).",
    )
    parser.add_argument("--n-exps", type=int, default=200,
                        help="Сколько последних экспериментов источника грузить в буфер.")
    parser.add_argument(
        "--use-comet",
        action="store_true",
        help="Логировать в Comet (нужен COMET_API_KEY в .env). По умолчанию "
             "Comet не используется вообще.",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Устарело и ничего не делает: Comet и так выключен по умолчанию.",
    )
    parser.add_argument(
        "--hf-results",
        type=str,
        default="danil-e/rlpinn-ablation-runs",
        help="Отдельный HF-датасет для логов и результатов (не тот, где буфер). "
             "Нужен HF_TOKEN с правом записи; без токена всё останется локально. "
             "Значение none полностью отключает выгрузку. Подразумевает --no-comet.",
    )
    parser.add_argument(
        "--hf-results-sync-sec",
        type=int,
        default=900,
        help="Как часто синхронизировать результаты на HF, секунд.",
    )
    parser.add_argument(
        "--hf-results-prefix",
        type=str,
        default="runs",
        help="Корневая папка результатов в HF-датасете. Отдельные кампании "
             "(сервер / Kaggle / проверки) разводите по разным префиксам, "
             "например runs, runs_kaggle, runs_checks.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Метка запуска в пути на HF (по умолчанию дата-время + hostname).",
    )
    parser.add_argument(
        "--value-type",
        type=str,
        default=None,
        help="Колонка value_type в CSV метрик (по умолчанию — режим абляции).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Пометить строки CSV как smoke_test=True (проверочный, не зачётный запуск).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Переопределить порог успеха траектории (по умолчанию 0.824311852455139 "
             "из таблицы tolerance-проектов).",
    )
    parser.add_argument(
        "--offline-pretrain-steps",
        type=int,
        default=50,
        help="Шагов оффлайн-претрена агента на буфере до онлайн-траекторий "
             "(0 = выключить). Каждый шаг = --offline-pretrain-iters батч-апдейтов.",
    )
    parser.add_argument(
        "--offline-pretrain-iters",
        type=int,
        default=5,
        help="Батч-апдейтов за один шаг оффлайн-претрена.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Бюджет времени на запуск, часов. По исчерпании новые траектории "
             "не начинаются: модель агента сохраняется, результаты уезжают на HF. "
             "Одна траектория идёт 1–2 часа, так что без бюджета запуск на "
             "n-trajectories=1000 не закончится никогда.",
    )

    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}-{args.ablation}")
    os.makedirs(save_path, exist_ok=True)

    # --- логирование результатов на HF (вместо Comet) ---
    hf_experiment = None
    if args.hf_results and args.hf_results.lower() != "none":
        import socket
        from RL.rl_utils.hf_logger import HFExperiment, tee_stdout

        run_tag = args.run_tag or f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{socket.gethostname()}"
        tee_stdout(os.path.join(save_path, "logs", "log.txt"))
        if not os.getenv("HF_TOKEN"):
            print("⚠️  HF_TOKEN не задан — результаты останутся только локально "
                  f"({save_path}). Для выгрузки на HF: export HF_TOKEN=<токен>.")
        else:
            hf_experiment = HFExperiment(
                repo_id=args.hf_results,
                repo_path=f"{args.hf_results_prefix}/{args.hf_subdir}/{args.ablation}/{run_tag}",
                run_dir=save_path,
                sync_every_sec=args.hf_results_sync_sec,
            )

    # --- источник буфера ---
    buffer_dir = None
    if args.buffer_src == "local":
        if not args.buffer_dir:
            raise SystemExit("--buffer-src local требует --buffer-dir")
        buffer_dir = args.buffer_dir
    elif args.buffer_src == "hf":
        from huggingface_hub import snapshot_download

        ds_root = snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            allow_patterns=[f"{args.hf_subdir}/*"],
        )
        buffer_dir = os.path.join(ds_root, args.hf_subdir)
        if not os.path.isdir(buffer_dir):
            raise SystemExit(
                f"В датасете {args.hf_repo} нет подпапки {args.hf_subdir} — "
                "буфер ещё не экспортирован (см. export_buffer_transitions.py)."
            )

    # Comet — только по явному --use-comet. Отключённая выгрузка на HF означает
    # «пишем локально», а не «идём в Comet».
    if hf_experiment is not None:
        experiment = hf_experiment
    elif args.use_comet:
        from comet_config import start_comet_experiment
        experiment = start_comet_experiment(
            project_name=f"{args.comet_project}-{args.ablation}"
        )
    else:
        experiment = None
        print(f"[local] Comet не используется; результаты только в {save_path}.")

    import deepxde as dde  # noqa: F401  (инициализация backend до rl_trainer)
    from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
    from rl_trainer import train_process_rl

    if experiment is not None:
        experiment.log_parameters({
            "param": "v_1",
            "reward_function": "v_2",
            "description": f"ablation_{args.ablation}_poisson3d_complexgeometry_rl_optimizer",
            "ablation": args.ablation,
            "buffer_src": args.buffer_src,
            "seed": args.seed,
        })

    # --- контроль запуска: бюджет времени, мягкая остановка, статус ---
    from RL.rl_utils.run_control import RunControl

    run_control = RunControl(
        max_seconds=args.max_hours * 3600 if args.max_hours else None,
        status_path=os.path.join(save_path, "results", "status.json"),
    )
    run_control.install_signal_handlers()
    run_control.write_status("running", "запуск стартовал")

    # --- построчный CSV по траекториям (ложится в run_dir -> уезжает на HF) ---
    from RL.rl_utils.trajectory_metrics import TrajectoryMetricsLogger

    trajectory_logger = TrajectoryMetricsLogger(
        csv_path=os.path.join(save_path, "results", "trajectory_metrics.csv"),
        run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pde_name=PDE_NAME,
        value_type=args.value_type or args.ablation,
        seed=args.seed,
        smoke_test=args.smoke_test,
        experiment=experiment,
    )

    pde_kwargs = dict(datapath=args.datapath)
    get_model = build_get_model_poisson3d_complexgeometry(args.hidden_layers, **pde_kwargs)
    get_model_rec = build_get_model_poisson3d_complexgeometry(args.hidden_layers, **pde_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [
            TesterCallback(log_every=args.log_every),
            PlotCallback(log_every=args.plot_every, fast=True),
            LossCallback(verbose=True),
        ],
        "n_trajectories": args.n_trajectories,
        "n_save_models": args.n_save_models,
        "operator_coeff": 1,
        "bnd_coeff": 1,
    }

    # Сетка действий должна совпадать с той, на которой фармились транзишены
    # poisson3d_complexgeometry (см. experiments/Poisson/poisson3d_complexgeometry_chain.py)
    optimizers = {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": device,
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False,
        "log_key": True,
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": "",
        "dde_pde_model": get_model_rec,
    }

    # Параметры загрузки буфера — из таблицы tolerance-проектов:
    # 'rlpinn-poisson3d-complexgeometry-tolerance': n_exps=200,
    # tolerance=0.824311852455139, prev_tol=0.0, use_tol=False, new_tol=True,
    # use_log_state=False
    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": args.tolerance if args.tolerance is not None else 0.824311852455139,
        "use_tol": False,
        "new_tol": True,
        "prev_tol": 0.0,
        "n_exps": args.n_exps,
        "stuck_threshold": 10,
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit": 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "agent_min_buffer": 32,
        "agent_update_iters": 5,
        "lr": 1e-3,
        "exp": experiment,
        "log_key": False,
        "proj_name": args.buffer_proj,
        "ablation": args.ablation,
        "buffer_dir": buffer_dir,
        "trajectory_logger": trajectory_logger,
        "run_control": run_control,
        "offline_pretrain_steps": args.offline_pretrain_steps,
        "offline_pretrain_iters": args.offline_pretrain_iters,
    }

    if experiment is not None:
        experiment.log_parameters(rl_agent_params)

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params))
    try:
        train_process_rl(data=data, save_path=save_path, device=args.device, seed=args.seed, rl_agent_params=rl_agent_params)
    except BaseException as exc:
        # Пишем причину в лог И в status.json: прошлый прогон оборвался
        # молча, и понять постфактум было нечего.
        import traceback as _tb
        _tb.print_exc()
        run_control.write_status(
            "failed",
            f"{type(exc).__name__}: {exc}",
            extra={"traceback": _tb.format_exc()[-4000:]},
        )
        raise
    else:
        run_control.write_status(
            "finished",
            run_control.stop_reason or "все траектории пройдены",
            extra={"trajectory_rows": trajectory_logger.rows_written},
        )
    finally:
        # Финальная выгрузка логов/результатов на HF даже при падении или Ctrl-C
        print(f"\n⏱  Время запуска: {run_control.elapsed / 3600:.2f} ч, "
              f"строк в CSV: {trajectory_logger.rows_written}")
        if hf_experiment is not None:
            hf_experiment.end()


if __name__ == "__main__":
    main()
