"""Абляция DQN-стека агента — один раннер на все уравнения бенчмарка.

Режимы (--ablation):
  none            — полный агент (PER + soft-Watkins + trust-region);
  no_per          — без prioritized replay (равномерная выборка, старый буфер);
  no_soft_watkins — без soft-Watkins Q(λ) (старый 1-step Double DQN таргет);
  no_trust_region — без trust-region маски в лоссе.

Уравнение задаётся ключом реестра: --pde burgers1d (список: --list-pdes).
Всё, что различает уравнения (класс задачи, порог успеха, сетка действий,
проект-источник буфера), лежит в pde_registry.py.

Отличие от прошлых per-PDE раннеров (experiments/optimization_multi_pde/
*_ablation_chain.py) только одно содержательное: оффлайн-претрен по умолчанию
500 шагов вместо 50 — с ним политика вычитывается из буфера целиком, агент
заметно меньше шумит и независимо обученные агенты дают сопоставимый
результат (настройка из ветки Saitama32:rlpinn_ablation_optimization,
experiments/agent_ablation/*_agent_ablation.py). Кампания v5 (три уравнения
в rebuttal) шла с 50 шагами, поэтому расширенную кампанию нужно считать
своим префиксом результатов и уравнения v5 пересчитать заново.

Источник буфера (--buffer-src):
  hf    (дефолт) — открытый HF-датасет (--hf-repo/--hf-subdir), COMET_API_KEY
                   для чтения не нужен; датасет наполняется export_buffers.py;
  local          — локальная папка --buffer-dir (формат экспорта);
  comet          — comet-проект уравнения из workspace saitama32.

Логирование результатов: HF-датасет (--hf-results), Comet только по
явному --use-comet.

Примеры:
    python experiments/agent_ablation/ablation_chain.py --pde burgers1d \
        --ablation no_per --max-hours 10.75 --hf-results-prefix runs_kaggle_v6
    python experiments/agent_ablation/ablation_chain.py --pde wave1d --print-config
"""
import os
import sys
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import json
import time
import argparse

import dill
import torch

from experiments.agent_ablation.pde_registry import (
    ABLATION_MODES,
    PDE_SPECS,
    build_get_model,
    get_spec,
)

# Настройка агента из ветки Saitama32:rlpinn_ablation_optimization, при которой
# независимо обученные агенты дают сопоставимый результат. Не трогать без
# причины — на этих числах считается вся расширенная кампания:
#   претрен 500 шагов x 5 батч-апдейтов; спад ε 200 шагов (после претрена ε ~0.09,
#   при 50 было бы уже 0.05); warmup PER 200 апдейтов.
DEFAULT_PRETRAIN_STEPS = 500
DEFAULT_PRETRAIN_ITERS = 5
DEFAULT_EPS_DECAY = 200.0
DEFAULT_WARMUP_UPDATES = 200


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default=None,
                        help="Ключ уравнения из pde_registry (см. --list-pdes).")
    parser.add_argument("--list-pdes", action="store_true",
                        help="Показать реестр уравнений и выйти.")
    parser.add_argument("--print-config", action="store_true",
                        help="Показать собранную конфигурацию запуска и выйти "
                             "(ничего не грузит и не обучает).")
    parser.add_argument("--name", type=str, default=None,
                        help="Имя запуска (по умолчанию <pde>_rl_ablation).")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default=None,
                        help="По умолчанию — из реестра уравнения.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Не используется (осталось от старых раннеров); "
                             "скорость обучения агента — --agent-lr.")
    parser.add_argument("--agent-lr", type=float, default=1e-3,
                        help="Скорость обучения DQN-агента.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs_single")

    parser.add_argument("--ablation", type=str, default="none", choices=list(ABLATION_MODES),
                        help="Какой компонент DQN-стека выключить (none = полный агент).")
    parser.add_argument("--comet-project", type=str, default=None,
                        help="Префикс comet-проекта; итоговое имя <prefix>-<ablation>.")
    parser.add_argument("--buffer-src", type=str, default="hf", choices=["hf", "local", "comet"],
                        help="Откуда грузить буфер: hf (датасет), local (--buffer-dir), comet.")
    parser.add_argument("--buffer-dir", type=str, default=None,
                        help="Папка с экспортированным буфером (для --buffer-src local).")
    parser.add_argument("--hf-repo", type=str, default="danil-e/rlpinn-ablation-buffers",
                        help="HF-датасет с буфером (для --buffer-src hf).")
    parser.add_argument("--hf-subdir", type=str, default=None,
                        help="Подпапка уравнения на HF (по умолчанию — ключ реестра). "
                             "Ею же именуется папка результатов, менять нельзя: "
                             "сломается резюм и агрегация.")
    parser.add_argument("--buffer-proj", type=str, default=None,
                        help="Comet-проект-источник транзишенов (по умолчанию — из реестра).")
    parser.add_argument("--n-exps", type=int, default=200,
                        help="Сколько последних экспериментов источника грузить в буфер.")
    parser.add_argument("--use-comet", action="store_true",
                        help="Логировать в Comet (нужен COMET_API_KEY в .env).")
    parser.add_argument("--hf-results", type=str, default="danil-e/rlpinn-ablation-runs",
                        help="HF-датасет для логов и результатов (нужен HF_TOKEN с правом "
                             "записи). Значение none отключает выгрузку.")
    parser.add_argument("--hf-results-sync-sec", type=int, default=900)
    parser.add_argument("--hf-results-prefix", type=str, default="runs",
                        help="Корневая папка результатов в HF-датасете. Кампании "
                             "разводятся по префиксам: runs, runs_kaggle_v5, runs_kaggle_v6.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Метка запуска в пути на HF (по умолчанию дата-время + hostname).")
    parser.add_argument("--value-type", type=str, default=None,
                        help="Колонка value_type в CSV метрик (по умолчанию — режим абляции).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Пометить строки CSV как smoke_test=True (не зачётный запуск).")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Переопределить порог успеха траектории (по умолчанию — из реестра: "
                             "для l2re это EPS_FACTOR x эталонного L2RE, для loss — tolerance).")
    parser.add_argument("--success-metric", choices=["l2re", "rmse", "loss"], default="l2re",
                        help="Чем меряется успех траектории. l2re — относительная L2-ошибка "
                             "против эталона, критерий статьи (формула 11), по умолчанию; "
                             "rmse — E = RMSE_op + RMSE_bc буквально по формуле (15); "
                             "loss — взвешенный train loss, как считалась кампания v5.")
    parser.add_argument("--max-chain-length", type=int, default=10,
                        help="Kmax из формулы (11) статьи: предел длины цепочки.")
    parser.add_argument("--offline-pretrain-steps", type=int, default=DEFAULT_PRETRAIN_STEPS,
                        help="Шагов оффлайн-претрена агента на буфере до онлайн-траекторий "
                             "(0 = выключить). Каждый шаг = --offline-pretrain-iters апдейтов.")
    parser.add_argument("--offline-pretrain-iters", type=int, default=DEFAULT_PRETRAIN_ITERS,
                        help="Батч-апдейтов за один шаг оффлайн-претрена.")
    parser.add_argument("--eps-decay", type=float, default=DEFAULT_EPS_DECAY,
                        help="Постоянная спада ε-жадности (в шагах агента, претрен "
                             "считается). 200 — как в ветке Saitama32 вместе с претреном "
                             "500; старые раннеры и кампания v5 шли с 50.")
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUP_UPDATES,
                        help="Апдейтов равномерной выборки до пересчёта приоритетов PER "
                             "(200 — как в ветке Saitama32; старые раннеры — 50).")
    parser.add_argument("--refine-steps", type=int, default=0,
                        help="Чанков доводки PINN после достижения порога "
                             "(0 = выключено; RL-семантика не меняется).")
    parser.add_argument("--refine-optimizer", type=str, default="LBFGS")
    parser.add_argument("--refine-lr", type=float, default=0.5)
    parser.add_argument("--refine-epochs", type=int, default=1500)
    parser.add_argument("--eval-only", action="store_true",
                        help="Жадная оценка загруженного агента без обучения "
                             "(буфер не грузится; требует --resume-from).")
    parser.add_argument("--fixed-steps", type=int, default=0,
                        help="Фиксированный бюджет шагов агента на траекторию "
                             "(done=1 игнорируется; для честного сравнения l2re).")
    parser.add_argument("--reset-success-done-to-failure", action="store_true",
                        help="При смене tolerance: сбросить старые done=1 буфера и "
                             "переразметить цепочки по новому порогу.")
    parser.add_argument("--resume-prefix", type=str, default=None,
                        help="Где искать чекпоинт для резюма (по умолчанию — "
                             "--hf-results-prefix).")
    parser.add_argument("--resume-from", type=str, default="auto",
                        help="auto — подхватить последний чекпоинт пары (pde, ablation) "
                             "из HF; none — с нуля; путь — локальный agent_final.pt. "
                             "При резюме претрен пропускается.")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Бюджет времени на запуск, часов. По исчерпании новые "
                             "траектории не начинаются: агент сохраняется, результаты "
                             "уезжают на HF.")
    return parser


def print_registry():
    header = f"{'key':26s} {'tier':11s} {'campaign':9s} {'tolerance':>22s} {'PELINE l2re':>12s}"
    print(header)
    print("-" * len(header))
    for key, spec in PDE_SPECS.items():
        tol = "не откалиброван" if spec.tolerance is None else f"{spec.tolerance:.10g}"
        l2re = "-" if spec.peline_l2re is None else f"{spec.peline_l2re:.2e}"
        print(f"{key:26s} {spec.tier:11s} {spec.campaign:9s} {tol:>22s} {l2re:>12s}")


def resolve_config(args):
    """Сводит CLI и реестр в один словарь настроек запуска."""
    spec = get_spec(args.pde)
    if args.tolerance is not None:
        tolerance = args.tolerance
    elif args.success_metric == "loss":
        tolerance = spec.tolerance
    else:
        tolerance = spec.eps_l2re
    if tolerance is None:
        raise SystemExit(
            f"У уравнения {spec.key} порог успеха не откалиброван. Посчитайте его по "
            f"буферу (experiments/agent_ablation/calibrate_tolerance.py --pde {spec.key}) "
            "и впишите в реестр либо передайте --tolerance явно."
        )
    if spec.tier == "unsolvable":
        l2re = f"L2RE ~ {spec.peline_l2re:.2g}" if spec.peline_l2re else "L2RE порядка 1"
        print(f"⚠️  {spec.key}: PELINE на этом уравнении даёт {l2re}. Абляция "
              "компонентов агента там неинформативна — запуск только для полноты.")
    if not spec.available:
        raise SystemExit(f"{spec.key}: {spec.note or 'класс задачи в этой ветке не реализован'}")
    return {
        "spec": spec,
        "name": args.name or f"{spec.key}_rl_ablation",
        "hidden_layers": args.hidden_layers or spec.hidden_layers,
        "hf_subdir": args.hf_subdir or spec.key,
        "buffer_proj": args.buffer_proj or spec.comet_project,
        "comet_project": args.comet_project or f"rlpinn-{spec.key.replace('_', '-')}-ablation",
        "tolerance": tolerance,
        "optimizers": spec.optimizers,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_pdes:
        print_registry()
        return
    if not args.pde:
        parser.error("нужен --pde <ключ> (список: --list-pdes)")

    cfg = resolve_config(args)
    spec = cfg["spec"]

    if args.print_config:
        print(json.dumps({
            "pde": spec.key,
            "title": spec.title,
            "class": f"{spec.module}.{spec.cls}",
            "kwargs": spec.kwargs,
            "tier": spec.tier,
            "campaign": spec.campaign,
            "ablation": args.ablation,
            "seed": args.seed,
            "hidden_layers": cfg["hidden_layers"],
            "tolerance": cfg["tolerance"],
            "success_metric": args.success_metric,
            "max_chain_length": args.max_chain_length,
            "optimizers": cfg["optimizers"],
            "buffer": {"src": args.buffer_src, "hf_repo": args.hf_repo,
                       "hf_subdir": cfg["hf_subdir"], "comet_project": cfg["buffer_proj"],
                       "n_exps": args.n_exps},
            "results_path": f"{args.hf_results}:{args.hf_results_prefix}/{cfg['hf_subdir']}/{args.ablation}/<run_tag>",
            "offline_pretrain": {"steps": args.offline_pretrain_steps,
                                 "iters": args.offline_pretrain_iters},
            "eps_decay": args.eps_decay,
            "warmup_updates": args.warmup_updates,
            "max_hours": args.max_hours,
            "resume_from": args.resume_from,
        }, ensure_ascii=False, indent=2))
        return

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{cfg['name']}-{args.ablation}")
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
                repo_path=f"{args.hf_results_prefix}/{cfg['hf_subdir']}/{args.ablation}/{run_tag}",
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
        from RL.rl_utils.hf_logger import hf_retry

        # Скачивание буфера — это запрос на каждый файл (их до ~400), а лимит
        # HF (1000 запросов / 5 мин) общий на все параллельные сессии кампании.
        # max_workers=2 сглаживает всплеск, hf_retry переживает 429.
        ds_root = hf_retry(
            snapshot_download,
            repo_id=args.hf_repo,
            repo_type="dataset",
            allow_patterns=[f"{cfg['hf_subdir']}/*"],
            max_workers=2,
            what=f"скачивание буфера {cfg['hf_subdir']}",
        )
        buffer_dir = os.path.join(ds_root, cfg["hf_subdir"])
        if not os.path.isdir(buffer_dir):
            raise SystemExit(
                f"В датасете {args.hf_repo} нет подпапки {cfg['hf_subdir']} — буфер "
                f"ещё не экспортирован. Сделайте это один раз с ключом Comet:\n"
                f"  COMET_API_KEY=<ключ> HF_TOKEN=<токен> python "
                f"experiments/agent_ablation/export_buffers.py --pde {spec.key}"
            )

    if hf_experiment is not None:
        experiment = hf_experiment
    elif args.use_comet:
        from comet_config import start_comet_experiment
        experiment = start_comet_experiment(project_name=f"{cfg['comet_project']}-{args.ablation}")
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
            "description": f"ablation_{args.ablation}_{spec.key}_rl_optimizer",
            "pde": spec.key,
            "pde_title": spec.title,
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

    # --- чекпоинт для продолжения обучения (цепочка сессий) ---
    resume_checkpoint = None
    if args.resume_from == "auto":
        from RL.rl_utils.resume import resolve_resume_checkpoint

        resume_repo = args.hf_results if args.hf_results and args.hf_results.lower() != "none" \
            else "danil-e/rlpinn-ablation-runs"
        # Сид в фильтре обязателен: ячейки одной пары (pde, режим) с разными
        # сидами — это независимо обученные агенты, чужой чекпоинт им нельзя.
        resume_checkpoint = resolve_resume_checkpoint(
            resume_repo, args.resume_prefix or args.hf_results_prefix,
            cfg["hf_subdir"], args.ablation, seed=args.seed)
    elif args.resume_from.lower() != "none":
        resume_checkpoint = {"kind": "final", "path": args.resume_from, "tag": "local"}

    # --- построчный CSV по траекториям (ложится в run_dir -> уезжает на HF) ---
    from RL.rl_utils.trajectory_metrics import TrajectoryMetricsLogger

    trajectory_logger = TrajectoryMetricsLogger(
        csv_path=os.path.join(save_path, "results", "trajectory_metrics.csv"),
        run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pde_name=spec.key,
        value_type=args.value_type or args.ablation,
        seed=args.seed,
        smoke_test=args.smoke_test,
        experiment=experiment,
    )

    get_model = build_get_model(spec, cfg["hidden_layers"])
    get_model_rec = build_get_model(spec, cfg["hidden_layers"])

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
    # уравнения (см. соответствующий *_chain.py в ветке-источнике).
    optimizers = cfg["optimizers"]

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

    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": cfg["tolerance"],
        "success_metric": args.success_metric,
        "success_op_coeff": 1.0,
        "success_bnd_coeff": 0.0,
        "max_chain_length": args.max_chain_length,
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
        "lr": args.agent_lr,
        "exp": experiment,
        "log_key": False,
        "proj_name": cfg["buffer_proj"],
        "ablation": args.ablation,
        "buffer_dir": buffer_dir,
        "trajectory_logger": trajectory_logger,
        "run_control": run_control,
        "resume_checkpoint": resume_checkpoint,
        "offline_pretrain_steps": args.offline_pretrain_steps,
        "offline_pretrain_iters": args.offline_pretrain_iters,
        "eps_decay": args.eps_decay,
        "warmup_updates": args.warmup_updates,
        "refine_steps": args.refine_steps,
        "refine_optimizer": args.refine_optimizer,
        "refine_lr": args.refine_lr,
        "refine_epochs": args.refine_epochs,
        "eval_only": args.eval_only,
        "fixed_steps": args.fixed_steps,
        "reset_success_done_to_failure": args.reset_success_done_to_failure,
    }

    if experiment is not None:
        experiment.log_parameters(rl_agent_params)

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params,
                       AE_train_params, loss_surface_params))
    try:
        train_process_rl(data=data, save_path=save_path, device=args.device,
                         seed=args.seed, rl_agent_params=rl_agent_params)
    except BaseException as exc:
        # Причина падения — и в лог, и в status.json: иначе постфактум не понять
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
        print(f"\n⏱  Время запуска: {run_control.elapsed / 3600:.2f} ч, "
              f"строк в CSV: {trajectory_logger.rows_written}")
        if hf_experiment is not None:
            hf_experiment.end()


if __name__ == "__main__":
    main()
