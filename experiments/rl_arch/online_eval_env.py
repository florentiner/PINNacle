#!/usr/bin/env python
"""
Online L2RE evaluation of an offline-trained agent using the AUTHORS' state
construction (autoencoder latent loss surface), not a reconstruction.

State pipeline (verified against branch rlpinn_pde_tolerance):
  1. every chunk saves n_save_models copies of the solver net (ModelSaverCallback)
  2. VisualizationModel trains an autoencoder over the flat concat of all
     state_dict tensors of those copies (input_dim = 40801 for FNN 100*5;
     layers_AE[0]/[2] are dead code, only the 125-wide hidden matters)
  3. PlotLossSurface decodes a 26x26 grid of the 2D latent (x_range
     [-1.25, 1.25, 25] -> step 0.1 -> 26 points) back to weights and evaluates
     PDE/BC losses at each point with a SECOND model factory
  4. save_equation_loss_surface(log_key=True) applies sign(x)*log1p(|x|)
  5. delta channel (env logic, replicated here):
         d = total_now - total_prev
         delta = sign(d)*log1p(|d|); delta /= max|delta|; clamp(-1, 1)
     channel order: loss_total, loss_oper, loss_bnd, delta

Action space (verified in RL/rl_algorithms.py: i2opt = dict key order):
    0 Adam   lr [1e-2, 1e-3, 1e-4]  epochs [100, 1000, 2500]
    1 LBFGS  lr [1, 5e-1, 1e-1]     epochs [100, 500, 1000]   (max_iter=10!)
    2 PSO    lr [0, 1e-3, 1e-4]     epochs [100, 200, 300]
    index = opt*9 + lr*3 + epochs

Comet/gym/dill are NOT imported: only landscape_visualization._aux is used and
the env's delta/reward logic is replicated locally.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("DDEBACKEND", "pytorch")

import torch  # noqa: E402
import deepxde as dde  # noqa: E402

OUT_REPO = "danil-e/pinnacle-optuna-db"
GRID_RANGE = [-1.25, 1.25, 25]

ACTION_TABLE = []
for _opt, _lrs, _eps in [
    ("Adam", [1e-2, 1e-3, 1e-4], [100, 1000, 2500]),
    ("LBFGS", [1.0, 5e-1, 1e-1], [100, 500, 1000]),
    ("PSO", [0.0, 1e-3, 1e-4], [100, 200, 300]),
]:
    for _lr in _lrs:
        for _ep in _eps:
            ACTION_TABLE.append((_opt, _lr, _ep))
assert len(ACTION_TABLE) == 27
# Действия вне пространства агента (трек 3, потолок среды): доступны открытым цепочкам,
# разведке и бандиту, агенту и маскам — нет. SOAP с теми же настройками, что в
# experiments/chain_eval (betas 0.95/0.95, пересчёт базиса каждые 2 шага)
EXTRA_ACTIONS = [("SOAP", 3e-3, 100), ("SOAP", 3e-3, 500), ("SOAP", 3e-3, 1000),
                 ("SOAP", 3e-3, 2500), ("SOAP", 1e-3, 1000), ("SOAP", 1e-2, 1000)]
ALL_ACTIONS = ACTION_TABLE + EXTRA_ACTIONS

AE_MODEL_PARAMS = dict(
    mode="NN", num_of_layers=3, layers_AE=[991, 125, 15], num_models=None,
    from_last=False, prefix="model-", every_nth=1, grid_step=0.1, d_max_latent=2,
    anchor_mode="circle", rec_weight=10000.0, anchor_weight=0.0, lastzero_weight=0.0,
    polars_weight=0.0, wellspacedtrajectory_weight=0.0, gridscaling_weight=0.0,
)
LOSS_TYPES = ["loss_total", "loss_oper", "loss_bnd"]


def reuse_optimizer(cache, opt_name, lr, net, keep):
    """Оптимизатор для очередного действия. При keep и том же семействе оптимизатора на той же
    сети возвращается прежний объект с обновлённым шагом: моменты Adam, история кривизны
    L-BFGS и предобусловливатель SOAP переживают границу действий (переход с сохранением
    состояния, как в AOS, arXiv 2608.01997). Иначе создаётся новый, как раньше."""
    if (keep and opt_name != "PSO" and cache.get("name") == opt_name
            and cache.get("net") is net and cache.get("obj") is not None):
        opt = cache["obj"]
        for g in opt.param_groups:
            g["lr"] = lr
        return opt
    opt = build_optimizer(opt_name, lr, net)
    cache.update(name=opt_name, net=net, obj=(opt if opt_name != "PSO" else None))
    return opt


def build_optimizer(opt_name, lr, net):
    """Mirrors rl_trainer._build_torch_optimizer (note LBFGS max_iter=10)."""
    from deepxde.optimizers.config import set_PSO_options

    if opt_name == "Adam":
        return torch.optim.Adam(net.parameters(), lr=lr)
    if opt_name == "LBFGS":
        return torch.optim.LBFGS(net.parameters(), lr=lr,
                                 line_search_fn="strong_wolfe", max_iter=10)
    if opt_name == "PSO":
        set_PSO_options(lr=lr)
        return "PSO"
    if opt_name == "SOAP":
        from experiments.chain_eval.vendor.soap import SOAP
        return SOAP(net.parameters(), lr=lr, betas=(0.95, 0.95), weight_decay=0.0,
                    precondition_frequency=2)
    raise ValueError(opt_name)


SCALAR_CH = 5


def add_scalar_ctx(state, step, k_max, spent, budget, last_action, err):
    """Дубликат из online_train_env: контекст постоянными каналами. Копия, чтобы
    оценка не импортировала обучающий скрипт целиком."""
    h, w = state.shape[1], state.shape[2]
    if last_action is None or last_action < 0:
        opt_i = ep_i = -1.0
    else:
        opt_i = (last_action // 9) / 2.0
        ep_i = (last_action % 3) / 2.0
    vals = [min(1.0, step / max(1, k_max)), min(1.0, spent / max(1, budget)),
            opt_i, ep_i,
            0.0 if err is None else float(np.clip(np.log10(max(err, 1e-8)) / 3.0 + 1.0, -1, 1))]
    planes = np.stack([np.full((h, w), v, dtype=np.float32) for v in vals])
    return np.concatenate([state, planes], axis=0)


def load_agent(model_file):
    from offline_rl import QNet

    if model_file and os.path.exists(model_file):
        path = model_file
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OUT_REPO, model_file, repo_type="dataset")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["variant"] == "cnx_smdp":       # другой класс сети (advanced_agents)
        from advanced_agents import SmdpAgent
        net = SmdpAgent(torch.device("cpu"))
        net.net.load_state_dict(ckpt["state_dict"])
        net.net.eval()
    else:
        # число входных каналов выводим из самого чекпоинта: агенты, обученные
        # со скалярным контекстом, ждут 4+5 каналов вместо четырёх
        in_ch = 4
        for k, v in ckpt["state_dict"].items():
            if hasattr(v, "dim") and v.dim() == 4 and v.shape[2] <= 7:
                in_ch = v.shape[1]
                break
        if ckpt["variant"] == "stat_dqn":
            # у кодировщика на статистиках свёрток нет: 8 признаков на канал
            in_ch = ckpt["state_dict"]["0.mlp.0.weight"].shape[1] // 8
        # число квантилей/бинов тоже выводим из чекпоинта: у HL-Gauss их 51,
        # у квантильных вариантов 32 — иначе голова не совпадёт по форме
        nq = 32
        hw = ckpt["state_dict"].get("1.weight")
        if hw is not None and hw.dim() == 2 and hw.shape[0] % 27 == 0:
            nq = max(1, hw.shape[0] // 27)
        net = QNet(ckpt["variant"], torch.device("cpu"), n_quantiles=nq, in_ch=in_ch)
        net.model.load_state_dict(ckpt["state_dict"])
        net.model.eval()
    hg = ckpt.get("hl_gauss")
    if hg:
        from advanced_agents import HLGauss
        net._hlg = HLGauss(v_min=hg[0], v_max=hg[1], n_bins=int(hg[2]))
        print(f"агент обучен с HL-Gauss: скаляр Q по центрам {int(hg[2])} бинов", flush=True)
    # условия обучения, которые оценка обязана воспроизвести (новые чекпоинты);
    # у старых чекпоинтов meta нет — действуют прежние константы
    net._meta = ckpt.get("meta") or {}
    return net, ckpt["mean"], ckpt["std"], ckpt["variant"]


def parse_action_spec(tok):
    """'LBFGS:0.5:1000' -> индекс действия в ACTION_TABLE."""
    o, lr, ep = tok.strip().split(":")
    for i, (on, l, e) in enumerate(ALL_ACTIONS):
        if on.lower() == o.lower() and abs(l - float(lr)) < 1e-12 and e == int(ep):
            return i
    raise ValueError(f"нет действия {tok!r}: в таблице 27 действий агента и "
                     f"{len(EXTRA_ACTIONS)} дополнительных (SOAP)")


def parse_chain(spec):
    return [parse_action_spec(t) for t in spec.split(",") if t.strip()] if spec else []


def parse_mask(spec):
    """True = действие разрешено. Токены: 'pso', 'adam:0.01', 'pso:0:300', номер."""
    allowed = np.ones(27, dtype=bool)
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            allowed[int(tok)] = False
            continue
        parts = tok.split(":")
        hit = False
        for i, (on, l, e) in enumerate(ACTION_TABLE):
            if on.lower() != parts[0].lower():
                continue
            if len(parts) > 1 and abs(l - float(parts[1])) > 1e-12:
                continue
            if len(parts) > 2 and e != int(parts[2]):
                continue
            allowed[i] = False
            hit = True
        if not hit:
            raise ValueError(f"маска: токен {tok!r} не совпал ни с одним действием")
    if not allowed.any():
        raise ValueError("маска запретила все 27 действий")
    return allowed


NOOP_ACTIONS = (18, 19, 20)     # PSO с lr=0: в 39-85% применений ошибку не меняет


Q_POLICY = "auto"   # auto | mean | cvar — как сводить квантили к скаляру


def q_values(agent, state, mean, std, variant):
    """Вектор Q по 27 действиям (numpy) — для маски, ансамбля и надбавки проводника."""
    dev = next(agent.model.parameters()).device
    x = torch.as_tensor((state[None] - mean) / std, device=dev).float()
    with torch.no_grad():
        hlg = getattr(agent, "_hlg", None)
        if hlg is not None:
            q = hlg.to_scalar(agent.q_online(x))
        elif Q_POLICY == "cvar":
            q = agent.q_cvar(x)
        elif Q_POLICY == "mean":
            q = agent.q_scalar(x)
        else:
            q = (agent.q_cvar(x) if variant in ("cnn_qrdqn", "cnx_cql_qr")
                 else agent.q_scalar(x))
    return q[0].detach().float().cpu().numpy()


def ensemble_action(agents, state, allowed, how="vote"):
    """Решение комитета замороженных агентов. vote — большинство голосов argmax
    (ничья в пользу более раннего в списке); mean — argmax среднего Q после
    стандартизации каждого агента (шкалы Q у агентов разные)."""
    qs = []
    for ag, mean, std, variant in agents:
        q = q_values(ag, state, mean, std, variant).astype(np.float64)
        q[~allowed] = -np.inf
        qs.append(q)
    if how == "mean":
        z = []
        for q in qs:
            f = q[np.isfinite(q)]
            z.append(np.where(np.isfinite(q), (q - f.mean()) / (f.std() + 1e-8), -np.inf))
        return int(np.argmax(np.mean(z, axis=0)))
    votes = [int(np.argmax(q)) for q in qs]
    best, cnt = votes[0], 0
    for v in votes:                      # порядок списка разрешает ничьи
        c = votes.count(v)
        if c > cnt:
            best, cnt = v, c
    return best


def pick_action(agent, state, mean, std, variant):
    # deepxde на GPU ставит default device = cuda, поэтому вход надо создавать
    # на том же устройстве, где лежат веса агента
    dev = next(agent.model.parameters()).device
    x = torch.as_tensor((state[None] - mean) / std, device=dev).float()
    with torch.no_grad():
        hlg = getattr(agent, "_hlg", None)
        if hlg is not None:
            q = hlg.to_scalar(agent.q_online(x))
        elif Q_POLICY == "cvar":
            q = agent.q_cvar(x)
        elif Q_POLICY == "mean":
            q = agent.q_scalar(x)
        else:
            # auto — прежнее поведение: у cnn_qrdqn и cnx_cql_qr политика по CVaR,
            # у остальных (включая cnx_qrdqn) по среднему. Оставлено ради
            # воспроизводимости уже снятых прогонов; для новых задавайте явно
            q = (agent.q_cvar(x) if variant in ("cnn_qrdqn", "cnx_cql_qr")
                 else agent.q_scalar(x))
    return int(q.argmax(1).item())


def landscape_stall_prob(state, trig):
    """Ландшафтный триггер коллеги в сильнейшей форме: логистическая модель,
    обученная на здоровом буфере (landscape_trigger.json, AUC 0.657)."""
    m = state[0].astype(np.float64)
    g = np.gradient(m)
    f = np.array([m.std(),
                  np.mean(np.abs(g[0])) + np.mean(np.abs(g[1])),
                  np.abs(np.gradient(g[0])[0] + np.gradient(g[1])[1]).mean(),
                  m[13, 13] - m.min(),
                  np.percentile(m, 95) - np.percentile(m, 5),
                  m[13, 13], m.mean()])
    z = (f - np.array(trig["mean"])) / np.array(trig["std"])
    return float(1.0 / (1.0 + np.exp(-(z @ np.array(trig["coef"]) + trig["intercept"]))))


class BoostedNet(torch.nn.Module):
    """u = base + eps*boost, база заморожена (arXiv 2307.08934)."""

    def __init__(self, base, boost, eps):
        super().__init__()
        self.base, self.boost, self.eps = base, boost, float(eps)
        self.regularizer = None
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.eps * self.boost(x)


CKPT_DIR = "rl_arch/online_env_ckpt"


def save_ckpt(name, payload):
    """Состояние прогона (веса PINN + цепочка + карты) — чтобы следующий кернел
    продолжил с того же места: сессия Kaggle живёт 12 ч, а бюджет 31000 эпох при
    мелких действиях агента требует больше."""
    local = f"ckpt_{name}.pt"
    torch.save(payload, local)
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    from huggingface_hub import upload_file
    # HF режет на 128 коммитов в час НА РЕПОЗИТОРИЙ, а кернелов десятки, поэтому
    # ждём долго: потерянный чекпоинт стоит целой сессии, потерянные минуты — нет
    for attempt in range(6):
        try:
            upload_file(path_or_fileobj=local, path_in_repo=f"{CKPT_DIR}/{name}.pt",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"ckpt {name}")
            return
        except Exception as e:
            print(f"ckpt upload retry {attempt}: {e}", flush=True)
            time.sleep(min(600, 20 * 2 ** attempt))


def load_ckpt(name):
    local = f"ckpt_{name}.pt"
    if os.path.exists(local):
        return torch.load(local, map_location="cpu", weights_only=False)
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(OUT_REPO, f"{CKPT_DIR}/{name}.pt", repo_type="dataset")
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"чекпоинта нет ({type(e).__name__}) — старт с нуля", flush=True)
        return None


def build_state(raw, prev_raw):
    """Replicates EnvRLOptimizer.step delta logic + channel order."""
    tot = raw["loss_total"].detach().float().cpu()
    op = raw["loss_oper"].detach().float().cpu()
    bn = raw["loss_bnd"].detach().float().cpu()
    if prev_raw is None:
        delta = torch.zeros_like(tot)
    else:
        d = tot - prev_raw["loss_total"].detach().float().cpu()
        delta = torch.sign(d) * torch.log1p(torch.abs(d))
        delta = delta / (delta.abs().max() + 1e-6)
        delta = delta.clamp(-1, 1)
    return torch.stack([tot, op, bn, delta]).numpy().astype(np.float32)


def run_seed(seed, args, progress_cb=None):
    from experiments.chain_eval.pde_registry import build_get_model
    from src.utils.callbacks import TesterCallback, ModelSaverCallback
    from landscape_visualization._aux.visualization_model import VisualizationModel
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    from landscape_visualization._aux.early_stopping_plot import EarlyStopping

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(args, "float64", False):
        # диагностика уровня среды: застой L-BFGS в float32 часто артефакт точности
        # линейного поиска (в контроле 1D float64 давал ошибку на 1-2 порядка ниже)
        dde.config.set_default_float("float64")
        torch.set_default_dtype(torch.float64)
    else:
        dde.config.set_default_float("float32")
        torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # two independent factories: one trains, one is overwritten 676x per state
    get_model = build_get_model(args.pde, args.hidden_layers, inverse_plain_fnn=args.plain_fnn)
    get_model_rec = build_get_model(args.pde, args.hidden_layers, inverse_plain_fnn=args.plain_fnn)

    model, loss_weights = get_model()

    def reinit(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.net.apply(reinit)
    if getattr(args, "float64", False):
        model.net.double()       # реестр приводит сеть к float32 — возвращаем двойную точность

    agent = mean = std = variant = None
    agents = []
    needs_agent = (args.policy == "agent"
                   or (args.policy == "script" and args.script_tail == "agent"))
    if needs_agent:
        files = [f.strip() for f in (args.model_files or args.model_file or "").split(",") if f.strip()]
        if not files:
            sys.exit("нужен --model-file (или --model-files для комитета)")
        for f in files:
            agents.append(load_agent(f))
        agent, mean, std, variant = agents[0]
        if Q_POLICY == "cvar" and variant not in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
            sys.exit(f"--q-policy cvar требует квантильного варианта, а в чекпоинте "
                     f"{variant}: у него один выход на действие, брать хвост распределения не из чего")
    meta = getattr(agent, "_meta", {}) if agent is not None else {}
    allowed = parse_mask(args.mask)
    for i in meta.get("mask", []):
        allowed[int(i)] = False          # маска обучения действует и на оценке
    script = parse_chain(args.script)
    guide = parse_chain(args.guide)
    ctx_kmax = int(meta.get("ctx_kmax", 10))
    ctx_budget = int(meta.get("ctx_budget", 31000))
    ctx_err = meta.get("ctx_err", "l2re")   # старые чекпоинты: прежнее поведение оценки
    stop_reason = "budget"
    # ---- трек 3: режим состояния, учёт времени, история по шагам ----
    from offline_rl import apply_state_mode, loss_state, pde_desc, pde_meta
    state_mode = meta.get("state_mode", "full") if args.state_mode == "auto" else args.state_mode
    mapless = state_mode in ("blind", "loss")
    if args.no_state and needs_agent and not mapless:
        sys.exit(f"--no-state: агент обучен в режиме состояния {state_mode!r}, ему нужны карты")
    if args.state_compare and state_mode == "loss":
        sys.exit("--state-compare не определён для режима loss: агент обучен на лоссах, не на картах")
    if state_mode == "rebuild" and not args.state_compare:
        sys.exit("--state-mode rebuild имеет смысл только с --state-compare: агент действует по первой "
                 "сборке карт, а записывается, что он выбрал бы по второй")
    build_maps = (not args.no_state) and (not mapless or args.state_compare or not needs_agent)
    guard_fb = parse_action_spec(args.guard_fallback) if args.guard_rollback else None
    guard_log, guard_spent = [], 0
    opt_cache = {}                   # --keep-opt: оптимизатор прошлого действия
    # агент, обученный в среде без сбросов оптимизатора, оценивается в ней же
    keep_opt = bool(args.keep_opt or meta.get("keep_opt"))
    if keep_opt and not args.keep_opt:
        print("агент обучен с --keep-opt: состояние оптимизатора сохраняется между действиями", flush=True)
    prev_raw2 = None                 # режим rebuild: предыдущая карта второй сборки
    stale_maps = None                # --state-every: последняя построенная карта
    # перенос между УрЧП: описатель задачи и масштаб ошибки — как при обучении
    desc_vals = pde_desc(args.pde) if meta.get("pde_ctx") else None
    err_scale = float(pde_meta()[args.pde]["init_err"]) if meta.get("err_norm") == "init" else 1.0
    prev_loss_tot = None

    def add_desc(st):
        if desc_vals is None:
            return st
        return np.concatenate([st, np.stack([np.full(st.shape[1:], v, dtype=np.float32)
                                             for v in desc_vals])], axis=0)
    agree, hist = [], []
    true_state = None
    t_ae_tot = t_srf_tot = t_opt_tot = 0.0
    bandit_arms = parse_chain(args.bandit_arms) if args.policy == "bandit" else []
    bd = dict(n=np.zeros(len(ALL_ACTIONS)), x=np.zeros(len(ALL_ACTIONS)), prev_loss=None)
    scout_set = parse_chain(args.scout_set) if args.policy == "scout" else []
    scout_commit = parse_chain(args.scout_commit) if args.policy == "scout" else []
    scout_log, scout_spent = [], 0

    def train_loss_now():
        return float(np.asarray(model.train_state.loss_train, dtype=float).sum())

    def greedy_on(st):
        """Жадное действие агента (или комитета) по состоянию st с учётом маски."""
        if len(agents) > 1:
            return ensemble_action(agents, st, allowed, args.ensemble)
        q = q_values(agent, st, mean, std, variant).astype(np.float64)
        q[~allowed] = -np.inf
        return int(np.argmax(q))

    vm = VisualizationModel(device=dev, path_to_plot_model=None,
                            path_to_trajectories=None, **AE_MODEL_PARAMS)

    rng = np.random.default_rng(seed * 613 + 7)
    save_dir = os.path.join(args.save_dir, f"{args.pde}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # initial state: zero maps (rl_trainer.zero_state)
    state = np.zeros((4, 26, 26), dtype=np.float32)
    want_ctx = bool(agent is not None and
                    (next(agent.model.parameters()).shape[1] > 4
                     if variant != "stat_dqn" else
                     next(agent.model.parameters()).shape[1] > 32))
    last_a = None
    if want_ctx:
        state = add_desc(add_scalar_ctx(state, 0, ctx_kmax, 0, ctx_budget, None,
                                        None if ctx_err == "none" else 1.0))  # нормировки обучения
    prev_raw = None
    trig = (json.load(open(os.path.join(SCRIPT_DIR, "landscape_trigger.json")))
            if args.boost_trigger.startswith("landscape") else None)
    boosted, loss_hist, p_hist = False, [], []
    boost_layers, boost_eps = None, None
    spent, chain, t0 = 0, [], time.time()
    last_prog = 0.0
    rmse = brmse = l2re_op = l2re_bnd = float("inf")

    ckpt_name = getattr(args, "_ckpt_name", None)
    ck = load_ckpt(ckpt_name) if (args.resume and ckpt_name) else None
    if ck is not None:
        if ck.get("boosted"):
            boost = dde.nn.FNN(ck["boost_layers"], "tanh", "Glorot normal").float()
            pde_ref = getattr(model, "pde", None)
            model = dde.Model(model.data, BoostedNet(model.net, boost, ck["boost_eps"]))
            model.pde = pde_ref
            boosted, boost_layers, boost_eps = True, ck["boost_layers"], ck["boost_eps"]
        model.net.load_state_dict(ck["net"])
        spent, chain, state = ck["spent"], ck["chain"], ck["state"]
        prev_raw = ck["prev_raw"]
        loss_hist, p_hist = ck["loss_hist"], ck["p_hist"]
        rmse, brmse, l2re_op, l2re_bnd = ck["metrics"]
        t3 = ck.get("t3") or {}
        prev_loss_tot = t3.get("prev_loss_tot")
        guard_log, guard_spent = t3.get("guard_log", []), t3.get("guard_spent", 0)
        prev_raw2, stale_maps = t3.get("prev_raw2"), t3.get("stale_maps")
        agree, hist = t3.get("agree", []), t3.get("hist", [])
        true_state = t3.get("true_state")
        t_ae_tot, t_srf_tot, t_opt_tot = t3.get("times", (0.0, 0.0, 0.0))
        bd = t3.get("bd", bd)
        scout_log, scout_spent = t3.get("scout_log", []), t3.get("scout_spent", 0)
        print(f"[seed {seed}] докатка: spent={spent}/{args.budget}, шагов уже {len(chain)}",
              flush=True)

    def dump_ckpt():
        if not ckpt_name:
            return
        save_ckpt(ckpt_name, dict(
            net={k: v.detach().cpu() for k, v in model.net.state_dict().items()},
            spent=spent, chain=chain, state=state,
            prev_raw=({k: v.detach().cpu() for k, v in prev_raw.items()}
                      if prev_raw is not None else None),
            loss_hist=loss_hist, p_hist=p_hist, boosted=boosted,
            boost_layers=boost_layers, boost_eps=boost_eps,
            rule_st=rule_st,
            t3=dict(agree=agree, hist=hist, true_state=true_state, prev_loss_tot=prev_loss_tot,
                    guard_log=guard_log, guard_spent=guard_spent, stale_maps=stale_maps,
                    prev_raw2=({k: v.detach().cpu() for k, v in prev_raw2.items()}
                               if prev_raw2 is not None else None),
                    times=(t_ae_tot, t_srf_tot, t_opt_tot), bd=bd,
                    scout_log=scout_log, scout_spent=scout_spent),
            metrics=(rmse, brmse, l2re_op, l2re_bnd)))

    rule_st = dict(kicks=0, stalled=False, last_loss=None)
    if ck is not None and ck.get("rule_st"):
        rule_st = ck["rule_st"]
    while spent < args.budget:
        use_agent = args.policy == "agent"
        if args.policy == "script":
            if len(chain) < len(script):
                a = script[len(chain)]
            elif args.script_tail == "repeat":
                a = script[-1]
            elif args.script_tail == "agent":
                use_agent = True
            else:
                stop_reason = "script_end"
                break
        elif args.policy == "rule":
            # эвристика «всплески L-BFGS, при застое — толчок Adam»: наблюдает только
            # обучающий лосс (истинной ошибки на оценке нет). Застой = лосс после
            # всплеска изменился меньше чем на rule_tol относительно
            if rule_st["stalled"]:
                if rule_st["kicks"] >= args.rule_max_kicks:
                    stop_reason = "rule_stop"
                    break
                a = parse_action_spec(args.rule_kick)
                rule_st["kicks"] += 1
            else:
                a = parse_action_spec(args.rule_burst)
        elif args.policy == "fixed":
            a = args.fixed_action
        elif args.policy == "random":
            # без маски — прежний поток случайных чисел (воспроизводимость старых прогонов);
            # с маской выбор идёт только среди разрешённых действий
            a = (int(rng.integers(0, 27)) if allowed.all()
                 else int(rng.choice(np.where(allowed)[0])))
        elif args.policy == "bandit":
            # UCB с забыванием по рукам-действиям: награда руки — убыль логарифма
            # обучающего лосса на тысячу эпох. Учится внутри одного прогона, без
            # предобучения и без карт: наблюдается только обучающий лосс
            untried = [x for x in bandit_arms if bd["n"][x] == 0]
            if untried:
                a = untried[0]
            else:
                tot = max(1e-9, float(sum(bd["n"][x] for x in bandit_arms)))
                ucb = {x: bd["x"][x] / bd["n"][x]
                          + args.bandit_c * math.sqrt(math.log(max(tot, 1.0 + 1e-9)) / bd["n"][x])
                       for x in bandit_arms}
                a = max(bandit_arms, key=lambda x: ucb[x])
        elif args.policy == "scout":
            # разведка: каждое действие из набора пробуется коротко из одной и той же
            # точки, веса откатываются; побеждает проба с наименьшим обучающим лоссом,
            # затем исполняется её «длинное» действие. Эпохи проб идут в счёт бюджета
            need = sum(ALL_ACTIONS[x][2] for x in scout_set)
            if args.budget - spent <= need:
                a = scout_commit[scout_log[-1]["win"]] if scout_log else scout_commit[0]
            else:
                snap = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
                losses = []
                for x in scout_set:
                    on_, lr_, ep_ = ALL_ACTIONS[x]
                    model.net.load_state_dict(snap)
                    model.compile(build_optimizer(on_, lr_, model.net), loss_weights=loss_weights)
                    t_op = time.time()
                    model.train(iterations=ep_, display_every=max(ep_, 1), callbacks=[],
                                model_save_path=save_dir, save_model=False)
                    t_opt_tot += time.time() - t_op
                    lv = train_loss_now()
                    losses.append(lv if np.isfinite(lv) else float("inf"))
                    spent += ep_
                    scout_spent += ep_
                # пробы шли на своих оптимизаторах, веса вернулись к снимку: оптимизатор прошлого
                # исполненного действия (opt_cache) по-прежнему соответствует этим весам
                model.net.load_state_dict(snap)
                win = int(np.argmin(losses))
                scout_log.append(dict(step=len(chain) + 1, losses=[float(v) for v in losses], win=win))
                a = scout_commit[win]
                print(f"[seed {seed}] разведка: лоссы {['%.3e' % v for v in losses]} -> "
                      f"{ALL_ACTIONS[a][0]} lr={ALL_ACTIONS[a][1]} ep={ALL_ACTIONS[a][2]}", flush=True)
        if use_agent:
            if len(agents) > 1:
                a = ensemble_action(agents, state, allowed, args.ensemble)
            elif allowed.all() and not guide:
                a = pick_action(agent, state, mean, std, variant)     # прежний путь
            else:
                q = q_values(agent, state, mean, std, variant).astype(np.float64)
                q[~allowed] = -np.inf
                a = int(np.argmax(q))
                g = guide[len(chain)] if len(chain) < len(guide) else None
                if g is not None and a != g and q[g] + args.guide_bonus >= q[a]:
                    a = g       # от эвристики отходим только при явном выигрыше по Q
            if args.state_compare and true_state is not None:
                agree.append([len(chain) + 1, int(a), int(greedy_on(true_state))])
            if (args.stop_on_noop and a in NOOP_ACTIONS
                    and any(c[0] != "PSO" for c in chain)):
                # «стоп» засчитываем только после настоящего обучения: лидер трека 1
                # начинает цепочку с пустого шага PSO, и остановка на нём оставила бы
                # необученную сеть
                stop_reason = "noop"
                break
        loss_before = bd["prev_loss"]
        snap = None
        if args.guard_rollback and use_agent and loss_before is not None:
            # страж: снимок весов до действия агента
            snap = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
        while True:
            opt_name, lr, epochs = ALL_ACTIONS[a]
            epochs = min(epochs, args.budget - spent)
            optimizer = reuse_optimizer(opt_cache, opt_name, lr, model.net, keep_opt)
            model.compile(optimizer, loss_weights=loss_weights)
            tester = TesterCallback(log_every=args.display_every)
            saver = ModelSaverCallback(total_iterations=epochs, n_save_models=args.n_save_models)
            t_op = time.time()
            model.train(iterations=epochs, display_every=args.display_every,
                        callbacks=[tester, saver], model_save_path=save_dir, save_model=False)
            t_opt_tot += time.time() - t_op
            spent += epochs
            cur_loss = train_loss_now()
            worse = (not np.isfinite(cur_loss)) or (loss_before is not None
                                                     and cur_loss > args.guard_rollback * loss_before)
            if snap is not None and worse and spent < args.budget and a != guard_fb:
                # действие агента ухудшило обучающий лосс: веса откатываются, вместо него
                # исполняется запасное действие. Потраченные эпохи остаются в счёте бюджета
                model.net.load_state_dict(snap)
                opt_cache.clear()        # после отката состояние оптимизатора не соответствует весам
                guard_log.append([len(chain) + 1, int(a), float(loss_before),
                                  float(cur_loss) if np.isfinite(cur_loss) else None])
                guard_spent += epochs
                print(f"[seed {seed}] страж: {opt_name} lr={lr} ep={epochs} поднял лосс "
                      f"{loss_before:.3e} -> {cur_loss:.3e}, откат и запасное действие", flush=True)
                a, snap = guard_fb, None
                continue
            break
        chain.append([opt_name, lr, epochs])
        last_a = a if a < 27 else -1      # действие вне пространства агента в контексте неизвестно
        if args.policy == "bandit":
            if loss_before is not None and np.isfinite(cur_loss) and cur_loss > 0 and loss_before > 0:
                r_b = float(np.clip((math.log10(loss_before) - math.log10(cur_loss))
                                    / max(1, epochs) * 1000.0, -5.0, 5.0))
                bd["n"] *= args.bandit_discount
                bd["x"] *= args.bandit_discount
                bd["n"][a] += 1.0
                bd["x"][a] += r_b
            elif loss_before is None:
                bd["n"][a] += 1e-3        # первая проба: базы для награды ещё нет
        bd["prev_loss"] = cur_loss if np.isfinite(cur_loss) else bd["prev_loss"]

        rmse = float(getattr(tester, "rmse", float("inf")))
        brmse = float(getattr(tester, "brmse", float("inf")))
        l2re_op = float(getattr(tester, "l2re", float("inf")))
        l2re_bnd = float(getattr(tester, "bc_l2re", float("inf")))
        print(f"[seed {seed}] step {len(chain)}: {opt_name} lr={lr} ep={epochs} "
              f"spent={spent}/{args.budget} l2re={math.hypot(l2re_op, l2re_bnd):.4e}", flush=True)
        # история по шагам: нужна для выбора «лучшего из k» по обучающему лоссу и для
        # имитации гонок между инициализациями без новых запусков
        hist.append([int(spent), float(cur_loss) if np.isfinite(cur_loss) else None,
                     float(math.hypot(l2re_op, l2re_bnd)), round(time.time() - t0, 1)])
        # строку прогресса шлём не чаще, чем раз в progress_every секунд: она нужна
        # только чтобы не потерять результат при срезе сессии, а десятки кернелов
        # пишут в один репозиторий с лимитом 128 коммитов в час
        want_prog = (spent >= args.budget
                     or time.time() - last_prog >= args.progress_every)
        if progress_cb is not None and want_prog:
            last_prog = time.time()
            progress_cb(dict(seed=seed, policy=args.policy, pde=args.pde, partial=True,
                             l2re=math.hypot(l2re_op, l2re_bnd), l2re_op=l2re_op,
                             l2re_bnd=l2re_bnd, rmse=rmse, brmse=brmse, spent=spent,
                             budget=args.budget, n_steps=len(chain), chain=chain,
                             boost_trigger=args.boost_trigger, boosted=boosted,
                             elapsed_s=round(time.time() - t0, 1)))

        if not (np.isfinite(rmse) or np.isfinite(brmse)):
            print(f"[seed {seed}] non-finite metrics — stopping (done=-1)", flush=True)
            break
        if spent >= args.budget:
            break

        # ---- решение о бустинге: проверка идеи статьи с ландшафтным триггером ----
        loss_hist.append(float(np.asarray(model.train_state.loss_train, dtype=float).sum()))
        if args.policy == "rule":
            cur_loss = loss_hist[-1]
            was_kick = rule_st["stalled"]
            prev_loss = rule_st["last_loss"]
            if was_kick:
                rule_st["stalled"] = False            # после толчка снова всплеск L-BFGS
            else:
                moved = (prev_loss is None
                         or abs(cur_loss - prev_loss) > args.rule_tol * max(abs(prev_loss), 1e-12))
                if moved and prev_loss is not None and cur_loss < prev_loss:
                    rule_st["kicks"] = 0              # прогресс есть — счётчик толчков сброшен
                rule_st["stalled"] = not moved
                rule_st["last_loss"] = cur_loss
        if args.boost_trigger != "none" and not boosted and spent < args.budget:
            fire = False
            if trig is not None and len(chain) > 1:
                p_stall = landscape_stall_prob(state, trig)
                p_hist.append(p_stall)
                if args.boost_trigger == "landscape":
                    fire = p_stall > args.boost_threshold
                else:  # landscape_peak: момент, который ландшафт считает самым застойным
                    fire = (len(p_hist) > args.boost_warmup
                            and p_stall >= max(p_hist[:-1]))
                print(f"[seed {seed}] ландшафтный триггер: P(застой)={p_stall:.3f} "
                      f"(max={max(p_hist):.3f})", flush=True)
            elif args.boost_trigger == "midpoint":
                fire = spent >= args.budget // 2
            elif args.boost_trigger == "plateau" and len(loss_hist) >= 4:
                r4 = loss_hist[-4:]
                fire = (max(r4) - min(r4)) < args.plateau_rel * abs(r4[-1] + 1e-12)
            if fire:
                eps = float(np.sqrt(max(float(np.asarray(model.train_state.loss_train,
                                                         dtype=float)[0]), 1e-30)))
                layers = [model.net.linears[0].in_features, 64, 64, 64,
                          model.net.linears[-1].out_features]
                boost = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
                pde_ref = getattr(model, "pde", None)
                model = dde.Model(model.data, BoostedNet(model.net, boost, eps))
                model.pde = pde_ref
                boosted, boost_layers, boost_eps = True, layers, eps
                chain.append(["BOOST", eps, 0])
                print(f"[seed {seed}] БУСТИНГ подключён (eps={eps:.3e}, триггер "
                      f"{args.boost_trigger})", flush=True)

        # ---- state: AE over the saved trajectory, then latent loss surface ----
        # армам none/plateau/midpoint карты не нужны: политика fixed, а триггер
        # смотрит на историю лосса или на счётчик эпох. Обучение PINN от этого не
        # зависит (проверено: l2re совпадает до последнего знака), а построение
        # карт — это ~90% времени прогона
        def build_raw():
            """Одна сборка карт: автокодировщик по траектории последнего действия и
            поверхность потерь в его латентной плоскости. Возвращает (raw, с_AE, с_поверхн)."""
            t_a = time.time()
            ae_ = vm.train(args.ae_lr, args.ae_cosine_patience, args.ae_epochs, 100,
                           args.ae_batch, True, finetune_AE_model=False,
                           callbacks=[EarlyStopping(patience=args.ae_es_patience)],
                           solver_models=saver.saved_models)
            t_a = time.time() - t_a
            t_s = time.time()
            pls_ = PlotLossSurface(solver_models=saver.saved_models, AE_model=ae_,
                                   dde_pde_model=get_model_rec, x_range=GRID_RANGE,
                                   batch_size=args.ae_batch, loss_types=LOSS_TYPES,
                                   loss_name="loss_total", path_to_plot_model=None,
                                   path_to_trajectories=None, img_dir="")
            raw_ = pls_.save_equation_loss_surface(log_key=True)
            t_s = time.time() - t_s
            del pls_, ae_
            return raw_, t_a, t_s

        # --state-every K: карты строятся после действий 1, 1+K, 1+2K, ...; между ними
        # агент видит последнюю построенную карту (цена наблюдения делится на K)
        due = (len(chain) - 1) % max(1, args.state_every) == 0
        alt_maps = None
        if build_maps and due:
            raw, t_ae, t_srf = build_raw()
            t_ae_tot += t_ae
            t_srf_tot += t_srf
            print(f"[seed {seed}] state built: AE {t_ae:.1f}s (epochs={args.ae_epochs}), "
                  f"surface {t_srf:.1f}s", flush=True)
            true_maps = build_state(raw, prev_raw)
            prev_raw = raw
            stale_maps = true_maps
            if state_mode == "rebuild":
                # вторая сборка тех же карт с другой случайностью автокодировщика: насколько
                # наблюдение воспроизводимо. Состояние генераторов возвращается, поэтому
                # сам прогон совпадает с обычным
                rs_t, rs_n = torch.random.get_rng_state(), np.random.get_state()
                rs_c = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(seed * 9973 + 17 * len(chain))
                np.random.seed((seed * 9973 + 17 * len(chain)) % (2 ** 31 - 1))
                raw2, t_ae2, t_srf2 = build_raw()
                torch.random.set_rng_state(rs_t); np.random.set_state(rs_n)
                if rs_c is not None:
                    torch.cuda.set_rng_state_all(rs_c)
                t_ae_tot += t_ae2
                t_srf_tot += t_srf2
                alt_maps = build_state(raw2, prev_raw2)
                prev_raw2 = raw2
        elif build_maps and stale_maps is not None:
            true_maps = stale_maps
        else:
            true_maps = np.zeros((4, 26, 26), dtype=np.float32)
        if build_maps or needs_agent:
            # вмешательство в состояние: агент получает искажённые карты, истинные
            # сохраняются для сравнения выбранных действий
            if state_mode == "loss":
                # дешёвое состояние: обучающие лоссы в текущей точке вместо карт
                lv = np.asarray(model.train_state.loss_train, dtype=float).ravel()
                n_pde = int(getattr(getattr(model, "pde", None), "num_pde", len(lv)))
                l3 = (float(lv.sum()), float(lv[:n_pde].sum()), float(lv[n_pde:].sum()))
                if all(np.isfinite(l3)):
                    maps = loss_state(l3[0], l3[1], l3[2], prev_loss_tot)
                    prev_loss_tot = l3[0]
                else:
                    maps = np.zeros((4, 26, 26), dtype=np.float32)
            elif state_mode == "rebuild":
                maps = true_maps          # агент действует по первой сборке
            else:
                maps = apply_state_mode(true_maps, state_mode, seed=seed * 7919 + len(chain), pde=args.pde)
            state = maps
            # с чем сравнивать при --state-compare: истинные карты либо вторая сборка
            ref_maps = true_maps if state_mode != "rebuild" else alt_maps
            if want_ctx:
                e_ctx = (None if ctx_err == "none" else
                         (rmse + brmse) / err_scale if ctx_err == "err"
                         else math.hypot(l2re_op, l2re_bnd))
                state = add_desc(add_scalar_ctx(maps, len(chain), ctx_kmax, spent, ctx_budget,
                                                last_a, e_ctx))
                if args.state_compare:
                    true_state = (None if ref_maps is None else
                                  add_desc(add_scalar_ctx(ref_maps, len(chain), ctx_kmax, spent,
                                                          ctx_budget, last_a, e_ctx)))
            elif args.state_compare:
                true_state = ref_maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # чекпоинт ставим ПОСЛЕ построения состояния: веса PINN и карты должны
        # соответствовать друг другу, иначе докатка стартует с рассогласования
        if ckpt_name and args.ckpt_every and len(chain) % args.ckpt_every == 0:
            dump_ckpt()
        if args.hours and (time.time() - t0) / 3600.0 >= args.hours:
            dump_ckpt()
            print(f"[seed {seed}] лимит времени ({args.hours} ч): чекпоинт на "
                  f"spent={spent}/{args.budget}, продолжит следующий кернел", flush=True)
            return dict(seed=seed, policy=args.policy, pde=args.pde, unfinished=True,
                        l2re=math.hypot(l2re_op, l2re_bnd), spent=spent,
                        budget=args.budget, n_steps=len(chain),
                        elapsed_s=round(time.time() - t0, 1))

    l2re = math.hypot(l2re_op, l2re_bnd)
    return dict(seed=seed, policy=args.policy, pde=args.pde, l2re=l2re,
                stop_reason=stop_reason, spent=spent,
                boost_trigger=args.boost_trigger, boosted=boosted, rmse=rmse,
                brmse=brmse, l2re_op=l2re_op, l2re_bnd=l2re_bnd, budget=args.budget,
                n_steps=len(chain), chain=chain, ae_epochs=args.ae_epochs,
                state_mode=(state_mode if needs_agent else ("none" if args.no_state else "unused")),
                t_ae_s=round(t_ae_tot, 1), t_surface_s=round(t_srf_tot, 1),
                t_opt_s=round(t_opt_tot, 1),
                loss_final=(hist[-1][1] if hist else None), hist=hist,
                agree=agree,
                agree_rate=(round(float(np.mean([x[1] == x[2] for x in agree])), 4)
                            if agree else None),
                scout_spent=int(scout_spent), scout_log=scout_log,
                guard_spent=int(guard_spent), guard_log=guard_log,
                state_every=int(args.state_every), keep_opt=bool(keep_opt),
                elapsed_s=round(time.time() - t0, 1))


def result_done(name):
    """Есть ли в HF законченная строка этого сида (не промежуточная и не оборванная). При
    повторном запуске задачи с --resume такие сиды пропускаются: иначе кернел, перезапущенный
    ради недосчитанных сидов, заново считал бы готовые (в двойной точности это часы на сид)."""
    url = f"https://huggingface.co/datasets/{OUT_REPO}/resolve/main/rl_arch/online_env/{name}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        return not d.get("partial") and not d.get("unfinished")
    except Exception:
        return False


def upload(row, name):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF token — result printed only", flush=True)
        return
    import io
    from huggingface_hub import upload_file
    # итоговую строку ждём долго: лимит HF (128 коммитов в час на репозиторий)
    # держится около часа, а три ретрая по 5 секунд теряют готовый результат
    tries = 3 if row.get("partial") else 9
    for attempt in range(tries):
        try:
            upload_file(path_or_fileobj=io.BytesIO(json.dumps(row, indent=1).encode()),
                        path_in_repo=f"rl_arch/online_env/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch online_env {name}")
            print(f"uploaded rl_arch/online_env/{name}.json", flush=True)
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(5 if row.get("partial") else min(600, 20 * 2 ** attempt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["agent", "random", "fixed", "script", "rule", "bandit", "scout"])
    ap.add_argument("--state-every", type=int, default=1,
                    help="Трек 3, цена наблюдения: строить карты после каждого K-го действия, "
                         "между ними агент видит последнюю построенную карту")
    ap.add_argument("--keep-opt", action="store_true",
                    help="Сохранять состояние оптимизатора между действиями, пока семейство оптимизатора "
                         "не меняется (моменты Adam, история L-BFGS, предобусловливатель SOAP); шаг "
                         "обновляется на месте. Без флага каждое действие создаёт оптимизатор заново. "
                         "При докатке с чекпоинта первое действие начинает с чистого состояния")
    ap.add_argument("--guard-rollback", type=float, default=0.0,
                    help="Трек 3, страж: если после действия агента обучающий лосс вырос более чем в "
                         "столько раз (например 1.0 — любой рост), веса откатываются и исполняется "
                         "--guard-fallback. Потраченные эпохи остаются в счёте бюджета. 0 = выключено")
    ap.add_argument("--guard-fallback", default="LBFGS:1:500",
                    help="запасное действие стража")
    ap.add_argument("--state-mode", default="auto",
                    choices=["auto", "full", "blind", "level", "shape", "shuffle", "loss", "rebuild", "tasknorm"],
                    help="Трек 3: что видит агент. auto — как при обучении (из чекпоинта); "
                         "blind — карты не строятся; level — только уровень потерь; shape — только "
                         "форма; shuffle — перемешанные пиксели; loss — обучающие лоссы вместо карт "
                         "(карты не строятся); rebuild — вторая сборка карт с другой случайностью "
                         "автокодировщика (с --state-compare). Явное значение — вмешательство на оценке")
    ap.add_argument("--state-compare", action="store_true",
                    help="Вместе с вмешательством: строить истинные карты и записывать, какое "
                         "действие агент выбрал бы по ним (доля совпадений в итоговой строке)")
    ap.add_argument("--bandit-arms", default="Adam:0.001:100,Adam:0.0001:100,LBFGS:1:100,LBFGS:1:500,LBFGS:0.1:500",
                    help="--policy bandit: набор действий-рук")
    ap.add_argument("--bandit-c", type=float, default=1.0, help="ширина доверительной надбавки UCB")
    ap.add_argument("--bandit-discount", type=float, default=0.8,
                    help="забывание статистик рук (награды нестационарны: лосс выходит на плато)")
    ap.add_argument("--scout-set", default="Adam:0.001:100,LBFGS:1:100,LBFGS:0.1:100",
                    help="--policy scout: короткие пробные запуски из общей точки")
    ap.add_argument("--scout-commit", default="Adam:0.001:1000,LBFGS:1:500,LBFGS:0.1:500",
                    help="--policy scout: действие, исполняемое после победы одноимённой пробы")
    ap.add_argument("--script", default="",
                    help="--policy script: цепочка действий без обратной связи, напр. "
                         "'LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500'")
    ap.add_argument("--script-tail", default="stop", choices=["stop", "repeat", "agent"],
                    help="что делать, когда цепочка кончилась, а бюджет нет: stop — "
                         "остановиться, repeat — повторять последнее действие, agent — "
                         "передать управление агенту (--model-file)")
    ap.add_argument("--rule-burst", default="LBFGS:1:100",
                    help="--policy rule: основное действие (всплеск L-BFGS)")
    ap.add_argument("--rule-kick", default="Adam:0.0001:100",
                    help="--policy rule: действие-толчок при застое лосса")
    ap.add_argument("--rule-max-kicks", type=int, default=3,
                    help="--policy rule: сколько толчков подряд без улучшения лосса до остановки")
    ap.add_argument("--rule-tol", type=float, default=1e-4,
                    help="--policy rule: относительное изменение лосса, ниже которого застой")
    ap.add_argument("--model-files", default=None,
                    help="комитет агентов: несколько чекпоинтов через запятую")
    ap.add_argument("--ensemble", default="vote", choices=["vote", "mean"],
                    help="как комитет выбирает действие")
    ap.add_argument("--mask", default="",
                    help="запрещённые действия ('pso,adam:0.01'); маска из чекпоинта "
                         "применяется автоматически")
    ap.add_argument("--guide", default="",
                    help="цепочка-эвристика: агент отходит от неё, только если его Q "
                         "выше Q действия эвристики больше чем на --guide-bonus")
    ap.add_argument("--guide-bonus", type=float, default=0.0)
    ap.add_argument("--float64", action="store_true",
                    help="решать PINN в двойной точности (только с --no-state: конвейер "
                         "карт рассчитан на float32). Диагностика: снимает ли точность "
                         "застой L-BFGS")
    ap.add_argument("--stop-on-noop", action="store_true",
                    help="выбор PSO с lr=0 после хотя бы одного шага Adam или L-BFGS "
                         "трактовать как «стоп» и закончить цепочку: "
                         "лидер трека 1 после 2000 эпох L-BFGS крутит это действие до "
                         "конца бюджета, 70%% времени оценки уходит на пустые шаги")
    ap.add_argument("--fixed-action", type=int, default=4,
                    help="Индекс повторяемого действия для --policy fixed "
                         "(4 = Adam lr 1e-3 x 1000 эпох)")
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--q-policy", default="auto", choices=["auto", "mean", "cvar"],
                    help="Как сводить квантили к скаляру у квантильных вариантов")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--pde", default="poissonboltzmann2d")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--budget", type=int, default=31000)
    ap.add_argument("--n-save-models", type=int, default=10)
    ap.add_argument("--display-every", type=int, default=100)
    ap.add_argument("--ae-epochs", type=int, default=10000)
    ap.add_argument("--ae-lr", type=float, default=5e-4)
    ap.add_argument("--ae-batch", type=int, default=32)
    ap.add_argument("--ae-cosine-patience", type=int, default=1200)
    ap.add_argument("--ae-es-patience", type=int, default=4000)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--boost-trigger", default="none",
                    choices=["none", "landscape", "landscape_peak", "plateau", "midpoint"])
    ap.add_argument("--boost-warmup", type=int, default=8,
                    help="Сколько шагов копить историю до срабатывания peak-триггера")
    ap.add_argument("--boost-threshold", type=float, default=0.5,
                    help="порог ландшафтного триггера. ВНИМАНИЕ: обученная модель "
                         "выдаёт P в диапазоне 0.10-0.45, порог 0.5 недостижим "
                         "(превышается на 0.19% состояний буфера) — калибруйте "
                         "по фактическому распределению, напр. 0.40 = верхняя дециль")
    ap.add_argument("--plateau-rel", type=float, default=1e-3,
                    help="относительный размах 4 подряд лоссов, ниже которого "
                         "считаем плато")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="мягкий лимит по времени: сохранить чекпоинт и выйти "
                         "(0 = без лимита). Сессия Kaggle живёт 12 ч")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прогон с чекпоинта (локального или с HF)")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="как часто страховочно сохранять чекпоинт, в шагах цепочки")
    ap.add_argument("--progress-every", type=float, default=1800.0,
                    help="минимальный интервал между строками прогресса, секунд "
                         "(лимит HF — 128 коммитов в час на репозиторий)")
    ap.add_argument("--plain-fnn", action="store_true",
                    help="обратные задачи: обычный FNN вместо PFNN (конвейер карт "
                         "не разбирает ветвящиеся сети)")
    ap.add_argument("--no-state", action="store_true",
                    help="не строить карты ландшафта (AE + поверхность лоссов). "
                         "Допустимо только для policy=fixed/random с триггерами "
                         "none/plateau/midpoint — там карты не используются")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global Q_POLICY
    Q_POLICY = args.q_policy
    if args.float64 and not args.no_state:
        sys.exit("--float64 поддержан только вместе с --no-state")
    if args.policy == "script" and not args.script:
        sys.exit("--policy script требует --script")
    if args.model_files and not args.tag:
        sys.exit("--model-files (комитет) требует --tag")
    if args.policy in ("script", "rule") and not args.tag:
        sys.exit("--policy script/rule требует --tag: иначе результат ляжет под именем random")
    uses_agent = args.policy == "agent" or (args.policy == "script" and args.script_tail == "agent")
    if args.no_state and args.boost_trigger.startswith("landscape"):
        sys.exit("--no-state несовместим с ландшафтными триггерами: им нужны карты")
    if args.no_state and uses_agent and args.state_mode not in ("auto", "blind", "loss"):
        sys.exit("--no-state с агентом допустим только при --state-mode blind или loss (или auto "
                 "для агента, обученного без карт)")
    if args.state_compare and (args.no_state or not uses_agent):
        sys.exit("--state-compare требует агента и построения карт (без --no-state)")
    if args.state_every < 1:
        sys.exit("--state-every должен быть не меньше 1")
    if any(g >= 27 for g in parse_chain(args.guide)):
        sys.exit("--guide: проводник сравнивается по Q агента, поэтому состоит только из 27 действий агента")
    if args.guard_rollback and not uses_agent:
        sys.exit("--guard-rollback действует только на действия агента")
    if args.policy in ("bandit", "scout") and not args.tag:
        sys.exit("--policy bandit/scout требует --tag")
    if args.policy == "scout":
        if len(parse_chain(args.scout_set)) != len(parse_chain(args.scout_commit)):
            sys.exit("--scout-set и --scout-commit должны быть одной длины")
    if args.smoke:  # только как дефолты — явные флаги не перезаписываем
        given = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("--"))
        if "--budget" not in given: args.budget = 300
        if "--ae-epochs" not in given: args.ae_epochs = 50
        if "--n-save-models" not in given: args.n_save_models = 3

    tag = args.tag or (os.path.basename(args.model_file).replace(".pt", "")
                       if args.model_file else "random")
    for seed in [int(s) for s in args.seeds.split(",")]:
        name = f"{args.pde}_{tag}_seed{seed}"
        if args.resume and not args.smoke and result_done(name):
            print(f"[seed {seed}] итоговая строка уже в HF — пропуск", flush=True)
            continue
        args._ckpt_name = name
        cb = None if args.smoke else (lambda r, n=name: upload(r, n))
        row = run_seed(seed, args, progress_cb=cb)
        row["smoke"] = args.smoke
        print(json.dumps({k: v for k, v in row.items() if k != "chain"}), flush=True)
        if row.get("unfinished"):
            # итог не заливаем: строка выглядела бы завершённой, а бюджет не выбран
            print(f"[seed {seed}] прогон не закончен — нужен ещё один кернел с --resume",
                  flush=True)
            continue
        if not args.smoke:
            upload(row, f"{args.pde}_{tag}_seed{seed}")


if __name__ == "__main__":
    main()
