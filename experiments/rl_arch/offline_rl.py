#!/usr/bin/env python
"""
Offline-RL architecture benchmark on the rlpinn poisson_boltzmann_2d buffer.

Data: 80 episode files (danil-e/rlpinn-ablation-buffers), 5841 transitions;
state = 4x26x26 loss-landscape maps, action = 27 discrete (3 opt x 3 lr x 3 ep),
reward in [-3.47, 0). Behavior policy ~uniform (good offline coverage).

Variants (same data split, training protocol and FQE evaluator for all):
    cnn_dqn       baseline: small CNN encoder + double-DQN
    convnext_dqn  H1: ConvNeXt-style encoder + double-DQN
    cnn_cql       H2a: CNN + conservative Q (CQL penalty)
    cnn_iql       H2b: CNN + implicit Q-learning (expectile V + TD-to-V)
    cnn_qrdqn     H3: CNN + QR-DQN (32 quantiles); policies: mean and CVaR@0.25
    cnn_vqc       H4: CNN + variational quantum circuit Q-head (PennyLane)

Metrics per (variant, seed), holdout = 20% of episodes (fixed split):
    td_error          holdout double-DQN Bellman residual (calibration)
    spearman_q_rtg    rank corr of Q(s, a_logged) vs discounted return-to-go
    fqe_*             fitted Q evaluation of the greedy policy (same FQE arch
                      for every variant): mean value over holdout states /
                      initial states, and CVaR@0.25 over holdout states
    agree_behavior    greedy-policy agreement with logged actions
Results: JSON per run, uploaded to HF dataset rl_arch/{variant}_seed{n}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)   # RL.* / src.* imports when run as a script

import torch  # module-level: QNet/soft_update/q_cvar are used by online_eval too

GAMMA = 0.9   # их значение (rl_agent_params в poisson_boltzmann2d_chain.py)
N_ACTIONS = 27
# эпохи на действие: a -> (opt=a//9, lr=(a//3)%3, ep=a%3); таблица и индексация
# те же, что в online_train_env.py (иначе SMDP-дисконт разойдётся с онлайном)
EPOCHS_TABLE = [100, 1000, 2500, 100, 500, 1000, 100, 200, 300]
REPO = "danil-e/rlpinn-ablation-buffers"
SUBDIR = "poisson_boltzmann_2d"
OUT_REPO = "danil-e/pinnacle-optuna-db"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _thin(names, max_files: int):
    """Не больше max_files имён, равномерно по отсортированному списку (детерминированно)."""
    names = sorted(names)
    if not max_files or len(names) <= max_files:
        return names
    idx = np.unique(np.linspace(0, len(names) - 1, max_files).round().astype(int))
    return [names[i] for i in idx]


def load_episodes(data_dir: str | None, subdir: str = SUBDIR, max_files: int = 0):
    import torch
    from huggingface_hub import list_repo_files, hf_hub_download

    if not data_dir and os.environ.get("RL_BUFFER_DIR"):
        # локальная копия буферов (тесты без сети): $RL_BUFFER_DIR/<subdir>/*.pt
        data_dir = os.path.join(os.environ["RL_BUFFER_DIR"], subdir)
    if data_dir and os.path.isdir(data_dir) and any(f.endswith(".pt") for f in os.listdir(data_dir)):
        paths = _thin([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if f.endswith(".pt")], max_files)
    else:
        names = _thin([f for f in list_repo_files(REPO, repo_type="dataset")
                       if f.startswith(subdir + "/") and f.endswith(".pt")], max_files)
        paths = [hf_hub_download(REPO, f, repo_type="dataset") for f in names]
    episodes = []
    for p in paths:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(d, list) or not d:
            continue
        episodes.append(d)
    return episodes


def episodes_to_arrays(episodes, fix_next_state: bool = False, chain_fix: bool = False,
                       reward_form: str = "logged", init_err=None, budget: int = 0,
                       err_scale: float = 1.0, verbose: bool = True):
    """fix_next_state: the poisson_boltzmann_2d dump logs next_state as a COPY of
    state (100% of transitions; healthy dumps satisfy next_state[t]==state[t+1]
    in 92%). Reconstruct s' = state[t+1] within each episode, last step terminal."""
    if chain_fix:
        return episodes_to_arrays_chains(episodes, reward_form=reward_form, verbose=verbose,
                                         init_err=init_err, budget=budget, err_scale=err_scale)
    if reward_form != "logged":
        raise ValueError("reward_form, отличная от 'logged', требует chain_fix=True: "
                         "без границ цепочек разности ошибок посчитать нельзя")
    base = ["loss_total", "loss_oper", "loss_bnd"]
    S, A, R, S2, D, EP = [], [], [], [], [], []
    for ei, ep in enumerate(episodes):
        prev_tot = prev_tot_ns = None
        for t in ep:
            # healthy dumps carry 3 channels; pb2d also stores `delta`. Keep the
            # 4-channel layout everywhere by deriving delta from consecutive maps
            # (same clip as the online env).
            cur = [np.asarray(t["state"][k], dtype=np.float32) for k in base]
            nxt = [np.asarray(t["next_state"][k], dtype=np.float32) for k in base]
            if "delta" in t["state"]:
                cur.append(np.asarray(t["state"]["delta"], dtype=np.float32))
                nxt.append(np.asarray(t["next_state"]["delta"], dtype=np.float32))
            else:
                cur.append(np.zeros_like(cur[0]) if prev_tot is None
                           else np.clip(cur[0] - prev_tot, -1, 1))
                nxt.append(np.clip(nxt[0] - cur[0], -1, 1))
                prev_tot = cur[0]
            S.append(np.stack(cur))
            S2.append(np.stack(nxt))
            a = t["action"]
            A.append(int(a[0]) * 9 + int(a[1]["lr"]) * 3 + int(a[1]["epochs"]))
            R.append(float(t["reward"]))
            D.append(float(t.get("done", 0)))
            EP.append(ei)
    if fix_next_state:
        by_ep = {}
        for i, e in enumerate(EP):
            by_ep.setdefault(e, []).append(i)
        for e, idxs in by_ep.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                S2[a] = S[b]
    S = np.stack(S); S2 = np.stack(S2)
    A = np.array(A, dtype=np.int64); R = np.array(R, dtype=np.float32)
    D = np.array(D, dtype=np.float32); EP = np.array(EP, dtype=np.int64)
    # force terminal at each episode's last transition (safety)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        D[idx[-1]] = 1.0
    # discounted return-to-go per episode
    RTG = np.zeros_like(R)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        run = 0.0
        for i in idx[::-1]:
            run = R[i] + GAMMA * run * (1.0 - D[i])
            RTG[i] = run
    FIRST = np.zeros_like(D)
    for ei in np.unique(EP):
        FIRST[np.where(EP == ei)[0][0]] = 1.0
    return dict(S=S, A=A, R=R, S2=S2, D=D, EP=EP, RTG=RTG, FIRST=FIRST)


REWARD_FORMS = ("logged", "level", "delta", "model", "dlog")

# ---- трек 3: лестница состояния -------------------------------------------------
# full    — карты как есть
# blind   — карты обнулены: агент видит только скалярный контекст (если он включён)
# level   — каждая карта заменена своим средним: остаётся уровень потерь, форма ландшафта убрана
# shape   — из каждой карты вычтено среднее и она поделена на свой разброс: остаётся
#           форма ландшафта, уровень потерь убран (зеркало режима level)
# shuffle — пиксели каждой карты перемешаны независимо: статистики те же, пространственной структуры нет
# loss    — карт нет, вместо них три обучающих лосса в текущей точке (общий, оператор,
#           границы) и их изменение за шаг. Наблюдение бесплатное: ни автокодировщика,
#           ни поверхности. В офлайновых буферах обучающего лосса нет, поэтому режим
#           доступен только в онлайновых скриптах
# tasknorm — три карты потерь приведены к нулевому среднему и единичному разбросу ПО ЗАДАЧЕ
#           (статистики буфера данного УрЧП из track3/pde_meta.json): одномерный оптимальный
#           перенос распределений уровня между задачами при гауссовом приближении; уровень
#           внутри задачи сохраняется, различия масштабов между задачами убраны
STATE_MODES = ("full", "blind", "level", "shape", "shuffle", "loss", "tasknorm")
SCALAR_CH = 5
PDE_DESC_CH = 6
_EPOCHS_TABLE = [100, 1000, 2500, 100, 500, 1000, 100, 200, 300]


def apply_state_mode(S, mode: str = "full", seed: int = 0, pde: str | None = None):
    """Преобразует первые четыре канала (карты) состояния или пачки состояний.
    Контекстные каналы (если уже приклеены) не трогаются. Возвращает копию.
    Режим tasknorm требует имя УрЧП (статистики карт из track3/pde_meta.json)."""
    if mode == "full":
        return S
    if mode not in STATE_MODES:
        raise ValueError(f"неизвестный режим состояния: {mode}")
    S = np.array(S, dtype=np.float32, copy=True)
    m = S[..., :4, :, :]
    if mode == "tasknorm":
        if not pde:
            raise ValueError("режим tasknorm требует имя УрЧП")
        row = pde_meta()[pde]
        if "map_mean" not in row:
            raise ValueError(f"в track3/pde_meta.json нет статистик карт для {pde}: "
                             f"запустите pde_meta.py --map-stats")
        mu = np.asarray(row["map_mean"], dtype=np.float32)[:, None, None]
        sd = np.asarray(row["map_std"], dtype=np.float32)[:, None, None]
        nz = m[..., :3, :, :] != 0.0           # нулевая стартовая карта остаётся нулевой
        m[..., :3, :, :] = np.where(nz, (m[..., :3, :, :] - mu) / np.maximum(sd, 1e-6), 0.0)
        return S
    if mode == "loss":
        raise ValueError("режим loss строится из обучающего лосса (loss_state) и для массивов "
                         "карт не определён: в офлайновых буферах обучающего лосса нет")
    if mode == "blind":
        m[...] = 0.0
    elif mode == "level":
        m[...] = m.mean(axis=(-2, -1), keepdims=True)
    elif mode == "shape":
        mu = m.mean(axis=(-2, -1), keepdims=True)
        sd = m.std(axis=(-2, -1), keepdims=True)
        m[...] = np.where(sd > 1e-6, (m - mu) / np.maximum(sd, 1e-6), 0.0)
    else:  # shuffle
        shp = m.shape
        flat = m.reshape(shp[:-2] + (-1,))
        rng = np.random.default_rng(seed)
        order = rng.random(flat.shape).argsort(axis=-1)
        m[...] = np.take_along_axis(flat, order, axis=-1).reshape(shp)
    return S


def _loss_code(x):
    """Лосс -> число в [-1, 1]: десятичный логарифм, диапазон 1e-8..1e4 (у необученных
    сетей лосс доходит до тысяч, у сошедшихся опускается ниже 1e-6)."""
    return float(np.clip(np.log10(max(float(x), 1e-12)) / 6.0 + 1.0 / 3.0, -1.0, 1.0))


def loss_state(loss_total, loss_oper, loss_bnd, prev_total=None, size: int = 26):
    """Состояние режима loss: четыре постоянных канала на месте карт. Первые три —
    уровни обучающих лоссов, четвёртый — на сколько порядков общий лосс упал за
    последнее действие (0 на первом шаге)."""
    d = 0.0
    if prev_total is not None and prev_total > 0 and loss_total > 0:
        d = float(np.clip(np.log10(float(prev_total)) - np.log10(float(loss_total)), -1.0, 1.0))
    vals = [_loss_code(loss_total), _loss_code(loss_oper), _loss_code(loss_bnd), d]
    return np.stack([np.full((size, size), v, dtype=np.float32) for v in vals])


_PDE_META = None


def pde_meta():
    """Таблица задач (track3/pde_meta.json): размерности, число слагаемых потерь,
    ошибка необученной сети. Строится скриптом track3/pde_meta.py."""
    global _PDE_META
    if _PDE_META is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track3", "pde_meta.json")
        with open(path) as f:
            _PDE_META = json.load(f)
    return _PDE_META


def subdir_to_pde(subdir: str) -> str:
    for name, row in pde_meta().items():
        if row.get("subdir") == subdir:
            return name
    raise KeyError(f"папка буфера {subdir!r} не найдена в track3/pde_meta.json")


def pde_desc(pde_name: str):
    """Дешёвый описатель задачи, известный до обучения: размерности, число слагаемых
    потерь, зависимость от времени, обратная задача. Шесть чисел в [0, 1]."""
    r = pde_meta()[pde_name]
    return [min(1.0, r["in_dim"] / 5.0), min(1.0, r["out_dim"] / 3.0),
            min(1.0, r["n_pde"] / 3.0), min(1.0, r["n_bnd"] / 8.0),
            float(r["time"]), float(r["inverse"])]


def _ctx_vals(step, k_max, spent, budget, last_action, err):
    """То же кодирование, что add_scalar_ctx в онлайновых скриптах."""
    if last_action is None or last_action < 0:
        opt_i = ep_i = -1.0
    else:
        opt_i = (int(last_action) // 9) / 2.0
        ep_i = (int(last_action) % 3) / 2.0
    e = 0.0 if err is None else float(np.clip(np.log10(max(float(err), 1e-8)) / 3.0 + 1.0, -1, 1))
    return [min(1.0, step / max(1, k_max)), min(1.0, spent / max(1, budget)), opt_i, ep_i, e]


def chain_ctx(data, k_max: int = 10, budget: int = 31000, with_err: bool = False):
    """Скалярный контекст офлайновых переходов, отдельно для s и для s'.
    Требует загрузки с chain_fix (нужны STEP, SPENT, ERR). Контекст s' — это контекст
    следующего шага: шаг+1, потрачено + эпохи действия, прошлое действие = текущее,
    ошибка после действия. with_err=False оставляет канал ошибки нулевым (только время)."""
    for k in ("STEP", "SPENT", "ERR", "FIRST"):
        if k not in data:
            raise ValueError("скалярный контекст офлайн требует --chain-fix")
    A = data["A"]; n = len(A)
    c = np.zeros((n, SCALAR_CH), dtype=np.float32)
    c2 = np.zeros((n, SCALAR_CH), dtype=np.float32)
    for i in range(n):
        first = bool(data["FIRST"][i]) or i == 0 or data["EP"][i - 1] != data["EP"][i]
        prev_a = -1 if first else int(A[i - 1])
        # на первом шаге онлайновая оценка подаёт уровень 1.0 (ошибка ещё не измерена)
        prev_e = (1.0 if first else float(data["ERR"][i - 1])) if with_err else None
        c[i] = _ctx_vals(float(data["STEP"][i]), k_max, float(data["SPENT"][i]), budget, prev_a, prev_e)
        a = int(A[i])
        ep = _EPOCHS_TABLE[(a // 9) * 3 + (a % 3)]
        c2[i] = _ctx_vals(float(data["STEP"][i]) + 1, k_max, float(data["SPENT"][i]) + ep, budget, a,
                          float(data["ERR"][i]) if with_err else None)
    return c, c2


def keep_consistent(A, EP, STEP, mode="all"):
    """Маска переходов буфера, чья динамика не зависит от сброса оптимизатора: действие исполнено
    на новом оптимизаторе и в среде с сохранением состояния (--keep-opt). Это первый шаг цепочки,
    шаг после действия другого семейства и любое действие PSO (у него состояния нет). При
    mode="safe" новый оптимизатор получает и Adam со сменой шага (сохраняется только история
    L-BFGS и Adam при том же шаге). Шаг, чей предшественник в массиве не найден (цепочка
    обрезана), считается несогласованным."""
    A, EP, STEP = np.asarray(A), np.asarray(EP), np.asarray(STEP)
    fam = A // 9
    lr = (A % 9) // 3
    ok = (STEP == 0) | (fam == 2)
    if len(A) > 1:
        adj = (EP[1:] == EP[:-1]) & (STEP[1:] == STEP[:-1] + 1)
        fresh = fam[1:] != fam[:-1]
        if mode == "safe":
            fresh |= (fam[1:] == 0) & (lr[1:] != lr[:-1])
        ok[1:] |= adj & fresh
    return ok


def attach_ctx(S, ctx):
    """Скаляры разворачиваются в постоянные каналы и приклеиваются к картам."""
    n, _, h, w = S.shape
    planes = np.broadcast_to(ctx[:, :, None, None], (n, ctx.shape[1], h, w))
    return np.concatenate([S, planes.astype(np.float32)], axis=1)


def _delta_like_env(tot, prev_tot):
    """Канал delta ровно так, как его строит онлайновая среда
    (online_eval_env.build_state): sign*log1p от разности карт, нормировка на
    максимум модуля. Без предыдущей карты (первое построение в цепочке) — нули."""
    if prev_tot is None:
        return np.zeros_like(tot, dtype=np.float32)
    d = tot - prev_tot
    dl = np.sign(d) * np.log1p(np.abs(d))
    dl = dl / (np.abs(dl).max() + 1e-6)
    return np.clip(dl, -1.0, 1.0).astype(np.float32)


def episodes_to_arrays_chains(episodes, reward_form: str = "delta", verbose: bool = True,
                              init_err=None, budget: int = 0, err_scale: float = 1.0):
    """Загрузка буфера с восстановлением структуры цепочек (включается --chain-fix).

    Что исправляет по сравнению с episodes_to_arrays (проверено на буферах
    ns2d_liddriven, poisson3d_complexgeometry, poisson_boltzmann_2d):

    1. Файл = несколько цепочек подряд, а не один эпизод. Граница цепочки —
       переход с done != 0 (1 = достигнут допуск, -1 = исчерпан K_max без
       допуска). EP теперь номер ЦЕПОЧКИ, поэтому n-шаговые возвраты, RTG,
       FIRST и скалярный контекст больше не пересекают границы.
    2. done = -1 раньше попадал в массив D как -1.0, и цель r + g*(1-D)*Q'
       удваивала бутстреп на последнем шаге каждой неудачной цепочки. Конец
       цепочки теперь терминален: D = 1.
    3. В дампах нового формата (ключ delta в state) next_state — копия state.
       Следующее состояние восстанавливается как state[t+1] внутри цепочки;
       у последнего перехода цепочки оно не нужно (терминал). Хвост файла без
       флага done в таком дампе отбрасывается: его следующего состояния нет.
    4. Состояния приводятся к определению онлайновой среды: старт цепочки —
       нулевая карта, канал delta после первого действия нулевой, далее
       sign*log1p от разности соседних карт с нормировкой на максимум.
    5. Награда. В поле reward дампа лежит минус УРОВЕНЬ ошибки после действия
       (на неудачном конце ещё и штраф -1), а онлайновая среда даёт РАЗНОСТЬ
       ошибок. reward_form:
         logged — как в дампе (прежнее поведение по награде);
         level  — минус уровень ошибки, без штрафа;
         delta  — E_до - E_после, на первом шаге 0 (как в онлайне);
         model  — поле reward_model (награда, на которой учился агент авторов);
         dlog   — log10(E_до) - log10(E_после): относительное улучшение, не зависит
                  от масштаба ошибки и потому годится для смеси разных УрЧП.
       err_scale — на что делить ошибку (обычно E0 задачи): при обучении на нескольких
       УрЧП ошибки приводятся к долям начальной.
       init_err — ошибка необученной сети. С ней первый шаг при delta получает
       награду init_err - E_1, а не 0. Без неё возврат цепочки равен E_1 - E_конец,
       то есть плохое первое действие ВЫГОДНО (больше «запас улучшения») — отсюда
       пустой первый шаг PSO у обученных политик.
    6. budget > 0 — цепочки обрезаются по бюджету эпох, как эпизод на оценке:
       переход, на котором расход достигает бюджета, терминален, дальнейшие шаги
       цепочки отбрасываются.

    Дополнительно отдаёт ERR (ошибка после действия), STEP (номер шага в
    цепочке), SPENT (эпох потрачено до действия) и сводку STATS.
    """
    if reward_form not in REWARD_FORMS:
        raise ValueError(f"reward_form={reward_form!r}, допустимо: {REWARD_FORMS}")
    base = ["loss_total", "loss_oper", "loss_bnd"]
    S, A, R, S2, D, EP, ERR, STEP, SPENT = ([] for _ in range(9))
    st = dict(files=0, copy_files=0, chains=0, ok=0, failed=0, tail=0,
              next_rebuilt=0, dropped_tail=0, diverged=0)
    cid = -1
    for ep in episodes:
        if not ep:
            continue
        st["files"] += 1
        probe = ep[: min(len(ep), 24)]
        same = sum(np.array_equal(np.asarray(t["state"]["loss_total"]),
                                  np.asarray(t["next_state"]["loss_total"])) for t in probe)
        copy_fmt = same >= 0.95 * len(probe)      # в здоровых дампах таких шагов ~12%
        st["copy_files"] += int(copy_fmt)
        chains, cur = [], []
        for t in ep:
            cur.append(t)
            if int(t.get("done", 0)) != 0:
                chains.append((cur, True)); cur = []
        if cur:
            chains.append((cur, False))
        for c, finished in chains:
            cid += 1
            st["chains"] += 1
            maps = [[np.asarray(t["state"][k], dtype=np.float32) for k in base] for t in c]
            states = []
            for j, m in enumerate(maps):
                if j == 0:
                    states.append(np.zeros((4,) + m[0].shape, dtype=np.float32))
                else:
                    prev_tot = maps[j - 1][0] if j >= 2 else None
                    states.append(np.stack(m + [_delta_like_env(m[0], prev_tot)]))
            prev_err, spent = None, 0
            for j, t in enumerate(c):
                last = j == len(c) - 1
                dn = int(t.get("done", 0))
                a = t["action"]
                a_idx = int(a[0]) * 9 + int(a[1]["lr"]) * 3 + int(a[1]["epochs"])
                ep_a = EPOCHS_TABLE[a_idx // 9 * 3 + a_idx % 3]
                cut = bool(budget and spent + ep_a >= budget)   # бюджет кончается на этом шаге
                raw_r = float(t["reward"])
                err = -raw_r - (1.0 if dn == -1 else 0.0)
                if not np.isfinite(err) or err < 0:
                    # расходимость: трейнер авторов пишет reward_scalar = 0 и обрывает цепочку
                    err = max(prev_err * err_scale if prev_err is not None else 1.0, 1.0)
                    st["diverged"] += 1
                err = err / err_scale
                if cut:
                    s2, d = states[j], 1.0
                    st["budget_cut"] = st.get("budget_cut", 0) + 1
                elif not last:
                    s2 = states[j + 1]
                    d = 0.0
                    if copy_fmt:
                        st["next_rebuilt"] += 1
                elif finished:
                    s2, d = states[j], 1.0
                elif copy_fmt:
                    st["dropped_tail"] += 1
                    break
                else:
                    nxt = [np.asarray(t["next_state"][k], dtype=np.float32) for k in base]
                    s2 = np.stack(nxt + [_delta_like_env(nxt[0], maps[j][0] if j >= 1 else None)])
                    d = 0.0
                if reward_form == "logged":
                    r = raw_r
                elif reward_form == "level":
                    r = -err
                elif reward_form == "delta":
                    if prev_err is None:
                        r = 0.0 if init_err is None else float(init_err) - err
                    else:
                        r = prev_err - err
                elif reward_form == "dlog":
                    base_e = prev_err if prev_err is not None else init_err
                    r = 0.0 if base_e is None else float(
                        np.clip(np.log10(max(float(base_e), 1e-8)) - np.log10(max(err, 1e-8)),
                                -3.0, 3.0))
                else:
                    r = float(t.get("reward_model", raw_r))
                S.append(states[j]); S2.append(s2); A.append(a_idx); R.append(r); D.append(d)
                EP.append(cid); ERR.append(err); STEP.append(j); SPENT.append(spent)
                spent += ep_a
                prev_err = err
                if cut:
                    break
            if finished:
                st["ok" if int(c[-1].get("done", 0)) == 1 else "failed"] += 1
            else:
                st["tail"] += 1
    S = np.stack(S); S2 = np.stack(S2)
    # в части дампов (например wave2d_longtime) встречаются карты с NaN: расходившийся прогон.
    # Одно такое состояние превращает в NaN статистики нормировки и всё обучение
    bad = int((~np.isfinite(S)).any(axis=(1, 2, 3)).sum() + (~np.isfinite(S2)).any(axis=(1, 2, 3)).sum())
    if bad:
        S = np.nan_to_num(S, nan=0.0, posinf=20.0, neginf=-20.0)
        S2 = np.nan_to_num(S2, nan=0.0, posinf=20.0, neginf=-20.0)
        st["nonfinite_states"] = bad
        print(f"chain-fix: состояний с нечисловыми значениями {bad} — заменены нулями", flush=True)
    A = np.array(A, dtype=np.int64); R = np.array(R, dtype=np.float32)
    D = np.array(D, dtype=np.float32); EP = np.array(EP, dtype=np.int64)
    RTG = np.zeros_like(R); FIRST = np.zeros_like(D)
    starts = np.r_[0, np.where(np.diff(EP) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(EP)]
    for a0, b0 in zip(starts, ends):
        run = 0.0
        for i in range(b0 - 1, a0 - 1, -1):
            run = R[i] + GAMMA * run * (1.0 - D[i])
            RTG[i] = run
        FIRST[a0] = 1.0
    if verbose:
        print(f"chain-fix: файлов {st['files']} (с копией next_state: {st['copy_files']}), "
              f"цепочек {st['chains']} (допуск {st['ok']}, неудача {st['failed']}, "
              f"хвостов {st['tail']}), переходов {len(A)}, следующих состояний восстановлено "
              f"{st['next_rebuilt']}, отброшено хвостов {st['dropped_tail']}, "
              f"терминальных {int(D.sum())}, награда: {reward_form}"
              + (f", первый шаг от init_err={init_err}" if init_err is not None else "")
              + (f", обрезка по бюджету {budget} ({st.get('budget_cut', 0)} цепочек)" if budget else ""),
              flush=True)
    return dict(S=S, A=A, R=R, S2=S2, D=D, EP=EP, RTG=RTG, FIRST=FIRST,
                ERR=np.array(ERR, dtype=np.float32), STEP=np.array(STEP, dtype=np.int64),
                SPENT=np.array(SPENT, dtype=np.int64), STATS=st)


def merge_online_buffers(data, paths, reward_form="delta", init_err=None, budget=0):
    """Подмешать к офлайновому набору онлайновые буферы прошлых прогонов
    (online_train_env.py --save-buffer). Это нужно для офлайнового переобучения
    итоговой политики на всех собранных данных (OOO, arXiv 2310.08558) и для
    переиспользования опыта всех прошлых экспериментов (RaE, NeurIPS 2023).

    data должен быть загружен с chain_fix=True: определения состояний, границ
    цепочек и терминалов тогда совпадают с онлайновой средой. Награды онлайновых
    переходов пересчитываются из сохранённых ошибок в ту же форму, что и у data."""
    import torch
    if "STEP" not in data:
        raise ValueError("merge_online_buffers требует data, загруженный с chain_fix=True")
    parts = {k: [data[k]] for k in ("S", "A", "R", "S2", "D", "EP", "ERR", "STEP", "SPENT")}
    ep_off = int(data["EP"].max()) + 1
    n_add = 0
    for p in paths:
        path = p
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(OUT_REPO, p if "/" in p else f"rl_arch/buffers_online/{p}",
                                   repo_type="dataset")
        b = torch.load(path, map_location="cpu", weights_only=False)
        S = np.asarray(b["S"], dtype=np.float32)[:, :4]      # контекстные каналы отбрасываем
        S2 = np.asarray(b["S2"], dtype=np.float32)[:, :4]
        A = np.asarray(b["A"], dtype=np.int64)
        ERR = np.asarray(b["ERR"], dtype=np.float32)
        STEP = np.asarray(b["STEP"], dtype=np.int64)
        SPENT = np.asarray(b["SPENT"], dtype=np.int64)
        D = np.asarray(b["D"], dtype=np.float32).copy()
        EP = np.asarray(b["EP"], dtype=np.int64)
        keep = np.ones(len(A), dtype=bool)
        if budget:
            ep_a = np.array([EPOCHS_TABLE[a // 9 * 3 + a % 3] for a in A])
            keep = SPENT < budget
            D[(SPENT + ep_a >= budget) & keep] = 1.0
        R = np.zeros(len(A), dtype=np.float32)
        for i in range(len(A)):
            if reward_form in ("level", "logged"):
                R[i] = -ERR[i]
            elif reward_form in ("delta", "dlog"):
                first = STEP[i] == 0 or i == 0 or EP[i - 1] != EP[i]
                base_e = init_err if first else ERR[i - 1]
                if base_e is None:
                    R[i] = 0.0
                elif reward_form == "delta":
                    R[i] = float(base_e) - ERR[i]
                else:
                    R[i] = float(np.clip(np.log10(max(float(base_e), 1e-8))
                                         - np.log10(max(float(ERR[i]), 1e-8)), -3.0, 3.0))
            else:
                raise ValueError("reward_form=model для онлайновых буферов не определена")
        _, ep_local = np.unique(EP, return_inverse=True)
        for k, v in (("S", S), ("A", A), ("R", R), ("S2", S2), ("D", D),
                     ("EP", ep_local + ep_off), ("ERR", ERR), ("STEP", STEP), ("SPENT", SPENT)):
            parts[k].append(v[keep])
        ep_off += int(ep_local.max()) + 1
        n_add += int(keep.sum())
    out = {k: np.concatenate(v, 0) for k, v in parts.items()}
    # последний переход каждой добавленной цепочки без флага конца оставляем
    # нетерминальным: его next_state настоящий (усечение по K_max)
    RTG = np.zeros_like(out["R"]); FIRST = np.zeros_like(out["D"])
    starts = np.r_[0, np.where(np.diff(out["EP"]) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(out["EP"])]
    for a0, b0 in zip(starts, ends):
        run = 0.0
        for i in range(b0 - 1, a0 - 1, -1):
            run = out["R"][i] + GAMMA * run * (1.0 - out["D"][i])
            RTG[i] = run
        FIRST[a0] = 1.0
    out["RTG"], out["FIRST"] = RTG, FIRST
    out["STATS"] = dict(data.get("STATS", {}), online_added=n_add, online_files=len(paths))
    print(f"онлайновые буферы: добавлено {n_add} переходов из {len(paths)} файлов; "
          f"всего {len(out['A'])}", flush=True)
    return out


ALL_SUBDIRS = ("burgers1d", "burgers2d", "grayscott", "heat2d_complexgeometry",
               "heat2d_longtime", "heat2d_multiscale", "heat2d_varyingcoef", "heatinv", "heatnd",
               "kuramoto_sivashinsky", "ns2d_backstep", "ns2d_liddriven", "ns2d_longtime",
               "poisson2d_classic", "poisson2d_manyarea", "poisson3d_complexgeometry",
               "poisson_boltzmann_2d", "poissoninv", "poissonnd", "wave1d",
               "wave2d_heterogeneous", "wave2d_longtime")
# УрЧП, которые на публичных данных не решает ни один метод (ошибка около 1 у всех):
# цепочки там неразличимы, в обучающую смесь по умолчанию не входят
UNSOLVED_SUBDIRS = ("burgers2d", "heat2d_longtime", "kuramoto_sivashinsky", "ns2d_longtime",
                    "poisson2d_manyarea", "wave2d_heterogeneous", "wave2d_longtime")


def resolve_subdirs(spec: str, holdout: str = ""):
    """'all' — все 22 папки; 'solvable' — без нерешаемых; иначе список через запятую.
    holdout — папки, которые надо исключить (задача, на которую проверяется перенос)."""
    if spec == "all":
        subs = list(ALL_SUBDIRS)
    elif spec == "solvable":
        subs = [x for x in ALL_SUBDIRS if x not in UNSOLVED_SUBDIRS]
    else:
        subs = [x.strip() for x in spec.split(",") if x.strip()]
    out = [x.strip() for x in holdout.split(",") if x.strip()]
    bad = [x for x in subs + out if x not in ALL_SUBDIRS]
    if bad:
        raise ValueError(f"неизвестные папки буфера: {bad}")
    subs = [x for x in subs if x not in out]
    if not subs:
        raise ValueError("после исключения не осталось ни одной папки буфера")
    return subs


def load_multi(subdirs, reward_form: str = "dlog", err_norm: str = "init", budget: int = 0,
               max_files: int = 0, data_dir=None, verbose: bool = True):
    """Офлайновый набор из нескольких УрЧП для переноса политики.

    Каждый буфер загружается с восстановлением цепочек; ошибка необученной сети E0
    берётся из track3/pde_meta.json. err_norm='init' делит ошибки на E0 задачи, и
    награды всех УрЧП оказываются в одних единицах (доли начальной ошибки);
    'none' оставляет ошибки как есть и передаёт E0 только в награду первого шага.
    Дополнительно отдаёт PDE (номер задачи перехода) и PDE_NAMES."""
    parts, names = [], []
    ep_off = 0
    keys = ("S", "A", "R", "S2", "D", "EP", "RTG", "FIRST", "ERR", "STEP", "SPENT")
    for k, sd in enumerate(subdirs):
        name = subdir_to_pde(sd)
        e0 = float(pde_meta()[name]["init_err"])
        eps = load_episodes(os.path.join(data_dir, sd) if data_dir else None, sd, max_files=max_files)
        if not eps:
            print(f"смесь УрЧП: {sd} пуст, пропущен", flush=True)
            continue
        d = episodes_to_arrays_chains(eps, reward_form=reward_form, verbose=False,
                                      init_err=(1.0 if err_norm == "init" else e0),
                                      budget=budget,
                                      err_scale=(e0 if err_norm == "init" else 1.0))
        d["EP"] = d["EP"] + ep_off
        ep_off = int(d["EP"].max()) + 1
        d["PDE"] = np.full(len(d["A"]), len(names), dtype=np.int64)
        parts.append(d)
        names.append(name)
        if verbose:
            print(f"смесь УрЧП: {sd} -> {name}: файлов {len(eps)}, цепочек "
                  f"{d['STATS']['chains']}, переходов {len(d['A'])}, E0={e0:.4g}, "
                  f"медиана ошибки после действия {float(np.median(d['ERR'])):.4g}", flush=True)
    if not parts:
        raise ValueError("смесь УрЧП пуста")
    out = {k: np.concatenate([d[k] for d in parts], 0) for k in keys + ("PDE",)}
    out["PDE_NAMES"] = names
    out["STATS"] = dict(pdes=len(names), chains=int(sum(d["STATS"]["chains"] for d in parts)))
    if verbose:
        print(f"смесь УрЧП: задач {len(names)}, переходов {len(out['A'])}, награда {reward_form}, "
              f"нормировка ошибки {err_norm}", flush=True)
    return out


def teacher_q(path, data, device):
    """Q-значения замороженного агента-учителя на всех переходах набора — цель для
    дистилляции в ученика с дешёвым состоянием. Учитель получает состояние в том виде,
    в каком обучался (режим состояния и контекст берутся из meta его чекпоинта).
    Возвращает (Q размера N x 27, meta учителя)."""
    import torch
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OUT_REPO, path, repo_type="dataset")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    if variant in ("cnx_smdp", "cnn_vqc") or ck.get("hl_gauss"):
        raise ValueError(f"учитель {variant}: дистилляция поддержана для обычных и квантильных Q-сетей")
    sd = ck["state_dict"]
    in_ch = 4
    for v in sd.values():
        if hasattr(v, "dim") and v.dim() == 4 and v.shape[2] <= 7:
            in_ch = int(v.shape[1]); break
    if variant == "stat_dqn":
        in_ch = int(sd["0.mlp.0.weight"].shape[1] // 8)
    nq = 32
    hw = sd.get("1.weight")
    if hw is not None and hw.dim() == 2 and hw.shape[0] % N_ACTIONS == 0:
        nq = max(1, hw.shape[0] // N_ACTIONS)
    net = QNet(variant, device, n_quantiles=nq, in_ch=in_ch)
    net.model.load_state_dict(sd)
    net.model.eval()
    meta = ck.get("meta") or {}
    tmode = meta.get("state_mode", "full")
    if tmode == "tasknorm":
        names = data.get("PDE_NAMES") or [subdir_to_pde(SUBDIR)]
        pde_idx = data.get("PDE", np.zeros(len(data["A"]), dtype=np.int64))
        S = np.array(data["S"], dtype=np.float32, copy=True)
        for k, name in enumerate(names):
            S[pde_idx == k] = apply_state_mode(S[pde_idx == k], "tasknorm", pde=name)
    else:
        S = apply_state_mode(data["S"], tmode, seed=101)
    if in_ch > 4:
        c, _ = chain_ctx(data, k_max=int(meta.get("ctx_kmax", 10)),
                         budget=int(meta.get("ctx_budget", 31000)),
                         with_err=(meta.get("ctx_err", "err") != "none"))
        if meta.get("pde_ctx"):
            if "PDE" not in data:
                raise ValueError("учитель обучен с описателем задачи: нужен набор --subdirs")
            c = np.concatenate([c, np.array([pde_desc(n) for n in data["PDE_NAMES"]],
                                            dtype=np.float32)[data["PDE"]]], 1)
        S = attach_ctx(S, c)
    if S.shape[1] != in_ch:
        raise ValueError(f"учитель ждёт {in_ch} каналов, собрано {S.shape[1]}")
    mean, std = np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32)
    out = np.zeros((len(S), N_ACTIONS), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(S), 512):
            x = torch.as_tensor((S[k:k + 512] - mean) / std, device=device).float()
            out[k:k + 512] = net.q_scalar(x).float().cpu().numpy()
    return out, meta


def split_by_episode(data, test_frac=0.2, split_seed=0):
    rng = np.random.default_rng(split_seed)
    eps = np.unique(data["EP"])
    order = rng.permutation(eps)
    counts = {ei: int((data["EP"] == ei).sum()) for ei in eps}
    total = sum(counts.values())
    test_eps, acc = [], 0
    for ei in order:
        if acc < test_frac * total:
            test_eps.append(ei); acc += counts[ei]
    test_mask = np.isin(data["EP"], test_eps)
    return ~test_mask, test_mask


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_encoder(kind: str, in_ch: int = 4):
    import torch
    import torch.nn as nn

    if kind == "cnn":
        class CNN(nn.Module):
            out_dim = 256
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 13
                    nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 6
                    nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU(),
                )
            def forward(self, x):
                return self.net(x)
        return CNN()

    if kind == "their":
        # ИХ боевой энкодер (RL/rl_utils/DQN_classes.py, идентичен ветке
        # rlpinn_pde_tolerance): 3 свёртки -> GAP -> MLP(64). Возвращает
        # кортеж (flat, h) — оборачиваем, чтобы отдавать h.
        from RL.rl_utils.DQN_classes import ConvEncoder

        class TheirEncoder(nn.Module):
            out_dim = 64
            def __init__(self):
                super().__init__()
                self.enc = ConvEncoder()
            def forward(self, x):
                return self.enc(x)[1]
        return TheirEncoder()

    if kind == "convnext":
        class Block(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
                self.ln = nn.LayerNorm(c)
                self.p1 = nn.Linear(c, 4 * c)
                self.p2 = nn.Linear(4 * c, c)
                self.act = nn.GELU()
            def forward(self, x):
                y = self.dw(x).permute(0, 2, 3, 1)
                y = self.p2(self.act(self.p1(self.ln(y)))).permute(0, 3, 1, 2)
                return x + y
        class ConvNeXtTiny(nn.Module):
            out_dim = 256
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(in_ch, 48, 2, stride=2)          # 13
                self.s1 = nn.Sequential(Block(48), Block(48))
                self.down = nn.Conv2d(48, 96, 2, stride=2)         # 6
                self.s2 = nn.Sequential(Block(96), Block(96))
                self.head = nn.Sequential(nn.Linear(96, 256), nn.ReLU())
            def forward(self, x):
                x = self.s2(self.down(self.s1(self.stem(x))))
                x = x.mean(dim=(2, 3))
                return self.head(x)
        return ConvNeXtTiny()

    if kind == "stat":
        class StatEncoder(nn.Module):
            """Сводные статистики карт вместо свёрток: по каждому каналу среднее,
            разброс, минимум, максимум, центр и квантили 10/50/90. Около 20 тыс.
            параметров против 250 тыс. у ConvNeXt — под буфер в несколько тысяч
            переходов. Мотивировка: на наших буферах тонкая геометрия карты почти
            не предсказывает застой (AUC 0.45–0.53), уровень — предсказывает."""
            out_dim = 128

            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(in_ch * 8, 128), nn.LayerNorm(128),
                                         nn.GELU(), nn.Linear(128, 128), nn.GELU())

            def forward(self, x):
                f = x.flatten(2)                                           # (B, C, H*W)
                qs = torch.quantile(f, torch.tensor([0.1, 0.5, 0.9], device=x.device,
                                                    dtype=f.dtype), dim=2)    # (3, B, C)
                c = x[:, :, x.shape[2] // 2, x.shape[3] // 2]
                feats = torch.cat([f.mean(2), f.std(2), f.amin(2), f.amax(2), c,
                                   qs[0], qs[1], qs[2]], dim=1)
                return self.mlp(feats)
        return StatEncoder()

    raise ValueError(kind)


def make_head(variant: str, in_dim: int, n_quantiles: int):
    import torch
    import torch.nn as nn

    if variant in ("their_dqn", "their_cql", "cnx_dueling"):
        # их дуэлинговая голова: Q = V + A - mean(A)
        from RL.rl_utils.DQN_classes import DuelingHead
        return DuelingHead(in_dim, N_ACTIONS)

    if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
        return nn.Linear(in_dim, N_ACTIONS * n_quantiles)
    if variant == "cnx_factored":
        class FactoredHead(nn.Module):
            """Q(o, l, e) = V + q_o[o] + q_ol[o, l] + q_oe[o, e]: действие — это
            оптимизатор, его шаг и число эпох. Слагаемые шага и длительности свои у
            каждого оптимизатора (их сетки различны), отброшено только взаимодействие
            шага с длительностью. Каждое слагаемое учится на втрое большем числе
            переходов, чем отдельный выход на действие; наружу те же 27 значений."""

            def __init__(self):
                super().__init__()
                self.v = nn.Linear(in_dim, 1)
                self.o = nn.Linear(in_dim, 3)
                self.ol = nn.Linear(in_dim, 9)
                self.oe = nn.Linear(in_dim, 9)
                a = torch.arange(N_ACTIONS)
                self.register_buffer("io", a // 9)
                self.register_buffer("iol", (a // 9) * 3 + (a % 9) // 3)
                self.register_buffer("ioe", (a // 9) * 3 + a % 3)

            def forward(self, z):
                return (self.v(z) + self.o(z)[:, self.io] + self.ol(z)[:, self.iol]
                        + self.oe(z)[:, self.ioe])
        return FactoredHead()
    if variant == "cnn_vqc":
        import pennylane as qml
        n_qubits, n_layers = 8, 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        qlayer = qml.qnn.TorchLayer(circuit, {"weights": wshape})

        class VQCHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.compress = nn.Linear(in_dim, n_qubits)
                self.q = qlayer
                self.out = nn.Linear(n_qubits, N_ACTIONS)
            def forward(self, z):
                z = torch.tanh(self.compress(z)) * math.pi / 2
                return self.out(self.q(z).float())
        return VQCHead()
    return nn.Linear(in_dim, N_ACTIONS)


class QNet:
    """Encoder + head with a target copy; variant-specific loss."""

    def __init__(self, variant, device, n_quantiles=32, in_ch=4):
        import torch
        import torch.nn as nn
        if variant in ("their_dqn", "their_cql"):
            enc_kind = "their"
        elif variant in ("convnext_dqn", "cnx_cql", "cnx_cql_qr", "cnx_dueling",
                         "cnx_bcq", "cnx_bbf", "cnx_qrdqn", "cnx_factored"):
            enc_kind = "convnext"
        elif variant == "stat_dqn":
            enc_kind = "stat"
        else:
            enc_kind = "cnn"
        self.variant = variant
        self.nq = n_quantiles
        self.enc = make_encoder(enc_kind, in_ch)
        self.head = make_head(variant, self.enc.out_dim, n_quantiles)
        self.v_head = nn.Linear(self.enc.out_dim, 1) if variant == "cnn_iql" else None
        mods = [self.enc, self.head] + ([self.v_head] if self.v_head is not None else [])
        self.model = nn.ModuleList(mods).to(device)
        import copy
        self.target = copy.deepcopy(self.model).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.device = device

    def params(self):
        return self.model.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def _q(self, model, x):
        z = model[0](x)
        out = model[1](z)
        if self.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
            return out.view(-1, N_ACTIONS, self.nq)
        return out

    def q_online(self, x):
        return self._q(self.model, x)

    def q_target(self, x):
        return self._q(self.target, x)

    def v_online(self, x):
        return self.model[2](self.model[0](x)).squeeze(-1)

    def q_scalar(self, x):
        """(B, N_ACTIONS) scalar Q for metrics/policies (mean over quantiles)."""
        q = self.q_online(x)
        return q.mean(-1) if self.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn") else q

    def q_cvar(self, x, alpha=0.25):
        import torch  # module-level `torch` only exists after main(); keep importable
        q = self.q_online(x)          # (B, A, nq), quantiles unsorted -> sort
        qs, _ = torch.sort(q, dim=-1)
        k = max(1, int(self.nq * alpha))
        return qs[..., :k].mean(-1)

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(), self.target.parameters()):
                tp.mul_(1 - tau).add_(tau * p)


# --------------------------------------------------------------------------
# Targets: n-step returns, potential-based shaping, SMDP discount
# --------------------------------------------------------------------------

def action_epochs(a) -> int:
    """Сколько эпох тратит действие a. Кодировка a = opt*9 + lr*3 + ep."""
    return EPOCHS_TABLE[(int(a) // 9) * 3 + (int(a) % 3)]


def build_targets(data, args, gamma):
    """n-шаговые цели + PBRS + SMDP-дисконт.

    Возвращает (NR, NG, NIDX, ND):
        NR[i]   сумма (возможно преобразованных) наград на m шагах вперёд
        NG[i]   произведение дисконтов на этих m шагах — множитель бутстрэпа
        NIDX[i] индекс перехода, из next_state которого делается бутстрэп
        ND[i]   признак терминальности в точке бутстрэпа
    Цель обучения: tgt = NR + NG * (1 - ND) * Q(s'_{NIDX}).

    При n_step=1, pbrs=0 и выключенном smdp это в точности прежние
    (R, gamma, i, D), то есть поведение не меняется.

    Награда в буфере — «absolute» (RL/rl_environment.compute_reward):
    r_t = -(coeff_op*err_op + coeff_bnd*err_bnd) состояния s_{t+1}.
    Проверено на данных: все 26 858 наград строго отрицательны. Поэтому
    потенциал восстанавливается точно: Ф(s_{t+1}) = -log10(-r_t), а
    Ф(s_t) = -log10(-r_{t-1}). У первого перехода эпизода Ф(s_0) неизвестен
    и добавка обнуляется — так же, как в онлайновой реализации.
    """
    R, D, EP, A = data["R"], data["D"], data["EP"], data["A"]
    n = len(R)

    gam = np.full(n, float(gamma), dtype=np.float64)
    if getattr(args, "smdp", False):
        cost = np.array([action_epochs(a) for a in A], dtype=np.float64)
        gam = float(gamma) ** (cost / float(args.smdp_scale))

    R_eff = R.astype(np.float64).copy()
    w_pbrs = float(getattr(args, "pbrs", 0.0) or 0.0)
    if w_pbrs:
        def phi(err):
            return -math.log10(max(float(err), 1e-8))
        for ei in np.unique(EP):
            idx = np.where(EP == ei)[0]
            for pos in range(1, len(idx)):
                i, prev = idx[pos], idx[pos - 1]
                # ошибка s' — из награды самого перехода, ошибка s — из предыдущей.
                # При chain-fix награда может быть разностью, поэтому ошибки берём
                # из массива ERR (там же снят штраф -1 за неудачную цепочку)
                if "ERR" in data:
                    R_eff[i] += w_pbrs * (gam[i] * phi(data["ERR"][i]) - phi(data["ERR"][prev]))
                else:
                    R_eff[i] += w_pbrs * (gam[i] * phi(-R[i]) - phi(-R[prev]))

    nst = max(1, int(getattr(args, "n_step", 1) or 1))
    NR = np.zeros(n, dtype=np.float64)
    NG = np.zeros(n, dtype=np.float64)
    NIDX = np.arange(n, dtype=np.int64)
    ND = np.zeros(n, dtype=np.float64)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        L = len(idx)
        for pos in range(L):
            i = idx[pos]
            m = min(nst, L - pos)
            acc, g = 0.0, 1.0
            for k in range(m):
                j = idx[pos + k]
                acc += g * R_eff[j]
                g *= gam[j]
            last = idx[pos + m - 1]
            NR[i], NG[i], NIDX[i], ND[i] = acc, g, last, D[last]
    return (NR.astype(np.float32), NG.astype(np.float32), NIDX,
            ND.astype(np.float32))


# группа симметрий латентной сетки. Сетка строится на квадрате [-1.2, 1.2]^2
# с одинаковым шагом по обеим осям (plot_loss_surface.py: min_y, max_y =
# min_x, max_x; один step_size), 26 узлов симметричны относительно нуля,
# поэтому D4 переставляет узлы точно. Базис 2-мерного латента автоэнкодера
# обучается заново на каждом эпизоде и канонической ориентации не имеет.
# Проверено на буфере ns2d (3229 карт): центроид области минимума
# (12.47, 12.48) при симметричном центре 12.5, баланс масс верх/низ 0.970 и
# лево/право 0.979, расхождение средней карты с её отражениями 0.011-0.031
# при шуме половина-к-половине 0.022.
def d4_apply(x, g: int):
    """g в 0..7: g>=4 — отражение по последней оси, затем g%4 поворотов."""
    if g >= 4:
        x = torch.flip(x, dims=[-1])
    k = g % 4
    return torch.rot90(x, k, dims=[-2, -1]) if k else x


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_smdp(data, train_mask, args, seed):
    """Полный стек SMDP-BBF-CQL (см. advanced_agents.py)."""
    import torch.nn.functional as F
    from advanced_agents import SmdpAgent, fit_behaviour, ACTION_DT

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    agent = SmdpAgent(device)
    opt = torch.optim.AdamW(agent.params(), lr=1e-4, weight_decay=0.1)

    idx = np.where(train_mask)[0]
    mean = data["S"][idx].mean(axis=(0, 2, 3), keepdims=True)
    std = data["S"][idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    S = torch.as_tensor((data["S"] - mean) / std, device=device).float()
    S2 = torch.as_tensor((data["S2"] - mean) / std, device=device).float()
    A = torch.as_tensor(data["A"], device=device)
    R = torch.as_tensor(data["R"], device=device)
    D = torch.as_tensor(data["D"], device=device)
    DT = torch.as_tensor(ACTION_DT[data["A"]], device=device)      # длительность действия

    fit_behaviour(agent, S[idx], data["A"][idx], device)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    bs = args.batch_size
    total = args.epochs * max(1, len(idx) // bs)
    for step in range(total):
        b = torch.as_tensor(rng.integers(0, len(idx), size=bs), device=device).long()
        b = torch.as_tensor(idx, device=device)[b]
        gamma = 0.97 + min(1.0, step / 2000) * (0.997 - 0.97)      # отжиг gamma
        with torch.no_grad():
            q_next_online = agent.q_scalar(S2[b])
            mask = agent.bcq_mask(S2[b])                            # BCQ-поддержка
            a_star = q_next_online.masked_fill(~mask, -1e9).argmax(-1)
            tl, _ = agent.target.all_logits(S2[b])
            tq = agent.hlg.to_scalar(tl)                            # (K,B,A)
            k1, k2 = rng.choice(agent.n_heads, size=2, replace=False)
            q_next = torch.minimum(tq[k1], tq[k2]).gather(-1, a_star[:, None])[:, 0]
            y = (R[b] + (gamma ** DT[b]) * (1 - D[b]) * q_next).clamp(-20.0, 2.0)
        logits, h = agent.net.all_logits(S[b])
        ii = A[b][None, :, None, None].expand(agent.n_heads, -1, 1, agent.net.n_bins)
        taken = logits.gather(2, ii).squeeze(2)
        td = torch.stack([agent.hlg.loss(taken[k], y) for k in range(agent.n_heads)]).mean()
        q_all = agent.hlg.to_scalar(logits).mean(0)
        cql = (torch.logsumexp(q_all, -1) - q_all.gather(-1, A[b][:, None])[:, 0]).mean()
        alpha_cql = 0.5
        phi = agent.net.act_emb(agent.net.action_feats)[A[b]]
        h_hat = agent.net.spr_tr(torch.cat([h, phi], -1))
        with torch.no_grad():
            h_tgt = agent.target.spr_proj(agent.target.embed(S2[b]))
        spr = -F.cosine_similarity(agent.net.spr_proj(h_hat), h_tgt, dim=-1).mean()
        loss = td + alpha_cql * cql + 1.0 * spr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 10.0)
        opt.step(); agent.soft_update()
        if (step + 1) % 2000 == 0:                                  # BBF-сброс
            agent.shrink_and_perturb()
            opt = torch.optim.AdamW(agent.params(), lr=1e-4, weight_decay=0.1)
        if (step + 1) % max(1, total // 4) == 0:
            print(f"  [cnx_smdp s{seed}] шаг {step+1}/{total} td={td.item():.4f} "
                  f"cql={cql.item():.4f} spr={spr.item():.4f}", flush=True)
    return agent, dict(mean=mean, std=std), time.time() - t0


def train_variant(variant, data, train_mask, args, seed):
    if variant == "cnx_smdp":
        return train_smdp(data, train_mask, args, seed)
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    in_ch = int(data["S"].shape[1])
    net = QNet(variant, device, in_ch=in_ch)
    opt = torch.optim.Adam(net.params(), lr=1e-3,
                           weight_decay=float(getattr(args, "l2", 0.0) or 0.0))

    idx = np.where(train_mask)[0]
    mean = data["S"][idx].mean(axis=(0, 2, 3), keepdims=True)
    std = data["S"][idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    if in_ch > 4:
        # контекстные каналы уже лежат в [-1, 1]: как и в онлайне, их не нормируем
        mean[:, 4:] = 0.0
        std[:, 4:] = 1.0
    # карты без разброса (режим blind) не нормируем: иначе деление на 1e-6
    flat_ch = std[0, :4, 0, 0] < 1e-5
    mean[0, :4][flat_ch] = 0.0
    std[0, :4][flat_ch] = 1.0

    def to_t(x):
        return torch.as_tensor(x, device=device)

    def norm(s):
        return (s - mean) / std

    S = to_t(norm(data["S"])).float(); S2 = to_t(norm(data["S2"])).float()
    A = to_t(data["A"]); R = to_t(data["R"]); D = to_t(data["D"])

    # n-шаговые цели / PBRS / SMDP: при значениях по умолчанию совпадают с (R, GAMMA, D)
    nr_np, ng_np, nidx_np, nd_np = build_targets(data, args, GAMMA)
    NR = to_t(nr_np); NG = to_t(ng_np); ND = to_t(nd_np)
    NIDX = to_t(nidx_np).long()
    # границы ценности: при награде-разности сумма будущих наград не больше текущей
    # ошибки (улучшить можно не больше, чем до нуля) и не меньше нуля (можно
    # остановиться). Без границ офлайновый DQN на этих данных завышает Q втрое
    # выше физического предела (проверено: Q(s0)=1.5 при максимуме 0.49)
    vbound = bool(getattr(args, "value_bound", False))
    UB = to_t(data["ERR"][nidx_np]).float() if vbound else None
    # дистилляция: ученик регрессирует Q учителя по всем 27 действиям, без TD-цели
    QT = to_t(data["QT"]).float() if "QT" in data else None
    aug_n = 8 if getattr(args, "aug", "none") == "d4" else 1
    w_margin = float(getattr(args, "dqfd", 0.0) or 0.0)
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    ckpts = []

    bs = args.batch_size
    n_epochs = args.epochs
    taus = None
    if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
        taus = (torch.arange(net.nq, device=device, dtype=torch.float32) + 0.5) / net.nq

    bcq_behaviour = None
    if variant == "cnx_bcq":
        from advanced_agents import fit_behaviour
        import torch.nn as nn
        bcq_behaviour = nn.Sequential(nn.Linear(4 * 26 * 26, 256), nn.LayerNorm(256),
                                      nn.GELU(), nn.Linear(256, N_ACTIONS)).to(device)
        class _Holder: pass
        _h = _Holder(); _h.behaviour = bcq_behaviour
        fit_behaviour(_h, S[np.where(train_mask)[0]], data["A"][train_mask], device)
        print(f"  [cnx_bcq s{seed}] модель поведения обучена", flush=True)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    best_td, stall = float("inf"), 0
    test_idx = np.where(~train_mask)[0]
    for epoch in range(n_epochs):
        order = rng.permutation(idx)
        for k in range(0, len(order), bs):
            b = to_t(order[k:k + bs]).long()
            # r/d/gm — n-шаговые: награда за m шагов, терминальность и дисконт
            # в точке бутстрэпа; s2 — next_state именно того перехода (NIDX)
            s, a, r, d, gm = S[b], A[b], NR[b], ND[b], NG[b]
            s2 = S2[NIDX[b]]
            s_raw = s                      # неаугментированное s: маска BCQ и margin
            if aug_n > 1:                  # одно преобразование на батч, общее для s и s'
                g = int(rng.integers(aug_n))
                s = d4_apply(s, g); s2 = d4_apply(s2, g)

            if QT is not None:
                loss = F.mse_loss(net.q_online(s), QT[b])
            elif variant == "cnn_iql":
                with torch.no_grad():
                    q_t = net.q_target(s)
                    q_data_t = q_t.gather(1, a[:, None]).squeeze(1)
                v = net.v_online(s)
                diff = q_data_t - v
                w = torch.where(diff > 0, torch.full_like(diff, 0.7), torch.full_like(diff, 0.3))
                v_loss = (w * diff ** 2).mean()
                with torch.no_grad():
                    v2 = net.v_online(s2)
                    tgt = r + gm * (1 - d) * v2
                q = net.q_online(s).gather(1, a[:, None]).squeeze(1)
                loss = v_loss + F.mse_loss(q, tgt)
            elif variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
                q = net.q_online(s)                                   # (B,A,nq)
                q_data = q.gather(1, a[:, None, None].expand(-1, 1, net.nq)).squeeze(1)
                with torch.no_grad():
                    a2 = net.q_online(s2).mean(-1).argmax(1)
                    q2 = net.q_target(s2).gather(
                        1, a2[:, None, None].expand(-1, 1, net.nq)).squeeze(1)
                    tgt = r[:, None] + (gm * (1 - d))[:, None] * q2    # (B,nq)
                u = tgt[:, None, :] - q_data[:, :, None]               # (B,nq_pred,nq_tgt)
                huber = torch.where(u.abs() <= 1.0, 0.5 * u ** 2, u.abs() - 0.5)
                loss = (torch.abs(taus[None, :, None] - (u.detach() < 0).float()) * huber).mean()
                if variant == "cnx_cql_qr":
                    qm = q.mean(-1)
                    loss = loss + args.cql_alpha * (
                        torch.logsumexp(qm, dim=1) - qm.gather(1, a[:, None]).squeeze(1)
                    ).mean()
            else:
                q = net.q_online(s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = net.q_online(s2).argmax(1)
                    q2 = net.q_target(s2).gather(1, a2[:, None]).squeeze(1)
                    if vbound:
                        q2 = torch.minimum(q2.clamp_min(0.0), UB[b])
                    tgt = r + gm * (1 - d) * q2
                loss = F.mse_loss(q, tgt)
                if variant == "cnx_bcq":      # discrete BCQ: маска поддержки
                    # модель поведения обучена на неаугментированных состояниях,
                    # поэтому маска берётся с s_raw; сама маска считается
                    # инвариантной к D4 — это и есть предпосылка аугментации
                    with torch.no_grad():
                        pb = bcq_behaviour(s_raw.flatten(1)).softmax(-1)
                        keep = pb >= 0.3 * pb.max(-1, keepdim=True).values
                    qs_all = net.q_online(s)
                    loss = loss + 0.5 * (qs_all.masked_fill(keep, 0.0).clamp_min(0) ** 2).mean()
                if variant in ("cnn_cql", "cnx_cql", "their_cql"):
                    qs = net.q_online(s)
                    loss = loss + args.cql_alpha * (
                        torch.logsumexp(qs, dim=1) - qs.gather(1, a[:, None]).squeeze(1)
                    ).mean()

            if w_margin and QT is None:
                # DQfD, большой отступ: J_E = max_a[Q(s,a)+L(a_E,a)] - Q(s,a_E).
                # Все переходы буфера считаются демонстрацией (чистый офлайн).
                qs_m = net.q_scalar(s)
                marg = torch.full_like(qs_m, float(args.dqfd_margin))
                marg.scatter_(1, a[:, None], 0.0)
                loss = loss + w_margin * (
                    (qs_m + marg).max(1).values
                    - qs_m.gather(1, a[:, None]).squeeze(1)).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            net.soft_update()
            if variant == "cnx_bbf" and (epoch * 1000 + k) % 4000 == 3999:
                # BBF: shrink-and-perturb — ствол сохраняет 50%, головы заново
                import copy as _copy
                fresh = QNet(variant, device, in_ch=in_ch)
                with torch.no_grad():
                    for (nm, p), pf in zip(net.model.named_parameters(),
                                           fresh.model.parameters()):
                        a_keep = 0.0 if nm.startswith("1") else 0.5
                        p.mul_(a_keep).add_((1 - a_keep) * pf)
                net.target.load_state_dict(net.model.state_dict())
                opt = torch.optim.Adam(net.params(), lr=1e-3)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  [{variant} s{seed}] epoch {epoch+1}/{n_epochs} loss={loss.item():.4f}", flush=True)
        if args.plateau_patience and (epoch + 1) % 25 == 0:
            with torch.no_grad():
                b = to_t(test_idx).long()
                qs = net.q_scalar(S[b]).gather(1, A[b][:, None]).squeeze(1)
                a2 = net.q_scalar(S2[b]).argmax(1)
                q2 = (net.q_target(S2[b]).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                      else net.q_target(S2[b])).gather(1, a2[:, None]).squeeze(1)
                td = float(((qs - (R[b] + GAMMA * (1 - D[b]) * q2)) ** 2).mean().sqrt())
            if td < best_td - 1e-3:
                best_td, stall = td, 0
            else:
                stall += 1
            print(f"  [{variant} s{seed}] plateau-check ep{epoch+1}: td={td:.3f} best={best_td:.3f} stall={stall}", flush=True)
            if stall >= args.plateau_patience:
                print(f"  [{variant} s{seed}] plateau reached at epoch {epoch+1}", flush=True)
                break
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            import copy as _c
            ckpts.append((epoch + 1, _c.deepcopy(net.model.state_dict())))

    train_time = time.time() - t0
    net.extra = {}
    if QT is not None:
        with torch.no_grad():
            b = to_t(test_idx).long()
            allow = torch.ones(N_ACTIONS, dtype=torch.bool, device=device)
            for i in (getattr(args, "_teacher_mask", None) or []):
                allow[int(i)] = False
            qs = net.q_scalar(S[b]).masked_fill(~allow[None], -1e9)
            qt = QT[b].masked_fill(~allow[None], -1e9)
            agree = float((qs.argmax(1) == qt.argmax(1)).float().mean())
            # потеря ценности при замене действий учителя действиями ученика (в Q учителя)
            regret = float((qt.max(1).values - qt.gather(1, qs.argmax(1)[:, None]).squeeze(1)).mean())
        net.extra = {"distill_agree": round(agree, 4), "distill_regret_q": round(regret, 5)}
        print(f"  [{variant} s{seed}] дистилляция: совпадение жадных действий с учителем на "
              f"отложенных переходах {agree:.3f}, потеря ценности {regret:.4f}", flush=True)
    if getattr(args, "fqe_select", False):
        # отбор чекпоинта по FQE на отложенных эпизодах: сравниваем ценность
        # жадной политики каждого снимка, берём лучший. Последняя эпоха всегда
        # в списке, поэтому отбор не может оказаться хуже обычного финала по
        # этому критерию (но по метрике в среде — может, критерий приблизителен)
        import copy as _c
        last_ep = epoch + 1
        pool = list(ckpts)
        if not pool or pool[-1][0] != last_ep:      # финальные веса всегда в пуле
            pool.append((last_ep, _c.deepcopy(net.model.state_dict())))
        stats_sel = dict(mean=mean, std=std)
        pol = "cvar" if (variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                         and getattr(args, "select_policy", "mean") == "cvar") else "mean"
        scored = []
        for pi, (ep_no, sd) in enumerate(pool):
            net.model.load_state_dict(sd)
            fq = fqe(net, stats_sel, data, train_mask, ~train_mask, pol, args, seed)
            v = fq["fqe_value_init"]
            if v is None:
                v = fq["fqe_value_all"]
            scored.append((float(v), pi))
            print(f"  [{variant} s{seed}] чекпоинт эпоха {ep_no}: FQE={v:.4f}", flush=True)
        best_v, best_pi = max(scored)
        best_ep = pool[best_pi][0]
        net.model.load_state_dict(pool[best_pi][1])
        net.target.load_state_dict(net.model.state_dict())
        net.extra = {"fqe_select_epoch": best_ep, "fqe_select_value": best_v,
                     "fqe_select_pool": [pool[pi][0] for _, pi in scored]}
        print(f"  [{variant} s{seed}] выбран чекпоинт эпохи {best_ep} (FQE={best_v:.4f})",
              flush=True)
    return net, dict(mean=mean, std=std), train_time


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(net, stats, data, test_mask, policy="mean"):
    import torch
    from scipy.stats import spearmanr

    device = net.device
    idx = np.where(test_mask)[0]
    S = torch.as_tensor((data["S"] - stats["mean"]) / stats["std"], device=device).float()
    S2 = torch.as_tensor((data["S2"] - stats["mean"]) / stats["std"], device=device).float()

    def batched(fn, X, bs=512):
        outs = []
        with torch.no_grad():
            for k in range(0, len(X), bs):
                outs.append(fn(X[k:k + bs]))
        return torch.cat(outs)

    qfun = (lambda x: net.q_cvar(x)) if policy == "cvar" else (lambda x: net.q_scalar(x))
    q_all = batched(qfun, S[idx])
    a_log = torch.as_tensor(data["A"][idx], device=device)
    q_data = q_all.gather(1, a_log[:, None]).squeeze(1).cpu().numpy()

    # holdout TD error (double-DQN residual on scalar Q)
    with torch.no_grad():
        q_scal = batched(lambda x: net.q_scalar(x), S[idx])
        a2 = batched(lambda x: net.q_scalar(x), S2[idx]).argmax(1)
        q2t = batched(lambda x: net.q_target(x).mean(-1)
                      if net.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                      else net.q_target(x), S2[idx])
        q2 = q2t.gather(1, a2[:, None]).squeeze(1)
        r = torch.as_tensor(data["R"][idx], device=device)
        d = torch.as_tensor(data["D"][idx], device=device)
        td = (q_scal.gather(1, a_log[:, None]).squeeze(1)
              - (r + GAMMA * (1 - d) * q2)).cpu().numpy()

    rho = spearmanr(q_data, data["RTG"][idx]).statistic
    greedy = q_all.argmax(1).cpu().numpy()
    agree = float((greedy == data["A"][idx]).mean())
    ent = 0.0
    counts = np.bincount(greedy, minlength=N_ACTIONS).astype(float)
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return dict(td_error=float(np.sqrt((td ** 2).mean())),
                spearman_q_rtg=float(rho),
                agree_behavior=agree,
                policy_entropy=ent), greedy


def fqe(net, stats, data, train_mask, test_mask, policy, args, seed):
    """Fitted Q Evaluation of the variant's greedy policy with a FIXED
    CNN architecture (identical for every variant)."""
    import torch
    import torch.nn.functional as F

    device = net.device
    torch.manual_seed(seed + 10_000)
    enc = make_encoder("cnn", in_ch=int(data["S"].shape[1])).to(device)
    head = torch.nn.Linear(enc.out_dim, N_ACTIONS).to(device)
    model = torch.nn.ModuleList([enc, head])
    import copy
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    S = torch.as_tensor((data["S"] - stats["mean"]) / stats["std"], device=device).float()
    S2 = torch.as_tensor((data["S2"] - stats["mean"]) / stats["std"], device=device).float()
    A = torch.as_tensor(data["A"], device=device)
    R = torch.as_tensor(data["R"], device=device)
    D = torch.as_tensor(data["D"], device=device)

    # policy actions on next states (from the trained variant net)
    qfun = (lambda x: net.q_cvar(x)) if policy == "cvar" else (lambda x: net.q_scalar(x))
    with torch.no_grad():
        pi_s2 = []
        for k in range(0, len(S2), 512):
            pi_s2.append(qfun(S2[k:k + 512]).argmax(1))
        pi_s2 = torch.cat(pi_s2)

    idx = np.where(train_mask)[0]
    rng = np.random.default_rng(seed + 1)
    for epoch in range(args.fqe_epochs):
        order = rng.permutation(idx)
        for k in range(0, len(order), args.batch_size):
            b = torch.as_tensor(order[k:k + args.batch_size], device=device).long()
            q = head(enc(S[b])).gather(1, A[b][:, None]).squeeze(1)
            with torch.no_grad():
                q2 = target[1](target[0](S2[b])).gather(1, pi_s2[b][:, None]).squeeze(1)
                tgt = R[b] + GAMMA * (1 - D[b]) * q2
            loss = F.mse_loss(q, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.mul_(0.995).add_(0.005 * p)

    tidx = np.where(test_mask)[0]
    with torch.no_grad():
        vals = []
        for k in range(0, len(tidx), 512):
            b = torch.as_tensor(tidx[k:k + 512], device=device).long()
            qv = head(enc(S[b]))
            pa = []
            for kk in range(0, len(b), 512):
                pa.append(qfun(S[b][kk:kk + 512]).argmax(1))
            pa = torch.cat(pa)
            vals.append(qv.gather(1, pa[:, None]).squeeze(1))
        vals = torch.cat(vals).cpu().numpy()
    first = data["FIRST"][tidx].astype(bool)
    vs = np.sort(vals)
    k25 = max(1, int(0.25 * len(vs)))
    return dict(fqe_value_all=float(vals.mean()),
                fqe_value_init=float(vals[first].mean()) if first.any() else None,
                fqe_cvar25=float(vs[:k25].mean()))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def upload_result(row: dict, name: str):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF write token; result kept local only", flush=True)
        return
    from huggingface_hub import upload_file
    import io
    payload = json.dumps(row, indent=1).encode()
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(payload),
                        path_in_repo=f"rl_arch/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch: {name}")
            print(f"uploaded rl_arch/{name}.json", flush=True)
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(5)


def main():
    global GAMMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["cnn_dqn", "convnext_dqn", "cnn_cql", "cnn_iql",
                             "cnn_qrdqn", "cnn_vqc", "cnx_cql", "cnx_cql_qr",
                             "their_dqn", "their_cql", "cnx_dueling",
                             "cnx_bcq", "cnx_bbf", "cnx_smdp", "cnx_qrdqn",
                             "cnx_factored", "stat_dqn"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--fqe-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--cql-alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--subdir", default=SUBDIR,
                    help="Buffer folder in the HF dataset (e.g. poisson3d_complexgeometry)")
    ap.add_argument("--fix-next-state", action="store_true",
                    help="Reconstruct s'=state[t+1] (poisson_boltzmann_2d dump has s'==s)")
    ap.add_argument("--chain-fix", action="store_true",
                    help="Восстановить структуру цепочек в буфере: границы по done, "
                         "терминал на конце цепочки (в дампах done=-1 давал 1-D=2), "
                         "s'=state[t+1] там, где next_state записан копией state")
    ap.add_argument("--reward-form", default="logged", choices=list(REWARD_FORMS),
                    help="Форма награды офлайновых переходов (нужен --chain-fix): "
                         "logged — как в дампе, level — минус ошибка, delta — разность "
                         "ошибок как в онлайне, model — награда агента авторов")
    ap.add_argument("--init-err", type=float, default=None,
                    help="Ошибка необученной сети (rmse+brmse): награда первого шага "
                         "при --reward-form delta = init_err - E_1 вместо нуля")
    ap.add_argument("--episode-budget", type=int, default=0,
                    help="Обрезать офлайновые цепочки по бюджету эпох (нужен --chain-fix)")
    ap.add_argument("--value-bound", action="store_true",
                    help="Ограничить бутстреп-цель физическими пределами: 0 <= V(s') <= "
                         "ошибка в s' (нужны --chain-fix и --reward-form delta, без --pbrs)")
    ap.add_argument("--extra-buffers", default="",
                    help="Онлайновые буферы прошлых прогонов через запятую (локальные пути "
                         "или имена в rl_arch/buffers_online/): переобучение на всех "
                         "собранных данных. Нужен --chain-fix")
    ap.add_argument("--save-model", action="store_true",
                    help="Save agent checkpoint and upload to HF rl_arch/models/")
    ap.add_argument("--plateau-patience", type=int, default=0,
                    help="Stop when holdout TD stops improving for N checks (every 25 epochs); 0=off")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # --- методы из статей (по умолчанию всё выключено: прежнее поведение) ---
    ap.add_argument("--aug", default="none", choices=["none", "d4"],
                    help="D4-аугментация карт: 8 симметрий квадратной латентной сетки")
    ap.add_argument("--n-step", type=int, default=1,
                    help="n-шаговые возвраты внутри эпизода (1 = как раньше)")
    ap.add_argument("--dqfd", type=float, default=0.0,
                    help="Вес отступного лосса DQfD (0 = выключен)")
    ap.add_argument("--dqfd-margin", type=float, default=0.8,
                    help="Отступ L(a_E,a) в лоссе DQfD")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="weight_decay в Adam (L2 из DQfD)")
    ap.add_argument("--pbrs", type=float, default=0.0,
                    help="Вес потенциального преобразования награды, Ф=-log10(ошибка)")
    ap.add_argument("--smdp", action="store_true",
                    help="Дисконт по длительности действия: gamma^(эпохи/scale)")
    ap.add_argument("--smdp-scale", type=float, default=100.0)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="Снимать чекпоинт каждые N эпох (для --fqe-select)")
    ap.add_argument("--fqe-select", action="store_true",
                    help="Выбрать чекпоинт с лучшей ценностью по FQE на отложенных эпизодах")
    ap.add_argument("--select-policy", default="mean", choices=["mean", "cvar"])
    ap.add_argument("--state-mode", default="full", choices=list(STATE_MODES),
                    help="Трек 3, лестница состояния: full карты; blind без карт; level только "
                         "уровень потерь; shuffle перемешанные пиксели")
    ap.add_argument("--scalar-ctx", action="store_true",
                    help="Пять скалярных каналов контекста (шаг, бюджет, прошлое действие, ошибка); "
                         "требует --chain-fix")
    ap.add_argument("--ctx-err", action="store_true",
                    help="Заполнять канал ошибки истинной ошибкой (привилегированная информация); "
                         "без флага канал нулевой и контекст содержит только время")
    ap.add_argument("--ctx-kmax", type=int, default=10)
    ap.add_argument("--ctx-budget", type=int, default=0,
                    help="Нормировка доли бюджета в контексте; 0 = --episode-budget либо 31000")
    ap.add_argument("--subdirs", default="",
                    help="Трек 3, перенос: обучение на смеси УрЧП. 'all', 'solvable' или список "
                         "папок буфера через запятую. Включает --chain-fix; награда dlog, если "
                         "--reward-form не задан")
    ap.add_argument("--holdout", default="",
                    help="Папки буфера, исключаемые из смеси (задача проверки переноса)")
    ap.add_argument("--per-pde-files", type=int, default=0,
                    help="Не больше стольких файлов буфера на УрЧП (0 = все): память и время")
    ap.add_argument("--err-norm", default="init", choices=["init", "none"],
                    help="Смесь УрЧП: init — ошибки в долях ошибки необученной сети; none — как есть")
    ap.add_argument("--pde-ctx", action="store_true",
                    help="Шесть каналов описателя задачи (размерности, число слагаемых потерь, "
                         "время, обратная задача); требует --scalar-ctx")
    ap.add_argument("--distill-from", default="",
                    help="Трек 3: чекпоинт учителя (локальный путь или путь в HF-датасете результатов). "
                         "Ученик с дешёвым состоянием (--state-mode, --scalar-ctx) регрессирует Q "
                         "учителя на переходах буфера вместо TD-обучения; в итог пишется доля "
                         "совпадения жадных действий. Нужен --chain-fix")
    ap.add_argument("--model-tag", default="",
                    help="Суффикс в именах результатов и чекпоинта: арм не перетирает базу")
    args = ap.parse_args()
    if args.fqe_select and not args.ckpt_every:
        ap.error("--fqe-select требует --ckpt-every N")
    if args.n_step < 1:
        ap.error("--n-step должен быть >= 1")
    if args.smoke:
        args.epochs, args.fqe_epochs = 2, 2

    GAMMA = args.gamma
    print(f"loading episodes... (gamma={GAMMA})", flush=True)
    multi = []
    if args.subdirs:
        given = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("--"))
        args.chain_fix = True
        if "--reward-form" not in given:
            args.reward_form = "dlog"
        if args.reward_form not in ("delta", "dlog"):
            ap.error("--subdirs: награда должна быть delta или dlog (масштабы ошибок у УрЧП разные)")
        if args.init_err is not None:
            ap.error("--subdirs: --init-err не задаётся, E0 каждой задачи берётся из track3/pde_meta.json")
        if args.extra_buffers:
            ap.error("--subdirs несовместим с --extra-buffers")
        if args.state_mode == "loss":
            ap.error("--state-mode loss офлайн недоступен: в буферах нет обучающего лосса")
        multi = resolve_subdirs(args.subdirs, args.holdout)
    elif args.holdout or args.pde_ctx:
        ap.error("--holdout и --pde-ctx работают только вместе с --subdirs")
    if args.state_mode == "loss":
        ap.error("--state-mode loss офлайн недоступен: в буферах нет обучающего лосса")
    if args.state_mode == "tasknorm" and not args.chain_fix:
        ap.error("--state-mode tasknorm требует --chain-fix (нужна принадлежность состояний задаче)")
    if args.reward_form != "logged" and not args.chain_fix:
        ap.error("--reward-form требует --chain-fix")
    if (args.init_err is not None or args.episode_budget) and not args.chain_fix:
        ap.error("--init-err и --episode-budget требуют --chain-fix")
    if args.value_bound and (not args.chain_fix or args.reward_form != "delta" or args.pbrs):
        ap.error("--value-bound требует --chain-fix --reward-form delta и несовместим с --pbrs")
    if args.value_bound and multi and args.err_norm != "init":
        ap.error("--value-bound на смеси УрЧП требует --err-norm init: граница выведена в единицах ошибки")
    if multi:
        data = load_multi(multi, reward_form=args.reward_form, err_norm=args.err_norm,
                          budget=args.episode_budget, max_files=args.per_pde_files,
                          data_dir=args.data_dir)
        episodes = list(range(int(data["EP"].max()) + 1))
        # имя без числа задач: путь к чекпоинту не должен зависеть от того, сколько буферов
        # фактически загрузилось (очередь ссылается на него заранее)
        args.subdir = "multi"
    else:
        # --per-pde-files действует и на один УрЧП: кривая «сколько данных буфера нужно»
        episodes = load_episodes(args.data_dir, args.subdir, max_files=args.per_pde_files)
        data = episodes_to_arrays(episodes, fix_next_state=args.fix_next_state,
                                  chain_fix=args.chain_fix, reward_form=args.reward_form,
                                  init_err=args.init_err, budget=args.episode_budget)
    if args.extra_buffers:
        if not args.chain_fix:
            ap.error("--extra-buffers требует --chain-fix")
        data = merge_online_buffers(
            data, [x.strip() for x in args.extra_buffers.split(",") if x.strip()],
            reward_form=args.reward_form, init_err=args.init_err, budget=args.episode_budget)
    if args.scalar_ctx and args.variant in ("cnx_bcq", "cnx_smdp"):
        ap.error("--scalar-ctx не поддержан для cnx_bcq и cnx_smdp")
    if args.scalar_ctx and not args.chain_fix:
        ap.error("--scalar-ctx требует --chain-fix")
    if args.ctx_err and not args.scalar_ctx:
        ap.error("--ctx-err требует --scalar-ctx")
    if args.pde_ctx and not args.scalar_ctx:
        ap.error("--pde-ctx требует --scalar-ctx")
    ctx_budget = args.ctx_budget or args.episode_budget or 31000
    args._teacher_mask = []
    teacher_meta = {}
    if args.distill_from:
        if not args.chain_fix:
            ap.error("--distill-from требует --chain-fix")
        if args.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn", "cnn_iql", "cnx_bcq", "cnx_smdp",
                            "cnn_vqc", "cnx_bbf"):
            ap.error("--distill-from: ученик должен быть обычной Q-сетью (convnext_dqn, stat_dqn, "
                     "cnn_dqn, cnx_dueling, cnx_factored, their_dqn)")
        if args.aug != "none" or args.fqe_select:
            ap.error("--distill-from несовместим с --aug и --fqe-select")
        import torch as _t
        _dev = _t.device("cuda" if _t.cuda.is_available() and not args.cpu else "cpu")
        data["QT"], teacher_meta = teacher_q(args.distill_from, data, _dev)
        args._teacher_mask = [int(i) for i in teacher_meta.get("mask", [])]
        print(f"дистилляция: Q учителя посчитаны на {len(data['QT'])} переходах "
              f"(учитель: режим {teacher_meta.get('state_mode', 'full')}, маска "
              f"{len(args._teacher_mask)} действий)", flush=True)
    if args.state_mode == "tasknorm":
        # нормировка по задаче: у каждого УрЧП свои статистики карт
        names = data.get("PDE_NAMES") or [subdir_to_pde(args.subdir)]
        pde_idx = data.get("PDE", np.zeros(len(data["A"]), dtype=np.int64))
        for k, name in enumerate(names):
            sel = pde_idx == k
            data["S"][sel] = apply_state_mode(data["S"][sel], "tasknorm", pde=name)
            data["S2"][sel] = apply_state_mode(data["S2"][sel], "tasknorm", pde=name)
        print(f"режим состояния: tasknorm по {len(names)} задачам", flush=True)
    elif args.state_mode != "full":
        data["S"] = apply_state_mode(data["S"], args.state_mode, seed=101)
        data["S2"] = apply_state_mode(data["S2"], args.state_mode, seed=202)
        print(f"режим состояния: {args.state_mode}", flush=True)
    if args.scalar_ctx:
        c1, c2 = chain_ctx(data, k_max=args.ctx_kmax, budget=ctx_budget, with_err=args.ctx_err)
        if args.pde_ctx:
            desc = np.array([pde_desc(n) for n in data["PDE_NAMES"]], dtype=np.float32)[data["PDE"]]
            c1, c2 = np.concatenate([c1, desc], 1), np.concatenate([c2, desc], 1)
        data["S"], data["S2"] = attach_ctx(data["S"], c1), attach_ctx(data["S2"], c2)
        print(f"скалярный контекст приклеен: {data['S'].shape}, канал ошибки "
              f"{'включён' if args.ctx_err else 'нулевой'}"
              + (", описатель задачи добавлен" if args.pde_ctx else ""), flush=True)
    run_meta = dict(ctx_kmax=int(args.ctx_kmax), ctx_budget=int(ctx_budget),
                    ctx_err=("err" if args.ctx_err else "none"),
                    state_mode=args.state_mode, gamma=float(GAMMA),
                    offline=True, scalar_ctx=bool(args.scalar_ctx),
                    episode_budget=int(args.episode_budget), init_err=args.init_err,
                    reward_form=args.reward_form, value_bound=bool(args.value_bound),
                    pde_ctx=bool(args.pde_ctx),
                    err_norm=(args.err_norm if multi else "none"),
                    train_pdes=(list(data["PDE_NAMES"]) if multi else []),
                    distill_from=os.path.basename(args.distill_from),
                    buffer_files=int(args.per_pde_files))
    if args._teacher_mask:
        run_meta["mask"] = list(args._teacher_mask)     # ученик наследует маску учителя
    train_mask, test_mask = split_by_episode(data)
    print(f"episodes={len(episodes)} transitions={len(data['A'])} "
          f"train={int(train_mask.sum())} test={int(test_mask.sum())}", flush=True)

    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time()
        net, stats, train_time = train_variant(args.variant, data, train_mask, args, seed)
        policies = ["mean", "cvar"] if args.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn") else ["mean"]
        for pol in policies:
            m, _ = evaluate(net, stats, data, test_mask, policy=pol)
            fq = fqe(net, stats, data, train_mask, test_mask, pol, args, seed)
            row = dict(variant=args.variant, policy=pol, seed=seed, gamma=GAMMA,
                       fixed_ns=bool(args.fix_next_state), dataset=args.subdir,
                       chain_fix=bool(args.chain_fix), reward_form=args.reward_form,
                       value_bound=bool(args.value_bound),
                       n_params=net.n_params(), train_time_s=round(train_time, 1),
                       epochs=args.epochs, smoke=args.smoke,
                       aug=args.aug, n_step=args.n_step, dqfd=args.dqfd,
                       l2=args.l2, pbrs=args.pbrs, smdp=bool(args.smdp),
                       model_tag=args.model_tag, state_mode=args.state_mode,
                       scalar_ctx=bool(args.scalar_ctx), ctx_err=bool(args.ctx_err),
                       **getattr(net, "extra", {}), **m, **fq)
            print(json.dumps(row), flush=True)
            if not args.smoke:
                tag = ("_fixns" if args.fix_next_state else "") + \
                      ("_cfix" if args.chain_fix else "") + \
                      (("_r" + args.reward_form) if args.reward_form != "logged" else "") + \
                      ("_vb" if args.value_bound else "") + \
                      ("" if args.subdir == SUBDIR else "_" + args.subdir.split("_")[0]) + \
                      (("_" + args.model_tag) if args.model_tag else "")
                upload_result(row, f"{args.variant}{tag}_{pol}_seed{seed}")
        if args.save_model and not args.smoke:
            import torch as _t
            ckpt = {"variant": args.variant, "seed": seed,
                    "state_dict": {k: v.cpu() for k, v in net.model.state_dict().items()},
                    "mean": stats["mean"], "std": stats["std"], "meta": run_meta}
            sfx = ("_fixns" if args.fix_next_state else "") + \
                  ("_cfix" if args.chain_fix else "") + \
                  (("_r" + args.reward_form) if args.reward_form != "logged" else "") + \
                  ("_vb" if args.value_bound else "") + \
                  ("" if args.subdir == SUBDIR else "_" + args.subdir.split("_")[0]) + \
                  (("_" + args.model_tag) if args.model_tag else "")
            fn = f"/tmp/agent_{args.variant}{sfx}_seed{seed}.pt"
            _t.save(ckpt, fn)
            tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
            if tok:
                # выгрузка НЕ критична: локальный чекпоинт уже сохранён и его
                # достаточно для онлайн-оценки в этом же кернеле. При параллельной
                # записи из десятков кернелов HF отдаёт 429/конфликт коммита —
                # раньше это роняло весь кернел.
                from huggingface_hub import upload_file
                for attempt in range(4):
                    try:
                        upload_file(path_or_fileobj=fn,
                                    path_in_repo=f"rl_arch/models/{args.variant}{sfx}_seed{seed}.pt",
                                    repo_id=OUT_REPO, repo_type="dataset", token=tok,
                                    commit_message=f"rl_arch model {args.variant} seed {seed}")
                        print(f"model uploaded: rl_arch/models/{args.variant}{sfx}_seed{seed}.pt", flush=True)
                        break
                    except Exception as e:
                        wait = 10 * (attempt + 1) + (seed % 7)
                        print(f"WARNING: model upload attempt {attempt+1}/4 failed: {e}", flush=True)
                        if attempt < 3:
                            time.sleep(wait)
                else:
                    print(f"model kept locally only: {fn}", flush=True)
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
