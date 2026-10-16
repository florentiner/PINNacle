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

Приёмы для агента на картах (всё выключено по умолчанию; сочетаются между собой и с --aug,
--n-step, --dqfd, --scalar-ctx):
    --encoder resnet|vit            энкодер карт у вариантов с ConvNeXt (определения общие с cv_probe.py)
    --encoder-init F                веса энкодера из файла (cv_probe.py --save-encoder или чекпоинт агента)
    --freeze-encoder                учить только головы
    --ens N --ens-min M             REDQ: N голов на общем энкодере, цель — минимум по M целевым головам
    --mb-aux W --mb-h H --mb-lam L  модельные цели по всем 27 действиям из суррогата (mb_targets.py)
    --mb-hs 1,2,3,5                 модельная цель — среднее по нескольким горизонтам (вместо --mb-h)
    --mb-trust state|gate|both      вес модельной цели строки по ошибке суррогата на её записанном исходе
                                    (суррогат без её кластера цепочек) и ворота по доле верных знаков
    --teacher-from F --teacher-w W  добавка к TD: Q учителя на скалярах (agent_combo.py) по всем 27 действиям
    --pani-w W --pani-sigma S       шум на нормированных картах, штраф за изменение вектора Q
    --distill-top P --distill-w W   отступ DQfD только на переходах лучших P% цепочек
    --mask-schedule                 первую треть эпох max в цели видит грубую сетку действий
    --target-chains N --data-seed K кривая «размер буфера»: ученик и суррогат — на тех же N цепочках цели,
                                    что учитель agent_combo.py с теми же N и K (mb_targets.target_subset)
    --target-buffer SPEC            данные задачи только из собранных заново файлов с картами
                                    (collect_active.py --save-buffer), исходный буфер не читается (target_buffer.py)
    --train-tol T --train-kmax K    правило остановки эпизода обучения из статьи (формула (11)): цепочка кончается
                                    терминальным переходом на ошибке <= T или на K-й стадии (stop_rule_cut);
                                    модельные цели и цели учителя за остановку не смотрят
Составляющие метода (волна 50; всё выключено по умолчанию):
    --wa ema|swa --wa-start F       развёртывается усреднение весов по итерациям обучения начиная с доли F эпох
    --wa-decay D                    (EMA с распадом D или равное среднее SWA); статистики BatchNorm пересчитываются
                                    на обучающих состояниях (Izmailov et al., UAI 2018; Morales-Brotons et al., TMLR 2024)
    --train-all                     учиться на всех переходах цели: отложенные 20% цепочек остаются только для
                                    диагностик (они теперь внутри обучения)
    --mc-lb W [--mc-lb-mode M]      нижняя граница Q по записанному исходу: дисконтированная сумма наград до конца
                                    цепочки (loss — штраф W * relu(G - Q(s,a))^2 / 2, как SIL / optimality
                                    tightening; target — цель TD max(цель, G), как MCAC)
    --teacher-from F1,F2,...        ансамбль учителей: цель учителя — среднее их Q по 27 действиям
Дообучение энкодера с малым числом параметров (волна 52; по умолчанию full — весь энкодер, как раньше):
    --ft-mode bn                    учатся только аффинные параметры BatchNorm энкодера (BitFit для норм)
    --ft-mode lora                  веса энкодера стоят, к каждой Conv2d и Linear добавлен низкоранговый адаптер
    --lora-rank R --lora-alpha A    (LoRA, Hu et al. ICLR 2022; свёртка kxk -> R каналов, затем 1x1, масштаб A/R)
    --ft-mode lastblock             учится только последний блок ResNet (body.3) и проекция head
Обучение на видах карт и дольше (волна 58; всё выключено по умолчанию):
    --view-aug SPEC [--view-aug-p P] виды карт состояний собранного буфера (map_views.py, render_views.py; SPEC как у
                                    cv_probe --views): у перехода, состояние которого (сид + начало цепочки + шаг)
                                    есть в файлах видов, карты s и независимо s' с вероятностью P заменяются
                                    случайным видом; delta — по тому же виду предыдущего шага (view_rows)
    --view-cons W                   согласованность: MSE Q(s) на двух разных наблюдениях одного состояния, вторая
                                    сторона — целевая сеть без градиента
    --lr-schedule cosine [--lr-min L] косинусный спад шага Adam по всем шагам обучения (арм: --epochs 1000)
Вид энкодера и число голов пишутся в meta чекпоинта: online_eval_env.load_agent строит ту же сеть.
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
import torch.nn as nn
import torch.nn.functional as F

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
# tele      — телеметрия обучения (tele_state): лоссы, статистики кривой внутри действия,
#           норма градиента, смещение весов; десять каналов, только онлайн
STATE_MODES = ("full", "blind", "level", "shape", "shuffle", "loss", "tasknorm", "tele")
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
    if mode in ("loss", "tele"):
        raise ValueError(f"режим {mode} строится из обучающего лосса (loss_state/tele_state) и для "
                         "массивов карт не определён: в офлайновых буферах обучающего лосса нет")
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


# Режим tele: телеметрия обучения вместо карт (сигналы состояния обучения AOS 2608.01997;
# статистический тест застоя по кривой лосса ExpTest 2411.16975 / SSS 2103.01205; проекция
# шага SWATS 1712.07628 — здесь в виде относительного смещения весов). Десять постоянных
# каналов на месте четырёх карт, контекст (--scalar-ctx) приклеивается после них.
TELE_CH = 10
TELE_NAMES = ("log_loss", "log_pde", "log_bnd", "drop", "slope100", "t_slope", "resid_sd",
              "improve_frac", "grad_norm", "rel_disp")
TELE_IMPROVE_TOL = 1e-6     # падение log10 лосса меньше этого (~20 ulp float32) — не улучшение


def state_channels(mode: str) -> int:
    """Число каналов наблюдения до контекста: 10 у tele, 4 (карты) у остальных режимов."""
    return TELE_CH if mode == "tele" else 4


def _symlog(x, floor):
    """Знаковый логарифм: sign(x)·log10(1 + |x|/floor), обрезка в [-3, 3]."""
    return float(np.clip(np.sign(x) * np.log10(1.0 + abs(float(x)) / floor), -3.0, 3.0))


def curve_stats(steps, losses, tail: float = 0.5):
    """Статистики кривой лосса внутри одного действия (ExpTest/SSS): МНК по log10 лосса на
    последней доле tail эпох действия. Возвращает сырые значения: slope100 — наклон в
    порядках на 100 эпох, t — его t-статистика, resid_sd — разброс остатков (порядки),
    improve — доля интервалов записи, где лосс упал больше чем на TELE_IMPROVE_TOL порядка,
    n — число точек. Меньше двух точек — нули."""
    st = np.asarray(steps, dtype=float).ravel()
    y = np.log10(np.maximum(np.asarray(losses, dtype=float).ravel(), 1e-30))
    ok = np.isfinite(st) & np.isfinite(y)
    st, y = st[ok], y[ok]
    out = dict(slope100=0.0, t=0.0, resid_sd=0.0, improve=0.0, n=int(len(st)))
    if len(st) < 2:
        return out
    cut = st[0] + (1.0 - tail) * (st[-1] - st[0])
    m = st >= cut
    if m.sum() < 2:
        m[-2:] = True
    x, yy = st[m], y[m]
    out["n"] = int(len(x))
    dy = np.diff(yy)
    out["improve"] = float(np.mean(dy < -TELE_IMPROVE_TOL)) if len(dy) else 0.0
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        return out
    b = float((xc * (yy - yy.mean())).sum() / sxx)
    out["slope100"] = b * 100.0
    if len(x) >= 3:
        res = yy - yy.mean() - b * xc
        sd = float(np.sqrt((res ** 2).sum() / (len(x) - 2)))
        out["resid_sd"] = sd
        se = sd / np.sqrt(sxx)
        out["t"] = float(np.clip(b / se, -1e4, 1e4)) if se > 1e-12 else (0.0 if b == 0 else float(np.sign(b) * 1e4))
    return out


def tele_state(loss_total, loss_oper, loss_bnd, prev_total=None, curve_steps=None, curve_loss=None,
               grad_norm=None, rel_disp=None, size: int = 26):
    """Состояние режима tele: TELE_CH постоянных каналов 26x26, каждый примерно в [-3, 3].
    Нормировка (фиксированная, без статистик буфера):
      0-3 log_loss, log_pde, log_bnd, drop — как loss_state: log10(L)/6 + 1/3 в [-1, 1];
          падение общего лосса за действие в порядках, обрезка [-1, 1];
      4 slope100 — наклон log10 лосса на 100 эпох на второй половине действия,
          symlog с полом 1e-3: sign·log10(1 + |s|/1e-3), [-3, 3] (застой 0, -0.1/100 эп -> -2);
      5 t_slope — t-статистика наклона, sign·log10(1 + |t|), [-3, 3] (|t| < 2 -> |код| < 0.48);
      6 resid_sd — разброс остатков МНК в порядках: (log10(sd + 1e-6) + 3) / 2, [-3, 3];
      7 improve_frac — доля интервалов записи с падением лосса: 2f - 1 в [-1, 1];
      8 grad_norm — норма градиента общего лосса в конце действия: log10(g) / 2, [-3, 3];
      9 rel_disp — ||θ_end − θ_start|| / ||θ_start|| за действие: (log10(d + 1e-12) + 4) / 1.5, [-3, 3].
    Нет кривой (меньше двух точек) — каналы 4-7 нулевые; нет градиента или смещения — нули."""
    head = loss_state(loss_total, loss_oper, loss_bnd, prev_total, size=size)
    cs = curve_stats(curve_steps if curve_steps is not None else [],
                     curve_loss if curve_loss is not None else [])
    if cs["n"] >= 2:
        c_curve = [_symlog(cs["slope100"], 1e-3), _symlog(cs["t"], 1.0),
                   float(np.clip((np.log10(cs["resid_sd"] + 1e-6) + 3.0) / 2.0, -3.0, 3.0)),
                   2.0 * cs["improve"] - 1.0]
    else:
        c_curve = [0.0, 0.0, 0.0, 0.0]
    g = (float(np.clip(np.log10(max(float(grad_norm), 1e-12)) / 2.0, -3.0, 3.0))
         if grad_norm is not None and np.isfinite(grad_norm) else 0.0)
    dsp = (float(np.clip((np.log10(max(float(rel_disp), 0.0) + 1e-12) + 4.0) / 1.5, -3.0, 3.0))
           if rel_disp is not None and np.isfinite(rel_disp) else 0.0)
    tail = np.stack([np.full((size, size), v, dtype=np.float32) for v in c_curve + [g, dsp]])
    return np.concatenate([head, tail], axis=0)


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


# --------------------------------------------------------------------------
# Правило остановки эпизода ОБУЧЕНИЯ (статья PINN-PELINE, раздел 2, формула (11), и раздел 3,
# «Stopping criterion»): эпизод кончается, когда ошибка относительно эталона дошла до допуска
# (e(θ_k) <= ε), после K_max стадий или при расходимости. Оценка по-прежнему идёт до конца бюджета
# (раздел 4 и приложение E статьи), правило касается только данных и целей обучения.
#
# Единицы ошибки — единицы тренера, те же, что у --init-err и награды: rmse+brmse (сумма RMSE
# решения и границы относительно эталона, формула (15) статьи при λ = 1), у смеси УрЧП с
# --err-norm init — доля ошибки необученной сети E0. Числа ε в статье нет.
# Расходимость правило не трогает: её обработка прежняя (загрузчик: нечисловая ошибка -> цепочка
# кончается терминальным переходом с ошибкой не меньше 1; суррогат: ошибка на верхнем пределе
# таблицы L_HI; онлайн: нечисловые метрики -> done=-1, переход не пишется).
# --------------------------------------------------------------------------

def stop_hit(err, tol) -> bool:
    """Ошибка дошла до допуска: tol > 0 и err <= tol (формула (11) статьи: e(θ_k) <= ε)."""
    return bool(tol is not None and float(tol) > 0 and np.isfinite(err) and float(err) <= float(tol))


def train_end(err, n_stages: int, tol=0.0, kmax: int = 0) -> str:
    """Причина конца эпизода обучения после стадии номер n_stages (с 1) с ошибкой err после неё:
    'tolerance' — ошибка не больше допуска, 'kmax' — исполнено kmax стадий, '' — эпизод идёт дальше.
    Оба конца терминальны (d = 1): ценности за остановкой нет."""
    if stop_hit(err, tol):
        return "tolerance"
    if kmax and int(n_stages) >= int(kmax):
        return "kmax"
    return ""


def stop_rule_cut(data, tol=0.0, kmax: int = 0, gamma=None, log=None):
    """Цепочки набора тренера обрезаются правилом остановки эпизода обучения: на первом переходе,
    после которого ошибка ERR <= tol (tol > 0), или на стадии номер kmax (STEP == kmax - 1, kmax > 0)
    цепочка кончается — переход терминален (D = 1), её дальнейшие переходы выбрасываются. RTG и FIRST
    пересчитываются. Нужны массивы загрузчика с chain_fix (EP, STEP, ERR, D, R).
    tol = 0 и kmax = 0 — правило выключено: возвращается тот же объект data без изменений.
    Возвращает (data, сводка: сколько цепочек обрезано по допуску и по K_max, сколько переходов
    выброшено, сколько концов стало терминальными)."""
    if not (tol and float(tol) > 0) and not (kmax and int(kmax) > 0):
        return data, {}
    for k in ("EP", "STEP", "ERR", "D", "R"):
        if k not in data:
            raise ValueError("правило остановки обучения требует набор, загруженный с --chain-fix")
    g = GAMMA if gamma is None else float(gamma)
    EP = np.asarray(data["EP"])
    STEP = np.asarray(data["STEP"], dtype=np.int64)
    ERR = np.asarray(data["ERR"], dtype=np.float64)
    n = len(EP)
    hit_t = (ERR <= float(tol)) & np.isfinite(ERR) if tol and float(tol) > 0 else np.zeros(n, dtype=bool)
    hit_k = STEP >= int(kmax) - 1 if kmax and int(kmax) > 0 else np.zeros(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    end = np.zeros(n, dtype=bool)
    st = dict(stop_tol=float(tol or 0.0), stop_kmax=int(kmax or 0), chains_tol=0, chains_kmax=0,
              dropped=0, made_terminal=0)
    starts = np.r_[0, np.where(EP[1:] != EP[:-1])[0] + 1] if n else np.zeros(0, dtype=np.int64)
    ends = np.r_[starts[1:], n] if n else np.zeros(0, dtype=np.int64)
    D0 = np.asarray(data["D"])
    for a0, b0 in zip(starts, ends):
        hit = np.where(hit_t[a0:b0] | hit_k[a0:b0])[0]
        if not len(hit):
            continue
        j = a0 + int(hit[0])
        st["chains_tol" if hit_t[j] else "chains_kmax"] += 1
        st["made_terminal"] += int(D0[j] == 0)
        keep[j + 1:b0] = False
        end[j] = True
    st["dropped"] = int((~keep).sum())
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n:
            out[k] = v[keep]
        else:
            out[k] = v
    D = np.asarray(out["D"], dtype=np.float32).copy()
    D[end[keep]] = 1.0
    out["D"] = D
    R, EP2 = np.asarray(out["R"]), np.asarray(out["EP"])
    RTG = np.zeros(len(R), dtype=np.float32)
    FIRST = np.zeros(len(R), dtype=np.float32)
    s2 = np.r_[0, np.where(EP2[1:] != EP2[:-1])[0] + 1] if len(R) else np.zeros(0, dtype=np.int64)
    e2 = np.r_[s2[1:], len(R)] if len(R) else np.zeros(0, dtype=np.int64)
    for a0, b0 in zip(s2, e2):
        run = 0.0
        for i in range(b0 - 1, a0 - 1, -1):
            run = float(R[i]) + g * run * (1.0 - float(D[i]))
            RTG[i] = run
        FIRST[a0] = 1.0
    if "RTG" in data:
        out["RTG"] = RTG
    if "FIRST" in data:
        out["FIRST"] = FIRST
    if isinstance(data.get("STATS"), dict):
        out["STATS"] = dict(data["STATS"], stop_rule=dict(st))
    if log is not None:
        log(f"правило остановки обучения (допуск {st['stop_tol']:g}, K_max {st['stop_kmax'] or 'нет'}): цепочек "
            f"обрезано по допуску {st['chains_tol']}, по K_max {st['chains_kmax']}; выброшено переходов "
            f"{st['dropped']} из {n}, новых терминальных концов {st['made_terminal']}")
    return out, st


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
    arch = ckpt_arch(ck)
    in_ch = arch["in_ch"]
    net = QNet(variant, device, **arch)
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


# нижний предел Q учителя (в декадах ошибки) при переводе в награду-разность: 10^(-Q) растёт
# экспоненциально, и без предела цель расходящегося действия (Q около -4) была бы -10^4 ошибок и
# забила бы остальные. -1 — «ошибка выросла вдесятеро», цель не ниже -9 текущих ошибок
TEACHER_QMIN = -1.0


def teacher_scalar_q(ck, data, device, names, err_scale):
    """Q учителя на скалярах (agent_combo.py: stat_dqn, state_mode=loss) на строках набора тренера.

    Состояние такого учителя — 15 чисел: 4 канала обучающих лоссов (их веса в первом слое нулевые
    — agent_combo.to_stat_dqn, учитель от них не зависит), шаг, эпохи, семейство и длина прошлого
    действия, log10((rmse+brmse)/E0) и описатель задачи. Обучающих лоссов в буферах нет, но они и
    не нужны; остальное восстанавливается из массивов тренера (STEP, SPENT, A, ERR) теми же
    функциями, какими учитель учился: состояние — mb_targets.chain_states (правило
    surrogate.build_table), наблюдение — agent_combo.agent_obs. Строки тренера при этом ни с какой
    внешней таблицей не выравниваются. Возвращает (Q (N, 27), valid, first, log10 текущей ошибки
    глазами учителя в исходных единицах): valid — у строки известны прошлая ошибка и расход эпох
    (цепочка идёт подряд с шага 0) и бюджет учителя не выбран."""
    import mb_targets as MB
    import agent_combo as AC
    S_ = MB.S
    q_of, rows = _teacher_scalar_net(ck, device, names)
    tab = dict(CTX=np.stack([S_.pde_ctx(rows[n]) for n in names]), NAMES=list(names))
    st, valid, first = MB.chain_states(data, tab, np.arange(len(names)), err_scale, budget=AC.BUDGET)
    obs = AC.agent_obs(tab["CTX"][st["pde"]], st["step"], st["spent"], st["pa"], st["l"])
    return q_of(obs), valid, first, st["l"]


def _teacher_scalar_net(ck, device, names):
    """Сеть учителя на скалярах (agent_combo.py: stat_dqn, state_mode=loss) после проверок чекпоинта:
    (функция наблюдения agent_combo.agent_obs (n, 15) -> Q (n, 27), строки pde_meta.json)."""
    import agent_combo as AC
    meta = ck.get("meta") or {}
    sd = ck["state_dict"]
    arch = ckpt_arch(ck)
    in_ch = arch["in_ch"]
    if ck["variant"] != "stat_dqn" or in_ch != AC.N_IN or ck.get("hl_gauss"):
        raise ValueError(f"учитель режима loss: ждём stat_dqn с {AC.N_IN} каналами от agent_combo.py, "
                         f"в чекпоинте {ck['variant']} с {in_ch}")
    w0 = sd["0.mlp.0.weight"]
    cols = [b * in_ch + c for b in range(w0.shape[1] // in_ch) for c in range(AC.N_LOSS)]
    if float(w0[:, cols].abs().max()) != 0.0:
        raise ValueError("учитель режима loss смотрит на обучающие лоссы (веса первого слоя на каналах "
                         "0-3 не нулевые), а в офлайновых буферах их нет: Q на строках буфера не посчитать")
    want = dict(ctx_err="err", err_norm="init", pde_ctx=True, ctx_kmax=AC.KMAX, ctx_budget=AC.BUDGET)
    bad = {k: meta.get(k) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"учитель режима loss: состояние собрано не как у agent_combo.py (ждём {want}, "
                         f"в meta {bad})")
    rows = pde_meta()
    miss = [n for n in names if n not in rows]
    if miss:
        raise ValueError(f"учитель: задач {miss} нет в track3/pde_meta.json")
    net = QNet("stat_dqn", device, **arch)
    net.model.load_state_dict(sd)
    net.model.eval()
    mean = np.asarray(ck["mean"], np.float32).reshape(1, in_ch)
    std = np.asarray(ck["std"], np.float32).reshape(1, in_ch)

    def q_of(obs):
        out = np.zeros((len(obs), N_ACTIONS), dtype=np.float32)
        with torch.no_grad():
            for k in range(0, len(obs), 512):
                # постоянные плоскости, как их подаёт оценка (add_scalar_ctx + описатель задачи)
                x = torch.as_tensor((obs[k:k + 512] - mean) / std, device=device).float()
                out[k:k + 512] = net.q_scalar(x[:, :, None, None].expand(-1, -1, 26, 26)).float().cpu().numpy()
        return out
    return q_of, rows


def teacher_rule_targets(ck, data, names, reward_form, err_scale, init_err, gamma, device, model, lam=0.5,
                         budget=0, tol=0.0, kmax=0, smdp_scale=0.0):
    """Цели учителя под правилом остановки эпизода обучения (--train-tol, --train-kmax).

    Q учителя agent_combo.py — убыль ошибки до конца ЕГО эпизода (до бюджета, без допуска и без
    K_max), обрезать её правилом нельзя: неизвестно, на какой стадии продолжение учителя дошло бы до
    допуска и сколько стадий оно заняло бы. Поэтому под правилом цель строки — ценность продолжения
    ПОЛИТИКИ учителя (жадное действие по его Q, его маска) в том же суррогате, что у модельных целей
    (mb_targets.policy_rollout): первое действие a, дальше действия учителя, пессимистичная награда
    r~ = среднее - lam * разброс по членам, дисконт ученика, конец — бюджет, расходимость, допуск
    (средняя ошибка суррогата <= tol), K_max стадий цепочки. Единицы награды — тренера (delta | dlog),
    первый шаг цепочки — от init_err, как у mb_targets.precompute. Возвращает (Q (N, 27), valid):
    valid — строка в цепочке с началом, расход меньше бюджета учителя и бюджета ученика."""
    import mb_targets as MB
    import agent_combo as AC
    S_ = MB.S
    tab, ens = model
    n = len(data["A"])
    q_of, rows = _teacher_scalar_net(ck, device, names)
    pidx = np.array([tab["NAMES"].index(nm) for nm in names], dtype=np.int64)
    err_scale = np.asarray(err_scale, dtype=np.float64)
    init_err = np.asarray(init_err, dtype=np.float64)
    st, valid, first = MB.chain_states(data, tab, pidx, err_scale, budget)
    valid &= np.asarray(data["SPENT"]) < AC.BUDGET
    pl = np.asarray(data["PDE"], dtype=np.int64) if "PDE" in data else np.zeros(n, dtype=np.int64)
    l_rew = st["l"].copy()
    l_rew[first] = S_._log(init_err[pl] * err_scale[pl])[first]
    # описатель задачи — как у teacher_scalar_q (pde_meta.json), по номеру задачи в таблице суррогата
    ctx_t = np.zeros((len(tab["NAMES"]), np.asarray(tab["CTX"]).shape[1]), dtype=np.float64)
    for k, nm in enumerate(names):
        ctx_t[pidx[k]] = S_.pde_ctx(rows[nm])
    allowed = np.ones(N_ACTIONS, dtype=bool)
    allowed[[int(i) for i in (ck.get("meta") or {}).get("mask", [])]] = False

    def act(s):
        q = q_of(AC.agent_obs(ctx_t[s["pde"]], s["step"], s["spent"], s["pa"], s["l"])).astype(np.float64)
        q[:, ~allowed] = -np.inf
        return q.argmax(1)

    Q = np.zeros((n, N_ACTIONS), dtype=np.float64)
    rows_v = np.where(valid)[0]
    if len(rows_v):
        sub = {k: v[rows_v] for k, v in st.items()}
        Q[rows_v] = MB.policy_rollout(ens, tab, sub, l_rew[rows_v], np.zeros(len(rows_v), dtype=bool),
                                      err_scale[pl][rows_v], reward_form, gamma, float(lam), act,
                                      budget=int(budget), smdp_scale=float(smdp_scale), kmax=int(kmax or 0),
                                      tl=MB.tol_log(tab, pidx, err_scale, tol))
    return Q, valid


def teacher_targets(path, data, names, reward_form, err_scale, init_err, gamma, device, budget=0, log=None,
                    tol=0.0, kmax=0, model=None, lam=0.5, smdp_scale=0.0):
    """--teacher-w: Q учителя в единицах награды тренера на строках его набора.

    Учитель agent_combo.py учится на награде «убыль log10 ошибки за действие» с дисконтом 1:
    Q(s, a) — на сколько декад упадёт ошибка от текущей до конца эпизода (со штрафом MOPO), то есть
    учитель предсказывает итоговую ошибку E_кон = 10^(l - Q), l — log10 текущей ошибки. Цель ученика
    — награда ЕГО формы за переход от текущей ошибки (у первого шага — от init_err тренера) к E_кон:
        dlog    log10 E - log10 E_кон   (у не первых шагов это само Q учителя);
        delta   (E - E_кон) / масштаб ошибки тренера, Q учителя снизу обрезано TEACHER_QMIN.
    Допущение: суммы наград складываются без дисконта (как у учителя). При gamma ученика < 1 и
    неотрицательных наградах цель завышена не больше чем в 1/gamma^(число оставшихся действий) раз
    (0.99 и 12 действий — до 13%). Формы logged, level и model не переводятся (отказ).
    names — задачи набора (в порядке data['PDE']), err_scale и init_err — по задачам в единицах
    тренера, как у mb_targets.precompute. Возвращает (Q float32 (N, 27), у каких строк цель есть
    bool (N,), сводка). Цель есть не у всех: строка вне цепочки с началом, расход не меньше бюджета
    учителя, строка до каркаса учителя (--guide: он решает только после каркаса), чужая задача.
    tol > 0 или kmax > 0 — правило остановки эпизода обучения (--train-tol, --train-kmax): цель — ценность
    продолжения политики учителя в суррогате model=(таблица, ансамбль) до остановки
    (teacher_rule_targets; lam — пессимизм, как у модельных целей). По умолчанию выключено."""
    log = log or (lambda *a: print(*a, flush=True))
    t0 = time.time()
    if reward_form not in ("delta", "dlog"):
        raise ValueError(f"--teacher-w: Q учителя — убыль log10 ошибки до конца эпизода; в награду "
                         f"{reward_form!r} она не переводится (нужна delta или dlog)")
    for k in ("EP", "STEP", "SPENT", "ERR"):
        if k not in data:
            raise ValueError("--teacher-w требует набор, загруженный с --chain-fix")
    if init_err is None:
        raise ValueError("--teacher-w требует --init-err: без него награда первого шага у тренера ноль, "
                         "а учитель считает первый шаг от ошибки необученной сети")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OUT_REPO, path, repo_type="dataset")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck.get("meta") or {}
    if meta.get("state_mode") != "loss":
        raise ValueError(f"--teacher-from: поддержан учитель на скалярах от agent_combo.py (state_mode=loss), "
                         f"в чекпоинте {meta.get('state_mode', 'full')!r}; учителя на картах берёт --distill-from")
    n = len(data["A"])
    err_scale = np.asarray(err_scale, dtype=np.float64)
    init_err = np.asarray(init_err, dtype=np.float64)
    assert err_scale.shape == (len(names),) and init_err.shape == (len(names),), (err_scale.shape, init_err.shape)
    rule = bool((tol and float(tol) > 0) or (kmax and int(kmax) > 0))
    pl = np.asarray(data["PDE"], dtype=np.int64) if "PDE" in data else np.zeros(n, dtype=np.int64)
    if rule:
        if model is None:
            raise ValueError("--teacher-w с правилом остановки обучения: нужен суррогат (model) — цель учителя "
                             "считается продолжением его политики в суррогате до остановки")
        out, valid = teacher_rule_targets(ck, data, names, reward_form, err_scale, init_err, gamma, device, model,
                                          lam=lam, budget=budget, tol=tol, kmax=kmax, smdp_scale=smdp_scale)
    else:
        q, valid, first, l = teacher_scalar_q(ck, data, device, names, err_scale)
        assert q.shape == (n, N_ACTIONS) and valid.shape == (n,) and l.shape == (n,)
    guide_ep = int(meta.get("guide_epochs", 0) or 0)
    valid = valid & (np.asarray(data["SPENT"]) >= guide_ep)
    known = meta.get("train_pdes")
    if known:
        valid &= np.isin(pl, [k for k, nm in enumerate(names) if nm in known])
    if not rule:
        # текущая ошибка в единицах тренера: у первого шага — init_err, дальше — ошибка после прошлого действия
        base = np.where(first, init_err[pl], np.r_[1.0, np.asarray(data["ERR"], dtype=np.float64)[:-1]])
        q64 = q.astype(np.float64)
        if reward_form == "dlog":
            out = (np.log10(np.maximum(base, 1e-8)) - (l - np.log10(err_scale[pl])))[:, None] + q64
        else:
            out = base[:, None] - 10.0 ** (l[:, None] - np.maximum(q64, TEACHER_QMIN)) / err_scale[pl][:, None]
    valid &= np.isfinite(out).all(1)
    out = np.where(valid[:, None], out, 0.0).astype(np.float32)
    g_t = float(meta.get("gamma", 1.0))
    info = dict(rows=int(valid.sum()), skipped=int(n - valid.sum()),
                units=(f"rollout->{reward_form}" if rule else f"dlog->{reward_form}"),
                gamma=g_t, guide_epochs=guide_ep, mask_n=len(meta.get("mask", [])),
                q_mean=float(out[valid].mean()) if valid.any() else 0.0,
                q_best_mean=float(out[valid].max(1).mean()) if valid.any() else 0.0,
                sec=round(time.time() - t0, 1))
    if rule:
        info.update(stop_tol=float(tol or 0.0), stop_kmax=int(kmax or 0))
    log(f"учитель {os.path.basename(path)}: цель у {info['rows']} строк из {n} (каркас учителя {guide_ep} "
        f"эпох, маска {info['mask_n']} действий), единицы {info['units']}: средний Q {info['q_mean']:.4f}, "
        f"лучшего действия {info['q_best_mean']:.4f}, {info['sec']} с")
    if int(budget) != int(meta.get("ctx_budget", 0) or 0) and not rule:
        log(f"ВНИМАНИЕ: учитель обучен на эпизодах в {meta.get('ctx_budget')} эпох, цепочки ученика обрезаны по "
            f"{budget or 'без бюджета'}: Q учителя считает остаток своего бюджета")
    if abs(g_t - float(gamma)) > 1e-9 and not rule:
        log(f"учитель: дисконт учителя {g_t:g}, ученика {float(gamma):g} — цель переведена как сумма наград "
            "без дисконта (см. teacher_targets)")
    if not valid.any():
        raise ValueError("--teacher-w: ни у одной строки набора нет цели учителя (каркас длиннее цепочек, "
                         "чужая задача или цепочки без начала)")
    return out, valid, dict(info, mask=[int(i) for i in meta.get("mask", [])])


def teacher_ensemble_targets(paths, data, names, reward_form, err_scale, init_err, gamma, device, budget=0,
                             log=None):
    """Ансамбль учителей (--teacher-from F1,F2,...): цель каждого — teacher_targets в единицах награды
    ученика, итог — их среднее по 27 действиям (дистилляция ансамбля: Hinton et al. 2015; Allen-Zhu, Li,
    ICLR 2023). Цель есть у строки, только если она есть у всех учителей. Маска учителя для диагностики
    совпадения — объединение масок. В сводке — согласие жадных действий каждой пары учителей на строках
    с целью (насколько учителя расходятся)."""
    log = log or (lambda *a: print(*a, flush=True))
    qs, vs, infos = [], [], []
    for p in paths:
        q, v, inf = teacher_targets(p, data, names=names, reward_form=reward_form, err_scale=err_scale,
                                    init_err=init_err, gamma=gamma, device=device, budget=budget, log=log)
        qs.append(q.astype(np.float64)); vs.append(v); infos.append(inf)
    valid = np.logical_and.reduce(vs)
    if not valid.any():
        raise ValueError("ансамбль учителей: ни у одной строки нет цели всех учителей")
    out = np.where(valid[:, None], np.mean(qs, axis=0), 0.0).astype(np.float32)
    greedy = [q[valid].argmax(1) for q in qs]
    agree = [float((greedy[i] == greedy[j]).mean()) for i in range(len(qs)) for j in range(i + 1, len(qs))]
    mask = sorted(set(int(i) for inf in infos for i in inf.get("mask", [])))
    info = dict(infos[0], rows=int(valid.sum()), skipped=int(len(valid) - valid.sum()), n_teachers=len(paths),
                teachers=",".join(os.path.basename(p) for p in paths),
                pair_agree_mean=round(float(np.mean(agree)), 4), pair_agree_min=round(float(np.min(agree)), 4),
                q_mean=float(out[valid].mean()), q_best_mean=float(out[valid].max(1).mean()), mask=mask)
    log(f"ансамбль из {len(paths)} учителей: цель у {info['rows']} строк, согласие жадных действий пар "
        f"{info['pair_agree_mean']:.3f} (мин. {info['pair_agree_min']:.3f}), средний Q {info['q_mean']:.4f}")
    return out, valid, info


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

# Энкодеры карт из зонда CV (cv_probe.py импортирует их отсюда: определение одно). Имена
# параметров менять нельзя — по ним читаются предобученные веса (--encoder-init) и чекпоинты.
class _Res(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class SmallResNet(nn.Module):
    """Малый ResNet под 26x26: первые блоки без уменьшения, затем 13x13 и 7x7, среднее по карте."""
    out_dim = 256

    def __init__(self, in_ch):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.body = nn.Sequential(_Res(32, 32, 1), _Res(32, 64, 2), _Res(64, 64, 1), _Res(64, 128, 2))
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU())

    def forward(self, x):
        return self.head(self.body(self.stem(x)).mean((2, 3)))


class SmallResNetPos(SmallResNet):
    """Тот же ResNet, но положение на карте не теряется: к входу добавлены два канала координат
    (CoordConv, 1807.03247), а признаки 7x7 не усредняются по карте, а разворачиваются в вектор.
    Центр карты — текущая точка обучения, и где впадина относительно неё, усреднение не видит."""

    def __init__(self, in_ch, size=26):
        super().__init__(in_ch + 2)
        g = torch.linspace(-1.0, 1.0, size)
        self.register_buffer("coord", torch.stack(torch.meshgrid(g, g, indexing="ij"))[None], persistent=False)
        side = (size + 3) // 4                      # два уменьшения вдвое с округлением вверх: 26 -> 13 -> 7
        self.head = nn.Sequential(nn.Linear(128 * side * side, 256), nn.ReLU())

    def forward(self, x):
        # координаты идут первыми: у входа с контекстом лишние каналы остаются в хвосте, и веса
        # первой свёртки из --encoder-init ложатся на те же каналы, что при предобучении
        x = torch.cat([self.coord.to(x.dtype).expand(x.shape[0], -1, -1, -1), x], dim=1)
        return self.head(self.body(self.stem(x)).flatten(1))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    """ViT: патч 2 (13x13 = 169 токенов у карты 26x26), 4 слоя, среднее по токенам."""

    def __init__(self, in_ch, size=26, patch=2, dim=96, depth=4, heads=4):
        super().__init__()
        self.out_dim = dim
        self.patch = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (size // patch) ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[_Block(dim, heads, 2 * dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.norm(self.blocks(t).mean(1))


MAP_ENCODERS = ("convnext", "convnextw", "convnexthr", "resnet", "resnetpos", "vit")      # что принимает --encoder у вариантов cnx_*


# --ft-mode: какие параметры энкодера карт учатся при дообучении (волна 52). full — все (прежнее);
# bn — только аффинные параметры BatchNorm; lora — низкоранговые адаптеры к каждой Conv2d и Linear
# (веса энкодера стоят) плюс аффинные параметры BatchNorm; lastblock — последний остаточный блок и
# проекция (FT_LASTBLOCK). Статистики BatchNorm в bn и lora копятся в режиме обучения, как при full;
# в lastblock замороженная часть энкодера — неизменная функция (режим eval, как у --freeze-encoder).
FT_MODES = ("full", "bn", "lora", "lastblock")
# последний блок у энкодеров, где он определён однозначно: у SmallResNet стадия 7x7 из одного блока
# _Res(64, 128, stride 2) — body.3 — и проекция 128 -> 256 (у resnetpos — 128*7*7 -> 256)
FT_LASTBLOCK = {"resnet": ("body.3.", "head."), "resnetpos": ("body.3.", "head.")}


class LoRAConv2d(nn.Conv2d):
    """Свёртка с низкоранговой добавкой (LoRA для свёрток): y = conv(x) + (alpha/r) * up(down(x)),
    down — свёртка с тем же ядром, шагом и отступом в r каналов (без групп: у глубинной свёртки
    добавка плотная), up — свёртка 1x1 из r каналов, нулевая при создании, так что исходная функция
    не меняется. Имена весов самой свёртки прежние (weight, bias): в state_dict добавляются только
    ключи lora_down.weight и lora_up.weight. Создаётся подменой класса готового слоя (lora_wrap)."""

    def _lora_setup(self, r, alpha):
        kw = dict(device=self.weight.device, dtype=self.weight.dtype)
        self.lora_down = nn.Conv2d(self.in_channels, r, self.kernel_size, stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, bias=False, padding_mode=self.padding_mode, **kw)
        self.lora_up = nn.Conv2d(r, self.out_channels, 1, bias=False, **kw)
        nn.init.zeros_(self.lora_up.weight)
        self.lora_scale = float(alpha) / float(r)

    def forward(self, x):
        return super().forward(x) + self.lora_scale * self.lora_up(self.lora_down(x))


class LoRALinear(nn.Linear):
    """Линейный слой с низкоранговой добавкой: y = W x + b + (alpha/r) * B A x, A (r x in) — как у
    nn.Linear по умолчанию, B (out x r) нулевая при создании (Hu et al., ICLR 2022)."""

    def _lora_setup(self, r, alpha):
        kw = dict(device=self.weight.device, dtype=self.weight.dtype)
        self.lora_down = nn.Linear(self.in_features, r, bias=False, **kw)
        self.lora_up = nn.Linear(r, self.out_features, bias=False, **kw)
        nn.init.zeros_(self.lora_up.weight)
        self.lora_scale = float(alpha) / float(r)

    def forward(self, x):
        return super().forward(x) + self.lora_scale * self.lora_up(self.lora_down(x))


def lora_wrap(enc, r, alpha):
    """Адаптеры LoRA ко всем Conv2d и Linear энкодера (подмена класса на месте: модуль и его
    параметры те же, имена прежних ключей не меняются). Возвращает число обёрнутых слоёв.
    Энкодер с вниманием (nn.MultiheadAttention: проекции вызываются не как слои) не поддержан."""
    if any(isinstance(m, nn.MultiheadAttention) for m in enc.modules()):
        raise ValueError("--ft-mode lora: у энкодера с вниманием (vit) проекции внимания не вызываются как слои, "
                         "адаптеры к ним не встают")
    mods = [m for m in enc.modules() if type(m) in (nn.Conv2d, nn.Linear)]
    for m in mods:
        m.__class__ = LoRAConv2d if type(m) is nn.Conv2d else LoRALinear
        m._lora_setup(int(r), float(alpha))
    return len(mods)


def is_lora_key(name):
    return ".lora_down." in f".{name}" or ".lora_up." in f".{name}"


def ft_trainable_names(enc, mode, enc_kind):
    """Имена параметров энкодера, которые учатся в режиме --ft-mode (без свежих каналов keep)."""
    if mode == "full":
        return {n for n, _ in enc.named_parameters()}
    bn = {f"{mn}.{pn}" if mn else pn for mn, m in enc.named_modules()
          if isinstance(m, nn.modules.batchnorm._BatchNorm) for pn, _ in m.named_parameters(recurse=False)}
    if mode == "bn":
        return bn
    if mode == "lora":
        return bn | {n for n, _ in enc.named_parameters() if is_lora_key(n)}
    if mode == "lastblock":
        pre = FT_LASTBLOCK[enc_kind]
        return {n for n, _ in enc.named_parameters() if n.startswith(pre)}
    raise ValueError(f"--ft-mode {mode!r}: выбор {FT_MODES}")


def ft_check(mode, enc_kind, enc):
    """Отказ, если режим у этого энкодера не определён (ValueError с объяснением)."""
    if mode not in FT_MODES:
        raise ValueError(f"--ft-mode {mode!r}: выбор {FT_MODES}")
    if mode == "bn" and not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in enc.modules()):
        raise ValueError(f"--ft-mode bn: у энкодера {enc_kind} нет BatchNorm (есть у {', '.join(FT_LASTBLOCK)}); "
                         "без них режим — просто замороженный энкодер (--freeze-encoder)")
    if mode == "lastblock" and enc_kind not in FT_LASTBLOCK:
        raise ValueError(f"--ft-mode lastblock определён только у {', '.join(FT_LASTBLOCK)} (последний остаточный "
                         f"блок body.3 и проекция head), не у {enc_kind}")


def make_encoder(kind: str, in_ch: int = 4, size: int = 26):
    import torch
    import torch.nn as nn

    if kind == "resnet":
        return SmallResNet(in_ch)
    if kind == "resnetpos":
        return SmallResNetPos(in_ch, size)
    if kind == "vit":
        return TinyViT(in_ch, size)

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

    if kind in ("convnext", "convnextw", "convnexthr"):
        # convnext — прежняя сеть (48/96 каналов, по два блока, вход сразу сжимается до 13x13);
        # convnextw — шире и глубже (64/128, по три блока); convnexthr — первая стадия на полной
        # карте 26x26 (вход не сжимается), вторая на 13x13: проверка, мешает ли раннее сжатие
        c1, c2, nb = (64, 128, 3) if kind == "convnextw" else (48, 96, 2)
        full_res = kind == "convnexthr"

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
                self.stem = (nn.Conv2d(in_ch, c1, 3, padding=1) if full_res        # 26
                             else nn.Conv2d(in_ch, c1, 2, stride=2))               # 13
                self.s1 = nn.Sequential(*[Block(c1) for _ in range(nb)])
                self.down = nn.Conv2d(c1, c2, 2, stride=2)         # 6 (13 у convnexthr)
                self.s2 = nn.Sequential(*[Block(c2) for _ in range(nb)])
                self.head = nn.Sequential(nn.Linear(c2, 256), nn.ReLU())
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


QUANTILE_VARIANTS = ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
# варианты с энкодером ConvNeXt: только у них выбирается энкодер карт (--encoder)
CNX_VARIANTS = ("convnext_dqn", "cnx_cql", "cnx_cql_qr", "cnx_dueling",
                "cnx_bcq", "cnx_bbf", "cnx_qrdqn", "cnx_factored")
NO_ENS_VARIANTS = ("cnn_iql", "cnn_vqc", "cnx_smdp")     # своя V-голова, квантовая голова, свой ансамбль
# варианты с простой линейной головой на 27 выходов: только им --hlg меняет голову на бины
HLG_VARIANTS = ("cnn_dqn", "convnext_dqn", "cnn_cql", "cnx_cql")
# варианты, которым --two-nets заменяет одну сеть двумя, как у агента статьи (обычный TD-лосс по
# скалярному Q на 27 действий; у остальных своя голова или свой лосс)
TWO_NETS_VARIANTS = ("convnext_dqn", "cnx_cql")
TWO_NETS_CENTER = ("max", "mean")
TWO_NETS_HEAD = ("linear", "dueling")
# как сеть параметров раскладывает преимущества шага и длительности (--two-nets-mode):
# additive — сумма двух независимых голов (прежнее), joint — одна голова на все пары (шаг,
# длительность) оптимизатора, auto — авторегрессия: шаг по [признаки, оптимизатор], длительность
# по [признаки, оптимизатор, шаг], lowrank — сумма additive плюс взаимодействие шага и длительности
# ранга TWO_NETS_RANK (у каждого значения шага и длительности вектор, взаимодействие — их скалярное
# произведение): выходов у сети B растёт как сумма числа значений параметров, а не как произведение
TWO_NETS_MODE = ("additive", "joint", "auto", "lowrank")
TWO_NETS_RANK = 1               # ранг взаимодействия шага и длительности у lowrank
# --two-nets-shared: идёт ли градиент сети параметров в общий энкодер (--two-nets-shared-grad)
TWO_NETS_SHARED_GRAD = ("both", "stop")
TWO_NETS_AUTO_EMB = 16          # размер вложений оптимизатора и шага у auto
TWO_NETS_AUTO_HIDDEN = 64       # скрытый слой голов auto при головах linear (dueling — ширина признаков)
# составляющие метода (волна 50): усреднение весов (--wa) и нижняя граница Q по исходу (--mc-lb)
WA_MODES = ("none", "ema", "swa")
MC_LB_MODES = ("loss", "target")
# --mc-lb-mode target меняет цель TD только у обычной головы DQN (не ансамбль, не бины, не квантили)
MC_LB_TARGET_VARIANTS = ("cnn_dqn", "convnext_dqn", "cnn_cql", "cnx_cql", "cnx_dueling", "cnx_factored",
                         "their_dqn", "their_cql", "cnx_bcq")
# дольше учиться (волна 58): план шага Adam по шагам всего обучения (--lr-schedule, --lr-min)
LR_SCHEDULES = ("const", "cosine")
VIEW_AUG_P = 0.5                # --view-aug-p по умолчанию


def mc_returns(data, gamma):
    """--mc-lb: G_i = r_i + gamma * r_{i+1} + ... до конца цепочки строки i (по data['EP'], строки цепочки
    идут подряд в порядке шагов, как у build_targets). Оборванная цепочка (последний переход не терминален)
    продолжается нулём: при неотрицательной ценности продолжения (--value-bound: можно остановиться) это
    нижняя граница Q*(s_i, a_i) записанного действия в детерминированной среде. float32 (N,)."""
    R = np.asarray(data["R"], dtype=np.float64)
    EP = np.asarray(data["EP"])
    G = np.zeros(len(R), dtype=np.float64)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        run = 0.0
        for i in idx[::-1]:
            run = R[i] + float(gamma) * run
            G[i] = run
    return G.astype(np.float32)


def wa_update(avg, model, n, mode, decay):
    """Одно обновление среднего весов --wa по state_dict модели (параметры и буферы BatchNorm).
    avg=None — первый вход: копия. Целые буферы (num_batches_tracked) копируются как есть.
    swa — равное среднее n входов, ema — avg <- decay * avg + (1 - decay) * текущее."""
    import torch
    with torch.no_grad():
        cur = model.state_dict()
        if avg is None:
            return {k: v.detach().clone() for k, v in cur.items()}
        for k, v in cur.items():
            if not v.is_floating_point():
                avg[k].copy_(v)
            elif mode == "swa":
                avg[k].add_((v - avg[k]) / float(n))
            else:
                avg[k].mul_(float(decay)).add_(v, alpha=1.0 - float(decay))
    return avg


def recompute_bn(model, forward, X, bs=256, skip=()):
    """Статистики BatchNorm для усреднённых весов (как torch.optim.swa_utils.update_bn): счётчики сброшены,
    накопительное среднее (momentum=None) по проходам forward на X в режиме train. skip — слои, которые не
    трогаются (на проходах стоят в режиме eval). Возвращает число пересчитанных слоёв BN."""
    import torch
    skip_ids = {id(m) for m in skip}
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
           and id(m) not in skip_ids]
    if not bns:
        return 0
    moms = [m.momentum for m in bns]
    was = model.training
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train(True)
    for m in skip:
        m.eval()
    with torch.no_grad():
        for k in range(0, len(X), bs):
            forward(X[k:k + bs])
    for m, mo in zip(bns, moms):
        m.momentum = mo
    model.train(was)
    return len(bns)


def lr_at(step, total, lr0, lr_min=0.0, schedule="const"):
    """Шаг Adam на шаге обучения step (с нуля) из total (--lr-schedule). const — lr0 всегда (прежнее поведение);
    cosine — lr_min + (lr0 - lr_min) * (1 + cos(pi * step / total)) / 2 (Loshchilov, Hutter, ICLR 2017, без
    перезапусков): первый шаг — lr0, середина — (lr0 + lr_min) / 2, к концу — почти lr_min (как
    CosineAnnealingLR с T_max = total и шагом планировщика после каждого шага оптимизатора)."""
    if schedule == "const":
        return float(lr0)
    if schedule != "cosine":
        raise ValueError(f"--lr-schedule {schedule!r}: допустимо {LR_SCHEDULES}")
    t = min(max(int(step), 0), max(int(total), 1))
    return float(lr_min) + (float(lr0) - float(lr_min)) * 0.5 * (1.0 + math.cos(math.pi * t / max(int(total), 1)))


# --------------------------------------------------------------------------
# Виды карт для обучения Q (--view-aug, волна 58)
# --------------------------------------------------------------------------

def view_row_keys(episodes, data):
    """Ключи состояний строк набора для сопоставления с файлами видов (map_views.state_key: сид + исполненное
    начало цепочки + шаг) — так же, как их строит render_views.py по файлу собранного буфера: сид — meta первого
    перехода файла, цепочка — номинальные действия всех переходов файла (таблица 27 действий), шаг состояния s
    строки — её номер в файле (STEP загрузчика с --chain-fix), шаг s' — на один больше.
    Строки сопоставляются с файлами по номеру цепочки EP: episodes_to_arrays_chains нумерует цепочки подряд по
    непустым файлам (граница — done != 0). Сопоставляется только первая цепочка файла (render_views повторяет
    файл как одну цепочку с шага 0; файлы сбора — одна цепочка на файл). Возвращает dict: KS, KS2 — ключ или
    None (шаг 0 — нулевая карта, у неё видов нет; терминальная строка — её s' в цель не входит), KP — ключ
    предыдущего состояния s (шаг s-1 >= 1; для канала delta), POS — номер перехода в файле (-1 у
    несопоставленных)."""
    import map_views as MV
    import target_buffer as TB
    first = {}                      # номер цепочки -> (сид, цепочка файла)
    cid = -1
    for ep in episodes:
        if not ep:
            continue
        meta = ep[0].get("meta") or {}
        seed = meta.get("seed")
        acts = [int(t["action"][0]) * 9 + int(t["action"][1]["lr"]) * 3 + int(t["action"][1]["epochs"]) for t in ep]
        chain = [list(TB.S.ACTIONS[a]) for a in acts]
        n_ch = sum(int(t.get("done", 0)) != 0 for t in ep) + int(int(ep[-1].get("done", 0)) == 0)
        for c in range(n_ch):
            cid += 1
            if c == 0 and seed is not None:
                first[cid] = (int(seed), chain)
    EP, STEP, D = (np.asarray(data[k]) for k in ("EP", "STEP", "D"))
    n = len(EP)
    KS, KS2, KP, POS = [None] * n, [None] * n, [None] * n, np.full(n, -1, dtype=np.int64)
    for i in range(n):
        src = first.get(int(EP[i]))
        if src is None:
            continue
        seed, chain = src
        p = int(STEP[i])
        POS[i] = p
        if p >= 1:
            KS[i] = MV.state_key(seed, chain, p)
        if p >= 2:
            KP[i] = MV.state_key(seed, chain, p - 1)
        if float(D[i]) == 0.0 and p + 1 <= len(chain):
            KS2[i] = MV.state_key(seed, chain, p + 1)
    return dict(KS=KS, KS2=KS2, KP=KP, POS=POS)


def view_bank(records, episodes, data):
    """Банк видов для --view-aug: каждое состояние строки (s и, отдельно, s'), у которого в файлах видов есть
    построенные виды, получает k альтернативных входов — та же строка data['S'] (data['S2']) с картами вида v
    вместо карт среды (каналы 0-2) и каналом delta (3) по тому же виду предыдущего шага, как у
    map_views.view_rows: шаг 1 — нули; дальше — разность с видом v состояния шага s-1, если он есть, иначе с
    картой среды шага s-1 из буфера. Каналы скалярного контекста (4+) не меняются: вид — другое наблюдение того
    же состояния. Ключ состояния — view_row_keys; первый встреченный файл видов с ключом (по порядку имён) —
    его, как dedupe у view_rows. Вид 0 (карта среды) — это сама строка буфера, в банк не входит.
    Возвращает dict: ALT (M, C, H, W) float32 — сырые (до нормировки) входы, OFF_S/K_S и OFF_S2/K_S2 (N,) —
    начало и число видов строки в ALT (K = 0 — видов нет), и сводку info."""
    import map_views as MV
    S, S2 = data["S"], data["S2"]
    if S.shape[1] < 4:
        raise ValueError("--view-aug: нужны 4 канала карт (3 карты среды и delta)")
    views = {}
    for rec in records:
        chain = rec["chain"]
        for s, v in (rec.get("views") or {}).items():
            if len(v):
                views.setdefault(MV.state_key(rec["seed"], chain, int(s)), np.asarray(v, dtype=np.float32))
    kk = view_row_keys(episodes, data)
    EP, STEP = np.asarray(data["EP"]), np.asarray(data["STEP"])
    n = len(EP)
    # карта среды (loss_total) шага p цепочки: s строки со STEP = p
    tot = {(int(EP[i]), int(STEP[i])): S[i, 0] for i in range(n)}
    alt = []
    OFF_S, K_S = np.zeros(n, np.int64), np.zeros(n, np.int64)
    OFF_S2, K_S2 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    m = 0
    for side, keys, base, OFF, K in (("s", kk["KS"], S, OFF_S, K_S), ("s2", kk["KS2"], S2, OFF_S2, K_S2)):
        for i in range(n):
            key = keys[i]
            if key is None or key not in views:
                continue
            p = int(kk["POS"][i]) + (1 if side == "s2" else 0)      # шаг состояния в цепочке файла
            vs = views[key]
            # виды предыдущего состояния: у s' — сама s строки, у s — шаг s-1 той же цепочки
            pk = kk["KS"][i] if side == "s2" else kk["KP"][i]
            prev_v = views.get(pk) if pk is not None else None
            prev_env = tot.get((int(EP[i]), p - 1)) if p >= 2 else None
            OFF[i], K[i] = m, len(vs)
            for v in range(len(vs)):
                x = np.array(base[i], dtype=np.float32, copy=True)
                x[:3] = vs[v]
                if p < 2:
                    x[3] = 0.0                      # после первого действия delta нулевой, как у среды
                else:
                    prev = prev_v[v][0] if (prev_v is not None and v < len(prev_v)) else prev_env
                    x[3] = _delta_like_env(vs[v][0], prev)
                alt.append(x)
                m += 1
    ALT = (np.stack(alt).astype(np.float32) if alt else np.zeros((0,) + S.shape[1:], np.float32))
    ms, ms2 = K_S > 0, K_S2 > 0
    info = dict(records=len(records), states=len(views), rows=int(n), rows_s=int(ms.sum()), rows_s2=int(ms2.sum()),
                mean_k_s=(round(float(K_S[ms].mean()), 3) if ms.any() else 0.0),
                mean_k_s2=(round(float(K_S2[ms2].mean()), 3) if ms2.any() else 0.0),
                rows_keyed=int(sum(k is not None for k in kk["KS"])), alt=int(m))
    return dict(ALT=ALT, OFF_S=OFF_S, K_S=K_S, OFF_S2=OFF_S2, K_S2=K_S2, info=info)


def load_view_records(spec, subdir, log=print):
    """Файлы видов по спецификации --view-aug (как cv_probe --views: префиксы rl_arch/views/<тег> или папки)."""
    import map_views as MV
    return MV.load_records(spec, subdir, log)


def attach_views(data, episodes, spec, subdir, log=print):
    """--view-aug: банк видов в data['VIEWS'] (не массив строк: срезы набора его не трогают). Возвращает сводку."""
    recs = load_view_records(spec, subdir, log)
    bank = view_bank(recs, episodes, data)
    data["VIEWS"] = bank
    return dict(bank["info"])


class ViewAug:
    """Замена карт выбранных строк пачки видом (--view-aug). Банк — view_bank, нормированный как вход сети и
    лежащий на устройстве. Случайность — свой генератор numpy (сид обучения): поток перестановок пачек и
    D4 тот же, что без флага, поэтому арм отличается от контроля только видами."""

    def __init__(self, bank, alt_t, p, seed):
        self.p = float(p)
        self.alt = alt_t
        self.off = {"s": bank["OFF_S"], "s2": bank["OFF_S2"]}
        self.k = {"s": bank["K_S"], "s2": bank["K_S2"]}
        self.rng = np.random.default_rng([int(seed), 58])
        self.n = {"s": 0, "s2": 0}
        self.hit = {"s": 0, "s2": 0}
        self.cand = {"s": 0, "s2": 0}

    def pick(self, x, rows, side):
        """x — входы строк rows (тензор пачки, копия); с вероятностью p строка с видами получает случайный из
        своих k видов (равновероятно). Два числа на строку при любом исходе. Возвращает x."""
        import torch
        k = self.k[side][rows]
        u, w = self.rng.random(len(rows)), self.rng.random(len(rows))
        sel = np.where((k > 0) & (u < self.p))[0]
        self.n[side] += len(rows)
        self.cand[side] += int((k > 0).sum())
        if len(sel):
            v = np.minimum((w[sel] * k[sel]).astype(np.int64), k[sel] - 1)
            j = torch.as_tensor(self.off[side][rows[sel]] + v, device=x.device).long()
            x[torch.as_tensor(sel, device=x.device).long()] = self.alt[j]
            self.hit[side] += len(sel)
        return x

    def pair(self, x, rows):
        """Пары разных наблюдений одного состояния для --view-cons: у строк с видами (k > 0) — два разных
        номера из 0..k (0 — карта среды, т.е. сама строка x). Возвращает (номера строк в пачке, xa, xb) или
        None, если в пачке нет строк с видами."""
        import torch
        k = self.k["s"][rows]
        u, w = self.rng.random(len(rows)), self.rng.random(len(rows))
        sel = np.where(k > 0)[0]
        if not len(sel):
            return None
        ks = k[sel]
        i1 = np.minimum((u[sel] * (ks + 1)).astype(np.int64), ks)
        i2 = (i1 + 1 + np.minimum((w[sel] * ks).astype(np.int64), ks - 1)) % (ks + 1)
        st = torch.as_tensor(sel, device=x.device).long()

        def obs(ii):
            out = x[st].clone()
            has = np.where(ii > 0)[0]
            if len(has):
                j = torch.as_tensor(self.off["s"][rows[sel[has]]] + ii[has] - 1, device=x.device).long()
                out[torch.as_tensor(has, device=x.device).long()] = self.alt[j]
            return out
        return sel, obs(i1), obs(i2)

    def summary(self):
        """Доли строк пачек: с видами и заменённых (по s и по s')."""
        out = {}
        for side in ("s", "s2"):
            n = max(self.n[side], 1)
            out[f"view_aug_cand_{side}"] = round(self.cand[side] / n, 4)
            out[f"view_aug_frac_{side}"] = round(self.hit[side] / n, 4)
        return out


def view_cons_loss(net, xa, xb):
    """--view-cons: MSE между скалярными Q всех действий на двух видах одного состояния; вторая сторона —
    целевая сеть без градиента (как цель TD). При одинаковых видах и целевой сети, равной онлайновой (в
    одном режиме), — ровно ноль."""
    import torch
    import torch.nn.functional as F
    qa = net.q_scalar(xa)
    with torch.no_grad():
        qb = net.scalar(net.q_target(xb))
    return F.mse_loss(qa, qb)


def compose_two_nets(q_opt, a_lr, a_ep, center="max"):
    """Совместное Q по 27 действиям из выходов двух сетей (--two-nets), a = opt*9 + lr*3 + ep:
        Q27[o, l, e] = Q_opt[o] + (A_lr[o, l] - c_l A_lr[o, .]) + (A_ep[o, e] - c_e A_ep[o, .]),
    q_opt — (B, 3) сети A, a_lr и a_ep — (B, 3, 3) сети B (оптимизатор, значение параметра).
    center='max' (c = max): обе добавки <= 0 и равны нулю на лучшем значении параметра, поэтому
    Q_opt[o] = max_{l,e} Q27[o, l, e], а жадное совместное действие совпадает с иерархическим
    выбором статьи — оптимизатор по сети A, затем шаг и число эпох независимо по сети B.
    center='mean' (c = среднее, как в дуэлинговой голове): Q_opt[o] — среднее Q27 по параметрам
    оптимизатора; шаг и эпохи по-прежнему выбирает сеть B, но оптимизатор жадного совместного
    действия — argmax_o [Q_opt[o] + размах A_lr[o] + размах A_ep[o]], то есть сеть A одна его уже
    не определяет."""
    a_lr, a_ep = _two_centered(a_lr, center), _two_centered(a_ep, center)
    q = q_opt[:, :, None, None] + a_lr[:, :, :, None] + a_ep[:, :, None, :]
    return q.reshape(q.shape[0], N_ACTIONS)


def _two_centered(a, center):
    """Преимущества минус их максимум или среднее по последней оси (значения одного параметра)."""
    if center == "max":
        return a - a.amax(-1, keepdim=True)
    if center == "mean":
        return a - a.mean(-1, keepdim=True)
    raise ValueError(f"центрирование двух сетей {center!r}: выбор {TWO_NETS_CENTER}")


def compose_two_nets_joint(q_opt, a_par, center="max"):
    """Совместное Q при --two-nets-mode joint: у оптимизатора одна голова на все пары (шаг,
    длительность), a_par — (B, 3, 9), пара p = lr*3 + ep:
        Q27[o, l, e] = Q_opt[o] + (A[o, l, e] - c_{l,e} A[o, ., .]).
    Сумма по шагу и длительности не предполагается: голова задаёт любую таблицу 3x3, в том числе
    со взаимодействием шага и длительности. center='max': Q_opt[o] = max_{l,e} Q27[o, l, e], жадное
    совместное действие = оптимизатор по сети A, затем лучшая пара по сети B. center='mean':
    Q_opt[o] — среднее по парам; пару по-прежнему выбирает сеть B, оптимизатор — сумма сетей."""
    q = q_opt[:, :, None] + _two_centered(a_par, center)
    return q.reshape(q.shape[0], N_ACTIONS)


def compose_two_nets_lowrank(q_opt, a_lr, a_ep, u_lr, v_ep, center="max"):
    """Совместное Q при --two-nets-mode lowrank: a_lr, a_ep — (B, 3, 3) как у additive, u_lr и v_ep —
    (B, 3, 3, k) векторы значений шага и длительности у каждого оптимизатора:
        T[o, l, e] = A_lr[o, l] + A_ep[o, e] + <U[o, l], V[o, e]>,
        Q27[o, l, e] = Q_opt[o] + (T[o, l, e] - c_{l,e} T[o, ., .]).
    Взаимодействие ранга k позволяет, например, «большой шаг хорош только при короткой длительности»,
    чего сумма additive выразить не может. center='max': Q_opt[o] = max_{l,e} Q27[o, ., .], жадное
    совместное действие = оптимизатор по сети A, затем лучшая пара (шаг, длительность) по таблице T."""
    t = a_lr[:, :, :, None] + a_ep[:, :, None, :] + torch.einsum("bolk,boek->bole", u_lr, v_ep)
    b, n_o = t.shape[:2]
    t = _two_centered(t.reshape(b, n_o, -1), center)
    return (q_opt[:, :, None] + t).reshape(b, N_ACTIONS)


def compose_two_nets_auto(q_opt, a_lr, a_ep, center="max"):
    """Совместное Q при --two-nets-mode auto (авторегрессия): a_lr — (B, 3, 3) преимущества шага
    при оптимизаторе, a_ep — (B, 3, 3, 3) преимущества длительности при оптимизаторе и шаге:
        Q27[o, l, e] = Q_opt[o] + (A_lr[o, l] - c_l A_lr[o, .]) + (A_ep[o, l, e] - c_e A_ep[o, l, .]).
    Длительность зависит от шага, поэтому суммы по шагу и длительности здесь нет. center='max':
    Q_opt[o] + A_lr[o, l] - max_l = max_e Q27[o, l, .], Q_opt[o] = max_{l,e} Q27[o, ., .], и жадное
    совместное действие равно последовательному выбору оптимизатор -> шаг -> длительность.
    center='mean': Q_opt[o] и Q_opt[o] + A_lr[o, l] - mean_l — средние по вложенным параметрам;
    совместный argmax тогда с последовательным выбором не совпадает (длительность при выбранном
    шаге по-прежнему даёт сеть B)."""
    q = q_opt[:, :, None, None] + _two_centered(a_lr, center)[:, :, :, None] + _two_centered(a_ep, center)
    return q.reshape(q.shape[0], N_ACTIONS)


class ParamHeads(nn.Module):
    """Головы сети параметров (--two-nets; DQN_params статьи): у каждого из трёх оптимизаторов своя
    голова на три шага обучения и своя на три длительности. forward -> (A_lr, A_ep), обе (B, 3, 3).
    kind='linear' — линейный слой, как голова лучшего рецепта; 'dueling' — поток преимуществ
    дуэлинговой головы статьи (Linear-ReLU-Linear). Поток ценности той головы здесь не создаётся:
    в compose_two_nets он сокращается при любом центрировании и остался бы без градиента."""

    def __init__(self, in_dim, kind="linear"):
        super().__init__()

        def mk():
            if kind == "dueling":
                return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 3))
            return nn.Linear(in_dim, 3)
        self.lr = nn.ModuleList([mk() for _ in range(3)])
        self.ep = nn.ModuleList([mk() for _ in range(3)])

    def forward(self, z):
        return (torch.stack([h(z) for h in self.lr], 1), torch.stack([h(z) for h in self.ep], 1))


class JointParamHeads(nn.Module):
    """Головы сети параметров при --two-nets-mode joint: у каждого оптимизатора ОДНА голова на все
    его пары (шаг, длительность) — n_pairs выходов, пара p = lr*3 + ep. forward -> (A,), A — (B, 3, 9).
    Новый оптимизатор — ещё одна голова на его собственное число пар; выходов у сети B — сумма
    по оптимизаторам, а не произведение на число оптимизаторов. kind — как у ParamHeads."""

    def __init__(self, in_dim, kind="linear", n_opt=3, n_pairs=9):
        super().__init__()

        def mk():
            if kind == "dueling":
                return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, n_pairs))
            return nn.Linear(in_dim, n_pairs)
        self.par = nn.ModuleList([mk() for _ in range(n_opt)])

    def forward(self, z):
        return (torch.stack([h(z) for h in self.par], 1),)


class AutoParamHeads(nn.Module):
    """Головы сети параметров при --two-nets-mode auto (авторегрессия): вложение оптимизатора и
    вложение шага; голова шага считает A_lr[o, .] по [признаки, emb(o)], голова длительности —
    A_ep[o, l, .] по [признаки, emb(o), emb(l)]. forward -> (A_lr (B, 3, 3), A_ep (B, 3, 3, 3)) за
    один проход: вложения перебирают все оптимизаторы и шаги. Выходов 3 + 3 при любом числе
    оптимизаторов: новый оптимизатор или новое значение параметра — одна строка вложения (и один
    выход своей головы), новый параметр — ещё одна голова и одно вложение, рост линейный.
    У обеих голов есть скрытый слой: линейный слой по [признаки, вложение] дал бы зависимость от
    оптимизатора и шага только сдвигом, не зависящим от состояния. kind='linear' — скрытый слой
    TWO_NETS_AUTO_HIDDEN, 'dueling' — шириной в признаки (как поток преимуществ головы статьи)."""

    def __init__(self, in_dim, kind="linear", n_opt=3, n_lr=3, n_ep=3, emb=TWO_NETS_AUTO_EMB):
        super().__init__()
        hid = in_dim if kind == "dueling" else TWO_NETS_AUTO_HIDDEN
        self.emb_o = nn.Embedding(n_opt, emb)
        self.emb_l = nn.Embedding(n_lr, emb)
        self.lr = nn.Sequential(nn.Linear(in_dim + emb, hid), nn.ReLU(), nn.Linear(hid, n_lr))
        self.ep = nn.Sequential(nn.Linear(in_dim + 2 * emb, hid), nn.ReLU(), nn.Linear(hid, n_ep))

    def forward(self, z):
        b, eo, el = z.shape[0], self.emb_o.weight, self.emb_l.weight
        zo = torch.cat([z[:, None, :].expand(b, len(eo), -1), eo[None].expand(b, -1, -1)], 2)
        zol = torch.cat([zo[:, :, None, :].expand(b, len(eo), len(el), -1),
                         el[None, None].expand(b, len(eo), -1, -1)], 3)
        return self.lr(zo), self.ep(zol)


class LowRankParamHeads(nn.Module):
    """Головы сети параметров при --two-nets-mode lowrank: у каждого оптимизатора голова шага и голова
    длительности, как у ParamHeads, но каждая отдаёт на значение параметра 1 + k чисел: преимущество и
    вектор ранга k для взаимодействия (compose_two_nets_lowrank). forward -> (A_lr, A_ep, U, V):
    (B, 3, 3), (B, 3, 3), (B, 3, 3, k), (B, 3, 3, k). Выходов (1 + k) * (n_lr + n_ep) на оптимизатор —
    сумма по параметрам, не произведение. kind — как у ParamHeads."""

    def __init__(self, in_dim, kind="linear", n_opt=3, n_lr=3, n_ep=3, rank=TWO_NETS_RANK):
        super().__init__()
        self.rank, self.n_lr, self.n_ep = int(rank), n_lr, n_ep

        def mk(n):
            out = n * (1 + self.rank)
            if kind == "dueling":
                return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out))
            return nn.Linear(in_dim, out)
        self.lr = nn.ModuleList([mk(n_lr) for _ in range(n_opt)])
        self.ep = nn.ModuleList([mk(n_ep) for _ in range(n_opt)])

    def forward(self, z):
        k = self.rank
        lr = torch.stack([h(z) for h in self.lr], 1).view(z.shape[0], -1, self.n_lr, 1 + k)
        ep = torch.stack([h(z) for h in self.ep], 1).view(z.shape[0], -1, self.n_ep, 1 + k)
        return lr[..., 0], ep[..., 0], lr[..., 1:], ep[..., 1:]


def make_param_heads(mode, in_dim, kind="linear"):
    """Головы сети параметров по --two-nets-mode."""
    if mode == "lowrank":
        return LowRankParamHeads(in_dim, kind)
    if mode == "joint":
        return JointParamHeads(in_dim, kind)
    if mode == "auto":
        return AutoParamHeads(in_dim, kind)
    return ParamHeads(in_dim, kind)


class QNet:
    """Encoder + head with a target copy; variant-specific loss.

    encoder — вид энкодера карт у вариантов с ConvNeXt ('auto' — прежний выбор по варианту,
    иначе один из MAP_ENCODERS). ens > 1 — столько Q-голов на общем энкодере (REDQ, 2101.05982):
    model[1] тогда список голов, q_online/q_target/q_scalar/q_cvar отдают СРЕДНЕЕ голов (у
    квантильных — среднее квантилей с одним номером), q_heads — все головы по отдельности.
    bins > 0 — голова HL-Gauss (--hlg; Farebrother et al., 2403.03950): у вариантов HLG_VARIANTS
    линейная голова отдаёт 27 x bins логитов, q_online/q_target — (B, A, bins), скаляр Q —
    матожидание по центрам бинов (scalar, q_scalar; сами бины — в self._hlg, его ставит тренер и
    online_eval_env.load_agent по полю hl_gauss чекпоинта).
    two_nets — две отдельные сети, как у агента статьи (--two-nets; RL/rl_utils/DQN_classes.py):
    сеть A — энкодер и голова на 3 оптимизатора (model[0], model[1]), сеть B — свой энкодер того же
    вида и головы шага и длительности у каждого оптимизатора (model[2], model[3] — ParamHeads).
    Наружу (q_online, q_target, q_heads, q_scalar) — совместное Q по 27 действиям из
    compose_two_nets с центрированием two_center; выходы сетей по отдельности — q_parts, выбор
    «оптимизатор, затем параметры» — hier_action. two_head — вид голов (TWO_NETS_HEAD).
    two_mode (TWO_NETS_MODE) — головы сети B и состав совместного Q: additive — ParamHeads и
    compose_two_nets (прежнее), joint — JointParamHeads и compose_two_nets_joint, auto —
    AutoParamHeads и compose_two_nets_auto. two_shared — у сети B нет своего энкодера, её головы
    берут признаки энкодера сети A (как в коде статьи: RL/rl_algorithms.py подаёт сети параметров
    свёрточные признаки сети оптимизатора — там это признаки до MLP энкодера, здесь выход
    энкодера целиком); model[2] тогда пустой nn.Identity, ключей '2.*' в чекпоинте нет. two_shared_grad: 'both' — лосс сети B учит и общий энкодер, 'stop' — признаки
    для сети B отрезаны от графа, энкодер учит только сеть A.
    ft_mode (FT_MODES, --ft-mode) — какие параметры энкодеров карт учатся; у lora адаптеры (ранг
    lora_rank, масштаб lora_alpha / lora_rank) встают в каждый энкодер сети при создании, до копии
    в целевую сеть, — их ключи есть в чекпоинте, load_agent строит их по meta (ckpt_arch). Заморозку
    по режиму включает apply_ft (тренер, после --encoder-init).
    При значениях по умолчанию сеть и имена параметров прежние."""

    def __init__(self, variant, device, n_quantiles=32, in_ch=4, encoder="auto", ens=1, bins=0,
                 two_nets=False, two_center="max", two_head="linear", two_mode="additive",
                 two_shared=False, two_shared_grad="both", ft_mode="full", lora_rank=4, lora_alpha=8.0):
        import torch
        import torch.nn as nn
        if variant in ("their_dqn", "their_cql"):
            enc_kind = "their"
        elif variant in CNX_VARIANTS:
            enc_kind = "convnext"
        elif variant == "stat_dqn":
            enc_kind = "stat"
        else:
            enc_kind = "cnn"
        if encoder not in (None, "", "auto", enc_kind):
            if enc_kind != "convnext" or encoder not in MAP_ENCODERS:
                raise ValueError(f"энкодер {encoder!r} у варианта {variant}: выбор {MAP_ENCODERS} есть "
                                 f"только у вариантов с ConvNeXt (здесь энкодер {enc_kind})")
            enc_kind = encoder
        self.ens = max(1, int(ens or 1))
        if self.ens > 1 and variant in NO_ENS_VARIANTS:
            raise ValueError(f"ансамбль голов (ens={self.ens}) не поддержан для варианта {variant}")
        self.variant = variant
        self.enc_kind = enc_kind
        self.nq = n_quantiles
        self.bins = max(0, int(bins or 0))
        self._hlg = None
        self.two_nets = bool(two_nets)
        self.two_center, self.two_head = two_center, two_head
        self.two_mode = two_mode or "additive"
        self.two_shared = bool(two_shared)
        self.two_shared_grad = two_shared_grad or "both"
        # где в model лежат энкодеры (у общего энкодера он один)
        self._enc_idx = (0, 2) if self.two_nets and not self.two_shared else (0,)
        if self.two_nets:
            if variant not in TWO_NETS_VARIANTS:
                raise ValueError(f"две сети (two_nets) есть только у {TWO_NETS_VARIANTS}, не у {variant}")
            if self.ens > 1 or self.bins:
                raise ValueError("две сети (two_nets) несовместимы с ансамблем голов (ens) и HL-Gauss (bins)")
            if two_center not in TWO_NETS_CENTER or two_head not in TWO_NETS_HEAD:
                raise ValueError(f"две сети: центрирование {two_center!r} из {TWO_NETS_CENTER}, "
                                 f"головы {two_head!r} из {TWO_NETS_HEAD}")
            if self.two_mode not in TWO_NETS_MODE or self.two_shared_grad not in TWO_NETS_SHARED_GRAD:
                raise ValueError(f"две сети: состав {self.two_mode!r} из {TWO_NETS_MODE}, градиент в общий "
                                 f"энкодер {self.two_shared_grad!r} из {TWO_NETS_SHARED_GRAD}")
            if self.two_shared_grad != "both" and not self.two_shared:
                raise ValueError("две сети: two_shared_grad='stop' имеет смысл только с общим энкодером (two_shared)")
        elif self.two_shared or self.two_mode != "additive" or self.two_shared_grad != "both":
            raise ValueError("two_mode, two_shared и two_shared_grad задаются только вместе с two_nets")
        self.enc = make_encoder(enc_kind, in_ch)
        if self.two_nets:
            # сеть A: голова на оптимизаторы (dueling — дуэлинговая голова статьи, Q = V + A - mean A)
            if two_head == "dueling":
                from RL.rl_utils.DQN_classes import DuelingHead
                self.head = DuelingHead(self.enc.out_dim, 3)
            else:
                self.head = nn.Linear(self.enc.out_dim, 3)
            # сеть B: свой энкодер (или признаки энкодера сети A при two_shared) и головы параметров
            self.enc_b = nn.Identity() if self.two_shared else make_encoder(enc_kind, in_ch)
            self.head_b = make_param_heads(self.two_mode, self.enc.out_dim, two_head)
        elif self.bins:
            if variant not in HLG_VARIANTS or self.ens > 1:
                raise ValueError(f"голова HL-Gauss (bins={self.bins}) есть только у {HLG_VARIANTS} с одной головой")
            self.head = nn.Linear(self.enc.out_dim, N_ACTIONS * self.bins)
        elif self.ens > 1:
            self.head = nn.ModuleList([make_head(variant, self.enc.out_dim, n_quantiles)
                                       for _ in range(self.ens)])
        else:
            self.head = make_head(variant, self.enc.out_dim, n_quantiles)
        self.v_head = nn.Linear(self.enc.out_dim, 1) if variant == "cnn_iql" else None
        mods = [self.enc, self.head] + ([self.v_head] if self.v_head is not None else [])
        if self.two_nets:
            mods = [self.enc, self.head, self.enc_b, self.head_b]
        self.model = nn.ModuleList(mods).to(device)
        # --ft-mode: проверка режима у этого энкодера; адаптеры lora — до копии в целевую сеть,
        # чтобы они были и там (soft_update идёт по тем же параметрам в том же порядке)
        self.ft_mode = ft_mode or "full"
        self.lora_rank, self.lora_alpha = int(lora_rank), float(lora_alpha)
        self._ft_eval = []                  # слои BatchNorm, которые при обучении стоят в режиме eval
        if self.ft_mode != "full":
            ft_check(self.ft_mode, enc_kind, self.enc)
            if self.ft_mode == "lora":
                if self.lora_rank < 1 or not self.lora_alpha > 0:
                    raise ValueError(f"--lora-rank >= 1 и --lora-alpha > 0, не {lora_rank} и {lora_alpha}")
                for i in self._enc_idx:
                    lora_wrap(self.model[i], self.lora_rank, self.lora_alpha)
        import copy
        self.target = copy.deepcopy(self.model).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.device = device
        # BatchNorm есть только у resnet: целевая сеть считается на накопленных статистиках
        # (режим eval), статистики копируются из обучаемой сети в soft_update
        self.has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.enc.modules())
        self.frozen = False
        if self.has_bn:
            self.target.eval()

    def params(self):
        return self.model.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def set_train(self, on=True):
        """Режим слоёв со статистиками (BatchNorm у resnet): обучение — статистики пачки, оценка —
        накопленные. У энкодеров без таких слоёв ничего не делает (прежнее поведение)."""
        if self.has_bn:
            self.model.train(on)
            if self.frozen:
                for i in self._enc_idx:
                    self.model[i].eval()
            for m in self._ft_eval:         # --ft-mode lastblock: замороженная часть энкодера
                m.eval()

    def apply_ft(self, keep=()):
        """--ft-mode: обучаемыми в энкодерах (у двух сетей — в обоих) остаются только параметры
        режима (ft_trainable_names) и keep — свежие входные каналы после --encoder-init (как у
        freeze_encoder: первая свёртка учится целиком; адаптер lora на ней тогда заморожен и, с нулевым
        up, ничего не добавляет). У lastblock слои BatchNorm вне последнего блока стоят в режиме eval:
        их статистики не меняются. Головы учатся целиком. Возвращает dict(обучаемых, всего) по
        параметрам энкодеров; при full ничего не меняет."""
        if self.ft_mode == "full":
            return None
        if self.frozen:
            raise ValueError("--ft-mode несовместим с --freeze-encoder")
        keep = set(keep)
        n_tr = n_all = 0
        self._ft_eval = []
        for i in self._enc_idx:
            enc = self.model[i]
            train = ft_trainable_names(enc, self.ft_mode, self.enc_kind)
            kept_layers = {k.rsplit(".", 1)[0] for k in keep}
            for name, p in enc.named_parameters():
                on = name in keep or (name in train and not (is_lora_key(name)
                                                             and name.split(".lora_")[0] in kept_layers))
                p.requires_grad_(on)
                n_tr += p.numel() * on
                n_all += p.numel()
            if self.ft_mode == "lastblock":
                pre = FT_LASTBLOCK[self.enc_kind]
                self._ft_eval += [m for mn, m in enc.named_modules()
                                  if isinstance(m, nn.modules.batchnorm._BatchNorm) and not (mn + ".").startswith(pre)]
        for m in self._ft_eval:
            m.eval()
        return dict(trainable=int(n_tr), total=int(n_all))

    def freeze_encoder(self, keep=()):
        """--freeze-encoder: градиент идёт только в головы. keep — имена параметров энкодера,
        которые остаются обучаемыми (первый слой, если у входа больше каналов, чем у
        предобученных весов). Возвращает число замороженных параметров. У двух сетей
        замораживаются оба энкодера."""
        if self.ft_mode != "full":
            raise ValueError("--freeze-encoder несовместим с --ft-mode")
        n = 0
        for i in self._enc_idx:
            for name, p in self.model[i].named_parameters():
                if name not in keep:
                    p.requires_grad_(False)
                    n += p.numel()
        self.frozen = True
        for i in self._enc_idx:
            self.model[i].eval()
        return n

    def n_params_nets(self):
        """Две сети: число параметров сети оптимизатора (A) и сети параметров (B). У общего
        энкодера он целиком засчитан сети A, у сети B — только головы."""
        cnt = lambda i: sum(p.numel() for p in self.model[i].parameters())  # noqa: E731
        return cnt(0) + cnt(1), cnt(2) + cnt(3)

    def n_outputs_nets(self):
        """Две сети: число выходов сети оптимизатора (A) и сети параметров (B) — выходные нейроны
        последних слоёв голов (у auto это 3 шага + 3 длительности при любом числе оптимизаторов)."""
        last = lambda m: [l for l in m.modules() if isinstance(l, nn.Linear)][-1].out_features  # noqa: E731
        b = self.model[3]
        heads = {"additive": lambda: list(b.lr) + list(b.ep), "joint": lambda: list(b.par),
                 "auto": lambda: [b.lr, b.ep], "lowrank": lambda: list(b.lr) + list(b.ep)}[self.two_mode]()
        return 3, sum(last(h) for h in heads)

    def _two_parts(self, model, x):
        """Выходы двух сетей: (Q_opt, *выходы голов сети B)."""
        z = model[0](x)
        if self.two_shared:
            zb = z.detach() if self.two_shared_grad == "stop" else z
        else:
            zb = model[2](x)
        return (model[1](z),) + tuple(model[3](zb))

    def q_parts(self, x, target=False):
        """Две сети по отдельности: Q_opt (B, 3) сети A и нецентрированные преимущества сети B.
        additive: A_lr, A_ep — (B, 3, 3) (оптимизатор, значение параметра); joint: A — (B, 3, 9)
        (оптимизатор, пара lr*3 + ep); auto: A_lr — (B, 3, 3), A_ep — (B, 3, 3, 3) (оптимизатор, шаг,
        длительность)."""
        return self._two_parts(self.target if target else self.model, x)

    def hier_action(self, x):
        """Выбор статьи: оптимизатор — argmax сети A, затем его параметры по сети B. additive: шаг
        и число эпох — независимые argmax; joint: лучшая пара (шаг, эпохи) оптимизатора; auto:
        последовательно шаг, затем число эпох при этом шаге. Возвращает номера действий (B,),
        a = opt*9 + lr*3 + ep; при центрировании max это argmax совместного Q."""
        q_o, *par = self.q_parts(x)
        o = q_o.argmax(1)
        rows = torch.arange(len(o), device=o.device)
        if self.two_mode == "joint":
            return o * 9 + par[0][rows, o].argmax(1)
        if self.two_mode == "lowrank":
            t = (par[0][rows, o][:, :, None] + par[1][rows, o][:, None, :]
                 + torch.einsum("blk,bek->ble", par[2][rows, o], par[3][rows, o]))
            return o * 9 + t.reshape(len(o), -1).argmax(1)
        l = par[0][rows, o].argmax(1)
        if self.two_mode == "auto":
            return o * 9 + l * 3 + par[1][rows, o, l].argmax(1)
        return o * 9 + l * 3 + par[1][rows, o].argmax(1)

    def _joint(self, model, x):
        compose = {"additive": compose_two_nets, "joint": compose_two_nets_joint,
                   "auto": compose_two_nets_auto, "lowrank": compose_two_nets_lowrank}[self.two_mode]
        return compose(*self._two_parts(model, x), center=self.two_center)

    def _heads(self, model, x):
        """Все головы: (головы, B, A) или (головы, B, A, nq)."""
        if self.two_nets:
            return self._joint(model, x)[None]
        z = model[0](x)
        hs = model[1] if self.ens > 1 else [model[1]]
        out = torch.stack([h(z) for h in hs], 0)
        if self.variant in QUANTILE_VARIANTS:
            return out.view(len(hs), -1, N_ACTIONS, self.nq)
        if self.bins:
            return out.view(len(hs), -1, N_ACTIONS, self.bins)
        return out

    def q_heads(self, x, target=False):
        return self._heads(self.target if target else self.model, x)

    def _q(self, model, x):
        if self.two_nets:
            return self._joint(model, x)
        if self.ens > 1:
            return self._heads(model, x).mean(0)
        z = model[0](x)
        out = model[1](z)
        if self.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
            return out.view(-1, N_ACTIONS, self.nq)
        if self.bins:
            return out.view(-1, N_ACTIONS, self.bins)
        return out

    def scalar(self, q):
        """Выход головы -> скаляр Q по действиям: среднее квантилей, матожидание бинов HL-Gauss,
        у обычной головы — сам выход."""
        if self.variant in QUANTILE_VARIANTS:
            return q.mean(-1)
        if self.bins:
            return self._hlg.to_scalar(q)
        return q

    def q_online(self, x):
        return self._q(self.model, x)

    def q_target(self, x):
        return self._q(self.target, x)

    def v_online(self, x):
        return self.model[2](self.model[0](x)).squeeze(-1)

    def q_scalar(self, x):
        """(B, N_ACTIONS) scalar Q for metrics/policies (mean over quantiles)."""
        return self.scalar(self.q_online(x))

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
            if self.has_bn:
                for i in self._enc_idx:
                    for b, tb in zip(self.model[i].buffers(), self.target[i].buffers()):
                        tb.copy_(b)


def ckpt_arch(ck):
    """Аргументы QNet по чекпоинту: число входных каналов и квантилей выводится из весов (агенты
    со скалярным контекстом ждут 4+5 каналов; у HL-Gauss 51 бин, у квантильных вариантов 32),
    вид энкодера и число голов — из meta. У чекпоинтов без этих полей — прежняя сеть."""
    sd = ck["state_dict"]
    meta = ck.get("meta") or {}
    ens = max(1, int(meta.get("ens", 1) or 1))
    in_ch = 4
    for v in sd.values():
        if hasattr(v, "dim") and v.dim() == 4 and v.shape[2] <= 7:
            in_ch = int(v.shape[1])
            break
    if meta.get("encoder") == "resnetpos":
        in_ch -= 2                               # два канала координат сеть добавляет сама
    if ck["variant"] == "stat_dqn":
        # у кодировщика на статистиках свёрток нет: 8 признаков на канал
        in_ch = int(sd["0.mlp.0.weight"].shape[1] // 8)
    nq = 32
    hw = sd.get("1.weight" if ens == 1 else "1.0.weight")
    if hw is not None and hw.dim() == 2 and hw.shape[0] % N_ACTIONS == 0:
        nq = max(1, hw.shape[0] // N_ACTIONS)
    out = dict(n_quantiles=nq, in_ch=in_ch, encoder=(meta.get("encoder") or "auto"), ens=ens)
    hg = ck.get("hl_gauss")
    if (hg and ck["variant"] in HLG_VARIANTS and ens == 1 and hw is not None
            and hw.shape[0] == N_ACTIONS * int(hg[2])):
        # офлайновый --hlg: линейная голова на 27 x bins логитов (у онлайновых чекпоинтов HL-Gauss
        # бины лежат в квантильной голове, там хватает n_quantiles); ключ есть только у таких
        out["bins"] = int(hg[2])
    if meta.get("two_nets"):
        # --two-nets: вторая сеть лежит в ключах '2.*' и '3.*'; поля есть только у таких чекпоинтов.
        # Чекпоинты до --two-nets-mode и --two-nets-shared этих полей не несут: additive, свой энкодер
        out.update(two_nets=True, two_center=(meta.get("two_nets_center") or "max"),
                   two_head=(meta.get("two_nets_head") or "linear"),
                   two_mode=(meta.get("two_nets_mode") or "additive"),
                   two_shared=bool(meta.get("two_nets_shared")))
        if out["two_shared"]:
            # на значения Q не влияет; нужен дообучению с чекпоинта (тот же путь градиента)
            out["two_shared_grad"] = meta.get("two_nets_shared_grad") or "both"
    ft = meta.get("ft_mode") or "full"
    if ft != "full":
        # --ft-mode: у lora в весах есть адаптеры (ключи *.lora_down.*, *.lora_up.*) — сеть строится с ними,
        # с тем же рангом и масштабом; у bn и lastblock сеть прежняя, режим только записан
        out["ft_mode"] = ft
        if ft == "lora":
            if "lora_rank" not in meta or "lora_alpha" not in meta:
                raise ValueError("чекпоинт с ft_mode=lora без lora_rank и lora_alpha в meta")
            out.update(lora_rank=int(meta["lora_rank"]), lora_alpha=float(meta["lora_alpha"]))
    return out


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


def load_encoder_init(net, path):
    """--encoder-init: веса энкодера из файла (локальный путь или путь в HF-датасете результатов).
    Форматы: файл зонда CV (cv_probe.py --save-encoder: encoder, ssl, state_dict, mean, std);
    чекпоинт агента (variant + state_dict, берутся ключи '0.*' — энкодер другого агента); голый
    state_dict энкодера. Если у сети больше входных каналов, чем у весов (скалярный контекст),
    в первую свёртку копируются веса первых каналов, остальные каналы остаются со свежей
    инициализацией (их имена — в partial). У двух сетей (--two-nets) веса получают оба энкодера
    (с --two-nets-shared он один); из чекпоинта такого агента берётся энкодер сети оптимизатора ('0.*').
    Возвращает dict(source, encoder, ssl, mean, std, partial)."""
    src = path
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OUT_REPO, path, repo_type="dataset")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = dict(source=os.path.basename(src), encoder=None, ssl=None, mean=None, std=None, partial=[])
    if isinstance(ck, dict) and "state_dict" in ck and "variant" in ck:
        sd = {k[2:]: v for k, v in ck["state_dict"].items() if k.startswith("0.")}
        info.update(encoder=(ck.get("meta") or {}).get("encoder"), mean=ck.get("mean"), std=ck.get("std"))
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
        info.update(encoder=ck.get("encoder"), ssl=ck.get("ssl"), mean=ck.get("mean"), std=ck.get("std"))
    else:
        sd = ck
    if info["encoder"] not in (None, "auto", net.enc_kind):
        raise ValueError(f"--encoder-init: в файле энкодер {info['encoder']!r}, у сети {net.enc_kind!r} "
                         f"(задайте --encoder {info['encoder']})")
    for i in getattr(net, "_enc_idx", (0,)):        # у двух сетей — оба энкодера (свежие каналы у каждого свои)
        enc = net.model[i]
        own = enc.state_dict()
        # --ft-mode lora: адаптеры в файле весов не лежат и остаются как созданы (up нулевой)
        ada = {k: v for k, v in own.items() if is_lora_key(k)}
        own = {k: v for k, v in own.items() if k not in ada}
        if set(sd) != set(own):
            raise ValueError(f"--encoder-init: имена параметров не совпали с энкодером {net.enc_kind}: "
                             f"лишние {sorted(set(sd) - set(own))[:4]}, нет {sorted(set(own) - set(sd))[:4]}")
        new = {}
        for k, v in own.items():
            w = torch.as_tensor(sd[k]).to(v.dtype)
            if w.shape == v.shape:
                new[k] = w
            elif (w.dim() == 4 and v.dim() == 4 and w.shape[0] == v.shape[0] and w.shape[2:] == v.shape[2:]
                  and w.shape[1] < v.shape[1]):
                m = v.detach().cpu().clone()
                m[:, :w.shape[1]] = w
                new[k] = m
                if k not in info["partial"]:
                    info["partial"].append(k)
            else:
                raise ValueError(f"--encoder-init: {k}: форма в файле {tuple(w.shape)}, у сети {tuple(v.shape)}")
        new.update(ada)
        enc.load_state_dict(new)
        net.target[i].load_state_dict(enc.state_dict())
    return info


def top_chain_rows(data, rows_mask, top_pct):
    """--distill-top: строки лучших top_pct% цепочек — по ошибке после последнего шага цепочки
    (с --episode-budget это ошибка в бюджете), в смеси УрЧП отдельно по каждой задаче. Ранжируются
    только цепочки из rows_mask (обучающая часть). Возвращает (маска строк, сводка)."""
    EP, ERR = data["EP"], data["ERR"]
    pde = data["PDE"] if "PDE" in data else np.zeros(len(EP), dtype=np.int64)
    last = np.r_[np.where(np.diff(EP) != 0)[0], len(EP) - 1]
    last = last[rows_mask[last]]
    top = []
    for k in np.unique(pde[last]):
        l = last[pde[last] == k]
        top.append(EP[l][ERR[l] <= np.percentile(ERR[l], top_pct)])
    top = np.concatenate(top) if top else np.zeros(0, dtype=EP.dtype)
    demo = np.isin(EP, top) & rows_mask
    return demo, dict(distill_chains=int(len(top)), distill_ranked=int(len(last)), distill_rows=int(demo.sum()))


def coarse_actions(lr_idx=1):
    """--mask-schedule: грубое подмножество действий — один шаг обучения (индекс lr_idx) на пару
    «оптимизатор x длина», как agent_combo.coarse_mask без итоговой маски."""
    return (np.arange(N_ACTIONS) % 9) // 3 == int(lr_idx)


def ens_td_loss(net, variant, s, a, r, d, gm, s2, taus, cql_alpha, n_min, rng, ub=None, allow=None,
                bcq_keep=None):
    """TD-лосс ансамбля голов (REDQ, 2101.05982). Действие в s' — argmax СРЕДНЕГО обучаемых голов
    (двойной DQN), цель — минимум по n_min случайным целевым головам, общая для всех голов; лосс
    усредняется по головам, так что его масштаб тот же, что у одной головы. У квантильных голов
    минимум берётся поквантильно после сортировки (квантили с одним номером у разных голов):
    результат — снова квантильная функция, не выше каждой из взятых. CQL и штраф BCQ — на каждой
    голове. ub — верхняя граница ценности (--value-bound), allow — действия, разрешённые в max
    (--mask-schedule)."""
    K = net.ens
    quant = variant in QUANTILE_VARIANTS
    qh = net.q_heads(s)                                               # (K,B,A[,nq])
    with torch.no_grad():
        q2m = net.q_heads(s2).mean(0)
        if quant:
            q2m = q2m.mean(-1)
        if allow is not None:
            q2m = q2m.masked_fill(~allow[None], -1e9)
        a2 = q2m.argmax(1)
        qt = net.q_heads(s2, target=True)
        if n_min < K:
            qt = qt[torch.as_tensor(rng.choice(K, n_min, replace=False), device=qt.device)]
        M = len(qt)
        if quant:
            q2 = qt.gather(2, a2[None, :, None, None].expand(M, -1, 1, net.nq)).squeeze(2)   # (M,B,nq)
            q2 = q2.sort(-1).values.min(0).values                                            # (B,nq)
            tgt = r[:, None] + (gm * (1 - d))[:, None] * q2
        else:
            q2 = qt.gather(2, a2[None, :, None].expand(M, -1, 1)).squeeze(2).min(0).values   # (B,)
            if ub is not None:
                q2 = torch.minimum(q2.clamp_min(0.0), ub)
            tgt = r + gm * (1 - d) * q2
    ai = a[None, :, None].expand(K, -1, 1)
    if quant:
        q_data = qh.gather(2, ai[..., None].expand(-1, -1, -1, net.nq)).squeeze(2)           # (K,B,nq)
        u = tgt[None, :, None, :] - q_data[:, :, :, None]                                    # (K,B,nq,nq)
        huber = torch.where(u.abs() <= 1.0, 0.5 * u ** 2, u.abs() - 0.5)
        loss = (torch.abs(taus[None, None, :, None] - (u.detach() < 0).float()) * huber).mean()
        qv = qh.mean(-1)
    else:
        qv = qh
        q = qv.gather(2, ai).squeeze(2)                                                      # (K,B)
        loss = F.mse_loss(q, tgt[None].expand_as(q))
    if bcq_keep is not None:
        loss = loss + 0.5 * (qv.masked_fill(bcq_keep[None], 0.0).clamp_min(0) ** 2).mean()
    if variant in ("cnn_cql", "cnx_cql", "their_cql", "cnx_cql_qr"):
        loss = loss + cql_alpha * (torch.logsumexp(qv, dim=2) - qv.gather(2, ai).squeeze(2)).mean()
    return loss


def hlg_range(data, idx, args):
    """--hlg: опора бинов (v_min, v_max) в единицах награды тренера; --hlg-vmin / --hlg-vmax задают
    её явно. Иначе: у награды-разности (delta) ценность не больше текущей ошибки (эту границу и
    использует --value-bound), а на старте цепочки текущая ошибка — ошибка необученной сети в
    единицах тренера (--init-err; на смеси с --err-norm init это 1). Она восстанавливается из
    данных как R + ERR первого шага (награда = ошибка до - ошибка после); v_max = 1.1 от
    максимума по обучающим цепочкам, v_min = -0.5 * v_max (действие, после которого ошибка выше
    стартовой в полтора раза и хуже, неотличимо от такого же плохого). Состояния с ошибкой выше
    стартовой (расходимость) и их цели обрезаются опорой. У остальных форм награды — отрезок
    квантилей 0.5%..99.5% возврата по записанным цепочкам (RTG), расширенный на 10%."""
    lo, hi = getattr(args, "hlg_vmin", None), getattr(args, "hlg_vmax", None)
    if lo is None or hi is None:
        idx = np.asarray(idx)
        a_lo = a_hi = None
        if getattr(args, "reward_form", "") == "delta" and "ERR" in data:
            first = np.asarray(data["FIRST"])[idx] > 0 if "FIRST" in data else np.zeros(len(idx), bool)
            before = (np.asarray(data["R"], np.float64) + np.asarray(data["ERR"], np.float64))[idx]
            top = float(before[first].max()) if first.any() else float(np.quantile(before, 0.99))
            if np.isfinite(top) and top > 0:
                a_lo, a_hi = -0.55 * top, 1.1 * top
        if a_hi is None:
            q0, q1 = np.quantile(np.asarray(data["RTG"], np.float64)[idx], [0.005, 0.995])
            pad = 0.1 * max(float(q1 - q0), 1e-3)
            a_lo, a_hi = float(q0) - pad, float(q1) + pad
        lo = a_lo if lo is None else float(lo)
        hi = a_hi if hi is None else float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"--hlg: опора бинов [{lo}, {hi}] пуста (задайте --hlg-vmin и --hlg-vmax)")
    return float(lo), float(hi)


def aux_next_targets(data, idx, chunk=20000):
    """--aux-next: цель вспомогательной головы — шесть чисел уровня СЛЕДУЮЩИХ карт
    (surrogate.level6 от S2: минимум и среднее трёх карт потерь) и награда перехода, нормированные
    по обучающим строкам idx. Следующие карты в массивах тренера есть (S2). Возвращает
    (Y float32 (N, 7), W float32 (N, 7)): вес 0 у уровня карт терминальных переходов (следующей
    карты там нет — как в surrogate.build_table) и у нечисловых значений."""
    import mb_targets as MB
    n = len(data["A"])
    lv = np.zeros((n, 6), dtype=np.float64)
    for k in range(0, n, chunk):                 # level6 считает в float64: по частям, без копии всего S2
        lv[k:k + chunk] = MB.S.level6(data["S2"][k:k + chunk, :3])
    y = np.concatenate([lv, np.asarray(data["R"], np.float64)[:, None]], 1)
    w = np.isfinite(y)
    w[:, :6] &= (np.asarray(data["D"]) != 1)[:, None]
    y = np.where(w, y, 0.0)
    sel = np.zeros(n, dtype=bool)
    sel[np.asarray(idx)] = True
    wt = w & sel[:, None]
    cnt = np.maximum(wt.sum(0), 1)
    mu = (y * wt).sum(0) / cnt
    sd = np.sqrt((((y - mu) ** 2) * wt).sum(0) / cnt) + 1e-6
    return ((y - mu) / sd * w).astype(np.float32), w.astype(np.float32)


def train_variant(variant, data, train_mask, args, seed):
    if variant == "cnx_smdp":
        return train_smdp(data, train_mask, args, seed)
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    in_ch = int(data["S"].shape[1])
    n_ens = max(1, int(getattr(args, "ens", 1) or 1))
    ens_min = max(1, min(n_ens, int(getattr(args, "ens_min", 2) or 2)))

    # --hlg: голова на бинах, обучение кросс-энтропией (по умолчанию выключено: прежняя голова)
    n_bins = int(getattr(args, "hlg_bins", 51) or 51) if getattr(args, "hlg", False) else 0

    # --two-nets: две сети статьи вместо одной (по умолчанию выключено: прежняя сеть)
    two = bool(getattr(args, "two_nets", False))
    if two and float(getattr(args, "aux_next", 0.0) or 0.0) > 0:
        raise ValueError("--two-nets несовместим с --aux-next: вспомогательная голова стоит на одном энкодере")

    def mk_net():
        return QNet(variant, device, in_ch=in_ch, encoder=getattr(args, "encoder", "auto"), ens=n_ens,
                    bins=n_bins, two_nets=two, two_center=getattr(args, "two_nets_center", "max"),
                    two_head=getattr(args, "two_nets_head", "linear"),
                    two_mode=getattr(args, "two_nets_mode", "additive"),
                    two_shared=bool(getattr(args, "two_nets_shared", False)),
                    two_shared_grad=getattr(args, "two_nets_shared_grad", "both"),
                    **ft_kw)

    # --ft-mode: сколько энкодера учится (по умолчанию full — весь, прежняя сеть и прежний оптимизатор)
    ft_mode = getattr(args, "ft_mode", "full") or "full"
    ft_kw = {}
    if ft_mode != "full":
        if variant in ("cnx_bbf",):
            raise ValueError("--ft-mode несовместим с cnx_bbf: он заново создаёт сеть и оптимизатор")
        ft_kw = dict(ft_mode=ft_mode, lora_rank=int(getattr(args, "lora_rank", 4)),
                     lora_alpha=float(getattr(args, "lora_alpha", 8.0)))
    net = mk_net()
    enc_info = None
    if getattr(args, "encoder_init", ""):
        enc_info = load_encoder_init(net, args.encoder_init)
        print(f"  [{variant} s{seed}] энкодер {net.enc_kind} инициализирован из {enc_info['source']}"
              + (f" (предобучение {enc_info['ssl']})" if enc_info["ssl"] else "")
              + (f"; свежие входные каналы у {enc_info['partial']}" if enc_info["partial"] else ""),
              flush=True)
    if getattr(args, "freeze_encoder", False):
        n_fr = net.freeze_encoder(keep=(enc_info["partial"] if enc_info else ()))
        print(f"  [{variant} s{seed}] энкодер заморожен ({n_fr} параметров), учатся головы"
              + ("" if enc_info else "; --encoder-init не задан: энкодер случайный"), flush=True)
        opt = torch.optim.Adam([p for p in net.params() if p.requires_grad], lr=float(getattr(args, "lr", 1e-3)),
                               weight_decay=float(getattr(args, "l2", 0.0) or 0.0))
    elif ft_mode != "full":
        ft_n = net.apply_ft(keep=(enc_info["partial"] if enc_info else ()))
        n_head = sum(p.numel() for i, m in enumerate(net.model) if i not in net._enc_idx
                     for p in m.parameters() if p.requires_grad)
        print(f"  [{variant} s{seed}] дообучение энкодера {ft_mode}"
              + (f" (LoRA ранг {net.lora_rank}, alpha {net.lora_alpha:g})" if ft_mode == "lora" else "")
              + f": учатся {ft_n['trainable']} из {ft_n['total']} параметров энкодера"
              + (f" (энкодеров {len(net._enc_idx)})" if len(net._enc_idx) > 1 else "")
              + (f", свежие каналы {list(enc_info['partial'])} целиком" if enc_info and enc_info["partial"] else "")
              + f"; головы — {n_head}" + ("" if enc_info else "; --encoder-init не задан: энкодер случайный"),
              flush=True)
        ft_info = dict(ft_enc_trainable=ft_n["trainable"], ft_enc_total=ft_n["total"], ft_head_trainable=int(n_head))
        # оптимизатору — только обучаемые параметры
        opt = torch.optim.Adam([p for p in net.params() if p.requires_grad], lr=float(getattr(args, "lr", 1e-3)),
                               weight_decay=float(getattr(args, "l2", 0.0) or 0.0))
    else:
        opt = torch.optim.Adam(net.params(), lr=float(getattr(args, "lr", 1e-3)),
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
    aux_info = {}
    if ft_mode != "full":
        aux_info.update(ft_info)
    if two:
        n_a, n_b = net.n_params_nets()
        o_a, o_b = net.n_outputs_nets()
        aux_info.update(n_params_opt_net=int(n_a), n_params_par_net=int(n_b),
                        n_out_opt_net=int(o_a), n_out_par_net=int(o_b))
        print(f"  [{variant} s{seed}] две сети: оптимизатор {n_a} параметров, параметры {n_b}; "
              f"совместное Q, центрирование {net.two_center}, головы {net.two_head}; "
              f"состав {net.two_mode}, выходов {o_a} + {o_b}, энкодер "
              + (f"общий (градиент сети параметров: {net.two_shared_grad})" if net.two_shared
                 else "у каждой сети свой"), flush=True)
    if ((net.frozen or ft_mode == "lastblock") and enc_info and enc_info["mean"] is not None
            and getattr(args, "state_mode", "full") == "full"):
        # замороженный энкодер ждёт вход в той нормировке, в которой предобучался: статистики
        # карт берутся из файла весов (в чекпоинт агента пишутся они же). То же у --ft-mode lastblock:
        # его замороженная часть с неизменными статистиками BatchNorm — тот же предобученный энкодер
        m0 = np.asarray(enc_info["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        s0 = np.asarray(enc_info["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        c = min(4, m0.shape[1], in_ch)
        mean[:, :c], std[:, :c] = m0[:, :c], s0[:, :c]
        aux_info["norm_from_encoder_init"] = True

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
    # --- приёмы для агента на картах (по умолчанию выключены: прежний лосс) ---
    # модельные цели: Q_mb(s, .) по всем 27 действиям, посчитаны заранее (mb_targets.py)
    w_mb = float(getattr(args, "mb_aux", 0.0) or 0.0) if QT is None else 0.0
    QMB = MBW = None
    if w_mb:
        if "QMB" not in data:
            raise ValueError("--mb-aux: в наборе нет модельных целей QMB (их считает mb_targets.precompute)")
        assert data["QMB"].shape == (len(data["A"]), N_ACTIONS), data["QMB"].shape
        QMB = to_t(data["QMB"]).float()
        MBW = to_t(data["MBW"].astype(np.float32))
    # --mb-trust: множитель слагаемого строки (вес доверия w(s) и ворота, mb_targets.trust_weights); без
    # флага столбца нет и лосс прежний бит в бит
    MBT = to_t(np.asarray(data["MBT"], dtype=np.float32)) if (w_mb and "MBT" in data) else None
    # Q учителя на скалярах в единицах награды тренера (teacher_targets): добавка к TD, не замена
    w_te = float(getattr(args, "teacher_w", 0.0) or 0.0) if QT is None else 0.0
    QTE = TEW = None
    if w_te:
        if "QTE" not in data:
            raise ValueError("--teacher-w: в наборе нет целей учителя QTE (их считает teacher_targets)")
        assert data["QTE"].shape == (len(data["A"]), N_ACTIONS), data["QTE"].shape
        QTE = to_t(data["QTE"]).float()
        TEW = to_t((data["TEV"] & train_mask).astype(np.float32))
    w_pani = float(getattr(args, "pani_w", 0.0) or 0.0) if QT is None else 0.0
    pani_sigma = float(getattr(args, "pani_sigma", 0.1) or 0.0)
    w_demo = float(getattr(args, "distill_w", 0.0) or 0.0) if QT is None else 0.0
    DEMO = None
    if w_demo and float(getattr(args, "distill_top", 0.0) or 0.0) > 0:
        demo_np, dinfo = top_chain_rows(data, train_mask, float(args.distill_top))
        aux_info.update(dinfo)
        DEMO = to_t(demo_np.astype(np.float32))
    coarse = None
    if getattr(args, "mask_schedule", False):
        coarse = to_t(coarse_actions(getattr(args, "mask_coarse_lr", 1)))
    hlg = None
    hlg_ce = getattr(args, "hlg_aux", "ev") == "ce"
    if n_bins:
        from advanced_agents import HLGauss     # тот же класс, что у онлайнового тренера и load_agent
        v_lo, v_hi = hlg_range(data, idx, args)
        hlg = HLGauss(v_min=v_lo, v_max=v_hi, n_bins=n_bins).to(device)
        net._hlg = hlg
        net.hl_gauss = (float(v_lo), float(v_hi), int(n_bins))      # поле hl_gauss чекпоинта
        aux_info.update(hlg_vmin=float(v_lo), hlg_vmax=float(v_hi), hlg_bins=int(n_bins))
        print(f"  [{variant} s{seed}] HL-Gauss: {n_bins} бин(ов) на [{v_lo:.4g}, {v_hi:.4g}], "
              f"sigma {hlg.sigma:.4g}, цель TD обрезается этим отрезком", flush=True)
    # --aux-next: голова на признаках энкодера и вложении действия предсказывает уровень СЛЕДУЮЩИХ
    # карт (surrogate.level6 от S2) и награду перехода; в чекпоинт не попадает (не часть net.model)
    w_nx = float(getattr(args, "aux_next", 0.0) or 0.0) if QT is None else 0.0
    nx_head = nx_emb = NXT = NXW = None
    if w_nx:
        import torch.nn as nn
        nx_y, nx_w = aux_next_targets(data, idx)
        NXT, NXW = to_t(nx_y).float(), to_t(nx_w).float()
        nx_emb = nn.Embedding(N_ACTIONS, 16).to(device)
        nx_head = nn.Sequential(nn.Linear(net.enc.out_dim + 16, 128), nn.GELU(),
                                nn.Linear(128, NXT.shape[1])).to(device)
        opt.add_param_group({"params": list(nx_emb.parameters()) + list(nx_head.parameters())})
        aux_info["aux_next_params"] = int(sum(p.numel() for m in (nx_emb, nx_head) for p in m.parameters()))
    aux_acc = {}
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    ckpts = []
    # --mc-lb: нижняя граница Q записанного действия — дисконтированная сумма наград до конца цепочки
    w_lb = float(getattr(args, "mc_lb", 0.0) or 0.0) if QT is None else 0.0
    lb_mode = getattr(args, "mc_lb_mode", "loss") or "loss"
    GLB = to_t(mc_returns(data, GAMMA)).float() if w_lb else None
    if w_lb:
        aux_info.update(mc_lb_rows_pos=int((mc_returns(data, GAMMA)[idx] > 0).sum()))
    # --wa: среднее весов по итерациям с эпохи wa_from (по умолчанию выключено)
    wa_mode = getattr(args, "wa", "none") or "none"
    wa_avg, wa_n = None, 0
    wa_from = int(round(float(getattr(args, "wa_start", 0.5)) * args.epochs)) if wa_mode != "none" else 0

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

    # --view-aug: виды карт строк (банк data['VIEWS'] строит attach_views); без флага ничего не меняется
    va, w_vc = None, 0.0
    if getattr(args, "view_aug", "") and QT is None:
        bank = data.get("VIEWS")
        if bank is None:
            raise ValueError("--view-aug: в наборе нет банка видов data['VIEWS'] (его строит attach_views)")
        va = ViewAug(bank, to_t(norm(bank["ALT"])).float(), float(getattr(args, "view_aug_p", VIEW_AUG_P)), seed)
        w_vc = float(getattr(args, "view_cons", 0.0) or 0.0)
        tr_s, tr_s2 = int((bank["K_S"][idx] > 0).sum()), int((bank["K_S2"][idx] > 0).sum())
        aux_info.update(view_rows_s_train=tr_s, view_rows_s2_train=tr_s2)
        print(f"  [{variant} s{seed}] виды карт: у {tr_s} обучающих переходов из {len(idx)} есть виды s, у {tr_s2} — "
              f"виды s'; замена с вероятностью {va.p:g}"
              + (f", согласованность видов с весом {w_vc:g}" if w_vc else ""), flush=True)
    # --lr-schedule: шаг Adam по шагам всего обучения (const — прежний постоянный шаг)
    lr_sched = getattr(args, "lr_schedule", "const") or "const"
    lr_min = float(getattr(args, "lr_min", 0.0) or 0.0)
    lr_total = n_epochs * max(1, int(math.ceil(len(idx) / bs)))
    lr_base = [pg["lr"] for pg in opt.param_groups]
    gstep = 0

    t0 = time.time()
    rng = np.random.default_rng(seed)
    best_td, stall = float("inf"), 0
    test_idx = np.where(~train_mask)[0]
    net.set_train(True)
    for epoch in range(n_epochs):
        order = rng.permutation(idx)
        # --mask-schedule: первую треть обучения max в цели видит только грубую сетку действий
        allow = coarse if (coarse is not None and epoch < n_epochs // 3) else None
        aux_acc = {}
        for k in range(0, len(order), bs):
            b = to_t(order[k:k + bs]).long()
            # r/d/gm — n-шаговые: награда за m шагов, терминальность и дисконт
            # в точке бутстрэпа; s2 — next_state именно того перехода (NIDX)
            s, a, r, d, gm = S[b], A[b], NR[b], ND[b], NG[b]
            s2 = S2[NIDX[b]]
            s_raw = s                      # неаугментированное s: маска BCQ и margin
            if va is not None:
                # виды: s и s' заменяются независимо (у каждого свой вид); строки без видов — как есть
                rows_np = order[k:k + bs]
                s = va.pick(s.clone(), rows_np, "s")
                s2 = va.pick(s2, nidx_np[rows_np], "s2")
            if aug_n > 1:                  # одно преобразование на батч, общее для s и s'
                g = int(rng.integers(aug_n))
                s = d4_apply(s, g); s2 = d4_apply(s2, g)

            if QT is not None:
                loss = F.mse_loss(net.q_online(s), QT[b])
            elif hlg is not None:
                # HL-Gauss (как online_train_env.hlg_update, но в общем лоссе с добавками): цель TD
                # двойного DQN по матожиданиям бинов -> сглаженное распределение по бинам,
                # кросс-энтропия с логитами взятого действия
                logits = net.q_online(s)                              # (B,A,bins)
                taken = logits.gather(1, a[:, None, None].expand(-1, 1, n_bins)).squeeze(1)
                with torch.no_grad():
                    q2o = hlg.to_scalar(net.q_online(s2))
                    if allow is not None:
                        q2o = q2o.masked_fill(~allow[None], -1e9)
                    a2 = q2o.argmax(1)
                    q2 = hlg.to_scalar(net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
                    if vbound:
                        q2 = torch.minimum(q2.clamp_min(0.0), UB[b])
                    tgt = (r + gm * (1 - d) * q2).clamp(v_lo, v_hi)
                loss = hlg.loss(taken, tgt).mean()
                if variant in ("cnn_cql", "cnx_cql"):
                    qs = hlg.to_scalar(logits)
                    loss = loss + args.cql_alpha * (
                        torch.logsumexp(qs, dim=1) - qs.gather(1, a[:, None]).squeeze(1)
                    ).mean()
            elif net.ens > 1:
                keep = None
                if variant == "cnx_bcq":
                    with torch.no_grad():
                        pb = bcq_behaviour(s_raw.flatten(1)).softmax(-1)
                        keep = pb >= 0.3 * pb.max(-1, keepdim=True).values
                loss = ens_td_loss(net, variant, s, a, r, d, gm, s2, taus, args.cql_alpha, ens_min, rng,
                                   ub=(UB[b] if vbound else None), allow=allow, bcq_keep=keep)
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
                    if allow is None:
                        a2 = net.q_online(s2).mean(-1).argmax(1)
                    else:
                        a2 = net.q_online(s2).mean(-1).masked_fill(~allow[None], -1e9).argmax(1)
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
                    if allow is None:
                        a2 = net.q_online(s2).argmax(1)
                    else:
                        a2 = net.q_online(s2).masked_fill(~allow[None], -1e9).argmax(1)
                    q2 = net.q_target(s2).gather(1, a2[:, None]).squeeze(1)
                    if vbound:
                        q2 = torch.minimum(q2.clamp_min(0.0), UB[b])
                    tgt = r + gm * (1 - d) * q2
                    if w_lb and lb_mode == "target":
                        # MCAC: цель не ниже записанного исхода цепочки
                        tgt = torch.maximum(tgt, GLB[b])
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

            if w_mb or w_pani or DEMO is not None or w_te:
                # плотные добавки считаются на каждой голове: (K,B,A), у квантильных — среднее квантилей
                qv = ql = net.q_heads(s)
                if variant in QUANTILE_VARIANTS:
                    qv = qv.mean(-1)
                elif hlg is not None:
                    qv = hlg.to_scalar(ql)      # --hlg: добавки действуют на матожидание головы

                def dense(tgt_q):
                    """Плотная добавка по 27 действиям, (B,): Huber по скаляру Q; при --hlg-aux ce —
                    кросс-энтропия логитов с распределением HL-Gauss цели (обрезанной опорой бинов)."""
                    if hlg is not None and hlg_ce:
                        return hlg.loss(ql[0], tgt_q.clamp(v_lo, v_hi)).mean(1)
                    return F.smooth_l1_loss(qv, tgt_q[None].expand_as(qv), reduction="none").mean((0, 2))
                if w_mb:
                    # W * Huber(Q(s, .) - Q_mb(s, .)) по всем 27 действиям; строки без модельной цели
                    # (отложенная часть, оборванная цепочка) имеют вес 0
                    wb = MBW[b]
                    if wb.sum() > 0:
                        hub = dense(QMB[b])
                        if MBT is None:
                            l_mb = (hub * wb).sum() / wb.sum()
                        else:
                            # доверие: вес строки умножается на её множитель, нормировка — по числу
                            # строк с целью, поэтому малое доверие уменьшает добавку, а не перераспределяет её
                            l_mb = (hub * wb * MBT[b]).sum() / wb.sum()
                        loss = loss + w_mb * l_mb
                        aux_acc.setdefault("mb_loss", []).append(l_mb.item())
                if w_te:
                    # W * Huber(Q(s, .) - Q_учителя(s, .)) по всем 27 действиям на строках с целью учителя
                    wt = TEW[b]
                    if wt.sum() > 0:
                        hub = dense(QTE[b])
                        l_te = (hub * wt).sum() / wt.sum()
                        loss = loss + w_te * l_te
                        aux_acc.setdefault("teacher_loss", []).append(l_te.item())
                if w_pani:
                    # сглаживание по входу: шум на нормированных картах (каналы контекста не трогаем),
                    # штраф за изменение вектора Q; чистый Q — цель без градиента
                    sn = s.clone()
                    sn[:, :4] = sn[:, :4] + pani_sigma * torch.randn_like(sn[:, :4])
                    qn = net.q_heads(sn)
                    if variant in QUANTILE_VARIANTS:
                        qn = qn.mean(-1)
                    elif hlg is not None:
                        qn = hlg.to_scalar(qn)
                    l_pani = F.smooth_l1_loss(qn, qv.detach())
                    loss = loss + w_pani * l_pani
                    aux_acc.setdefault("pani_loss", []).append(l_pani.item())
                if DEMO is not None:
                    # отступ DQfD только на строках лучших цепочек (у --dqfd — на всех переходах)
                    wd = DEMO[b]
                    if wd.sum() > 0:
                        ai = a[None, :, None].expand(len(qv), -1, 1)
                        marg = torch.full_like(qv, float(args.dqfd_margin))
                        marg.scatter_(2, ai, 0.0)
                        jm = ((qv + marg).max(2).values - qv.gather(2, ai).squeeze(2)).mean(0)
                        l_demo = (jm * wd).sum() / wd.sum()
                        loss = loss + w_demo * l_demo
                        aux_acc.setdefault("distill_top_loss", []).append(l_demo.item())

            if w_lb and lb_mode == "loss":
                # SIL / optimality tightening: W * relu(G - Q(s, a))^2 / 2 на каждой голове, только снизу
                qh = net.q_heads(s)
                if variant in QUANTILE_VARIANTS:
                    qh = qh.mean(-1)
                elif hlg is not None:
                    qh = hlg.to_scalar(qh)
                q_taken = qh.gather(2, a[None, :, None].expand(len(qh), -1, 1)).squeeze(2)    # (K,B)
                gap = (GLB[b][None] - q_taken).clamp_min(0.0)
                l_lb = 0.5 * (gap ** 2).mean()
                loss = loss + w_lb * l_lb
                aux_acc.setdefault("mc_lb_loss", []).append(l_lb.item())
                aux_acc.setdefault("mc_lb_active", []).append(float((gap > 0).float().mean()))

            if w_nx:
                # W * Huber(прогноз - [уровень следующих карт (6), награда]) в нормированных единицах;
                # у терминальных переходов следующей карты нет — там только награда (вес NXW)
                z = net.model[0](s)
                pred = nx_head(torch.cat([z, nx_emb(a)], 1))
                hub = F.smooth_l1_loss(pred, NXT[b], reduction="none")
                l_nx = (hub * NXW[b]).sum() / NXW[b].sum().clamp_min(1.0)
                loss = loss + w_nx * l_nx
                aux_acc.setdefault("aux_next_loss", []).append(l_nx.item())

            if w_vc:
                # согласованность видов: Q(s) на двух разных наблюдениях одного состояния (карта среды и
                # виды), вторая сторона — целевая сеть без градиента; только строки с видами
                pr = va.pair(S[b], rows_np)
                if pr is not None:
                    _, xa, xb = pr
                    if aug_n > 1:
                        xa, xb = d4_apply(xa, g), d4_apply(xb, g)
                    l_vc = view_cons_loss(net, xa, xb)
                    loss = loss + w_vc * l_vc
                    aux_acc.setdefault("view_cons_loss", []).append(l_vc.item())

            if lr_sched != "const":
                for pg, b0 in zip(opt.param_groups, lr_base):
                    pg["lr"] = lr_at(gstep, lr_total, b0, lr_min, lr_sched)
            gstep += 1
            opt.zero_grad(); loss.backward(); opt.step()
            net.soft_update(float(getattr(args, "tau", 0.005)))
            if wa_mode != "none" and epoch >= wa_from:
                wa_n += 1
                wa_avg = wa_update(wa_avg, net.model, wa_n, wa_mode, float(getattr(args, "wa_decay", 0.99)))
            if variant == "cnx_bbf" and (epoch * 1000 + k) % 4000 == 3999:
                # BBF: shrink-and-perturb — ствол сохраняет 50%, головы заново
                import copy as _copy
                fresh = mk_net()
                with torch.no_grad():
                    for (nm, p), pf in zip(net.model.named_parameters(),
                                           fresh.model.parameters()):
                        a_keep = 0.0 if nm.startswith("1") else 0.5
                        p.mul_(a_keep).add_((1 - a_keep) * pf)
                net.target.load_state_dict(net.model.state_dict())
                opt = torch.optim.Adam(net.params(), lr=1e-3)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  [{variant} s{seed}] epoch {epoch+1}/{n_epochs} loss={loss.item():.4f}"
                  + (f" lr={opt.param_groups[0]['lr']:.3g}" if lr_sched != "const" else ""), flush=True)
        if args.plateau_patience and (epoch + 1) % 25 == 0:
            net.set_train(False)
            with torch.no_grad():
                b = to_t(test_idx).long()
                qs = net.q_scalar(S[b]).gather(1, A[b][:, None]).squeeze(1)
                a2 = net.q_scalar(S2[b]).argmax(1)
                q2 = net.scalar(net.q_target(S2[b])).gather(1, a2[:, None]).squeeze(1)
                td = float(((qs - (R[b] + GAMMA * (1 - D[b]) * q2)) ** 2).mean().sqrt())
            net.set_train(True)
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
    net.set_train(False)
    aux_info.update({k: float(np.mean(v)) for k, v in aux_acc.items()})   # средние последней эпохи
    if va is not None:
        aux_info.update(va.summary())
    if lr_sched != "const":
        aux_info.update(lr_steps=int(lr_total), lr_last=float(opt.param_groups[0]["lr"]))
    if wa_avg is not None:
        # --wa: в сеть кладутся усреднённые веса; BatchNorm (кроме замороженного энкодера) пересчитывается
        # на обучающих состояниях. Доля обучающих строк, где жадное действие среднего совпало с последней
        # итерацией, — мера дрожания политики в конце обучения
        def greedy_rows():
            with torch.no_grad():
                return torch.cat([net.q_scalar(S[to_t(idx[k:k + 512]).long()]).argmax(1)
                                  for k in range(0, len(idx), 512)])
        g_last = greedy_rows()
        net.model.load_state_dict(wa_avg)
        n_bn = 0
        if not net.frozen:
            # у --ft-mode lastblock статистики замороженной части не пересчитываются
            n_bn = recompute_bn(net.model, net.q_online, S[to_t(idx).long()], bs=bs, skip=net._ft_eval)
        net.set_train(False)
        g_avg = greedy_rows()
        # сам режим и распад пишет main() в поля прогона (wa, wa_start, wa_decay); здесь — только итог
        aux_info.update(wa_n=int(wa_n), wa_from_epoch=int(wa_from), wa_bn_layers=int(n_bn),
                        wa_agree_last=round(float((g_last == g_avg).float().mean()), 4))
        print(f"  [{variant} s{seed}] веса усреднены ({wa_mode}, {wa_n} итераций с эпохи {wa_from}, BatchNorm "
              f"пересчитан в {n_bn} слоях): жадное действие совпадает с последней итерацией на "
              f"{aux_info['wa_agree_last']:.3f} обучающих строк", flush=True)
    net.extra = {}
    if w_te:
        # совпадение жадных действий ученика и учителя на отложенных строках с целью учителя
        # (среди действий, разрешённых учителю его маской)
        tv = np.where(~train_mask & data["TEV"])[0]
        allow = torch.ones(N_ACTIONS, dtype=torch.bool, device=device)
        for i in (getattr(args, "_teacher_soft_mask", None) or []):
            allow[int(i)] = False
        same = 0
        with torch.no_grad():
            for k in range(0, len(tv), 512):
                b = to_t(tv[k:k + 512]).long()
                same += int((net.q_scalar(S[b]).masked_fill(~allow[None], -1e9).argmax(1)
                             == QTE[b].masked_fill(~allow[None], -1e9).argmax(1)).sum())
        aux_info["teacher_agree"] = round(same / len(tv), 4) if len(tv) else None
        aux_info["teacher_agree_rows"] = int(len(tv))
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
    net.extra.update(aux_info)
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
        q2t = batched(lambda x: net.scalar(net.q_target(x)), S2[idx])
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


def run_suffix(args, subdir=None):
    """Суффикс имён результата и чекпоинта (его же повторяет queue/make_queue.py: off_model)."""
    subdir = args.subdir if subdir is None else subdir
    return (("_fixns" if args.fix_next_state else "")
            + ("_cfix" if args.chain_fix else "")
            + (("_r" + args.reward_form) if args.reward_form != "logged" else "")
            + ("_vb" if args.value_bound else "")
            + ("" if subdir == SUBDIR else "_" + subdir.split("_")[0])
            + (("_" + args.model_tag) if args.model_tag else ""))


def verify_ckpt(path, net, n=4):
    """Чекпоинт глазами оценки (как agent_combo.verify_ckpt): online_eval_env.load_agent строит ту
    же сеть (энкодер, число голов, каналы), q_values на нулевом (старт цепочки) и случайных
    состояниях отдаёт 27 конечных чисел и совпадает с обучавшейся сетью."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import online_eval_env as E
    # импорт среды тянет deepxde, а он на GPU делает тензором по умолчанию cuda: возвращаем
    # умолчание обратно, иначе входы CPU-сети создаются на cuda (см. agent_combo.verify_ckpt)
    torch.set_default_tensor_type(torch.FloatTensor)
    ag, mean, std, variant = E.load_agent(path)
    in_ch = int(np.asarray(mean).shape[1])
    out = dict(variant=variant, in_ch=in_ch, encoder=getattr(ag, "enc_kind", None),
               ens=int(getattr(ag, "ens", 1)))
    rng = np.random.default_rng(0)
    worst, scale, finite = 0.0, 1.0, True
    for k in range(n):
        st = np.zeros((in_ch, 26, 26), dtype=np.float32)
        if k:
            st = (np.asarray(mean)[0] + np.asarray(std)[0]
                  * rng.normal(size=st.shape)).astype(np.float32)
        q = E.q_values(ag, st, mean, std, variant)
        finite = finite and q.shape == (N_ACTIONS,) and bool(np.isfinite(q).all())
        if finite:
            scale = max(scale, float(np.abs(q).max()))
            worst = max(worst, float(np.abs(q - E.q_values(net, st, mean, std, variant)).max()))
    # допуск относительный: обучавшаяся сеть может жить на GPU, загруженная — на CPU
    out.update(max_abs_dq=worst, finite=finite,
               ok=bool(finite and worst <= 1e-3 * scale and out["encoder"] == getattr(net, "enc_kind", None)
                       and out["ens"] == int(getattr(net, "ens", 1))))
    two = (bool(getattr(net, "two_nets", False)), bool(getattr(ag, "two_nets", False)))
    if any(two):
        # --two-nets: загруженная сеть — тоже две сети с тем же центрированием и теми же головами
        # и тем же составом совместного Q, и так же устроенным энкодером сети параметров
        same = (two[0] == two[1] and all(getattr(ag, k, None) == getattr(net, k, None)
                                         for k in ("two_center", "two_head", "two_mode", "two_shared")))
        out.update(two_nets=two[1], two_nets_center=getattr(ag, "two_center", None),
                   two_nets_mode=getattr(ag, "two_mode", None), two_nets_shared=getattr(ag, "two_shared", None),
                   ok=bool(out["ok"] and same))
    ft = (getattr(net, "ft_mode", "full"), getattr(ag, "ft_mode", "full"))
    if any(f != "full" for f in ft):
        # --ft-mode: тот же режим, у lora — те же ранг и масштаб адаптеров
        keys = ("ft_mode", "lora_rank", "lora_alpha") if "lora" in ft else ("ft_mode",)
        same = all(getattr(ag, k, None) == getattr(net, k, None) for k in keys)
        out.update(ft_mode=ft[1], ok=bool(out["ok"] and same))
        if "lora" in ft:
            out.update(lora_rank=getattr(ag, "lora_rank", None), lora_alpha=getattr(ag, "lora_alpha", None))
    return out


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
    ap.add_argument("--train-tol", type=float, default=0.0,
                    help="Правило остановки эпизода обучения (статья, формула (11)): цепочка кончается на первом "
                         "переходе с ошибкой после действия <= T (единицы тренера, как --init-err: rmse+brmse, у "
                         "смеси с --err-norm init — доля E0); переход терминален, дальше переходов нет. Модельные "
                         "цели (--mb-aux) и цели учителя (--teacher-w) кончаются там же. 0 — выключено. Нужен --chain-fix")
    ap.add_argument("--train-kmax", type=int, default=0,
                    help="Правило остановки эпизода обучения: не больше K стадий (статья: K_max = 10); K-я стадия "
                         "терминальна, модельные цели и цели учителя дальше K-й стадии не смотрят. 0 — выключено. "
                         "Нормировку шага в контексте (--ctx-kmax) не меняет. Нужен --chain-fix")
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
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="шаг Adam у обучения агента (подбор гиперпараметров; прежнее значение 1e-3)")
    ap.add_argument("--tau", type=float, default=0.005,
                    help="скорость мягкого обновления целевой сети (прежнее значение 0.005)")
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
    ap.add_argument("--target-chains", type=int, default=0,
                    help="Кривая «размер буфера»: оставить N случайных записанных цепочек задачи --subdir "
                         "(0 = все) — те же, что оставляет agent_combo.py --target-chains N --data-seed K. На них "
                         "учится суррогат модельных целей (--mb-aux, он же щит оценки; остальные задачи его "
                         "таблицы не трогаются), и только их переходы идут ученику (TD, --mb-aux, --teacher-w, "
                         "отложенные 20%%). Выбор делается по таблице суррогата (--mb-subdirs, --mb-table, "
                         "--mb-work; учителю с --table нужна та же таблица). Нужен --chain-fix; без --subdirs, "
                         "--extra-buffers и --per-pde-files")
    ap.add_argument("--data-seed", type=int, default=0,
                    help="Сид выбора цепочек для --target-chains (как --data-seed у agent_combo.py)")
    ap.add_argument("--target-buffer", default="",
                    help="Данные задачи --subdir ТОЛЬКО из собранных заново файлов с картами "
                         "(collect_active.py --save-buffer): через запятую префиксы HF rl_arch/buffers_new/<тег> "
                         "и/или локальные папки. Исходный буфер задачи не читается ни учеником, ни суррогатом "
                         "модельных целей (--mb-aux: остальные задачи его таблицы — прежние буферы). Нужен "
                         "--chain-fix; без --subdirs, --extra-buffers, --target-chains, --per-pde-files, --data-dir")
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
    # --- приёмы для агента на картах (по умолчанию всё выключено: прежняя сеть и прежний лосс) ---
    ap.add_argument("--encoder", default="auto", choices=["auto"] + list(MAP_ENCODERS),
                    help="Энкодер карт у вариантов с ConvNeXt (convnext_dqn, cnx_*): auto — как раньше "
                         "(convnext); resnet — малый ResNet под 26x26; resnetpos — он же с каналами координат и "
                         "без усреднения по карте; vit — ViT с патчем 2 "
                         "(определения общие с cv_probe.py). Вид энкодера пишется в meta чекпоинта")
    ap.add_argument("--encoder-init", default="",
                    help="Веса энкодера для старта: локальный путь или путь в HF-датасете результатов. "
                         "Файл зонда CV (cv_probe.py --save-encoder), чекпоинт агента или state_dict")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Заморозить энкодер, учить только головы. С --encoder-init и режимом full "
                         "нормировка карт берётся из файла весов")
    ap.add_argument("--ft-mode", default="full", choices=list(FT_MODES),
                    help="Какие параметры энкодера карт учатся (у --two-nets — обоих энкодеров): full — все "
                         "(как раньше); bn — только аффинные параметры BatchNorm (статистики копятся как при full); "
                         "lora — веса энкодера стоят, к каждой Conv2d и Linear добавлен адаптер LoRA (плюс аффинные "
                         "параметры BatchNorm); lastblock — только последний блок ResNet (body.3) и проекция head, "
                         "остальное — в режиме eval, нормировка карт из файла --encoder-init (как у --freeze-encoder). "
                         "Головы учатся целиком; свежие каналы первой свёртки после --encoder-init — тоже. "
                         "Режим, ранг и масштаб пишутся в meta чекпоинта")
    ap.add_argument("--lora-rank", type=int, default=4, help="--ft-mode lora: ранг адаптеров")
    ap.add_argument("--lora-alpha", type=float, default=8.0,
                    help="--ft-mode lora: масштаб добавки alpha / rank")
    ap.add_argument("--ens", type=int, default=1,
                    help="REDQ: число Q-голов на общем энкодере (1 = одна голова, как раньше). "
                         "Действие и Q на оценке — по среднему голов")
    ap.add_argument("--ens-min", type=int, default=2,
                    help="REDQ: цель TD — минимум по стольким случайным целевым головам")
    ap.add_argument("--mb-aux", type=float, default=0.0,
                    help="Вес модельных целей (0 = выключено): W * Huber(Q(s,.) - Q_mb(s,.)) по всем 27 "
                         "действиям; Q_mb считает ансамбль суррогата (mb_targets.py) один раз до "
                         "обучения. Нужны --chain-fix и --reward-form delta или dlog")
    ap.add_argument("--mb-h", type=int, default=3,
                    help="Модельные цели: глубина жадного просмотра вперёд в суррогате (1 = одно действие)")
    ap.add_argument("--mb-lam", type=float, default=0.5,
                    help="Модельные цели: награда = среднее по членам ансамбля - L * разброс (MOPO)")
    ap.add_argument("--mb-full", action="store_true",
                    help="Модельные цели: немиопический просмотр вперёд (mb_targets.lookahead_full) — лучевой "
                         "поиск глубины --mb-h и ширины --mb-beam по пессимистичной СУММЕ наград до конца "
                         "бюджета или глубины; остановиться можно после любого действия. Без флага — прежний "
                         "жадный просмотр (лучшая немедленная награда, обрыв на неположительной)")
    ap.add_argument("--mb-hs", default="",
                    help="Модельные цели по нескольким горизонтам через запятую (например 1,2,3,5; SHARP, ICLR 2027 "
                         "su2U0XecdI): цель — среднее Q_h по горизонтам (mb_targets._mix_horizons). Путь просмотра "
                         "от горизонта не зависит, поэтому все горизонты считаются за один проход глубины max — "
                         "вдвое дороже H=3 при max=5 (жадный просмотр). Вместо --mb-h; пусто — один горизонт --mb-h")
    ap.add_argument("--mb-trust", default="none", choices=("none", "state", "gate", "both"),
                    help="Доверие к модельным целям по ошибке суррогата на записанных переходах задачи-цели "
                         "(mb_targets.trust_weights; RG-TAIL, ICLR 2027 gspGh6aGPl). state — вес строки "
                         "w(s) = exp(-|log10 E предсказанная - log10 E записанная| / --mb-trust-tau); gate — вся "
                         "добавка умножается на ворота по доле верных знаков изменения ошибки (--mb-trust-gate-min); "
                         "both — оба. Предсказание — суррогатом, не видевшим кластер цепочек строки (перекрёстная "
                         "подгонка, --mb-trust-folds). none — прежний лосс. Только один УрЧП")
    ap.add_argument("--mb-trust-tau", type=float, default=0.3,
                    help="--mb-trust state/both: масштаб ошибки в порядках (w = exp(-ошибка / tau))")
    ap.add_argument("--mb-trust-gate-min", type=float, default=0.7,
                    help="--mb-trust gate/both: доля верных знаков, с которой добавка берётся целиком; ниже — "
                         "линейно до нуля при 0.5 (знак не лучше монеты)")
    ap.add_argument("--mb-trust-folds", type=int, default=5,
                    help="--mb-trust: фолдов перекрёстной подгонки по кластерам цепочек цели (столько же "
                         "дополнительных обучений суррогата); 0 — по кластеру на фолд (исключение своей цепочки)")
    ap.add_argument("--mb-beam", type=int, default=3,
                    help="--mb-full: ширина луча на пару (состояние, первое действие). Предсказаний ансамбля "
                         "на состояние: 27 * (1 + 27 + (H - 2) * 27 * луч) при H >= 2")
    ap.add_argument("--mb-max-rows", type=int, default=0,
                    help="Модельные цели: не больше стольких обучающих строк с целью Q_mb (случайные, сид "
                         "--data-seed; 0 = все). У остальных строк вес добавки 0 — для смеси УрЧП, где "
                         "просмотр по всем строкам слишком долог")
    ap.add_argument("--mb-subdirs", default="all",
                    help="Модельные цели: буферы, на которых учится суррогат (all | solvable | список). "
                         "С --subdirs и --holdout отложенные папки из этого набора исключаются: суррогат "
                         "не видит задачу проверки переноса")
    ap.add_argument("--mb-table", default="",
                    help="Модельные цели: готовая таблица переходов npz (локально или в HF) вместо сборки")
    ap.add_argument("--mb-members", type=int, default=5, help="Модельные цели: членов ансамбля суррогата")
    ap.add_argument("--mb-work", default="agent_combo_work",
                    help="Модельные цели: кэш таблицы и суррогата (общий с agent_combo.py)")
    ap.add_argument("--teacher-from", default="",
                    help="Чекпоинт учителя на скалярах от agent_combo.py (локальный путь или путь в "
                         "HF-датасете результатов), например rl_arch/agents_online/<тег>.pt. Его состояние "
                         "(шаг, эпохи, прошлое действие, rmse+brmse, описатель задачи) восстанавливается из "
                         "строк буфера; задаётся вместе с --teacher-w")
    ap.add_argument("--teacher-w", type=float, default=0.0,
                    help="Вес дистилляции учителя (0 = выключено): W * Huber(Q(s,.) - Q_учителя(s,.)) по "
                         "всем 27 действиям ДОБАВЛЯЕТСЯ к TD и --mb-aux (у --distill-from регрессия "
                         "заменяет TD). Q учителя переводится в единицы награды тренера; нужны --chain-fix, "
                         "--reward-form delta или dlog и --init-err (на смеси УрЧП E0 берётся сам)")
    ap.add_argument("--pani-w", type=float, default=0.0,
                    help="Вес сглаживания по входу (0 = выключено): шум на нормированных картах, штраф "
                         "за изменение вектора Q")
    ap.add_argument("--pani-sigma", type=float, default=0.1,
                    help="Разброс шума --pani-w в единицах нормированной карты")
    ap.add_argument("--distill-top", type=float, default=0.0,
                    help="Лучшие P%% цепочек буфера (по ошибке в конце цепочки, по каждой задаче): "
                         "отступ DQfD только на их переходах (--dqfd действует на все). Нужен --chain-fix")
    ap.add_argument("--distill-w", type=float, default=0.0,
                    help="Вес отступа на лучших цепочках (отступ — --dqfd-margin)")
    ap.add_argument("--mask-schedule", action="store_true",
                    help="От грубого к точному: первую треть эпох max в цели TD видит один шаг обучения "
                         "на пару «оптимизатор x длина», дальше все 27 действий")
    ap.add_argument("--mask-coarse-lr", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--hlg", action="store_true",
                    help="HL-Gauss (Farebrother et al., 2403.03950): голова отдаёт логиты по бинам ценности, "
                         "цель TD — сглаженное гауссианой распределение, лосс — кросс-энтропия вместо MSE. "
                         f"Варианты {', '.join(HLG_VARIANTS)}, одна голова. Добавки --mb-aux, --teacher-w, "
                         "--pani-w, --distill-w, --dqfd действуют на матожидание головы. В чекпоинт пишется "
                         "hl_gauss=(v_min, v_max, бины): оценка сводит бины к скаляру Q")
    ap.add_argument("--hlg-bins", type=int, default=51, help="--hlg: число бинов")
    ap.add_argument("--hlg-vmin", type=float, default=None,
                    help="--hlg: нижняя граница опоры бинов (не задана: по данным, см. hlg_range)")
    ap.add_argument("--hlg-vmax", type=float, default=None,
                    help="--hlg: верхняя граница опоры (не задана: у delta — 1.1 ошибки на старте цепочки "
                         "в единицах тренера, то есть граница --value-bound)")
    ap.add_argument("--hlg-aux", default="ev", choices=["ev", "ce"],
                    help="--hlg: как считать --mb-aux и --teacher-w. ev — Huber по матожиданию головы "
                         "(как без --hlg); ce — кросс-энтропия с распределением HL-Gauss плотной цели")
    ap.add_argument("--aux-next", type=float, default=0.0,
                    help="Вес вспомогательной цели (0 = выключено): малая голова на признаках энкодера и "
                         "вложении действия предсказывает уровень следующих карт (6 чисел surrogate.level6 "
                         "от S2) и награду, W * Huber в нормированных единицах. В чекпоинт голова не "
                         "пишется, на оценку не влияет")
    ap.add_argument("--two-nets", action="store_true",
                    help="Две отдельные сети, как у агента статьи (RL/rl_utils/DQN_classes.py), вместо одной "
                         "с головой на 27 действий: сеть A — свой энкодер и голова на 3 оптимизатора, сеть B — "
                         "свой энкодер и головы шага и длительности у каждого оптимизатора. Энкодеры одного "
                         "вида (--encoder), оба стартуют с --encoder-init. Лоссы (TD, --mb-aux, --teacher-w, "
                         "--value-bound) считаются по совместному Q на 27 действий: Q_opt[o] + (A_lr[o,l] - "
                         "c A_lr[o,.]) + (A_ep[o,e] - c A_ep[o,.]); целевая сеть — та же сумма по целевым "
                         f"копиям. Варианты {', '.join(TWO_NETS_VARIANTS)}; без --ens, --hlg, --aux-next. "
                         "В meta чекпоинта пишется two_nets: оценка строит обе сети и получает те же 27 значений")
    ap.add_argument("--two-nets-center", default="max", choices=list(TWO_NETS_CENTER),
                    help="--two-nets: что вычитается из преимуществ параметров. max — жадное совместное "
                         "действие равно выбору статьи (оптимизатор по сети A, затем его шаг и эпохи по сети "
                         "B), Q_opt — ценность оптимизатора с лучшими параметрами; mean — среднее, как в "
                         "дуэлинговой голове: вычитаемое гладкое и тянет все три значения параметра, а не одно "
                         "лучшее, но Q_opt — уже средняя ценность оптимизатора, и выбирает его сумма двух сетей")
    ap.add_argument("--two-nets-head", default="linear", choices=list(TWO_NETS_HEAD),
                    help="--two-nets: головы. linear — линейные слои, как голова лучшего рецепта; dueling — "
                         "как в статье: дуэлинговая голова со скрытым слоем у сети A, её поток преимуществ у "
                         "голов сети B. У --two-nets-mode auto головы сети B всегда со скрытым слоем: linear — "
                         f"{TWO_NETS_AUTO_HIDDEN} нейронов, dueling — шириной в признаки энкодера")
    ap.add_argument("--two-nets-mode", default="additive", choices=list(TWO_NETS_MODE),
                    help="--two-nets: как сеть B раскладывает шаг и длительность. additive (прежнее) — у "
                         "оптимизатора голова шага и голова длительности, их преимущества складываются: 3 + 3 "
                         "выхода на оптимизатор, но взаимодействие шага с длительностью не выразить, плотные "
                         "цели по 27 действиям (--mb-aux, --teacher-w) с таким взаимодействием не выучить. "
                         "joint — у оптимизатора одна голова на все его пары (шаг, длительность): "
                         "Q27[o,l,e] = Q_opt[o] + (A[o,l,e] - c A[o,.,.]), 9 выходов на оптимизатор, любая "
                         "таблица; новый оптимизатор — один выход сети A и одна голова сети B. auto — "
                         "авторегрессия: вложение оптимизатора, голова шага по [признаки, emb(o)], голова "
                         "длительности по [признаки, emb(o), emb(l)]: Q27[o,l,e] = Q_opt[o] + (A_lr[o,l] - c) "
                         "+ (A_ep[o,l,e] - c A_ep[o,l,.]); выходов 3 + 3 при любом числе оптимизаторов, рост "
                         "по числу параметров линейный, при --two-nets-center max жадное действие — "
                         "последовательный выбор оптимизатор -> шаг -> длительность. lowrank — как additive "
                         "плюс взаимодействие шага и длительности ранга TWO_NETS_RANK: у значения параметра "
                         "вектор, T[o,l,e] = A_lr[o,l] + A_ep[o,e] + <U[o,l], V[o,e]>; (1 + k) * (3 + 3) "
                         "выходов на оптимизатор, рост по числу значений параметров — сумма, не произведение. "
                         "В meta: two_nets_mode")
    ap.add_argument("--two-nets-shared", action="store_true",
                    help="--two-nets: у сети B нет своего энкодера, её головы берут признаки энкодера сети A "
                         "(как в коде статьи: сети параметров подаются свёрточные признаки сети оптимизатора). "
                         "Параметров почти вдвое меньше; работает со всеми --two-nets-mode. В meta: "
                         "two_nets_shared")
    ap.add_argument("--two-nets-shared-grad", default="both", choices=list(TWO_NETS_SHARED_GRAD),
                    help="--two-nets-shared: both — лосс сети B учит и общий энкодер (так и в коде статьи: сумма "
                         "двух лоссов идёт назад через признаки сети оптимизатора); stop — признаки для сети B "
                         "отрезаны от графа, энкодер учит только сеть A, головы сети B учатся на готовых признаках")
    ap.add_argument("--no-verify", action="store_true",
                    help="Не проверять сохранённый чекпоинт через online_eval_env.load_agent")
    # --- составляющие метода, волна 50 (по умолчанию выключено: прежнее поведение бит в бит) ---
    ap.add_argument("--wa", default="none", choices=list(WA_MODES),
                    help="Усреднение весов по итерациям одного обучения; в чекпоинт и диагностики идут "
                         "усреднённые веса: ema — экспоненциальное среднее с распадом --wa-decay, swa — равное "
                         "среднее (Izmailov et al., UAI 2018). Статистики BatchNorm пересчитываются по обучающим "
                         "состояниям. Цель — меньше дрожания жадной политики в конце обучения (policy churn)")
    ap.add_argument("--wa-start", type=float, default=0.5,
                    help="--wa: доля эпох, после которой итерации входят в среднее (0 — с первой)")
    ap.add_argument("--wa-decay", type=float, default=0.99,
                    help="--wa ema: распад среднего за шаг обучения (0.99 — окно около 100 шагов)")
    ap.add_argument("--train-all", action="store_true",
                    help="Учиться на всех переходах (без отложенных 20%% цепочек). Диагностики evaluate/FQE "
                         "считаются на тех же цепочках, что и раньше, но они теперь внутри обучения "
                         "(в строке diag_in_sample=true). Несовместим с --fqe-select и --plateau-patience")
    ap.add_argument("--mc-lb", type=float, default=0.0,
                    help="Нижняя граница Q по записанному исходу (0 = выключено): G — дисконтированная сумма "
                         "наград от перехода до конца его цепочки; при неотрицательной ценности продолжения "
                         "(--value-bound) Q*(s, a_записанное) >= G. Вид — --mc-lb-mode")
    ap.add_argument("--mc-lb-mode", default="loss", choices=list(MC_LB_MODES),
                    help="--mc-lb: loss — штраф W * relu(G - Q(s,a))^2 / 2 на всех головах (SIL, Oh et al. "
                         "2018; optimality tightening, He et al. 2017); target — цель TD max(цель, G) (MCAC, "
                         "Wilcox et al. 2022), только у обычной головы DQN; W > 0 лишь включает её")
    # --- обучение дольше и на видах карт, волна 58 (по умолчанию выключено: прежнее поведение бит в бит) ---
    ap.add_argument("--view-aug", default="",
                    help="Виды карт собранного буфера цели (map_views.py, render_views.py): префиксы rl_arch/views/<тег> "
                         "через запятую или локальные папки, как cv_probe --views. У перехода, состояние которого "
                         "(сид + исполненное начало цепочки + шаг) есть в файлах видов, карты s и, независимо, s' "
                         "с вероятностью --view-aug-p заменяются случайным видом; канал delta — по тому же виду "
                         "предыдущего шага (map_views.view_rows). Переходы без видов (плоские карты) — как есть; "
                         "модельные цели и цели учителя от карт не зависят. Нужен --target-buffer")
    ap.add_argument("--view-aug-p", type=float, default=VIEW_AUG_P,
                    help="--view-aug: вероятность замены карт состояния с видами (по умолчанию 0.5)")
    ap.add_argument("--view-cons", type=float, default=0.0,
                    help="--view-aug: вес согласованности видов — MSE между Q(s) на двух разных наблюдениях одного "
                         "состояния (карта среды или вид), вторая сторона — целевая сеть без градиента; 0 — выключено")
    ap.add_argument("--lr-schedule", default="const", choices=list(LR_SCHEDULES),
                    help="План шага Adam по шагам всего обучения: const — постоянный --lr (прежнее), cosine — "
                         "косинусный спад от --lr до --lr-min (вместе с --epochs 1000: дольше учиться и не "
                         "развёртывать шумную последнюю итерацию)")
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="--lr-schedule cosine: шаг в конце обучения (по умолчанию 0)")
    args = ap.parse_args()
    if args.fqe_select and not args.ckpt_every:
        ap.error("--fqe-select требует --ckpt-every N")
    if args.n_step < 1:
        ap.error("--n-step должен быть >= 1")
    if args.smoke:
        args.epochs, args.fqe_epochs = 2, 2
    # выключатель задач: тег — имя результата без сида (вариант + суффикс + --model-tag)
    from killswitch import exit_if_killed
    kill_tag = f"offline_{args.variant}" + run_suffix(args, "multi" if args.subdirs else args.subdir)
    exit_if_killed(kill_tag, remote=True, what="обучение")
    new_arch = (args.encoder != "auto" or args.ens != 1 or args.encoder_init or args.freeze_encoder
                or args.two_nets or args.ft_mode != "full")
    if args.ft_mode != "full":
        # отказы до загрузки данных; энкодер, у которого режим не определён, отвергает ft_check
        if args.freeze_encoder:
            ap.error("--ft-mode несовместим с --freeze-encoder (тот замораживает энкодер целиком)")
        if args.variant in ("cnx_bbf", "cnx_smdp"):
            ap.error("--ft-mode несовместим с cnx_bbf (он заново создаёт сеть и оптимизатор) и cnx_smdp (свой тренер)")
        _kind = {"their_dqn": "their", "their_cql": "their", "stat_dqn": "stat"}.get(
            args.variant, ("convnext" if args.encoder == "auto" else args.encoder) if args.variant in CNX_VARIANTS
            else "cnn")
        try:
            ft_check(args.ft_mode, _kind, make_encoder(_kind, 4))
            if args.ft_mode == "lora":
                lora_wrap(make_encoder(_kind, 4), max(1, args.lora_rank), 1.0)
        except ValueError as e:
            ap.error(str(e))
        if args.ft_mode == "lora" and (args.lora_rank < 1 or not args.lora_alpha > 0):
            ap.error("--lora-rank >= 1 и --lora-alpha > 0")
    elif args.lora_rank != 4 or args.lora_alpha != 8.0:
        ap.error("--lora-rank и --lora-alpha задаются вместе с --ft-mode lora")
    if args.ft_mode in ("bn", "lastblock") and (args.lora_rank != 4 or args.lora_alpha != 8.0):
        ap.error("--lora-rank и --lora-alpha задаются вместе с --ft-mode lora")
    new_loss = (args.mb_aux > 0 or args.pani_w > 0 or args.distill_w > 0 or args.distill_top > 0
                or args.mask_schedule or args.teacher_w > 0)
    if args.ens < 1 or (args.ens > 1 and not 1 <= args.ens_min <= args.ens):
        ap.error("--ens >= 1 и 1 <= --ens-min <= --ens")
    if args.hlg:
        if args.variant not in HLG_VARIANTS:
            ap.error(f"--hlg поддержан для {', '.join(HLG_VARIANTS)} (обычная линейная голова)")
        if args.ens > 1 or args.distill_from:
            ap.error("--hlg несовместим с --ens > 1 и --distill-from")
        if args.hlg_bins < 2:
            ap.error("--hlg-bins >= 2")
        if args.hlg_vmin is not None and args.hlg_vmax is not None and args.hlg_vmax <= args.hlg_vmin:
            ap.error("--hlg-vmax должен быть больше --hlg-vmin")
    if args.two_nets:
        if args.variant not in TWO_NETS_VARIANTS:
            ap.error(f"--two-nets поддержан для {', '.join(TWO_NETS_VARIANTS)}: у квантильных вариантов, "
                     "cnx_dueling, cnx_factored, cnx_bcq, cnx_bbf и cnx_smdp своя голова, свой лосс или свой тренер")
        if args.ens > 1:
            ap.error("--two-nets несовместим с --ens > 1: ансамбль голов на общем энкодере, а сетей две")
        if args.hlg:
            ap.error("--two-nets несовместим с --hlg: совместное Q складывается из скаляров двух сетей, "
                     "а голова HL-Gauss отдаёт распределение по бинам")
        if args.aux_next > 0:
            ap.error("--two-nets несовместим с --aux-next: вспомогательная голова стоит на одном энкодере")
        if args.two_nets_shared_grad != "both" and not args.two_nets_shared:
            ap.error("--two-nets-shared-grad задаётся вместе с --two-nets-shared (у сети параметров --two-nets "
                     "без него свой энкодер)")
    elif (args.two_nets_center != "max" or args.two_nets_head != "linear" or args.two_nets_mode != "additive"
          or args.two_nets_shared or args.two_nets_shared_grad != "both"):
        ap.error("--two-nets-center, --two-nets-head, --two-nets-mode, --two-nets-shared и "
                 "--two-nets-shared-grad задаются вместе с --two-nets")
    if args.aux_next < 0:
        ap.error("--aux-next >= 0")
    if args.aux_next > 0:
        if args.variant in ("cnx_smdp", "cnx_bbf") or args.distill_from:
            ap.error("--aux-next несовместим с cnx_smdp, cnx_bbf (он заново создаёт сеть и оптимизатор) "
                     "и --distill-from")
        if args.freeze_encoder:
            ap.error("--aux-next несовместим с --freeze-encoder: цель учит признаки энкодера")
    if args.mb_beam < 1 or args.mb_max_rows < 0:
        ap.error("--mb-beam >= 1, --mb-max-rows >= 0")
    if (args.mb_full or args.mb_max_rows) and not args.mb_aux > 0:
        ap.error("--mb-full и --mb-max-rows задаются вместе с --mb-aux > 0")
    args._mb_hs = None
    if args.mb_hs:
        try:
            args._mb_hs = sorted(set(int(x) for x in args.mb_hs.split(",") if x.strip()))
        except ValueError:
            ap.error(f"--mb-hs: целые горизонты через запятую, не {args.mb_hs!r}")
        if not args._mb_hs or args._mb_hs[0] < 1:
            ap.error("--mb-hs: горизонты >= 1")
        if not args.mb_aux > 0:
            ap.error("--mb-hs задаётся вместе с --mb-aux > 0")
        if any(a == "--mb-h" or a.startswith("--mb-h=") for a in sys.argv[1:]) and args.mb_h != args._mb_hs[-1]:
            ap.error("--mb-h и --mb-hs вместе: глубина --mb-h должна равняться наибольшему горизонту "
                     "(лучше задать только --mb-hs)")
        args.mb_h = args._mb_hs[-1]
    if args.mb_trust != "none":
        if not args.mb_aux > 0:
            ap.error("--mb-trust задаётся вместе с --mb-aux > 0")
        if args.subdirs:
            ap.error("--mb-trust: только один УрЧП (--subdir), доверие считается на записанных переходах задачи-цели")
        if not args.mb_trust_tau > 0 or not 0.5 < args.mb_trust_gate_min <= 1.0 or args.mb_trust_folds < 0:
            ap.error("--mb-trust-tau > 0, 0.5 < --mb-trust-gate-min <= 1, --mb-trust-folds >= 0")
    # составляющие метода (волна 50)
    if args.wa != "none":
        if not 0.0 <= args.wa_start < 1.0 or not 0.0 < args.wa_decay < 1.0:
            ap.error("--wa-start в [0, 1), --wa-decay в (0, 1)")
        if args.fqe_select or args.variant in ("cnx_smdp", "cnx_bbf") or args.distill_from:
            ap.error("--wa несовместим с --fqe-select (свой выбор снимка), cnx_smdp, cnx_bbf (сеть пересоздаётся) "
                     "и --distill-from")
    if args.train_all and (args.fqe_select or args.plateau_patience):
        ap.error("--train-all несовместим с --fqe-select и --plateau-patience: отложенных цепочек больше нет")
    if args.mc_lb < 0:
        ap.error("--mc-lb >= 0")
    if args.mc_lb > 0:
        if not args.chain_fix or args.distill_from or args.variant in ("cnx_smdp", "cnn_iql", "cnn_vqc"):
            ap.error("--mc-lb требует --chain-fix (границы цепочек) и обычный тренер Q (не cnx_smdp, cnn_iql, "
                     "cnn_vqc, --distill-from)")
        if args.mc_lb_mode == "target" and (args.variant not in MC_LB_TARGET_VARIANTS or args.ens > 1 or args.hlg):
            ap.error("--mc-lb-mode target меняет цель TD только у обычной головы DQN (без --ens > 1 и --hlg); "
                     "иначе --mc-lb-mode loss")
    # волна 58: виды карт и план шага
    if args.view_aug:
        if not args.target_buffer:
            ap.error("--view-aug требует --target-buffer: виды сопоставляются с файлами собранного буфера (сид и "
                     "цепочка из файла)")
        if not 0.0 < args.view_aug_p <= 1.0 or args.view_cons < 0:
            ap.error("--view-aug-p в (0, 1], --view-cons >= 0")
        if args.state_mode != "full":
            ap.error("--view-aug: только --state-mode full (виды — сырые карты среды)")
        if args.variant in ("cnx_smdp", "cnn_vqc") or args.distill_from:
            ap.error("--view-aug несовместим с cnx_smdp (свой тренер), cnn_vqc и --distill-from")
    elif args.view_aug_p != VIEW_AUG_P or args.view_cons != 0.0:
        ap.error("--view-aug-p и --view-cons задаются вместе с --view-aug")
    if args.lr_schedule != "const":
        if args.variant in ("cnx_smdp", "cnx_bbf"):
            ap.error("--lr-schedule несовместим с cnx_smdp (свой тренер) и cnx_bbf (оптимизатор пересоздаётся)")
        if not 0.0 <= args.lr_min < args.lr:
            ap.error("--lr-min в [0, --lr)")
    elif args.lr_min != 0.0:
        ap.error("--lr-min задаётся вместе с --lr-schedule cosine")
    teacher_paths = [x.strip() for x in args.teacher_from.split(",") if x.strip()]
    if len(teacher_paths) > 1 and (args.train_tol > 0 or args.train_kmax > 0):
        ap.error("ансамбль учителей (--teacher-from через запятую) несовместим с правилом остановки обучения")
    if args.variant == "cnx_smdp" and (new_arch or new_loss):
        ap.error("cnx_smdp — отдельный тренер (advanced_agents.SmdpAgent): --encoder, --encoder-init, "
                 "--freeze-encoder, --ens, --mb-aux, --pani-w, --distill-top, --mask-schedule, --teacher-w "
                 "не поддержаны")
    if args.encoder != "auto" and args.variant not in CNX_VARIANTS:
        ap.error(f"--encoder задаётся только у вариантов с ConvNeXt: {', '.join(CNX_VARIANTS)}")
    if args.ens > 1 and args.variant in NO_ENS_VARIANTS:
        ap.error(f"--ens не поддержан для {args.variant}: у cnn_iql своя V-голова, у cnn_vqc квантовая")
    if (args.encoder_init or args.freeze_encoder) and args.variant == "cnx_bbf":
        ap.error("--encoder-init и --freeze-encoder несовместимы с cnx_bbf (он заново создаёт сеть)")
    if args.variant == "cnn_iql" and args.mask_schedule:
        ap.error("--mask-schedule не поддержан для cnn_iql: в его цели нет max по действиям")
    if (args.distill_w > 0) != (args.distill_top > 0):
        ap.error("--distill-top и --distill-w задаются вместе")
    if args.distill_top > 100:
        ap.error("--distill-top — проценты цепочек, не больше 100")
    if args.distill_from and (args.ens > 1 or new_loss):
        ap.error("--distill-from (регрессия Q учителя) несовместим с --ens, --mb-aux, --pani-w, "
                 "--distill-top, --mask-schedule и --teacher-w")
    if (args.teacher_w > 0) != bool(args.teacher_from) or args.teacher_w < 0:
        ap.error("--teacher-from и --teacher-w > 0 задаются вместе")
    if args.teacher_w > 0 and (args.pbrs or args.extra_buffers):
        ap.error("--teacher-w несовместим с --pbrs (цель учителя считается в исходной награде) и с "
                 "--extra-buffers (у онлайновых цепочек своя среда и бюджет)")
    if args.target_chains < 0:
        ap.error("--target-chains >= 0")
    if args.target_chains > 0:
        if args.subdirs:
            ap.error("--target-chains несовместим с --subdirs: цель — один буфер (--subdir), а в смеси УрЧП "
                     "задачи-цели нет")
        if not args.chain_fix:
            ap.error("--target-chains требует --chain-fix: без границ цепочек выбирать нечего")
        if args.extra_buffers:
            ap.error("--target-chains несовместим с --extra-buffers: онлайновых цепочек нет в таблице "
                     "суррогата, размер буфера перестал бы быть N")
        if args.per_pde_files:
            ap.error("--target-chains несовместим с --per-pde-files: буфер урезается по цепочкам, а не по файлам")
    if args.target_buffer:
        bad = [f for f, v in (("--subdirs", args.subdirs), ("--extra-buffers", args.extra_buffers),
                              ("--target-chains", args.target_chains > 0), ("--per-pde-files", args.per_pde_files),
                              ("--data-dir", args.data_dir)) if v]
        if bad:
            ap.error(f"--target-buffer несовместим с {', '.join(bad)}: данные задачи — только собранные файлы")
        if not args.chain_fix:
            ap.error("--target-buffer требует --chain-fix: файлы сбора — цепочки, без границ их не прочитать")
    if args.mb_aux > 0:
        if args.mb_h < 1 or args.mb_lam < 0 or args.mb_members < 1:
            ap.error("--mb-h >= 1, --mb-lam >= 0, --mb-members >= 1")
        if args.pbrs:
            ap.error("--mb-aux несовместим с --pbrs: модельная цель считается в исходной награде")
        if args.extra_buffers:
            ap.error("--mb-aux несовместим с --extra-buffers: у онлайновых цепочек своя среда и бюджет")

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
    if args.state_mode in ("loss", "tele"):
        ap.error(f"--state-mode {args.state_mode} офлайн недоступен: в буферах нет обучающего лосса")
    if args.state_mode == "tasknorm" and not args.chain_fix:
        ap.error("--state-mode tasknorm требует --chain-fix (нужна принадлежность состояний задаче)")
    if args.reward_form != "logged" and not args.chain_fix:
        ap.error("--reward-form требует --chain-fix")
    if (args.init_err is not None or args.episode_budget) and not args.chain_fix:
        ap.error("--init-err и --episode-budget требуют --chain-fix")
    if args.train_tol < 0 or args.train_kmax < 0:
        ap.error("--train-tol >= 0, --train-kmax >= 0")
    stop_on = args.train_tol > 0 or args.train_kmax > 0
    if stop_on and not args.chain_fix:
        ap.error("--train-tol и --train-kmax требуют --chain-fix: без границ цепочек правило остановки не применить")
    if stop_on and args.reward_form not in ("delta", "dlog"):
        ap.error("--train-tol и --train-kmax требуют --reward-form delta или dlog: награда-уровень (logged, level) "
                 "и награда модели авторов (model) не складываются в ценность до остановки")
    if stop_on and args.distill_from:
        ap.error("--train-tol и --train-kmax несовместимы с --distill-from: Q учителя на картах правило не обрезает")
    if args.value_bound and (not args.chain_fix or args.reward_form != "delta" or args.pbrs):
        ap.error("--value-bound требует --chain-fix --reward-form delta и несовместим с --pbrs")
    if args.value_bound and multi and args.err_norm != "init":
        ap.error("--value-bound на смеси УрЧП требует --err-norm init: граница выведена в единицах ошибки")
    if args.mb_aux > 0 and (not args.chain_fix or args.reward_form not in ("delta", "dlog")):
        ap.error("--mb-aux требует --chain-fix и --reward-form delta или dlog: суррогат предсказывает "
                 "ошибку после действия, а сумма наград-уровней (logged, level) за горизонт --mb-h "
                 "не равна ценности")
    if args.distill_top > 0 and not args.chain_fix:
        ap.error("--distill-top требует --chain-fix (нужны границы цепочек и ошибка в их конце)")
    if args.teacher_w > 0 and (not args.chain_fix or args.reward_form not in ("delta", "dlog")
                               or (not multi and args.init_err is None)):
        ap.error("--teacher-w требует --chain-fix, --reward-form delta или dlog и --init-err: Q учителя — "
                 "убыль log10 ошибки от текущей до конца эпизода, считая первый шаг от ошибки "
                 "необученной сети; в награды logged, level и model она не переводится")
    if multi:
        data = load_multi(multi, reward_form=args.reward_form, err_norm=args.err_norm,
                          budget=args.episode_budget, max_files=args.per_pde_files,
                          data_dir=args.data_dir)
        episodes = list(range(int(data["EP"].max()) + 1))
        # имя без числа задач: путь к чекпоинту не должен зависеть от того, сколько буферов
        # фактически загрузилось (очередь ссылается на него заранее)
        args.subdir = "multi"
    else:
        # --per-pde-files действует и на один УрЧП, но режет по файлам и только данные ученика;
        # для кривой «сколько данных буфера нужно» — --target-chains (ниже)
        tb_info = {}
        if args.target_buffer:
            # исходный буфер задачи не читается: только файлы collect_active.py --save-buffer
            import target_buffer as TB
            episodes, tb_info = TB.load(args.target_buffer, args.subdir,
                                        log=lambda *x: print(*x, flush=True))
        else:
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
    tc_info, tc_tab = {}, None
    if args.target_buffer:
        # сколько собранных цепочек и переходов досталось ученику (после обрезки по --episode-budget)
        tb_info = dict(tb_info, used_chains=int(len(np.unique(data["EP"]))), used_transitions=int(len(data["A"])))
        print(f"собранный буфер цели у ученика: цепочек {tb_info['used_chains']}, переходов "
              f"{tb_info['used_transitions']}; исходный буфер {args.subdir} не используется", flush=True)
        if args.mb_aux > 0:
            # таблица суррогата — как у учителя agent_combo.py --target-buffer: строки цели только собранные
            tc_tab = TB.surrogate_table(episodes, subdir_to_pde(args.subdir), args.mb_subdirs, args.mb_table,
                                        args.mb_work, log=lambda *x: print(*x, flush=True))
    if args.target_chains > 0:
        # до всего остального: цели учителя, контекст, модельные цели и отложенная часть считаются
        # уже на оставленных цепочках
        import mb_targets as MB
        if args.mb_table:
            print("ВНИМАНИЕ: --target-chains с --mb-table: цепочки совпадут с цепочками учителя, только если "
                  "agent_combo.py получил ту же таблицу (--table)", flush=True)
        data, tc_tab, tc_info = MB.target_subset(
            data, subdir_to_pde(args.subdir), args.target_chains, args.data_seed,
            budget=int(args.episode_budget), subdirs=args.mb_subdirs, table=args.mb_table, work=args.mb_work)
    stop_info = {}
    if stop_on:
        # правило остановки эпизода обучения: после выбора цепочек (--target-chains сопоставляет цепочки
        # тренера и таблицы по полному содержимому), до контекста, отложенной части и плотных целей
        data, stop_info = stop_rule_cut(data, args.train_tol, args.train_kmax, gamma=GAMMA,
                                        log=lambda *x: print(*x, flush=True))
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
    if args.train_all:
        # --train-all: учатся все переходы; прежние отложенные цепочки остаются только для диагностик
        # (evaluate, FQE), теперь они внутри обучения
        train_mask = np.ones_like(train_mask)
        print(f"--train-all: учатся все {int(train_mask.sum())} переходов; диагностики на {int(test_mask.sum())} "
              "переходах прежних отложенных цепочек (внутри обучения)", flush=True)
    # новые флаги — в meta чекпоинта и в строку результата
    arm = dict(encoder_init=os.path.basename(args.encoder_init), freeze_encoder=bool(args.freeze_encoder),
               ens_min=int(args.ens_min), mb_aux=float(args.mb_aux), mb_h=int(args.mb_h),
               mb_lam=float(args.mb_lam), mb_subdirs=args.mb_subdirs, mb_members=int(args.mb_members),
               mb_table=os.path.basename(args.mb_table), pani_w=float(args.pani_w),
               pani_sigma=float(args.pani_sigma), distill_top=float(args.distill_top),
               distill_w=float(args.distill_w), mask_schedule=bool(args.mask_schedule),
               mask_coarse_lr=int(args.mask_coarse_lr), teacher_w=float(args.teacher_w),
               teacher_from=",".join(os.path.basename(x) for x in args.teacher_from.split(",")),
               target_chains=int(args.target_chains), data_seed=int(args.data_seed))
    # новые приёмы пишутся, только когда включены: строки и meta прежних рецептов не меняются
    if args.hlg:
        arm.update(hlg=True, hlg_aux=args.hlg_aux)
    if args.aux_next > 0:
        arm.update(aux_next=float(args.aux_next))
    if args.mb_full:
        arm.update(mb_full=True, mb_beam=int(args.mb_beam))
    if args.mb_max_rows:
        arm.update(mb_max_rows=int(args.mb_max_rows))
    if args._mb_hs:
        # несколько горизонтов: mb_h в строке — наибольший из них
        arm.update(mb_hs=",".join(str(x) for x in args._mb_hs))
    if args.wa != "none":
        arm.update(wa=args.wa, wa_start=float(args.wa_start), wa_decay=float(args.wa_decay))
    if args.train_all:
        arm.update(train_all=True, diag_in_sample=True)
    if args.mc_lb > 0:
        arm.update(mc_lb=float(args.mc_lb), mc_lb_mode=args.mc_lb_mode)
    if len(teacher_paths) > 1:
        arm.update(n_teachers=len(teacher_paths))
    if args.mb_trust != "none":
        arm.update(mb_trust=args.mb_trust, mb_trust_tau=float(args.mb_trust_tau),
                   mb_trust_gate_min=float(args.mb_trust_gate_min), mb_trust_folds=int(args.mb_trust_folds))
    if stop_on:
        # правило остановки обучения: в строке результата и в meta чекпоинта, только когда включено
        arm.update(train_tol=float(args.train_tol), train_kmax=int(args.train_kmax),
                   stop_rule={k: v for k, v in stop_info.items() if not k.startswith("stop_")})
    if args.two_nets:
        # по полю two_nets в meta чекпоинта offline_rl.ckpt_arch строит обе сети
        arm.update(two_nets=True, two_nets_center=args.two_nets_center, two_nets_head=args.two_nets_head,
                   two_nets_mode=args.two_nets_mode, two_nets_shared=bool(args.two_nets_shared))
        if args.two_nets_shared:
            arm.update(two_nets_shared_grad=args.two_nets_shared_grad)
    if args.view_aug:
        arm.update(view_aug=args.view_aug, view_aug_p=float(args.view_aug_p), view_cons=float(args.view_cons))
    if args.lr_schedule != "const":
        arm.update(lr_schedule=args.lr_schedule, lr_min=float(args.lr_min))
    if args.ft_mode != "full":
        # по полям ft_mode, lora_rank, lora_alpha в meta чекпоинта offline_rl.ckpt_arch строит адаптеры;
        # число обучаемых параметров энкодера — в строке сида (ft_enc_trainable, ft_enc_total)
        arm.update(ft_mode=args.ft_mode)
        if args.ft_mode == "lora":
            arm.update(lora_rank=int(args.lora_rank), lora_alpha=float(args.lora_alpha))
    # суррогат модельных целей не должен видеть отложенную задачу (--holdout): её папки убираются
    # из --mb-subdirs. Список — в том же порядке, что у agent_combo.py --extra-only-target без
    # --extra-data (учитель без строк цели), поэтому таблица и ансамбль в кэше --mb-work общие
    mb_subdirs = args.mb_subdirs
    if multi and args.holdout and args.mb_aux > 0:
        if args.mb_table:
            print("ВНИМАНИЕ: --mb-table с --holdout: готовая таблица берётся как есть, отложенная задача "
                  "из неё НЕ убирается", flush=True)
        else:
            mb_subdirs = ",".join(resolve_subdirs(args.mb_subdirs, args.holdout))
            arm.update(mb_holdout=args.holdout)
            print(f"модельные цели: суррогат без отложенных задач ({args.holdout}): {mb_subdirs}", flush=True)
    run_meta.update(arm)
    tb_row = {}
    if args.target_buffer:
        run_meta["target_buffer"] = dict(tb_info)       # источники, число цепочек и переходов
        tb_row = {"tb_" + k: v for k, v in tb_info.items()}
    if tc_info:
        run_meta["target_data"] = dict(tc_info)         # сколько цепочек цели осталось и где
    mb_info, te_info = {}, {}
    if args.mb_aux > 0 or args.teacher_w > 0:
        # модельные цели Q_mb(s, .) для обучающих переходов и цели учителя: один расчёт на все сиды
        names = list(data["PDE_NAMES"]) if multi else [subdir_to_pde(args.subdir)]
        e0 = np.array([float(pde_meta()[n]["init_err"]) for n in names])
        # единицы награды — те же, что дал загрузчик: в смеси с --err-norm init ошибки поделены на
        # E0 задачи и первый шаг считается от 1.0 (load_multi); у одного УрЧП ошибки как есть, а
        # первый шаг — от --init-err (не задан: награда первого шага 0)
        if multi and args.err_norm == "init":
            mb_scale, mb_init = e0, np.ones(len(names))
        elif multi:
            mb_scale, mb_init = np.ones(len(names)), e0
        else:
            mb_scale = np.ones(1)
            mb_init = None if args.init_err is None else np.array([float(args.init_err)])
    # правило остановки: суррогат нужен и целям учителя (их ценность считается продолжением политики
    # учителя в суррогате до остановки, см. teacher_targets) — один и тот же ансамбль на обе добавки
    stop_kw = dict(tol=float(args.train_tol), kmax=int(args.train_kmax)) if stop_on else {}
    surr = {}
    if args.mb_aux > 0:
        import mb_targets as MB
        data["QMB"], data["MBW"], mb_info = MB.precompute(
            data, train_mask, names=names, reward_form=args.reward_form,
            err_scale=mb_scale, init_err=mb_init,
            gamma=float(GAMMA), h=args.mb_h, lam=args.mb_lam, budget=int(args.episode_budget),
            smdp_scale=(float(args.smdp_scale) if args.smdp else 0.0),
            subdirs=mb_subdirs, table=args.mb_table, members=args.mb_members,
            work=args.mb_work, quick=bool(args.smoke), tab=tc_tab,
            full=bool(args.mb_full), beam=int(args.mb_beam), max_rows=int(args.mb_max_rows),
            rows_seed=int(args.data_seed), **stop_kw,
            **({"keep": surr} if (stop_on or args.mb_trust != "none") else {}),
            **({"hs": args._mb_hs} if args._mb_hs else {}))
        if args.mb_trust != "none":
            # множитель слагаемого модельной цели каждой строки (вес доверия, ворота): суррогаты фолдов
            # учатся на той же таблице без кластеров цепочек цели своего фолда (mb_targets.trust_weights)
            data["MBT"], tr_info = MB.trust_weights(
                data, data["MBW"], names, mb_scale, int(args.episode_budget), surr["model"], mode=args.mb_trust,
                tau=float(args.mb_trust_tau), gate_min=float(args.mb_trust_gate_min), folds=int(args.mb_trust_folds),
                seed=int(args.data_seed), members=int(args.mb_members), work=args.mb_work, quick=bool(args.smoke))
            mb_info.update(tr_info)
        run_meta["mb"] = dict(mb_info)
    args._teacher_soft_mask = []
    if args.teacher_w > 0:
        import torch as _t
        te_kw = {}
        if stop_on:
            if not surr:
                import mb_targets as MB
                _lg = lambda *x: print(*x, flush=True)  # noqa: E731
                if tc_tab is not None:
                    tab_s, ens_s, _ = MB.fit_model(tc_tab, args.mb_members, args.mb_work, bool(args.smoke), _lg)
                else:
                    tab_s, ens_s, _ = MB.build_model(names, mb_subdirs, args.mb_table, args.mb_members,
                                                     args.mb_work, bool(args.smoke), _lg)
                surr["model"] = (tab_s, ens_s)
            te_kw = dict(stop_kw, model=surr["model"], lam=float(args.mb_lam),
                         smdp_scale=(float(args.smdp_scale) if args.smdp else 0.0))
        _te_dev = _t.device("cuda" if _t.cuda.is_available() and not args.cpu else "cpu")
        if len(teacher_paths) == 1:
            data["QTE"], data["TEV"], te_info = teacher_targets(
                args.teacher_from, data, names=names, reward_form=args.reward_form, err_scale=mb_scale,
                init_err=mb_init, gamma=float(GAMMA), budget=int(args.episode_budget), device=_te_dev, **te_kw)
        else:
            data["QTE"], data["TEV"], te_info = teacher_ensemble_targets(
                teacher_paths, data, names=names, reward_form=args.reward_form, err_scale=mb_scale,
                init_err=mb_init, gamma=float(GAMMA), budget=int(args.episode_budget), device=_te_dev)
        # маску учителя ученик НЕ наследует (у него свой TD по всем действиям); она только в meta
        args._teacher_soft_mask = te_info.pop("mask")
        te_info["train_rows"] = int((data["TEV"] & train_mask).sum())
        run_meta["teacher"] = dict(te_info, mask=list(args._teacher_soft_mask))
    va_row = {}
    if args.view_aug:
        # банк видов — после всех срезов набора и контекста: альтернативные входы — те же строки с другими картами
        va_info = attach_views(data, episodes, args.view_aug, args.subdir, log=lambda *x: print(*x, flush=True))
        print(f"виды карт ({args.view_aug}): файлов {va_info['records']}, состояний с видами {va_info['states']}; "
              f"у переходов: s с видами {va_info['rows_s']} из {va_info['rows']} (в среднем {va_info['mean_k_s']} "
              f"видов), s' с видами {va_info['rows_s2']} (в среднем {va_info['mean_k_s2']}); ключ есть у "
              f"{va_info['rows_keyed']} переходов", flush=True)
        if not va_info["rows_s"] and not va_info["rows_s2"]:
            raise SystemExit("--view-aug: ни у одного перехода буфера нет видов (сид, цепочка и шаг не совпали ни с "
                             "одним файлом видов) — не тот тег видов или не тот буфер")
        run_meta["views"] = dict(va_info)
        va_row = {"view_" + k: v for k, v in va_info.items()}
    verify_bad = False

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
                       l2=args.l2, lr=float(args.lr), tau=float(args.tau), pbrs=args.pbrs, smdp=bool(args.smdp),
                       model_tag=args.model_tag, state_mode=args.state_mode,
                       scalar_ctx=bool(args.scalar_ctx), ctx_err=bool(args.ctx_err),
                       encoder=getattr(net, "enc_kind", "convnext"), ens=int(getattr(net, "ens", 1)), **arm,
                       **{"tc_" + k: v for k, v in tc_info.items()}, **tb_row,
                       **{"mb_" + k: v for k, v in mb_info.items()},
                       **{"teacher_" + k: v for k, v in te_info.items()}, **va_row,
                       **getattr(net, "extra", {}), **m, **fq)
            print(json.dumps(row), flush=True)
            if not args.smoke:
                tag = run_suffix(args)
                upload_result(row, f"{args.variant}{tag}_{pol}_seed{seed}")
        if args.save_model and not args.smoke:
            import torch as _t
            # архитектура — в meta: по ней load_agent строит ту же сеть (у старых чекпоинтов
            # этих полей нет, они читаются как раньше)
            ckpt = {"variant": args.variant, "seed": seed,
                    "state_dict": {k: v.cpu() for k, v in net.model.state_dict().items()},
                    "mean": stats["mean"], "std": stats["std"],
                    "meta": dict(run_meta, encoder=getattr(net, "enc_kind", "convnext"),
                                 ens=int(getattr(net, "ens", 1)),
                                 n_quantiles=int(getattr(net, "nq", 0) or 0),
                                 in_ch=int(data["S"].shape[1]))}
            if getattr(net, "hl_gauss", None):
                # по этому полю load_agent строит голову на бинах и сводит её к скаляру Q
                ckpt["hl_gauss"] = tuple(net.hl_gauss)
            sfx = run_suffix(args)
            fn = f"/tmp/agent_{args.variant}{sfx}_seed{seed}.pt"
            _t.save(ckpt, fn)
            verify = None
            if not args.no_verify and args.variant != "cnx_smdp":
                try:
                    verify = verify_ckpt(fn, net)
                except Exception as e:      # noqa: BLE001 — чекпоинт уже сохранён
                    verify = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
                print(f"проверка загрузки через online_eval_env: {verify}", flush=True)
                if not verify.get("ok"):
                    verify_bad = True
                    print("ПРОВЕРКА НЕ ПРОШЛА: чекпоинт в HF не заливается, код возврата 3 "
                          "(оценку не запускать)", flush=True)
            tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
            if tok and not (verify is not None and not verify.get("ok")):
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
    if verify_bad:
        sys.exit(3)         # следующий шаг задачи (оценка на GPU) не должен стартовать с таким чекпоинтом


if __name__ == "__main__":
    main()
