#!/usr/bin/env python
"""
Сбор обучающих цепочек для агента с четырьмя приёмами экономии GPU-часов. Каждая записанная
цепочка стоит одного обучения PINN, поэтому приёмы включаются флагами, а контрольный сбор —
тот же код с другими флагами.

  1. Направленный сбор (--acq lcb|greedy|random, --lcb-k). Раундами по --batch цепочек: суррогат
     (track3/surrogate.py, ансамбль из 5 HGB) учится на буферах остальных УрЧП, по --use-old-target
     ещё на старом буфере целевой задачи, и на всём, что уже собрано здесь; кандидаты — лучевой
     поиск surrogate.plan (несколько k), мутации и скрещивания лучших собранных цепочек
     (операторы ga_chains.py) и случайные цепочки; пачка берётся по J = среднее − k·разброс по
     членам ансамбля (lcb) или по среднему (greedy). random — равномерно случайные допустимые
     цепочки, суррогат не строится (контроль). Модель переучивается каждый раунд.
  2. Общий префикс (--share-prefix, --prefix-bonus, --share-min). Цепочки пачки с общим началом
     исполняются так: префикс один раз, чекпоинт весов локально, концовки от снимка — механизм
     branch_endings.py (докатка среды с чекпоинта, токены HF на время прогона спрятаны). Эпохи
     префикса входят в счёт один раз. --prefix-bonus (декады log10 ошибки за общий префикс
     длиной в весь бюджет) склоняет выбор к концовкам уже выбранного префикса при близких оценках.
  3. Ранняя остановка (--truncate C, --truncate-min, --truncate-min-ref). После каждого действия
     l2re цепочки сравнивается с лучшей l2re собранных цепочек при тех же потраченных эпохах
     (линейная интерполяция log10 l2re по их hist; за концом цепочки держится её последнее
     значение). Хуже в C раз после --truncate-min эпох (по умолчанию половина бюджета) — цепочка
     останавливается, её переходы остаются в данных, запись помечается truncated. Остановленный
     общий префикс отменяет все свои концовки (они перечислены в skipped записи).
  4. Низкая точность (--budget E, --fidelity-note, --no-clip). Бюджет любой; он пишется в каждую
     запись (env.budget), нормировка spent/budget — дело обучения агента. --no-clip не исполняет
     действие, которое бюджет обрезал бы (agent_combo.py такие действия при бюджете меньше 7000
     в таблицу не берёт, их эпохи пропали бы).

Как устроена остановка и почему не докатка по одному действию. Докатка среды с чекпоинта
начинает первое действие с чистым оптимизатором (opt_cache в чекпоинт не входит), поэтому
исполнение по одному действию молча превратило бы среду --keep-opt в среду со сбросом, а
каждая докатка ещё и заново строит модель. Здесь цепочка идёт одним вызовом run_seed; решение
принимает progress_cb (--progress-every 0: вызов после каждого действия), а остановка — штатный
мягкий выход среды по args.hours: чекпоинт (с полным hist) и строка unfinished. Среда не меняется.
По той же причине ветвление от префикса допускается только в точке, где среда и так создала бы
оптимизатор заново (reuse_optimizer: смена семейства, PSO, при --keep-opt-mode safe ещё Adam с
другим шагом); без --keep-opt — в любой.

Сиды: одна цепочка — один сид из --seed-pool (перестановка по --collect-seed); у цепочек общего
префикса сид один на группу (снимок весов общий).

Выход: <save-dir>/<tag>_collect.json, тот же словарь в HF: rl_arch/online_env/<pde>_<tag>_collect.json
(без токена не заливается). --resume продолжает с сохранённого состояния (локальный файл, иначе HF).

Формат (его читает agent_combo.py --extra-data):

    records: [ {
      chain:  [[opt, lr, epochs], ...]      исполненные действия, эпохи фактические (последнее
                                            действие среда обрезает по бюджету)
      hist:   [[spent, loss, l2re, t], ...] по строке на действие: эпох потрачено всего, обучающий
                                            лосс, l2re после действия, секунд от начала цепочки
                                            (loss и l2re — null, если не число)
      seed:   int
      env:    {keep_opt: bool, mode: "all"|"safe"|null, budget: int}
      truncated: bool                       остановлена приёмом 3 (chain и hist — исполненная часть)
      shared_prefix_len: int                сколько первых действий взято от общего снимка (0 — нет)
      round:  int
      acq:    "lcb"|"greedy"|"random"|"init"|"prefix"   чем выбрана: init — из --init, random — случайная
                                            (контроль или добор пачки), prefix — запись одного
                                            префикса (все его концовки упали)
      pred_mean, pred_std: float|null       прогноз суррогата для задуманной цепочки: log10(rmse+brmse)
                                            после последнего действия, среднее и разброс по членам
      -- дополнительно --
      err:    [rmse+brmse после действия]   единицы суррогата и буферов (null, если неизвестно)
      l2re, spent, planned (строка --script), actions (индексы задуманной цепочки), origin,
      group, prefix_owner, epochs_real, elapsed_s, stop, error, skipped
    } ]
    У записей с одним group первые shared_prefix_len действий — один и тот же физический прогон
    (тот же сид и те же веса): эти переходы считать один раз, у записи с prefix_owner=true.

    rounds: [ {round, acq, n_chains, n_truncated, n_shared, cum_chains, cum_epochs (обученные
               эпохи, общий префикс один раз), cum_epochs_planned (те же цепочки без приёмов 2 и 3),
               cum_gpu_s (секунды всех вызовов среды), best_l2re, fit_s (обучение суррогата в этом
               раунде), sel_s, cum_fit_s, n_candidates, n_train_rows, n_target_rows, pred_mae, wall_s} ]
    settings: флаги запуска; env: {keep_opt, mode, budget, lbfgs_tol, plain_fnn, hidden_layers};
    n_chains, n_runs, epochs_real, epochs_planned, gpu_s, fit_s, table_s, wall_s, best, done,
    partial, stop_reason; round, pending, sel, init_left, seed_cursor — для докатки.

Буфер с картами (--save-buffer). Среда запускается без --no-state: после каждого действия строятся
карты ландшафта теми же настройками автокодировщика и той же build_state, что на оценке
(online_eval_env.py; после действия, выбравшего бюджет, карта не строится — как там). Каждая
записанная цепочка пишется отдельным файлом .pt в формате исходных офлайновых буферов (список
переходов: state, next_state — словари карт loss_total, loss_oper, loss_bnd 26x26; action =
(оптимизатор, {lr, epochs}); reward = −(rmse+brmse) после действия; done), его без изменений читают
offline_rl.load_episodes + episodes_to_arrays_chains и surrogate.build_table:

    <save-dir>/buffers_new/<tag>/<папка буфера задачи>/<tag>_cNNNN.pt       (--buffer-dir меняет корень)
    HF: rl_arch/buffers_new/<tag>/<папка буфера задачи>/<tag>_cNNNN.pt

  * файл = одна цепочка с шага 0. Цепочка с общим префиксом записана целиком: переходы префикса
    (их карты уже посчитаны) скопированы в файл каждой концовки; у переходов одной группы общая
    метка group (тренеры кладут их в одну половину данных);
  * конец цепочки: бюджет выбран — done=1; ошибка не число — done=−1 и reward=0 (как пишет трейнер
    авторов); цепочка остановлена приёмом 3 или кончилась раньше бюджета (--no-clip, префикс) —
    done=0 и настоящая карта в next_state: переход не терминальный (хвост файла у загрузчика);
  * действие — задуманный номер из 27 (последнее, обрезанное бюджетом, — тоже), фактические эпохи —
    в поле epochs_real; у первого перехода поле meta (сид, среда, бюджет, запись сбора);
  * заливка: все новые файлы одним коммитом перед каждой заливкой состояния (граница раунда,
    остановка, не чаще --progress-every внутри раунда): оборванный кернел теряет не больше
    незалитого хвоста. В записи: buf_file, buf_n (переходов), buf_up (залит).
Обучение на таком буфере вместо исходного: agent_combo.py / offline_rl.py --target-buffer
rl_arch/buffers_new/<tag>[,...] (target_buffer.py).

Несколько сборщиков параллельно (--worker k/K, свой --tag у каждого): кандидаты делятся между
сборщиками по crc ключа цепочки (одна и та же цепочка двум сборщикам не достанется), сиды среды —
срез пула [k::K], --collect-seed по умолчанию k. Выходы независимы и складываются перечислением
префиксов в --target-buffer.

    python experiments/rl_arch/collect_active.py --pde poissonboltzmann2d --acq lcb --lcb-k 1 \\
        --share-prefix --truncate 3 --keep-opt --keep-opt-mode safe --rounds 8 --batch 5 \\
        --tag ca_lcb --hours 14 --kernel-hours 10.5 --resume
    python experiments/rl_arch/collect_active.py --pde burgers_1d --acq random --smoke --rounds 1 --batch 2 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
import zlib

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
_T3 = os.path.join(SCRIPT_DIR, "track3")
if _T3 not in sys.path:
    sys.path.append(_T3)
os.environ.setdefault("DDEBACKEND", "pytorch")

import online_eval_env as E  # noqa: E402
from online_eval_env import OUT_REPO, run_seed, upload  # noqa: E402
import branch_endings as B  # noqa: E402
import ga_chains as G  # noqa: E402
import surrogate as S  # noqa: E402

FORMAT = 1
EPOCHS = [int(t[2]) for t in E.ACTION_TABLE]
BUF_HF = "rl_arch/buffers_new"  # корень буферов с картами в датасете результатов HF
MAP_KEYS = ("loss_total", "loss_oper", "loss_bnd")
_BASE = {}                      # кэш процесса: базовая таблица переходов


# --------------------------------------------------------------------------------------------
# Цепочки: допустимая форма, ключ, префиксы
# --------------------------------------------------------------------------------------------

def eff(genome, budget, max_len, clip=True):
    """Действия, которые успевают начаться до исчерпания бюджета (не больше max_len), и их
    фактические эпохи: последнее среда обрезает. clip=False: цепочка кончается перед действием,
    которое не помещается целиком."""
    acts, eps, spent = [], [], 0
    for g in list(genome)[:max_len]:
        if spent >= budget:
            break
        ep = EPOCHS[int(g)]
        if spent + ep > budget:
            if not clip:
                break
            ep = budget - spent
        acts.append(int(g))
        eps.append(int(ep))
        spent += ep
    return acts, eps


def norm(genome, a):
    """Допустимая форма цепочки при флагах запуска: (действия, фактические эпохи)."""
    return eff(genome, a.budget, a.max_len, clip=not getattr(a, "no_clip", False))


def key_of(acts, eps):
    """Ключ физической цепочки: (оптимизатор и шаг, фактические эпохи) по действиям."""
    return tuple((int(a) // 3, int(e)) for a, e in zip(acts, eps))


def chain_str(acts):
    return ",".join(G.action_str(a) for a in acts)


def fresh_opt(pa, a, keep_opt, mode):
    """Начнёт ли среда действие a после pa с нового оптимизатора (reuse_optimizer). Только в
    такой точке докатка с чекпоинта не меняет среду."""
    if not keep_opt:
        return True
    return bool(S.consistent_ok(np.array([pa]), np.array([a]), "safe" if mode == "safe" else "all")[0])


def can_branch(chain, j, a):
    """Можно ли исполнить chain как «префикс chain[:j] от снимка + остаток»."""
    if j >= len(chain):
        return j == len(chain)          # цепочка сама и есть префикс: её итог — прогон префикса
    return fresh_opt(chain[j - 1], chain[j], a.keep_opt, a.keep_opt_mode)


def shared_epochs(x, y, a):
    """Эпохи самого длинного общего префикса x и y, от которого обе цепочки можно ветвить."""
    best = spent = 0
    for j in range(1, min(len(x), len(y)) + 1):
        if x[j - 1] != y[j - 1]:
            break
        spent += EPOCHS[x[j - 1]]
        if spent >= a.budget:
            break
        if can_branch(x, j, a) and can_branch(y, j, a):
            best = spent
    return best


def plan_groups(chains, a):
    """Разбиение пачки на группы с общим префиксом: [(длина префикса, [номера цепочек])], длина 0 —
    цепочка идёт одна. Жадно по сбережённым эпохам: префикс × (число цепочек − 1). Один уровень
    ветвления: префикс исполняется раз, концовки — от его снимка."""
    left, groups = set(range(len(chains))), []
    while a.share_prefix:
        best = None
        for i in sorted(left):
            c = chains[i]
            for j in range(1, len(c) + 1):
                ep = sum(EPOCHS[x] for x in c[:j])
                if ep >= a.budget:          # чекпоинт пишется только при spent < budget
                    break
                if ep < a.share_min:
                    continue
                mem = [m for m in sorted(left) if chains[m][:j] == c[:j] and can_branch(chains[m], j, a)]
                if len(mem) >= 2 and (best is None or ep * (len(mem) - 1) > best[0]):
                    best = (ep * (len(mem) - 1), j, mem)
        if best is None:
            break
        groups.append((best[1], best[2]))
        left -= set(best[2])
    groups += [(0, [i]) for i in sorted(left)]
    return sorted(groups, key=lambda g: min(g[1]))


def edit_dist(a, b):
    """Расстояние Левенштейна между цепочками."""
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1]


# --------------------------------------------------------------------------------------------
# Приём 3: опорная кривая и решение об остановке
# --------------------------------------------------------------------------------------------

def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def ref_curves(records):
    """Кривые log10 l2re от потраченных эпох у собранных цепочек."""
    out = []
    for r in records:
        pts = [(h[0], math.log10(h[2])) for h in (r.get("hist") or []) if h[2] is not None and h[2] > 0]
        if pts:
            out.append((np.array([p[0] for p in pts], dtype=float), np.array([p[1] for p in pts])))
    return out


def best_at(curves, spent):
    """Лучший log10 l2re при spent эпохах среди цепочек, у которых есть данные к этому моменту
    (за концом цепочки — её последнее значение), и число таких цепочек."""
    vals = [float(np.interp(spent, sp, lg)) for sp, lg in curves if sp[0] <= spent]
    return (min(vals) if vals else None), len(vals)


def make_watch(records, a):
    """Функция (spent, l2re) -> остановить ли цепочку; None, если приём 3 выключен."""
    if not a.truncate:
        return None
    curves = ref_curves(records)

    def watch(spent, l2re):
        l2re = _num(l2re)
        if spent < a.truncate_min or l2re is None or l2re <= 0:
            return False
        ref, n = best_at(curves, spent)
        return n >= a.truncate_min_ref and math.log10(l2re) > ref + math.log10(a.truncate)
    return watch


# --------------------------------------------------------------------------------------------
# Один вызов среды
# --------------------------------------------------------------------------------------------

def env_args(a, script, ckpt_every=0, resume=False):
    """Namespace для run_seed: флаги среды как у ga_chains.py; строка прогресса после каждого действия."""
    argv = ["--policy", "script", "--script", script, "--script-tail", "stop",
            "--pde", a.pde, "--budget", str(a.budget), "--save-dir", a.save_dir, "--tag", a.tag,
            "--hidden-layers", a.hidden_layers, "--keep-opt-mode", a.keep_opt_mode,
            "--ckpt-every", str(ckpt_every), "--ckpt-min-interval", "0", "--progress-every", "0"]
    # --save-buffer: карты строятся (умолчания автокодировщика — те же, что у оценки)
    argv += [] if getattr(a, "save_buffer", False) else ["--no-state"]
    argv += ["--resume"] if resume else []
    argv += ["--keep-opt"] if a.keep_opt else []
    argv += ["--plain-fnn"] if a.plain_fnn else []
    argv += ["--lbfgs-tol", str(a.lbfgs_tol)] if a.lbfgs_tol is not None else []
    return E.build_parser().parse_args(argv)


def _ckpt_local(name):
    return f"ckpt_{name}.pt"


def _rm(path):
    if path and os.path.exists(path):
        os.remove(path)


def _ckpt_read(name):
    """(chain, hist, spent) из локального чекпоинта среды или None."""
    if not name or not os.path.exists(_ckpt_local(name)):
        return None
    try:
        ck = E.load_ckpt(name)
        return ([list(c) for c in ck["chain"]], [list(h) for h in (ck.get("t3") or {}).get("hist", [])],
                int(ck["spent"]))
    except Exception as e:
        print(f"чекпоинт {name} не прочитан ({type(e).__name__}: {e})", flush=True)
        return None


def run_env(a, acts, seed, n_planned, watch=None, ckpt=None, ckpt_every=0, resume=False, base_trace=(),
            base_maps=None):
    """Один вызов run_seed со скриптом acts. watch решает об остановке после каждого действия,
    пока исполнено меньше n_planned действий. Возвращает dict: ok, error, truncated, chain, hist,
    err, spent, l2re, stop, trace, elapsed_s, maps. maps (--save-buffer): {номер действия с 1:
    карты (3, 26, 26) после него}; base_maps — карты общего префикса (докатка их не строит заново)."""
    args = env_args(a, chain_str(acts), ckpt_every=ckpt_every, resume=resume)
    args._ckpt_name = ckpt
    trace = [dict(p) for p in base_trace]
    maps = dict(base_maps or {})
    if getattr(a, "save_buffer", False):
        # среда отдаёт состояние build_state (4, 26, 26); канал delta загрузчик буфера строит сам
        args._map_sink = lambda step, m: maps.__setitem__(int(step), np.array(m, dtype=np.float32)[:3])

    def cb(info):
        e = _num(info.get("rmse", float("nan")) + info.get("brmse", float("nan")))
        trace.append(dict(spent=int(info["spent"]), l2re=_num(info["l2re"]), err=e,
                          t=float(info.get("elapsed_s", 0.0))))
        if watch is not None and int(info["n_steps"]) < n_planned and watch(info["spent"], info["l2re"]):
            args.hours = 1e-9       # штатный мягкий выход среды: чекпоинт и строка unfinished

    saved, t, row, error = B._hide_tokens(), time.time(), None, None
    try:
        row = run_seed(seed, args, progress_cb=cb)
    except Exception as e:
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"[:200]
    finally:
        B._restore_tokens(saved)
    elapsed = round(time.time() - t, 1)

    emap = {p["spent"]: p["err"] for p in trace}
    truncated = bool(row is not None and row.get("unfinished"))
    if row is not None and not truncated:
        chain, hist, spent = row.get("chain") or [], row.get("hist") or [], int(row.get("spent", 0))
        emap[spent] = _num(row.get("rmse", float("nan")) + row.get("brmse", float("nan")))
        l2re, stop = _num(row.get("l2re")), row.get("stop_reason")
        if l2re is None:
            stop = "nonfinite"
    else:
        ck = _ckpt_read(ckpt) if truncated else None
        if ck is not None:
            chain, hist, spent = ck
        else:                       # падение прогона или чекпоинта нет: что видел progress_cb
            _, eps = eff(acts, a.budget, len(acts))
            n = min(len(trace), len(eps))
            chain = [[E.ACTION_TABLE[acts[i]][0], E.ACTION_TABLE[acts[i]][1], eps[i]] for i in range(n)]
            hist = [[p["spent"], None, p["l2re"], p["t"]] for p in trace[:n]]
            spent = trace[n - 1]["spent"] if n else 0
        l2re = _num(hist[-1][2]) if hist else None
        stop = "truncated" if truncated else "error"
    hist = [[int(h[0]), _num(h[1]), _num(h[2]), round(float(h[3]), 1)] for h in hist]
    return dict(ok=error is None, error=error, truncated=truncated,
                chain=[[c[0], float(c[1]), int(c[2])] for c in chain], hist=hist,
                err=[emap.get(h[0]) for h in hist], spent=int(spent), l2re=l2re, stop=stop,
                trace=trace, elapsed_s=elapsed, maps=maps)


# --------------------------------------------------------------------------------------------
# Буфер с картами (--save-buffer)
# --------------------------------------------------------------------------------------------

def buffer_subdir(pde):
    """Папка буфера задачи (как у исходных офлайновых буферов: pde_meta.json, поле subdir)."""
    import offline_rl as O
    return (O.pde_meta().get(pde) or {}).get("subdir") or pde


def buffer_dir(a):
    return os.path.join(a.buffer_dir or os.path.join(a.save_dir, "buffers_new", a.tag), buffer_subdir(a.pde))


def chain_transitions(res, acts, budget):
    """Исполненная цепочка -> переходы в формате офлайнового буфера (описание — в докстринге
    модуля). res — итог run_env (chain, hist, err, maps), acts — задуманные номера действий.
    Цепочка кончается на первом переходе без числовой ошибки и на первом действии, перед которым
    нет карты (упавший прогон)."""
    chain, hist, maps = res["chain"], res["hist"], res.get("maps") or {}
    n = min(len(chain), len(hist))
    errs = list(res.get("err") or []) + [None] * n
    zero = {k: np.zeros((26, 26), np.float32) for k in MAP_KEYS}
    out, has_next, bad = [], False, False
    for i in range(n):
        if i and i not in maps:
            break
        x = acts[i] if acts is not None and i < len(acts) else None
        if x is None or E.ACTION_TABLE[x][0] != chain[i][0] or abs(E.ACTION_TABLE[x][1] - float(chain[i][1])) > 1e-12:
            x = action_index(*chain[i])
        if x is None:
            break
        cur = {k: maps[i][j].copy() for j, k in enumerate(MAP_KEYS)} if i else {k: v.copy() for k, v in zero.items()}
        has_next = (i + 1) in maps
        nxt = {k: (maps[i + 1][j] if has_next else cur[k]).copy() for j, k in enumerate(MAP_KEYS)}
        e = _num(errs[i])
        bad = e is None or e <= 0
        out.append(dict(state=cur, next_state=nxt, action=(x // 9, {"lr": (x % 9) // 3, "epochs": x % 3}),
                        reward=(0.0 if bad else -float(e)), done=0, epochs_real=int(chain[i][2]),
                        spent=int(hist[i][0]), l2re=_num(hist[i][2])))
        if bad:
            break
    if out:
        out[-1]["done"] = -1 if bad else (1 if out[-1]["spent"] >= budget or not has_next else 0)
    return out


def write_buffer(st, a, rec, res):
    """Файл буфера записи rec (последняя в st['records']); в запись — buf_file, buf_n, buf_up."""
    import torch
    idx = len(st["records"]) - 1
    if rec.get("error") and not rec["prefix_owner"] and len(rec["hist"]) <= rec["shared_prefix_len"]:
        return                          # упавшая концовка без новых действий: это был бы повтор префикса
    tr = chain_transitions(res, rec.get("actions"), int(a.budget))
    if not tr:
        return
    grp = None if rec.get("group") is None else f"{a.tag}#g{rec['group']}"
    for t in tr:
        t["chain"], t["group"] = f"{a.tag}#{idx}", grp
    tr[0]["meta"] = dict(format=FORMAT, pde=a.pde, tag=a.tag, record=idx, seed=rec["seed"], env=rec["env"],
                         truncated=rec["truncated"], shared_prefix_len=rec["shared_prefix_len"],
                         stop=rec["stop"], round=rec["round"], acq=rec["acq"], l2re=rec["l2re"])
    d = buffer_dir(a)
    os.makedirs(d, exist_ok=True)
    fname = f"{a.tag}_c{idx:04d}.pt"
    torch.save(tr, os.path.join(d, fname))
    rec.update(buf_file=fname, buf_n=len(tr), buf_up=False)
    buf = st.setdefault("buffer", dict(dir=d, hf=f"{BUF_HF}/{a.tag}/{buffer_subdir(a.pde)}", files=0,
                                       transitions=0, uploaded=0, lost=0))
    buf["dir"] = d
    buf["files"] += 1
    buf["transitions"] += len(tr)


def upload_files(pairs, msg, tries=9):
    """Несколько файлов одним коммитом в датасет результатов HF: pairs — [(локальный путь, путь в
    репозитории)]. Токен и повторы — как у online_eval_env.upload (лимит HF — 128 коммитов в час
    на репозиторий, поэтому один коммит на заливку, а не на файл)."""
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF token — буфер только локально", flush=True)
        return False
    from huggingface_hub import CommitOperationAdd, HfApi
    for attempt in range(tries):
        try:
            HfApi().create_commit(repo_id=OUT_REPO, repo_type="dataset", token=tok, commit_message=msg,
                                  operations=[CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
                                              for src, dst in pairs])
            print(f"uploaded {len(pairs)} файлов буфера: {os.path.dirname(pairs[0][1])}", flush=True)
            return True
        except Exception as e:
            print(f"buffer upload retry {attempt}: {str(e)[:120]}", flush=True)
            time.sleep(min(600, 20 * 2 ** attempt))
    return False


def flush_buffer(st, a):
    """Залить ещё не залитые файлы буфера одним коммитом; отметить записи (buf_up)."""
    d, sub = buffer_dir(a), buffer_subdir(a.pde)
    todo = [r for r in st["records"] if r.get("buf_file") and not r.get("buf_up") and not r.get("buf_lost")]
    lost = [r for r in todo if not os.path.exists(os.path.join(d, r["buf_file"]))]
    for r in lost:                      # кернел оборвался до заливки, локального файла уже нет
        r["buf_lost"] = True
    if lost:
        st["buffer"]["lost"] = st["buffer"].get("lost", 0) + len(lost)
        print(f"буфер: карты {len(lost)} цепочек потеряны (файлы не были залиты до обрыва кернела); "
              "их действия и ошибки остались в записях", flush=True)
    todo = [r for r in todo if not r.get("buf_lost")]
    if not todo:
        return 0
    pairs = [(os.path.join(d, r["buf_file"]), f"{BUF_HF}/{a.tag}/{sub}/{r['buf_file']}") for r in todo]
    if not upload_files(pairs, f"rl_arch buffers_new {a.tag}: {len(pairs)} цепочек, раунд {st['round']}"):
        return 0
    for r in todo:
        r["buf_up"] = True
    st["buffer"]["uploaded"] = sum(bool(r.get("buf_up")) for r in st["records"])
    return len(todo)


# --------------------------------------------------------------------------------------------
# Приём 1: таблица переходов, суррогат, кандидаты, выбор пачки
# --------------------------------------------------------------------------------------------

def base_table(a):
    """Таблица переходов буферов (surrogate.build_table с кэшем .npz) и описатель целевой задачи.
    Возвращает dict(tab, ctx, k_l2re, table_s). Буферы берутся из RL_BUFFER_DIR, иначе
    скачиваются из HF (snapshot_download), как это делает загрузчик offline_rl."""
    key = (a.pde, a.table, a.subdirs)
    if _BASE.get("key") == key:
        return _BASE
    import offline_rl as O
    meta = O.pde_meta()
    if a.pde not in meta:
        sys.exit(f"задачи {a.pde!r} нет в track3/pde_meta.json: суррогату нужен её описатель")
    t = time.time()
    path = a.table or os.path.join(a.save_dir, "surrogate_table.npz")
    if not os.path.exists(path) and a.table_hf:
        try:
            from huggingface_hub import hf_hub_download
            shutil.copyfile(hf_hub_download(OUT_REPO, a.table_hf, repo_type="dataset"), path)
            print(f"таблица переходов взята из HF: {a.table_hf}", flush=True)
        except Exception as e:
            print(f"таблицы {a.table_hf} в HF нет ({type(e).__name__}): собирается из буферов", flush=True)
    if not os.path.exists(path):
        subs = [s for s in a.subdirs.split(",") if s.strip()] or list(O.ALL_SUBDIRS)
        root = os.environ.get("RL_BUFFER_DIR")
        if not root:
            try:
                from huggingface_hub import snapshot_download
                root = snapshot_download(O.REPO, repo_type="dataset",
                                         allow_patterns=[f"{s}/*.pt" for s in subs])
            except Exception as e:
                print(f"буферы не скачаны ({type(e).__name__}: {e}): пробуется локальный кэш", flush=True)
                root = S.buffer_root()
        tab = S.build_table(subdirs=subs, root=root)
        S.save_table(tab, path)
    tab = S.load_table(path)
    m = meta[a.pde]
    k = float(m["init_l2re"]) / float(m["init_err"]) if m.get("init_l2re") and m.get("init_err") else 1.0
    _BASE.clear()
    _BASE.update(key=key, tab=tab, ctx=S.pde_ctx(m), k_l2re=k, table_s=round(time.time() - t, 1))
    print(f"таблица переходов: УрЧП {len(tab['NAMES'])}, переходов {len(tab['A'])}, "
          f"{_BASE['table_s']:.0f} с ({path})", flush=True)
    return _BASE


def action_index(opt, lr, ep):
    """Индекс действия из 27 по оптимизатору и шагу; эпохи — ближайшие по логарифму (последнее
    действие могло быть обрезано бюджетом), как в surrogate.load_eval_rows."""
    hit = [i for i, (o, l, _) in enumerate(E.ACTION_TABLE) if o == opt and abs(l - float(lr)) < 1e-12]
    if not hit:
        return None
    return min(hit, key=lambda j: abs(math.log10(EPOCHS[j]) - math.log10(max(float(ep), 1.0))))


def k_l2re(records, default=1.0):
    """l2re / (rmse + brmse) по собранным цепочкам (перевод l2re в единицы суррогата)."""
    k = [h[2] / e for r in records for h, e in zip(r.get("hist") or [], r.get("err") or [])
         if h[2] is not None and e is not None and e > 0 and h[2] > 0]
    return float(np.median(k)) if k else float(default)


def records_to_rows(records, ctx, k=1.0):
    """Собранные цепочки -> столбцы таблицы переходов surrogate (без PDE, FILE, CTX). Ошибка —
    rmse+brmse после действия (err; если нет — l2re / k), перед первым действием E0 задачи.
    Переходы общего префикса берутся один раз, у записи-владельца."""
    cols = {c: [] for c in ("CH", "STEP", "SPENT", "A", "PA", "L", "LN", "SPF", "LAST")}
    n_ch = 0
    for r in records:
        chain, hist = r.get("chain") or [], r.get("hist") or []
        n = min(len(chain), len(hist))
        acts = [action_index(*c) for c in chain[:n]]
        if not n or any(x is None for x in acts):
            continue
        errs = list(r.get("err") or []) + [None] * n
        start = 0 if r.get("prefix_owner", True) else int(r.get("shared_prefix_len") or 0)
        l, pa, spent, spf = float(S._log(10.0 ** float(ctx[6]))), -1, 0, np.zeros(3)
        for i in range(n):
            e = errs[i] if errs[i] is not None else (hist[i][2] / k if hist[i][2] is not None else None)
            ln = float(S._log(e)) if e is not None and e > 0 else S.L_HI      # не число: расходимость
            if i >= start:
                for c, v in (("CH", n_ch), ("STEP", i), ("SPENT", spent), ("A", acts[i]), ("PA", pa),
                             ("L", l), ("LN", ln), ("SPF", spf.copy()), ("LAST", i == n - 1)):
                    cols[c].append(v)
            spf[S.FAM[acts[i]]] += int(chain[i][2])
            spent, pa, l = int(hist[i][0]), acts[i], ln
        n_ch += 1
    out = {c: np.array(v) for c, v in cols.items()}
    out["SPF"] = np.array(cols["SPF"], dtype=np.float64).reshape(-1, 3)
    return out


def training_table(base, records, a):
    """Таблица для обучения суррогата: базовая плюс собранные переходы целевой задачи отдельной
    «задачей» с тем же описателем. Возвращает (таблица, строки обучения, число собранных строк)."""
    tab, ctx = base["tab"], np.asarray(base["ctx"], dtype=np.float64)
    new = records_to_rows(records, ctx, k=k_l2re(records, base.get("k_l2re", 1.0)))
    n_old, n_new = len(tab["A"]), len(new["A"])
    names = list(tab["NAMES"])
    out = dict(NAMES=names + [a.pde + "@collect"], SUBDIRS=list(tab["SUBDIRS"]) + [""],
               CTX=np.vstack([tab["CTX"], ctx[None]]))
    new["PDE"] = np.full(n_new, len(names))
    new["FILE"] = np.full(n_new, (int(tab["FILE"].max()) + 1) if n_old else 0)
    new["CH"] = new["CH"] + ((int(tab["CH"].max()) + 1) if n_old else 0)
    for c in ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "L", "LN", "SPF", "LAST"):
        out[c] = np.concatenate([tab[c], new[c].astype(tab[c].dtype).reshape((n_new,) + tab[c].shape[1:])])
    for c, fill in (("LV", 0.0), ("LVN", np.nan)):
        if c in tab:
            out[c] = np.concatenate([tab[c], np.full((n_new,) + tab[c].shape[1:], fill)])
    keep = np.ones(n_old + n_new, dtype=bool)
    if a.pde in names and not a.use_old_target:
        keep[:n_old] = tab["PDE"] != names.index(a.pde)
    return out, np.where(keep)[0], n_new


def fit_surrogate(tab, rows, a, rnd):
    """Ансамбль модели одного шага на строках rows (точка подмены в тестах)."""
    return S.fit_ensemble(tab, rows, members=a.members, seed=a.collect_seed + rnd, quick=a.quick)


def score_chains(ens, ctx, chains, eps):
    """Среднее и разброс по членам ансамбля log10 итоговой ошибки цепочек."""
    L = S.rollout(ens, np.asarray(ctx, dtype=np.float64), [list(c) for c in chains], eps=eps)
    return L.mean(0), L.std(0)


def plan_ks(a):
    """Штрафы за разброс для лучевого поиска: без штрафа, оптимистичный (как у выбора), осторожный."""
    ks = [0.0] + ([-float(a.lcb_k)] if a.acq == "lcb" and a.lcb_k else []) + [1.0]
    return list(dict.fromkeys(ks))


def parse_worker(spec):
    """'k/K' -> (k, K); пусто — один сборщик."""
    if not spec:
        return 0, 1
    try:
        k, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        sys.exit(f"--worker {spec!r}: нужен вид k/K, например 0/4")
    if not 0 <= k < n:
        sys.exit(f"--worker {spec!r}: нужно 0 <= k < K")
    return k, n


def mine(key, a):
    """Достаётся ли цепочка с таким ключом этому сборщику (--worker k/K): crc ключа по модулю K."""
    k, n = getattr(a, "_worker", (0, 1))
    return n <= 1 or zlib.crc32(repr(tuple(key)).encode()) % n == k


def gen_candidates(ens, ctx, a, allowed, parents, tried, rng):
    """Кандидаты раунда: {ключ: (действия, эпохи, происхождение)}. Лучи surrogate.plan, мутации и
    скрещивания parents и лучей, случайные цепочки; только допустимые, без уже исполненных."""
    idx = np.where(allowed)[0].tolist()
    ok = set(idx)
    cons = None if a.consistent == "none" else a.consistent
    cand = {}

    def add(genome, origin):
        acts, eps = norm(genome, a)
        if not acts or any(x not in ok for x in acts):
            return
        if cons and not bool(S.consistent_ok(np.r_[-1, acts[:-1]], np.asarray(acts), cons).all()):
            return
        k = key_of(acts, eps)
        if k not in tried and k not in cand and mine(k, a):
            cand[k] = (acts, eps, origin)

    beams = []
    for k in plan_ks(a):
        r = S.plan(ens, np.asarray(ctx, dtype=np.float64), budget=a.budget, k=k, width=a.plan_width,
                   cap=a.plan_cap, max_len=a.max_len, allowed=allowed, consistent=cons, top=a.plan_top)
        for c, *_ in r["best"]:
            if c:
                beams.append([int(x) for x in c])
                add(c, f"beam k={k:g}")
    pool = [list(p) for p in parents] + beams[:max(0, a.n_parents - len(parents))]
    for p in pool:
        for _ in range(a.n_mut):
            g = G.mutate(p, rng, idx, a.max_len)
            if rng.random() < 0.3:
                g = G.mutate(g, rng, idx, a.max_len)
            add(g, "mutation")
    for _ in range(a.n_cross if len(pool) > 1 else 0):
        i, j = (int(x) for x in rng.choice(len(pool), size=2, replace=False))
        add(G.crossover(pool[i], pool[j], rng, a.max_len), "crossover")
    for _ in range(a.n_random):
        add(G.random_genome(rng, idx, a.budget, a.max_len), "random")
    return cand


def select_batch(chains, mu, sd, a, n, fixed=()):
    """Номера n цепочек по оценке J = mu − k·sd (lcb) или mu (greedy), меньше — лучше. Жадно по
    одной: с --share-prefix оценка кандидата уменьшается на prefix_bonus × (эпохи общего с уже
    выбранными префикса / бюджет); цепочки пачки различаются не меньше чем на --min-dist правок."""
    mu, sd = np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)
    score = mu - (a.lcb_k * sd if a.acq == "lcb" else 0.0)
    order = [int(i) for i in np.argsort(score, kind="stable")[:max(200, 20 * n)]]
    chosen, got = [list(c) for c in fixed], []
    while len(got) < n:
        best, best_v = None, float("inf")
        for i in order:
            if i in got or any(edit_dist(chains[i], c) < a.min_dist for c in chosen):
                continue
            v = float(score[i])
            if a.share_prefix and a.prefix_bonus and chosen:
                v -= a.prefix_bonus * max(shared_epochs(chains[i], c, a) for c in chosen) / a.budget
            if v < best_v:
                best, best_v = i, v
        if best is None:
            break
        got.append(best)
        chosen.append(list(chains[best]))
    return got


def tried_keys(st, a):
    out = set()
    for g in ([r.get("actions") for r in st["records"]] + [x for r in st["records"] for x in r.get("skipped") or []]
              + [p["actions"] for p in st["pending"]]):
        if g:
            out.add(key_of(*norm(g, a)))
    return out


def best_parents(st, a):
    """Задуманные цепочки лучших по итоговой l2re записей (остановленные приёмом 3 не берутся)."""
    rows = [r for r in st["records"] if r.get("l2re") is not None and not r.get("truncated") and r.get("actions")]
    rows.sort(key=lambda r: r["l2re"])
    out = []
    for r in rows:
        if r["actions"] not in out:
            out.append(list(r["actions"]))
    return out[:a.n_parents]


def select_round(st, a, allowed):
    """Пачка очередного раунда -> st['pending'] и st['sel'] (сведения о выборе)."""
    rnd = st["round"]
    rng = np.random.default_rng([a.collect_seed, rnd, 1])
    idx = np.where(allowed)[0].tolist()
    tried = tried_keys(st, a)
    sel = dict(fit_s=0.0, sel_s=0.0, n_candidates=0, n_train_rows=0, n_target_rows=0)
    items = []

    def push(acts, origin, acq, mu=None, sd=None):
        acts, eps = norm(acts, a)
        k = key_of(acts, eps)
        if not acts or k in tried:
            return False
        tried.add(k)
        items.append(dict(id=st["n_planned"], actions=acts, origin=origin, acq=acq,
                          pred_mean=_num(mu), pred_std=_num(sd)))
        st["n_planned"] += 1
        return True

    init = st["init_left"][:a.batch]
    st["init_left"] = st["init_left"][len(init):]
    if a.acq == "random":
        for g in init:
            push(g, "init", "init")
    else:
        base = base_table(a)
        if base.get("table_s"):
            st["table_s"] = round(st["table_s"] + base["table_s"], 1)
            base["table_s"] = 0.0                       # время сборки таблицы учитывается один раз
        t = time.time()
        tab, rows, n_new = training_table(base, st["records"], a)
        ens = fit_surrogate(tab, rows, a, rnd)
        sel.update(fit_s=round(time.time() - t, 1), n_train_rows=int(len(rows)), n_target_rows=int(n_new))
        t = time.time()
        cand = gen_candidates(ens, base["ctx"], a, allowed, best_parents(st, a), tried, rng)
        fixed = []
        for g in init:
            acts, eps = norm(g, a)
            if acts:
                mu, sd = score_chains(ens, base["ctx"], [acts], [eps])
                if push(acts, "init", "init", mu[0], sd[0]):
                    fixed.append(acts)
                cand.pop(key_of(acts, eps), None)
        keys = list(cand)
        if keys and len(items) < a.batch:
            chains, eps = [cand[k][0] for k in keys], [cand[k][1] for k in keys]
            mu, sd = score_chains(ens, base["ctx"], chains, eps)
            for i in select_batch(chains, mu, sd, a, a.batch - len(items), fixed=fixed):
                push(chains[i], cand[keys[i]][2], a.acq, mu[i], sd[i])
        sel.update(sel_s=round(time.time() - t, 1), n_candidates=len(keys))
    for _ in range(400):                                # случайный сбор и добор пачки
        if len(items) >= a.batch:
            break
        g = G.random_genome(rng, idx, a.budget, a.max_len)
        if mine(key_of(*norm(g, a)), a):
            push(g, "random", "random")
    st["pending"], st["sel"] = items, sel
    st["fit_s"] = round(st["fit_s"] + sel["fit_s"], 1)
    return items


# --------------------------------------------------------------------------------------------
# Исполнение пачки (приёмы 2 и 3) и учёт
# --------------------------------------------------------------------------------------------

def take_seed(st, a):
    k, n = getattr(a, "_worker", (0, 1))
    pool = np.asarray(G.parse_pool(a.seed_pool))[k::n]          # --worker k/K: свой срез пула сидов
    pool = np.random.default_rng(a.collect_seed).permutation(pool)
    seed = int(pool[st["seed_cursor"] % len(pool)])
    st["seed_cursor"] += 1
    return seed


def _account(st, res, epochs):
    st["n_runs"] += 1
    st["gpu_s"] = round(st["gpu_s"] + res["elapsed_s"], 1)
    st["epochs_real"] += int(epochs)


def _planned_epochs(acts, a):
    return int(sum(eff(acts, a.budget, len(acts))[1]))


def add_record(st, a, res, seed, item=None, shared=0, group=None, owner=True, epochs_real=0,
               elapsed=None, skipped=()):
    """Запись о цепочке в st['records'] (формат — в описании модуля)."""
    rec = dict(
        chain=res["chain"], hist=res["hist"], seed=int(seed),
        env=dict(keep_opt=bool(a.keep_opt), mode=(a.keep_opt_mode if a.keep_opt else None), budget=int(a.budget)),
        truncated=bool(res["truncated"]),
        shared_prefix_len=int(shared), round=int(st["round"]),
        acq=(item["acq"] if item else "prefix"),
        pred_mean=(item["pred_mean"] if item else None), pred_std=(item["pred_std"] if item else None),
        err=res["err"], l2re=res["l2re"], spent=int(res["spent"]),
        planned=(chain_str(item["actions"]) if item else None),
        actions=(list(item["actions"]) if item else None), origin=(item["origin"] if item else "prefix"),
        group=group, prefix_owner=bool(owner), epochs_real=int(epochs_real),
        elapsed_s=round(res["elapsed_s"] if elapsed is None else elapsed, 1),
        stop=res["stop"], error=res["error"], skipped=[list(x) for x in skipped])
    st["records"].append(rec)
    st["n_chains"] = len(st["records"])
    if getattr(a, "save_buffer", False):
        write_buffer(st, a, rec, res)
    if rec["l2re"] is not None and (st.get("best") is None or rec["l2re"] < st["best"]["l2re"]):
        st["best"] = dict(l2re=rec["l2re"], chain=chain_str_rec(rec["chain"]), seed=rec["seed"],
                          record=len(st["records"]) - 1, truncated=rec["truncated"])
    print(f"[collect r{rec['round']}] {len(st['records'])}: l2re="
          f"{'nan' if rec['l2re'] is None else '%.4e' % rec['l2re']} spent={rec['spent']} "
          f"{'ОСТАНОВЛЕНА ' if rec['truncated'] else ''}{'префикс=%d ' % shared if shared else ''}"
          f"({rec['origin']}; эпох обучено всего {st['epochs_real']}, GPU {st['gpu_s'] / 3600:.2f} ч) "
          f"{chain_str_rec(rec['chain'])}", flush=True)
    return rec


def chain_str_rec(chain):
    return ",".join("%s:%g:%d" % (c[0], c[1], c[2]) for c in chain)


def _drop(st, items, a):
    ids = {it["id"] for it in items}
    st["pending"] = [p for p in st["pending"] if p["id"] not in ids]
    st["epochs_planned"] += sum(_planned_epochs(it["actions"], a) for it in items)


def run_group(st, a, plen, items, over, save):
    """Одна группа пачки: цепочка без общего префикса (plen = 0) или префикс и концовки от его
    снимка. Возвращает причину остановки по лимиту времени или None."""
    seed = take_seed(st, a)
    watch = make_watch(st["records"], a)
    if plen == 0:
        it = items[0]
        name = f"{a.pde}_{a.tag}_csolo" if a.truncate else None
        _rm(_ckpt_local(name) if name else None)
        res = run_env(a, it["actions"], seed, len(it["actions"]), watch=watch, ckpt=name)
        _rm(_ckpt_local(name) if name else None)
        _account(st, res, res["spent"])
        _drop(st, [it], a)
        add_record(st, a, res, seed, item=it, epochs_real=res["spent"])
        save()
        return over()

    gid = st["n_groups"]
    st["n_groups"] += 1
    items = sorted(items, key=lambda it: len(it["actions"]) != plen)      # цепочка-префикс идёт первой
    prefix = items[0]["actions"][:plen]
    pre_name, br_name = f"{a.pde}_{a.tag}_cpre", f"{a.pde}_{a.tag}_cbr"
    pre_local, br_local = _ckpt_local(pre_name), _ckpt_local(br_name)
    _rm(pre_local)                      # префикс считается заново: чекпоинт должен быть его концом
    pre = run_env(a, prefix, seed, max(len(it["actions"]) for it in items), watch=watch,
                  ckpt=pre_name, ckpt_every=1)
    _account(st, pre, pre["spent"])
    exact = items[0] if len(items[0]["actions"]) == plen else None
    whole = pre["ok"] and len(pre["hist"]) == plen and pre["l2re"] is not None
    if not (whole and not pre["truncated"] and os.path.exists(pre_local)):
        # префикс остановлен приёмом 3, разошёлся или упал: концовки не исполняются. Запись одна:
        # цепочка-префикс, если она в пачке и исполнена целиком (тогда она закончена), иначе первая
        # цепочка группы с исполненной частью; остальные перечислены в skipped
        if whole and not pre["truncated"]:
            print(f"после префикса нет чекпоинта {pre_local}: концовки пропущены", flush=True)
            pre = dict(pre, stop="no_ckpt")
        it0 = exact if (whole and exact is not None) else items[0]
        if it0 is exact and pre["truncated"]:
            pre = dict(pre, truncated=False, stop="script_end")
        _drop(st, items, a)
        add_record(st, a, pre, seed, item=it0, shared=0, group=gid, owner=True, epochs_real=pre["spent"],
                   skipped=[it["actions"] for it in items if it is not it0])
        _rm(pre_local)
        save()
        return over()

    t_pre, owned, reason = pre["hist"][-1][3], False, None
    if exact is not None:
        _drop(st, [exact], a)
        add_record(st, a, pre, seed, item=exact, shared=plen, group=gid, owner=True, epochs_real=pre["spent"])
        owned = True
        save()
    for it in items:
        if it is exact:
            continue
        # лимит времени проверяется, когда группа уже дала хотя бы одну цепочку: префикс без
        # концовки был бы потрачен впустую. Остальные концовки остаются в pending, докатка
        # перегруппирует их и исполнит префикс заново
        reason = over() if owned else None
        if reason:
            break
        shutil.copyfile(pre_local, br_local)
        try:
            res = run_env(a, it["actions"], seed, len(it["actions"]), watch=make_watch(st["records"], a),
                          ckpt=br_name, resume=True, base_trace=pre["trace"], base_maps=pre.get("maps"))
        finally:
            _rm(br_local)
        new = max(0, res["spent"] - pre["spent"])
        _account(st, res, new)
        for h in res["hist"][plen:]:
            h[3] = round(h[3] + t_pre, 1)       # время цепочки: префикс и концовка подряд
        own = not owned and res["ok"] and len(res["hist"]) >= plen
        owned = owned or own
        _drop(st, [it], a)
        add_record(st, a, res, seed, item=it, shared=plen, group=gid, owner=own,
                   epochs_real=new + (pre["spent"] if own else 0),
                   elapsed=res["elapsed_s"] + (pre["elapsed_s"] if own else 0.0))
        save()
    if not owned:
        # ни одна концовка не дошла до записи: переходы префикса сохраняются отдельной записью
        add_record(st, a, pre, seed, shared=plen, group=gid, owner=True, epochs_real=pre["spent"])
        save()
    _rm(pre_local)
    return reason or over()


def run_pending(st, a, over, save):
    """Исполнить st['pending']; группы пересчитываются по оставшимся цепочкам (так докатка
    посреди группы честно исполняет префикс заново)."""
    while st["pending"]:
        plen, mem = plan_groups([p["actions"] for p in st["pending"]], a)[0]
        reason = run_group(st, a, plen, [st["pending"][i] for i in mem], over, save)
        if reason:
            return reason
    return None


def close_round(st, a, wall_s):
    recs = [r for r in st["records"] if r["round"] == st["round"]]
    pm = [abs(r["pred_mean"] - math.log10(r["err"][-1])) for r in recs
          if r["pred_mean"] is not None and not r["truncated"] and r["err"] and r["err"][-1]]
    sel = st.get("sel") or {}
    row = dict(round=st["round"], acq=a.acq, n_chains=len(recs),
               n_truncated=sum(r["truncated"] for r in recs),
               n_shared=sum(r["shared_prefix_len"] > 0 for r in recs),
               cum_chains=len(st["records"]), cum_epochs=st["epochs_real"],
               cum_epochs_planned=st["epochs_planned"], cum_gpu_s=st["gpu_s"],
               best_l2re=(st["best"]["l2re"] if st.get("best") else None),
               fit_s=sel.get("fit_s", 0.0), sel_s=sel.get("sel_s", 0.0), cum_fit_s=st["fit_s"],
               n_candidates=sel.get("n_candidates", 0), n_train_rows=sel.get("n_train_rows", 0),
               n_target_rows=sel.get("n_target_rows", 0),
               pred_mae=(round(float(np.mean(pm)), 4) if pm else None), wall_s=round(wall_s, 1))
    st["rounds"].append(row)
    print(f"[collect r{row['round']}] цепочек {row['cum_chains']} (в раунде {row['n_chains']}, остановлено "
          f"{row['n_truncated']}, с общим префиксом {row['n_shared']}); эпох {row['cum_epochs']} из "
          f"{row['cum_epochs_planned']} задуманных; GPU {row['cum_gpu_s'] / 3600:.2f} ч; лучшая l2re "
          f"{row['best_l2re']}; суррогат {row['fit_s']} с, ошибка прогноза {row['pred_mae']} дек.", flush=True)
    return row


# --------------------------------------------------------------------------------------------
# Состояние и запуск
# --------------------------------------------------------------------------------------------

SETTINGS = ("budget", "fidelity_note", "acq", "lcb_k", "batch", "rounds", "share_prefix", "prefix_bonus",
            "share_min", "truncate", "truncate_min", "truncate_min_ref", "mask", "max_len", "keep_opt",
            "keep_opt_mode", "lbfgs_tol", "plain_fnn", "hidden_layers", "seed_pool", "collect_seed",
            "use_old_target", "consistent", "members", "quick", "plan_width", "plan_cap", "plan_top",
            "n_parents", "n_mut", "n_cross", "n_random", "min_dist", "subdirs", "init", "no_clip", "save_buffer",
            "worker")
SAME = ("budget", "acq", "keep_opt", "keep_opt_mode", "lbfgs_tol", "plain_fnn", "hidden_layers", "mask")


def load_state(path, name, smoke):
    if os.path.exists(path):
        return json.load(open(path))
    if smoke:
        return None
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(OUT_REPO, f"rl_arch/online_env/{name}.json", repo_type="dataset")))
    except Exception as e:
        print(f"сохранённого сбора нет ({type(e).__name__}): старт с нуля", flush=True)
        return None


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pde", required=True)
    ap.add_argument("--budget", type=int, default=7000, help="бюджет эпох цепочки (приём 4: любой, напр. 3500)")
    ap.add_argument("--fidelity-note", default="", help="пометка о точности сбора, пишется в settings")
    ap.add_argument("--no-clip", action="store_true",
                    help="не исполнять действие, которое бюджет обрезал бы: цепочка кончается перед ним. Нужно "
                         "сбору с бюджетом меньше 7000: agent_combo.py выбрасывает действия, обрезанные таким бюджетом")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5, help="цепочек в раунде")
    ap.add_argument("--acq", default="lcb", choices=["lcb", "greedy", "random"])
    ap.add_argument("--lcb-k", type=float, default=1.0, help="J = среднее − k·разброс ансамбля")
    ap.add_argument("--use-old-target", action="store_true",
                    help="учить суррогат и на старом буфере целевой задачи (среда со сбросом оптимизатора)")
    ap.add_argument("--share-prefix", action="store_true", help="приём 2: общий префикс исполнять один раз")
    ap.add_argument("--prefix-bonus", type=float, default=0.1,
                    help="скидка к оценке кандидата, декад log10 ошибки за общий префикс длиной в бюджет")
    ap.add_argument("--share-min", type=int, default=300,
                    help="общий префикс короче этого числа эпох не выделять (докатка заново строит модель)")
    ap.add_argument("--truncate", type=float, default=0.0,
                    help="приём 3: остановить цепочку, если l2re хуже лучшей при тех же эпохах в C раз (0 выкл., совет 3.0)")
    ap.add_argument("--truncate-min", type=int, default=None,
                    help="не останавливать раньше этого числа эпох (по умолчанию половина бюджета: цепочки "
                         "с разгоном Adam и поздним L-BFGS в начале отстают)")
    ap.add_argument("--truncate-min-ref", type=int, default=5,
                    help="не останавливать, пока опорных цепочек с данными на этих эпохах меньше")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="лимит GPU-часов всего сбора, накопительно через докатки (0 без лимита)")
    ap.add_argument("--kernel-hours", type=float, default=0.0,
                    help="мягкий лимит одного процесса (сессия Kaggle): сохранить и выйти, продолжит --resume")
    ap.add_argument("--seed-pool", default="1000-1999", help="сиды цепочек (список или диапазон)")
    ap.add_argument("--collect-seed", type=int, default=0, help="сид выбора кандидатов и порядка сидов")
    ap.add_argument("--mask", default="", help="запрещённые действия, синтаксис среды: 'pso,adam:0.01'")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--init", action="append", default=[],
                    help="цепочки первых пачек (каркасы), несколько через ';' или повтором флага")
    ap.add_argument("--consistent", default="none", choices=["none", "safe", "all"],
                    help="брать только цепочки, которые среды со сбросом и без исполняют одинаково")
    ap.add_argument("--min-dist", type=int, default=1, help="минимум правок между цепочками пачки")
    ap.add_argument("--members", type=int, default=5, help="членов ансамбля суррогата")
    ap.add_argument("--quick", action="store_true", help="короткий бустинг суррогата (60 итераций)")
    ap.add_argument("--plan-width", type=int, default=256)
    ap.add_argument("--plan-cap", type=int, default=3000, help="до этого размера фронт луча перебирается целиком")
    ap.add_argument("--plan-top", type=int, default=10, help="лучших цепочек с каждого луча")
    ap.add_argument("--n-parents", type=int, default=6, help="лучших собранных цепочек для мутаций")
    ap.add_argument("--n-mut", type=int, default=40, help="мутаций на родителя")
    ap.add_argument("--n-cross", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=20, help="случайных цепочек среди кандидатов")
    ap.add_argument("--table", default="", help="кэш таблицы переходов (.npz); по умолчанию <save-dir>/surrogate_table.npz")
    ap.add_argument("--table-hf", default="", help="путь готовой таблицы в датасете результатов HF (если есть)")
    ap.add_argument("--subdirs", default="", help="папки буферов для таблицы через запятую (по умолчанию все)")
    ap.add_argument("--keep-opt", action="store_true")
    ap.add_argument("--keep-opt-mode", default="all", choices=["all", "safe"])
    ap.add_argument("--lbfgs-tol", type=float, default=None)
    ap.add_argument("--plain-fnn", action="store_true")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=float, default=1800.0,
                    help="заливать состояние в HF не чаще, чем раз в столько секунд, кроме границ раундов")
    ap.add_argument("--save-buffer", action="store_true",
                    help="строить карты ландшафта после каждого действия (как оценка) и писать цепочки файлами "
                         "в формате офлайнового буфера: <save-dir>/buffers_new/<tag>/<папка задачи>/, в HF — "
                         "rl_arch/buffers_new/<tag>/<папка задачи>/ (читает --target-buffer тренеров)")
    ap.add_argument("--buffer-dir", default="",
                    help="корень локальной копии буфера вместо <save-dir>/buffers_new/<tag>")
    ap.add_argument("--worker", default="",
                    help="k/K — сборщик k из K независимых (у каждого свой --tag): своя доля кандидатов, свой "
                         "срез --seed-pool, --collect-seed по умолчанию k")
    ap.add_argument("--smoke", action="store_true", help="бюджет 300 (если не задан), без HF")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    given = {x.split("=")[0] for x in (sys.argv[1:] if argv is None else argv) if x.startswith("--")}
    if a.smoke and "--budget" not in given:
        a.budget = 300
    if a.batch < 1 or a.max_len < 1 or a.rounds < 1 or a.budget < 1:
        sys.exit("нужно --batch >= 1, --rounds >= 1, --max-len >= 1, --budget >= 1")
    if a.truncate_min is None:
        a.truncate_min = a.budget // 2
    if a.truncate and a.truncate <= 1.0:
        sys.exit("--truncate: множитель должен быть больше 1 (или 0 — выключено)")
    a._worker = parse_worker(a.worker)
    if a._worker[1] > 1 and "--collect-seed" not in given:
        a.collect_seed = a._worker[0]
    if a._worker[1] > 1 and len(G.parse_pool(a.seed_pool)) < a._worker[1]:
        sys.exit("--worker: в --seed-pool меньше сидов, чем сборщиков")
    from killswitch import exit_if_killed
    allowed = E.parse_mask(a.mask)
    idx = set(np.where(allowed)[0].tolist())
    name = f"{a.pde}_{a.tag}_collect"
    os.makedirs(a.save_dir, exist_ok=True)
    path = os.path.join(a.save_dir, f"{a.tag}_collect.json")
    env = dict(keep_opt=bool(a.keep_opt), mode=(a.keep_opt_mode if a.keep_opt else None), budget=int(a.budget),
               lbfgs_tol=a.lbfgs_tol, plain_fnn=bool(a.plain_fnn), hidden_layers=a.hidden_layers)

    st = load_state(path, name, a.smoke) if a.resume else None
    if st is None:
        init = []
        for spec in (s for v in a.init for s in v.split(";") if s.strip()):
            g = E.parse_chain(spec)
            if not g or any(x >= G.N_AGENT or x not in idx for x in g):
                sys.exit(f"--init {spec!r}: действия вне пространства агента или под маской")
            init.append(g)
        st = dict(kind="collect_active", format=FORMAT, pde=a.pde, tag=a.tag,
                  settings={k: getattr(a, k) for k in SETTINGS}, env=env,
                  round=0, pending=[], sel={}, init_left=init, seed_cursor=0, n_planned=0, n_groups=0,
                  records=[], rounds=[], n_chains=0, n_runs=0, epochs_real=0, epochs_planned=0,
                  gpu_s=0.0, fit_s=0.0, table_s=0.0, wall_s=0.0, best=None, done=False, partial=True,
                  stop_reason=None, smoke=a.smoke)
    else:
        diff = [k for k in SAME if st["settings"].get(k) != getattr(a, k)]
        if diff:
            sys.exit(f"докатка {name}: флаги {diff} отличаются от сохранённых — это был бы другой сбор")
        if st.get("done") and st["round"] >= a.rounds:
            print(f"сбор {name} уже завершён: раундов {st['round']}, цепочек {st['n_chains']}, "
                  f"лучшая l2re {st['best']['l2re'] if st.get('best') else None}", flush=True)
            return st
        print(f"докатка {name}: раунд {st['round']}, цепочек {st['n_chains']}, в очереди {len(st['pending'])}, "
              f"{st['wall_s'] / 3600:.2f} ч", flush=True)
        st["done"], st["partial"], st["stop_reason"] = False, True, None
        st["settings"].update({k: getattr(a, k) for k in SETTINGS if k not in SAME and k != "init"})
    t0, t_proc, last_up = time.time() - st["wall_s"], time.time(), [time.time()]

    def save(up=False):
        st["wall_s"] = round(time.time() - t0, 1)
        do_up = not a.smoke and (up or time.time() - last_up[0] >= a.progress_every)
        if do_up and a.save_buffer and st.get("buffer"):
            flush_buffer(st, a)         # до состояния: залитое состояние не ссылается на незалитые карты
        with open(path, "w") as f:
            json.dump(st, f, indent=1)
        if do_up:
            last_up[0] = time.time()
            upload(st, name)

    def over_total():
        return "hours" if a.hours and (time.time() - t0) / 3600.0 >= a.hours else None

    def over():
        if over_total():
            return "hours"
        return "kernel" if a.kernel_hours and (time.time() - t_proc) / 3600.0 >= a.kernel_hours else None

    if a.save_buffer:
        print(f"буфер с картами: {buffer_dir(a)}"
              + ("" if a.smoke else f"; HF: {BUF_HF}/{a.tag}/{buffer_subdir(a.pde)}")
              + (f"; сборщик {a._worker[0]} из {a._worker[1]}" if a._worker[1] > 1 else ""), flush=True)
    reason = None
    while st["round"] < a.rounds:
        # выключатель задач между раундами: состояние и буфер прошлого раунда уже залиты
        exit_if_killed(name, remote=not a.smoke, what="сбор")
        reason = over_total()
        if reason:
            break
        if not st["pending"]:
            if not select_round(st, a, allowed):
                reason = "empty"            # под маской и бюджетом не осталось новых цепочек
                break
            print(f"[collect r{st['round']}] пачка ({a.acq}): "
                  + "; ".join(f"{chain_str(p['actions'])} [{p['origin']}"
                              + ("" if p["pred_mean"] is None else f", {p['pred_mean']:+.2f}±{p['pred_std']:.2f}")
                              + "]" for p in st["pending"]), flush=True)
            save()
        reason = run_pending(st, a, over, save)
        if not st["pending"]:
            close_round(st, a, time.time() - t0)
            st["round"] += 1
            st["sel"] = {}
            save(up=True)
        if reason:
            break
    if st["round"] >= a.rounds:
        reason = None                       # последний раунд закрыт: лимит кернела уже не важен
    if reason == "kernel":
        save(up=True)
        print(f"лимит кернела ({a.kernel_hours} ч): состояние сохранено, продолжит --resume", flush=True)
        return st
    st["done"], st["partial"], st["stop_reason"] = True, False, (reason or "rounds")
    save(up=True)
    print(f"сбор завершён ({st['stop_reason']}): цепочек {st['n_chains']}, вызовов среды {st['n_runs']}, эпох "
          f"{st['epochs_real']} (без приёмов 2 и 3 было бы {st['epochs_planned']}), GPU {st['gpu_s'] / 3600:.2f} ч, "
          f"суррогат {st['fit_s']:.0f} с, таблица {st['table_s']:.0f} с, стена {(time.time() - t0) / 3600:.2f} ч; "
          f"лучшая l2re {st['best']['l2re'] if st.get('best') else None}: "
          f"{st['best']['chain'] if st.get('best') else ''}", flush=True)
    if st.get("buffer"):
        b = st["buffer"]
        print(f"буфер с картами: файлов {b['files']} (переходов {b['transitions']}), залито {b['uploaded']}, "
              f"потеряно {b.get('lost', 0)}; локально {b['dir']}, HF {b['hf']}", flush=True)
    return st


if __name__ == "__main__":
    main()
