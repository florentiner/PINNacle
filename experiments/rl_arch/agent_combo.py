#!/usr/bin/env python
"""Боевой тренер офлайнового агента на таблице переходов: комбинация приёмов из литературы, каждый
включается своим флагом («полная комбинация» и «минус один приём» — один и тот же код).

Работает на CPU внутри кернела (GPU не нужен): собирает таблицу переходов из буферов HF
(surrogate.build_table, кэш в --work), учит ансамбль модели одного шага (surrogate.Ensemble),
учит агента и пишет чекпоинт в формате online_eval_env.load_agent (stat_dqn, один Q-выход на
действие). Второй шаг той же задачи очереди оценивает чекпоинт скриптом online_eval_env.py —
точная команда печатается в конце и лежит в отчёте (eval_cmd).

Состояние агента — то же, что у прототипа MBPO (paper/.../mbpo_agent.py): 15 каналов =
4 канала лоссов (веса первого слоя на них нулевые, агент их не видит) + 5 чисел скалярного
контекста online_eval_env.add_scalar_ctx (шаг/12, эпохи/7000, семейство и длина прошлого действия,
log10((rmse+brmse)/E0)/3+1) + 6 чисел описателя задачи (offline_rl.pde_desc). Награда —
убыль log10 ошибки за действие, дисконт 1.

Приёмы
  --mbpo-h H --mopo-lam L   MBPO (1906.08253): модельные продолжения длины H из реальных состояний,
                            награда штрафуется разбросом ансамбля, r - L*std (MOPO, 2005.13239).
  --cql-alpha A             CQL (2006.04779) на реальных переходах: A * (logsumexp Q по разрешённым
                            действиям - Q действия из данных).
  --iql-tau T               IQL (2110.06169), упрощённо: V-голова на общем стволе учится экспектилем T
                            от Q цели, бутстреп r + V(s') вместо max. Политика на оценке всё равно
                            argmax Q (взвешенного клонирования нет): имеет смысл только вместе с
                            маской/CQL. С --mve цели MVE имеют приоритет на своих строках.
  --guide CHAIN             каркас-префикс. Агент решает только ПОСЛЕ префикса: реальные переходы
                            берутся с потраченными эпохами >= длины префикса, модельные продолжения
                            ветвятся из них и из состояний, куда модель приводит префикс со старта
                            задачи (--guide-frac). Префикс пишется в meta чекпоинта (guide), оценка:
                            --policy script --script <префикс> --script-tail agent --model-file F.
  --ens N --ens-min M --utd U   REDQ (2101.05982): N голов Q на ОБЩЕМ стволе, цель — минимум по
                            случайным M головам, U шагов градиента на пачку. Чекпоинт — одна сеть
                            stat_dqn с головой, равной среднему голов: головы линейные и ствол общий,
                            поэтому среднее весов голов ТОЧНО равно среднему Q ансамбля (дистилляция
                            не нужна). Действие выбирается по среднему голов.
  --mve K                   расширение ценности через модель (MVE, 1803.00101): для части реальных
                            переходов бутстреп в s' заменяется K-уровневым просмотром вперёд по
                            суррогату: на каждом уровне перебираются все разрешённые действия,
                            берётся максимум r~ + Q_цели(следующее), r~ со штрафом MOPO. При K=1 это
                            точный одношаговый модельный оператор Беллмана; при K>1 путь жадный
                            (полное дерево 27^K не строится). Значения пересчитываются раз в --regen.
  --distill-top P --distill-w W   DQfD (1704.03732): маржинальная потеря в сторону действий лучших
                            P% записанных цепочек задачи-цели (по итоговой ошибке в бюджете).
  --pani-w W                сглаживание по сетке действий (в духе PANI): для реального перехода
                            соседние действия того же оптимизатора (соседний шаг или соседняя длина)
                            тянутся к той же цели минус --pani-pen, с весом W.
  --mask-schedule           от грубого к точному: первую треть обучения политика и max в целях видят
                            только один шаг обучения на пару «оптимизатор x длина» (--mask-coarse-lr),
                            дальше — все разрешённые действия. Итоговая маска (--mask) — в meta.

  --pbrs НЕ реализован намеренно: награда уже равна Phi(s') - Phi(s) с Phi = -log10 ошибки при
  дисконте 1, то есть это и есть потенциальное шейпирование награды «-log10 итоговой ошибки».
  Добавка F = gamma*Phi(s') - Phi(s) с тем же потенциалом просто удвоила бы награду (оптимальная
  политика та же, меняется только масштаб Q относительно штрафов CQL/MOPO/маржи).

Данные
  --pde                     задача-цель (имя из pde_meta.json; pb2d и ns2d — сокращения)
  --subdirs                 папки буферов (all | solvable | список через запятую)
  --target-chains N         только N случайных записанных цепочек цели (кривая «размер буфера»)
  --target-only             без других задач
  --extra-data PATH_OR_HF   JSON новых цепочек цели (формат ниже), путь локальный или в HF
                            (danil-e/pinnacle-optuna-db); несколько — через запятую
  --extra-only-target       старый буфер цели выбросить, для цели только --extra-data
  --table PATH_OR_HF        готовая таблица (npz от surrogate.save_table) вместо сборки из буферов

Формат --extra-data (его пишет experiments/rl_arch/collect_active.py; годятся и строки оценки
online_eval_env): JSON-словарь {"records": [...], "env": {...}, ...} или просто список записей,
одна запись = одна исполненная цепочка задачи-цели:
    {"chain": [[opt, lr, epochs], ...],        # как row["chain"] у online_eval_env: epochs — фактические
     "hist":  [[spent, loss, l2re, t], ...],   # по строке на действие, spent — эпох ПОСЛЕ действия;
                                               # loss и l2re — null, если не число
     "seed": 42,
     "env": {"keep_opt": false, "mode": null, "budget": 7000},   # среда сбора (или строка "reset")
     # необязательные поля:
     "err": [e1, e2, ...],                     # rmse + brmse ПОСЛЕ каждого действия (null — неизвестно)
     "shared_prefix_len": 0, "prefix_owner": true, "group": id,  # общий префикс (см. ниже)
     "rmse": r, "brmse": b, "pde": "poissonboltzmann2d", "budget": 7000}
Как читается:
  * Единицы. Таблица и состояние агента живут в rmse+brmse (ошибка до первого действия — E0 задачи из
    pde_meta.json), а в hist только l2re. Ошибка шага: (1) err[i], если число; иначе l2re[i] * k, где
    k = (rmse+brmse)/l2re: (2) по последнему шагу записи (поля rmse, brmse); (3) медиана по всем шагам
    файлов --extra-data, где известны и err, и l2re; (4) --extra-err-scale. Если k взять неоткуда,
    цепочка обрывается на этом шаге.
  * Расходимость. Шаг, где нет ни err, ни l2re (не число), — расходимость: log10 ошибки = +4 (верхний
    предел таблицы), цепочка на нём кончается. Переход с такой ошибкой после действия считается
    терминальным (и в старых буферах тоже).
  * Общий префикс. У записи с prefix_owner=false первые shared_prefix_len действий — тот же
    физический прогон, что у владельца группы: их переходы в таблицу не идут (считаются один раз),
    но состояние (эпохи, прошлое действие, ошибка) через них ведётся. Записи одной group попадают в
    одну половину при офлайновой оценке на половинах.
  * Бюджет. Последнее действие, обрезанное бюджетом 7000 (spent == 7000, эпох меньше номинала),
    относится к ближайшему большему номиналу и хранится с фактическими эпохами. Действие, обрезанное
    ДРУГИМ бюджетом сбора (низкая точность, env.budget < 7000), в таблицу не идёт: агент в среде на
    7000 эпох исполнил бы его целиком. Переходы за пределами 7000 эпох отбрасываются.
  * Цепочка обрывается на первом действии вне 27 (SOAP, BOOST) и при рассинхроне spent и epochs
    (страж, разведка).
  * Среда. Буферы и модель среды — со сбросом оптимизатора. Если env.keep_opt у новых цепочек true,
    динамика смешивается: скрипт предупреждает и пишет это в отчёт (extra.envs).

Выход
  --tag T    чекпоинт <out-dir>/rl_arch/agents_online/T.pt (и в HF под тем же путём, если есть токен
             HF_TOKEN_WRITE или HF_TOKEN) и отчёт rl_arch/online_env/<pde>_<T>_train.json
             (настройки, размеры, кривые обучения, офлайновые оценки на независимой половине, время).
             Локальный путь совпадает с путём в HF: --model-file rl_arch/agents_online/T.pt в том же
             кернеле читает локальный файл, в другом — качает из HF.
  --smoke    крошечные настройки, без заливки
  После сохранения чекпоинт проверяется через online_eval_env.load_agent + q_values (--no-verify
  отключает): при расхождении Q с обучавшейся сетью чекпоинт не заливается и код возврата 3.

Пример (полная комбинация, pb2d):
    env DDEBACKEND=pytorch python experiments/rl_arch/agent_combo.py --pde pb2d --tag pb2d_combo_full \\
        --guide Adam:0.01:1000,Adam:0.0001:2500 --mbpo-h 5 --mopo-lam 0.5 --cql-alpha 0.1 \\
        --ens 5 --ens-min 2 --utd 2 --mve 1 --distill-top 10 --distill-w 0.5 --pani-w 0.1 --mask-schedule
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(HERE, "track3"), HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DDEBACKEND", "pytorch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import surrogate as S  # noqa: E402

BUDGET = S.BUDGET
KMAX = S.K_MAX
MAX_STEPS = 40                       # предел длины жадной цепочки в модели
N_IN, N_LOSS, HID = 15, 4, 128       # 4 канала лоссов (игнорируются) + 5 контекст + 6 описатель
OUT_REPO = "danil-e/pinnacle-optuna-db"
ALIASES = {"pb2d": "poissonboltzmann2d", "ns2d": "ns2d_liddriven"}
ROWKEYS = ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "L", "LN", "LV", "LVN", "SPF", "LAST", "EPA", "GRP")
PSO_LR = np.arange(21, 27)           # PSO с ненулевым шагом
STAT_BLOCKS = 8                      # StatEncoder: mean, std, min, max, центр, q10, q50, q90
NON_STD = [0, 2, 3, 4, 5, 6, 7]


@contextlib.contextmanager
def one_thread():
    """Предсказания HGB на нескольких строках (замкнутый контур): запуск потоков OpenMP стоит
    в десятки раз дороже самой работы."""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:       # noqa: BLE001
        yield
        return
    with threadpool_limits(limits=1, user_api="openmp"):
        yield


# --------------------------------------------------------------------------------------------
# состояние агента: то же кодирование, что online_eval_env.add_scalar_ctx + описатель задачи
# --------------------------------------------------------------------------------------------

def agent_obs(ctx, step, spent, pa, l):
    """(n, 15): 4 нулевых канала лоссов, 5 чисел контекста, 6 чисел описателя."""
    ctx = np.asarray(ctx, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    spent = np.asarray(spent, dtype=np.float64)
    pa = np.asarray(pa, dtype=np.int64)
    l = np.asarray(l, dtype=np.float64)
    out = np.zeros((len(step), N_IN), dtype=np.float32)
    out[:, 4] = np.minimum(1.0, step / KMAX)
    out[:, 5] = np.minimum(1.0, spent / BUDGET)
    out[:, 6] = np.where(pa < 0, -1.0, (np.maximum(pa, 0) // 9) / 2.0)
    out[:, 7] = np.where(pa < 0, -1.0, (np.maximum(pa, 0) % 3) / 2.0)
    # оценка до первого действия подаёт err=1.0, дальше (rmse+brmse)/E0
    rel = np.maximum(10.0 ** (l - ctx[:, 6]), 1e-8)
    out[:, 8] = np.where(step == 0, 1.0, np.clip(np.log10(rel) / 3.0 + 1.0, -1.0, 1.0))
    out[:, 9:] = ctx[:, :6]
    return out


def norm_stats(s):
    mean = s.mean(0).astype(np.float32)
    std = np.maximum(s.std(0), 1e-3).astype(np.float32)
    mean[:N_LOSS], std[:N_LOSS] = 0.0, 1.0
    return mean, std


# --------------------------------------------------------------------------------------------
# таблица переходов
# --------------------------------------------------------------------------------------------

def hf_fetch(path, repo=OUT_REPO):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def ensure_buffers(subdirs):
    """Корень буферов для surrogate.build_table. Внутри кернела локального снимка нет:
    качаем нужные папки (snapshot_download; при полном кэше сеть не нужна)."""
    if os.environ.get("RL_BUFFER_DIR"):
        return os.environ["RL_BUFFER_DIR"]
    import offline_rl as O
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(O.REPO, repo_type="dataset",
                                 allow_patterns=[f"{sd}/*.pt" for sd in subdirs])
    except Exception as e:      # noqa: BLE001
        root = S.buffer_root()
        if not root:
            raise SystemExit(f"буферы не скачались ({type(e).__name__}) и локального снимка нет: "
                             "задайте RL_BUFFER_DIR или --table")
        print(f"snapshot_download не удался ({type(e).__name__}), беру локальный снимок", flush=True)
        return root


def with_epa(tab):
    """EPA — фактические эпохи действия (у буферов равны номиналу; у новых цепочек последнее
    действие может быть обрезано бюджетом). GRP — группа цепочки для разбиения на половины (у
    буферов сама цепочка; у новых цепочек с общим префиксом — одна на группу)."""
    if "EPA" not in tab:
        tab["EPA"] = S.EPOCHS[tab["A"]].astype(np.int64)
    if "GRP" not in tab:
        tab["GRP"] = tab["CH"].astype(np.int64)
    return tab


def get_table(args, subdirs, log=print):
    """Таблица переходов: --table, иначе кэш в --work, иначе сборка из буферов."""
    if args.table:
        return with_epa(S.load_table(hf_fetch(args.table)))
    import offline_rl as O
    os.makedirs(args.work, exist_ok=True)
    full = tuple(subdirs) == tuple(O.ALL_SUBDIRS)
    key = "" if full else f"_{zlib.crc32(','.join(subdirs).encode()):08x}"
    p = os.path.join(args.work, f"surrogate_table{key}.npz")     # полная — то же имя, что у surrogate.Work
    if not os.path.exists(p):
        t0 = time.time()
        tab = S.build_table(subdirs=list(subdirs), root=ensure_buffers(subdirs))
        S.save_table(tab, p)
        log(f"таблица собрана за {time.time() - t0:.0f} с: {p}")
    return with_epa(S.load_table(p))


def subset(tab, keep):
    out = dict(tab)
    for k in ROWKEYS:
        out[k] = tab[k][keep]
    return out


def ensure_pde(tab, name, meta):
    """Номер задачи в таблице; если её буфера нет (только --extra-data) — добавить строку CTX."""
    if name not in tab["NAMES"]:
        tab["NAMES"] = list(tab["NAMES"]) + [name]
        tab["SUBDIRS"] = list(tab["SUBDIRS"]) + [meta[name].get("subdir") or name]
        ctx = S.pde_ctx(meta[name])[None]
        tab["CTX"] = np.concatenate([tab["CTX"], ctx]) if len(tab["CTX"]) else ctx
    return tab["NAMES"].index(name)


def chain_finals(tab, pi=None):
    """Цепочки таблицы как статические: строки в бюджете (по фактическим эпохам), log10 ошибки
    после последней из них. Строки одной цепочки в таблице идут подряд."""
    ok = np.where(tab["SPENT"] + tab["EPA"] <= BUDGET)[0]
    if pi is not None:
        ok = ok[tab["PDE"][ok] == pi]
    out = dict(CH=[], final=[], spent=[], chains=[], rows=[])
    if len(ok):
        ch = tab["CH"][ok]
        cut = np.r_[0, np.where(np.diff(ch) != 0)[0] + 1, len(ok)]
        for s, e in zip(cut[:-1], cut[1:]):
            r = ok[s:e]
            out["CH"].append(int(ch[s]))
            out["final"].append(float(tab["LN"][r[-1]]))
            out["spent"].append(int(tab["SPENT"][r[-1]] + tab["EPA"][r[-1]]))
            out["chains"].append([int(a) for a in tab["A"][r]])
            out["rows"].append(r)
    for k in ("CH", "final", "spent"):
        out[k] = np.array(out[k])
    return out


def _act_index(opt, lr, ep, truncated):
    hit = [i for i, (o, l, e) in enumerate(S.ACTIONS)
           if o.lower() == str(opt).lower() and abs(l - float(lr)) < 1e-12]
    exact = [i for i in hit if S.EPOCHS[i] == int(ep)]
    if exact:
        return exact[0]
    up = [i for i in hit if S.EPOCHS[i] > int(ep)] if truncated else []
    return min(up, key=lambda i: S.EPOCHS[i]) if up else None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _step_errors(rec, n, k_all, scale):
    """rmse+brmse после каждого из n действий записи (nan — взять неоткуда) и источник; diverged[i] —
    на шаге нет ни err, ни l2re (не число)."""
    hist = rec["hist"]
    l2 = [_num(h[2]) for h in hist[:n]]
    e = [_num(x) for x in (list(rec.get("err") or []) + [None] * n)[:n]]
    div = [not np.isfinite(x) and not np.isfinite(l) for x, l in zip(e, l2)]
    r, b, last = _num(rec.get("rmse")), _num(rec.get("brmse")), _num(hist[-1][2])
    if np.isfinite(r + b) and np.isfinite(last) and last > 0 and len(hist) == len(rec["chain"]):
        k, src = (r + b) / last, "l2re*k_записи"
    elif k_all:
        k, src = float(k_all), "l2re*k_сбора"
    elif scale:
        k, src = float(scale), "l2re*scale"
    else:
        k, src = float("nan"), "нет"
    need = [not np.isfinite(x) and not dv for x, dv in zip(e, div)]
    if not any(need):
        return e, div, "err"
    out = [l * k if nd else x for x, l, nd in zip(e, l2, need)]
    return out, div, (src if all(need[i] or div[i] for i in range(n)) else "err+" + src)


def _rec_env(rec):
    """(метка среды сбора, бюджет сбора): env — словарь collect_active {keep_opt, mode, budget}
    или строка."""
    env, bud = rec.get("env"), rec.get("budget")
    if isinstance(env, dict):
        bud = bud or env.get("budget")
        label = f"keep_{env.get('mode') or 'all'}" if env.get("keep_opt") else "reset"
    else:
        label = str(env or "reset")
    return label, int(bud or BUDGET)


def load_extra(spec, allow_partial=False):
    recs = []
    for p in (x.strip() for x in spec.split(",") if x.strip()):
        with open(hf_fetch(p)) as f:
            d = json.load(f)
        # сбор заливает состояние после каждого раунда: незаконченный файл уже лежит в HF, и
        # обучение на нём молча получило бы меньше данных, чем заявлено
        if isinstance(d, dict) and d.get("done") is False and not allow_partial:
            sys.exit(f"{p}: сбор данных ещё не закончен (done=false); дождаться конца или --extra-allow-partial")
        rs = d["records"] if isinstance(d, dict) else d
        if isinstance(d, dict) and d.get("env") is not None:
            rs = [dict(r, env=d["env"]) if r.get("env") is None else r for r in rs]
        recs += [dict(r, _file=p) for r in rs]
    return recs


def extra_rows(tab, records, pi, err_scale=None):
    """Новые цепочки задачи pi -> строки таблицы (те же столбцы, что у surrogate.build_table).
    Возвращает (таблица с добавленными строками, сводка). Правила — в докстринге модуля."""
    tab = with_epa(dict(tab))
    name = tab["NAMES"][pi]
    e0 = float(10.0 ** tab["CTX"][pi, 6])
    cols = {k: [] for k in ROWKEYS}
    st = dict(records=len(records), chains=0, rows=0, cut=0, skipped=0, truncated=0, diverged=0,
              shared_skipped=0, envs={}, err_src={}, budgets={})
    ks = [_num(e) / _num(h[2]) for rec in records
          for h, e in zip(rec.get("hist") or [], rec.get("err") or [])
          if np.isfinite(_num(e)) and np.isfinite(_num(h[2])) and _num(e) > 0 and _num(h[2]) > 0]
    k_all = float(np.median(ks)) if ks else None
    st["k_err_per_l2re"] = k_all
    fid = int(tab["FILE"].max()) + 1 if len(tab["FILE"]) else 0
    cid = int(tab["CH"].max()) + 1 if len(tab["CH"]) else 0
    gid_of = {}
    for rec in records:
        chain, hist = rec.get("chain") or [], rec.get("hist") or []
        n = min(len(chain), len(hist))
        if rec.get("pde") not in (None, name) or not n:
            st["skipped"] += 1
            continue
        errs, div, src = _step_errors(rec, n, k_all, err_scale)
        label, bud = _rec_env(rec)
        start = 0 if rec.get("prefix_owner", True) else int(rec.get("shared_prefix_len") or 0)
        gkey = (rec.get("_file"), rec["group"]) if rec.get("group") is not None else None
        grp = gid_of.setdefault(gkey, cid) if gkey is not None else cid
        prev, spent, pa, spf, m = e0, 0, -1, np.zeros(3), 0
        for i in range(n):
            ep, after, e = int(chain[i][2]), int(hist[i][0]), (10.0 ** S.L_HI if div[i] else errs[i])
            cut_by_7000 = after >= bud and after == BUDGET
            a = (_act_index(chain[i][0], chain[i][1], ep, truncated=cut_by_7000)
                 if (ep > 0 and after - spent == ep and str(chain[i][0]) != "BOOST") else None)
            if a is None or not np.isfinite(e) or e <= 0:
                st["cut"] += 1
                break
            if i >= start:
                st["truncated"] += int(ep != S.EPOCHS[a])
                for key, v in (("PDE", pi), ("FILE", fid), ("CH", cid), ("STEP", i), ("SPENT", spent),
                               ("A", a), ("PA", pa), ("L", float(S._log(prev))), ("LN", float(S._log(e))),
                               ("LV", np.zeros(6)), ("LVN", np.full(6, np.nan)), ("SPF", spf.copy()),
                               ("LAST", False), ("EPA", ep), ("GRP", grp)):
                    cols[key].append(v)
                m += 1
            else:
                st["shared_skipped"] += 1
            spf[S.FAM[a]] += ep
            spent, pa, prev = after, a, e
            if div[i]:
                st["diverged"] += 1
                break
        if m:
            cols["LAST"][-1] = True
            st["chains"] += 1
            st["rows"] += m
            cid += 1
            fid += 1
        else:
            st["skipped"] += 1
            src = "нет" if not np.isfinite(_num(errs[0])) and not div[0] else src
        st["err_src"][src] = st["err_src"].get(src, 0) + 1
        if m:
            st["envs"][label] = st["envs"].get(label, 0) + 1
            st["budgets"][str(bud)] = st["budgets"].get(str(bud), 0) + 1
    if not st["rows"]:
        return tab, st
    out = dict(tab)
    for k in ROWKEYS:
        new = np.asarray(cols[k], dtype=tab[k].dtype)
        out[k] = np.concatenate([tab[k], new.reshape((len(new),) + tab[k].shape[1:])])
    return out, st


# --------------------------------------------------------------------------------------------
# ансамбли модели одного шага (surrogate.Ensemble), кэш в --work
# --------------------------------------------------------------------------------------------

class Models:
    """'full' — все цепочки, 'half0' / 'half1' — половины цепочек (то же разбиение, что
    surrogate.Work.half_model). Имя кэша содержит отпечаток данных: чужая таблица не подхватится."""

    def __init__(self, tab, work, members=5, quick=False, seed=0, log=print):
        self.tab, self.work, self.members, self.quick, self.seed, self.log = tab, work, members, quick, seed, log
        perm = np.random.default_rng(seed + 1).permutation(int(tab["GRP"].max()) + 1)
        self.half_of_row = perm[tab["GRP"]] % 2
        crc = 0
        for k in ("PDE", "CH", "A", "EPA", "L", "LN"):
            crc = zlib.crc32(np.ascontiguousarray(tab[k]).tobytes(), crc)
        self.sig = f"{len(tab['A'])}_{crc:08x}"
        self.fit_s, self._m = {}, {}
        os.makedirs(work, exist_ok=True)

    def rows(self, key):
        if key == "full":
            return np.arange(len(self.tab["A"]))
        return np.where(self.half_of_row == int(key[4]))[0]

    def get(self, key):
        if key in self._m:
            return self._m[key]
        p = os.path.join(self.work, f"combo_surr_{self.sig}_{key}_m{self.members}"
                                    f"{'_q' if self.quick else ''}_s{self.seed}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                self._m[key] = pickle.load(f)
            return self._m[key]
        t0 = time.time()
        t, r = self.tab, self.rows(key)
        # как surrogate.table_features, но с фактическими эпохами (обрезанные действия новых
        # цепочек) и шагом, ограниченным как в прогонах модели
        X = S.featurize(t["CTX"][t["PDE"][r]], np.minimum(t["STEP"][r], KMAX - 1), t["SPENT"][r],
                        t["SPF"][r], t["PA"][r], t["L"][r], t["A"][r], ep=t["EPA"][r])
        hp = dict(max_iter=60, learning_rate=0.15) if self.quick else {}
        seed = self.seed if key == "full" else self.seed + 10 * int(key[4]) + 1
        ens = S.Ensemble(members=self.members, seed=seed, **hp).fit(X, t["L"][r], t["LN"][r], t["CH"][r])
        with open(p, "wb") as f:
            pickle.dump(ens, f)
        self.fit_s[key] = round(time.time() - t0, 1)
        self.log(f"  модель среды {key}: {len(r)} переходов, {self.fit_s[key]} с")
        self._m[key] = ens
        return ens


# --------------------------------------------------------------------------------------------
# состояние модели и один шаг
# --------------------------------------------------------------------------------------------

def model_state(tab, rows):
    return dict(pde=tab["PDE"][rows].copy(), step=tab["STEP"][rows].astype(np.int64),
                spent=tab["SPENT"][rows].astype(np.int64), spf=tab["SPF"][rows].astype(np.float64),
                pa=tab["PA"][rows].astype(np.int64), l=tab["L"][rows].astype(np.float64))


def start_state(tab, pi, n):
    return dict(pde=np.full(n, pi), step=np.zeros(n, np.int64), spent=np.zeros(n, np.int64),
                spf=np.zeros((n, 3)), pa=np.full(n, -1, np.int64), l=np.full(n, tab["CTX"][pi, 6]))


def obs_of(tab, st):
    return agent_obs(tab["CTX"][st["pde"]], st["step"], st["spent"], st["pa"], st["l"])


def model_predict(ens, tab, st, a):
    """Следующая log10 ошибка по всем членам (члены, n) и фактические эпохи (обрезка бюджетом)."""
    a = np.asarray(a, dtype=np.int64)
    ep = np.minimum(S.EPOCHS[a], BUDGET - st["spent"])
    X = S.featurize(tab["CTX"][st["pde"]], np.minimum(st["step"], KMAX - 1), st["spent"],
                    st["spf"], st["pa"], st["l"], a, ep=ep)
    return ens.predict_ln(X, st["l"]), ep


def next_state(st, a, ep, ln):
    spf = st["spf"].copy()
    spf[np.arange(len(a)), S.FAM[a]] += ep
    return dict(pde=st["pde"], step=st["step"] + 1, spent=st["spent"] + ep, spf=spf,
                pa=np.asarray(a, np.int64), l=np.asarray(ln, np.float64))


def take(st, idx):
    return {k: v[idx] for k, v in st.items()}


def cat_states(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def prefix_states(ens, tab, pi, guide, n, rng):
    """n состояний после каркаса-префикса со старта задачи: на каждом шаге предсказание случайного
    члена ансамбля (разброс по членам = разброс стартов агента)."""
    st = start_state(tab, pi, n)
    for a in guide:
        P, ep = model_predict(ens, tab, st, np.full(n, a))
        st = next_state(st, np.full(n, a), ep, P[rng.integers(0, P.shape[0], n), np.arange(n)])
    return st


# --------------------------------------------------------------------------------------------
# сеть: общий ствол, N линейных голов Q, V-голова (IQL)
# --------------------------------------------------------------------------------------------

class QEns(nn.Module):
    def __init__(self, n_heads=1):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(N_IN, HID), nn.LayerNorm(HID), nn.GELU(),
                                   nn.Linear(HID, HID), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(HID, 27) for _ in range(n_heads)])
        self.v = nn.Linear(HID, 1)

    def forward(self, x):                               # (головы, B, 27)
        z = self.trunk(x)
        return torch.stack([h(z) for h in self.heads], 0)

    def value(self, x):
        return self.v(self.trunk(x)).squeeze(-1)


class Agent:
    def __init__(self, mean, std, n_heads=1, seed=0):
        torch.manual_seed(seed)
        self.net = QEns(n_heads)
        self.tgt = copy.deepcopy(self.net)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.info = {}

    def norm(self, s):
        return torch.as_tensor((np.asarray(s, dtype=np.float32) - self.mean) / self.std, dtype=torch.float32)

    def q(self, s, allowed=None, target=False):
        """Среднее голов (numpy), запрещённые действия = -inf."""
        with torch.no_grad():
            q = (self.tgt if target else self.net)(self.norm(s)).mean(0).numpy().astype(np.float64)
        if allowed is not None:
            q[:, ~np.asarray(allowed, dtype=bool)] = -np.inf
        return q

    def act(self, s, allowed=None):
        return np.argmax(self.q(s, allowed), 1)


def to_stat_dqn(net):
    """Ствол и среднее голов -> state_dict offline_rl.QNet('stat_dqn'). На постоянных плоскостях
    8 статистик канала равны (v, 0, v, v, v, v, v, v): первый слой раскладывается по семи блокам.
    Веса на 4 канала лоссов обнуляются — агент к ним точно инвариантен."""
    sd = net.state_dict()
    W = sd["trunk.0.weight"].clone()
    W[:, :N_LOSS] = 0.0
    big = torch.zeros(W.shape[0], STAT_BLOCKS * N_IN)
    for b in NON_STD:
        big[:, b * N_IN:(b + 1) * N_IN] = W / len(NON_STD)
    n = len(net.heads)
    return {"0.mlp.0.weight": big, "0.mlp.0.bias": sd["trunk.0.bias"].clone(),
            "0.mlp.1.weight": sd["trunk.1.weight"].clone(), "0.mlp.1.bias": sd["trunk.1.bias"].clone(),
            "0.mlp.3.weight": sd["trunk.3.weight"].clone(), "0.mlp.3.bias": sd["trunk.3.bias"].clone(),
            "1.weight": torch.stack([sd[f"heads.{i}.weight"] for i in range(n)]).mean(0),
            "1.bias": torch.stack([sd[f"heads.{i}.bias"] for i in range(n)]).mean(0)}


# --------------------------------------------------------------------------------------------
# маски и сетка действий
# --------------------------------------------------------------------------------------------

def make_allowed(mask):
    """'all' | 'mpso' (PSO с шагом > 0 запрещён) | 'long' (только действия от 500 эпох: Adam
    1000/2500, L-BFGS 500/1000) | токены surrogate.parse_mask ('pso', 'adam:0.01:100', номер)."""
    if mask == "all":
        return np.ones(27, bool)
    if mask == "mpso":
        allowed = np.ones(27, bool)
        allowed[PSO_LR] = False
        return allowed
    if mask == "long":
        return S.EPOCHS >= 500
    return S.parse_mask(mask)


def coarse_mask(allowed, lr_idx=1):
    """Грубое подмножество: один шаг обучения (индекс lr_idx) на пару «оптимизатор x длина»."""
    c = np.asarray(allowed, bool) & (S.LRI == lr_idx)
    return c if c.any() else np.asarray(allowed, bool)


def neighbours(allowed):
    """Соседи действия на сетке того же оптимизатора: соседняя длина или соседний (по величине)
    ненулевой шаг. (27, 4) индексы и маска; соседи вне разрешённых действий не берутся."""
    NB, NM = np.zeros((27, 4), np.int64), np.zeros((27, 4), bool)
    for a in range(27):
        o, lr, ep = S.ACTIONS[a]
        lrs = sorted({l for oo, l, _ in S.ACTIONS if oo == o and l > 0})
        eps = sorted({e for oo, _, e in S.ACTIONS if oo == o})
        k = 0
        for b in range(27):
            ob, lb, eb = S.ACTIONS[b]
            if ob != o or b == a or not allowed[b]:
                continue
            near_ep = lb == lr and abs(eps.index(eb) - eps.index(ep)) == 1
            near_lr = (eb == ep and lr > 0 and lb > 0 and abs(lrs.index(lb) - lrs.index(lr)) == 1)
            if near_ep or near_lr:
                NB[a, k], NM[a, k] = b, True
                k += 1
    return NB, NM


# --------------------------------------------------------------------------------------------
# MBPO: модельные продолжения; MVE: просмотр вперёд через модель
# --------------------------------------------------------------------------------------------

def gen_rollouts(ag, ens, tab, st, h, lam, eps, rng, allowed):
    """Продолжения длины h из состояний st: на каждом шаге предсказание случайного члена (MBPO),
    награда = убыль log10 ошибки - lam * разброс по членам (MOPO)."""
    st = take(st, np.where(st["spent"] < BUDGET)[0])
    out = dict(s=[], a=[], r=[], s2=[], d=[], u=[])
    al = np.where(allowed)[0]
    for _ in range(h):
        if not len(st["l"]):
            break
        s = obs_of(tab, st)
        a = ag.act(s, allowed)
        rnd = rng.random(len(a)) < eps
        if rnd.any():
            a[rnd] = rng.choice(al, int(rnd.sum()))
        P, ep = model_predict(ens, tab, st, a)
        ln = P[rng.integers(0, P.shape[0], len(a)), np.arange(len(a))]
        u = P.std(0)
        nst = next_state(st, a, ep, ln)
        d = nst["spent"] >= BUDGET
        out["s"].append(s)
        out["a"].append(a)
        out["r"].append((st["l"] - ln - lam * u).astype(np.float32))
        out["s2"].append(obs_of(tab, nst))
        out["d"].append(d.astype(np.float32))
        out["u"].append(u)
        st = take(nst, np.where(~d)[0])
    return {k: np.concatenate(v) for k, v in out.items()} if out["s"] else None


def mve_values(ag, ens, tab, rows, K, lam, gamma, allowed):
    """Бутстреп B(s') реальных переходов rows через модель: K уровней, на каждом перебор всех
    разрешённых действий, максимум r~ + gamma * max Q_цели(следующее); следующее состояние — среднее
    членов, r~ = убыль log10 ошибки - lam * разброс. Терминальное s' -> 0."""
    n = len(rows)
    st = next_state(model_state(tab, rows), tab["A"][rows], tab["EPA"][rows], tab["LN"][rows])
    B, disc = np.zeros(n), np.ones(n)
    idx = np.where(st["spent"] < BUDGET)[0]
    cur = take(st, idx)
    acts = np.where(allowed)[0]
    na = len(acts)
    for k in range(K):
        m = len(idx)
        if not m:
            break
        rep = take(cur, np.repeat(np.arange(m), na))
        a = np.tile(acts, m)
        P, ep = model_predict(ens, tab, rep, a)
        mu = P.mean(0)
        rt = rep["l"] - mu - lam * P.std(0)
        nst = next_state(rep, a, ep, mu)
        dn = nst["spent"] >= BUDGET
        qn = ag.q(obs_of(tab, nst), allowed, target=True).max(1)
        val = (rt + gamma * np.where(dn, 0.0, qn)).reshape(m, na)
        best = val.argmax(1)
        flat = np.arange(m) * na + best
        if k == K - 1:
            B[idx] += disc[idx] * val[np.arange(m), best]
            break
        B[idx] += disc[idx] * rt[flat]
        disc[idx] *= gamma
        go = ~dn[flat]
        cur, idx = take(nst, flat[go]), idx[go]
    return B


# --------------------------------------------------------------------------------------------
# обучение
# --------------------------------------------------------------------------------------------

def real_transitions(tab, r):
    ctx = tab["CTX"][tab["PDE"][r]]
    ep = tab["EPA"][r]
    return dict(rows=r, s=agent_obs(ctx, tab["STEP"][r], tab["SPENT"][r], tab["PA"][r], tab["L"][r]),
                a=tab["A"][r].astype(np.int64), r=(tab["L"][r] - tab["LN"][r]).astype(np.float32),
                s2=agent_obs(ctx, tab["STEP"][r] + 1, tab["SPENT"][r] + ep, tab["A"][r], tab["LN"][r]),
                # конец эпизода: бюджет выбран либо расходимость (ошибка на верхнем пределе таблицы)
                d=(((tab["SPENT"][r] + ep) >= BUDGET) | (tab["LN"][r] >= S.L_HI - 1e-9)).astype(np.float32),
                pde=tab["PDE"][r])


def demo_rows(tab, rows, pi, top_pct, min_spent, allowed, log=print):
    """Строки лучших top_pct% цепочек задачи-цели (по итоговой ошибке в бюджете) среди rows,
    с разрешённым действием. Цепочки короче min_spent эпох не ранжируются (если таких, что
    длиннее, меньше пяти — ранжируются все)."""
    lc = chain_finals(tab, pi)
    inrow = np.zeros(len(tab["A"]), bool)
    inrow[rows] = True
    cand = [i for i in range(len(lc["CH"])) if inrow[lc["rows"][i]].any()]   # с каркасом начало цепочки вне rows
    full = [i for i in cand if lc["spent"][i] >= min_spent]
    cand = full if len(full) >= 5 else cand
    if not cand:
        return np.zeros(0, np.int64), dict(chains=0, rows=0)
    fin = lc["final"][cand]
    thr = np.percentile(fin, top_pct)
    top = [i for i, f in zip(cand, fin) if f <= thr]
    r = np.concatenate([lc["rows"][i] for i in top])
    r = r[inrow[r] & allowed[tab["A"][r]]]
    return r, dict(chains=len(top), ranked=len(cand), rows=int(len(r)), thr_log10=float(thr),
                   best_log10=float(fin.min()))


def train(tab, rows, ens, pi, cfg, allowed, seed=0, steps=None, guide=(), log=print, name=""):
    """Двойной DQN (Хьюбер, жёсткое обновление цели раз в --tgt-every) на реальных переходах rows,
    плюс включённые приёмы (см. докстринг модуля). ens — модель среды этой же части данных
    (None: без модельных приёмов)."""
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    steps = int(steps or cfg.steps)
    rng = np.random.default_rng(seed)
    R = real_transitions(tab, rows)
    n_real = len(R["a"])
    if n_real < 2:
        raise SystemExit(f"{name}: реальных переходов {n_real} — учить не на чем (проверьте --guide, "
                         "--target-only, --extra-data)")
    mean, std = norm_stats(np.concatenate([R["s"], R["s2"]]))
    N = int(cfg.ens)
    ag = Agent(mean, std, n_heads=N, seed=seed)
    opt = torch.optim.Adam(ag.net.parameters(), lr=cfg.lr)
    tm = R["pde"] == pi
    if tm.any() and (~tm).any():
        p = np.where(tm, cfg.target_frac / tm.sum(), (1 - cfg.target_frac) / (~tm).sum())
    else:
        p = np.ones(n_real)
    cdf = np.cumsum(p / p.sum())

    def draw(k):
        return np.minimum(np.searchsorted(cdf, rng.random(k)), n_real - 1)

    use_model = ens is not None and cfg.mbpo_h > 0
    use_mve = ens is not None and cfg.mve > 0
    coarse = coarse_mask(allowed, cfg.mask_coarse_lr) if cfg.mask_schedule else None
    Xs, Xs2 = ag.norm(R["s"]), ag.norm(R["s2"])
    At, Rt, Dt = torch.as_tensor(R["a"]), torch.as_tensor(R["r"]), torch.as_tensor(R["d"])
    NB, NM = (torch.as_tensor(x) for x in neighbours(allowed))
    demo, dinfo = None, None
    if cfg.distill_w > 0 and cfg.distill_top > 0:
        dr, dinfo = demo_rows(tab, rows, pi, cfg.distill_top, cfg.distill_min_spent, allowed)
        if len(dr):
            ctx = tab["CTX"][tab["PDE"][dr]]
            demo = (ag.norm(agent_obs(ctx, tab["STEP"][dr], tab["SPENT"][dr], tab["PA"][dr], tab["L"][dr])),
                    torch.as_tensor(tab["A"][dr].astype(np.int64)))
        else:
            log(f"  {name}: у лучших цепочек нет строк с разрешённым действием — дистилляция выключена")
    gens, uncs, M, pool = [], [], None, None
    curves = dict(it=[], td=[], cql=[], iql=[], dqfd=[], pani=[], q=[], chain=[])
    acc = {k: [] for k in ("td", "cql", "iql", "dqfd", "pani", "q")}
    log_every = max(1, steps // 40)
    chain_every = max(1, steps // 8)
    t0 = time.time()
    for it in range(steps):
        now = coarse if (coarse is not None and it < steps // 3) else allowed
        now_t = torch.as_tensor(now)
        if it % cfg.regen == 0 and (use_model or use_mve):
            if use_model:
                st = model_state(tab, R["rows"][draw(cfg.n_branch)])
                if len(guide) and cfg.guide_frac > 0:
                    ng = max(1, int(cfg.n_branch * cfg.guide_frac))
                    st = cat_states(take(st, np.arange(max(0, cfg.n_branch - ng))),
                                    prefix_states(ens, tab, pi, guide, ng, rng))
                g = gen_rollouts(ag, ens, tab, st, cfg.mbpo_h, cfg.mopo_lam, cfg.eps, rng, now)
                if g is not None:
                    uncs.append(float(np.mean(g["u"])))
                    gens = (gens + [g])[-cfg.keep_gens:]
                if gens:
                    M = dict(s=ag.norm(np.concatenate([x["s"] for x in gens])),
                             s2=ag.norm(np.concatenate([x["s2"] for x in gens])),
                             **{k: torch.as_tensor(np.concatenate([x[k] for x in gens])) for k in ("a", "r", "d")})
            if use_mve:
                pi_ = draw(min(cfg.mve_n, max(n_real, 1)))
                pool = (torch.as_tensor(pi_), torch.as_tensor(
                    mve_values(ag, ens, tab, R["rows"][pi_], cfg.mve, cfg.mopo_lam, cfg.gamma, now),
                    dtype=torch.float32))
        nr = cfg.batch if M is None else max(1, int(cfg.batch * cfg.real_ratio))
        n_mv = int(nr * cfg.mve_frac) if pool is not None else 0
        i = torch.as_tensor(draw(nr - n_mv))
        bfix = torch.zeros(nr - n_mv)
        if n_mv:
            j = torch.as_tensor(rng.integers(0, len(pool[0]), n_mv))
            i, bfix = torch.cat([i, pool[0][j]]), torch.cat([bfix, pool[1][j]])
        xs, a, r, xs2, d = Xs[i], At[i], Rt[i], Xs2[i], Dt[i]
        is_mve = torch.zeros(nr, dtype=torch.bool)
        is_mve[nr - n_mv:] = True
        if M is not None and cfg.batch > nr:
            j = torch.as_tensor(rng.integers(0, len(M["a"]), cfg.batch - nr))
            xs, a, r = torch.cat([xs, M["s"][j]]), torch.cat([a, M["a"][j]]), torch.cat([r, M["r"][j]])
            xs2, d = torch.cat([xs2, M["s2"][j]]), torch.cat([d, M["d"][j]])
            is_mve = torch.cat([is_mve, torch.zeros(len(j), dtype=torch.bool)])
            bfix = torch.cat([bfix, torch.zeros(len(j))])
        for _ in range(max(1, cfg.utd)):
            with torch.no_grad():
                if cfg.iql_tau > 0:
                    b = ag.net.value(xs2)
                else:
                    an = ag.net(xs2).mean(0).masked_fill(~now_t, -1e9).argmax(1)
                    qt = ag.tgt(xs2)
                    if N > 1 and cfg.ens_min < N:       # REDQ: минимум по случайным M головам
                        qt = qt[torch.as_tensor(rng.choice(N, cfg.ens_min, replace=False))]
                    b = qt.gather(2, an.expand(len(qt), -1)[..., None]).squeeze(2).min(0).values
                y = r + cfg.gamma * (1 - d) * torch.where(is_mve, bfix, b)
            qa = ag.net(xs)                                             # (N, B, 27)
            q_sa = qa.gather(2, a.expand(N, -1)[..., None]).squeeze(2)  # (N, B)
            td = F.smooth_l1_loss(q_sa, y.expand_as(q_sa))
            loss = td
            acc["td"].append(td.item())
            acc["q"].append(q_sa.mean().item())
            if cfg.cql_alpha > 0:
                ok = now_t[None, :] | F.one_hot(a[:nr], 27).bool()
                cql = (torch.logsumexp(qa[:, :nr].masked_fill(~ok, -1e9), 2) - q_sa[:, :nr]).mean()
                loss = loss + cfg.cql_alpha * cql
                acc["cql"].append(cql.item())
            if cfg.iql_tau > 0:
                with torch.no_grad():
                    qmin = ag.tgt(xs).gather(2, a.expand(N, -1)[..., None]).squeeze(2).min(0).values
                diff = qmin - ag.net.value(xs)
                iql = (torch.where(diff > 0, cfg.iql_tau, 1 - cfg.iql_tau) * diff ** 2).mean()
                loss = loss + iql
                acc["iql"].append(iql.item())
            if demo is not None:
                j = torch.as_tensor(rng.integers(0, len(demo[1]), cfg.distill_batch))
                qd, ad = ag.net(demo[0][j]), demo[1][j]
                oh = F.one_hot(ad, 27).bool()
                aug = (qd + cfg.distill_margin * (~oh).float()).masked_fill(~(now_t[None, :] | oh), -1e9)
                dq = (aug.max(2).values - qd.gather(2, ad.expand(N, -1)[..., None]).squeeze(2)).mean()
                loss = loss + cfg.distill_w * dq
                acc["dqfd"].append(dq.item())
            if cfg.pani_w > 0:
                nb, nm = NB[a[:nr]], NM[a[:nr]]
                if nm.any():
                    qn = qa[:, :nr].gather(2, nb.expand(N, -1, -1))
                    tg = (y[:nr] - cfg.pani_pen)[None, :, None].expand_as(qn)
                    pani = (F.smooth_l1_loss(qn, tg, reduction="none") * nm[None]).sum() / (nm.sum() * N)
                    loss = loss + cfg.pani_w * pani
                    acc["pani"].append(pani.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.net.parameters(), 10.0)
            opt.step()
        if it % cfg.tgt_every == 0:
            ag.tgt.load_state_dict(ag.net.state_dict())
        if (it + 1) % log_every == 0 or it == steps - 1:
            curves["it"].append(it + 1)
            for k, v in acc.items():
                curves[k].append(round(float(np.mean(v)), 5) if v else None)
                v.clear()
        if ens is not None and ((it + 1) % chain_every == 0 or it == steps - 1):
            with one_thread():
                ch, _ = greedy_nominal(ag, ens, tab, pi, guide, now)
            curves["chain"].append([it + 1, S.chain_str(ch)])
            log(f"    {name} it {it + 1}/{steps} td {curves['td'][-1] if curves['td'] else float('nan')} "
                f"({time.time() - t0:.0f} с) {S.chain_str(ch)[:90]}")
    ag.tgt.load_state_dict(ag.net.state_dict())
    ag.info = dict(sec=round(time.time() - t0, 1), steps=steps, n_real=int(n_real),
                   n_real_target=int(tm.sum()), n_chains=int(len(np.unique(tab["CH"][rows]))),
                   n_model=(int(len(M["a"])) if M is not None else 0),
                   model_unc=(float(np.mean(uncs)) if uncs else None), demo=dinfo, curves=curves)
    return ag


# --------------------------------------------------------------------------------------------
# офлайновая оценка в ансамбле модели
# --------------------------------------------------------------------------------------------

def greedy_closed(ag, ens, tab, pi, guide=(), allowed=None):
    """Замкнутый контур по членам (PETS): член m ведёт свою траекторию, агент видит его ошибку.
    Первые len(guide) действий — каркас. Возвращает итоговые log10 ошибки по членам и цепочки."""
    Mm = ens.members
    st = start_state(tab, pi, Mm)
    chains = [[] for _ in range(Mm)]
    live = np.arange(Mm)
    for t in range(MAX_STEPS):
        sub = take(st, live)
        a = (np.full(len(live), guide[t], np.int64) if t < len(guide)
             else np.asarray(ag.act(obs_of(tab, sub), allowed), np.int64))
        P, ep = model_predict(ens, tab, sub, a)
        nst = next_state(sub, a, ep, P[live, np.arange(len(live))])
        for j, m in enumerate(live):
            chains[m].append(int(a[j]))
        for k in st:
            st[k] = st[k].copy()
            st[k][live] = nst[k]
        live = live[nst["spent"] < BUDGET]
        if not len(live):
            break
    return st["l"].copy(), chains


def greedy_nominal(ag, ens, tab, pi, guide=(), allowed=None):
    """Агент на траектории среднего ансамбля: одна цепочка (с каркасом) и её фактические эпохи."""
    st = start_state(tab, pi, 1)
    ch, eps = [], []
    while st["spent"][0] < BUDGET and len(ch) < MAX_STEPS:
        a = (np.array([guide[len(ch)]], np.int64) if len(ch) < len(guide)
             else np.asarray(ag.act(obs_of(tab, st), allowed), np.int64))
        P, ep = model_predict(ens, tab, st, a)
        st = next_state(st, a, ep, P.mean(0))
        ch.append(int(a[0]))
        eps.append(int(ep[0]))
    return ch, eps


def score_chain(ens, tab, pi, chain, eps=None):
    """Открытая цепочка в ансамбле (эпохи обрезаны бюджетом): итог log10 ошибки по членам."""
    if eps is None:
        eps, left = [], BUDGET
        for a in chain:
            e = int(min(S.EPOCHS[a], left))
            eps.append(e)
            left -= e
        keep = [i for i, e in enumerate(eps) if e > 0]
        chain, eps = [chain[i] for i in keep], [eps[i] for i in keep]
    return S.rollout(ens, tab["CTX"][pi], [list(chain)], eps=[list(eps)])[:, 0]


def static_best(ens, tab, name):
    """Замеренные статические цепочки задачи (surrogate.MEASURED, варианты среды со сбросом) в
    этом ансамбле; лучшая по среднему. Для задач без замеренных цепочек — (None, {})."""
    pi = tab["NAMES"].index(name)
    res = {}
    for tag, spec, note in S.MEASURED.get(name, []):
        if "без сброса" in note:
            continue                      # суррогат описывает среду со сбросом
        res[tag] = float(score_chain(ens, tab, pi, S.parse_chain(spec)).mean())
    return (min(res, key=res.get) if res else None), res


def score_agent(ag, ens, tab, pi, guide, allowed):
    """Оценка агента одним ансамблем: замкнутый контур, номинальная цепочка, лучшая статическая.
    Δ = log10 итоговой ошибки агента − лучшей статической (< 0 — агент лучше)."""
    name = tab["NAMES"][pi]
    with one_thread():
        fin, chs = greedy_closed(ag, ens, tab, pi, guide, allowed)
        nom, nom_ep = greedy_nominal(ag, ens, tab, pi, guide, allowed)
        nom_L = float(score_chain(ens, tab, pi, nom, nom_ep).mean())
        best, stat = static_best(ens, tab, name)
    ref = stat[best] if best else None
    return dict(closed=float(fin.mean()), closed_sd=float(fin.std()),
                closed_chains=[S.chain_str(c) for c in chs], nominal=S.chain_str(nom),
                nominal_len=len(nom), nominal_L=nom_L, static_best=best, static=stat,
                d_closed=(float(fin.mean() - ref) if best else None),
                d_nominal=(nom_L - ref if best else None))


def first_actions_check(tab, half_of_row, held, pi, chain):
    """Проверка на реальных данных отложенной половины: итог цепочек задачи (от 6000 эпох), которые
    начинаются с первого / первых двух действий цепочки агента."""
    lc = chain_finals(tab, pi)
    sel = [i for i in range(len(lc["CH"]))      # STEP == 0: цепочки без начала (общий префикс) не годятся
           if lc["spent"][i] >= 6000 and half_of_row[lc["rows"][i][0]] == held
           and tab["STEP"][lc["rows"][i][0]] == 0]
    if not sel or not chain:
        return dict(n=len(sel))
    fin = lc["final"][sel]
    chs = [lc["chains"][i] for i in sel]
    m1 = [f for f, c in zip(fin, chs) if c[0] == chain[0]]
    m2 = [f for f, c in zip(fin, chs) if c[:2] == chain[:2]]
    return dict(n=len(fin), median_all=float(np.median(fin)), best_all=float(fin.min()),
                n_first1=len(m1), med_first1=(float(np.median(m1)) if m1 else None),
                pct_first1=(float((fin < np.median(m1)).mean()) if m1 else None),
                n_first2=len(m2), med_first2=(float(np.median(m2)) if m2 else None))


# --------------------------------------------------------------------------------------------
# чекпоинт, проверка загрузки, заливка
# --------------------------------------------------------------------------------------------

def hf_upload(src, path_in_repo, msg, tries=8):
    """Как online_train_env.save_agent / upload: токен HF_TOKEN_WRITE или HF_TOKEN, повторы с
    нарастающей паузой (лимит HF — 128 коммитов в час на репозиторий)."""
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print(f"нет токена HF — {path_in_repo} только локально", flush=True)
        return False
    from huggingface_hub import upload_file
    for attempt in range(tries):
        try:
            upload_file(path_or_fileobj=src, path_in_repo=path_in_repo, repo_id=OUT_REPO,
                        repo_type="dataset", token=tok, commit_message=msg)
            print(f"uploaded {path_in_repo}", flush=True)
            return True
        except Exception as e:      # noqa: BLE001
            print(f"upload retry {attempt}: {str(e)[:80]}", flush=True)
            time.sleep(min(600, 20 * 2 ** attempt))
    return False


def save_ckpt(ag, tag, meta, out_dir):
    """Чекпоинт в формате online_train_env.save_agent (его читает online_eval_env.load_agent)."""
    payload = dict(variant="stat_dqn", hl_gauss=None, state_dict=to_stat_dqn(ag.net),
                   mean=ag.mean.reshape(1, N_IN, 1, 1), std=ag.std.reshape(1, N_IN, 1, 1),
                   n_chains=int(ag.info.get("n_chains", 0)), l2re_best=None, tag=tag, meta=meta)
    path = os.path.join(out_dir, "rl_arch", "agents_online", f"{tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def verify_ckpt(path, ag, tab, pi, guide, n=16):
    """Чекпоинт через online_eval_env.load_agent + q_values на состояниях, собранных так же, как
    их собирает цикл оценки (loss_state + add_scalar_ctx + описатель): Q совпадает с обучавшейся
    сетью и не зависит от каналов лоссов. Первым проверяется состояние сразу после каркаса."""
    import online_eval_env as E
    from offline_rl import loss_state, pde_desc, pde_meta
    name = tab["NAMES"][pi]
    net, mean, std, variant = E.load_agent(path)
    meta = net._meta
    desc = pde_desc(name)
    e0 = float(pde_meta()[name]["init_err"])
    out = dict(variant=variant, in_ch=int(np.asarray(mean).shape[1]), state_mode=meta.get("state_mode"),
               desc_match=bool(np.allclose(desc, tab["CTX"][pi, :6])),
               e0_match=bool(abs(np.log10(e0) - tab["CTX"][pi, 6]) < 1e-9))
    rng = np.random.default_rng(0)
    allowed = np.ones(27, bool)
    allowed[[int(i) for i in meta.get("mask", [])]] = False
    worst, same, first = 0.0, 0, None
    P = int(sum(S.EPOCHS[a] for a in guide))
    for k in range(n):
        if k == 0:      # решение сразу после каркаса (без каркаса — стартовое состояние оценки)
            step, spent, last_a = len(guide), P, (guide[-1] if len(guide) else None)
            err = e0 * (0.05 if len(guide) else 1.0)
        else:
            step, spent, last_a = int(rng.integers(1, 12)), int(rng.integers(100, 6900)), int(rng.integers(0, 27))
            err = float(10 ** rng.uniform(-4, 0))
        maps = (loss_state(*(10 ** rng.uniform(-6, 2, 3)), 10 ** rng.uniform(-6, 2)) if step
                else np.zeros((4, 26, 26), np.float32))
        st = E.add_scalar_ctx(maps, step, meta["ctx_kmax"], spent, meta["ctx_budget"], last_a, err / e0)
        st = np.concatenate([st, np.stack([np.full(st.shape[1:], v, dtype=np.float32) for v in desc])])
        q = E.q_values(net, st, mean, std, variant)
        st2 = st.copy()
        st2[:4] = loss_state(*(10 ** rng.uniform(-6, 2, 3)), None)
        q2 = E.q_values(net, st2, mean, std, variant)
        obs = agent_obs(tab["CTX"][[pi]], np.array([step]), np.array([spent]),
                        np.array([-1 if last_a is None else last_a]), np.array([np.log10(err)]))
        qm = ag.q(obs)[0]
        worst = max(worst, float(np.abs(q - qm).max()), float(np.abs(q - q2).max()))
        qa, qb = np.where(allowed, q, -np.inf), np.where(allowed, qm, -np.inf)
        same += int(np.argmax(qa) == np.argmax(qb))
        if k == 0:
            first = S.action_str(int(np.argmax(qa)))
    # argmax_agree — справочно: при почти равных Q два пути вычисления могут разойтись в выборе
    out.update(max_abs_dq=worst, argmax_agree=f"{same}/{n}", first_action_after_guide=first,
               ok=bool(worst < 1e-3 and out["desc_match"] and out["e0_match"]))
    return out


def eval_command(name, tag, guide_spec, inverse=False, seeds="42,43,44,45,46"):
    mf = f"rl_arch/agents_online/{tag}.pt"
    pol = (f"--policy script --script {guide_spec} --script-tail agent --model-file {mf}" if guide_spec
           else f"--policy agent --model-file {mf}")
    return (f"python experiments/rl_arch/online_eval_env.py {pol} --no-state --state-mode loss "
            f"--pde {name}{' --plain-fnn' if inverse else ''} --budget {BUDGET} --seeds {seeds} --tag n_{tag}")


# --------------------------------------------------------------------------------------------
# основной прогон
# --------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("данные")
    g.add_argument("--pde", required=True)
    g.add_argument("--subdirs", default="all", help="all | solvable | папки буферов через запятую")
    g.add_argument("--target-chains", type=int, default=0, help="N случайных цепочек цели (0 — все)")
    g.add_argument("--target-only", action="store_true")
    g.add_argument("--extra-data", default="", help="JSON новых цепочек цели, локально или в HF; через запятую")
    g.add_argument("--extra-only-target", action="store_true")
    g.add_argument("--extra-allow-partial", action="store_true",
                   help="брать файл сбора, даже если сбор ещё не закончен (done=false)")
    g.add_argument("--extra-err-scale", type=float, default=0.0,
                   help="(rmse+brmse)/l2re для записей без err и без итоговых rmse/brmse")
    g.add_argument("--table", default="", help="готовая таблица npz (локально или в HF) вместо сборки")
    g.add_argument("--data-seed", type=int, default=0)
    g = ap.add_argument_group("приёмы")
    g.add_argument("--mbpo-h", type=int, default=0)
    g.add_argument("--mopo-lam", type=float, default=0.0)
    g.add_argument("--cql-alpha", type=float, default=0.0)
    g.add_argument("--iql-tau", type=float, default=0.0)
    g.add_argument("--guide", default="", help="каркас-префикс, напр. Adam:0.01:1000,Adam:0.0001:2500")
    g.add_argument("--guide-frac", type=float, default=0.5,
                   help="доля стартов модельных продолжений из состояний после каркаса")
    g.add_argument("--ens", type=int, default=1)
    g.add_argument("--ens-min", type=int, default=2)
    g.add_argument("--utd", type=int, default=1)
    g.add_argument("--mve", type=int, default=0)
    g.add_argument("--mve-n", type=int, default=2000, help="реальных переходов в пуле MVE на пересчёт")
    g.add_argument("--mve-frac", type=float, default=0.5, help="доля реальной части пачки с целями MVE")
    g.add_argument("--distill-top", type=float, default=0.0, help="лучшие P%% цепочек цели")
    g.add_argument("--distill-w", type=float, default=0.0)
    g.add_argument("--distill-margin", type=float, default=0.1)
    g.add_argument("--distill-batch", type=int, default=64)
    g.add_argument("--distill-min-spent", type=int, default=6000)
    g.add_argument("--pani-w", type=float, default=0.0)
    g.add_argument("--pani-pen", type=float, default=0.05)
    g.add_argument("--mask", default="long", help="all | mpso | long | токены маски (pso,adam:0.01:100,...)")
    g.add_argument("--mask-schedule", action="store_true")
    g.add_argument("--mask-coarse-lr", type=int, default=1, choices=[0, 1, 2])
    g = ap.add_argument_group("обучение")
    g.add_argument("--steps", type=int, default=12000)
    g.add_argument("--batch", type=int, default=256)
    g.add_argument("--real-ratio", type=float, default=0.5)
    g.add_argument("--n-branch", type=int, default=3000)
    g.add_argument("--regen", type=int, default=1000)
    g.add_argument("--keep-gens", type=int, default=6)
    g.add_argument("--target-frac", type=float, default=0.5, help="доля реальной пачки из задачи-цели")
    g.add_argument("--eps", type=float, default=0.2)
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--tgt-every", type=int, default=250)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--members", type=int, default=5, help="членов ансамбля модели среды")
    g.add_argument("--surr-quick", action="store_true", help="быстрые настройки HGB (отладка)")
    g = ap.add_argument_group("оценка и выход")
    g.add_argument("--offline-eval", default="half", choices=["none", "half", "both"],
                   help="half: ещё один агент на половине цепочек, оценка моделью другой половины")
    g.add_argument("--offline-steps", type=int, default=0, help="шагов у агентов на половинах (0 — как --steps)")
    g.add_argument("--tag", required=True)
    g.add_argument("--work", default="agent_combo_work", help="кэш таблицы и моделей среды")
    g.add_argument("--out-dir", default=".")
    g.add_argument("--no-upload", action="store_true")
    g.add_argument("--no-verify", action="store_true")
    g.add_argument("--eval-seeds", default="42,43,44,45,46")
    g.add_argument("--smoke", action="store_true")
    return ap


def run(args, log=None):
    log = log or (lambda *a: print(*a, flush=True))
    t_start = time.time()
    import offline_rl as O
    if args.smoke:      # только как умолчания: явные флаги не перезаписываются
        given = set(a.split("=")[0] for a in getattr(args, "_argv", []) if a.startswith("--"))
        for flag, v in (("--steps", 300), ("--batch", 64), ("--n-branch", 200), ("--regen", 100),
                        ("--mve-n", 100), ("--tgt-every", 50), ("--distill-batch", 16)):
            if flag not in given:
                setattr(args, flag[2:].replace("-", "_"), v)
        args.surr_quick = args.no_upload = True
    meta_all = O.pde_meta()
    name = ALIASES.get(args.pde, args.pde)
    if name not in meta_all:
        raise SystemExit(f"задачи {name!r} нет в track3/pde_meta.json")
    if args.ens < 1 or (args.ens > 1 and not 1 <= args.ens_min <= args.ens):
        raise SystemExit("--ens >= 1 и 1 <= --ens-min <= --ens")
    guide = S.parse_chain(args.guide) if args.guide else []
    guide_spec = S.chain_str(guide)
    P = int(sum(S.EPOCHS[a] for a in guide))
    if P >= BUDGET:
        raise SystemExit(f"каркас занимает {P} эпох из {BUDGET}: агенту нечего решать")
    allowed = make_allowed(args.mask)
    if not allowed.any():
        raise SystemExit("маска запретила все действия")

    # ---- данные ----
    subdirs = O.resolve_subdirs(args.subdirs)
    tsub = meta_all[name].get("subdir")
    if args.target_only and not args.table:
        subdirs = [tsub] if tsub else []        # чужие буферы не качаем
    drop_old = args.extra_only_target
    if drop_old and not args.table:
        subdirs = [s for s in subdirs if s != tsub]
    if args.table or subdirs:
        tab = get_table(args, subdirs, log)
    else:                                       # только --extra-data
        tab = dict({k: np.zeros((0,), np.int64) for k in ROWKEYS}, CTX=np.zeros((0, 7)), NAMES=[], SUBDIRS=[])
        for k, shp, dt in (("L", (0,), float), ("LN", (0,), float), ("LV", (0, 6), float),
                           ("LVN", (0, 6), float), ("SPF", (0, 3), float), ("LAST", (0,), bool)):
            tab[k] = np.zeros(shp, dt)
    pi = ensure_pde(tab, name, meta_all)
    sizes = dict(table_rows=int(len(tab["A"])), table_pdes=len(tab["NAMES"]),
                 target_rows_buffer=int((tab["PDE"] == pi).sum()),
                 target_chains_buffer=int(len(np.unique(tab["CH"][tab["PDE"] == pi]))))
    keep = np.ones(len(tab["A"]), bool)
    if args.target_only:
        keep &= tab["PDE"] == pi
    if drop_old:
        keep &= tab["PDE"] != pi
    if args.target_chains > 0 and not drop_old:
        chs = np.unique(tab["CH"][tab["PDE"] == pi])
        pick = np.random.default_rng(args.data_seed).choice(chs, min(args.target_chains, len(chs)), replace=False)
        keep &= (tab["PDE"] != pi) | np.isin(tab["CH"], pick)
    tab = subset(tab, keep)
    extra_info = None
    if args.extra_data:
        tab, extra_info = extra_rows(tab, load_extra(args.extra_data, args.extra_allow_partial), pi, args.extra_err_scale or None)
        log(f"новые цепочки: {extra_info}")
        if set(extra_info["envs"]) - {"reset"}:
            log("ВНИМАНИЕ: часть новых цепочек собрана без сброса оптимизатора, а буферы и модель среды — "
                "со сбросом: динамика смешана")
        if set(extra_info["budgets"]) - {str(BUDGET)}:
            log(f"ВНИМАНИЕ: бюджеты сбора {sorted(extra_info['budgets'])}: агент учится в среде на {BUDGET} "
                "эпох, цепочки короче — только начало эпизода")
    if not len(tab["A"]):
        raise SystemExit("после отбора данных не осталось ни одного перехода")
    tm = tab["PDE"] == pi
    sizes.update(rows=int(len(tab["A"])), chains=int(len(np.unique(tab["CH"]))), target_rows=int(tm.sum()),
                 target_chains=int(len(np.unique(tab["CH"][tm]))),
                 pdes=int(len(np.unique(tab["PDE"]))))
    log(f"данные: {sizes}")

    models = Models(tab, args.work, members=args.members, quick=args.surr_quick, seed=0, log=log)
    ok = (tab["SPENT"] + tab["EPA"] <= BUDGET) & (tab["SPENT"] >= P)   # в бюджете и после каркаса
    sizes.update(real_used=int(ok.sum()), real_used_target=int((ok & tm).sum()))

    # ---- офлайновая оценка: агент на половине цепочек, модель независимой половины ----
    offline = {}
    halves = {"none": [], "half": [0], "both": [0, 1]}[args.offline_eval]
    for h in halves:
        rows = np.where(ok & (models.half_of_row == h))[0]
        if len(rows) < 2 or (models.half_of_row == 1 - h).sum() < 2:
            log(f"половина {h}: мало данных, офлайновая оценка пропущена")
            continue
        ag_h = train(tab, rows, models.get(f"half{h}"), pi, args, allowed, seed=args.seed,
                     steps=(args.offline_steps or args.steps), guide=guide, log=log, name=f"half{h}")
        row = dict(own=score_agent(ag_h, models.get(f"half{h}"), tab, pi, guide, allowed),
                   ind=score_agent(ag_h, models.get(f"half{1 - h}"), tab, pi, guide, allowed))
        row["first"] = first_actions_check(tab, models.half_of_row, 1 - h, pi, S.parse_chain(row["ind"]["nominal"]))
        row["info"] = {k: v for k, v in ag_h.info.items() if k != "curves"}
        offline[f"half{h}"] = row
        log(f"половина {h}: Δ независимой модели {row['ind']['d_closed']}, своей {row['own']['d_closed']}; "
            f"{row['ind']['nominal']}")

    # ---- итоговый агент: все цепочки, полная модель ----
    need_model = args.mbpo_h > 0 or args.mve > 0
    ens_full = models.get("full")
    ag = train(tab, np.where(ok)[0], ens_full if need_model else None, pi, args, allowed, seed=args.seed,
               guide=guide, log=log, name="full")
    final = {"full": score_agent(ag, ens_full, tab, pi, guide, allowed)}
    for key in sorted(k for k in models._m if k != "full"):     # не независимы: агент видел все цепочки
        final[key] = score_agent(ag, models.get(key), tab, pi, guide, allowed)
    log(f"итоговый агент: {final['full']['nominal']} (Δ в своей модели {final['full']['d_closed']})")

    cmd = eval_command(name, args.tag, guide_spec, bool(meta_all[name].get("inverse")), args.eval_seeds)
    combo = {k: getattr(args, k) for k in (
        "mbpo_h", "mopo_lam", "cql_alpha", "iql_tau", "guide_frac", "ens", "ens_min", "utd", "mve", "mve_n",
        "mve_frac", "distill_top", "distill_w", "distill_margin", "pani_w", "pani_pen", "mask",
        "mask_schedule", "steps", "batch", "real_ratio", "target_frac", "seed", "target_chains",
        "target_only", "extra_data", "extra_only_target")}
    meta = dict(ctx_kmax=KMAX, ctx_budget=BUDGET, ctx_err="err", err_norm="init", pde_ctx=True,
                scalar_ctx=True, state_mode="loss", mask=[int(i) for i in np.where(~allowed)[0]],
                gamma=float(args.gamma), keep_opt=False, keep_opt_mode=None, lbfgs_tol=None,
                pde=name, train_pdes=[tab["NAMES"][i] for i in np.unique(tab["PDE"])],
                source="agent_combo.py (офлайн, таблица переходов + суррогат)",
                guide=guide_spec, guide_actions=[int(a) for a in guide], guide_epochs=P,
                eval_cmd=cmd, combo=combo,
                note="каналы лоссов 0-3 имеют нулевые веса; агент видит шаг, эпохи, прошлое действие, "
                     "(rmse+brmse)/E0 и описатель задачи. При непустом guide агент обучен решать только "
                     "после каркаса: оценивать через --policy script --script <guide> --script-tail agent")
    path = save_ckpt(ag, args.tag, meta, args.out_dir)
    log(f"чекпоинт: {path}")
    verify = None
    if not args.no_verify:
        try:
            verify = verify_ckpt(path, ag, tab, pi, guide)
        except Exception as e:      # noqa: BLE001 — чекпоинт уже сохранён, проверка его не отменяет
            verify = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        log(f"проверка загрузки через online_eval_env: {verify}")
    uploaded = False
    bad = verify is not None and not verify.get("ok")
    if bad:
        log("ПРОВЕРКА НЕ ПРОШЛА: чекпоинт в HF не заливается, код возврата 3 (оценку не запускать)")
    if not args.no_upload and not bad:
        uploaded = hf_upload(path, f"rl_arch/agents_online/{args.tag}.pt",
                             f"agent_combo {args.tag} ({ag.info['n_chains']} цепочек)")
    # partial=True: сборщики строк оценки (collect_results, surrogate.load_eval_rows) этот файл
    # пропускают — это отчёт обучения, а не строка оценки
    report = dict(kind="agent_combo_train", partial=True, pde=name, tag=args.tag, smoke=bool(args.smoke),
                  settings={k: v for k, v in vars(args).items() if not k.startswith("_")},
                  guide=guide_spec, guide_epochs=P, mask=meta["mask"], sizes=sizes, extra=extra_info,
                  surrogate=dict(members=args.members, quick=bool(args.surr_quick), fit_s=models.fit_s,
                                 data_sig=models.sig),
                  train={k: v for k, v in ag.info.items() if k != "curves"}, curves=ag.info["curves"],
                  offline=dict(final=final, halves=offline), verify=verify,
                  checkpoint=f"rl_arch/agents_online/{args.tag}.pt", checkpoint_uploaded=uploaded,
                  eval_cmd=cmd, wall_s=round(time.time() - t_start, 1))
    rname = f"{name}_{args.tag}_train"
    rpath = os.path.join(args.out_dir, "rl_arch", "online_env", f"{rname}.json")
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    if not args.no_upload:
        hf_upload(rpath, f"rl_arch/online_env/{rname}.json", f"rl_arch agent_combo {rname}", tries=9)
    log(f"отчёт: {rpath} ({report['wall_s']} с)")
    log("команда оценки чекпоинта:")
    log(cmd)
    report["_paths"] = dict(ckpt=path, report=rpath)
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args._argv = argv
    report = run(args)
    if report["verify"] is not None and not report["verify"].get("ok"):
        sys.exit(3)         # следующий шаг задачи (оценка на GPU) не должен стартовать с таким чекпоинтом
    return report


if __name__ == "__main__":
    main()
