#!/usr/bin/env python
"""Зонд «компьютерное зрение на картах ландшафта»: извлекает ли свёрточная/трансформерная модель
из КАРТ что-то полезное сверх их уровня и какие приёмы CV помогают.

Без RL и без обучения PINN: обучение с учителем на записанных буферах (22 УрЧП, ~195 тыс.
переходов, действия случайны). Образец = переход: карты S (4x26x26: общий лосс, оператор, границы,
изменение общего — online_eval_env.build_state), карты предыдущего шага той же цепочки, скаляры
(шаг, потрачено/7000, прошлое действие, описатель задачи), действие. Цель y = log10 ошибки после
действия минус log10 текущей ошибки. Известно (TRACK3, находка E): бустинг по (действие, шаг,
прошлое действие) даёт R² следующей ошибки 0.74, с шестью числами уровня карт 0.958, со всеми
статистиками карт 0.961; перемешивание пикселей не меняет результат обученного агента.

Конфиги (имя[-энкодер]; без энкодера берётся выбранный по фолду 0 конфигов maps-*):
    scalars     состояния нет: скаляры и действие, MLP
    level       скаляры + log10 текущей ошибки + шесть чисел уровня карт (surrogate.level6), MLP
    level-hgb   то же бустингом surrogate (контроль: не слаб ли MLP как базовая линия уровня)
    maps        энкодер на сырых картах + скаляры и текущая ошибка (уровень энкодер достаёт сам)
    mapslv      maps + шесть чисел уровня отдельно (не в порядке по умолчанию)
    shape       каждая карта приведена к нулевому среднему и единичному разбросу (как режим shape
                в offline_rl.apply_state_mode), шесть чисел уровня поданы отдельно: энкодер видит
                только форму
    shuffle     пиксели переставлены (одна перестановка на образец, общая для каналов): контроль,
                распределение значений то же, пространственной структуры нет
    stack2      текущие + предыдущие карты, 8 каналов
    delta       текущие карты + разность с предыдущими, 8 каналов
    aug-d4      maps + случайный элемент D4 при обучении (скрытая плоскость автокодировщика
                канонической ориентации не имеет — offline_rl.d4_apply)
    aug-shift   maps + случайный сдвиг на несколько пикселей
    ssl-mae     предобучение энкодера восстановлением замаскированных блоков на ВСЕХ картах
                (цель — форма карты, декодер из вектора признаков), затем дообучение
    ssl-con     контрастное предобучение SimCLR (виды: D4 + сдвиг + шум), затем дообучение
    frozen      энкодер после ssl-mae заморожен, учится только голова (качество представления;
                признаки считаются один раз — так подешевело бы обучение агента); frozen-con — то же
                после ssl-con
    Замены MAE для предобучения энкодера агента (волна 49; не в порядке по умолчанию, обычно с --pretrain-only):
    ssl-vae     β-VAE: из вектора признаков — среднее и лог-дисперсия латента (--vae-dim), декодер
                восстанавливает всю нормированную карту (уровень и форму), лосс = восстановление + --vae-beta * KL
    ssl-dyn     самопредсказание в латенте с действием (SPR/SGI): модель перехода по (z, действие)
                предсказывает проекцию следующих карт цепочки от EMA-копии энкодера (--dyn-ema), плюс обратная
                динамика — действие по (z, z') с весом --dyn-inv; пара (карты, следующие карты) под общим
                элементом D4. Нужны переходы внутри цепочки (последний шаг цепочки пропускается)
    ssl-out     исход с учителем по всем задачам: признаки энкодера + скаляры и действие (без текущей ошибки:
                у агента её нет, уровень энкодер должен достать из карт) -> изменение log10 ошибки (Huber)
    ssl-maeout  ssl-mae + голова исхода ssl-out на незамаскированных картах с весом --out-w
                У ssl-out и ssl-maeout предобучение видит цель y всех строк: их R² в режиме фолдов
                оптимистичен (флаг ssl_saw_labels в отчёте), в «лучший конфиг с картами» они не идут.
                --save-encoder пишет веса предобученного энкодера (cvprobe_<tag>_enc_<вид>_<энкодер>.pt,
                вид — mae, con, vae, dyn, out, maeout;
                в HF rl_arch/encoders/) для offline_rl.py --encoder-init
    lopo:level, lopo:best   «исключить УрЧП»: level и лучший конфиг с картами на --lopo задачах

Предобучение без чужих буферов (волна 55; только с --pretrain-only):
    --views СПЕЦ   карты из файлов видов целевой задачи (map_views.py: render_views.py или collect_active.py --views;
                   спецификация — префиксы rl_arch/views/<тег> через запятую или локальные папки, задача — единственная
                   папка --subdirs). У состояния карта среды (вид 0) и V видов того же снимка траектории (другие старты
                   автокодировщика); повторы состояний (общий префикс) убираются по ключу сид + начало цепочки.
                   --views-max K — только виды 0..K-1 (1 — одна карта среды на состояние: контроль без видов)
    --procedural N N процедурных цепочек (гауссовы поля, чаша или седло, рост к краю, плоские карты застоя) вместо
                   буферов — ноль уравнений; вместе с --views — смесь
    ssl-mv         MAE на всех видах + согласованность видов одного состояния: VICReg (Bardes et al., ICLR 2022) между
                   проекциями признаков двух разных видов (--mv-w вес, --mv-dim ширина проектора); строки без второго
                   вида (плоские карты, карта среды без видов) — только MAE. Нужны --views

--pretrain-only — только предобучение и сохранение энкодеров (без level, фолдов и lopo; волна 46: MAE
на картах одной задачи, --subdirs <папка> --max-per-pde N); --ssl-steps — число шагов предобучения
вместо числа эпох, чтобы наборы разного размера получали одинаковое число шагов.

Энкодеры: conv (энкодер статьи RL/rl_utils/DQN_classes.ConvEncoder; между свёртками нет
нелинейностей, до MLP это линейное отображение со средним по карте), convnext (энкодер агентов,
offline_rl.make_encoder), resnet (малый ResNet под 26x26), vit (патч 2, 4 слоя), cnn (первый
свёрточный вариант rl_arch, не в порядке по умолчанию).

Метрики на отложенных цепочках (GroupKFold по цепочкам; у lopo — по отложенной задаче): R² и
Spearman по y, R² следующей log10 ошибки, по каждой задаче; выигрыш над level с бутстреп-интервалом
90% по цепочкам; ранговая корреляция предсказанного и реального улучшения внутри (задача, корзина
шага) и внутри (задача, корзина шага, действие); секунды обучения и число параметров.
Правило вердикта: «карты добавляют информацию», если лучший конфиг с картами выигрывает у level
(и у level-hgb, если он посчитан) не меньше --min-gain R² с интервалом выше нуля И средний выигрыш
на lopo не отрицателен.

Отчёт <out-dir>/cvprobe_<tag>.json пишется после КАЖДОГО конфига и заливается в HF как
rl_arch/online_env/cvprobe_<tag>.json; --resume пропускает посчитанное; --hours — мягкий предел.

    python experiments/rl_arch/cv_probe.py --tag w20 --hours 4.5 --resume          # кернел, GPU
    python experiments/rl_arch/cv_probe.py --tag w20s --smoke                      # проверка буферов
    python experiments/rl_arch/cv_probe.py --tag syn --synthetic 400 --smoke       # без сети и буферов
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
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
ROWKEYS = ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "L", "LN", "LV", "LVN", "SPF", "LAST")
ENCODERS = ("conv", "convnext", "convnextw", "convnexthr", "resnet", "resnetpos", "vit", "cnn")
DEFAULT_ENCODERS = ("conv", "convnext", "resnet", "vit")
FALLBACK_ENCODER = "convnext"
BASES = {
    "scalars":    dict(),
    "level":      dict(err=True, lv=True),
    "level-hgb":  dict(hgb=True),
    "maps":       dict(maps="raw", err=True),
    "mapslv":     dict(maps="raw", err=True, lv=True),
    "shape":      dict(maps="shape", err=True, lv=True),
    "shuffle":    dict(maps="shuffle", err=True),
    "stack2":     dict(maps="stack2", err=True),
    "delta":      dict(maps="delta", err=True),
    "aug-d4":     dict(maps="raw", err=True, aug="d4"),
    "aug-shift":  dict(maps="raw", err=True, aug="shift"),
    "ssl-mae":    dict(maps="raw", err=True, ssl="mae"),
    "ssl-con":    dict(maps="raw", err=True, ssl="con"),
    "frozen":     dict(maps="raw", err=True, ssl="mae", frozen=True),
    "frozen-con": dict(maps="raw", err=True, ssl="con", frozen=True),
    "ssl-vae":    dict(maps="raw", err=True, ssl="vae"),
    "ssl-dyn":    dict(maps="raw", err=True, ssl="dyn"),
    "ssl-out":    dict(maps="raw", err=True, ssl="out"),
    "ssl-maeout": dict(maps="raw", err=True, ssl="maeout"),
    "ssl-mv":     dict(maps="raw", err=True, ssl="mv"),
}
SSL_LABELS = ("out", "maeout")      # предобучение видит исход y: в режиме фолдов R² оптимистичен
NOT_MAPS = ("scalars", "level", "level-hgb", "shuffle")     # не кандидаты в «лучший конфиг с картами»
DEFAULT_CONFIGS = (["scalars", "level", "level-hgb"] + [f"maps-{e}" for e in DEFAULT_ENCODERS]
                   + ["shape", "shuffle", "stack2", "delta", "aug-d4", "aug-shift", "ssl-mae", "ssl-con",
                      "frozen", "frozen-con", "lopo:level", "lopo:best"])
DEFAULT_LOPO = "poissonboltzmann2d,ns2d_liddriven,heat2d_multiscale,wave1d,poisson3d_complexgeometry"
STEP_EDGES = (1, 3, 6)              # корзины шага: 0 | 1-2 | 3-5 | 6+
HF_DIR = "rl_arch/online_env"
ENC_DIR = "rl_arch/encoders"       # предобученные энкодеры (--save-encoder) в HF-датасете результатов


# --------------------------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------------------------

def load_data(args, log=print):
    """Первый тяжёлый вызов: все буферы с картами. Возвращает (tab, maps): tab — таблица переходов
    surrogate.build_table (PDE, CH, STEP, SPENT, A, PA, L, LN, LV, SPF, CTX, NAMES, ...), maps —
    (N, 4, H, W) состояния ДО действия в том же порядке строк. Таблица и карты строятся одним
    загрузчиком (offline_rl.episodes_to_arrays_chains по файлу), совпадение строк проверяется по
    шести числам уровня. --max-per-pde оставляет случайные цепочки целиком."""
    import offline_rl as O
    if getattr(args, "synthetic", 0):
        return synthetic_data(args.synthetic, seed=args.seed)
    if getattr(args, "views", "") or getattr(args, "procedural", 0):
        return map_only_data(args, log)
    meta = O.pde_meta()
    subdirs = O.resolve_subdirs(args.subdirs)
    if not os.environ.get("RL_BUFFER_DIR") and not args.max_files:
        import agent_combo as AC
        os.environ["RL_BUFFER_DIR"] = AC.ensure_buffers(subdirs)
    rng = np.random.default_rng(args.seed)
    dtype = np.float16 if args.map_dtype == "float16" else np.float32
    tabs, parts = [], []
    ch_off = file_off = 0
    for sd in subdirs:
        t0 = time.time()
        eps = O.load_episodes(None, sd, max_files=args.max_files)
        if not eps:
            log(f"  {sd}: буфер пуст, пропущен")
            continue
        try:
            t = S.build_table(subdirs=[sd], episodes={sd: eps}, meta=meta, verbose=False)
        except ValueError:
            log(f"  {sd}: ни одной цепочки, пропущен")
            continue
        ms = []
        for ep in eps:
            try:
                ms.append(O.episodes_to_arrays_chains([ep], reward_form="delta", verbose=False)["S"])
            except ValueError:          # как в build_table: в файле не осталось ни одной цепочки
                continue
        del eps
        m = np.concatenate(ms, 0)
        del ms
        if len(m) != len(t["A"]):
            raise SystemExit(f"{sd}: карт {len(m)}, строк таблицы {len(t['A'])} — загрузчики разошлись")
        if not np.allclose(S.level6(m), t["LV"], atol=1e-4, rtol=1e-4):
            raise SystemExit(f"{sd}: уровень карт не совпал с таблицей — строки не выровнены")
        if args.max_per_pde and len(m) > args.max_per_pde:
            ug, inv, cnt = np.unique(t["CH"], return_inverse=True, return_counts=True)
            order = rng.permutation(len(ug))
            take = order[np.cumsum(cnt[order]) <= args.max_per_pde]
            if not len(take):
                take = order[:1]
            keep = np.isin(inv, take)
            for k in ROWKEYS:
                t[k] = t[k][keep]
            m = m[keep]
        t["PDE"] = np.full(len(m), len(tabs), dtype=np.int64)
        t["CH"] = t["CH"] + ch_off
        t["FILE"] = t["FILE"] + file_off
        ch_off, file_off = int(t["CH"].max()) + 1, int(t["FILE"].max()) + 1
        tabs.append(t)
        parts.append(m.astype(dtype))
        log(f"  {sd}: переходов {len(m)}, цепочек {len(np.unique(t['CH']))}, {time.time() - t0:.0f} с")
    if not tabs:
        raise SystemExit("ни одного буфера не загружено")
    shp = {p.shape[1:] for p in parts}
    if len(shp) != 1:
        raise SystemExit(f"размеры карт различаются между задачами: {sorted(shp)}")
    tab = {k: np.concatenate([t[k] for t in tabs], 0) for k in ROWKEYS}
    tab["CTX"] = np.concatenate([t["CTX"] for t in tabs], 0)
    tab["NAMES"] = [t["NAMES"][0] for t in tabs]
    tab["SUBDIRS"] = [t["SUBDIRS"][0] for t in tabs]
    maps = np.empty((len(tab["A"]),) + parts[0].shape[1:], dtype=dtype)   # без второй полной копии
    k = 0
    while parts:
        p = parts.pop(0)
        maps[k:k + len(p)] = p
        k += len(p)
    return tab, maps


def map_only_data(args, log=print):
    """--views / --procedural: карты без переходов буфера (только для предобучения). Строки — map_views.view_rows:
    цепочка (файл, вид) с нулевой картой на шаге 0; таблица минимальная: действие — следующее по цепочке (если
    известно), ошибка — из файла видов, исхода нет (y = 0), VKEY — номер состояния, общий у его видов."""
    import map_views as MV
    import offline_rl as O
    recs, names = [], []
    if getattr(args, "views", ""):
        subs = O.resolve_subdirs(args.subdirs)
        if len(subs) != 1:
            raise SystemExit(f"--views: нужна ровно одна папка задачи в --subdirs, а не {args.subdirs!r}")
        recs += MV.load_records(args.views, subs[0], log)
        names.append(f"views:{subs[0]}")
    if getattr(args, "procedural", 0):
        recs += MV.procedural_records(args.procedural, seed=args.seed, chain_len=args.procedural_len)
        names.append("procedural")
    r = MV.view_rows(recs, max_views=getattr(args, "views_max", 0))
    maps = r["maps"]
    n = len(maps)
    step, ch = r["STEP"], r["CH"]
    first = step == 0
    A = np.where(r["A"] >= 0, r["A"], 0).astype(np.int64)
    pa = np.where(first, -1, np.r_[-1, A[:-1]]).astype(np.int64)
    L = np.array([np.log10(e) if (e is not None and np.isfinite(e) and e > 0) else 0.0 for e in r["ERR"]])
    tab = dict(PDE=np.zeros(n, np.int64), FILE=r["FILE"], CH=ch, STEP=step, SPENT=np.zeros(n, np.int64), A=A, PA=pa,
               L=L, LN=L.copy(), LV=S.level6(maps), LVN=np.full((n, 6), np.nan), SPF=np.zeros((n, 3)),
               LAST=np.r_[first[1:], True], CTX=np.zeros((1, 7)), NAMES=["+".join(names)],
               SUBDIRS=["+".join(names)], VKEY=r["VKEY"], VIEW=r["VIEW"], MAPONLY=True)
    nv = np.bincount(r["VIEW"][~first]) if (~first).any() else np.zeros(1, np.int64)
    log(f"  карты без буферов ({'+'.join(names)}): состояний {r['n_states']}, строк {int((~first).sum())} "
        f"(по видам {nv.tolist()}), цепочек (файл, вид) {int(first.sum())}")
    return tab, maps.astype(np.float16 if args.map_dtype == "float16" else np.float32)


def synthetic_data(n=400, seed=0, n_pde=2, size=26, chain_len=5):
    """Крошечный набор с известным ответом (тесты, прогон без сети): y зависит от знака наклона
    карты вдоль оси x. Среднее карты от наклона не зависит, минимум зависит только от его модуля:
    level наклон не видит, shuffle теряет знак, энкодер на картах видит."""
    rng = np.random.default_rng(seed)
    n_ch = max(2 * n_pde, n // chain_len)
    n = n_ch * chain_len
    step = np.tile(np.arange(chain_len), n_ch)
    ch = np.repeat(np.arange(n_ch), chain_len)
    pde = ch % n_pde
    first = step == 0
    A = rng.integers(0, 27, n)
    pa = np.where(first, -1, np.r_[-1, A[:-1]])
    ramp = np.linspace(-1.0, 1.0, size)[None, None, :]
    lvl, g = rng.uniform(0.5, 3.0, n), rng.normal(0, 1, n)
    base = lvl[:, None, None] + 0.6 * g[:, None, None] * ramp + rng.normal(0, 0.02, (n, size, size))
    maps = np.stack([base, 0.6 * base, 0.4 * base, rng.normal(0, 0.1, (n, size, size))], 1)
    maps[first] = 0.0
    y = -0.15 * (S.FAM[A] == 1) + 0.5 * g * (~first) + rng.normal(0, 0.05, n)
    e0 = np.linspace(1.0, 0.5, n_pde)
    L, LN, spent, spf = np.zeros(n), np.zeros(n), np.zeros(n, np.int64), np.zeros((n, 3))
    for i in range(n):
        if first[i]:
            L[i] = np.log10(e0[pde[i]])
        else:
            L[i] = LN[i - 1]
            spent[i] = spent[i - 1] + S.EPOCHS[A[i - 1]]
            spf[i] = spf[i - 1]
            spf[i, S.FAM[A[i - 1]]] += S.EPOCHS[A[i - 1]]
        LN[i] = L[i] + y[i]
    ctx = np.array([[0.4 + 0.2 * (p % 2), 1 / 3, 1 / 3, 0.25 * (1 + p % 2), p % 2, 0.0, np.log10(e0[p])]
                    for p in range(n_pde)])
    maps = maps.astype(np.float32)
    tab = dict(PDE=pde.astype(np.int64), FILE=(ch // 10).astype(np.int64), CH=ch.astype(np.int64),
               STEP=step.astype(np.int64), SPENT=spent, A=A.astype(np.int64), PA=pa.astype(np.int64),
               L=L, LN=LN, LV=S.level6(maps), LVN=np.full((n, 6), np.nan), SPF=spf,
               LAST=np.r_[first[1:], True], CTX=ctx, NAMES=[f"syn_{p}" for p in range(n_pde)],
               SUBDIRS=[f"syn_{p}" for p in range(n_pde)])
    return tab, maps


def features(tab):
    """Скалярные признаки по группам: base (шаг, потрачено/7000 всего и по семействам, прошлое
    действие, описатель задачи и log10 E0), act (действие: 27 позиций + семейство, log шага,
    log эпох), err (log10 текущей ошибки и она же относительно E0), lv (шесть чисел уровня карт).
    Возвращает (N, F) и {группа: срез}."""
    a, pa = tab["A"], tab["PA"]
    pai = np.where(pa < 0, S.NONE, pa)
    ctx = tab["CTX"][tab["PDE"]]
    same_fam = (pa >= 0) & (S.FAM[np.maximum(pa, 0)] == S.FAM[a])
    one = np.zeros((len(a), 27))
    one[np.arange(len(a)), a] = 1.0
    groups = dict(
        base=[tab["STEP"][:, None] / KMAX, tab["SPENT"][:, None] / BUDGET, tab["SPF"] / BUDGET,
              S.ENC[pai], (pa < 0)[:, None], same_fam[:, None], (pa == a)[:, None], ctx],
        act=[one, S.ENC[a]],
        err=[tab["L"][:, None], (tab["L"] - ctx[:, 6])[:, None]],
        lv=[tab["LV"]])
    cols, parts, k = {}, [], 0
    for name, blocks in groups.items():
        x = np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], 1)
        cols[name] = slice(k, k + x.shape[1])
        k += x.shape[1]
        parts.append(x)
    return np.concatenate(parts, 1).astype(np.float32), cols


class Data:
    """Подготовленный набор: карты на --data-device (по умолчанию CPU, пачки уходят на GPU),
    скаляры и цель на устройстве обучения, статистики цепочек для бутстрепа."""

    def __init__(self, tab, maps, dev, data_device="cpu"):
        self.tab, self.dev = tab, dev
        self.n = len(tab["A"])
        self.names, self.subdirs = list(tab["NAMES"]), list(tab["SUBDIRS"])
        self.pde, self.step, self.A = tab["PDE"], tab["STEP"], tab["A"]
        ug, self.ch = np.unique(tab["CH"], return_inverse=True)
        self.n_ch = len(ug)
        self.L = np.asarray(tab["L"], dtype=np.float64)
        self.LN = np.asarray(tab["LN"], dtype=np.float64)
        self.y = self.LN - self.L
        prev = np.where(self.step > 0, np.arange(self.n) - 1, -1)
        has = prev >= 0
        if has.any() and (prev[has].min() < 0 or (self.ch[prev[has]] != self.ch[has]).any()
                          or (self.step[prev[has]] != self.step[has] - 1).any()):
            raise SystemExit("строки цепочек идут не подряд: предыдущий шаг не найден")
        self.prev = prev
        # следующий шаг той же цепочки (карты после действия: ssl-dyn); -1 у последнего шага цепочки
        nxt = np.arange(self.n) + 1
        ok = nxt < self.n
        ok[ok] = (self.ch[nxt[ok]] == self.ch[ok]) & (self.step[nxt[ok]] == self.step[ok] + 1)
        self.nxt = np.where(ok, nxt, -1)
        feat, self.cols = features(tab)
        self.feat_t = torch.as_tensor(feat, device=dev)
        self.y_t = torch.as_tensor(self.y, dtype=torch.float32, device=dev)
        mdev = torch.device("cpu")
        if dev.type == "cuda" and data_device in ("cuda", "auto"):
            free = torch.cuda.mem_get_info(dev)[0]
            if data_device == "cuda" or free > maps.nbytes + 6 * 2 ** 30:
                mdev = dev                       # auto: карты на GPU, если после них остаётся 6 ГБ
        self.maps = torch.as_tensor(maps).to(mdev)
        self.prev_t = torch.as_tensor(prev, device=mdev)
        self.C, self.H, self.W = (int(x) for x in maps.shape[1:])
        self.bucket = np.digitize(self.step, STEP_EDGES)
        # статистики цепочек: R² на бутстрепе цепочек считается из сумм (центрировано — float32 хватает)
        self.chain_pde = np.zeros(self.n_ch, dtype=np.int64)
        self.chain_pde[self.ch] = self.pde
        self.cn = np.bincount(self.ch, minlength=self.n_ch).astype(np.float64)
        self.sums = {}
        for key, v in (("y", self.y), ("next", self.LN)):
            c = v - v.mean()
            self.sums[key] = (np.bincount(self.ch, c, minlength=self.n_ch),
                              np.bincount(self.ch, c * c, minlength=self.n_ch))
        self._hgb_x = None
        # номер состояния у видов одной карты (--views); у обычных буферов виды не известны
        self.vkey = np.asarray(tab["VKEY"], dtype=np.int64) if "VKEY" in tab else None
        self.map_only = bool(tab.get("MAPONLY", False))
        sig = np.concatenate([self.A, self.ch, self.step, self.pde]).astype(np.int64).tobytes()
        if self.vkey is not None:
            sig = sig + self.vkey.tobytes()
        self.fingerprint = f"{self.n}-{self.n_ch}-{zlib.crc32(sig):08x}"

    def fetch(self, idx, prev=False):
        """Сырые карты пачки на устройстве обучения (float32) и, если надо, карты предыдущего шага
        цепочки (нули у первого шага — как стартовое состояние среды)."""
        i = idx.to(self.maps.device)
        x = self.maps[i].to(self.dev, non_blocking=True).float()
        if not prev:
            return x, None
        p = self.prev_t[i]
        xp = self.maps[p.clamp_min(0)].to(self.dev, non_blocking=True).float()
        return x, xp * (p >= 0).to(self.dev).view(-1, 1, 1, 1)

    def hgb_x(self):
        if self._hgb_x is None:
            self._hgb_x = S.table_features(self.tab, level=True)
        return self._hgb_x

    def sizes(self):
        out = {}
        zero = np.all(self.tab["LV"] == 0, 1)
        for p, name in enumerate(self.names):
            r = np.where(self.pde == p)[0]
            sub = torch.as_tensor(r[np.linspace(0, len(r) - 1, min(len(r), 2000)).astype(int)])
            m = self.maps[sub.to(self.maps.device)].float()
            out[name] = dict(n=int(len(r)), chains=int(len(np.unique(self.ch[r]))),
                             step0=float((self.step[r] == 0).mean()), zero_maps=float(zero[r].mean()),
                             zero_maps_after_start=int((zero[r] & (self.step[r] > 0)).sum()),
                             y_mean=float(self.y[r].mean()), y_std=float(self.y[r].std()),
                             l_median=float(np.median(self.L[r])), map_mean=float(m.mean()),
                             map_std=float(m.std()), map_absmax=float(m.abs().max()),
                             map_nonfinite=int((~torch.isfinite(m)).sum()))
        return out


def map_stats(D, rows, cap=20000, seed=0):
    """Среднее и разброс каждого канала по (подвыборке) строк rows: (1, C, 1, 1) на устройстве."""
    rows = np.asarray(rows)
    if len(rows) > cap:
        rows = np.random.default_rng(seed).choice(rows, cap, replace=False)
    m = D.maps[torch.as_tensor(np.sort(rows), device=D.maps.device)].float()
    mean = m.mean((0, 2, 3), keepdim=True).to(D.dev)
    std = m.std((0, 2, 3), keepdim=True).clamp_min(1e-6).to(D.dev)
    return mean, std


# --------------------------------------------------------------------------------------------
# Построение входа и аугментации (на устройстве обучения)
# --------------------------------------------------------------------------------------------

def shape_norm(x):
    """Форма без уровня: каждая карта каждого образца — нулевое среднее, единичный разброс
    (то же, что offline_rl.apply_state_mode(..., 'shape')); постоянная карта -> нули."""
    mu = x.mean((-2, -1), keepdim=True)
    sd = x.flatten(-2).std(-1, unbiased=False)[..., None, None]
    return torch.where(sd > 1e-6, (x - mu) / sd.clamp_min(1e-6), torch.zeros_like(x))


def shuffle_pixels(x, gen=None, indep=False):
    """Перестановка пикселей: одна на образец, общая для каналов (indep=True — своя у каждого
    канала, как режим shuffle в offline_rl.apply_state_mode)."""
    b, c, h, w = x.shape
    r = torch.rand(b, c if indep else 1, h * w, generator=gen, device=x.device)
    order = r.argsort(-1).expand(b, c, h * w)
    return x.flatten(2).gather(2, order).view(b, c, h, w)


def aug_d4(x, gen=None):
    """Случайный элемент группы D4 на образец (offline_rl.d4_apply)."""
    import offline_rl as O
    g = torch.randint(0, 8, (len(x),), generator=gen, device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = g == k
        if m.any():
            out[m] = O.d4_apply(x[m], k)
    return out


def aug_shift(x, gen=None, s=2):
    """Случайный сдвиг на образец до s пикселей по каждой оси, края продолжаются крайним значением."""
    b, _, h, w = x.shape
    if s <= 0:
        return x
    dy = torch.randint(-s, s + 1, (b,), generator=gen, device=x.device)
    dx = torch.randint(-s, s + 1, (b,), generator=gen, device=x.device)
    iy = (torch.arange(h, device=x.device)[None] + dy[:, None]).clamp(0, h - 1)
    ix = (torch.arange(w, device=x.device)[None] + dx[:, None]).clamp(0, w - 1)
    bi = torch.arange(b, device=x.device)[:, None, None]
    return x[bi, :, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)


def block_mask(b, h, w, ratio, gen=None, device=None, block=4):
    """Маска блоками block x block: (b, 1, h, w), True — скрыто."""
    gh, gw = math.ceil(h / block), math.ceil(w / block)
    m = torch.rand(b, 1, gh, gw, generator=gen, device=device) < ratio
    return m.repeat_interleave(block, 2).repeat_interleave(block, 3)[:, :, :h, :w]


def build_input(mode, x, xp, stats, gen=None, indep=False):
    """Вход энкодера из сырых карт: raw — нормировка по каналам; shape — форма без уровня;
    shuffle — перестановка пикселей; stack2 — текущие и предыдущие; delta — текущие и разность."""
    if mode == "shape":
        return shape_norm(x)
    mean, std = stats
    xn = (x - mean) / std
    if mode == "shuffle":
        return shuffle_pixels(xn, gen, indep)
    if mode == "stack2":
        return torch.cat([xn, (xp - mean) / std], 1)
    if mode == "delta":
        return torch.cat([xn, (x - xp) / std], 1)
    return xn


def in_channels(mode, c):
    return 2 * c if mode in ("stack2", "delta") else c


# --------------------------------------------------------------------------------------------
# Энкодеры и модель
# --------------------------------------------------------------------------------------------

def make_encoder(kind, in_ch=4, size=26):
    """conv — энкодер статьи (offline_rl.make_encoder('their') = RL/rl_utils/DQN_classes.ConvEncoder;
    для входа не из 4 каналов заменяется первая свёртка); convnext, cnn, resnet (малый ResNet под
    26x26) и vit (патч 2) — offline_rl.make_encoder: определения общие с тренером агента
    (offline_rl.py --encoder), чтобы предобученные здесь веса читались там (--encoder-init)."""
    import offline_rl as O
    if kind == "conv":
        enc = O.make_encoder("their")
        if in_ch != 4:
            old = enc.enc.backbone[0]
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, stride=old.stride, padding=old.padding)
            nn.init.kaiming_uniform_(new.weight, nonlinearity="relu")
            enc.enc.backbone[0] = new
        return enc
    if kind in ("convnext", "convnextw", "convnexthr", "cnn", "resnet", "resnetpos", "vit"):
        return O.make_encoder(kind, in_ch, size)
    raise ValueError(f"неизвестный энкодер {kind!r}: {ENCODERS}")


class Probe(nn.Module):
    """concat(признаки энкодера, вложение скаляров и действия) -> MLP -> число."""

    def __init__(self, enc, n_scal, width=256):
        super().__init__()
        self.enc = enc
        zdim = int(enc.out_dim) if enc is not None else 0
        self.zn = nn.LayerNorm(zdim) if zdim else None
        self.scal = nn.Sequential(nn.Linear(n_scal, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(zdim + 128, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, s, x=None, z=None):
        h = self.scal(s)
        if self.enc is not None:
            if z is None:
                z = self.enc(x)
            h = torch.cat([self.zn(z.float()), h.float()], 1)
        return self.head(h).squeeze(-1)


def n_params(m, trainable=False):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable))


# --------------------------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------------------------

def _amp(dev, args):
    return dev.type == "cuda" and not args.no_amp


def _autocast(on):
    return torch.autocast("cuda", dtype=torch.float16) if on else contextlib.nullcontext()


def _scaler(on):
    if not on:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _schedule(opt, steps):
    warm = max(1, int(0.05 * steps))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm
        else 0.5 * (1.0 + math.cos(math.pi * (k - warm) / max(1, steps - warm))))


def _step(loss, opt, sched, scaler, params):
    """Шаг оптимизатора; нечисловой лосс пропускается (возвращает False)."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        sched.step()
        return False
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt)
        scaler.update()
    sched.step()
    return True


def _gen(dev, seed):
    return torch.Generator(device=dev).manual_seed(int(seed))


def nt_xent(z, temp=0.2):
    """SimCLR: z — (2B, d), строки i и i+B — два вида одного образца."""
    z = F.normalize(z.float(), dim=1)
    b = len(z) // 2
    sim = z @ z.T / temp
    sim = sim.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def vicreg(z1, z2, inv=25.0, var=25.0, cov=1.0, eps=1e-4):
    """VICReg (Bardes et al., ICLR 2022), как в официальном коде (коэффициенты 25/25/1), делённый на 25, чтобы
    масштаб был сравним с MAE. Возвращает (лосс, части)."""
    n, d = z1.shape
    rep_l = F.mse_loss(z1, z2)
    std_l = cov_l = 0.0
    for z in (z1, z2):
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + eps)
        std_l = std_l + F.relu(1.0 - std).mean() / 2.0
        c = (z.T @ z) / max(n - 1, 1)
        cov_l = cov_l + (c - torch.diag(torch.diag(c))).pow(2).sum() / d
    loss = (inv * rep_l + var * std_l + cov * cov_l) / 25.0
    return loss, dict(inv=float(rep_l.item()), std=float(std_l.item()), cov=float(cov_l.item()))


def view_partners(D, rows):
    """Для ssl-mv: строки, сгруппированные по состоянию (VKEY). Возвращает (по позиции в rows: начало группы,
    размер, место внутри группы; строки в порядке групп) — второй вид выбирается из своей группы."""
    vk = D.vkey[rows]
    order = np.argsort(vk, kind="stable")
    svk = vk[order]
    start = np.r_[0, np.where(np.diff(svk) != 0)[0] + 1]
    size = np.diff(np.r_[start, len(svk)])
    gid = np.repeat(np.arange(len(start)), size)
    g_start, g_size = np.empty(len(rows), np.int64), np.empty(len(rows), np.int64)
    g_rank = np.empty(len(rows), np.int64)
    g_start[order], g_size[order] = start[gid], size[gid]
    g_rank[order] = np.arange(len(rows)) - start[gid]
    g_size[vk < 0] = 1
    return g_start, g_size, g_rank, rows[order]


class OutHead(nn.Module):
    """Голова исхода (ssl-out, ssl-maeout): как у Probe, но признаки энкодера подаются готовыми."""

    def __init__(self, zdim, n_scal, width=256):
        super().__init__()
        self.zn = nn.LayerNorm(zdim)
        self.scal = nn.Sequential(nn.Linear(n_scal, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(zdim + 128, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, z, s):
        return self.head(torch.cat([self.zn(z.float()), self.scal(s).float()], 1)).squeeze(-1)


class VaeHead(nn.Module):
    """β-VAE поверх энкодера (ssl-vae): среднее и лог-дисперсия латента из вектора признаков, декодер —
    MLP во всю карту (C x H x W)."""

    def __init__(self, zdim, dim, out):
        super().__init__()
        self.mu, self.lv = nn.Linear(zdim, dim), nn.Linear(zdim, dim)
        self.dec = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, out))

    def forward(self, z, gen=None):
        z = z.float()
        mu, lv = self.mu(z), self.lv(z).clamp(-10.0, 10.0)
        eps = torch.randn(mu.shape, generator=gen, device=mu.device)
        return self.dec(mu + eps * torch.exp(0.5 * lv)), mu, lv


class DynHead(nn.Module):
    """Самопредсказание с действием (ssl-dyn, SPR/SGI): остаточная модель перехода в пространстве признаков
    энкодера (действие — вложение), проекция и предсказатель как у BYOL, обратная динамика (действие по z, z')."""

    def __init__(self, zdim, n_act=27, proj=128):
        super().__init__()
        self.act = nn.Embedding(n_act, 64)
        self.trans = nn.Sequential(nn.Linear(zdim + 64, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, zdim))
        self.proj = nn.Sequential(nn.Linear(zdim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, proj))
        self.pred = nn.Sequential(nn.Linear(proj, 256), nn.ReLU(), nn.Linear(256, proj))
        self.inv = nn.Sequential(nn.Linear(2 * zdim, 256), nn.ReLU(), nn.Linear(256, n_act))

    def forward(self, z, a):
        h = z + self.trans(torch.cat([z, self.act(a).to(z.dtype)], 1))
        return self.pred(self.proj(h))


def pretrain_rows(kind, D):
    """Строки предобучения: карты не нулевые (шаг > 0); у ssl-dyn — ещё и следующий шаг цепочки есть,
    у ssl-out / ssl-maeout — исход конечен. Возвращает (строки статистик нормировки, строки обучения):
    статистики у всех видов считаются по одним и тем же строкам, что и у mae."""
    rows = np.where(D.step > 0)[0]
    if len(rows) < 8:
        rows = np.arange(D.n)
    if kind == "dyn":
        tr = rows[D.nxt[rows] >= 0]
        if len(tr) < 2:
            raise SystemExit("ssl-dyn: нет переходов внутри цепочек (нужны карты следующего шага)")
        return rows, tr
    if kind in SSL_LABELS:
        return rows, rows[np.isfinite(D.y[rows])]
    return rows, rows


def pretrain(kind, enc_name, D, args, dev, log=print):
    """Самообучение энкодера на ВСЕХ картах (стартовые нулевые карты пропущены). mae — блоки 4x4 скрыты
    (обнулены после нормировки), декодер из вектора признаков восстанавливает ФОРМУ карты (shape_norm) в
    скрытых пикселях: обобщённый вариант SimMIM/MAE, одинаковый для свёрточных энкодеров и ViT. con — SimCLR:
    два вида (D4, сдвиг, шум), NT-Xent. Замены MAE (волна 49): vae, dyn, out, maeout — см. описание модуля;
    dyn и out/maeout используют переходы буферов (действие, карты следующего шага, исход), метки mae и con
    не используют. Возвращает dict(state, stats, seconds, loss_first, loss_last, loss_parts, ...)."""
    t0 = time.time()
    torch.manual_seed(args.seed + 17)
    stat_rows, rows = pretrain_rows(kind, D)
    stats = map_stats(D, stat_rows, seed=args.seed)
    enc = make_encoder(enc_name, D.C, D.H).to(dev)
    zdim = int(enc.out_dim)
    tgt = Fn = yn = None
    if kind in ("out", "maeout"):
        cols = [i for g in ("base", "act") for i in range(D.cols[g].start, D.cols[g].stop)]
        rt = torch.as_tensor(rows, device=dev)
        Fs = D.feat_t[:, cols]
        sd = Fs[rt].std(0)
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        Fn = ((Fs - Fs[rt].mean(0)) / sd).clamp(-10.0, 10.0)
        yn = (D.y_t - D.y_t[rt].mean()) / D.y_t[rt].std().clamp_min(1e-6)
    if kind == "mae":
        aux = nn.Sequential(nn.Linear(enc.out_dim, 512), nn.GELU(), nn.Linear(512, D.C * D.H * D.W)).to(dev)
    elif kind == "con":
        aux = nn.Sequential(nn.Linear(enc.out_dim, 256), nn.ReLU(), nn.Linear(256, 64)).to(dev)
    elif kind == "vae":
        aux = VaeHead(zdim, args.vae_dim, D.C * D.H * D.W).to(dev)
    elif kind == "dyn":
        import copy
        aux = DynHead(zdim).to(dev)
        tgt = nn.ModuleList([copy.deepcopy(enc), copy.deepcopy(aux.proj)]).to(dev)   # EMA-цель (BYOL/SPR)
        src = nn.ModuleList([enc, aux.proj])
        for p in tgt.parameters():
            p.requires_grad_(False)
        nxt_t = torch.as_tensor(D.nxt, device=dev)
        act_t = torch.as_tensor(D.A, device=dev)
    elif kind == "out":
        aux = OutHead(zdim, Fn.shape[1]).to(dev)
    elif kind == "maeout":
        aux = nn.ModuleDict(dict(
            mae=nn.Sequential(nn.Linear(zdim, 512), nn.GELU(), nn.Linear(512, D.C * D.H * D.W)),
            out=OutHead(zdim, Fn.shape[1]))).to(dev)
    elif kind == "mv":
        if D.vkey is None:
            raise SystemExit("ssl-mv: нужны виды карт (--views): номер состояния у строк неизвестен")
        k_ = int(args.mv_dim)
        aux = nn.ModuleDict(dict(
            mae=nn.Sequential(nn.Linear(zdim, 512), nn.GELU(), nn.Linear(512, D.C * D.H * D.W)),
            proj=nn.Sequential(nn.Linear(zdim, k_), nn.BatchNorm1d(k_), nn.ReLU(), nn.Linear(k_, k_)))).to(dev)
        g_start, g_size, g_rank, g_rows = (torch.as_tensor(x, device=dev) for x in view_partners(D, rows))
        pos_of = torch.full((D.n,), -1, dtype=torch.long, device=dev)
        pos_of[torch.as_tensor(rows, device=dev)] = torch.arange(len(rows), device=dev)
        n_pair_rows = int((g_size > 1).sum())
        if not n_pair_rows:
            raise SystemExit("ssl-mv: ни у одного состояния нет второго вида (процедурные карты, --views-max 1?)")
        log(f"    mv: строк {len(rows)}, из них со вторым видом {n_pair_rows}, состояний с видами "
            f"{len(np.unique(D.vkey[rows][D.vkey[rows] >= 0]))}")
    else:
        raise ValueError(f"неизвестный вид предобучения {kind!r}")
    params = list(enc.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    bs = min(args.batch, len(rows))
    per = max(1, len(rows) // bs)
    # --ssl-steps: число шагов задано, эпохи подбираются под объём карт (малые наборы — больше эпох);
    # без флага — прежние --ssl-epochs
    n_ep = (max(1, math.ceil(args.ssl_steps / per)) if getattr(args, "ssl_steps", 0) else args.ssl_epochs)
    steps = n_ep * per
    sched = _schedule(opt, steps)
    amp = _amp(dev, args)
    scaler = _scaler(amp)
    gen = _gen(dev, args.seed + 23)
    rows_t = torch.as_tensor(rows, device=dev)
    first = last = None
    parts_last = {}
    log_every = 1 if n_ep <= 20 else n_ep // 10
    npix = D.C * D.H * D.W

    def mae_loss(out, raw, m):
        rec = out.float().view_as(raw)
        return (((rec - shape_norm(raw)) ** 2) * m).sum() / (m.sum().clamp_min(1) * D.C)

    for ep in range(n_ep):
        perm = rows_t[torch.randperm(len(rows_t), device=dev)]
        tot, k = 0.0, 0
        parts = {}
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            raw, _ = D.fetch(idx)
            x = build_input("raw", raw, None, stats)
            part = {}
            if kind == "mae":
                m = block_mask(len(x), D.H, D.W, args.mae_ratio, gen, dev)
                v = x.masked_fill(m, 0.0)
                with _autocast(amp):
                    out = aux(enc(v))
                loss = mae_loss(out, raw, m)     # лосс в float32, вне смешанной точности
            elif kind == "con":
                v = torch.cat([aug_shift(aug_d4(x, gen), gen, args.shift) for _ in range(2)])
                v = v + 0.05 * torch.randn(v.shape, generator=gen, device=dev)
                with _autocast(amp):
                    out = aux(enc(v))
                loss = nt_xent(out, args.con_temp)
            elif kind == "vae":
                with _autocast(amp):
                    z = enc(x)
                rec, mu, lv = aux(z, gen)
                r = ((rec.float().view_as(x) - x.clamp(-10.0, 10.0)) ** 2).sum((1, 2, 3)).mean()
                kl = (-0.5 * (1.0 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
                loss = (r + args.vae_beta * kl) / npix
                part = dict(rec=r.item() / npix, kl=kl.item())
            elif kind == "dyn":
                raw2, _ = D.fetch(nxt_t[idx])
                xx = aug_d4(torch.cat([x, build_input("raw", raw2, None, stats)], 1), gen)  # общий элемент D4
                x1, x2 = xx[:, :D.C], xx[:, D.C:]
                a = act_t[idx]
                with _autocast(amp):
                    zz = enc(torch.cat([x1, x2]))
                    z1, z2 = zz[:len(x1)], zz[len(x1):]
                    p = aux(z1, a)
                    logits = aux.inv(torch.cat([z1, z2], 1))
                    with torch.no_grad():
                        t = tgt[1](tgt[0](x2))
                spr = (2.0 - 2.0 * F.cosine_similarity(p.float(), t.float(), dim=1)).mean()
                inv = F.cross_entropy(logits.float(), a)
                loss = spr + args.dyn_inv * inv
                part = dict(spr=spr.item(), inv=inv.item(),
                            inv_acc=(logits.argmax(1) == a).float().mean().item())
            elif kind == "out":
                with _autocast(amp):
                    pr = aux(enc(x), Fn[idx])
                loss = F.huber_loss(pr.float(), yn[idx], delta=1.0)
            elif kind == "mv":
                m = block_mask(len(x), D.H, D.W, args.mae_ratio, gen, dev)
                v = x.masked_fill(m, 0.0)
                p_ = pos_of[idx]
                sz = g_size[p_]
                pair = sz > 1
                n_p = int(pair.sum())
                if n_p >= 4:
                    # второй вид того же состояния: сдвиг на 1..size-1 внутри группы (сам себя не выбирает)
                    off = torch.randint(0, 1 << 30, (len(idx),), generator=gen, device=dev) % (sz - 1).clamp_min(1) + 1
                    other = g_rows[g_start[p_] + (g_rank[p_] + off) % sz][pair]
                    raw2, _ = D.fetch(other)
                    x2 = build_input("raw", raw2, None, stats)
                    with _autocast(amp):
                        zz = enc(torch.cat([v, x[pair], x2]))
                        out = aux["mae"](zz[:len(x)])
                        zp = aux["proj"](zz[len(x):])
                    lv, lp = vicreg(zp[:n_p].float(), zp[n_p:].float())
                else:
                    with _autocast(amp):
                        out = aux["mae"](enc(v))
                    lv, lp = torch.zeros((), device=dev), dict(inv=0.0, std=0.0, cov=0.0)
                lm = mae_loss(out, raw, m)
                loss = lm + args.mv_w * lv
                part = dict(mae=lm.item(), vic=float(lv.item()), inv=lp["inv"], pairs=n_p / len(idx))
            else:                                # maeout
                m = block_mask(len(x), D.H, D.W, args.mae_ratio, gen, dev)
                v = x.masked_fill(m, 0.0)
                with _autocast(amp):
                    zz = enc(torch.cat([v, x]))
                    out = aux["mae"](zz[:len(x)])
                    pr = aux["out"](zz[len(x):], Fn[idx])
                lm = mae_loss(out, raw, m)
                lo = F.huber_loss(pr.float(), yn[idx], delta=1.0)
                loss = lm + args.out_w * lo
                part = dict(mae=lm.item(), out=lo.item())
            if _step(loss, opt, sched, scaler, params):
                tot, k = tot + loss.item(), k + 1
                for key, val in part.items():
                    parts[key] = parts.get(key, 0.0) + val
                if tgt is not None:              # EMA-цель: веса — скользящее среднее, буферы BatchNorm — копия
                    with torch.no_grad():
                        for pt, po in zip(tgt.parameters(), src.parameters()):
                            pt.mul_(args.dyn_ema).add_(po.detach(), alpha=1.0 - args.dyn_ema)
                        for bt, bo in zip(tgt.buffers(), src.buffers()):
                            bt.copy_(bo)
        last = tot / max(1, k)
        first = last if first is None else first
        parts_last = {key: val / max(1, k) for key, val in parts.items()}
        if (ep + 1) % log_every == 0 or ep + 1 == n_ep:
            log(f"    {kind}/{enc_name} эпоха {ep + 1}/{n_ep}: лосс {last:.4f}"
                + "".join(f", {key} {val:.4f}" for key, val in parts_last.items()))
    enc.eval()
    state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    return dict(state=state, stats=stats, seconds=time.time() - t0, loss_first=first, loss_last=last,
                epochs=int(n_ep), steps=int(steps), n_maps=int(len(rows)), loss_parts=parts_last)


def ssl_args(kind, args):
    """Параметры предобучения вида kind (в файл энкодера и отчёт)."""
    keys = dict(mae=("mae_ratio",), con=("con_temp", "shift"), vae=("vae_beta", "vae_dim"),
                dyn=("dyn_ema", "dyn_inv"), out=(), maeout=("mae_ratio", "out_w"),
                mv=("mae_ratio", "mv_w", "mv_dim")).get(kind, ())
    return {k: getattr(args, k) for k in keys}


def save_encoder(pre, kind, enc_name, D, args, log=print):
    """--save-encoder: веса предобученного энкодера в <out-dir>/cvprobe_<tag>_enc_<mae|con>_<энкодер>.pt
    и в HF под ENC_DIR. Формат читает offline_rl.py --encoder-init: state_dict энкодера, его вид и
    статистики нормировки карт (1, C, 1, 1), с которыми шло предобучение."""
    path = os.path.join(args.out_dir, f"cvprobe_{args.tag}_enc_{kind}_{enc_name}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mean, std = (t.detach().float().cpu().numpy() for t in pre["stats"])
    torch.save(dict(kind="ssl_encoder", encoder=enc_name, ssl=kind, in_ch=D.C, size=D.H,
                    state_dict=pre["state"], mean=mean, std=std, tag=args.tag,
                    ssl_epochs=int(pre.get("epochs", args.ssl_epochs)), ssl_steps=pre.get("steps"),
                    n_maps=pre.get("n_maps"), loss_first=pre["loss_first"], loss_last=pre["loss_last"],
                    loss_parts=pre.get("loss_parts") or {}, ssl_args=ssl_args(kind, args),
                    pdes=list(D.names), fingerprint=D.fingerprint), path)
    up = False
    if not args.no_upload:
        import agent_combo as AC
        up = AC.hf_upload(path, f"{ENC_DIR}/{os.path.basename(path)}", f"rl_arch cv_probe encoder {args.tag}", tries=3)
    log(f"  энкодер {kind}/{enc_name} сохранён: {path}"
        + (f" (в HF: {ENC_DIR}/{os.path.basename(path)})" if up else " (только локально)"))
    return path


def get_pre(sp, D, args, dev, cache, log=print):
    key = (sp["ssl"], sp["encoder"])
    if key not in cache["pre"]:
        log(f"  предобучение {key[0]} энкодера {key[1]} на всех картах")
        cache["pre"][key] = pretrain(key[0], key[1], D, args, dev, log)
        if getattr(args, "save_encoder", False):
            cache["pre"][key]["file"] = save_encoder(cache["pre"][key], key[0], key[1], D, args, log)
    return cache["pre"][key]


@torch.no_grad()
def encode_all(enc, D, stats, dev, amp, bs=4096):
    """Признаки замороженного энкодера для всех строк (режим raw), на устройстве обучения."""
    enc.eval()
    out = []
    for i in range(0, D.n, bs):
        raw, _ = D.fetch(torch.arange(i, min(D.n, i + bs)))
        with _autocast(amp):
            out.append(enc(build_input("raw", raw, None, stats)).float())
    return torch.cat(out)


def spec_cols(sp, D):
    idx = []
    for g in ("base", "act") + (("err",) if sp.get("err") else ()) + (("lv",) if sp.get("lv") else ()):
        idx += list(range(D.cols[g].start, D.cols[g].stop))
    return idx


def fit_predict(sp, D, tr, te, args, dev, seed=0, pre=None, cache=None):
    """Одна модель: обучение на строках tr, предсказание y на строках te.
    Один оптимизатор и расписание у всех конфигов: AdamW, прогрев 5% и косинус, Huber по
    стандартизованной цели. Возвращает (pred, info)."""
    if sp.get("hgb"):
        t0 = time.time()
        X = D.hgb_x()
        h = S._hgb(seed, *((60, 0.15) if args.smoke else (300, 0.06)), 31, 1.0)
        h.fit(X[tr], D.y[tr])
        sub = tr[::max(1, len(tr) // 4000)]
        return h.predict(X[te]), dict(train_s=time.time() - t0, params=0, params_trainable=0, skipped=0,
                                      train_r2=S.r2(D.y[sub], h.predict(X[sub])))
    t0 = time.time()
    torch.manual_seed(seed)
    amp = _amp(dev, args)
    tr_t, te_t = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    Fs = D.feat_t[:, spec_cols(sp, D)]
    sd = Fs[tr_t].std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)     # постоянный на обучении признак не раздувается
    Fn = ((Fs - Fs[tr_t].mean(0)) / sd).clamp(-10.0, 10.0)   # на отложенной задаче (lopo) он может отличаться
    ym, ys = D.y_t[tr_t].mean(), D.y_t[tr_t].std().clamp_min(1e-6)
    mode, aug, frozen = sp.get("maps"), sp.get("aug"), bool(sp.get("frozen"))
    enc = stats = z_all = None
    if mode:
        enc = make_encoder(sp["encoder"], in_channels(mode, D.C), D.H)
        if pre is not None:
            enc.load_state_dict(pre["state"])
            stats = pre["stats"]                 # тот же вход, что при предобучении
        else:
            stats = map_stats(D, tr, seed=seed)
    model = Probe(enc, Fn.shape[1]).to(dev)
    if frozen:
        for p in model.enc.parameters():
            p.requires_grad_(False)
        model.enc.eval()
        if not aug:                              # признаки не зависят от фолда: считаются один раз
            zc = cache.setdefault("z", {}) if cache is not None else {}
            key = (sp.get("ssl"), sp["encoder"])
            if key not in zc:
                zc[key] = encode_all(model.enc, D, stats, dev, amp)
            z_all = zc[key]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    bs = max(2, min(args.batch, len(tr_t)))
    sched = _schedule(opt, args.epochs * math.ceil(len(tr_t) / bs))
    scaler = _scaler(amp)
    gen = _gen(dev, seed + 1)
    need_prev = mode in ("stack2", "delta")

    def forward(idx, train):
        if not mode:
            return model(Fn[idx])
        if z_all is not None:
            return model(Fn[idx], z=z_all[idx])
        x, xp = D.fetch(idx, prev=need_prev)
        x = build_input(mode, x, xp, stats, gen, indep=args.shuffle_indep)
        if train and aug == "d4":
            x = aug_d4(x, gen)
        elif train and aug == "shift":
            x = aug_shift(x, gen, args.shift)
        return model(Fn[idx], x=x)

    skipped = 0
    for _ in range(args.epochs):
        model.train()
        if frozen:
            model.enc.eval()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            with _autocast(amp):
                pred = forward(idx, True)
            loss = F.huber_loss(pred.float(), (D.y_t[idx] - ym) / ys, delta=1.0)
            skipped += not _step(loss, opt, sched, scaler, params)
    model.eval()
    gen = _gen(dev, seed + 2)                    # перестановка shuffle на оценке воспроизводима

    def predict(rows_t):
        out = []
        with torch.no_grad():
            for i in range(0, len(rows_t), 4 * args.batch):
                with _autocast(amp):
                    out.append(forward(rows_t[i:i + 4 * args.batch], False).float())
        return (torch.cat(out) * ys + ym).double().cpu().numpy()

    pred = predict(te_t)
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        pred = np.nan_to_num(pred, nan=float(ym), posinf=float(ym), neginf=float(ym))
    secs = time.time() - t0
    sub = tr[::max(1, len(tr) // 4000)]          # R² на обучении: виден разрыв обобщения
    info = dict(train_s=secs, params=n_params(model), params_trainable=n_params(model, True),
                skipped=int(skipped) + bad,
                train_r2=S.r2(D.y[sub], np.nan_to_num(predict(torch.as_tensor(sub, device=dev)))))
    return pred, info


# --------------------------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------------------------

def rank_within(y, p, groups, min_n=20):
    """Средний по группам (с весом числа строк) Spearman предсказанного и реального улучшения."""
    order = np.argsort(groups, kind="stable")
    cut = np.r_[0, np.where(np.diff(groups[order]) != 0)[0] + 1, len(order)]
    num = den = used = 0.0
    for s, e in zip(cut[:-1], cut[1:]):
        if e - s < min_n:
            continue
        r = order[s:e]
        rho = S.spearman(p[r], y[r])
        if np.isfinite(rho):
            num, den, used = num + rho * (e - s), den + (e - s), used + 1
    return dict(rho=(num / den if den else float("nan")), groups=int(used), rows=int(den))


def metrics(D, rows, pred, per_pde=True):
    y, l, ln = D.y[rows], D.L[rows], D.LN[rows]
    pde, bucket, a, step = D.pde[rows], D.bucket[rows], D.A[rows], D.step[rows]
    later = step >= 1
    out = dict(n=int(len(rows)), r2_y=S.r2(y, pred), spearman_y=S.spearman(pred, y),
               r2_next=S.r2(ln, l + pred), mae_y=float(np.abs(y - pred).mean()),
               r2_y_after_start=(S.r2(y[later], pred[later]) if later.sum() > 2 else float("nan")),
               rank_within=rank_within(y, pred, pde * 4 + bucket)["rho"],
               rank_within_action=rank_within(y, pred, (pde * 4 + bucket) * 27 + a, min_n=8)["rho"])
    if per_pde:
        pp = {}
        for p in np.unique(pde):
            m = pde == p
            pp[D.names[p]] = dict(n=int(m.sum()), r2_y=S.r2(y[m], pred[m]),
                                  spearman_y=S.spearman(pred[m], y[m]), r2_next=S.r2(ln[m], l[m] + pred[m]))
        out["per_pde"] = pp
        out["r2_y_pde_median"] = float(np.nanmedian([v["r2_y"] for v in pp.values()]))
    return out


def chain_sse(D, rows, pred):
    """Сумма квадратов ошибок по цепочкам (длина n_ch; ноль у цепочек вне rows)."""
    return np.bincount(D.ch[rows], (D.y[rows] - pred) ** 2, minlength=D.n_ch)


class Boot:
    """Бутстреп по цепочкам (пуассоновские веса) для разности R² двух моделей на одних строках:
    R²_a - R²_b = (SSE_b - SSE_a) / SST, всё из сумм по цепочкам."""

    def __init__(self, D, n=1000, seed=0):
        self.D, self.n, self.seed, self._w = D, n, seed, None

    def w(self):
        if self._w is None:
            self._w = np.random.default_rng(self.seed + 991).poisson(1.0, (self.n, self.D.n_ch)).astype(np.float32)
        return self._w

    def _terms(self, sse_a, sse_b, chains, target):
        s1, s2 = self.D.sums[target]
        v = [self.D.cn, s1, s2, np.asarray(sse_b) - np.asarray(sse_a)]
        if chains is not None:
            v = [x[chains] for x in v]
        return [x.astype(np.float32) for x in v]

    def samples(self, sse_a, sse_b, chains=None, target="y"):
        n, s1, s2, d = self._terms(sse_a, sse_b, chains, target)
        w = self.w() if chains is None else self.w()[:, chains]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (w @ d) / (w @ s2 - (w @ s1) ** 2 / (w @ n))

    def gain(self, sse_a, sse_b, chains=None, target="y"):
        """[оценка, 5%, 95%] разности R²(a) - R²(b) на строках цепочек chains (None — все)."""
        n, s1, s2, d = (x.astype(np.float64) for x in self._terms(sse_a, sse_b, chains, target))
        sst = s2.sum() - s1.sum() ** 2 / max(n.sum(), 1.0)
        if not sst > 0:
            return [float("nan")] * 3
        bs = self.samples(sse_a, sse_b, chains, target)
        lo, hi = np.nanpercentile(bs, [5, 95]) if np.isfinite(bs).any() else (float("nan"),) * 2
        return [float(d.sum() / sst), float(lo), float(hi)]


# --------------------------------------------------------------------------------------------
# Конфиги, отчёт, продолжение
# --------------------------------------------------------------------------------------------

def parse_config(name):
    """'shape-vit' -> ('shape', 'vit'); 'maps' -> ('maps', None)."""
    for b in sorted(BASES, key=len, reverse=True):
        if name == b:
            return b, None
        if name.startswith(b + "-") and name[len(b) + 1:] in ENCODERS:
            return b, name[len(b) + 1:]
    raise SystemExit(f"неизвестный конфиг {name!r}: основы {sorted(BASES)}, энкодеры {ENCODERS}, "
                     f"lopo:<конфиг>|lopo:best")


def expand_configs(spec):
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in ("all", "default"):
            out += DEFAULT_CONFIGS
        elif tok == "maps":
            out += [f"maps-{e}" for e in DEFAULT_ENCODERS]
        else:
            out.append(tok)
    names = list(dict.fromkeys(out))
    for n in names:
        if n.startswith("lopo:"):
            if n != "lopo:best":
                parse_config(n[5:])
        else:
            parse_config(n)
    return names


def spec_of(name, encoder):
    base, enc = parse_config(name)
    sp = dict(BASES[base], base=base)
    if sp.get("maps"):
        sp["encoder"] = enc or encoder
    return sp


def select_encoder(report, args):
    """Энкодер для конфигов без явного суффикса: --encoder, иначе лучший по R² фолда 0 среди
    maps-*; из уложившихся в --select-tol от лучшего берётся самый дешёвый по времени обучения."""
    if args.encoder != "auto":
        return dict(name=args.encoder, by="--encoder")
    cand = {}
    for name, e in report["configs"].items():
        if e.get("kind") == "cv" and e.get("base") == "maps" and e.get("fold_r2"):
            r = e["fold_r2"][0]
            if np.isfinite(r) and (e["encoder"] not in cand or r > cand[e["encoder"]][0]):
                cand[e["encoder"]] = (float(r), float(e.get("train_s", 0.0)))
    if not cand:
        return dict(name=FALLBACK_ENCODER, by="нет посчитанных maps-*: энкодер по умолчанию")
    top = max(v[0] for v in cand.values())
    near = {k: v for k, v in cand.items() if v[0] >= top - args.select_tol}
    name = min(near, key=lambda k: near[k][1])
    return dict(name=name, by=f"фолд 0 maps-*, допуск {args.select_tol}", fold0_r2={k: v[0] for k, v in cand.items()},
                train_s={k: v[1] for k, v in cand.items()})


def best_maps_config(report):
    """Лучший конфиг с картами в распределении: по выигрышу над level, иначе по R²."""
    best, score = None, -np.inf
    for name, e in report["configs"].items():
        if e.get("kind") != "cv" or e.get("base") in NOT_MAPS or e.get("ssl_saw_labels"):
            continue
        g = (e.get("gain_over_level") or {}).get("r2_y")
        s = g[0] if g and np.isfinite(g[0]) else e.get("r2_y", float("nan")) - 10.0
        if np.isfinite(s) and s > score:
            best, score = name, s
    return best


def _key(*parts):
    return "__".join("".join(c if c.isalnum() else "_" for c in p) for p in parts)


def paths(args):
    base = os.path.join(args.out_dir, f"cvprobe_{args.tag}")
    return base + ".json", base + "_chains.npz"


def upload(path, args, tries=3):
    """Заливка в HF по образцу agent_combo.hf_upload (без токена — ничего не делает)."""
    if args.no_upload:
        return False
    import agent_combo as AC
    return AC.hf_upload(path, f"{HF_DIR}/{os.path.basename(path)}", f"rl_arch cv_probe {args.tag}", tries=tries)


def save_report(report, args, t0, do_upload=True):
    path, _ = paths(args)
    report["elapsed_s"] = round(report.get("elapsed_before_s", 0.0) + time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)
    if do_upload:
        upload(path, args)
    return path


def save_sse(sse, args, do_upload=False):
    _, path = paths(args)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:      # через открытый файл: np.savez сам дописал бы .npz к имени
        np.savez_compressed(f, **{k: np.asarray(v, dtype=np.float32) for k, v in sse.items()})
    if do_upload:
        upload(path, args)


def load_prev(args, log=print):
    """Прежний отчёт и суммы по цепочкам: локальные файлы, иначе HF. (None, {}) — начинать заново."""
    jpath, npath = paths(args)
    found = {}
    for kind, p in (("json", jpath), ("npz", npath)):
        if os.path.exists(p):
            found[kind] = p
            continue
        try:
            import agent_combo as AC
            found[kind] = AC.hf_fetch(f"{HF_DIR}/{os.path.basename(p)}")
        except Exception as e:      # noqa: BLE001 — файла нет или сеть недоступна
            log(f"  продолжение: {os.path.basename(p)} не найден ({type(e).__name__})")
    if "json" not in found:
        return None, {}
    with open(found["json"]) as f:
        report = json.load(f)
    sse = {}
    if "npz" in found:
        try:
            with np.load(found["npz"]) as z:
                sse = {k: z[k].astype(np.float64) for k in z.files}
        except Exception as e:      # noqa: BLE001
            log(f"  продолжение: суммы по цепочкам не читаются ({type(e).__name__})")
    return report, sse


def kill_pattern(args):
    """Выключатель задач (killswitch.py): запись списка killed.txt, под которую попал --tag, иначе
    None. Работает только при запуске через main(): при старте и перед каждым конфигом."""
    if not getattr(args, "_kill", False):
        return None
    try:
        from killswitch import killed
    except ImportError:
        return None
    return killed(args.tag, remote=True)


def is_done(report, name):
    e = report["configs"].get(name)
    return bool(e) and (e.get("kind") != "lopo" or bool(e.get("done")))


# --------------------------------------------------------------------------------------------
# Прогоны
# --------------------------------------------------------------------------------------------

def run_cv(name, sp, D, fold_row, args, dev, cache, log=print):
    """Конфиг на GroupKFold по цепочкам: предсказания на отложенных фолдах для всех строк."""
    oof = np.full(D.n, np.nan)
    pre = get_pre(sp, D, args, dev, cache, log) if sp.get("ssl") else None
    fold_r2, train_r2, secs, info, skipped = [], [], 0.0, {}, 0
    for f in range(int(fold_row.max()) + 1):
        te, tr = np.where(fold_row == f)[0], np.where(fold_row != f)[0]
        pred, info = fit_predict(sp, D, tr, te, args, dev, seed=args.seed * 1000 + f, pre=pre, cache=cache)
        oof[te] = pred
        secs += info["train_s"]
        skipped += info["skipped"]
        fold_r2.append(S.r2(D.y[te], pred))
        train_r2.append(info["train_r2"])
        log(f"  {name} фолд {f}: R² {fold_r2[-1]:.4f} (на обучении {train_r2[-1]:.4f}), {info['train_s']:.0f} с")
    ent = dict(kind="cv", base=sp["base"], encoder=sp.get("encoder"), fold_r2=fold_r2,
               fold_train_r2=train_r2, train_s=round(secs, 1),
               pretrain_s=round(pre["seconds"], 1) if pre else 0.0,
               ssl_loss=[pre["loss_first"], pre["loss_last"]] if pre else None,
               params=info["params"], params_trainable=info["params_trainable"], skipped_steps=skipped,
               epochs=args.epochs, folds=len(fold_r2))
    if pre and pre.get("file"):
        ent["encoder_file"] = f"{ENC_DIR}/{os.path.basename(pre['file'])}"
    if sp.get("ssl") in SSL_LABELS:
        ent["ssl_saw_labels"] = True             # предобучение видело y отложенных фолдов: R² оптимистичен
    ent.update(metrics(D, np.arange(D.n), oof))
    return ent, oof


def lopo_ids(D, spec, log=print):
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in D.names:
            out.append(D.names.index(tok))
        elif tok in D.subdirs:
            out.append(D.subdirs.index(tok))
        else:
            log(f"  lopo: задачи {tok!r} нет в данных, пропущена")
    return list(dict.fromkeys(out))


def run_lopo(name, D, ids, report, sse, args, dev, cache, boot, encoder, save, time_left, log=print):
    """«Исключить УрЧП»: обучение без задачи, оценка на ней. Отчёт пишется после каждой задачи.
    Возвращает False, если остановлен пределом времени."""
    ent = report["configs"].get(name) or dict(kind="lopo", pdes={}, done=False)
    if "config" not in ent:
        ent["config"] = best_maps_config(report) if name == "lopo:best" else name[5:]
    if ent["config"] is None:
        log(f"  {name}: нет ни одного посчитанного конфига с картами, пропущен")
        return True
    src = report["configs"].get(ent["config"], {})
    sp = spec_of(ent["config"], src.get("encoder") or encoder)
    ent["encoder"] = sp.get("encoder")
    ent["ssl_saw_heldout"] = bool(sp.get("ssl"))      # предобучение шло на картах всех задач
    pre = get_pre(sp, D, args, dev, cache, log) if sp.get("ssl") else None
    for p in ids:
        nm = D.names[p]
        if nm in ent["pdes"]:
            continue
        if not time_left():
            return False
        te, tr = np.where(D.pde == p)[0], np.where(D.pde != p)[0]
        if len(te) < 3 or len(tr) < 3:
            continue
        pred, info = fit_predict(sp, D, tr, te, args, dev, seed=args.seed * 1000 + 100 + p, pre=pre, cache=cache)
        ent["pdes"][nm] = dict(train_s=round(info["train_s"], 1), train_r2=info["train_r2"],
                               **metrics(D, te, pred, per_pde=False))
        sse[_key("lopo", name, nm)] = chain_sse(D, te, pred)
        report["configs"][name] = ent
        lopo_gains(report, sse, D, boot)
        log(f"  {name} [{ent['config']}] без {nm}: R² {ent['pdes'][nm]['r2_y']:.4f}, {info['train_s']:.0f} с")
        save()
    ent["done"] = True
    report["configs"][name] = ent
    lopo_gains(report, sse, D, boot)
    return True


def cv_gains(report, sse, D, boot, only=None):
    """Выигрыш над level (и над level-hgb) с интервалом по цепочкам; only — имена для пересчёта."""
    cfgs = report["configs"]
    for ref, field in (("level", "gain_over_level"), ("level-hgb", "gain_over_level_hgb")):
        kb = _key("cv", ref)
        if ref not in cfgs or kb not in sse:
            continue
        for name, e in cfgs.items():
            ka = _key("cv", name)
            if e.get("kind") != "cv" or ka not in sse or (only is not None and name not in only):
                continue
            g = dict(r2_y=boot.gain(sse[ka], sse[kb]), r2_next=boot.gain(sse[ka], sse[kb], target="next"))
            for m in ("spearman_y", "rank_within", "rank_within_action"):
                g[m] = float(e[m] - cfgs[ref][m])
            if ref == "level":
                g["per_pde"] = {nm: boot.gain(sse[ka], sse[kb], chains=D.chain_pde == p)
                                for p, nm in enumerate(D.names)}
            e[field] = g


def lopo_gains(report, sse, D, boot):
    """У каждого lopo:<конфиг>, кроме lopo:level: выигрыш над lopo:level по отложенной задаче и
    средний по задачам (интервал — по цепочкам; без сумм по цепочкам — только разность R²)."""
    cfgs = report["configs"]
    ref = cfgs.get("lopo:level")
    for name, e in cfgs.items():
        if e.get("kind") != "lopo" or name == "lopo:level" or not ref:
            continue
        per, samp = {}, []
        for nm, r in e["pdes"].items():
            if nm not in ref["pdes"]:
                continue
            ka, kb = _key("lopo", name, nm), _key("lopo", "lopo:level", nm)
            if ka in sse and kb in sse:
                ch = D.chain_pde == D.names.index(nm)
                per[nm] = boot.gain(sse[ka], sse[kb], chains=ch)
                samp.append(boot.samples(sse[ka], sse[kb], chains=ch))
            else:
                per[nm] = [float(r["r2_y"] - ref["pdes"][nm]["r2_y"]), None, None]
        e["gain_over_level"] = per
        if per:
            mean = float(np.mean([v[0] for v in per.values()]))
            lo = hi = None
            if len(samp) == len(per):
                lo, hi = (float(x) for x in np.nanpercentile(np.mean(samp, 0), [5, 95]))
            e["mean_gain"] = [mean, lo, hi]
            e["n_positive"] = int(sum(v[0] > 0 for v in per.values()))


def summarize(report, sse, D, boot, args):
    cfgs = report["configs"]
    rows = []
    for name, e in cfgs.items():
        if e.get("kind") != "cv":
            continue
        g = (e.get("gain_over_level") or {}).get("r2_y") or [None] * 3
        if name == "level":
            g = [0.0, 0.0, 0.0]
        rows.append(dict(config=name, encoder=e.get("encoder"), r2_y=e["r2_y"], gain=g[0], lo=g[1], hi=g[2],
                         r2_next=e["r2_next"], spearman_y=e["spearman_y"], rank_within=e["rank_within"],
                         rank_within_action=e["rank_within_action"],
                         train_s=round(e["train_s"] + e.get("pretrain_s", 0.0), 1), params=e["params"]))
    rows.sort(key=lambda r: (r["gain"] is None or not np.isfinite(r["gain"]),
                             -(r["gain"] if r["gain"] is not None and np.isfinite(r["gain"]) else 0.0)))
    best = best_maps_config(report)
    out = dict(table=rows, best_maps_config=best, selected_encoder=report.get("selected_encoder"),
               rule=(f"карты добавляют информацию, если лучший конфиг с картами выигрывает у level (и у "
                     f"level-hgb, если он посчитан) >= {args.min_gain} R² по y с 90%-интервалом выше нуля "
                     f"И средний по отложенным задачам выигрыш lopo:best над lopo:level >= 0"),
               in_dist=None, lopo=None, spatial=None, maps_add_information=None)
    if best is None or "level" not in cfgs:
        out["verdict"] = "неполно: нужен level и хотя бы один конфиг с картами"
        return out
    e = cfgs[best]
    g = (e.get("gain_over_level") or {}).get("r2_y")
    gh = (e.get("gain_over_level_hgb") or {}).get("r2_y")
    if not g:
        out["verdict"] = "неполно: нет выигрыша над level (нет сумм по цепочкам)"
        return out
    ok = bool(g[0] >= args.min_gain and g[1] is not None and g[1] > 0)
    ok_h = None if not gh else bool(gh[0] >= args.min_gain and gh[1] > 0)
    out["in_dist"] = dict(config=best, gain=g, gain_over_hgb=gh, beats_level=ok, beats_level_hgb=ok_h)
    shuf = [n for n, c in cfgs.items() if c.get("kind") == "cv" and c.get("base") == "shuffle"
            and _key("cv", n) in sse]
    shuf.sort(key=lambda n: cfgs[n].get("encoder") != e.get("encoder"))     # сначала с тем же энкодером
    if shuf and _key("cv", best) in sse:
        sg = boot.gain(sse[_key("cv", best)], sse[_key("cv", shuf[0])])
        out["spatial"] = dict(shuffle_config=shuf[0], gain_over_shuffle=sg, structure_matters=bool(sg[1] > 0))
    lb = cfgs.get("lopo:best")
    if lb and lb.get("mean_gain") and lb.get("done") and cfgs.get("lopo:level", {}).get("done"):
        mg = lb["mean_gain"]
        out["lopo"] = dict(config=lb["config"], mean_gain=mg, per_pde=lb["gain_over_level"],
                           n_positive=lb.get("n_positive"), not_worse=bool(mg[0] >= 0),
                           significantly_worse=bool(mg[2] is not None and mg[2] < 0),
                           same_as_best=bool(lb["config"] == best))
    ind = ok and ok_h is not False
    txt = f"лучший конфиг с картами {best}: выигрыш над level {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] R²"
    if gh:
        txt += f", над level-hgb {gh[0]:+.4f} [{gh[1]:+.4f}, {gh[2]:+.4f}]"
    if out["spatial"]:
        sg = out["spatial"]["gain_over_shuffle"]
        txt += f"; над shuffle {sg[0]:+.4f} [{sg[1]:+.4f}, {sg[2]:+.4f}]"
    if not ind:
        out["maps_add_information"] = False
        out["verdict"] = "карты НЕ добавляют информации сверх уровня — " + txt
    elif out["lopo"] is None:
        out["verdict"] = "в распределении карты добавляют информацию, lopo не посчитан (вердикт неполный) — " + txt
    else:
        mg = out["lopo"]["mean_gain"]
        out["maps_add_information"] = bool(out["lopo"]["not_worse"])
        txt += f"; lopo [{out['lopo']['config']}] средний выигрыш {mg[0]:+.4f}"
        if mg[1] is not None:
            txt += f" [{mg[1]:+.4f}, {mg[2]:+.4f}]"
        out["verdict"] = ("карты добавляют информацию — " if out["maps_add_information"]
                          else "в распределении карты добавляют, при переносе на новую задачу хуже level — ") + txt
    return out


def print_summary(summary, log=print):
    def f(x, w=8, p=4, sign=False):
        if x is None or not np.isfinite(x):
            return " " * (w - 1) + "-"
        return f"{x:{'+' if sign else ''}{w}.{p}f}"
    log("\nКонфиги по выигрышу над level (R² по y; интервал 90% по цепочкам)")
    log(f"{'конфиг':<14}{'энкодер':<10}{'R² y':>8}{'выигрыш':>9}{'5%':>9}{'95%':>9}{'R² след':>9}{'ρ':>8}"
        f"{'ранг':>8}{'ранг|a':>8}{'с':>8}{'парам.':>10}")
    for r in summary["table"]:
        log(f"{r['config']:<14}{(r['encoder'] or '-'):<10}{f(r['r2_y'])}{f(r['gain'], 9, sign=True)}"
            f"{f(r['lo'], 9, sign=True)}{f(r['hi'], 9, sign=True)}{f(r['r2_next'], 9)}{f(r['spearman_y'])}"
            f"{f(r['rank_within'])}{f(r['rank_within_action'])}{r['train_s']:>8.0f}{r['params']:>10d}")
    lp = summary.get("lopo")
    if lp:
        log(f"lopo [{lp['config']}] выигрыш над lopo:level по отложенным задачам:")
        for nm, v in lp["per_pde"].items():
            log(f"  {nm:<28}{f(v[0], 9, sign=True)}{f(v[1], 9, sign=True)}{f(v[2], 9, sign=True)}")
    log("ВЕРДИКТ: " + summary["verdict"])


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("данные")
    g.add_argument("--subdirs", default="all", help="all | solvable | папки буферов через запятую")
    g.add_argument("--max-per-pde", type=int, default=0, help="не больше N переходов на задачу (цепочки целиком)")
    g.add_argument("--max-files", type=int, default=0, help="не больше N файлов буфера на задачу (по одному из HF)")
    g.add_argument("--map-dtype", default="float32", choices=["float32", "float16"])
    g.add_argument("--data-device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="где лежат карты: cpu — пачки переносятся на GPU; auto — на GPU, если хватает памяти")
    g.add_argument("--synthetic", type=int, default=0, help="N синтетических переходов вместо буферов")
    g.add_argument("--views", default="", help="карты из файлов видов (map_views.py) вместо буфера задачи --subdirs "
                                                "(только с --pretrain-only): префиксы rl_arch/views/<тег> или папки")
    g.add_argument("--views-max", type=int, default=0, help="--views: только виды 0..K-1 (1 — карта среды); 0 — все")
    g.add_argument("--procedural", type=int, default=0,
                   help="N процедурных цепочек карт без уравнений (map_views.procedural_records; только с --pretrain-only)")
    g.add_argument("--procedural-len", type=int, default=8, help="карт в процедурной цепочке")
    g = ap.add_argument_group("разбиения и конфиги")
    g.add_argument("--configs", default="all", help="имена через запятую; all — порядок по умолчанию")
    g.add_argument("--folds", type=int, default=3)
    g.add_argument("--lopo", default=DEFAULT_LOPO, help="отложенные задачи (имя или папка буфера)")
    g.add_argument("--encoder", default="auto", choices=("auto",) + ENCODERS,
                   help="энкодер конфигов без суффикса; auto — лучший на фолде 0 среди maps-*")
    g.add_argument("--select-tol", type=float, default=0.003,
                   help="из энкодеров в этом допуске R² от лучшего берётся самый дешёвый")
    g = ap.add_argument_group("обучение")
    g.add_argument("--epochs", type=int, default=12)
    g.add_argument("--batch", type=int, default=512)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--wd", type=float, default=0.01)
    g.add_argument("--ssl-epochs", type=int, default=4)
    g.add_argument("--ssl-steps", type=int, default=0,
                   help="предобучение: столько шагов оптимизатора вместо --ssl-epochs эпох (эпохи подбираются под "
                        "объём карт). Для сравнения предобучения на наборах разного размера при равном числе шагов; "
                        "0 — по --ssl-epochs, как раньше")
    g.add_argument("--mae-ratio", type=float, default=0.6)
    g.add_argument("--con-temp", type=float, default=0.2)
    g.add_argument("--vae-beta", type=float, default=1.0, help="ssl-vae: вес KL (β-VAE; лосс на пиксель)")
    g.add_argument("--vae-dim", type=int, default=32, help="ssl-vae: размерность латента")
    g.add_argument("--dyn-ema", type=float, default=0.99, help="ssl-dyn: постоянная EMA целевого энкодера")
    g.add_argument("--dyn-inv", type=float, default=1.0, help="ssl-dyn: вес обратной динамики")
    g.add_argument("--out-w", type=float, default=1.0, help="ssl-maeout: вес головы исхода рядом с MAE")
    g.add_argument("--mv-w", type=float, default=1.0, help="ssl-mv: вес согласованности видов (VICReg / 25) рядом с MAE")
    g.add_argument("--mv-dim", type=int, default=256, help="ssl-mv: ширина проектора VICReg")
    g.add_argument("--shift", type=int, default=2, help="наибольший сдвиг aug-shift и видов ssl-con, пикселей")
    g.add_argument("--shuffle-indep", action="store_true", help="shuffle: своя перестановка у каждого канала")
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("оценка и выход")
    g.add_argument("--boot", type=int, default=1000, help="повторов бутстрепа по цепочкам")
    g.add_argument("--min-gain", type=float, default=0.01, help="порог выигрыша R² в правиле вердикта")
    g.add_argument("--hours", type=float, default=0.0, help="мягкий предел: новый конфиг не начинается (0 — нет)")
    g.add_argument("--tag", required=True)
    g.add_argument("--out-dir", default=".")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--no-upload", action="store_true")
    g.add_argument("--save-preds", action="store_true", help="предсказания на отложенных фолдах в <out-dir>")
    g.add_argument("--save-encoder", action="store_true",
                   help="сохранять и заливать в HF веса энкодера после предобучения (ssl-mae, ssl-con, "
                        "frozen): файл для offline_rl.py --encoder-init. Пишется, когда предобучение "
                        "идёт в этом запуске (при --resume уже посчитанные конфиги его не повторяют)")
    g.add_argument("--pretrain-only", action="store_true",
                   help="только предобучение энкодеров конфигов с самообучением (ssl-mae-<энкодер>, ssl-con-..., "
                        "frozen...) и их сохранение (подразумевает --save-encoder): без level, без фолдов и lopo. "
                        "Отчёт — cvprobe_<tag>.json с полем encoders (файлы, лосс, шаги, данные)")
    g.add_argument("--smoke", action="store_true")
    return ap


def run(args, log=None):
    log = log or (lambda *a: print(*a, flush=True))
    t0 = time.time()
    if args.smoke:      # только как умолчания: явные флаги не перезаписываются
        given = set(a.split("=")[0] for a in getattr(args, "_argv", []) if a.startswith("--"))
        for flag, v in (("--epochs", 1), ("--folds", 2), ("--max-per-pde", 400), ("--max-files", 2),
                        ("--ssl-epochs", 1), ("--boot", 200), ("--batch", 128)):
            if flag not in given:
                setattr(args, flag[2:].replace("-", "_"), v)
        args.no_upload = True
    names = expand_configs(args.configs)
    if (getattr(args, "views", "") or getattr(args, "procedural", 0)) and not getattr(args, "pretrain_only", False):
        raise SystemExit("--views и --procedural дают карты без исходов: только с --pretrain-only")
    if getattr(args, "views", "") and getattr(args, "synthetic", 0):
        raise SystemExit("--views и --synthetic вместе не используются")
    if getattr(args, "pretrain_only", False):
        return run_pretrain_only(args, names, log, t0)
    if args.folds < 2:
        raise SystemExit("--folds >= 2")
    if any(not n.startswith("lopo:") and n not in ("scalars", "level") for n in names) and "level" not in names:
        names.insert(0, "level")                # без level нет выигрыша
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log(f"cv_probe {args.tag}: устройство {dev}, конфиги: {', '.join(names)}")
    tab, maps = load_data(args, log)
    D = Data(tab, maps, dev, args.data_device)
    del maps
    if D.H != D.W or D.H % 2:
        raise SystemExit(f"карты {D.H}x{D.W}: энкодеры и D4 рассчитаны на квадратные карты чётного размера")
    sizes = D.sizes()
    log(f"данные: переходов {D.n}, цепочек {D.n_ch}, задач {len(D.names)}, карты {D.C}x{D.H}x{D.W} "
        f"{D.maps.dtype} на {D.maps.device}, загрузка {time.time() - t0:.0f} с")
    for nm, s in sizes.items():
        log(f"  {nm:<28} n={s['n']:<7} цепочек={s['chains']:<6} нулевых карт={s['zero_maps']:.3f} "
            f"(после старта {s['zero_maps_after_start']}) карта: среднее {s['map_mean']:.2f}, разброс "
            f"{s['map_std']:.2f}, |max| {s['map_absmax']:.1f}; y: {s['y_mean']:+.3f} ± {s['y_std']:.3f}")
    report = dict(kind="cv_probe", tag=args.tag, smoke=bool(args.smoke), finished=False,
                  settings={k: v for k, v in vars(args).items() if not k.startswith("_")},
                  data=dict(n=D.n, chains=D.n_ch, pdes=D.names, fingerprint=D.fingerprint, per_pde=sizes,
                            y_mean=float(D.y.mean()), y_std=float(D.y.std())),
                  device=str(dev), configs={}, summary=None)
    sse = {}
    if args.resume:
        prev, psse = load_prev(args, log)
        if prev is not None:
            pf = (prev.get("data") or {}).get("fingerprint")
            if pf != D.fingerprint:
                raise SystemExit(f"--resume: отчёт {args.tag} посчитан на других данных ({pf} против "
                                 f"{D.fingerprint}); возьмите другой --tag")
            report["configs"] = prev.get("configs") or {}
            report["elapsed_before_s"] = float(prev.get("elapsed_s") or 0.0)
            sse = {k: v for k, v in psse.items() if len(v) == D.n_ch}
            if "level" in report["configs"] and _key("cv", "level") not in sse and \
                    any(not is_done(report, n) and not n.startswith("lopo:") for n in names):
                del report["configs"]["level"]   # без его сумм по цепочкам интервал не посчитать: он дёшев
                log("  продолжение: у level нет сумм по цепочкам — считается заново")
            log(f"  продолжение: готово {sorted(report['configs'])}")
    fold_row = S.chain_folds(dict(CH=D.ch, A=D.A), n_splits=args.folds, seed=args.seed)[D.ch]
    boot = Boot(D, n=args.boot, seed=args.seed)
    cache = dict(pre={}, z={})
    ids = lopo_ids(D, args.lopo, log) if any(n.startswith("lopo:") for n in names) else []

    def time_left():
        return not args.hours or (time.time() - t0) < args.hours * 3600.0

    saves = [0]

    def save(do_upload=True):
        report["summary"] = summarize(report, sse, D, boot, args)
        saves[0] += 1                           # суммы по цепочкам в HF — каждое четвёртое сохранение
        save_sse(sse, args, do_upload=do_upload and saves[0] % 4 == 0)     # (лимит коммитов HF)
        return save_report(report, args, t0, do_upload)

    stopped = None
    for name in names:
        if is_done(report, name):
            log(f"{name}: уже посчитан, пропуск")
            continue
        if not time_left():
            stopped = name
            break
        pat = kill_pattern(args)
        if pat:
            stopped, report["killed_by"] = name, pat
            log(f"задача {args.tag!r} отменена записью {pat!r} списка killed.txt")
            break
        log(f"{name}: старт ({(time.time() - t0) / 60:.1f} мин)")
        report["selected_encoder"] = select_encoder(report, args)
        enc = report["selected_encoder"]["name"]
        if name.startswith("lopo:"):
            if not run_lopo(name, D, ids, report, sse, args, dev, cache, boot, enc, save, time_left, log):
                stopped = name
                break
        else:
            sp = spec_of(name, enc)
            ent, oof = run_cv(name, sp, D, fold_row, args, dev, cache, log)
            report["configs"][name] = ent
            sse[_key("cv", name)] = chain_sse(D, np.arange(D.n), oof)
            cv_gains(report, sse, D, boot, only=None if name in ("level", "level-hgb") else {name})
            if args.save_preds:
                np.save(os.path.join(args.out_dir, f"cvprobe_{args.tag}_pred_{_key(name)}.npy"), oof.astype(np.float32))
            g = (ent.get("gain_over_level") or {}).get("r2_y")
            log(f"{name}: R² y {ent['r2_y']:.4f}, R² след. {ent['r2_next']:.4f}, ρ {ent['spearman_y']:.4f}"
                + (f", выигрыш над level {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]" if g else "")
                + f", {ent['train_s']:.0f} с, параметров {ent['params']}")
        save()
    report["finished"] = stopped is None
    report["stopped_before"] = stopped
    if stopped:
        log(f"остановка перед {stopped} (предел --hours {args.hours} или отмена); продолжение — тот же "
            f"--tag и --resume")
    report["selected_encoder"] = select_encoder(report, args)
    path = save(do_upload=False)
    upload(path, args, tries=6)
    save_sse(sse, args, do_upload=True)
    print_summary(report["summary"], log)
    log(f"отчёт: {path} ({report['elapsed_s']} с)")
    report["_paths"] = dict(report=path, chains=paths(args)[1])
    return report


def run_pretrain_only(args, names, log, t0):
    """--pretrain-only: загрузка карт (--subdirs, --max-per-pde), предобучение и сохранение энкодера каждого
    конфига с самообучением; ни level, ни фолдов, ни lopo. Энкодер файла — суффикс конфига, иначе --encoder
    (auto — энкодер по умолчанию: выбирать по фолду здесь не из чего)."""
    specs = []
    for n in names:
        if n.startswith("lopo:"):
            raise SystemExit(f"--pretrain-only: конфиг {n!r} — оценка «исключить УрЧП», а не предобучение")
        _, enc = parse_config(n)
        sp = spec_of(n, enc or (args.encoder if args.encoder != "auto" else FALLBACK_ENCODER))
        if not sp.get("ssl"):
            raise SystemExit(f"--pretrain-only: у конфига {n!r} нет предобучения (нужны ssl-mae, ssl-con, frozen...)")
        specs.append((n, sp))
    args.save_encoder = True
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"cv_probe {args.tag}: только предобучение, устройство {dev}, конфиги: {', '.join(names)}")
    tab, maps = load_data(args, log)
    D = Data(tab, maps, dev, args.data_device)
    del maps
    if D.H != D.W or D.H % 2:
        raise SystemExit(f"карты {D.H}x{D.W}: энкодеры рассчитаны на квадратные карты чётного размера")
    log(f"данные: переходов {D.n}, цепочек {D.n_ch}, задач {len(D.names)} ({', '.join(D.names)}), "
        f"карты {D.C}x{D.H}x{D.W}, загрузка {time.time() - t0:.0f} с")
    report = dict(kind="cv_probe_pretrain", tag=args.tag, smoke=bool(args.smoke), finished=False,
                  settings={k: v for k, v in vars(args).items() if not k.startswith("_")},
                  data=dict(n=D.n, chains=D.n_ch, pdes=D.names, subdirs=D.subdirs, fingerprint=D.fingerprint),
                  device=str(dev), encoders={})
    cache = dict(pre={}, z={})
    for n, sp in specs:
        pre = get_pre(sp, D, args, dev, cache, log)
        report["encoders"][n] = dict(ssl=sp["ssl"], encoder=sp["encoder"],
                                     file=f"{ENC_DIR}/{os.path.basename(pre['file'])}",
                                     seconds=round(pre["seconds"], 1), loss_first=pre["loss_first"],
                                     loss_last=pre["loss_last"], epochs=pre["epochs"], steps=pre["steps"],
                                     n_maps=pre["n_maps"], loss_parts=pre.get("loss_parts") or {},
                                     ssl_args=ssl_args(sp["ssl"], args))
        log(f"{n}: {pre['steps']} шагов ({pre['epochs']} эпох по {pre['n_maps']} картам), лосс "
            f"{pre['loss_first']:.4f} -> {pre['loss_last']:.4f}, {pre['seconds']:.0f} с")
    report["finished"] = True
    path = save_report(report, args, t0, do_upload=False)
    upload(path, args, tries=6)
    log(f"отчёт: {path} ({report['elapsed_s']} с)")
    report["_paths"] = dict(report=path)
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args._argv = argv
    args._kill = True
    try:        # как agent_combo.main: отменённая задача выходит до загрузки буферов
        from killswitch import exit_if_killed
        exit_if_killed(args.tag, remote=True, what="зонд CV")
    except ImportError:
        pass
    return run(args)


if __name__ == "__main__":
    main()
