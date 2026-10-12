#!/usr/bin/env python
"""Модельные цели для агента на картах (offline_rl.py --mb-aux): плотный сигнал по всем 27 действиям
из суррогатной среды.

У перехода буфера известен исход одного действия из 27. Ансамбль модели одного шага
(track3/surrogate.py: бустинг по действию, шагу, прошлому действию, ошибке и описателю задачи)
предсказывает исход КАЖДОГО действия в реальном состоянии буфера, и для него считается
пессимистичная ценность (MOPO, 2005.13239; модельное расширение ценности MVE, 1803.00101):

    Q_mb(s, a) = r~(s, a) + g(a) * [ r~(s1, a1*) + g(a1*) * [ ... ] ],        не больше H действий,
    r~ = среднее по членам ансамбля награды - L * её разброс по членам,
    a_k* = argmax_a r~(s_k, a)   (жадный просмотр: полное дерево 27^H не строится),

следующее состояние — среднее членов. Продолжение обрывается, когда выбран бюджет эпох (последнее
действие обрезается остатком бюджета, как в среде оценки), когда ошибка ушла на верхний предел
таблицы (расходимость) и когда лучшая награда r~ не положительна: хвост считается нулём (в среде
есть почти пустое действие — PSO с нулевым шагом на 100 эпох). Жадный путь конечной длины — нижняя
оценка ценности в модели, поэтому это добавка к TD-лоссу, а не его замена:
W * Huber(Q(s, .) - Q_mb(s, .)).

Единицы награды — те же, что у тренера (offline_rl.episodes_to_arrays_chains):
    delta   (E_до - E_после) / err_scale;
    dlog    log10 E_до - log10 E_после, обрезка [-3, 3] (от масштаба ошибки не зависит);
    первый шаг цепочки считается от init_err (в единицах тренера), а без него награда первого
    шага равна нулю — как в загрузчике.
Формы logged и level не поддержаны: там награда — уровень ошибки, и сумма за H шагов без хвоста не
равна ценности. Дисконт — gamma тренера (при --smdp: gamma^(эпохи/scale) по номиналу действия).

Состояние суррогата строки буфера восстанавливается из массивов тренера (EP, STEP, SPENT, A, ERR)
тем же правилом, что surrogate.build_table: ошибка перед первым действием — E0 задачи из
pde_meta.json, дальше — ошибка после предыдущего действия. Цели считаются только для обучающих
строк; суррогат учится на всех цепочках таблицы (--mb-subdirs), включая отложенные цепочки тренера.

Таблица и ансамбль строятся как в agent_combo.py (get_table, Models) и кэшируются в --mb-work.

Щит на оценке (online_eval_env.py --shield): тот же ансамбль перед каждым решением агента
предсказывает исход каждого разрешённого действия в ЖИВОМ состоянии прогона и запрещает действия,
у которых пессимистичное изменение log10 ошибки (среднее членов + L * разброс) не отрицательно.
В состоянии суррогата есть текущая ошибка rmse+brmse: на оценке это измеренная истинная ошибка
(привилегированная информация, как --ctx-err у агента) либо оценка самой модели (model_err).
Состояние живого прогона (live_state) собирается той же функцией chain_states, что и состояние
строки буфера, признаки — той же step_features (её же зовёт _predict), поэтому вход ансамбля на
оценке совпадает с входом при обучении (tests/test_shield_teacher.py сверяет три пути: таблица,
массивы тренера, живой прогон). Ансамбль — файл combo_surr_<отпечаток>_full_m<N>_s0.pkl из --mb-work
(его путь тренер пишет в meta чекпоинта: mb.surr_file).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(HERE, "track3"), HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surrogate as S  # noqa: E402

FORMS = ("delta", "dlog")


def build_model(names, subdirs="all", table="", members=5, work="agent_combo_work", quick=False, log=print):
    """Таблица переходов и ансамбль суррогата на всех её цепочках — теми же функциями, что у
    agent_combo.py (кэш в work общий). names — задачи набора тренера: если буфера задачи нет в
    таблице, для неё добавляется строка описателя (суррогат тогда экстраполирует)."""
    import agent_combo as AC
    import offline_rl as O
    meta = O.pde_meta()
    tab = AC.get_table(argparse.Namespace(table=table, work=work), O.resolve_subdirs(subdirs), log)
    missing = [n for n in names if n not in tab["NAMES"]]
    for n in names:
        AC.ensure_pde(tab, n, meta)
    if missing:
        log(f"модельные цели: буферов задач {missing} нет в таблице суррогата — он экстраполирует")
    models = AC.Models(tab, work, members=members, quick=quick, seed=0, log=log)
    # surr_file — файл ансамбля в кэше: его читает щит оценки (online_eval_env.py --shield auto)
    return tab, models.get("full"), dict(surr_rows=int(len(tab["A"])), surr_sig=models.sig,
                                         surr_members=int(members),
                                         surr_file=os.path.abspath(models.path("full")))


def chain_states(data, tab, pidx, err_scale, budget=0):
    """Состояние суррогата перед действием для каждой строки набора тренера. pidx[k] — номер
    задачи k набора в таблице суррогата, err_scale[k] — на что тренер делил её ошибки.
    Возвращает (состояние, valid, first): valid — строка стоит в цепочке, которая начинается с
    шага 0 и идёт подряд (иначе прошлая ошибка и расход эпох неизвестны), и бюджет ещё не выбран.
    data["EPA"] (необязательно) — фактические эпохи действий: у буферов они равны номиналу, у
    живого прогона последнее действие может быть обрезано бюджетом."""
    A = np.asarray(data["A"], dtype=np.int64)
    EP, STEP = np.asarray(data["EP"]), np.asarray(data["STEP"], dtype=np.int64)
    SPENT = np.asarray(data["SPENT"], dtype=np.int64)
    ERR = np.asarray(data["ERR"], dtype=np.float64)
    n = len(A)
    pl = np.asarray(data["PDE"], dtype=np.int64) if "PDE" in data else np.zeros(n, dtype=np.int64)
    first = np.r_[True, EP[1:] != EP[:-1]]
    link = np.where(first, STEP == 0, np.r_[True, STEP[1:] == STEP[:-1] + 1])
    cid = np.cumsum(first) - 1
    first_bad = np.full(int(cid.max()) + 1, n, dtype=np.int64)
    np.minimum.at(first_bad, cid, np.where(link, n, np.arange(n)))
    valid = np.arange(n) < first_bad[cid]
    if budget:
        valid &= SPENT < budget
    e0 = 10.0 ** tab["CTX"][pidx[pl], 6]
    prev = np.where(first, e0, np.r_[1.0, ERR[:-1]] * np.asarray(err_scale, dtype=np.float64)[pl])
    pa = np.where(first, -1, np.r_[-1, A[:-1]])
    oh = np.zeros((n, 3))
    oh[np.arange(n), S.FAM[A]] = np.asarray(data["EPA"], dtype=np.float64) if "EPA" in data else S.EPOCHS[A]
    cs = np.cumsum(oh, 0) - oh                                   # эпох по семействам до строки
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    st = dict(pde=pidx[pl], step=STEP, spent=SPENT, spf=cs - cs[start], pa=pa.astype(np.int64),
              l=S._log(prev))
    return st, valid, first


def step_features(tab, st, a, budget=0):
    """Вход ансамбля для действий a в состояниях st и фактические эпохи: при заданном бюджете
    действие обрезается его остатком (как agent_combo.model_predict). Единственное место, где
    состояние превращается в признаки и для модельных целей, и для щита оценки; то же выражение,
    на котором ансамбль учится (agent_combo.fit_features: шаг ограничен K_MAX - 1, эпохи фактические)."""
    a = np.asarray(a, dtype=np.int64)
    ep = np.minimum(S.EPOCHS[a], budget - st["spent"]) if budget else S.EPOCHS[a]
    X = S.featurize(tab["CTX"][st["pde"]], np.minimum(st["step"], S.K_MAX - 1), st["spent"],
                    st["spf"], st["pa"], st["l"], a, ep=ep)
    return X, ep


def _predict(ens, tab, st, a, budget):
    """log10 ошибки после действия по членам (члены, n) и фактические эпохи."""
    X, ep = step_features(tab, st, a, budget)
    return ens.predict_ln(X, st["l"]), ep


def _reward(form, l, ln, scale):
    """Награда по членам (члены, n) в единицах тренера: l — log10 ошибки до действия (n,),
    ln — после (члены, n), scale — делитель ошибки (n,)."""
    if form == "dlog":
        return np.clip(l[None] - ln, -3.0, 3.0)
    return (10.0 ** l[None] - 10.0 ** ln) / scale[None]


def _done(st, budget):
    """Эпизод кончился: бюджет выбран (без бюджета — K_MAX шагов цепочки буфера) или расходимость."""
    end = st["spent"] >= budget if budget else st["step"] >= S.K_MAX
    return end | (st["l"] >= S.L_HI - 1e-9)


def lookahead(ens, tab, st, l_rew, zero_first, scale, form, gamma, h, lam, budget=0, smdp_scale=0.0,
              chunk=1500, log=None):
    """Q_mb (n, 27) для состояний st (см. докстринг модуля). l_rew — log10 ошибки «до» для награды
    первого действия (у первого шага цепочки это init_err тренера, а не E0 таблицы), zero_first —
    строки, где награда первого действия по правилу тренера равна нулю. Возвращает (Q, средний
    разброс log10 ошибки по членам на первом действии)."""
    import agent_combo as AC
    if form not in FORMS:
        raise ValueError(f"модельные цели: награда {form!r} не поддержана, нужна одна из {FORMS}")
    n = len(st["l"])
    na = len(S.ACTIONS)
    acts = np.arange(na)
    g_act = (float(gamma) ** (S.EPOCHS / float(smdp_scale)) if smdp_scale
             else np.full(na, float(gamma)))
    Q = np.zeros((n, na))
    unc, t0 = [], time.time()
    for c0 in range(0, n, chunk):
        rows = np.arange(c0, min(n, c0 + chunk))
        m = len(rows)
        rep_i = np.repeat(np.arange(m), na)
        rep = AC.take(st, rows[rep_i])
        a = np.tile(acts, m)
        P, ep = _predict(ens, tab, rep, a, budget)
        Rm = _reward(form, l_rew[rows][rep_i], P, scale[rows][rep_i])
        q = Rm.mean(0) - lam * Rm.std(0)
        q[zero_first[rows][rep_i]] = 0.0
        unc.append(P.std(0).mean())
        disc = g_act[a].copy()
        cur = AC.next_state(rep, a, ep, P.mean(0))
        idx = np.where(~_done(cur, budget))[0]
        cur, sc = AC.take(cur, idx), scale[rows][rep_i]
        for _ in range(1, h):
            k = len(idx)
            if not k:
                break
            rep2 = AC.take(cur, np.repeat(np.arange(k), na))
            a2 = np.tile(acts, k)
            P2, ep2 = _predict(ens, tab, rep2, a2, budget)
            R2 = _reward(form, rep2["l"], P2, np.repeat(sc[idx], na))
            r2 = (R2.mean(0) - lam * R2.std(0)).reshape(k, na)
            best = r2.argmax(1)
            rb = r2[np.arange(k), best]
            go = rb > 0                                           # иначе хвост нулевой
            q[idx[go]] += disc[idx[go]] * rb[go]
            disc[idx] *= g_act[best]
            flat = np.arange(k) * na + best
            nxt = AC.next_state(AC.take(rep2, flat), best, ep2[flat], P2.mean(0)[flat])
            keep = np.where(go & ~_done(nxt, budget))[0]
            cur, idx = AC.take(nxt, keep), idx[keep]
        Q[rows] = q.reshape(m, na)
        if log is not None and c0 == 0 and n > chunk:
            log(f"модельные цели: первые {m} строк за {time.time() - t0:.1f} с, всего строк {n}")
    return Q, float(np.mean(unc)) if unc else 0.0


def precompute(data, train_mask, names, reward_form, err_scale, init_err, gamma, h=3, lam=0.5,
               budget=0, smdp_scale=0.0, subdirs="all", table="", members=5, work="agent_combo_work",
               quick=False, model=None, log=None):
    """Цели Q_mb для обучающих строк набора тренера. names — задачи набора (в порядке data['PDE']),
    err_scale и init_err — по задачам, в единицах тренера (init_err=None: награда первого шага 0).
    model=(tab, ens) — готовые таблица и ансамбль (тесты), иначе они строятся build_model.
    Возвращает (QMB float32 (N, 27), MBW bool (N,) — у каких строк цель есть, сводка)."""
    log = log or (lambda *a: print(*a, flush=True))
    t0 = time.time()
    for k in ("EP", "STEP", "SPENT", "ERR"):
        if k not in data:
            raise ValueError("модельные цели требуют набор, загруженный с --chain-fix")
    n = len(data["A"])
    if model is None:
        tab, ens, info = build_model(names, subdirs, table, members, work, quick, log)
    else:
        tab, ens = model
        info = dict(surr_rows=int(len(tab["A"])), surr_sig="given", surr_members=int(ens.members))
    pidx = np.array([tab["NAMES"].index(nm) for nm in names], dtype=np.int64)
    err_scale = np.asarray(err_scale, dtype=np.float64)
    assert err_scale.shape == (len(names),), (err_scale.shape, len(names))
    st, valid, first = chain_states(data, tab, pidx, err_scale, budget)
    pl = np.asarray(data["PDE"], dtype=np.int64) if "PDE" in data else np.zeros(n, dtype=np.int64)
    l_rew = st["l"].copy()
    zero_first = np.zeros(n, dtype=bool)
    if init_err is None:
        zero_first = first.copy()
    else:
        init_err = np.asarray(init_err, dtype=np.float64)
        assert init_err.shape == (len(names),), (init_err.shape, len(names))
        l_rew[first] = S._log(init_err[pl] * err_scale[pl])[first]
    rows = np.where(np.asarray(train_mask, dtype=bool) & valid)[0]
    QMB = np.zeros((n, len(S.ACTIONS)), dtype=np.float32)
    MBW = np.zeros(n, dtype=bool)
    unc = 0.0
    if len(rows):
        q, unc = lookahead(ens, tab, {k: v[rows] for k, v in st.items()}, l_rew[rows], zero_first[rows],
                           err_scale[pl][rows], reward_form, gamma, int(h), float(lam), int(budget),
                           float(smdp_scale), log=log)
        assert q.shape == (len(rows), QMB.shape[1]), q.shape
        if not np.isfinite(q).all():
            raise ValueError("модельные цели: нечисловые значения Q_mb")
        QMB[rows] = q
        MBW[rows] = True
    assert QMB.shape == (n, 27) and MBW.shape == (n,)
    info.update(rows=int(len(rows)), skipped=int(np.asarray(train_mask, dtype=bool).sum() - len(rows)),
                q_mean=float(QMB[rows].mean()) if len(rows) else 0.0,
                q_best_mean=float(QMB[rows].max(1).mean()) if len(rows) else 0.0,
                unc_log10=round(unc, 5), sec=round(time.time() - t0, 1))
    log(f"модельные цели: строк {info['rows']} (пропущено {info['skipped']}), награда {reward_form}, "
        f"H={h}, lam={lam}, бюджет {budget or 'нет'}: средний Q_mb {info['q_mean']:.4f}, лучшего действия "
        f"{info['q_best_mean']:.4f}, разброс ансамбля {info['unc_log10']} порядка, {info['sec']} с")
    return QMB, MBW, info


# --------------------------------------------------------------------------------------------
# щит на оценке (online_eval_env.py --shield)
# --------------------------------------------------------------------------------------------

def load_shield(spec, pde, agent_meta=None):
    """Ансамбль суррогата для щита и строка описателя задачи. spec — файл ансамбля (pickle
    surrogate.Ensemble из кэша --mb-work / agent_combo.py --work), папка кэша (в ней должен быть
    ровно один combo_surr_*_full_*.pkl) или 'auto' — путь из meta чекпоинта агента (mb.surr_file,
    его пишет offline_rl.py --mb-aux). Файл только локальный: pickle из сети не читается.
    Возвращает (tab, ens, путь): tab — словарь с CTX и NAMES одной задачи, как ждёт chain_states."""
    import offline_rl as O
    path = spec
    if spec == "auto":
        path = ((agent_meta or {}).get("mb") or {}).get("surr_file") or ""
        if not path:
            raise SystemExit("--shield auto: в meta чекпоинта нет пути к суррогату (mb.surr_file): агент "
                             "обучен без --mb-aux или до появления щита; задайте файл или папку кэша")
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "combo_surr_*_full_m*.pkl")))
        if len(found) != 1:
            raise SystemExit(f"--shield {path}: ансамблей combo_surr_*_full_*.pkl в папке {len(found)}, "
                             f"нужен ровно один ({[os.path.basename(f) for f in found]}); задайте файл")
        path = found[0]
    if not os.path.exists(path):
        raise SystemExit(f"--shield: нет файла ансамбля {path!r} (его создаёт шаг обучения этого же "
                         "кернела: offline_rl.py --mb-aux или agent_combo.py)")
    meta = O.pde_meta()
    if pde not in meta:
        raise SystemExit(f"--shield: задачи {pde!r} нет в track3/pde_meta.json — описатель взять неоткуда")
    with open(path, "rb") as f:
        ens = pickle.load(f)
    if getattr(ens, "level", False):
        raise SystemExit("--shield: ансамбль с уровнем карт на входе (level=True) не поддержан — "
                         "agent_combo.Models таких не учит")
    return dict(CTX=S.pde_ctx(meta[pde])[None], NAMES=[pde]), ens, path


def chain_actions(chain):
    """Исполненная цепочка в записи online_eval_env ([оптимизатор, шаг, фактические эпохи]) ->
    (индексы действий, фактические эпохи). Обрезанное бюджетом действие относится к ближайшему
    большему номиналу (как agent_combo.extra_rows). None — в цепочке есть действие вне 27
    (SOAP, BOOST): такого состояния суррогат не знает."""
    acts, eps = [], []
    for opt, lr, ep in chain:
        hit = [i for i, (o, l, e) in enumerate(S.ACTIONS)
               if o.lower() == str(opt).lower() and abs(l - float(lr)) < 1e-12 and e >= int(ep)]
        if not hit or int(ep) <= 0:
            return None
        acts.append(min(hit, key=lambda i: S.EPOCHS[i]))
        eps.append(int(ep))
    return np.array(acts, dtype=np.int64), np.array(eps, dtype=np.int64)


def live_state(tab, chain, err, pi=0):
    """Состояние суррогата живого прогона перед очередным решением: chain — исполненные и
    оставленные действия (откаченные стражем в цепочку не попадают), err — rmse+brmse после
    последнего из них (до первого действия не нужна: берётся E0 задачи, как в таблице).
    Считается функцией chain_states на цепочке с подставной последней строкой — тем же кодом, что
    состояние строки буфера. spent состояния — эпохи оставленных действий (в буферах расход всегда
    равен их сумме); остаток бюджета живого прогона передаётся отдельно (shield, left).
    None — состояние суррогату неизвестно (действие вне 27 или нечисловая ошибка)."""
    ca = chain_actions(chain)
    if ca is None:
        return None
    acts, eps = ca
    k = len(acts)
    if k and not (err is not None and np.isfinite(err) and err > 0):
        return None
    data = dict(A=np.r_[acts, 0], EPA=np.r_[eps, 0], EP=np.zeros(k + 1, np.int64),
                STEP=np.arange(k + 1), SPENT=np.r_[0, np.cumsum(eps)].astype(np.int64),
                ERR=np.full(k + 1, float(err) if k else 1.0))
    st, valid, _ = chain_states(data, tab, np.array([pi]), np.ones(1))
    assert valid[k]
    return {key: v[k:k + 1] for key, v in st.items()}


def model_err(ens, tab, chain, pi=0):
    """Ошибка живого прогона по самой модели, без измерения: открытый прогон среднего ансамбля по
    исполненной цепочке от E0 задачи (как agent_combo.next_state). Для щита без привилегированной
    информации (--shield-err model): истинную rmse+brmse на развёртывании взять неоткуда. Точность
    падает с длиной цепочки. None — в цепочке действие вне 27."""
    ca = chain_actions(chain)
    if ca is None:
        return None
    err = 10.0 ** float(tab["CTX"][pi, 6])
    for k in range(len(ca[0])):
        st = live_state(tab, chain[:k], err, pi)
        P, _ = _predict(ens, tab, st, ca[0][k:k + 1], int(st["spent"][0]) + int(ca[1][k]))
        err = 10.0 ** float(P.mean())
    return err


def shield(ens, tab, st, allowed, lam=0.5, keep=3, left=0):
    """Вето щита в состоянии st (одна строка, live_state). Для каждого из 27 действий ансамбль даёт
    изменение log10 ошибки по членам d = ln - l (меньше нуля — улучшение; награда dlog модельных
    целей — это -d). Пессимистичная оценка score = среднее d + lam * разброс d; действие запрещено,
    если score >= 0. Если разрешённых и не запрещённых меньше keep, остаются keep лучших по score
    среди разрешённых маской. left — остаток бюджета живого прогона в эпохах (действие обрезается
    им, как в среде оценки). Возвращает (ok bool (27,), score, среднее d, разброс d)."""
    allowed = np.asarray(allowed, dtype=bool)
    na = len(S.ACTIONS)
    rep = {k: np.repeat(v[:1], na, 0) for k, v in st.items()}
    P, _ = _predict(ens, tab, rep, np.arange(na), (int(st["spent"][0]) + int(left)) if left else 0)
    d = P - rep["l"][None]
    mu, sd = d.mean(0), d.std(0)
    score = mu + float(lam) * sd
    ok = allowed & (score < 0)
    if ok.sum() < keep:
        idx = np.where(allowed)[0]
        ok = np.zeros(na, dtype=bool)
        ok[idx[np.argsort(score[idx], kind="stable")[:max(1, int(keep))]]] = True
    return ok, score, mu, sd


def shield_step(ens, tab, chain, err, allowed, a_free, lam=0.5, keep=3, left=0, spent=0):
    """Одно решение щита в цикле оценки. chain и err — как у live_state, allowed — маска оценки,
    a_free — что агент выбрал бы без щита, left — остаток бюджета, spent — расход (для журнала).
    Возвращает (маска для выбора агента, запись журнала); действие, которое агент выберет по этой
    маске, дописывает shield_chosen. Состояние суррогату неизвестно — маска прежняя, skip=True."""
    st = live_state(tab, chain, err)
    sh = dict(step=len(chain) + 1, spent=int(spent), free=int(a_free), skip=st is None)
    if st is None:
        return np.asarray(allowed, dtype=bool), sh
    ok, score, mu, sd = shield(ens, tab, st, allowed, lam, keep, left)
    sh.update(n_allowed=int(np.sum(allowed)), vetoed=int((np.asarray(allowed, dtype=bool) & ~ok).sum()),
              forced=bool((score[ok] >= 0).any()), free_vetoed=bool(not ok[int(a_free)]),
              _mu=mu, _sd=sd)
    return ok, sh


def shield_chosen(sh, a):
    """Дописать в запись журнала выбранное действие и его прогноз (среднее и разброс изменения
    log10 ошибки по членам); служебные поля записи убираются."""
    mu, sd = sh.pop("_mu", None), sh.pop("_sd", None)
    if mu is not None:
        sh.update(a=int(a), pred=round(float(mu[int(a)]), 4), pred_sd=round(float(sd[int(a)]), 4))
    return sh


def shield_summary(log):
    """Сводка журнала щита для строки результата: доля решений, где щит запретил собственный выбор
    агента (veto_rate), средняя доля запрещённых действий среди разрешённых маской, число решений,
    где улучшения не обещало почти ни одно действие (forced). Шаги со skip не в счёт."""
    act = [x for x in log if not x.get("skip")]
    return dict(shield_steps=len(act), shield_skips=len(log) - len(act),
                shield_veto_rate=(round(float(np.mean([x["free_vetoed"] for x in act])), 4) if act else None),
                shield_vetoed_frac=(round(float(np.mean([x["vetoed"] / max(1, x["n_allowed"]) for x in act])), 4)
                                    if act else None),
                shield_forced=int(sum(x["forced"] for x in act)))
