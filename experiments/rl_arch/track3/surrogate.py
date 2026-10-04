#!/usr/bin/env python
"""Суррогатная среда из офлайновых буферов: можно ли планировать цепочки оптимизаторов в модели.

Литература: ESP (arXiv 2002.05368) — поиск предписаний по выученному суррогату; MBPO и MOPO —
офлайновое RL в модели со штрафом за разброс ансамбля; находка E (TRACK3.md) — следующая ошибка
предсказуема по действию, шагу и прошлому действию (R² 0.74), с шестью числами уровня карт — 0.958.

Модель одного шага: ансамбль градиентного бустинга (HistGradientBoosting), члены учатся на
бутстрепе цепочек; каждый член — своя модель среды и в прогоне ведёт свою траекторию (PETS).
Состояние: log10 текущей ошибки (rmse + brmse, как в буфере и в строках оценки), шаг, потрачено
эпох (всего и по семействам), прошлое действие, описатель задачи (6 чисел и log10 E0); вариант
level добавляет шесть чисел уровня карт (минимум и среднее трёх карт потерь). Цель — изменение
log10 ошибки за действие.

Буферы собраны в среде СО СБРОСОМ оптимизатора (находка N), модель описывает её. Цепочки без пар
«L-BFGS -> L-BFGS» и «Adam -> Adam с тем же шагом» (--consistent safe) в среде без сбросов
(--keep-opt --keep-opt-mode safe) исполняются так же: каждое их действие и там начинается с нового
оптимизатора.

Подкоманды (каждая печатает таблицу и пишет surrogate_<команда>.json в --work):
    table      таблица переходов всех буферов, размеры по УрЧП (шаг 1)
    onestep    R² модели одного шага: GroupKFold по цепочкам и «исключить УрЧП» (шаг 2)
    multistep  прогон записанных цепочек через модель: Spearman итоговой ошибки, разброс (шаг 3)
    gpucheck   то же на строках GPU-оценки (среда со сбросом, бюджет 7000): статические цепочки
    plan       планирование для ns2d_liddriven и poissonboltzmann2d, место замеренных цепочек (шаг 4)
    halves     план по модели одной половины буфера, оценка моделью другой (проклятие оптимизатора)
    lopo       планирование моделью без задачи против универсальной цепочки (шаг 4)
    mpc        перепланирование после каждого действия, plan_next, на записанных траекториях (шаг 5)
    all        всё по порядку

    env -u HF_TOKEN DDEBACKEND=pytorch HF_HUB_OFFLINE=1 \\
        python experiments/rl_arch/track3/surrogate.py all --work /tmp/surrogate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(RL, "..", ".."))
for _p in (RL, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUDGET = 7000
K_MAX = 12                      # шагов в цепочке буфера
L_LO, L_HI = -6.0, 4.0          # пределы log10 ошибки
# та же таблица и тот же порядок, что ACTION_TABLE в online_eval_env.py (a = opt*9 + lr*3 + ep)
ACTIONS = [(o, lr, ep) for o, lrs, eps in (("Adam", (1e-2, 1e-3, 1e-4), (100, 1000, 2500)),
                                            ("LBFGS", (1.0, 5e-1, 1e-1), (100, 500, 1000)),
                                            ("PSO", (0.0, 1e-3, 1e-4), (100, 200, 300)))
           for lr in lrs for ep in eps]
EPOCHS = np.array([a[2] for a in ACTIONS], dtype=np.int64)
FAM = np.arange(27) // 9
LRI = (np.arange(27) % 9) // 3
NONE = 27                       # код «прошлого действия нет»
MAIN = ("ns2d_liddriven", "poissonboltzmann2d")
# замеренные на GPU цепочки (10 сидов, медиана l2re). Скрипт оценки по умолчанию останавливается в
# конце цепочки; армы t3b_sk* исполнялись с повтором последнего действия до бюджета («+хвост»)
_RAND6 = ("PSO:0:200,LBFGS:1:100,Adam:0.0001:2500,LBFGS:0.1:500,LBFGS:1:500,LBFGS:1:1000,"
          "LBFGS:0.1:1000,PSO:0:100,PSO:0.0001:300,PSO:0.001:300")
MEASURED = {
    "ns2d_liddriven": [
        ("sk1", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000", "0.0388 без сброса (safe, с хвостом)"),
        ("sk1+хвост", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000,LBFGS:1:1000", "0.0414 со сбросом"),
        ("t3b_log3", "PSO:0.0001:300,PSO:0:300,PSO:0.001:200,PSO:0.0001:300,Adam:0.01:100,"
                     "Adam:0.01:2500,Adam:0.0001:1000,Adam:0.001:1000,PSO:0:200,LBFGS:1:1000",
         "0.0376 со сбросом"),
        ("rand6", _RAND6, "0.0640 со сбросом (универсальная цепочка)"),
    ],
    "poissonboltzmann2d": [
        ("sk2", "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000", "0.0075 без сброса (safe, с хвостом)"),
        ("sk2+хвост", "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:500",
         "0.0096 со сбросом"),
        ("sk1+хвост", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000,LBFGS:1:1000", "0.0143 со сбросом"),
        ("rand6", _RAND6, "0.0158 со сбросом (универсальная цепочка)"),
    ],
}
LOPO_MIN_FILES = 40
# УрЧП, которые не решает ни один метод (offline_rl.UNSOLVED_SUBDIRS): цепочки там неразличимы
UNSOLVED = ("burgers2d", "heat2d_longtime", "kuramoto_sivashinsky", "ns2d_longtime",
            "poisson2d_manyarea", "wave2d_heterogeneous", "wave2d_longtime")


def _enc_table():
    """Кодирование действия: семейство (3), log10 шага (PSO с нулевым шагом -> -6), log10 эпох.
    Строка NONE — «действия нет»."""
    rows = []
    for o, lr, ep in ACTIONS:
        f = ("Adam", "LBFGS", "PSO").index(o)
        rows.append([f == 0, f == 1, f == 2, np.log10(lr) if lr > 0 else -6.0, np.log10(ep)])
    rows.append([0, 0, 0, -7.0, 0.0])
    return np.array(rows, dtype=np.float64)


ENC = _enc_table()


def action_str(a):
    o, lr, ep = ACTIONS[int(a)]
    return f"{o}:{lr:g}:{ep}"


def chain_str(chain):
    return ",".join(action_str(a) for a in chain)


def parse_chain(spec):
    """'Adam:0.01:2500,LBFGS:1:1000' -> индексы действий (та же запись, что --script оценки)."""
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        o, lr, ep = tok.split(":")
        hit = [i for i, (on, l, e) in enumerate(ACTIONS)
               if on.lower() == o.lower() and abs(l - float(lr)) < 1e-12 and e == int(ep)]
        if not hit:
            raise ValueError(f"нет действия {tok!r} среди 27")
        out.append(hit[0])
    return out


def parse_mask(spec):
    """True = действие разрешено. Токены как в online_eval_env.parse_mask: 'pso', 'pso:0.001',
    'adam:0.01:100', номер."""
    allowed = np.ones(27, dtype=bool)
    for tok in (t.strip() for t in (spec or "").split(",") if t.strip()):
        if tok.isdigit():
            allowed[int(tok)] = False
            continue
        p = tok.split(":")
        hit = [i for i, (on, l, e) in enumerate(ACTIONS) if on.lower() == p[0].lower()
               and (len(p) < 2 or abs(l - float(p[1])) < 1e-12) and (len(p) < 3 or e == int(p[2]))]
        if not hit:
            raise ValueError(f"маска: токен {tok!r} не совпал ни с одним действием")
        allowed[hit] = False
    return allowed


def consistent_ok(pa, a, mode):
    """Переход pa -> a исполняется на новом оптимизаторе и в среде без сбросов (как keep_consistent
    в offline_rl.py): первый шаг, PSO, смена семейства; при mode='safe' ещё Adam со сменой шага."""
    pa, a = np.asarray(pa), np.asarray(a)
    if mode in (None, "", "none"):
        return np.ones(np.broadcast(pa, a).shape, dtype=bool)
    pf = FAM[np.maximum(pa, 0)]
    ok = (pa < 0) | (FAM[a] == 2) | (pf != FAM[a])
    if mode == "safe":
        ok |= (FAM[a] == 0) & (pf == 0) & (LRI[a] != LRI[np.maximum(pa, 0)])
    return ok


# --------------------------------------------------------------------------------------------
# Шаг 1. Таблица переходов
# --------------------------------------------------------------------------------------------

def buffer_root():
    r = os.environ.get("RL_BUFFER_DIR")
    if r:
        return r
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--danil-e--rlpinn-ablation-buffers/snapshots/*"))
    return max(snaps, key=os.path.getmtime) if snaps else None


def pde_ctx(row):
    """Описатель задачи (как offline_rl.pde_desc) и log10 ошибки необученной сети E0."""
    return np.array([min(1.0, row["in_dim"] / 5.0), min(1.0, row["out_dim"] / 3.0),
                     min(1.0, row["n_pde"] / 3.0), min(1.0, row["n_bnd"] / 8.0),
                     float(row["time"]), float(row["inverse"]),
                     np.log10(float(row["init_err"]))], dtype=np.float64)


def level6(S):
    """Шесть чисел уровня карт (находка E): минимум и среднее трёх карт потерь."""
    F = np.asarray(S[:, :3], dtype=np.float64).reshape(len(S), 3, -1)
    return np.concatenate([F.min(2), F.mean(2)], 1)


def _log(x):
    return np.log10(np.clip(np.asarray(x, dtype=np.float64), 10.0 ** L_LO, 10.0 ** L_HI))


def build_table(subdirs=None, root=None, meta=None, episodes=None, verbose=True):
    """Переходы всех буферов одной таблицей. Цепочки восстанавливаются загрузчиком
    episodes_to_arrays_chains по одному файлу (так известен номер файла). Ошибка перед первым
    действием цепочки — E0 задачи из pde_meta.json. episodes: {папка: [файлы]} вместо диска."""
    import offline_rl as O
    meta = meta if meta is not None else O.pde_meta()
    sub2pde = {r["subdir"]: k for k, r in meta.items() if r.get("subdir")}
    if subdirs is None:
        subdirs = list(episodes) if episodes is not None else list(O.ALL_SUBDIRS)
    if episodes is None:
        root = root or buffer_root()
        if not root:
            raise SystemExit("нет буферов: задайте RL_BUFFER_DIR")
        os.environ["RL_BUFFER_DIR"] = root
    keys = ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "L", "LN", "LV", "LVN", "SPF", "LAST")
    cols = {k: [] for k in keys}
    names, ctxs, subs = [], [], []
    fid = cid = 0
    for sd in subdirs:
        name = sub2pde[sd]
        eps = episodes.get(sd, []) if episodes is not None else O.load_episodes(None, sd)
        if not eps:
            continue
        pi = len(names)
        e0 = float(meta[name]["init_err"])
        used = 0
        for ep in eps:
            try:
                d = O.episodes_to_arrays_chains([ep], reward_form="delta", verbose=False)
            except ValueError:          # в файле не осталось ни одной цепочки
                continue
            A, STEP, SPENT, ERR = d["A"], d["STEP"], d["SPENT"], d["ERR"].astype(np.float64)
            n = len(A)
            first = STEP == 0
            prev = np.where(first, e0, np.r_[e0, ERR[:-1]])
            pa = np.where(first, -1, np.r_[-1, A[:-1]])
            ch = np.cumsum(first) - 1 + cid
            last = np.r_[first[1:], True]
            spf = np.zeros((n, 3))
            for i in range(n):
                if not first[i]:
                    spf[i] = spf[i - 1]
                    spf[i, FAM[A[i - 1]]] += EPOCHS[A[i - 1]]
            lvn = level6(d["S2"])
            lvn[d["D"] == 1] = np.nan                  # у терминального перехода следующей карты нет
            for k, v in (("PDE", np.full(n, pi)), ("FILE", np.full(n, fid)), ("CH", ch),
                         ("STEP", STEP), ("SPENT", SPENT), ("A", A), ("PA", pa), ("L", _log(prev)),
                         ("LN", _log(ERR)), ("LV", level6(d["S"])), ("LVN", lvn), ("SPF", spf),
                         ("LAST", last)):
                cols[k].append(v)
            cid = int(ch[-1]) + 1
            fid += 1
            used += 1
        names.append(name)
        subs.append(sd)
        ctxs.append(pde_ctx(meta[name]))
        if verbose:
            print(f"  {sd}: файлов {used}", flush=True)
    tab = {k: np.concatenate(v, 0) for k, v in cols.items()}
    for k in ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA"):
        tab[k] = tab[k].astype(np.int64)
    tab["LAST"] = tab["LAST"].astype(bool)
    tab["CTX"] = np.stack(ctxs)
    tab["NAMES"] = names
    tab["SUBDIRS"] = subs
    return tab


def save_table(tab, path):
    np.savez_compressed(path, **{k: (np.array(v) if isinstance(v, list) else v) for k, v in tab.items()})


def load_table(path):
    z = np.load(path, allow_pickle=False)
    tab = {k: z[k] for k in z.files}
    tab["NAMES"] = [str(x) for x in tab["NAMES"]]
    tab["SUBDIRS"] = [str(x) for x in tab["SUBDIRS"]]
    return tab


def logged_chains(tab, mask=None, budget=BUDGET):
    """Записанные цепочки как статические: префикс действий, укладывающийся в бюджет, и log10
    ошибки после него. Возвращает dict: CH, PDE, FILE, chains (списки действий), final, spent."""
    rows = np.where(tab["SPENT"] + EPOCHS[tab["A"]] <= budget)[0]
    if mask is not None:
        rows = rows[mask[rows]]
    out = dict(CH=[], PDE=[], FILE=[], chains=[], final=[], spent=[])
    if not len(rows):
        return {k: np.array(v) for k, v in out.items()}
    ch = tab["CH"][rows]
    cut = np.r_[0, np.where(np.diff(ch) != 0)[0] + 1, len(rows)]
    for s, e in zip(cut[:-1], cut[1:]):
        r = rows[s:e]
        out["CH"].append(int(tab["CH"][r[0]]))
        out["PDE"].append(int(tab["PDE"][r[0]]))
        out["FILE"].append(int(tab["FILE"][r[0]]))
        out["chains"].append([int(a) for a in tab["A"][r]])
        out["final"].append(float(tab["LN"][r[-1]]))
        out["spent"].append(int(tab["SPENT"][r[-1]] + EPOCHS[tab["A"][r[-1]]]))
    res = {k: np.array(v) for k, v in out.items() if k != "chains"}
    res["chains"] = out["chains"]
    return res


# --------------------------------------------------------------------------------------------
# Модель одного шага
# --------------------------------------------------------------------------------------------

FEATS = (["a", "pa", "a_adam", "a_lbfgs", "a_pso", "a_loglr", "a_logep",
          "pa_adam", "pa_lbfgs", "pa_pso", "pa_loglr", "pa_logep", "same_fam", "same_a",
          "step", "spent", "sp_adam", "sp_lbfgs", "sp_pso", "l", "l_rel",
          "d_in", "d_out", "d_npde", "d_nbnd", "d_time", "d_inv", "log_e0"])
LEVEL_FEATS = ["lv_min_tot", "lv_min_op", "lv_min_bnd", "lv_mean_tot", "lv_mean_op", "lv_mean_bnd"]
NO_ERR = ("l", "l_rel")


def featurize(ctx, step, spent, spf, pa, l, a, lv=None, ep=None):
    """Признаки одного шага для пачки: ctx (n, 7), spf (n, 3) эпох по семействам до действия,
    pa — прошлое действие (-1 нет), l — log10 текущей ошибки, a — действие, lv (n, 6) — уровень
    карт (вариант level), ep — фактические эпохи действия (обрезка последнего по бюджету)."""
    a = np.asarray(a, dtype=np.int64)
    pa = np.asarray(pa, dtype=np.int64)
    pai = np.where(pa < 0, NONE, pa)
    ea = ENC[a].copy()
    if ep is not None:
        ea[:, 4] = np.log10(np.maximum(np.asarray(ep, dtype=np.float64), 1.0))
    same_fam = (pa >= 0) & (FAM[np.maximum(pa, 0)] == FAM[a])
    cols = [a[:, None], pai[:, None], ea, ENC[pai], same_fam[:, None], (pa == a)[:, None],
            np.asarray(step, dtype=np.float64)[:, None], np.asarray(spent, dtype=np.float64)[:, None],
            np.asarray(spf, dtype=np.float64), np.asarray(l, dtype=np.float64)[:, None],
            (np.asarray(l, dtype=np.float64) - ctx[:, 6])[:, None], ctx]
    if lv is not None:
        cols.append(np.asarray(lv, dtype=np.float64))
    return np.concatenate([np.asarray(c, dtype=np.float64) for c in cols], 1)


def table_features(tab, rows=None, level=False):
    rows = np.arange(len(tab["A"])) if rows is None else rows
    return featurize(tab["CTX"][tab["PDE"][rows]], tab["STEP"][rows], tab["SPENT"][rows],
                     tab["SPF"][rows], tab["PA"][rows], tab["L"][rows], tab["A"][rows],
                     lv=(tab["LV"][rows] if level else None))


def _hgb(seed, max_iter, lr, leaves, l2):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    return HGB(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
               categorical_features=[0, 1], early_stopping=False, random_state=seed)


class Ensemble:
    """Ансамбль HGB, члены учатся на бутстрепе цепочек (MBPO/PETS: член = модель среды).
    target='delta' — учится изменение log10 ошибки (по умолчанию), 'abs' — сама следующая
    ошибка (для признаков без текущей ошибки). level=True: на входе уровень карт и шесть
    дополнительных моделей на член предсказывают уровень после действия (для прогонов)."""

    def __init__(self, members=5, level=False, seed=0, max_iter=300, learning_rate=0.06,
                 max_leaf_nodes=31, l2=1.0, target="delta", drop=()):
        self.members, self.level, self.seed = members, level, seed
        self.hp = (max_iter, learning_rate, max_leaf_nodes, l2)
        self.target, self.drop = target, tuple(drop)
        names = FEATS + (LEVEL_FEATS if level else [])
        self.keep = np.array([n not in self.drop for n in names])
        self.models, self.lv_models = [], []

    def _x(self, X):
        return X[:, self.keep] if not self.keep.all() else X

    def fit(self, X, l, ln, groups, lvn=None):
        rng = np.random.default_rng(self.seed)
        ug, inv = np.unique(groups, return_inverse=True)
        y = ln - l if self.target == "delta" else ln
        Xk = self._x(X)
        for m in range(self.members):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv] \
                if self.members > 1 else np.ones(len(inv), dtype=np.int64)
            use = w > 0
            h = _hgb(self.seed * 101 + m, *self.hp)
            h.fit(Xk[use], y[use], sample_weight=w[use].astype(np.float64))
            self.models.append(h)
            if self.level and lvn is not None:
                ok = use & np.isfinite(lvn).all(1)
                lvm = []
                for j in range(lvn.shape[1]):
                    g = _hgb(self.seed * 101 + m + 7 * (j + 1), *self.hp)
                    g.fit(Xk[ok], lvn[ok, j], sample_weight=w[ok].astype(np.float64))
                    lvm.append(g)
                self.lv_models.append(lvm)
        return self

    def predict_ln(self, X, l, member=None):
        """log10 ошибки после действия: (члены, n) или (n,) для одного члена."""
        Xk = self._x(X)
        ms = range(self.members) if member is None else [member]
        out = []
        for m in ms:
            p = self.models[m].predict(Xk)
            out.append(np.clip((l + p) if self.target == "delta" else p, L_LO, L_HI))
        return np.stack(out) if member is None else out[0]

    def predict_lv(self, X, member):
        Xk = self._x(X)
        return np.stack([g.predict(Xk) for g in self.lv_models[member]], 1)


def fit_ensemble(tab, rows, level=False, members=5, seed=0, quick=False, target="delta", drop=()):
    X = table_features(tab, rows, level=level)
    hp = dict(max_iter=60, learning_rate=0.15) if quick else {}
    ens = Ensemble(members=members, level=level, seed=seed, target=target, drop=drop, **hp)
    return ens.fit(X, tab["L"][rows], tab["LN"][rows], tab["CH"][rows],
                   lvn=(tab["LVN"][rows] if level else None))


# --------------------------------------------------------------------------------------------
# Прогон в модели
# --------------------------------------------------------------------------------------------

def init_state(ctx, n, members, history=(), l=None, lv=None):
    """Состояние пачки из n одинаковых цепочек после исполненной истории history. l — наблюдаемая
    log10 ошибка (по умолчанию E0 задачи: старт цепочки), lv — наблюдаемый уровень карт."""
    ctx = np.asarray(ctx, dtype=np.float64)
    spf = np.zeros(3)
    for a in history:
        spf[FAM[a]] += EPOCHS[a]
    l0 = ctx[6] if l is None else float(l)
    lv0 = np.zeros(6) if lv is None else np.asarray(lv, dtype=np.float64)
    return dict(ctx=np.tile(ctx, (n, 1)), step=np.full(n, len(history)),
                spent=np.full(n, int(spf.sum())), spf=np.tile(spf, (n, 1)),
                pa=np.full(n, history[-1] if len(history) else -1),
                l=np.full((members, n), l0), lv=np.tile(lv0, (members, n, 1)))


def take(st, idx):
    return {k: (v[:, idx] if k in ("l", "lv") else v[idx]) for k, v in st.items()}


def advance(ens, st, a, ep=None):
    """Один шаг всех цепочек пачки действиями a (n,): каждый член ансамбля двигает свою копию
    состояния. ep — фактические эпохи (если действие обрезано бюджетом)."""
    a = np.asarray(a, dtype=np.int64)
    epa = EPOCHS[a] if ep is None else np.asarray(ep, dtype=np.int64)
    new_l = np.empty_like(st["l"])
    new_lv = st["lv"].copy()
    for m in range(ens.members):
        X = featurize(st["ctx"], st["step"], st["spent"], st["spf"], st["pa"], st["l"][m], a,
                      lv=(st["lv"][m] if ens.level else None), ep=ep)
        new_l[m] = ens.predict_ln(X, st["l"][m], member=m)
        if ens.level:
            new_lv[m] = ens.predict_lv(X, m)
    spf = st["spf"].copy()
    spf[np.arange(len(a)), FAM[a]] += epa
    return dict(ctx=st["ctx"], step=st["step"] + 1, spent=st["spent"] + epa, spf=spf, pa=a,
                l=new_l, lv=new_lv)


def rollout(ens, ctxs, chains, eps=None):
    """Прогон фиксированных цепочек (разной длины) из старта. ctxs: (n, 7) или (7,).
    Возвращает log10 итоговой ошибки по членам (члены, n)."""
    n = len(chains)
    ctxs = np.asarray(ctxs, dtype=np.float64)
    if ctxs.ndim == 1:
        ctxs = np.tile(ctxs, (n, 1))
    st = init_state(ctxs[0], n, ens.members)
    st["ctx"] = ctxs.copy()
    st["l"] = np.tile(ctxs[:, 6], (ens.members, 1))
    lens = np.array([len(c) for c in chains])
    for t in range(int(lens.max()) if n else 0):
        act = np.where(lens > t)[0]
        a = np.array([chains[i][t] for i in act])
        e = None if eps is None else np.array([eps[i][t] for i in act])
        sub = advance(ens, take(st, act), a, ep=e)
        for k in st:
            if k in ("l", "lv"):
                st[k][:, act] = sub[k]
            elif k != "ctx":
                st[k][act] = sub[k]
    return st["l"]


# --------------------------------------------------------------------------------------------
# Планирование: лучевой поиск по цепочкам (полный перебор, пока фронт меньше cap)
# --------------------------------------------------------------------------------------------

def plan(ens, ctx, budget=BUDGET, k=0.0, width=256, cap=30000, max_len=K_MAX, allowed=None,
         consistent=None, start=None, top=5, keep_depth=0):
    """Цепочки, минимизирующие J = среднее по членам log10 итоговой ошибки + k * разброс по членам
    (MOPO). Остановиться можно после любого действия. Фронт перебирается целиком, пока в нём не
    больше cap цепочек, дальше — лучевой поиск ширины width по J.
    start: dict(history, l, lv) — планировать продолжение из наблюдённого состояния (MPC); тогда
    вариант «не делать ничего» тоже участвует (stop). keep_depth > 0: вернуть J всех цепочек до
    этой длины (место заданной цепочки среди всех).
    Возвращает dict(best=[(цепочка, среднее, разброс, J)], allJ=массив, stop_J)."""
    allowed = np.ones(27, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    acts = np.where(allowed)[0]
    hist = tuple(start["history"]) if start else ()
    st = init_state(ctx, 1, ens.members, history=hist,
                    l=(start.get("l") if start else None), lv=(start.get("lv") if start else None))
    chains = [()]
    pool = []                           # (J, mean, std, цепочка)
    short = []                          # то же среди цепочек не длиннее keep_depth (полный перебор)
    stop_J = None
    if hist:
        stop_J = float(st["l"][:, 0].mean())
        pool.append((stop_J, stop_J, 0.0, ()))
    allJ = []
    for depth in range(max_len - len(hist)):
        par = np.repeat(np.arange(len(chains)), len(acts))
        a = np.tile(acts, len(chains))
        ok = st["spent"][par] + EPOCHS[a] <= budget
        ok &= consistent_ok(st["pa"][par], a, consistent)
        par, a = par[ok], a[ok]
        if not len(a):
            break
        ch = advance(ens, take(st, par), a)
        mu, sd = ch["l"].mean(0), ch["l"].std(0)
        J = mu + k * sd
        kids = [chains[p] + (int(x),) for p, x in zip(par, a)]
        best = np.argsort(J)[:max(top * 20, 100)]
        found = [(float(J[i]), float(mu[i]), float(sd[i]), kids[i]) for i in best]
        pool = sorted(pool + found, key=lambda r: r[0])[:max(top * 20, 100)]
        if depth < keep_depth:
            allJ.append(J)
            short = sorted(short + found, key=lambda r: r[0])[:top]
        keep = np.arange(len(a)) if len(a) <= cap else np.argsort(J)[:width]
        st = take(ch, keep)
        chains = [kids[i] for i in keep]
    return dict(best=[(list(c), m, s, j) for j, m, s, c in pool[:top]],
                best_short=[(list(c), m, s, j) for j, m, s, c in short],
                allJ=(np.concatenate(allJ) if allJ else np.zeros(0)), stop_J=stop_J)


def chain_score(ens, ctx, chain, k=0.0):
    """Среднее, разброс и J одной цепочки в модели."""
    L = rollout(ens, ctx, [list(chain)])[:, 0]
    return float(L.mean()), float(L.std()), float(L.mean() + k * L.std())


def plan_next(ens, ctx, history, l_obs, lv_obs=None, budget=BUDGET, k=0.0, width=48, cap=800,
              max_len=K_MAX, allowed=None, consistent=None):
    """Замкнутый контур (MPC): после исполненных действий history и наблюдённой log10 ошибки
    l_obs (и уровня карт lv_obs для варианта level) перепланировать остаток и вернуть первое
    действие лучшего плана. action=None — остановиться: по модели ни одно продолжение не
    лучше текущей ошибки, или бюджет и шаги кончились.
    Возвращает dict(action, plan, mean, std, J, stop_J)."""
    history = [int(x) for x in history]
    if l_obs is None and history:
        raise ValueError("после первого действия нужна наблюдённая ошибка l_obs")
    if sum(EPOCHS[a] for a in history) >= budget or len(history) >= max_len:
        return dict(action=None, plan=[], mean=l_obs, std=0.0, J=l_obs, stop_J=l_obs)
    start = dict(history=history, l=l_obs, lv=lv_obs) if history else None
    r = plan(ens, ctx, budget=budget, k=k, width=width, cap=cap, max_len=max_len,
             allowed=allowed, consistent=consistent, start=start, top=1)
    if not r["best"]:
        return dict(action=None, plan=[], mean=l_obs, std=0.0, J=l_obs, stop_J=r["stop_J"])
    c, m, s, j = r["best"][0]
    return dict(action=(c[0] if c else None), plan=c, mean=m, std=s, J=j, stop_J=r["stop_J"])


# --------------------------------------------------------------------------------------------
# Строки GPU-оценки (внешняя проверка): среда со сбросом, бюджет 7000
# --------------------------------------------------------------------------------------------

def eval_rows_dirs():
    return glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--danil-e--pinnacle-optuna-db/snapshots/*/rl_arch/online_env"))


def load_eval_rows(pdes=MAIN, dirs=None, budget=BUDGET):
    """Строки оценки (последняя версия каждого файла среди снимков кэша HF), пригодные для
    проверки модели: бюджет 7000, без стража, разведки, толчков, последнего слоя, допусков
    L-BFGS и двойной точности, все действия из 27 (эпохи последнего могут быть обрезаны)."""
    files = {}
    for d in (dirs if dirs is not None else eval_rows_dirs()):
        for p in glob.glob(os.path.join(d, "*.json")):
            b = os.path.basename(p)
            if b not in files or os.path.getmtime(p) > os.path.getmtime(files[b]):
                files[b] = p
    out = []
    for b, p in sorted(files.items()):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("pde") not in pdes or r.get("budget") != budget or r.get("smoke") or r.get("partial"):
            continue
        if r.get("boosted") or r.get("boost_trigger") not in (None, "none") or r.get("guard_spent") \
                or r.get("scout_spent") or r.get("ls") or r.get("lbfgs_tol") is not None or "f64" in b:
            continue
        if not r.get("chain") or not np.isfinite(r.get("rmse", np.inf)) or not np.isfinite(r.get("brmse", np.inf)):
            continue
        acts, eps, bad = [], [], False
        for o, lr, ep in r["chain"]:
            hit = [i for i, (on, l, e) in enumerate(ACTIONS) if on == o and abs(l - float(lr)) < 1e-12]
            if not hit:
                bad = True
                break
            i = min(hit, key=lambda j: abs(np.log10(EPOCHS[j]) - np.log10(max(float(ep), 1.0))))
            acts.append(i)
            eps.append(int(ep))
        if bad or sum(eps) > budget:
            continue
        arm = re.sub(r"_seed\d+\.json$", "", b)[len(r["pde"]) + 1:]
        out.append(dict(arm=arm, pde=r["pde"], seed=r.get("seed"), policy=r.get("policy"),
                        keep=bool(r.get("keep_opt")), chain=acts, eps=eps,
                        err=float(r["rmse"] + r["brmse"]), l2re=float(r["l2re"])))
    return out


# --------------------------------------------------------------------------------------------
# Оценки
# --------------------------------------------------------------------------------------------

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    v = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / v) if v > 0 else float("nan")


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def chain_folds(tab, n_splits=5, seed=0):
    """Номер фолда каждой цепочки: GroupKFold по цепочкам со случайной перестановкой групп."""
    from sklearn.model_selection import GroupKFold
    n_ch = int(tab["CH"].max()) + 1
    perm = np.random.default_rng(seed).permutation(n_ch)
    fold = np.zeros(n_ch, dtype=np.int64)
    grp = perm[tab["CH"]]
    for f, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(tab["A"], groups=grp)):
        fold[np.unique(tab["CH"][te])] = f
    return fold


def lopo_pdes(tab):
    files = {i: len(np.unique(tab["FILE"][tab["PDE"] == i])) for i in range(len(tab["NAMES"]))}
    return [i for i, n in files.items() if n >= LOPO_MIN_FILES]


class Work:
    """Кэш таблицы и обученных моделей в --work (подкоманды переиспользуют друг друга)."""

    def __init__(self, args):
        self.args = args
        self.dir = args.work
        os.makedirs(self.dir, exist_ok=True)
        self._tab = None

    def path(self, name):
        return os.path.join(self.dir, name)

    def table(self):
        if self._tab is None:
            p = self.path("surrogate_table.npz")
            if os.path.exists(p) and not self.args.rebuild:
                self._tab = load_table(p)
            else:
                t0 = time.time()
                subs = self.args.subdirs.split(",") if self.args.subdirs else None
                tab = build_table(subdirs=subs)
                save_table(tab, p)
                self._tab = load_table(p)
                print(f"таблица собрана за {time.time() - t0:.0f} с: {p}", flush=True)
        return self._tab

    def model(self, key, fit):
        p = self.path(f"surrogate_model_{key}{'_q' if self.args.quick else ''}.pkl")
        if os.path.exists(p) and not self.args.refit:
            with open(p, "rb") as f:
                return pickle.load(f)
        t0 = time.time()
        m = fit()
        with open(p, "wb") as f:
            pickle.dump(m, f)
        print(f"  модель {key}: {time.time() - t0:.0f} с", flush=True)
        return m

    def cv_models(self, level=False):
        tab = self.table()
        fold = chain_folds(tab, seed=self.args.seed)
        ms = []
        for f in range(5):
            rows = np.where(fold[tab["CH"]] != f)[0]
            ms.append(self.model(f"cv{f}{'_lv' if level else ''}",
                                 lambda rows=rows: fit_ensemble(tab, rows, level=level, seed=self.args.seed,
                                                                quick=self.args.quick)))
        return fold, ms

    def full_model(self, level=False):
        tab = self.table()
        rows = np.arange(len(tab["A"]))
        return self.model(f"full{'_lv' if level else ''}",
                          lambda: fit_ensemble(tab, rows, level=level, seed=self.args.seed,
                                               quick=self.args.quick))

    def half_model(self, h):
        """Модель на половине цепочек (все задачи): план одной половиной, оценка другой."""
        tab = self.table()
        perm = np.random.default_rng(self.args.seed + 1).permutation(int(tab["CH"].max()) + 1)
        rows = np.where(perm[tab["CH"]] % 2 == h)[0]
        return self.model(f"half{h}", lambda: fit_ensemble(tab, rows, seed=self.args.seed + 10 * h,
                                                           quick=self.args.quick))

    def lopo_model(self, pi):
        tab = self.table()
        rows = np.where(tab["PDE"] != pi)[0]
        return self.model(f"lopo_{tab['NAMES'][pi]}",
                          lambda: fit_ensemble(tab, rows, seed=self.args.seed, quick=self.args.quick))

    def dump(self, name, obj):
        with open(self.path(f"surrogate_{name}.json"), "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


def cmd_table(w):
    tab = w.table()
    lc = logged_chains(tab)
    print("\nШаг 1. Таблица переходов (буферы среды со сбросом; итог — префикс цепочки в 7000 эпох)")
    print(f"{'УрЧП':26s} {'файлов':>6s} {'цепочек':>7s} {'переходов':>9s} {'≥6000эп':>7s} "
          f"{'E0':>8s} {'медиана итога':>13s} {'лучший итог':>11s} {'ур.карт':>7s}")
    out = {}
    for i, name in enumerate(tab["NAMES"]):
        m = tab["PDE"] == i
        sel = lc["PDE"] == i
        full = sel & (lc["spent"] >= 6000)
        fin = 10 ** lc["final"][full] if full.any() else np.array([np.nan])
        row = dict(files=int(len(np.unique(tab["FILE"][m]))), chains=int(len(np.unique(tab["CH"][m]))),
                   transitions=int(m.sum()), chains_6000=int(full.sum()),
                   e0=float(10 ** tab["CTX"][i, 6]), median_final=float(np.median(fin)),
                   best_final=float(np.min(fin)),
                   level_next_known=float(np.isfinite(tab["LVN"][m]).all(1).mean()))
        out[name] = row
        print(f"{name:26s} {row['files']:6d} {row['chains']:7d} {row['transitions']:9d} {row['chains_6000']:7d} "
              f"{row['e0']:8.3g} {row['median_final']:13.4g} {row['best_final']:11.4g} "
              f"{row['level_next_known']:7.2f}")
    print(f"всего: УрЧП {len(tab['NAMES'])}, файлов {len(np.unique(tab['FILE']))}, цепочек "
          f"{len(np.unique(tab['CH']))}, переходов {len(tab['A'])}")
    w.dump("table", out)
    return out


def cmd_onestep(w):
    """Шаг 2: R² следующей log10 ошибки. GroupKFold(5) по цепочкам на всех УрЧП вместе; столбцы:
    персистентность (ошибка не меняется), без текущей ошибки (как G1 находки E: действие, шаг,
    прошлое действие, задача), модель ансамбля, она же с уровнем карт; R² изменения ошибки;
    «исключить УрЧП» для задач с ≥40 файлами; модель только на своей задаче для ns2d и pb2d."""
    tab = w.table()
    n = len(tab["A"])
    fold, ms = w.cv_models()
    _, ms_lv = w.cv_models(level=True)
    X = table_features(tab)
    Xl = table_features(tab, level=True)
    P = np.zeros((5, n))
    Pl = np.zeros(n)
    Pn = np.zeros(n)
    Sd = np.zeros(n)
    for f in range(5):
        te = np.where(fold[tab["CH"]] == f)[0]
        p = ms[f].predict_ln(X[te], tab["L"][te])
        P[:, te] = p
        Sd[te] = p.std(0)
        Pl[te] = ms_lv[f].predict_ln(Xl[te], tab["L"][te]).mean(0)
        tr = np.where(fold[tab["CH"]] != f)[0]
        ne = w.model(f"cv{f}_noerr", lambda tr=tr: fit_ensemble(tab, tr, members=1, target="abs",
                                                                drop=NO_ERR, seed=w.args.seed,
                                                                quick=w.args.quick))
        Pn[te] = ne.predict_ln(X[te], tab["L"][te]).mean(0)
    Pm = P.mean(0)
    lopo = {}
    for pi in lopo_pdes(tab):
        te = np.where(tab["PDE"] == pi)[0]
        lopo[pi] = w.lopo_model(pi).predict_ln(X[te], tab["L"][te]).mean(0)
    own = {}
    for name in MAIN:
        if name not in tab["NAMES"]:
            continue
        pi = tab["NAMES"].index(name)
        po = np.zeros(n)
        for f in range(5):
            tr = np.where((fold[tab["CH"]] != f) & (tab["PDE"] == pi))[0]
            te = np.where((fold[tab["CH"]] == f) & (tab["PDE"] == pi))[0]
            mo = w.model(f"cv{f}_own_{name}", lambda tr=tr: fit_ensemble(tab, tr, seed=w.args.seed,
                                                                          quick=w.args.quick))
            po[te] = mo.predict_ln(X[te], tab["L"][te]).mean(0)
        own[pi] = po
    print("\nШаг 2. Модель одного шага, R² следующей log10 ошибки (внефолдово; GroupKFold по цепочкам)")
    print(f"{'УрЧП':26s} {'n':>6s} {'персист':>7s} {'без ошибки':>10s} {'ансамбль':>8s} {'+уровень':>8s} "
          f"{'R²(Δ)':>6s} {'свой':>6s} {'LOPO':>6s} {'MAE':>5s}")
    out = {}
    for i, name in enumerate(tab["NAMES"]):
        m = tab["PDE"] == i
        y, l = tab["LN"][m], tab["L"][m]
        row = dict(n=int(m.sum()), persist=r2(y, l), noerr=r2(y, Pn[m]), ens=r2(y, Pm[m]),
                   level=r2(y, Pl[m]), delta=r2(y - l, Pm[m] - l), mae=float(np.abs(y - Pm[m]).mean()),
                   own=(r2(y, own[i][m]) if i in own else None),
                   lopo=(r2(y, lopo[i]) if i in lopo else None),
                   lopo_delta=(r2(y - l, lopo[i] - l) if i in lopo else None),
                   sd_vs_abs=spearman(Sd[m], np.abs(y - Pm[m])))
        out[name] = row
        f = lambda v: "     -" if v is None else f"{v:6.3f}"
        print(f"{name:26s} {row['n']:6d} {row['persist']:7.3f} {row['noerr']:10.3f} {row['ens']:8.3f} "
              f"{row['level']:8.3f} {row['delta']:6.3f} {f(row['own'])} {f(row['lopo'])} {row['mae']:5.3f}")
    allm = np.ones(n, dtype=bool)
    out["_all"] = dict(persist=r2(tab["LN"], tab["L"]), noerr=r2(tab["LN"], Pn), ens=r2(tab["LN"], Pm),
                       level=r2(tab["LN"], Pl), delta=r2(tab["LN"] - tab["L"], Pm - tab["L"]),
                       sd_vs_abs=spearman(Sd[allm], np.abs(tab["LN"] - Pm)))
    print(f"все вместе: персист {out['_all']['persist']:.3f}, без ошибки {out['_all']['noerr']:.3f}, "
          f"ансамбль {out['_all']['ens']:.3f}, +уровень {out['_all']['level']:.3f}, R²(Δ) {out['_all']['delta']:.3f}; "
          f"Spearman(разброс ансамбля, |ошибка|) {out['_all']['sd_vs_abs']:.2f}")
    w.dump("onestep", out)
    return out


def _chain_metrics(pred, sd, actual):
    """Сводка многошагового прогноза по цепочкам одной задачи."""
    pred, sd, actual = map(np.asarray, (pred, sd, actual))
    n = len(actual)
    if n < 5:
        return dict(n=n)
    q = np.argsort(np.argsort(actual)) / max(n - 1, 1)          # процентиль факта (0 = лучший)
    top = actual <= np.quantile(actual, 0.25)
    pick = int(np.argmin(pred))
    top10 = np.argsort(pred)[:max(1, n // 10)]
    absd = np.abs(pred - actual)
    return dict(n=n, spearman=spearman(pred, actual), spearman_top25=spearman(pred[top], actual[top]),
                mae=float(absd.mean()), bias=float((pred - actual).mean()),
                pick_pct=float(q[pick]), top10_pct=float(q[top10].mean()),
                sd_vs_abs=spearman(sd, absd), cover2sd=float((absd <= 2 * np.maximum(sd, 1e-9)).mean()),
                actual_sd=float(actual.std()))


def cmd_multistep(w, level_too=True):
    """Шаг 3: цепочки отложенных фолдов (и отложенной задачи) прогоняются через модель от старта
    по записанным действиям; сравнивается итоговая log10 ошибка после префикса в 7000 эпох."""
    tab = w.table()
    lc = logged_chains(tab)
    fold, ms = w.cv_models()
    variants = [("err", ms)]
    if level_too:
        variants.append(("level", w.cv_models(level=True)[1]))
    ctx_of = tab["CTX"][lc["PDE"]]
    res = {}
    for vname, models in variants:
        pred = np.zeros(len(lc["CH"]))
        sd = np.zeros(len(lc["CH"]))
        for f in range(5):
            idx = np.where(fold[lc["CH"]] == f)[0]
            L = rollout(models[f], ctx_of[idx], [lc["chains"][i] for i in idx])
            pred[idx], sd[idx] = L.mean(0), L.std(0)
        res[vname] = (pred, sd)
    lopo = {}
    for pi in lopo_pdes(tab):
        idx = np.where(lc["PDE"] == pi)[0]
        L = rollout(w.lopo_model(pi), ctx_of[idx], [lc["chains"][i] for i in idx])
        lopo[pi] = (idx, L.mean(0), L.std(0))
    print("\nШаг 3. Многошаговый прогноз: записанные действия от старта, итог после префикса ≤7000 эпох")
    print(f"{'УрЧП':26s} {'цеп':>5s} {'ρ':>6s} {'ρ топ25%':>8s} {'ρ уров':>6s} {'MAE':>5s} {'сдвиг':>6s} "
          f"{'выбор':>5s} {'топ10%':>6s} {'ρ(sd,|e|)':>9s} | {'LOPO ρ':>6s} {'ρ топ25':>7s} {'сдвиг':>6s} {'выбор':>5s}")
    out = {}
    for i, name in enumerate(tab["NAMES"]):
        sel = np.where(lc["PDE"] == i)[0]
        pe, se = res["err"][0][sel], res["err"][1][sel]
        mrow = _chain_metrics(pe, se, lc["final"][sel])
        if "level" in res:
            mrow["spearman_level"] = spearman(res["level"][0][sel], lc["final"][sel])
        if i in lopo:
            idx, pm, ps = lopo[i]
            mrow["lopo"] = _chain_metrics(pm, ps, lc["final"][idx])
        out[name] = mrow
        if mrow["n"] < 5:
            continue
        lo = mrow.get("lopo", {})
        f = lambda v, fmt="6.2f": format(v, fmt) if v is not None else "-".rjust(int(fmt.split(".")[0]))
        print(f"{name:26s} {mrow['n']:5d} {mrow['spearman']:6.2f} {mrow['spearman_top25']:8.2f} "
              f"{mrow.get('spearman_level', float('nan')):6.2f} {mrow['mae']:5.2f} {mrow['bias']:6.2f} "
              f"{mrow['pick_pct']:5.2f} {mrow['top10_pct']:6.2f} {mrow['sd_vs_abs']:9.2f} | "
              f"{f(lo.get('spearman'))} {f(lo.get('spearman_top25'), '7.2f')} {f(lo.get('bias'))} "
              f"{f(lo.get('pick_pct'), '5.2f')}")
    print("ρ — Spearman прогноза и факта по цепочкам задачи; «выбор» — процентиль факта у цепочки с лучшим "
          "прогнозом (0 = лучшая из записанных); топ10% — средний процентиль факта у 10% лучших по прогнозу")
    w.dump("multistep", out)
    return out


def _k_l2re(rows):
    """l2re / (rmse + brmse) по строкам оценки задачи (перевод ошибки буфера в l2re)."""
    k = [r["l2re"] / r["err"] for r in rows if r["err"] > 0]
    return float(np.median(k)) if k else 1.0


def _prefix_keys(tab, pi, fold):
    """Префиксы цепочек буфера задачи -> фолд цепочки. Последнее действие префикса берётся по
    оптимизатору и шагу (a // 3): на оценке его эпохи могли быть обрезаны бюджетом (так в армы
    t3b_log* попал шаг, пересекающий 7000 эпох)."""
    rows = np.where(tab["PDE"] == pi)[0]
    keys = {}
    ch, a = tab["CH"][rows], tab["A"][rows]
    cut = np.r_[0, np.where(np.diff(ch) != 0)[0] + 1, len(rows)]
    for s, e in zip(cut[:-1], cut[1:]):
        acts = [int(x) for x in a[s:e]]
        f = int(fold[ch[s]])
        for t in range(1, len(acts) + 1):
            keys.setdefault(tuple(acts[:t - 1]) + (acts[t - 1] // 3,), f)
    return keys


def cmd_gpucheck(w):
    """Внешняя проверка: цепочки, исполненные на GPU (строки оценки), прогоняются через модель;
    сравнивается медиана по сидам log10(rmse + brmse) по армам и отдельные прогоны. Берутся цепочки
    не длиннее K_MAX действий (длиннее в буфере не бывает). Прогноз перекрёстный: цепочку, которая
    целиком есть в буфере (армы t3b_log*), предсказывает модель фолда, где эта цепочка отложена,
    остальные — среднее пяти моделей фолдов; для сравнения — полная модель (видела все цепочки)."""
    tab = w.table()
    ens = w.full_model()
    fold, ms = w.cv_models()
    rows = load_eval_rows()
    out = {}
    print("\nПроверка на строках GPU-оценки (бюджет 7000, цепочки ≤%d действий)" % K_MAX)
    for name in MAIN:
        if name not in tab["NAMES"]:
            continue
        pi = tab["NAMES"].index(name)
        ctx = tab["CTX"][pi]
        long_ = sum(1 for r in rows if r["pde"] == name and len(r["chain"]) > K_MAX)
        rr = [r for r in rows if r["pde"] == name and len(r["chain"]) <= K_MAX]
        if not rr:
            continue
        logged = _prefix_keys(tab, pi, fold)
        chains, eps = [r["chain"] for r in rr], [r["eps"] for r in rr]
        L = rollout(ens, ctx, chains, eps=eps)
        Lf = np.stack([rollout(m, ctx, chains, eps=eps).mean(0) for m in ms])      # (фолды, n)
        for j, (r, p, s) in enumerate(zip(rr, L.mean(0), L.std(0))):
            f = logged.get(tuple(r["chain"][:-1]) + (r["chain"][-1] // 3,))
            r["in_buffer"] = f is not None
            r["full"], r["sd"] = float(p), float(s)
            r["pred"] = float(Lf[f, j]) if f is not None else float(Lf[:, j].mean())
        print(f"{name}: строк {len(rr)} (длиннее {K_MAX} действий отброшено {long_}), цепочек из буфера "
              f"{sum(r['in_buffer'] for r in rr)}")
        res = {}
        for env, keep in (("со сбросом", False), ("без сброса", True)):
            sub = [r for r in rr if r["keep"] == keep]
            arms = {}
            for r in sub:
                arms.setdefault(r["arm"], []).append(r)
            stat = [a for a, v in arms.items() if v[0]["policy"] == "script" and len(v) >= 3]
            noise = [float(np.std(np.log10([r["err"] for r in arms[a]]))) for a in stat]
            fact = {a: float(np.median(np.log10([r["err"] for r in v]))) for a, v in arms.items() if len(v) >= 3}
            pr = {a: float(np.median([r["pred"] for r in arms[a]])) for a in fact}
            pf = {a: float(np.median([r["full"] for r in arms[a]])) for a in fact}
            sa = [a for a in stat if a in fact]
            sn = [a for a in sa if not arms[a][0]["in_buffer"]]
            best = min(sa, key=lambda a: fact[a]) if sa else None
            res[env] = dict(
                rows=len(sub), arms=len(fact), static_arms=len(sa), static_new=len(sn),
                rho_rows=spearman([r["pred"] for r in sub], np.log10([r["err"] for r in sub])),
                rho_arms=spearman([pr[a] for a in fact], [fact[a] for a in fact]),
                rho_static=spearman([pr[a] for a in sa], [fact[a] for a in sa]),
                rho_static_full=spearman([pf[a] for a in sa], [fact[a] for a in sa]),
                rho_static_new=spearman([pr[a] for a in sn], [fact[a] for a in sn]),
                bias_static=float(np.mean([pr[a] - fact[a] for a in sa])) if sa else None,
                mae_static=float(np.mean([abs(pr[a] - fact[a]) for a in sa])) if sa else None,
                seed_sd=float(np.median(noise)) if noise else None,
                arm_sd=float(np.std([fact[a] for a in sa])) if sa else None,
                model_pick=(min(sa, key=lambda a: pr[a]) if sa else None), fact_best=best,
                static={a: dict(fact=fact[a], pred=pr[a], full=pf[a], n=len(arms[a]),
                                in_buffer=arms[a][0]["in_buffer"],
                                chain=chain_str(arms[a][0]["chain"])) for a in sa})
        res["K_l2re"] = _k_l2re([r for r in rr if not r["keep"]])
        out[name] = res
        for env in ("со сбросом", "без сброса"):
            x = res[env]
            if not x["rows"]:
                continue
            print(f"{name} {env}: прогонов {x['rows']}, армов {x['arms']} (статических {x['static_arms']}, "
                  f"не из буфера {x['static_new']}); ρ по прогонам {x['rho_rows']:.2f}, по армам {x['rho_arms']:.2f}, "
                  f"по статическим {x['rho_static']:.2f} (полная модель {x['rho_static_full']:.2f}, только не из "
                  f"буфера {x['rho_static_new']:.2f}); сдвиг {x['bias_static']:+.2f}, MAE {x['mae_static']:.2f} дек.; "
                  f"разброс сидов {x['seed_sd']:.2f}, разброс армов {x['arm_sd']:.2f} дек.; модель выбрала "
                  f"{x['model_pick']}, лучший по факту {x['fact_best']}")
        print(f"{name}: l2re/(rmse+brmse) = {res['K_l2re']:.2f}")
        st = res["со сбросом"]["static"]
        for a in sorted(st, key=lambda a: st[a]["fact"])[:40]:
            print(f"   {a:14s} факт {10 ** st[a]['fact']:.4f} прогноз {10 ** st[a]['pred']:.4f} "
                  f"(полная {10 ** st[a]['full']:.4f}) {'буфер ' if st[a]['in_buffer'] else ''}"
                  f"(n={st[a]['n']}) {st[a]['chain'][:80]}")
    w.dump("gpucheck", out)
    return out


def _measured(name):
    return [(t, parse_chain(s), note) for t, s, note in MEASURED.get(name, [])]


def _gpu_k(w):
    p = w.path("surrogate_gpucheck.json")
    if os.path.exists(p):
        return {k: v.get("K_l2re", 1.0) for k, v in json.load(open(p)).items()}
    return {}


def _support(tab, chain, pi):
    """Сколько цепочек буфера (всех задач и своей) начинаются с тех же двух действий."""
    first = np.where(tab["STEP"] == 0)[0]
    nxt = first + 1
    ok = (nxt < len(tab["A"])) & (tab["CH"][np.minimum(nxt, len(tab["A"]) - 1)] == tab["CH"][first])
    f, n = first[ok], nxt[ok]
    if len(chain) < 2:
        hit = tab["A"][first] == chain[0]
        return int(hit.sum()), int((hit & (tab["PDE"][first] == pi)).sum())
    hit = (tab["A"][f] == chain[0]) & (tab["A"][n] == chain[1])
    return int(hit.sum()), int((hit & (tab["PDE"][f] == pi)).sum())


def _is_safe(chain):
    """Цепочка исполняется одинаково со сбросом и без (--keep-opt-mode safe)."""
    chain = [int(a) for a in chain]
    return not chain or bool(consistent_ok(np.r_[-1, chain[:-1]], np.asarray(chain), "safe").all())


def cmd_plan(w, ks=(0.0, 1.0), modes=(None, "safe")):
    """Шаг 4: планирование для ns2d и pb2d полной моделью; место замеренных цепочек среди всех
    цепочек до четырёх действий (полный перебор) и прогноз для них."""
    tab = w.table()
    ens = w.full_model()
    Kl = _gpu_k(w)
    allowed = parse_mask(w.args.mask) if w.args.mask else None
    out = {}
    for name in MAIN:
        if name not in tab["NAMES"]:
            continue
        pi = tab["NAMES"].index(name)
        ctx = tab["CTX"][pi]
        K = Kl.get(name, 1.0)
        res = {}
        for mode in modes:
            for k in ks:
                t0 = time.time()
                r = plan(ens, ctx, k=k, width=w.args.width, cap=w.args.cap, allowed=allowed,
                         consistent=mode, top=5, keep_depth=4)
                allJ = r["allJ"]
                meas = []
                for tag, ch, note in _measured(name):
                    m, s, j = chain_score(ens, ctx, ch, k=k)
                    meas.append(dict(tag=tag, chain=chain_str(ch), mean=m, sd=s, J=j,
                                     rank4=int((allJ < j).sum()) + 1, of=int(len(allJ)), note=note,
                                     safe=_is_safe(ch)))
                row = lambda c, m, s, j: dict(chain=chain_str(c), mean=m, sd=s, J=j, len=len(c),
                                              epochs=int(sum(EPOCHS[a] for a in c)), safe=_is_safe(c),
                                              support=_support(tab, c, pi), l2re=K * 10 ** m)
                key = f"{mode or 'все'}|k={k:g}"
                res[key] = dict(best=[row(*x) for x in r["best"]], short=[row(*x) for x in r["best_short"]],
                                measured=meas, n4=int(len(allJ)), sec=time.time() - t0)
                print(f"\n{name}: ограничение {mode or 'нет'}, k={k:g} ({time.time() - t0:.0f} с; "
                      f"цепочек до 4 действий: {len(allJ)}); l2re ≈ {K:.2f}·ошибка; "
                      f"опора — цепочек буфера с теми же двумя первыми действиями (все задачи / своя)")
                for tag, lst in (("лучшие", res[key]["best"]), ("до 4 действий", res[key]["short"][:3])):
                    print(f"  {tag}:")
                    for x in lst:
                        print(f"   J={x['J']:+.3f}  ошибка {10 ** x['mean']:.4g} (l2re≈{x['l2re']:.4f}) ±{x['sd']:.2f} дек.  "
                              f"опора {x['support'][0]}/{x['support'][1]}  {x['chain']}")
                for x in meas:
                    print(f"   [{x['tag']}] J={x['J']:+.3f} ошибка {10 ** x['mean']:.4g} (l2re≈{K * 10 ** x['mean']:.4f}) "
                          f"±{x['sd']:.2f}; место среди цепочек ≤4 действий {x['rank4']}/{x['of']}; факт: {x['note']}")
        out[name] = res
    w.dump("plan", out)
    return out


def cmd_halves(w, ks=(0.0, 1.0)):
    """Проклятие оптимизатора: план, найденный по модели половины A, оценивается моделью
    половины B (и наоборот) вместе с замеренными цепочками. Выигрыш плана, который видит только
    модель, по которой он найден, — подгонка под её ошибки."""
    tab = w.table()
    halves = [w.half_model(0), w.half_model(1)]
    allowed = parse_mask(w.args.mask) if w.args.mask else None
    out = {}
    print("\nПроверка планов на независимой половине буфера (J — log10 итоговой ошибки)")
    for name in MAIN:
        if name not in tab["NAMES"]:
            continue
        ctx = tab["CTX"][tab["NAMES"].index(name)]
        res = []
        cands = []
        for k in ks:
            for h in (0, 1):
                a, b = halves[h], halves[1 - h]
                r = plan(a, ctx, k=k, width=w.args.width, cap=w.args.cap, allowed=allowed, top=20, keep_depth=4)
                for c, *_ in r["best"] + r["best_short"]:
                    if c not in cands:
                        cands.append(c)
                for kind, lst in (("лучший", r["best"]), ("до 4", r["best_short"])):
                    c = lst[0][0]
                    row = dict(k=k, half=h, kind=kind, chain=chain_str(c), J_own=lst[0][3],
                               J_other=chain_score(b, ctx, c, k)[2],
                               meas={t: (chain_score(a, ctx, ch, k)[2], chain_score(b, ctx, ch, k)[2])
                                     for t, ch, _ in _measured(name)})
                    res.append(row)
                    best_m = min(row["meas"], key=lambda t: row["meas"][t][1])
                    print(f"{name} k={k:g} план по половине {h} ({kind}): J своей {row['J_own']:+.3f}, другой "
                          f"{row['J_other']:+.3f}; лучшая замеренная по другой: {best_m} {row['meas'][best_m][1]:+.3f} "
                          f"-> выигрыш плана по другой {row['meas'][best_m][1] - row['J_other']:+.3f} дек.  {row['chain']}")
        # устойчивый выбор: кандидаты из планов обеих половин, оценка — худшая из двух половин
        meas = _measured(name)
        L = [rollout(m, ctx, cands + [ch for _, ch, _ in meas]) for m in halves]
        worst = np.maximum(L[0].mean(0), L[1].mean(0))
        nc = len(cands)
        order = np.argsort(worst[:nc])
        mw = {t: float(worst[nc + i]) for i, (t, _, _) in enumerate(meas)}
        best_m = min(mw, key=mw.get)
        robust = [dict(chain=chain_str(cands[i]), worst=float(worst[i]), len=len(cands[i]),
                       safe=_is_safe(cands[i]), epochs=int(sum(EPOCHS[a] for a in cands[i])))
                  for i in order[:8]]
        out[name] = dict(plans=res, robust=robust, measured_worst=mw)
        print(f"{name}: устойчивый выбор из {nc} кандидатов (худшая из двух половин); лучшая замеренная "
              f"{best_m} {mw[best_m]:+.3f}")
        for x in robust:
            print(f"   {x['worst']:+.3f} ({mw[best_m] - x['worst']:+.3f} дек. к {best_m}) "
                  f"{'safe ' if x['safe'] else ''}{x['len']} д., {x['epochs']} эп.  {x['chain']}")
    w.dump("halves", out)
    return out


def _universal(ens, ctxs, cands, k=0.0):
    """Универсальная цепочка по модели: среди кандидатов минимум среднего по задачам отношения
    к лучшему кандидату задачи (в логарифмах; как «лучший одиночный метод» находки H)."""
    n, P = len(cands), len(ctxs)
    J = np.zeros((P, n))
    for p in range(P):
        L = rollout(ens, ctxs[p], cands)
        J[p] = L.mean(0) + k * L.std(0)
    reg = J - J.min(1, keepdims=True)
    return int(np.argmin(reg.mean(0))), J


def cmd_lopo(w, k=0.0):
    """Шаг 4, перенос: для каждой задачи с ≥40 файлами модель без неё планирует цепочку по
    описателю задачи; универсальная цепочка — лучшая в среднем по остальным задачам той же моделью.
    Судьи: (1) полная модель (видела задачу), (2) отбор среди записанных цепочек задачи с
    настоящими итогами: по прогнозу для задачи против прогноза «в среднем по остальным»."""
    tab = w.table()
    full = w.full_model()
    lc = logged_chains(tab)
    allowed = parse_mask(w.args.mask) if w.args.mask else None
    out = {}
    pis = lopo_pdes(tab)
    print("\nШаг 4, перенос между УрЧП (k=%g): модель без задачи планирует для неё" % k)
    print(f"{'УрЧП':26s} {'план/факт%':>10s} {'полн: план':>10s} {'универс':>8s} {'Δдек':>6s} {'свой план':>9s} | "
          f"{'отбор: по задаче':>16s} {'по остальным':>12s} {'случайно':>8s}")
    for pi in pis:
        name = tab["NAMES"][pi]
        ens = w.lopo_model(pi)
        ctx = tab["CTX"][pi]
        others = [j for j in range(len(tab["NAMES"])) if j != pi]
        r = plan(ens, ctx, k=k, width=w.args.width, cap=w.args.lopo_cap, allowed=allowed, top=5)
        cp = r["best"][0][0]
        # кандидаты универсальной цепочки: лучшие планы модели для каждой из остальных задач
        cands = []
        for j in others:
            for c, *_ in plan(ens, tab["CTX"][j], k=k, width=w.args.width, cap=w.args.lopo_cap,
                              allowed=allowed, top=3)["best"]:
                if c not in cands:
                    cands.append(c)
        ui, _ = _universal(ens, tab["CTX"][others], cands, k=k)
        up = cands[ui]
        own = plan(full, ctx, k=k, width=w.args.width, cap=w.args.lopo_cap, allowed=allowed, top=1)["best"][0]
        jp = chain_score(full, ctx, cp, k)[2]
        ju = chain_score(full, ctx, up, k)[2]
        jl = chain_score(ens, ctx, cp, k)[2]
        sel = np.where(lc["PDE"] == pi)[0]
        fin = lc["final"][sel]
        pct = lambda v: float((fin < v).mean())
        # отбор среди записанных цепочек задачи (настоящие итоги, одна инициализация на цепочку)
        chs = [lc["chains"][i] for i in sel]
        Lt = rollout(ens, ctx, chs)
        pt = Lt.mean(0) + k * Lt.std(0)
        Lo = np.stack([rollout(ens, tab["CTX"][j], chs).mean(0) for j in others])
        po = (Lo - Lo.min(1, keepdims=True)).mean(0)
        q = np.argsort(np.argsort(fin)) / max(len(fin) - 1, 1)
        row = dict(plan=chain_str(cp), universal=chain_str(up), own_plan=chain_str(own[0]),
                   lopo_pred=jl, full_plan=jp, full_universal=ju, full_own=own[3],
                   plan_pct_lopo=pct(jl), plan_pct_full=pct(jp), univ_pct_full=pct(ju),
                   pick_target=float(q[int(np.argmin(pt))]), pick_others=float(q[int(np.argmin(po))]),
                   top10_target=float(q[np.argsort(pt)[:max(1, len(pt) // 10)]].mean()),
                   top10_others=float(q[np.argsort(po)[:max(1, len(po) // 10)]].mean()),
                   rho_target=spearman(pt, fin), rho_others=spearman(po, fin), n=len(sel),
                   solvable=tab["SUBDIRS"][pi] not in UNSOLVED)
        out[name] = row
        print(f"{name:26s} {row['plan_pct_lopo']:10.2f} {jp:10.2f} {ju:8.2f} {jp - ju:+6.2f} {own[3]:9.2f} | "
              f"{row['pick_target']:8.2f} ({row['top10_target']:.2f}) {row['pick_others']:7.2f} ({row['top10_others']:.2f}) "
              f"{0.5:8.2f}")
        print(f"   план: {row['plan']}\n   универсальная: {row['universal']}\n   своя модель: {row['own_plan']}")
    print("план/факт% — доля записанных цепочек задачи с фактом лучше прогноза модели без задачи для её плана; "
          "полн — J полной модели (log10 ошибки); Δдек < 0 — план для задачи лучше универсальной; "
          "отбор — процентиль факта выбранной записанной цепочки (в скобках — средний у 10% лучших)")
    w.dump("lopo", out)
    return out


def cmd_mpc(w, k=0.0, n_states=150):
    """Шаг 5: на записанных цепочках отложенного фолда сравниваются действия plan_next из
    наблюдённого состояния и из состояния, предсказанного моделью для той же истории (то есть
    без обратной связи). Доля смен действия по величине отклонения и ожидаемый моделью выигрыш."""
    tab = w.table()
    fold, ms = w.cv_models()
    lc = logged_chains(tab)
    allowed = parse_mask(w.args.mask) if w.args.mask else None
    rng = np.random.default_rng(w.args.seed)
    out = {}
    print("\nШаг 5. Перепланирование (MPC) на записанных траекториях отложенного фолда")
    for name in MAIN:
        if name not in tab["NAMES"]:
            continue
        pi = tab["NAMES"].index(name)
        ctx = tab["CTX"][pi]
        open_plan = plan(w.full_model(), ctx, k=k, width=w.args.width, cap=w.args.cap,
                         allowed=allowed, top=1)["best"][0][0]
        cand = []
        for i in np.where(lc["PDE"] == pi)[0]:
            ch = lc["chains"][i]
            for t in range(1, len(ch)):
                cand.append((i, t, ch[:t] == open_plan[:t]))
        on_plan = [c for c in cand if c[2]]
        pick = [cand[j] for j in rng.choice(len(cand), min(n_states, len(cand)), replace=False)]
        recs = []
        t0 = time.time()
        for i, t, onp in pick + [c for c in on_plan if c not in pick]:
            ens = ms[fold[lc["CH"][i]]]
            hist = lc["chains"][i][:t]
            row = np.where((tab["CH"] == lc["CH"][i]) & (tab["STEP"] == t))[0][0]
            l_obs = float(tab["L"][row])
            l_hat = float(rollout(ens, ctx, [hist]).mean())
            a_obs = plan_next(ens, ctx, hist, l_obs, k=k, allowed=allowed)
            a_hat = plan_next(ens, ctx, hist, l_hat, k=k, allowed=allowed)
            # чего стоит не перепланировать: план из предсказанного состояния, исполненный из настоящего
            if a_hat["plan"]:
                st = init_state(ctx, 1, ens.members, history=hist, l=l_obs)
                for a in a_hat["plan"]:
                    st = advance(ens, st, np.array([a]))
                j_keep = float(st["l"][:, 0].mean() + k * st["l"][:, 0].std())
            else:
                j_keep = l_obs
            recs.append(dict(dev=l_obs - l_hat, change=a_obs["action"] != a_hat["action"],
                             gain=j_keep - a_obs["J"], on_plan=bool(onp), step=t,
                             stop_obs=a_obs["action"] is None, stop_hat=a_hat["action"] is None))
        dev = np.array([r["dev"] for r in recs])
        chg = np.array([r["change"] for r in recs])
        gain = np.array([r["gain"] for r in recs])
        onp = np.array([r["on_plan"] for r in recs])
        bins = [(0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 99)]
        tbl = []
        for lo, hi in bins:
            m = (np.abs(dev) >= lo) & (np.abs(dev) < hi)
            tbl.append(dict(lo=lo, hi=hi, n=int(m.sum()), change=float(chg[m].mean()) if m.any() else None,
                            gain_med=float(np.median(gain[m & chg])) if (m & chg).any() else None))
        out[name] = dict(open_plan=chain_str(open_plan), n=len(recs), n_on_plan=int(onp.sum()),
                         change=float(chg.mean()), change_on_plan=float(chg[onp].mean()) if onp.any() else None,
                         gain_med=float(np.median(gain[chg])) if chg.any() else None,
                         gain_on_plan=float(np.median(gain[onp & chg])) if (onp & chg).any() else None,
                         dev_abs_med=float(np.median(np.abs(dev))), bins=tbl, sec=time.time() - t0)
        o = out[name]
        pc = lambda v: "-" if v is None else f"{100 * v:.0f}%"
        dk = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"{name}: открытый план {o['open_plan']}")
        print(f"   состояний {o['n']} (на плане {o['n_on_plan']}), |отклонение| медиана {o['dev_abs_med']:.2f} дек.; "
              f"действие меняется в {pc(o['change'])} (на плане {pc(o['change_on_plan'])}); ожидаемый моделью "
              f"выигрыш при смене: медиана {dk(o['gain_med'])} дек. (на плане {dk(o['gain_on_plan'])})")
        for b in tbl:
            print(f"   |откл| {b['lo']}–{b['hi']} дек.: n={b['n']}, смена {pc(b['change'])}, выигрыш {dk(b['gain_med'])}")
    w.dump("mpc", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "onestep", "multistep", "gpucheck", "plan", "halves", "lopo",
                                    "mpc", "all"])
    ap.add_argument("--work", default=os.environ.get("SURR_WORK", "/tmp/surrogate"),
                    help="папка кэша таблицы, моделей и JSON результатов")
    ap.add_argument("--subdirs", default="", help="папки буфера через запятую (по умолчанию все 22)")
    ap.add_argument("--rebuild", action="store_true", help="пересобрать таблицу")
    ap.add_argument("--refit", action="store_true", help="переобучить модели")
    ap.add_argument("--quick", action="store_true", help="короткий бустинг (дымовой прогон)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=256, help="ширина луча после полного перебора")
    ap.add_argument("--cap", type=int, default=30000, help="фронт перебирается целиком, пока не больше cap")
    ap.add_argument("--lopo-cap", type=int, default=800, help="cap для переноса (много запусков плана)")
    ap.add_argument("--mask", default="", help="запрещённые действия, как --mask оценки ('pso:0.001,pso:0.0001')")
    ap.add_argument("--k", type=float, default=0.0, help="штраф за разброс в lopo и mpc")
    args = ap.parse_args(argv)
    w = Work(args)
    cmds = (["table", "onestep", "multistep", "gpucheck", "plan", "halves", "lopo", "mpc"]
            if args.cmd == "all" else [args.cmd])
    for c in cmds:
        t0 = time.time()
        if c == "table":
            cmd_table(w)
        elif c == "onestep":
            cmd_onestep(w)
        elif c == "multistep":
            cmd_multistep(w)
        elif c == "gpucheck":
            cmd_gpucheck(w)
        elif c == "plan":
            cmd_plan(w)
        elif c == "halves":
            cmd_halves(w)
        elif c == "lopo":
            cmd_lopo(w, k=args.k)
        elif c == "mpc":
            cmd_mpc(w, k=args.k)
        print(f"[{c}: {time.time() - t0:.0f} с]", flush=True)


if __name__ == "__main__":
    # модели в pickle должны ссылаться на модуль surrogate, а не на __main__: тогда их можно
    # загрузить из другого кода (plan_next в оценке, тесты)
    sys.path.insert(0, HERE)
    import surrogate as _self
    _self.main()
