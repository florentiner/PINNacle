"""Трек 3: разбор уже снятых результатов без новых запусков.

Источник — JSON-строки замороженной оценки (rl_arch/online_env/*.json) и обучения
(rl_arch/online_train/*.json), лежащие в локальной папке. Отчёты:

  vbs      лучший одиночный арм (SBS), виртуально лучший по сидам (VBS), контроль
           перемешиванием сидов, вклад армов по Шепли
  bestk    «лучший из k» сидов одного арма: выбор по обучающему лоссу (честно) и по
           истинной ошибке (оракул), стоимость в эпохах
  race     гонка между k инициализациями с отсевом по обучающему лоссу (нужна история hist)
  pair     парное сравнение двух армов: бутстреп-интервалы, вероятность улучшения, знаковый тест
  cost     модель времени: секунды на шаг состояния и на эпоху каждого оптимизатора
  pool     лучшая из k обучающих цепочек (по файлам online_train)
  even     число применений, после которого мета-обучение окупается
  ladder   таблица лестницы: для каждого арма медиана, разброс, время, доля времени на
           состояние, доля совпадений с действиями по истинным картам, число разных цепочек;
           парное сравнение с опорным армом (--ref)
  pick     победитель случайного поиска статических цепочек: лучшая цепочка по обучающему
           лоссу (честный выбор) и по истинной ошибке (оракул) в формате --script
  target   «шаги до цели»: сколько эпох и секунд нужно арму, чтобы ошибка впервые опустилась
           до порога (--target; по умолчанию итоговая медиана опорного арма --ref); недошедшие
           прогоны цензурируются бюджетом. Нужна история hist (прогоны нового кода)
  stall    где прогон перестаёт двигаться: эпоха последнего изменения ошибки, доля бюджета до
           неё и доля пустых действий (ошибка изменилась меньше чем на 0.1%) по номеру шага.
           Показывает застой L-BFGS после пересоздания оптимизатора; пары армов со сбросом
           и без (--keep-opt) сравниваются по этим числам
  shrink   проклятие победителя: значение, по которому арм был отобран (--sel имя=значение или
           лучший сид арма-источника), против его переоценки на новых сидах
  tree     что выучила политика: дерево решений предсказывает действие арма по дешёвым
           признакам (шаг, доля бюджета, прошлое действие, обучающий лосс); точность с
           проверкой по сидам и сами правила

Примеры:
  python posthoc.py --fetch --dir /tmp/online_env --pde ns2d_liddriven --report vbs,cost
  python posthoc.py --dir /tmp/online_env --report ladder --arms t3s_ --ref t3s_full
  python posthoc.py --dir /tmp/online_env --report pick --arms t3m_rs
"""
from __future__ import annotations

import argparse
import ast
import glob
import itertools
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


def _chain(v):
    if isinstance(v, str):
        try:
            return json.loads(v.replace("'", '"'))
        except Exception:
            return ast.literal_eval(v)
    return v or []


def load_eval(dir_, pde, budget=None):
    """{арм: {сид: строка}} для одного УрЧП."""
    arms = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(dir_, f"{pde}_*_seed*.json"))):
        m = re.match(rf"{re.escape(pde)}_(.+)_seed(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        if d.get("partial") or d.get("unfinished"):
            continue
        if budget is not None and int(float(d.get("budget", budget))) != int(budget):
            continue
        d["chain"] = _chain(d.get("chain"))
        d["l2re"] = float(d["l2re"])
        arms[m.group(1)][int(m.group(2))] = d
    return arms


def matrix(arms, seeds=None, min_seeds=3):
    seeds = seeds or sorted(set.intersection(*[set(v) for v in arms.values()])) if arms else []
    names = [a for a in sorted(arms) if all(s in arms[a] for s in seeds)]
    A = np.array([[arms[a][s]["l2re"] for s in seeds] for a in names], dtype=float)
    return names, seeds, A


def shapley_vbs(A, n_perm=2000, seed=0):
    """Вклад каждого арма в среднюю ошибку виртуально лучшего (меньше — лучше).
    Ценность коалиции = -среднее по сидам от минимума по армам коалиции; пустая
    коалиция оценивается худшим армом."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    worst = A.mean(1).max()
    phi = np.zeros(n)
    perms = (list(itertools.permutations(range(n))) if n <= 7
             else [rng.permutation(n) for _ in range(n_perm)])
    for perm in perms:
        cur = np.full(A.shape[1], np.inf)
        prev_v = -worst
        for j in perm:
            cur = np.minimum(cur, A[j])
            v = -cur.mean()
            phi[j] += v - prev_v
            prev_v = v
    return phi / len(perms)


def report_vbs(arms, drop_above=None):
    names, seeds, A = matrix(arms)
    if not names:
        print("нет армов с общим набором сидов"); return
    if drop_above:
        keep = np.median(A, 1) < drop_above
        names, A = [n for n, k in zip(names, keep) if k], A[keep]
    med, mean = np.median(A, 1), A.mean(1)
    i = int(np.argmin(med))
    vbs = A.min(0)
    print(f"армов {len(names)}, сидов {len(seeds)}: {seeds}")
    print(f"лучший одиночный (SBS) по медиане: {names[i]}  медиана {med[i]:.4f}  среднее {mean[i]:.4f}")
    print(f"виртуально лучший (VBS) по сидам:   медиана {np.median(vbs):.4f}  среднее {vbs.mean():.4f}")
    rng = np.random.default_rng(0)
    sh = [np.array([rng.permutation(r) for r in A]).min(0).mean() for _ in range(3000)]
    print(f"VBS при перемешанных сидах (экземпляр не важен): среднее {np.mean(sh):.4f} "
          f"[{np.percentile(sh, 2.5):.4f}, {np.percentile(sh, 97.5):.4f}]")
    gap = mean[i] - vbs.mean()
    print(f"зазор SBS−VBS по среднему: {gap:.4f} ({100 * gap / mean[i]:.1f}% от SBS). "
          f"{'Взаимодополняемости по сидам нет: VBS не лучше, чем при перемешанных сидах.' if vbs.mean() >= np.percentile(sh, 2.5) else 'VBS лучше случайного — есть эффект экземпляра.'}")
    phi = shapley_vbs(A)
    order = np.argsort(-phi)
    print("вклад армов в VBS по Шепли (первые 8):")
    for j in order[:8]:
        print(f"   {names[j]:28s} вклад {phi[j]:+.4f}  медиана {med[j]:.4f}  побед {(A.argmin(0) == j).sum()}")
    nch = {a: len({json.dumps(arms[a][s]["chain"]) for s in seeds}) for a in names}
    same = [a for a in names if nch[a] == 1]
    print(f"армов с одной и той же цепочкой на всех сидах: {len(same)} из {len(names)}: {same[:8]}")


def report_bestk(arms, ks=(1, 2, 3, 5), n_boot=20000):
    rng = np.random.default_rng(0)
    print("арм | k | медиана l2re при выборе по обучающему лоссу | по истинной ошибке (оракул) | эпох на результат")
    for a in sorted(arms):
        rows = list(arms[a].values())
        if len(rows) < 3:
            continue
        l2 = np.array([r["l2re"] for r in rows])
        loss = np.array([r["loss_final"] if r.get("loss_final") is not None else np.nan for r in rows], float)
        budget = float(rows[0].get("spent") or rows[0].get("budget") or 0)
        for k in ks:
            if k > len(rows):
                continue
            if math.comb(len(rows), k) <= 5000:     # все подмножества без повторов
                idx = np.array(list(itertools.combinations(range(len(rows)), k)))
            else:
                idx = np.array([rng.choice(len(rows), size=k, replace=False) for _ in range(5000)])
            orc = l2[idx].min(1)
            if np.isfinite(loss).all():
                pick = idx[np.arange(len(idx)), loss[idx].argmin(1)]
                hon = f"{np.median(l2[pick]):.4f}"
            else:
                hon = "нет loss_final"
            print(f"{a:28s} | {k} | {hon} | {np.median(orc):.4f} | {int(k * budget)}")


def report_race(arms, k=4, eta=2, rungs=(0.25, 0.5, 1.0), n_boot=5000):
    """Гонка между k инициализациями одного арма: на каждой ступени бюджета остаётся
    доля 1/eta лучших по обучающему лоссу. Итог — ошибка победителя и потраченные эпохи."""
    rng = np.random.default_rng(0)
    print(f"гонка: k={k}, отсев 1/{eta}, ступени {rungs}")
    for a in sorted(arms):
        rows = [r for r in arms[a].values() if r.get("hist")]
        if len(rows) < k:
            continue
        budget = float(rows[0].get("budget") or 0)

        def at(r, frac):
            h = [x for x in r["hist"] if x[0] <= frac * budget + 1e-9 and x[1] is not None]
            return h[-1] if h else None
        res, cost = [], []
        for _ in range(n_boot):
            alive = list(rng.choice(len(rows), size=k, replace=False))
            spent = 0.0
            prev = 0.0
            for fr in rungs:
                spent += len(alive) * (fr - prev) * budget
                prev = fr
                vals = [(at(rows[i], fr) or [0, float("inf")])[1] for i in alive]
                if fr < rungs[-1]:
                    keep = max(1, len(alive) // eta)
                    alive = [alive[j] for j in np.argsort(vals)[:keep]]
                else:
                    alive = [alive[int(np.argmin(vals))]]
            res.append(rows[alive[0]]["l2re"]); cost.append(spent)
        print(f"{a:28s} медиана победителя {np.median(res):.4f}  средние эпохи {np.mean(cost):.0f} "
              f"(против {k * budget:.0f} у «лучшего из {k}»)")


def report_pair(arms, a, b, n_boot=20000):
    seeds = sorted(set(arms[a]) & set(arms[b]))
    x = np.array([arms[a][s]["l2re"] for s in seeds]); y = np.array([arms[b][s]["l2re"] for s in seeds])
    d = x - y
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1); bmed = np.median(d[idx], 1)
    wins = int((d < 0).sum())
    p_sign = sum(math.comb(len(d), i) for i in range(min(wins, len(d) - wins) + 1)) * 2 / 2 ** len(d)
    print(f"{a} против {b}: сидов {len(d)}; разность (a−b) среднее {d.mean():+.4f} "
          f"[{np.percentile(bm, 2.5):+.4f}, {np.percentile(bm, 97.5):+.4f}], медиана {np.median(d):+.4f} "
          f"[{np.percentile(bmed, 2.5):+.4f}, {np.percentile(bmed, 97.5):+.4f}]")
    print(f"   вероятность, что a лучше на случайном сиде: {wins / len(d):.2f}; знаковый тест p = {min(1.0, p_sign):.3f}; "
          f"SD парной разности {d.std(ddof=1):.4f}")


def report_cost(arms):
    from scipy.optimize import nnls
    X, y, direct = [], [], []
    for a in arms:
        for r in arms[a].values():
            ep = defaultdict(int)
            for o, _, e in r["chain"]:
                ep[o] += e
            X.append([1, len(r["chain"]), ep["Adam"], ep["LBFGS"], ep["PSO"]])
            y.append(float(r.get("elapsed_s") or 0))
            if r.get("t_opt_s") is not None:
                direct.append((float(r.get("t_ae_s") or 0), float(r.get("t_surface_s") or 0),
                               float(r["t_opt_s"]), float(r.get("elapsed_s") or 0)))
    X, y = np.array(X, float), np.array(y)
    names = ["константа (запуск, первый AE)", "на шаг (состояние и оценка)", "эпоха Adam", "эпоха L-BFGS", "эпоха PSO"]
    for title, m in (("все прогоны", np.ones(len(y), bool)), ("прогоны без PSO", X[:, 4] == 0)):
        if m.sum() < 8:
            continue
        cols = [0, 1, 2, 3] + ([4] if title == "все прогоны" else [])
        w, _ = nnls(X[m][:, cols], y[m])
        pred = X[m][:, cols] @ w
        r2 = 1 - ((y[m] - pred) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
        share = (X[m][:, cols] * w).sum(0) / pred.sum()
        print(f"{title}: n={int(m.sum())}, R2={r2:.3f}")
        for j, c in enumerate(cols):
            print(f"   {names[c]:32s} {w[j]:9.3f} с   доля времени {100 * share[j]:.1f}%")
    if direct:
        D = np.array(direct)
        print(f"прямые замеры (n={len(D)}): автокодировщик {100 * D[:, 0].sum() / D[:, 3].sum():.1f}%, "
              f"поверхность {100 * D[:, 1].sum() / D[:, 3].sum():.1f}%, оптимизатор {100 * D[:, 2].sum() / D[:, 3].sum():.1f}% времени")


def report_pool(train_dir, pde, ks=(1, 3, 6, 12, 24)):
    allc, best, hours, n = [], [], 0.0, 0
    for f in sorted(glob.glob(os.path.join(train_dir, "*.json"))):
        d = json.load(open(f))
        if d.get("pde") != pde:
            continue
        ch = d.get("chains")
        ch = ast.literal_eval(ch) if isinstance(ch, str) else (ch or [])
        l = [c["l2re"] for c in ch if c.get("l2re") is not None and np.isfinite(c["l2re"])]
        if not l:
            continue
        allc += l; best.append(min(l)); hours += float(d.get("elapsed_h") or 0); n += 1
    if not allc:
        print("нет обучающих прогонов для", pde); return
    allc = np.array(allc)
    print(f"{pde}: прогонов {n}, цепочек {len(allc)}, GPU-часов {hours:.0f}, минут на цепочку {60 * hours / len(allc):.1f}")
    print(f"лучшая цепочка прогона: квартили {np.round(np.percentile(best, [25, 50, 75]), 4)}")
    rng = np.random.default_rng(0)
    for k in ks:
        m = allc[rng.integers(0, len(allc), size=(20000, k))].min(1)
        print(f"   лучшая из {k:2d} случайных цепочек пула: медиана {np.median(m):.4f}")


def _script(chain):
    """Цепочка из строки результата -> аргумент --script."""
    return ",".join(f"{o}:{lr:g}:{int(e)}" for o, lr, e in chain if o != "BOOST")


def report_ladder(arms, prefix="", ref=""):
    names = [a for a in sorted(arms) if a.startswith(prefix) or a == ref]
    if not names:
        print("нет армов с префиксом", prefix); return
    print("арм | сидов | медиана l2re | квартили | среднее log10 | часов на прогон | доля состояния | "
          "совпадение действий | разных цепочек | против опорного")
    for a in names:
        rows = arms[a]
        l2 = np.array([r["l2re"] for r in rows.values()])
        el = np.array([float(r.get("elapsed_s") or 0) for r in rows.values()])
        st = np.array([float(r.get("t_ae_s") or 0) + float(r.get("t_surface_s") or 0) for r in rows.values()])
        ag = [r["agree_rate"] for r in rows.values() if r.get("agree_rate") is not None]
        nch = len({json.dumps(r["chain"]) for r in rows.values()})
        q = np.percentile(l2, [25, 75])
        vs = ""
        if ref and ref in arms and a != ref:
            seeds = sorted(set(rows) & set(arms[ref]))
            if seeds:
                d = np.array([rows[s]["l2re"] - arms[ref][s]["l2re"] for s in seeds])
                wins = int((d < 0).sum())
                k = min(wins, len(d) - wins)
                p = min(1.0, sum(math.comb(len(d), i) for i in range(k + 1)) * 2 / 2 ** len(d))
                rel = np.median([rows[s]["l2re"] / arms[ref][s]["l2re"] for s in seeds])
                vs = f"лучше на {wins}/{len(d)}, отношение медиан по сидам {rel:.2f}, p={p:.3f}"
        print(f"{a:26s} | {len(l2):2d} | {np.median(l2):.4f} | {q[0]:.4f}–{q[1]:.4f} | "
              f"{np.mean(np.log10(l2)):+.3f} | {np.median(el) / 3600:.2f} | "
              f"{(100 * st.sum() / el.sum() if el.sum() else 0):.0f}% | "
              f"{(np.mean(ag) if ag else float('nan')):.2f} | {nch} | {vs}")


def report_target(arms, prefix="", ref="", target=0.0):
    """Эпохи и время до порога ошибки. История hist: [эпох потрачено, лосс, l2re, секунд]."""
    names = [a for a in sorted(arms) if a.startswith(prefix) or a == ref]
    if not target:
        if not (ref and ref in arms):
            print("нужен --target или опорный арм --ref"); return
        target = float(np.median([r["l2re"] for r in arms[ref].values()]))
    print(f"порог l2re = {target:.4f}" + (f" (итоговая медиана {ref})" if ref in arms else ""))
    print("арм | сидов | дошли до порога | эпох до порога: медиана (недошедшие = бюджет) | "
          "секунд до порога: медиана по дошедшим | итоговая медиана l2re")
    for a in names:
        ep_to, sec_to, n, nohist = [], [], 0, 0
        for r in arms[a].values():
            h = r.get("hist") or []
            if not h:
                nohist += 1
                continue
            n += 1
            hit = next((x for x in h if x[2] is not None and x[2] <= target), None)
            ep_to.append(float(hit[0]) if hit else float(r.get("budget") or h[-1][0]))
            if hit:
                sec_to.append(float(hit[3]))
        if not n:
            print(f"{a:26s} | нет истории hist (прогоны старого кода: {nohist})"); continue
        fin = np.median([r["l2re"] for r in arms[a].values()])
        sec = f"{np.median(sec_to):.0f}" if sec_to else "—"
        print(f"{a:26s} | {n:2d} | {len(sec_to)}/{n} | {np.median(ep_to):.0f} | {sec} | {fin:.4f}")


def report_stall(arms, prefix="", tol=1e-3):
    """Застой по истории hist: [эпох потрачено, лосс, l2re, секунд] на каждое действие."""
    names = [a for a in sorted(arms) if a.startswith(prefix)]
    print("арм | сидов | медиана l2re | действий | эпоха последнего изменения ошибки: медиана (мин–макс) | "
          "доля бюджета до неё | пустых действий | доля сидов с пустым действием №2, №3, №4, №5")
    for a in names:
        last, used, nst, empty, by_k = [], [], [], [], defaultdict(list)
        for r in arms[a].values():
            h = [x for x in (r.get("hist") or []) if x[2] is not None]
            if not h:
                continue
            e, sp = [x[2] for x in h], [x[0] for x in h]
            flat = [k > 0 and e[k - 1] > 0 and abs(e[k] - e[k - 1]) / e[k - 1] < tol for k in range(len(e))]
            k = len(e) - 1
            while k > 0 and flat[k]:
                k -= 1
            # доля считается от бюджета прогона, а не от потраченного: агент может остановиться раньше
            last.append(sp[k]); used.append(sp[k] / max(1.0, float(r.get("budget") or sp[-1])))
            nst.append(len(e))
            empty.append(sum(flat) / max(1, len(e) - 1) if len(e) > 1 else 0.0)
            for j in range(1, min(len(e), 5)):
                by_k[j].append(flat[j])
        if not last:
            print(f"{a:26s} | нет истории hist"); continue
        l2 = np.median([r["l2re"] for r in arms[a].values()])
        ks = ", ".join(f"{100 * np.mean(by_k[j]):.0f}%" for j in sorted(by_k))
        print(f"{a:26s} | {len(last):2d} | {l2:.4f} | {int(np.median(nst))} | {int(np.median(last))} "
              f"({int(min(last))}–{int(max(last))}) | {100 * np.median(used):.0f}% | "
              f"{100 * np.mean(empty):.0f}% | {ks}")


def report_shrink(arms, sel):
    """Проклятие победителя. sel: список «арм=значение при отборе» или «арм=арм-источник» (тогда
    значение при отборе — лучший сид источника). Для каждого — медиана переоценки и отношение."""
    if not sel:
        print("нужен --sel арм=значение[,арм=значение...] или арм=арм-источник"); return
    print("арм переоценки | значение при отборе | сидов | медиана переоценки | квартили | "
          "отношение переоценки к отбору")
    ratios = []
    for item in sel.split(","):
        if "=" not in item:
            continue
        a, v = [x.strip() for x in item.split("=", 1)]
        if a not in arms:
            print(f"{a:26s} | нет прогонов"); continue
        try:
            picked, src = float(v), ""
        except ValueError:
            src_rows = [r["l2re"] for s_ in sorted(arms) if s_.startswith(v) and s_ != a
                        for r in arms[s_].values()]
            if not src_rows:
                print(f"{a:26s} | нет арма-источника {v}"); continue
            picked, src = float(min(src_rows)), f" (лучший из {len(src_rows)} прогонов {v})"
        l2 = np.array([r["l2re"] for r in arms[a].values()])
        q = np.percentile(l2, [25, 75])
        ratios.append(np.median(l2) / picked)
        print(f"{a:26s} | {picked:.4f}{src} | {len(l2)} | {np.median(l2):.4f} | {q[0]:.4f}–{q[1]:.4f} | "
              f"{np.median(l2) / picked:.2f}")
    if ratios:
        print(f"среднее геометрическое отношения: {float(np.exp(np.mean(np.log(ratios)))):.2f} "
              f"(1.00 — отбор не завышал качество)")


def report_pick(arms, prefix):
    rows = [r for a in sorted(arms) if a.startswith(prefix) for r in arms[a].values()]
    if not rows:
        print("нет прогонов с префиксом", prefix); return
    l2 = np.array([r["l2re"] for r in rows])
    loss = np.array([r["loss_final"] if r.get("loss_final") is not None else np.inf for r in rows], float)
    hours = sum(float(r.get("elapsed_s") or 0) for r in rows) / 3600
    print(f"случайный поиск: цепочек {len(rows)}, GPU-часов {hours:.1f}; l2re: медиана {np.median(l2):.4f}, "
          f"лучшая {l2.min():.4f}")
    i, j = int(np.argmin(loss)), int(np.argmin(l2))
    if np.isfinite(loss).any():
        print(f"выбор по обучающему лоссу (честный): сид {rows[i]['seed']}, l2re {l2[i]:.4f}")
        print(f"   RS_CHAIN_LOSS = {_script(rows[i]['chain'])}")
    else:
        print("выбор по обучающему лоссу невозможен: в строках нет loss_final (прогоны старого кода)")
    print(f"выбор по истинной ошибке (оракул, тот же сигнал, что награда агента): сид {rows[j]['seed']}, "
          f"l2re {l2[j]:.4f}")
    print(f"   RS_CHAIN_ERR = {_script(rows[j]['chain'])}")
    order = np.argsort(l2)[:5]
    print("пять лучших цепочек по истинной ошибке:")
    for k in order:
        print(f"   {l2[k]:.4f}  лосс {loss[k]:.3e}  {_script(rows[k]['chain'])}")


def report_tree(arms, prefix, depth=3):
    from sklearn.tree import DecisionTreeClassifier, export_text
    for a in sorted(arms):
        if not a.startswith(prefix):
            continue
        X, y, g = [], [], []
        has_loss = True
        for sd, r in arms[a].items():
            ch, hist = r["chain"], r.get("hist") or []
            budget = float(r.get("budget") or 1)
            for j, (o, lr, e) in enumerate(ch):
                if o == "BOOST":
                    continue
                po = {"Adam": 0, "LBFGS": 1, "PSO": 2}.get(ch[j - 1][0], -1) if j else -1
                pe = float(ch[j - 1][2]) if j else 0.0
                spent_b = float(sum(c[2] for c in ch[:j]))
                lb = hist[j - 1][1] if (j and len(hist) >= j and hist[j - 1][1]) else None
                lbb = hist[j - 2][1] if (j > 1 and len(hist) >= j - 1 and hist[j - 2][1]) else None
                if j and lb is None:
                    has_loss = False
                ll = math.log10(lb) if lb else 3.0
                dl = (math.log10(lbb) - math.log10(lb)) if (lb and lbb) else 0.0
                X.append([j, spent_b / budget, po, pe, ll, dl])
                # последний шаг обрезан бюджетом: метка без числа эпох была бы шумом
                y.append(f"{o}:{lr:g}" + ("" if j == len(ch) - 1 else f":{int(e)}"))
                g.append(sd)
        if len(set(g)) < 3 or len(set(y)) < 2:
            nch = len({json.dumps(r["chain"]) for r in arms[a].values()})
            print(f"{a}: {'одно и то же действие на всех шагах' if len(set(y)) < 2 else 'мало сидов'}; "
                  f"разных цепочек {nch}")
            continue
        X, y, g = np.array(X, float), np.array(y), np.array(g)
        names = ["шаг", "доля_бюджета", "прошлый_оптимизатор", "прошлые_эпохи", "log10_лосса", "падение_лосса"]
        maj = max(np.mean(y == v) for v in set(y))
        out = [f"{a}: решений {len(y)}, сидов {len(set(g))}, разных действий {len(set(y))}, "
               f"самое частое действие {100 * maj:.0f}%"]
        for title, cols in (("время", [0, 1, 2, 3]), ("время и обучающий лосс", [0, 1, 2, 3, 4, 5])):
            if len(cols) > 4 and not has_loss:
                out.append("   в строках нет истории лосса (прогоны старого кода): только признаки времени")
                continue
            acc = []
            for sd in sorted(set(g)):
                tr, te = g != sd, g == sd
                t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[tr][:, cols], y[tr])
                acc.append(float((t.predict(X[te][:, cols]) == y[te]).mean()))
            out.append(f"   дерево глубины {depth}, признаки «{title}»: точность на отложенном сиде "
                       f"{100 * np.mean(acc):.0f}% (от {100 * min(acc):.0f}% до {100 * max(acc):.0f}%)")
            last = (cols, DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[:, cols], y))
        print("\n".join(out))
        print("   правила (дерево по всем сидам):")
        print("   " + export_text(last[1], feature_names=[names[c] for c in last[0]]).replace("\n", "\n   "))


def fetch(dir_, pde):
    """Скачать строки оценки из публичного датасета результатов (токен не нужен)."""
    from huggingface_hub import snapshot_download
    root = snapshot_download("danil-e/pinnacle-optuna-db", repo_type="dataset", token=False,
                             allow_patterns=[f"rl_arch/online_env/{pde}_*.json",
                                             "rl_arch/online_train/*.json"])
    os.makedirs(dir_, exist_ok=True)
    n = 0
    for sub in ("online_env", "online_train"):
        src = os.path.join(root, "rl_arch", sub)
        dst = dir_ if sub == "online_env" else os.path.join(dir_, "_train")
        os.makedirs(dst, exist_ok=True)
        for f in (os.listdir(src) if os.path.isdir(src) else []):
            with open(os.path.join(src, f), "rb") as a, open(os.path.join(dst, f), "wb") as b:
                b.write(a.read())
            n += 1
    print(f"скачано файлов: {n} -> {dir_}")


def report_even(c_meta_h, c_agent_h, c_base_h):
    """Безубыточность: сколько применений нужно, чтобы мета-обучение окупилось."""
    save = c_base_h - c_agent_h
    if save <= 0:
        print(f"применение агента не дешевле бейзлайна ({c_agent_h} против {c_base_h} ч): окупаемости нет")
        return
    print(f"мета-затраты {c_meta_h} ч, экономия на одном применении {save:.2f} ч: "
          f"окупаемость после {math.ceil(c_meta_h / save)} применений")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="папка с JSON оценки (online_env)")
    ap.add_argument("--train-dir", default="", help="папка с JSON обучения (online_train)")
    ap.add_argument("--pde", default="ns2d_liddriven")
    ap.add_argument("--budget", type=int, default=7000)
    ap.add_argument("--report", default="vbs,cost")
    ap.add_argument("--pair", default="", help="арм_a,арм_b")
    ap.add_argument("--drop-above", type=float, default=0.12)
    ap.add_argument("--even", default="", help="мета_ч,агент_ч,бейзлайн_ч")
    ap.add_argument("--arms", default="", help="префикс имён армов для отчётов ladder и pick")
    ap.add_argument("--ref", default="", help="опорный арм для парных сравнений в отчёте ladder")
    ap.add_argument("--target", type=float, default=0.0,
                    help="порог l2re для отчёта target (по умолчанию итоговая медиана --ref)")
    ap.add_argument("--sel", default="",
                    help="отчёт shrink: арм=значение_при_отборе или арм=арм-источник, через запятую")
    ap.add_argument("--fetch", action="store_true",
                    help="сначала скачать строки оценки и обучения из HF в --dir")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.dir, args.pde)
        args.train_dir = args.train_dir or os.path.join(args.dir, "_train")
    arms = load_eval(args.dir, args.pde, args.budget or None)
    for rep in [r.strip() for r in args.report.split(",") if r.strip()]:
        print(f"\n===== {rep} =====")
        if rep == "vbs": report_vbs(arms, args.drop_above)
        elif rep == "bestk": report_bestk(arms)
        elif rep == "race": report_race(arms)
        elif rep == "cost": report_cost(arms)
        elif rep == "pool": report_pool(args.train_dir or args.dir, args.pde)
        elif rep == "pair":
            a, b = args.pair.split(","); report_pair(arms, a, b)
        elif rep == "even":
            report_even(*[float(x) for x in args.even.split(",")])
        elif rep == "ladder": report_ladder(arms, args.arms, args.ref)
        elif rep == "pick": report_pick(arms, args.arms)
        elif rep == "target": report_target(arms, args.arms, args.ref, args.target)
        elif rep == "shrink": report_shrink(arms, args.sel)
        elif rep == "stall": report_stall(arms, args.arms)
        elif rep == "tree": report_tree(arms, args.arms)
        else:
            print("неизвестный отчёт", rep)


if __name__ == "__main__":
    main()
