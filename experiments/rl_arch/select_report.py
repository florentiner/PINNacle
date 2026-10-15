#!/usr/bin/env python
"""Сверка отбора агентов без оценки в среде (select_agents.py) с измеренными медианами (без GPU, секунды):

    python experiments/rl_arch/select_report.py --tag w45 [--fetch] [--sel <каталог JSON>] [--rows <каталог строк>]
        [--min-seeds 3] [--out report.json]

--fetch скачивает результаты отбора rl_arch/select/<pde>_<tag>.json и строки оценок агентов по их меткам
(rl_arch/online_env/<pde>_<метка>_seed*.json) во временные каталоги (или в --sel / --rows); токен
чтения — как у seq_rank.fetch. Измерено — медиана l2re по законченным сидам метки (не меньше
--min-seeds). Для каждого способа (surrogate: surr_score — B1 с первым продолжением, surrogate:<продолжение> —
B1 с каждым продолжением --cont, fqe: fqe_pred_log; меньше — лучше) и набора
агентов (основные; все, включая --extra-models):
    spearman, kendall — корреляция оценки способа с измеренной медианой (log10; ранговые, ничьи средним);
    regret         — медиана агента, выбранного способом (лучший по оценке), минус лучшая медиана набора;
    regret_top3    — то же для лучшего из трёх выбранных (комитет из трёх) и общих с тремя лучшими;
    random_regret  — средняя медиана набора минус лучшая (ожидание при случайном выборе).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import seq_rank as SR  # noqa: E402

SCORES = {"surrogate": "surr_score", "fqe": "fqe_pred_log"}


def rankdata(x):
    """Ранги с 1, ничьи — средним рангом."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall(a, b):
    """tau-b."""
    n = len(a)
    if n < 3:
        return None
    c = d = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif x == y:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + ta) * (c + d + tb))
    return float((c - d) / den) if den else None


def score_of(a, method):
    """Оценка агента способом method (меньше — лучше): surrogate, fqe или surrogate:<продолжение>."""
    if method.startswith("surrogate:"):
        v = (a.get("surr_by_cont") or {}).get(method.split(":", 1)[1]) or {}
        return v.get("surr_score")
    return a.get(SCORES[method])


def methods_of(sel):
    """Способы в результате: основные и B1 с каждым продолжением (--cont select_agents)."""
    conts = (sel.get("surrogate") or {}).get("conts") or []
    return list(SCORES) + [f"surrogate:{c}" for c in conts]


def evaluate(sel, med, method, main_only):
    """Метрики одного способа на одном наборе агентов. sel — JSON select_agents, med — {метка: медиана}."""
    ag = sel.get("agents", {})
    sc = {x: score_of(a, method) for x, a in ag.items()}
    labs = [x for x, a in ag.items() if sc[x] is not None and x in med and (a.get("main") or not main_only)]
    if len(labs) < 2:
        return dict(method=method, set="main" if main_only else "all", n=len(labs), note="мало агентов с оценкой и строками")
    s = np.array([sc[x] for x in labs])
    m = np.array([med[x] for x in labs])
    lm = np.log10(m)
    order = np.argsort(s, kind="mergesort")
    best = float(m.min())
    chosen = labs[int(order[0])]
    top3 = [labs[int(i)] for i in order[:3]]
    true3 = [labs[int(i)] for i in np.argsort(m, kind="mergesort")[:3]]
    return dict(method=method, set="main" if main_only else "all", n=len(labs), spearman=spearman(s, lm),
                kendall=kendall(s, lm), chosen=chosen, chosen_median=float(med[chosen]), best=labs[int(m.argmin())],
                best_median=best, regret=float(med[chosen] - best), top3=top3,
                regret_top3=float(min(med[x] for x in top3) - best), overlap_top3=len(set(top3) & set(true3)),
                random_regret=float(m.mean() - best),
                table=sorted(([x, round(float(sc[x]), 5), round(float(med[x]), 5)] for x in labs), key=lambda r: r[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="сверка отбора агентов с измеренными медианами")
    ap.add_argument("--tag", default="w45")
    ap.add_argument("--pdes", default="poissonboltzmann2d,ns2d_liddriven")
    ap.add_argument("--sel", default="", help="каталог JSON select_agents (<pde>_<tag>.json)")
    ap.add_argument("--rows", default="", help="каталог строк оценок")
    ap.add_argument("--fetch", action="store_true", help="скачать JSON отбора и строки из HF")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    a.sel = a.sel or tempfile.mkdtemp(prefix="sel_json_")
    a.rows = a.rows or tempfile.mkdtemp(prefix="sel_rows_")
    out = []
    for pde in [SR.pde_name(p) for p in a.pdes.split(",") if p]:
        fn = f"{pde}_{a.tag}.json"
        if a.fetch:
            SR.fetch([fn], a.sel, sub="rl_arch/select")
        path = os.path.join(a.sel, fn)
        if not os.path.exists(path):
            print(f"{pde}: нет {fn} в {a.sel}")
            continue
        with open(path) as f:
            sel = json.load(f)
        labs = list(sel.get("agents", {}))
        if a.fetch:
            SR.fetch([f"{pde}_{x}_seed*.json" for x in labs], a.rows)
        rows = SR.load_rows(a.rows, pde, set(labs))
        med = {x: SR.median(r.values()) for x, r in rows.items() if len(r) >= a.min_seeds}
        print(f"{pde}: агентов в отборе {len(labs)}, с измерением не меньше {a.min_seeds} сидов {len(med)} "
              f"(нет: {sorted(set(labs) - set(med))}); нет файлов: {[m['label'] for m in sel.get('missing', [])]}")
        for method in methods_of(sel):
            for main_only in (True, False):
                r = evaluate(sel, med, method, main_only)
                r["pde"] = pde
                out.append(r)
                if "spearman" not in r:
                    print(f"  {method:22s} {r['set']}: {r.get('note')} (n={r['n']})")
                    continue
                fmt = lambda v: "—" if v is None else f"{v:+.2f}"  # noqa: E731
                print(f"  {method:22s} {r['set']:4s} n={r['n']:2d}: Spearman {fmt(r['spearman'])}, Kendall "
                      f"{fmt(r['kendall'])}; выбран {r['chosen']} ({r['chosen_median']:.4f}), лучший {r['best']} "
                      f"({r['best_median']:.4f}), сожаление {r['regret']:.4f} (случайный выбор {r['random_regret']:.4f}); "
                      f"лучший из трёх выбранных — сожаление {r['regret_top3']:.4f}, общих с тремя лучшими {r['overlap_top3']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()
