#!/usr/bin/env python
"""Локальная проверка: помогают ли модели среды итогового метода признаки карт (уровни лосса на карте).

Модель среды (track3/surrogate.py, ансамбль из 5 HGB одного шага) сейчас скалярная: действие, прошлое действие, эпохи
по семействам, шаг, потраченный бюджет, log10 ошибки, описатель уравнения. Вариант level добавляет 6 чисел уровня
карт (минимум и среднее трёх карт лосса, surrogate.level6) и для прогона вперёд сам предсказывает их следующие
значения (по 6 моделей на член). Сравнение на псевдоцелях полной таблицы журналов, как conformal_check.py --pseudo:
каждое решаемое уравнение, кроме источников (burgers1d, poisson2d_classic, ns2d_backstep по 20 цепочек), — цель;
модель учится на источниках и 20 своих цепочках цели; тест — сегменты по H действий из остальных цепочек цели
(до --max-test). Варианты:
  scalar        — как сейчас;
  level         — + уровни карт, будущие уровни предсказывает модель;
  level_oracle  — + уровни карт, будущие уровни настоящие (верхняя граница пользы карт);
  hybrid        — первый шаг моделью с уровнями карт (уровни текущего состояния известны), дальше — скалярной.
Метрики по горизонтам 1..H: MAE log10 ошибки; Spearman предсказанного и настоящего изменения log10 ошибки (насколько
модель ранжирует исходы — это нужно модельным целям и учителю).

    python -I experiments/rl_arch/surr_maps_check.py --work <папка с полной таблицей> --out surr_maps_check.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "track3")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DDEBACKEND", "pytorch")

import numpy as np  # noqa: E402

SRC = ("burgers1d", "poisson2d_classic", "ns2d_backstep")


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def segments(tab, rows, H):
    by_ch = {}
    for r in rows:
        by_ch.setdefault(int(tab["CH"][r]), []).append(int(r))
    out = []
    for rs in by_ch.values():
        rs = sorted(rs, key=lambda r: tab["STEP"][r])
        for i in range(len(rs) - H + 1):
            seg = rs[i:i + H]
            if all(tab["STEP"][seg[k + 1]] == tab["STEP"][seg[k]] + 1 for k in range(H - 1)):
                out.append(seg)
    return out


def rollout(S, ens, tab, segs, oracle=False, then=None):
    """Прогон записанных действий сегментов: (H, n) — log10 ошибки по шагам, среднее по членам."""
    m = ens.members
    r0 = np.array([seg[0] for seg in segs])
    H = len(segs[0])
    st = dict(ctx=tab["CTX"][tab["PDE"][r0]].astype(np.float64), step=tab["STEP"][r0].copy(),
              spent=tab["SPENT"][r0].astype(np.int64), spf=tab["SPF"][r0].astype(np.float64),
              pa=tab["PA"][r0].copy(), l=np.tile(tab["L"][r0], (m, 1)).astype(np.float64),
              lv=np.tile(np.nan_to_num(tab["LV"][r0])[None], (m, 1, 1)).astype(np.float64))
    out = []
    for h in range(H):
        rh = np.array([seg[h] for seg in segs])
        if oracle and h > 0:
            st["lv"] = np.tile(np.nan_to_num(tab["LV"][rh])[None], (m, 1, 1)).astype(np.float64)
        st = S.advance(ens if (then is None or h == 0) else then, st, tab["A"][rh], ep=tab["EPA"][rh])
        out.append(st["l"].mean(0))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--own-chains", type=int, default=20)
    ap.add_argument("--max-test", type=int, default=1500)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--seeds", default="0,1", help="сиды выбора своих цепочек (повторы)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import agent_combo as AC
    import offline_rl as O
    import surrogate as S
    full = AC.get_table(argparse.Namespace(table="", work=a.work), list(O.ALL_SUBDIRS), lambda *x: None)
    names = list(full["SUBDIRS"])
    src_idx = [i for i, s in enumerate(names) if s in SRC]
    src_tab, _ = AC.cap_source_chains(AC.subset(full, np.isin(full["PDE"], src_idx)), 20, 0)
    src_ch = set(src_tab["CH"].tolist())
    targets = [s for s in names if s not in SRC and s not in O.UNSOLVED_SUBDIRS]
    H = a.horizon
    res = {}
    for ti, tsub in enumerate(targets):
        pi = names.index(tsub)
        keep = (full["PDE"] == pi) | (np.isin(full["PDE"], src_idx) & np.isin(full["CH"], list(src_ch)))
        tab = AC.subset(full, keep)
        tr = np.where(tab["PDE"] == pi)[0]
        srcr = np.where(tab["PDE"] != pi)[0]
        chs = np.unique(tab["CH"][tr])
        res[tsub] = {}
        for sd in [int(x) for x in a.seeds.split(",")]:
            rng = np.random.default_rng(1000 * ti + sd)
            own_ch = rng.choice(chs, a.own_chains, replace=False)
            own = tr[np.isin(tab["CH"][tr], own_ch)]
            rest = tr[~np.isin(tab["CH"][tr], own_ch)]
            segs = segments(tab, rest, H)
            if len(segs) > a.max_test:
                segs = [segs[i] for i in sorted(rng.choice(len(segs), a.max_test, replace=False))]
            y = np.stack([tab["LN"][np.array([sg[h] for sg in segs])] for h in range(H)])
            l0 = tab["L"][np.array([sg[0] for sg in segs])]
            train = np.concatenate([srcr, own])
            ens_s = S.fit_ensemble(tab, train, level=False, members=a.members, seed=sd)
            ens_l = S.fit_ensemble(tab, train, level=True, members=a.members, seed=sd)
            preds = {"scalar": rollout(S, ens_s, tab, segs), "level": rollout(S, ens_l, tab, segs),
                     "level_oracle": rollout(S, ens_l, tab, segs, oracle=True),
                     "hybrid": rollout(S, ens_l, tab, segs, then=ens_s)}
            row = {}
            for k, pr in preds.items():
                row[k] = dict(mae=[float(np.abs(pr[h] - y[h]).mean()) for h in range(H)],
                              rank=[spearman(pr[h] - l0, y[h] - l0) for h in range(H)])
            res[tsub][str(sd)] = dict(n_test=len(segs), **row)
            print(f"{tsub:26s} сид {sd} тест {len(segs)}: " + "; ".join(
                f"{k} MAE {[round(x, 3) for x in v['mae']]} ранг {[round(x, 2) for x in v['rank']]}"
                for k, v in row.items()), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("\nитог (среднее по целям и повторам):")
    for k in ("scalar", "level", "level_oracle", "hybrid"):
        mae = np.array([[v[k]["mae"] for v in r.values()] for r in res.values()]).reshape(-1, H)
        rk = np.array([[v[k]["rank"] for v in r.values()] for r in res.values()]).reshape(-1, H)
        wins = None
        if k != "scalar":
            base = np.array([[v["scalar"]["mae"] for v in r.values()] for r in res.values()]).reshape(-1, H)
            wins = (mae < base).mean(0)
        print(f"  {k:13s} MAE по горизонтам {np.round(mae.mean(0), 4).tolist()}  ранг {np.round(np.nanmean(rk, 0), 3).tolist()}"
              + (f"  лучше скалярной по MAE в доле случаев {np.round(wins, 2).tolist()}" if wins is not None else ""))


if __name__ == "__main__":
    main()
