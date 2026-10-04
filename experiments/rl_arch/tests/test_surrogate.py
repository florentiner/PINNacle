#!/usr/bin/env python
"""
Проверки суррогатной среды (track3/surrogate.py) без GPU и без сети, на синтетическом буфере
с известной динамикой (две «задачи»; L-BFGS после L-BFGS ничего не меняет, как в среде со сбросом):

  * таблица действий совпадает с online_eval_env, разбор цепочек и маски, согласованность
    переходов совпадает с keep_consistent;
  * таблица переходов: ошибка перед первым действием = E0 задачи, прошлое действие, эпохи по
    семействам, префиксы цепочек в бюджете;
  * ансамбль одного шага и прогон цепочек: высокий R² и Spearman итога на отложенных цепочках;
  * планирование находит почти лучшую по истинной динамике цепочку, уважает бюджет, маску и
    согласованность; plan_next возвращает допустимое действие и останавливается без бюджета.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_surrogate.py
"""
import itertools
import os
import sys
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402

KEYS = ["loss_total", "loss_oper", "loss_bnd"]
BUDGET = 3000
META = {
    "fake_a": dict(in_dim=2, out_dim=1, n_pde=1, n_bnd=2, time=0, inverse=0, init_err=1.0, subdir="fa"),
    "fake_b": dict(in_dim=3, out_dim=2, n_pde=1, n_bnd=4, time=1, inverse=0, init_err=0.5, subdir="fb"),
}
SCALE = {"fake_a": 1.0, "fake_b": 0.5}


def true_delta(a, pa, pde):
    """Изменение log10 ошибки за действие в синтетической среде."""
    o, lr, ep = S.ACTIONS[a]
    li = S.LRI[a]
    if o == "Adam":
        d = -0.2 * (1.5, 1.0, 0.5)[li] * (ep / 1000) ** 0.5
    elif o == "LBFGS":
        d = 0.0 if (pa >= 0 and S.FAM[pa] == 1) else -0.5 * (1.0, 0.8, 0.5)[li] * (ep / 1000) ** 0.3
    else:
        d = 0.1
    return SCALE[pde] * d


def true_final(chain, pde):
    l, pa = np.log10(META[pde]["init_err"]), -1
    for a in chain:
        l += true_delta(a, pa, pde)
        pa = a
    return l


def make_files(pde, n_files, chains_per_file, rng):
    """Файлы в формате дампа (старый формат: старт — нули, next_state верный)."""
    files = []
    for _ in range(n_files):
        out = []
        for _ in range(chains_per_file):
            chain, spent = [], 0
            while len(chain) < 5:
                a = int(rng.integers(0, 27))
                if spent + S.EPOCHS[a] > BUDGET + 2500:
                    break
                chain.append(a)
                spent += S.EPOCHS[a]
            l, pa = np.log10(META[pde]["init_err"]), -1
            maps = [{k: np.zeros((26, 26), np.float32) for k in KEYS}]
            errs = []
            for a in chain:
                l += true_delta(a, pa, pde) + rng.normal(0, 0.02)
                pa = a
                errs.append(10 ** l)
                maps.append({k: np.full((26, 26), l + 3.0, np.float32) for k in KEYS})
            for j, a in enumerate(chain):
                done = -1 if j == len(chain) - 1 else 0
                out.append(dict(state=dict(maps[j]), next_state=dict(maps[j + 1]),
                                action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                                reward=-(errs[j] + (1.0 if done == -1 else 0.0)), done=done))
        files.append(out)
    return files


def test_tables_and_parsing():
    try:
        import online_eval_env as E
        assert [tuple(x) for x in E.ACTION_TABLE] == [tuple(x) for x in S.ACTIONS], "таблица действий разошлась"
    except ImportError:
        print("online_eval_env не импортируется: сверка таблицы по эпохам offline_rl")
    assert [O.action_epochs(a) for a in range(27)] == S.EPOCHS.tolist()
    assert tuple(S.UNSOLVED) == tuple(O.UNSOLVED_SUBDIRS)
    spec = "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000"
    assert S.parse_chain(spec) == [2, 8, 11] and S.chain_str([2, 8, 11]) == spec
    m = S.parse_mask("pso:0.001,pso:0.0001")
    assert m.sum() == 21 and not m[21:].any() and m[:21].all()
    rng = np.random.default_rng(3)
    A = rng.integers(0, 27, 400)
    EP = np.repeat(np.arange(40), 10)
    STEP = np.tile(np.arange(10), 40)
    PA = np.where(STEP == 0, -1, np.r_[-1, A[:-1]])
    for mode in ("all", "safe"):
        assert (S.consistent_ok(PA, A, mode) == O.keep_consistent(A, EP, STEP, mode=mode)).all(), mode
    print("таблица действий, разбор цепочек и маски, согласованность: OK")


def build():
    rng = np.random.default_rng(0)
    eps = {"fa": make_files("fake_a", 12, 25, rng), "fb": make_files("fake_b", 12, 25, rng)}
    return S.build_table(episodes=eps, meta=META, verbose=False)


def test_table(tab):
    assert tab["NAMES"] == ["fake_a", "fake_b"]
    first = tab["STEP"] == 0
    e0 = np.log10([META[n]["init_err"] for n in tab["NAMES"]])
    assert np.allclose(tab["L"][first], e0[tab["PDE"][first]]), "ошибка перед первым действием — E0"
    assert (tab["PA"][first] == -1).all() and (tab["PA"][~first] == tab["A"][np.where(~first)[0] - 1]).all()
    assert np.allclose(tab["L"][~first], tab["LN"][np.where(~first)[0] - 1]), "текущая ошибка = прошлая следующая"
    assert np.allclose(tab["SPF"].sum(1), tab["SPENT"]), "эпохи по семействам в сумме = потрачено"
    assert (tab["LV"][first] == 0).all(), "стартовая карта нулевая (как в онлайновой среде)"
    lc = S.logged_chains(tab, budget=BUDGET)
    assert (lc["spent"] <= BUDGET).all() and len(lc["chains"]) == len(np.unique(tab["CH"]))
    i = 7
    rows = np.where(tab["CH"] == lc["CH"][i])[0]
    assert lc["chains"][i] == tab["A"][rows][:len(lc["chains"][i])].tolist()
    print(f"таблица переходов ({len(tab['A'])} переходов, {len(lc['chains'])} цепочек): OK")


def test_model_and_rollout(tab):
    ch = tab["CH"]
    test_ch = np.unique(ch)[::5]
    tr = np.where(~np.isin(ch, test_ch))[0]
    te = np.where(np.isin(ch, test_ch))[0]
    ens = S.fit_ensemble(tab, tr, members=3, quick=True)
    p = ens.predict_ln(S.table_features(tab, te), tab["L"][te])
    assert p.shape == (3, len(te))
    r2 = S.r2(tab["LN"][te], p.mean(0))
    assert r2 > 0.9, r2
    # прогон записанных цепочек отложенных цепочек от старта
    lc = S.logged_chains(tab, mask=np.isin(ch, test_ch), budget=BUDGET)
    L = S.rollout(ens, tab["CTX"][lc["PDE"]], lc["chains"])
    rho = S.spearman(L.mean(0), lc["final"])
    assert rho > 0.85, rho
    # пустое продолжение не двигает состояние; одна цепочка = тот же результат, что в пачке
    one = S.rollout(ens, tab["CTX"][lc["PDE"][0]], [lc["chains"][0]])
    assert np.allclose(one[:, 0], L[:, 0])
    # вариант с уровнем карт: шесть моделей уровня на член, прогон работает
    lv = S.fit_ensemble(tab, tr, members=2, quick=True, level=True)
    assert len(lv.lv_models) == 2 and len(lv.lv_models[0]) == 6
    L2 = S.rollout(lv, tab["CTX"][lc["PDE"]], lc["chains"])
    assert np.isfinite(L2).all() and S.spearman(L2.mean(0), lc["final"]) > 0.8
    print(f"ансамбль одного шага R²={r2:.3f}, прогон цепочек ρ={rho:.2f}: OK")
    return S.fit_ensemble(tab, np.arange(len(tab["A"])), members=3, quick=True)


def test_plan(tab, ens):
    pde = "fake_a"
    ctx = tab["CTX"][tab["NAMES"].index(pde)]
    best_true = min(true_final(c, pde) for n in range(1, 4) for c in itertools.product(range(27), repeat=n)
                    if sum(S.EPOCHS[a] for a in c) <= BUDGET)
    r = S.plan(ens, ctx, budget=BUDGET, max_len=3, cap=30000, top=5, keep_depth=3)
    c, m, s, j = r["best"][0]
    assert sum(S.EPOCHS[a] for a in c) <= BUDGET and 1 <= len(c) <= 3
    assert true_final(c, pde) <= best_true + 0.15, (S.chain_str(c), true_final(c, pde), best_true)
    assert len(r["allJ"]) == sum(1 for n in range(1, 4) for x in itertools.product(range(27), repeat=n)
                                 if sum(S.EPOCHS[a] for a in x) <= BUDGET)
    mm, ss, jj = S.chain_score(ens, ctx, c)
    assert abs(jj - j) < 1e-9, "J плана совпадает с прогоном той же цепочки"
    # штраф за разброс: J = среднее + k * разброс
    r1 = S.plan(ens, ctx, budget=BUDGET, max_len=3, k=1.0, top=3)
    for c1, m1, s1, j1 in r1["best"]:
        assert abs(j1 - (m1 + s1)) < 1e-9
    # маска и согласованность: без L-BFGS -> L-BFGS и Adam -> Adam с тем же шагом
    no_pso = S.parse_mask("pso")
    rs = S.plan(ens, ctx, budget=BUDGET, max_len=4, allowed=no_pso, consistent="safe", cap=3000, width=64, top=5)
    for c2, *_ in rs["best"]:
        assert all(S.FAM[a] != 2 for a in c2)
        assert S.consistent_ok(np.r_[-1, c2[:-1]], np.array(c2), "safe").all(), S.chain_str(c2)
    print(f"план: {S.chain_str(c)} (истинный итог {true_final(c, pde):+.2f}, лучший {best_true:+.2f}): OK")


def test_plan_next(tab, ens):
    ctx = tab["CTX"][0]
    t0 = time.time()
    r = S.plan_next(ens, ctx, [], None, budget=BUDGET, max_len=4)
    assert r["action"] is not None and S.EPOCHS[r["action"]] <= BUDGET and r["plan"][0] == r["action"]
    hist = [11]                                       # L-BFGS 1 x 1000 исполнен
    r2 = S.plan_next(ens, ctx, hist, -0.5, budget=BUDGET, max_len=4)
    assert r2["action"] is None or (1000 + S.EPOCHS[r2["action"]] <= BUDGET)
    assert r2["stop_J"] == -0.5
    # после L-BFGS повтор L-BFGS по истинной динамике пуст: модель не должна начинать с него
    assert r2["action"] is None or S.FAM[r2["action"]] != 1, S.chain_str(r2["plan"])
    full = [2, 8]                                     # 5000 эпох при бюджете 3000: продолжать нечем
    r3 = S.plan_next(ens, ctx, full, -1.0, budget=BUDGET)
    assert r3["action"] is None and r3["plan"] == []
    try:
        S.plan_next(ens, ctx, [11], None)
        raise AssertionError("без наблюдённой ошибки после действия plan_next должен падать")
    except ValueError:
        pass
    print(f"plan_next: первое действие {S.action_str(r['action'])}, после L-BFGS "
          f"{'стоп' if r2['action'] is None else S.action_str(r2['action'])} ({time.time() - t0:.1f} с): OK")


if __name__ == "__main__":
    t0 = time.time()
    test_tables_and_parsing()
    tab = build()
    test_table(tab)
    ens = test_model_and_rollout(tab)
    test_plan(tab, ens)
    test_plan_next(tab, ens)
    print(f"ВСЕ ПРОВЕРКИ СУРРОГАТА ПРОЙДЕНЫ за {time.time() - t0:.0f} с")
