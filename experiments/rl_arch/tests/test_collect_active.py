#!/usr/bin/env python
"""
Проверки сбора данных с приёмами экономии (collect_active.py) без GPU и без сети. Среда подменена
маленькой копией контракта run_seed (скрипт, обрезка по бюджету, progress_cb после каждого действия,
чекпоинт и докатка, мягкий выход по args.hours), обучение суррогата — ансамблем с известной
динамикой; сеть нигде не обучается.

  * форма цепочек, точки ветвления в среде с --keep-opt, группы общего префикса;
  * правило ранней остановки и выбор пачки (lcb против greedy, скидка за общий префикс);
  * таблица переходов из собранных записей принимается настоящим surrogate.fit_ensemble;
  * раунды: суррогат переучивается на новых переходах, lcb и random выбирают разное, формат
    JSON совпадает с описанием модуля;
  * общий префикс исполняется один раз, эпохи считаются один раз;
  * безнадёжная цепочка останавливается, её переходы остаются;
  * докатка (в том числе посреди группы) и лимит часов; сбор с половинным бюджетом.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_collect_active.py
"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import collect_active as C  # noqa: E402

E, S, G = C.E, C.S, C.G
PDE = "burgers_1d"
ALL = list(range(27))
TRAINED = [0]           # эпох «обучено» подменной средой
CALLS = []              # вызовы run_seed
FITS = []               # число строк обучения суррогата по раундам
SK = "Adam:0.01:1000,Adam:0.0001:2500"
C1 = SK + ",LBFGS:1:1000"
C2 = "Adam:0.001:1000,LBFGS:1:1000,LBFGS:1:1000"
PSO4 = ",".join(["PSO:0.001:300"] * 4)


def delta(a, pa, ep):
    """Изменение log10 ошибки за действие в подменной среде (PSO портит)."""
    a, pa, ep = np.asarray(a), np.asarray(pa), np.asarray(ep, dtype=float)
    li = (a % 9) // 3
    adam = -0.25 * np.array([1.5, 1.0, 0.5])[li] * (ep / 1000) ** 0.5
    lbfgs = -0.4 * np.array([1.0, 0.8, 0.5])[li] * (ep / 1000) ** 0.3 * np.where((pa >= 9) & (pa < 18), 0.5, 1.0)
    return np.where(a < 9, adam, np.where(a < 18, lbfgs, 0.3))


def fake_run_seed(seed, args, progress_cb=None):
    """Контракт online_eval_env.run_seed для --policy script --no-state: тот же порядок шагов."""
    assert args.policy == "script" and args.script_tail == "stop" and args.no_state
    assert args.progress_every == 0 and args.ckpt_min_interval == 0
    assert not (os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_WRITE")), "токены не спрятаны"
    t0 = time.time()
    script = E.parse_chain(args.script)
    name = getattr(args, "_ckpt_name", None)
    CALLS.append(dict(script=args.script, seed=seed, resume=bool(args.resume), ckpt=name,
                      keep=bool(args.keep_opt), mode=args.keep_opt_mode, budget=args.budget))
    spent, chain, hist, l, pa = 0, [], [], 0.01 * (seed % 5 - 2), -1
    if args.resume and name and os.path.exists(f"ckpt_{name}.pt"):
        ck = torch.load(f"ckpt_{name}.pt", weights_only=False)
        spent, chain, hist, l, pa = ck["spent"], ck["chain"], ck["t3"]["hist"], ck["l"], ck["pa"]
    start = spent

    def dump():
        torch.save(dict(spent=spent, chain=chain, t3=dict(hist=hist), l=l, pa=pa), f"ckpt_{name}.pt")

    stop = "budget"
    while spent < args.budget:
        if len(chain) >= len(script):
            stop = "script_end"
            break
        x = script[len(chain)]
        o, lr, ep = E.ACTION_TABLE[x]
        ep = min(ep, args.budget - spent)
        l += float(delta(x, pa, ep))
        pa, spent = x, spent + ep
        TRAINED[0] += ep
        chain.append([o, lr, ep])
        l2 = 10 ** l
        hist.append([spent, 3 * l2, l2, 0.01 * (spent - start)])
        if progress_cb is not None and spent < args.budget:
            progress_cb(dict(seed=seed, policy="script", pde=args.pde, partial=True, l2re=l2, rmse=0.4 * l2,
                             brmse=0.1 * l2, spent=spent, budget=args.budget, n_steps=len(chain),
                             chain=chain, elapsed_s=0.01 * (spent - start)))
        if spent >= args.budget:
            break
        if name and args.ckpt_every and len(chain) % args.ckpt_every == 0:
            dump()
        if args.hours and (time.time() - t0) / 3600.0 >= args.hours:
            if name:
                dump()
            return dict(seed=seed, policy="script", pde=args.pde, unfinished=True, l2re=l2, spent=spent,
                        budget=args.budget, n_steps=len(chain), elapsed_s=0.01 * (spent - start))
    l2 = 10 ** l
    return dict(seed=seed, policy="script", pde=args.pde, l2re=l2, stop_reason=stop, spent=spent,
                rmse=0.4 * l2, brmse=0.1 * l2, budget=args.budget, n_steps=len(chain), chain=chain,
                hist=hist, keep_opt=bool(args.keep_opt), elapsed_s=0.01 * (spent - start))


class FakeEns:
    """Ансамбль с истинной динамикой подменной среды; члены расходятся на 15% изменения."""
    members, level = 3, False

    def predict_ln(self, X, l, member=None):
        a, pa = X[:, 0].astype(int), X[:, 1].astype(int)
        d = delta(a, np.where(pa == S.NONE, -1, pa), 10 ** X[:, 6])
        out = [np.clip(l + d * (1 + 0.15 * (m - 1)), S.L_LO, S.L_HI)
               for m in (range(self.members) if member is None else [member])]
        return np.stack(out) if member is None else out[0]


def make_base(n_chains=25, seed=0):
    """Маленькая таблица переходов двух «задач» (вторая — целевая) в формате surrogate.build_table."""
    rng = np.random.default_rng(seed)
    keys = ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "L", "LN", "LV", "LVN", "SPF", "LAST")
    cols = {k: [] for k in keys}
    ch = 0
    for pi in range(2):
        for _ in range(n_chains):
            g, eps = C.eff(G.random_genome(rng, ALL, 7000, 6), 7000, 6)
            l, pa, spent, spf = math.log10(0.5), -1, 0, np.zeros(3)
            for i, (x, ep) in enumerate(zip(g, eps)):
                ln = l + float(delta(x, pa, ep))
                for k, v in (("PDE", pi), ("FILE", pi), ("CH", ch), ("STEP", i), ("SPENT", spent), ("A", x),
                             ("PA", pa), ("L", l), ("LN", ln), ("LV", np.zeros(6)), ("LVN", np.zeros(6)),
                             ("SPF", spf.copy()), ("LAST", i == len(g) - 1)):
                    cols[k].append(v)
                spf[S.FAM[x]] += ep
                l, pa, spent = ln, x, spent + ep
            ch += 1
    tab = {k: np.array(v) for k, v in cols.items()}
    tab["CTX"] = np.array([[0.4, 0.33, 0.33, 0.25, 1, 0, math.log10(0.5)]] * 2)
    tab["CTX"][0, 0] = 0.6
    tab["NAMES"], tab["SUBDIRS"] = ["other_pde", PDE], ["other", "burgers1d"]
    return dict(key=None, tab=tab, ctx=tab["CTX"][1], k_l2re=2.0, table_s=0.0)


BASE = make_base()


@contextlib.contextmanager
def patched(real_fit=False):
    saved = (C.run_seed, C.base_table, C.fit_surrogate)
    tmp, cwd = tempfile.mkdtemp(prefix="collect_test_"), os.getcwd()
    TRAINED[0] = 0
    del CALLS[:], FITS[:]

    def fit(tab, rows, a, rnd):
        FITS.append(len(rows))
        return FakeEns()

    C.run_seed, C.base_table = fake_run_seed, (lambda a: BASE)
    if not real_fit:
        C.fit_surrogate = fit
    os.chdir(tmp)                       # чекпоинты среды пишутся в текущую папку
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        C.run_seed, C.base_table, C.fit_surrogate = saved
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp, extra, tag="t"):
    argv = ["--pde", PDE, "--smoke", "--budget", "7000", "--save-dir", tmp, "--tag", tag,
            "--plan-width", "8", "--plan-cap", "60", "--plan-top", "4", "--n-mut", "6", "--n-cross", "6",
            "--n-random", "4"] + extra
    with contextlib.redirect_stdout(io.StringIO()):
        return C.main(argv)


def ns(**kw):
    base = dict(budget=7000, max_len=12, keep_opt=False, keep_opt_mode="all", share_prefix=True, share_min=300,
                prefix_bonus=0.1, acq="lcb", lcb_k=1.0, min_dist=1, truncate=0.0, truncate_min=1000,
                truncate_min_ref=5, use_old_target=False, pde=PDE)
    base.update(kw)
    return C.argparse.Namespace(**base)


def check_record(r, budget):
    """Запись согласована сама с собой: действие — строка hist, эпохи сходятся, время растёт."""
    assert len(r["chain"]) == len(r["hist"]) == len(r["err"]) > 0
    spent = 0
    for (o, lr, ep), h in zip(r["chain"], r["hist"]):
        spent += ep
        assert h[0] == spent <= budget and len(h) == 4
        assert abs(h[3] - 0.01 * spent) < 1e-6, "время цепочки с общим префиксом идёт подряд"
    assert r["spent"] == spent and r["env"]["budget"] == budget


def test_chain_helpers():
    p = E.parse_chain
    acts, eps = C.eff(p("Adam:0.01:2500,Adam:0.001:2500,Adam:0.0001:2500,LBFGS:1:1000"), 7000, 12)
    assert len(acts) == 3 and eps == [2500, 2500, 2000], "последнее действие обрезано бюджетом, дальше — спят"
    assert C.eff(p(C1), 3500, 12)[1] == [1000, 2500] and C.eff(p(C1), 7000, 2)[0] == p(SK)
    assert C.eff(p("Adam:0.01:2500,Adam:0.001:2500,LBFGS:1:100"), 3000, 12, clip=False) == ([2], [2500])
    assert C.norm(p(C1), ns(budget=4000, no_clip=True))[1] == [1000, 2500] and C.norm(p(C1), ns(budget=4000))[1][-1] == 500
    # одна физическая цепочка при разной записи обрезанного действия
    assert C.key_of(*C.eff(p("Adam:0.01:2500,Adam:0.01:1000"), 3000, 12)) == \
        C.key_of(*C.eff(p("Adam:0.01:2500,Adam:0.01:2500"), 3000, 12))
    assert E.parse_chain(C.chain_str(acts)) == acts
    ad1, ad2, lb1, lb2, pso = p("Adam:0.01:1000,Adam:0.0001:1000,LBFGS:1:500,LBFGS:0.5:500,PSO:0:100")
    assert all(C.fresh_opt(x, y, False, "all") for x in (ad1, lb1) for y in (ad1, lb2))
    assert not C.fresh_opt(lb1, lb2, True, "safe") and not C.fresh_opt(lb1, lb2, True, "all")
    assert C.fresh_opt(ad1, ad2, True, "safe") and not C.fresh_opt(ad1, ad2, True, "all")
    assert not C.fresh_opt(ad1, ad1, True, "safe")
    assert C.fresh_opt(ad1, lb1, True, "safe") and C.fresh_opt(lb1, pso, True, "all") and C.fresh_opt(pso, lb1, True, "safe")
    # группы: общий префикс Adam, Adam (3500 эпох) у трёх цепочек, четвёртая отдельно
    chains = [p(C1), p(SK + ",LBFGS:0.5:1000,LBFGS:1:500"), p(C2), p(SK + ",PSO:0:100")]
    safe = ns(keep_opt=True, keep_opt_mode="safe")
    assert C.plan_groups(chains, safe) == [(2, [0, 1, 3]), (0, [2])]
    assert C.plan_groups(chains, ns(share_prefix=False)) == [(0, [i]) for i in range(4)]
    assert C.shared_epochs(chains[0], chains[1], safe) == 3500 and C.shared_epochs(chains[0], chains[2], safe) == 0
    # в среде без сбросов нельзя ветвить между L-BFGS и L-BFGS: общий префикс короче
    x, y = p("Adam:0.01:1000,LBFGS:1:1000,LBFGS:1:500"), p("Adam:0.01:1000,LBFGS:1:1000,LBFGS:0.5:500")
    assert C.plan_groups([x, y], safe) == [(1, [0, 1])] and C.shared_epochs(x, y, safe) == 1000
    assert C.plan_groups([x, y], ns()) == [(2, [0, 1])] and C.shared_epochs(x, y, ns()) == 2000
    # цепочка-префикс входит в группу своей длинной версии; короткий префикс не выделяется
    assert C.plan_groups([p(SK), p(C1)], safe) == [(2, [0, 1])]
    assert C.plan_groups([p("Adam:0.01:100,LBFGS:1:500"), p("Adam:0.01:100,PSO:0:100")], safe) == [(0, [0]), (0, [1])]
    # префикс, доходящий до бюджета, не ветвится (чекпоинт пишется только при spent < budget)
    assert C.plan_groups([p(SK + ",LBFGS:1:100"), p(SK + ",LBFGS:1:500")], ns(budget=3500)) == [(1, [0, 1])]
    assert C.edit_dist([1, 2, 3], [1, 2, 3]) == 0 and C.edit_dist([1, 2, 3], [1, 3]) == 1
    print("цепочки, точки ветвления, группы: OK")


def test_truncation_rule():
    recs = [dict(hist=[[1000, 1.0, 0.1, 1.0], [3000, 1.0, 0.01, 2.0]]),
            dict(hist=[[500, 1.0, 0.5, 1.0], [2000, 1.0, 0.05, 2.0]]),
            dict(hist=[[2500, None, None, 1.0]]), dict(hist=[])]
    cur = C.ref_curves(recs)
    assert len(cur) == 2
    assert C.best_at(cur, 400) == (None, 0)
    ref, n = C.best_at(cur, 500)
    assert n == 1 and abs(ref - math.log10(0.5)) < 1e-9
    ref, n = C.best_at(cur, 2000)                       # 0.05 у второй против 10^-1.5 у первой
    assert n == 2 and abs(ref - (-1.5)) < 1e-9
    ref, n = C.best_at(cur, 6000)                       # за концом цепочки — её последнее значение
    assert n == 2 and abs(ref - (-2.0)) < 1e-9
    assert C.make_watch(recs, ns(truncate=0.0)) is None
    w = C.make_watch(recs, ns(truncate=3.0, truncate_min=1000, truncate_min_ref=2))
    assert not w(500, 100.0), "раньше --truncate-min не останавливать"
    assert w(2000, 0.2) and not w(2000, 0.09), "хуже лучшей в 3 раза при тех же эпохах"
    assert not w(2000, float("nan"))
    assert not C.make_watch(recs, ns(truncate=3.0, truncate_min=0, truncate_min_ref=3))(2000, 9.0), "мало опорных"
    print("правило ранней остановки: OK")


def test_select_batch():
    p = E.parse_chain
    chains = [p(C1), p(C2), p(SK + ",LBFGS:0.5:1000"), p("LBFGS:1:1000")]
    mu, sd = np.array([-2.0, -1.9, -1.95, -1.5]), np.array([0.05, 0.4, 0.05, 0.1])
    a = ns(share_prefix=False)
    assert C.select_batch(chains, mu, sd, ns(share_prefix=False, acq="greedy"), 2) == [0, 2]
    assert C.select_batch(chains, mu, sd, a, 2) == [1, 0], "lcb берёт цепочку с большим разбросом"
    assert C.select_batch(chains, mu, sd, ns(share_prefix=False, lcb_k=0.0), 4) == [0, 2, 1, 3]
    # скидка за общий префикс: при близких оценках вторая — концовка того же префикса
    mu2, sd0 = np.array([-2.0, -1.97, -1.95, -1.5]), np.zeros(4)
    assert C.select_batch(chains, mu2, sd0, ns(prefix_bonus=0.0), 2) == [0, 1]
    assert C.select_batch(chains, mu2, sd0, ns(prefix_bonus=0.1), 2) == [0, 2]
    assert C.select_batch(chains, np.array([-2.0, -2.5, -1.95, -1.5]), sd0, ns(prefix_bonus=0.1), 2) == [1, 0]
    assert C.select_batch(chains, mu2, sd0, ns(prefix_bonus=0.1), 1, fixed=[p(SK + ",PSO:0:100")]) == [0]
    assert C.select_batch(chains, mu2, sd0, ns(min_dist=2, share_prefix=False), 3) == [0, 1, 3], "разнообразие пачки"
    assert C.select_batch(chains[:2], mu[:2], sd[:2], a, 5) == [1, 0], "кандидатов меньше пачки"
    print("выбор пачки: OK")


def test_table_and_real_fit():
    ctx = BASE["ctx"]
    rec = dict(chain=[["Adam", 0.01, 1000], ["LBFGS", 1.0, 1000], ["Adam", 0.001, 400]],
               hist=[[1000, 1.0, 0.2, 1.0], [2000, 0.5, 0.02, 2.0], [2400, 0.4, None, 3.0]],
               err=[0.1, None, None], prefix_owner=True, shared_prefix_len=0)
    rows = C.records_to_rows([rec], ctx, k=2.0)
    assert rows["A"].tolist() == [1, 11, 4], "обрезанное действие — ближайшее по эпохам"
    assert rows["PA"].tolist() == [-1, 1, 11] and rows["STEP"].tolist() == [0, 1, 2]
    assert rows["SPENT"].tolist() == [0, 1000, 2000] and rows["LAST"].tolist() == [False, False, True]
    assert rows["SPF"].tolist() == [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0]]
    assert np.allclose(rows["L"], [math.log10(0.5), -1.0, -2.0]), "err, иначе l2re / k"
    assert np.allclose(rows["LN"], [-1.0, -2.0, S.L_HI]), "нечисловая ошибка — расходимость"
    # переходы общего префикса берутся один раз: у владельца
    sib = dict(rec, prefix_owner=False, shared_prefix_len=2)
    two = C.records_to_rows([rec, sib], ctx, k=2.0)
    assert two["STEP"].tolist() == [0, 1, 2, 2] and two["CH"].tolist() == [0, 0, 0, 1]
    assert abs(C.k_l2re([rec], 7.0) - 2.0) < 1e-9 and C.k_l2re([], 7.0) == 7.0
    a = ns(members=2, quick=True, collect_seed=0)
    n_old = len(BASE["tab"]["A"])
    tab, tr, n_new = C.training_table(BASE, [], a)
    assert n_new == 0 and (tab["PDE"][tr] == 0).all() and len(tr) == int((BASE["tab"]["PDE"] == 0).sum())
    tab, tr, n_new = C.training_table(BASE, [rec, sib], a)
    assert n_new == 4 and len(tab["A"]) == n_old + 4 and tab["NAMES"][-1] == PDE + "@collect"
    assert (tab["PDE"][-4:] == 2).all() and np.allclose(tab["CTX"][2], ctx) and tab["CH"][-1] == tab["CH"][:n_old].max() + 2
    assert set(np.unique(tab["PDE"][tr])) == {0, 2}, "старый буфер целевой задачи без --use-old-target не берётся"
    _, tr_old, _ = C.training_table(BASE, [rec], ns(use_old_target=True))
    assert len(tr_old) == n_old + 3
    # настоящий ансамбль HGB на этой таблице и прогон цепочек через него
    ens = C.fit_surrogate(tab, tr, a, 0)
    mu, sd = C.score_chains(ens, ctx, [E.parse_chain(C1), E.parse_chain(PSO4)], [[1000, 2500, 1000], [300] * 4])
    assert mu.shape == sd.shape == (2,) and np.isfinite(mu).all() and mu[0] < mu[1], "Adam+L-BFGS лучше PSO"
    # base_table: готовый кэш .npz читается без сборки из буферов, описатель задачи — из pde_meta.json
    tmp = tempfile.mkdtemp(prefix="collect_tab_")
    try:
        path = os.path.join(tmp, "surrogate_table.npz")
        S.save_table(BASE["tab"], path)
        b = C.base_table(ns(table=path, table_hf="", subdirs="", save_dir=tmp))
        assert b["tab"]["NAMES"] == ["other_pde", PDE] and len(b["tab"]["A"]) == n_old
        assert b["ctx"].shape == (7,) and b["k_l2re"] > 0 and C.base_table(ns(table=path, subdirs="")) is b
        assert C.training_table(b, [rec], a)[2] == 3
    finally:
        C._BASE.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    print("таблица переходов из записей и настоящий суррогат: OK")


REC_KEYS = {"chain", "hist", "seed", "env", "truncated", "shared_prefix_len", "round", "acq", "pred_mean",
            "pred_std", "err", "l2re", "spent", "planned", "actions", "origin", "group", "prefix_owner",
            "epochs_real", "elapsed_s", "stop", "error", "skipped"}
ROUND_KEYS = {"round", "acq", "n_chains", "n_truncated", "n_shared", "cum_chains", "cum_epochs",
              "cum_epochs_planned", "cum_gpu_s", "best_l2re", "fit_s", "sel_s", "cum_fit_s", "n_candidates",
              "n_train_rows", "n_target_rows", "pred_mae", "wall_s"}
TOP_KEYS = {"records", "rounds", "settings", "env", "n_chains", "n_runs", "epochs_real", "epochs_planned",
            "gpu_s", "fit_s", "table_s", "wall_s", "best", "done", "partial", "stop_reason", "round",
            "pending", "sel", "init_left", "seed_cursor"}


def test_rounds_format_and_acquisition():
    with patched() as tmp:
        st = run(tmp, ["--acq", "lcb", "--rounds", "2", "--batch", "3", "--mask", "pso:0"], tag="lcb")
        assert st["done"] and st["stop_reason"] == "rounds" and st["round"] == 2 and not st["pending"]
        assert len(st["records"]) == 6 == st["n_chains"] == st["n_runs"] == len(CALLS)
        assert st["epochs_real"] == TRAINED[0] == st["epochs_planned"] == sum(r["spent"] for r in st["records"])
        assert len(FITS) == 2 and FITS[1] > FITS[0], "суррогат переучен на новых переходах"
        r0, r1 = st["rounds"]
        assert r0["n_target_rows"] == 0 and r1["n_target_rows"] == sum(len(r["chain"]) for r in st["records"][:3])
        assert r0["n_train_rows"] == FITS[0] and r0["n_candidates"] > 3
        assert (r0["cum_chains"], r1["cum_chains"]) == (3, 6) and r1["cum_epochs"] == TRAINED[0]
        assert r1["cum_gpu_s"] == st["gpu_s"] >= 0 and r1["best_l2re"] == st["best"]["l2re"] == min(r["l2re"] for r in st["records"])
        assert r0["pred_mae"] is not None and r0["pred_mae"] < 0.2, "ансамбль знает динамику среды"
        seeds = [r["seed"] for r in st["records"]]
        assert len(set(seeds)) == 6 and all(1000 <= s <= 1999 for s in seeds), "по сиду из пула на цепочку"
        for r in st["records"]:
            check_record(r, 7000)
            assert set(r) == REC_KEYS and r["acq"] == "lcb" and not r["truncated"] and r["shared_prefix_len"] == 0
            assert isinstance(r["pred_mean"], float) and r["pred_std"] > 0
            assert r["env"] == dict(keep_opt=False, mode=None, budget=7000)
            assert E.parse_chain(r["planned"]) == r["actions"] and not any(x in (18, 19, 20) for x in r["actions"])
            assert C.chain_str_rec(r["chain"]) == C.chain_str_rec([list(E.ACTION_TABLE[x]) for x in r["actions"]]) \
                or r["spent"] == 7000
        assert len({r["planned"] for r in st["records"]}) == 6, "цепочки не повторяются"
        assert all(set(x) == ROUND_KEYS for x in st["rounds"]) and TOP_KEYS <= set(st)
        for k in sorted(REC_KEYS | ROUND_KEYS | TOP_KEYS):
            assert k in C.__doc__, f"поле {k!r} не описано в документации модуля"
        disk = json.load(open(os.path.join(tmp, "lcb_collect.json")))
        assert disk == json.loads(json.dumps(st)) and disk["settings"]["acq"] == "lcb" and disk["env"]["budget"] == 7000
        lcb = st

        n_fit = len(FITS)
        st = run(tmp, ["--acq", "random", "--rounds", "2", "--batch", "3", "--mask", "pso:0"], tag="rnd")
        assert len(FITS) == n_fit, "случайному сбору суррогат не нужен"
        assert len(st["records"]) == 6 and all(r["acq"] == "random" and r["pred_mean"] is None for r in st["records"])
        assert {r["planned"] for r in st["records"]} != {r["planned"] for r in lcb["records"]}
        assert lcb["best"]["l2re"] < st["best"]["l2re"], "направленный сбор находит цепочку лучше случайного"
        assert st["rounds"][0]["fit_s"] == 0.0 and st["rounds"][1]["pred_mae"] is None

        st = run(tmp, ["--acq", "greedy", "--rounds", "1", "--batch", "2", "--keep-opt", "--keep-opt-mode", "safe",
                       "--lbfgs-tol", "0", "--init", C1], tag="gr")
        assert [r["acq"] for r in st["records"]] == ["init", "greedy"] and st["records"][0]["planned"] == C1
        assert st["records"][0]["pred_mean"] is not None
        assert st["records"][0]["env"] == dict(keep_opt=True, mode="safe", budget=7000)
        assert all(c["keep"] and c["mode"] == "safe" for c in CALLS[-2:]) and st["env"]["lbfgs_tol"] == 0.0
    print("раунды, формат, lcb против random: OK")


def test_cold_start():
    """Таблица только из цели (--subdirs с папкой цели, без --use-old-target): в первом раунде суррогату не на чем
    учиться — пачка случайная, суррогат не строится; со второго раунда он учится только на собранном здесь."""
    tab = BASE["tab"]
    m = tab["PDE"] == 1
    own = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in tab.items()}
    own["PDE"] = np.zeros(int(m.sum()), dtype=tab["PDE"].dtype)
    own["CTX"], own["NAMES"], own["SUBDIRS"] = tab["CTX"][1:], [PDE], ["burgers1d"]
    base_own = dict(BASE, tab=own)
    with patched() as tmp:
        C.base_table = lambda a: base_own
        st = run(tmp, ["--acq", "lcb", "--rounds", "3", "--batch", "3", "--mask", "pso:0"], tag="cold")
        r0, r1, r2 = st["rounds"]
        assert r0["n_train_rows"] == 0 and r0["fit_s"] == 0.0 and r0["pred_mae"] is None
        assert all(r["acq"] == "random" for r in st["records"][:3]), "первая пачка — случайная"
        n0 = sum(len(r["chain"]) for r in st["records"][:3])
        assert len(FITS) == 2 and FITS[0] == r1["n_train_rows"] == r1["n_target_rows"] == n0, \
            "со второго раунда суррогат учится только на собранном"
        assert r2["n_train_rows"] == sum(len(r["chain"]) for r in st["records"][:6])
        assert all(r["acq"] == "lcb" for r in st["records"][3:]) and len(st["records"]) == 9
    print("холодный старт без чужих буферов: OK")


def test_prefix_sharing():
    c_b, c_c = SK + ",LBFGS:0.5:1000,LBFGS:1:500", SK + ",PSO:0:100"
    with patched() as tmp:
        st = run(tmp, ["--acq", "random", "--rounds", "1", "--batch", "3", "--share-prefix", "--keep-opt",
                       "--keep-opt-mode", "safe", "--init", f"{C1};{c_b};{c_c}"])
        assert [c["script"] for c in CALLS] == [SK, C1, c_b, c_c], "префикс исполнен один раз"
        assert CALLS[0]["ckpt"].endswith("_cpre") and not CALLS[0]["resume"]
        assert all(c["resume"] and c["ckpt"].endswith("_cbr") for c in CALLS[1:])
        assert len({c["seed"] for c in CALLS}) == 1, "сид один на группу"
        assert st["epochs_real"] == TRAINED[0] == 3500 + 1000 + 1500 + 100, "эпохи префикса посчитаны один раз"
        assert st["epochs_planned"] == 4500 + 5000 + 3600 and st["n_runs"] == 4 and st["n_chains"] == 3
        recs = st["records"]
        assert [r["planned"] for r in recs] == [C1, c_b, c_c]
        assert all(r["shared_prefix_len"] == 2 and r["group"] == recs[0]["group"] for r in recs)
        assert [r["prefix_owner"] for r in recs] == [True, False, False]
        assert [r["epochs_real"] for r in recs] == [4500, 1500, 100] and st["rounds"][0]["n_shared"] == 3
        for r in recs:
            check_record(r, 7000)
            assert C.chain_str_rec(r["chain"]) == r["planned"] and r["hist"][:2] == recs[0]["hist"][:2]
        assert not [f for f in os.listdir(".") if f.startswith("ckpt_")], "чекпоинты убраны"
        # цепочка-префикс в пачке: её итог — сам прогон префикса
        del CALLS[:]
        TRAINED[0] = 0
        st = run(tmp, ["--acq", "random", "--rounds", "1", "--batch", "2", "--share-prefix", "--init", f"{C1};{SK}"],
                 tag="ex")
        assert [c["script"] for c in CALLS] == [SK, C1] and st["epochs_real"] == TRAINED[0] == 4500
        assert [r["planned"] for r in st["records"]] == [SK, C1] and [r["prefix_owner"] for r in st["records"]] == [True, False]
        assert st["records"][0]["stop"] == "script_end" and st["records"][0]["spent"] == 3500
        # без флага те же цепочки идут по одной
        del CALLS[:]
        st = run(tmp, ["--acq", "random", "--rounds", "1", "--batch", "2", "--init", f"{C1};{SK}"], tag="no")
        assert [c["script"] for c in CALLS] == [C1, SK] and st["epochs_real"] == 8000
        assert all(c["ckpt"] is None and not c["resume"] for c in CALLS)
        # направленный сбор со скидкой: в пачке появляются концовки общего префикса
        st = run(tmp, ["--acq", "lcb", "--rounds", "2", "--batch", "4", "--share-prefix", "--prefix-bonus", "0.5",
                       "--mask", "pso"], tag="dir")
        assert st["rounds"][-1]["n_shared"] >= 2 and st["epochs_real"] < st["epochs_planned"]
        assert st["epochs_real"] <= sum(r["epochs_real"] for r in st["records"])
        for r in st["records"]:
            check_record(r, 7000)
    print("общий префикс: OK")


def test_truncation():
    bad = PSO4 + ",PSO:0.001:300,PSO:0.001:300"
    flags = ["--acq", "random", "--truncate", "3", "--truncate-min", "300", "--truncate-min-ref", "2"]
    with patched() as tmp:
        st = run(tmp, flags + ["--rounds", "1", "--batch", "3", "--init", f"{C1};{C2};{bad}"])
        r = st["records"][2]
        assert [x["truncated"] for x in st["records"]] == [False, False, True]
        assert r["planned"] == bad and r["stop"] == "truncated" and r["spent"] == 1200 and len(r["chain"]) == 4
        assert all(h[1] is not None for h in r["hist"]), "hist остановленной цепочки — из чекпоинта среды, с лоссом"
        assert all(e is not None for e in r["err"])
        check_record(r, 7000)
        assert st["epochs_real"] == TRAINED[0] == 4500 + 3000 + 1200 and st["epochs_planned"] == 4500 + 3000 + 1800
        assert st["rounds"][0]["n_truncated"] == 1 and st["best"]["l2re"] == st["records"][0]["l2re"]
        assert len(C.records_to_rows([r], BASE["ctx"])["A"]) == 4, "переходы остановленной цепочки остаются данными"
        assert not [f for f in os.listdir(".") if f.startswith("ckpt_")]
        # чекпоинт не прочитался: запись собирается из того, что видел progress_cb (лосса нет)
        saved = C._ckpt_read
        C._ckpt_read = lambda name: None
        try:
            st = run(tmp, flags + ["--rounds", "1", "--batch", "3", "--init", f"{C1};{C2};{bad}"], tag="nock")
        finally:
            C._ckpt_read = saved
        r = st["records"][2]
        assert r["truncated"] and len(r["chain"]) == 4 and r["hist"][0][1] is None and r["hist"][3][2] is not None
        # без флага та же цепочка идёт до конца
        st = run(tmp, ["--acq", "random", "--rounds", "1", "--batch", "3", "--init", f"{C1};{C2};{bad}"], tag="full")
        assert not st["records"][2]["truncated"] and st["records"][2]["spent"] == 1800
        # остановлен общий префикс: концовки не исполняются и не теряются из учёта
        del CALLS[:]
        TRAINED[0] = 0
        e1, e2 = PSO4 + ",Adam:0.01:100", PSO4 + ",LBFGS:1:100"
        st = run(tmp, flags + ["--rounds", "2", "--batch", "2", "--share-prefix", "--init", f"{C1};{C2};{e1};{e2}"],
                 tag="pre")
        assert [c["script"] for c in CALLS] == [C1, C2, PSO4] and len(st["records"]) == 3
        r = st["records"][2]
        assert r["truncated"] and r["planned"] == e1 and r["skipped"] == [E.parse_chain(e2)] and r["spent"] == 1200
        assert st["epochs_real"] == TRAINED[0] == 4500 + 3000 + 1200 and st["epochs_planned"] == 7500 + 1300 + 1300
        assert not st["pending"] and st["done"]
    print("ранняя остановка: OK")


def test_resume_and_limits():
    args = ["--acq", "lcb", "--rounds", "2", "--batch", "3", "--mask", "pso:0"]
    with patched() as tmp:
        full = run(tmp, args, tag="full")
        n_full = len(CALLS)
        part = run(tmp, args + ["--kernel-hours", "1e-9"], tag="part")
        assert not part["done"] and len(part["records"]) == 1 and len(part["pending"]) == 2
        for i in range(20):
            part = run(tmp, args + ["--kernel-hours", "1e-9", "--resume"], tag="part")
            if part["done"]:
                break
        assert part["done"] and i == 4, "по цепочке на кернел: шесть запусков на шесть цепочек"
        assert [(r["planned"], r["seed"], r["round"], r["l2re"]) for r in part["records"]] == \
            [(r["planned"], r["seed"], r["round"], r["l2re"]) for r in full["records"]], "докатка ничего не меняет"
        assert part["epochs_real"] == full["epochs_real"] and part["n_runs"] == full["n_runs"] == n_full
        assert [x["cum_chains"] for x in part["rounds"]] == [3, 6]
        n = len(CALLS)
        again = run(tmp, args + ["--resume"], tag="part")
        assert again["done"] and len(CALLS) == n, "завершённый сбор не пересчитывается"
        more = run(tmp, args[:3] + ["3"] + args[4:] + ["--resume"], tag="part")
        assert more["round"] == 3 and len(more["records"]) == 9 and len(CALLS) == n + 3, "больше раундов — продолжение"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                C.main(["--pde", PDE, "--smoke", "--budget", "3500", "--save-dir", tmp, "--tag", "part", "--resume"])
            raise AssertionError("докатка с другим бюджетом должна отказать")
        except SystemExit as e:
            assert "budget" in str(e)
        # лимит часов всего сбора: закончить цепочку и выйти с итогом
        st = run(tmp, args + ["--hours", "1e-9"], tag="h")
        assert st["done"] and st["stop_reason"] == "hours" and len(st["records"]) <= 1 and st["rounds"] == []
        # докатка посреди группы общего префикса: префикс исполняется заново, эпохи считаются честно
        c_b, c_c = SK + ",LBFGS:0.5:1000,LBFGS:1:500", SK + ",PSO:0:100"
        sh = ["--acq", "random", "--rounds", "1", "--batch", "3", "--share-prefix", "--init", f"{C1};{c_b};{c_c}",
              "--kernel-hours", "1e-9"]
        del CALLS[:]
        TRAINED[0] = 0
        st = run(tmp, sh, tag="sh")
        assert [c["script"] for c in CALLS] == [SK, C1] and len(st["pending"]) == 2 and not st["done"]
        st = run(tmp, sh + ["--resume"], tag="sh")
        st = run(tmp, sh + ["--resume"], tag="sh")
        assert st["done"] and [c["script"] for c in CALLS] == [SK, C1, SK, c_b, c_c]
        assert st["epochs_real"] == TRAINED[0] == 4500 + 5000 + 3600 and st["epochs_planned"] == 13100
        assert [r["planned"] for r in st["records"]] == [C1, c_b, c_c]
        assert [r["shared_prefix_len"] for r in st["records"]] == [2, 2, 0]
        assert [r["prefix_owner"] for r in st["records"]] == [True, True, True], "у каждого прогона префикса свой владелец"
        assert len({r["seed"] for r in st["records"]}) == 3
        assert not [f for f in os.listdir(".") if f.startswith("ckpt_")]
    print("докатка и лимиты: OK")


def test_low_fidelity():
    with patched() as tmp:
        argv = ["--pde", PDE, "--smoke", "--budget", "3500", "--fidelity-note", "half", "--save-dir", tmp, "--tag",
                "lf", "--acq", "lcb", "--rounds", "1", "--batch", "3", "--plan-width", "8", "--plan-cap", "60",
                "--n-mut", "6", "--n-cross", "6", "--n-random", "4", "--init", C1]
        with contextlib.redirect_stdout(io.StringIO()):
            st = C.main(argv)
        assert st["settings"]["budget"] == 3500 and st["settings"]["fidelity_note"] == "half" and st["env"]["budget"] == 3500
        assert all(c["budget"] == 3500 for c in CALLS)
        for r in st["records"]:
            check_record(r, 3500)
            assert r["env"]["budget"] == 3500 and r["spent"] <= 3500
        r = st["records"][0]                # каркас в 4500 эпох обрезан средой: третье действие не началось
        assert r["planned"] == SK and r["spent"] == 3500 and r["stop"] == "budget"
        with contextlib.redirect_stdout(io.StringIO()):
            st = C.main(["--pde", PDE, "--smoke", "--save-dir", tmp, "--tag", "sm", "--acq", "random", "--rounds", "1",
                         "--batch", "2"])
        assert st["settings"]["budget"] == 300 and all(r["spent"] <= 300 for r in st["records"])
        assert st["settings"]["truncate_min"] == 150, "порог ранней остановки по умолчанию — половина бюджета"
        # --no-clip: действие, которое бюджет обрезал бы, не исполняется
        with contextlib.redirect_stdout(io.StringIO()):
            st = C.main(["--pde", PDE, "--smoke", "--no-clip", "--budget", "4000", "--save-dir", tmp, "--tag", "nc",
                         "--acq", "lcb", "--rounds", "1", "--batch", "4", "--plan-width", "8", "--plan-cap", "60",
                         "--n-mut", "6", "--n-cross", "6", "--n-random", "4",
                         "--init", C1 + ";Adam:0.01:2500,Adam:0.001:2500"])
        nominal = {(o, lr): {e for o2, lr2, e in E.ACTION_TABLE if (o2, lr2) == (o, lr)} for o, lr, _ in E.ACTION_TABLE}
        assert [r["planned"] for r in st["records"][:2]] == [SK, "Adam:0.01:2500"] and st["records"][1]["stop"] == "script_end"
        assert all(c[2] in nominal[(c[0], c[1])] for r in st["records"] for c in r["chain"]), "обрезанных действий нет"
        assert all(r["spent"] <= 4000 and r["env"]["budget"] == 4000 for r in st["records"])
    print("половинный бюджет: OK")


if __name__ == "__main__":
    t = time.time()
    test_chain_helpers()
    test_truncation_rule()
    test_select_batch()
    test_table_and_real_fit()
    test_rounds_format_and_acquisition()
    test_cold_start()
    test_prefix_sharing()
    test_truncation()
    test_resume_and_limits()
    test_low_fidelity()
    print(f"ВСЕ ПРОВЕРКИ СБОРА С ПРИЁМАМИ ЭКОНОМИИ ПРОЙДЕНЫ ({time.time() - t:.1f} с)")
