#!/usr/bin/env python
"""
Проверки тренера комбинированного агента (agent_combo.py) без GPU и без сети, на крошечной
синтетической таблице переходов (три задачи с настоящими именами из pde_meta.json, ~500 переходов,
несколько десятков шагов градиента):

  * новые цепочки (--extra-data) превращаются в строки таблицы: E0 перед первым действием,
    обрезанное бюджетом последнее действие, обрыв на действии вне 27, три источника ошибки;
  * соседи действия на сетке (PANI), грубая маска, бутстреп MVE;
  * каждый приём по отдельности и все вместе отрабатывают; чекпоинт грузится через
    online_eval_env.load_agent, q_values работает на состоянии режима loss со скалярным контекстом
    и не зависит от каналов лоссов; каркас записан в meta; маска соблюдается.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_agent_combo.py
"""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("HF_TOKEN", None)            # тест ничего не заливает
os.environ.pop("HF_TOKEN_WRITE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import agent_combo as AC  # noqa: E402

GUIDE = "Adam:0.01:1000,Adam:0.0001:2500"
QUIET = lambda *a: None  # noqa: E731


def true_delta(a, pa):
    o, lr, ep = S.ACTIONS[a]
    li = S.LRI[a]
    if o == "Adam":
        return -0.2 * (1.5, 1.0, 0.5)[li] * (ep / 1000) ** 0.5
    if o == "LBFGS":
        return 0.0 if (pa >= 0 and S.FAM[pa] == 1) else -0.5 * (1.0, 0.8, 0.5)[li] * (ep / 1000) ** 0.3
    return 0.1


def make_table(rng, n_chains=14):
    meta = O.pde_meta()
    names = ["poissonboltzmann2d", "ns2d_liddriven", O.subdir_to_pde("burgers1d")]
    cols = {k: [] for k in AC.ROWKEYS}
    cid = 0
    for pi, name in enumerate(names):
        for _ in range(n_chains):
            l, pa, spent, spf = np.log10(meta[name]["init_err"]), -1, 0, np.zeros(3)
            for k in range(12):
                a = int(rng.integers(0, 27))
                ln = l + true_delta(a, pa) + rng.normal(0, 0.02)
                for key, v in (("PDE", pi), ("FILE", cid // 4), ("CH", cid), ("STEP", k), ("SPENT", spent),
                               ("A", a), ("PA", pa), ("L", l), ("LN", ln), ("LV", np.zeros(6)),
                               ("LVN", np.zeros(6)), ("SPF", spf.copy()), ("LAST", k == 11),
                               ("EPA", int(S.EPOCHS[a])), ("GRP", cid)):
                    cols[key].append(v)
                spf[S.FAM[a]] += S.EPOCHS[a]
                spent, pa, l = spent + int(S.EPOCHS[a]), a, ln
            cid += 1
    tab = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "EPA", "GRP"):
        tab[k] = tab[k].astype(np.int64)
    tab["CTX"] = np.stack([S.pde_ctx(meta[n]) for n in names])
    tab["NAMES"] = names
    tab["SUBDIRS"] = [meta[n]["subdir"] for n in names]
    return tab


def make_extra(tab):
    """Записи в формате collect_active.py: (1) с err, обрезанным бюджетом последним действием, владелец
    группы; (2) без err, с итоговыми rmse/brmse; (3) действие вне 27 (SOAP) на втором шаге, среда без
    сбросов; (4) без единиц ошибки; (5) концовка от общего префикса группы; (6) расходимость на втором
    шаге; (7) сбор с бюджетом 1500 — последнее действие обрезано чужим бюджетом."""
    e0 = float(10 ** tab["CTX"][0, 6])
    full = [["Adam", 0.01, 1000], ["Adam", 0.0001, 2500], ["LBFGS", 1.0, 1000], ["LBFGS", 1.0, 1000],
            ["LBFGS", 1.0, 1000], ["LBFGS", 1.0, 100], ["LBFGS", 1.0, 400]]    # 400 из 500: обрезано бюджетом
    spent = np.cumsum([c[2] for c in full]).tolist()
    err = [e0 * 10 ** (-0.3 * (i + 1)) for i in range(len(full))]
    hist = [[s, 1e-3, e * 2.0, 1.0 * i] for i, (s, e) in enumerate(zip(spent, err))]
    env = dict(keep_opt=False, mode=None, budget=7000)
    return [
        dict(chain=full, hist=hist, err=err, seed=1, env=env, pde="poissonboltzmann2d", group=7,
             prefix_owner=True, shared_prefix_len=2),
        dict(chain=full[:3], hist=hist[:3], rmse=err[2] * 0.75, brmse=err[2] * 0.25, seed=2, env="reset"),
        dict(chain=[full[0], ["SOAP", 0.003, 1000], full[2]],
             hist=[hist[0], [2000, 1e-3, 0.1, 2.0], [3000, 1e-3, 0.05, 3.0]], err=[err[0], 0.1, 0.05],
             seed=3, env=dict(keep_opt=True, mode="safe", budget=7000)),
        dict(chain=full[:2], hist=hist[:2], seed=4),
        dict(chain=full[:2] + [["LBFGS", 0.5, 1000]], hist=hist[:2] + [[4500, 1e-3, err[2], 3.0]],
             err=err[:2] + [err[2] / 2], seed=1, group=7, prefix_owner=False, shared_prefix_len=2),
        dict(chain=[full[0], ["LBFGS", 1.0, 1000]], hist=[hist[0], [2000, None, None, 2.0]],
             err=[err[0], None], seed=6, env=env),
        dict(chain=[full[0], ["Adam", 0.0001, 500]], hist=[hist[0], [1500, 1e-3, err[1] * 2, 2.0]],
             err=err[:2], seed=7, env=dict(keep_opt=False, mode=None, budget=1500)),
    ]


N_X_CHAINS, N_X_ROWS = 7, 7 + 3 + 1 + 2 + 1 + 2 + 1


def test_extra_rows(tab, work):
    recs = make_extra(tab)
    n0 = len(tab["A"])
    t2, st = AC.extra_rows(tab, recs, 0)
    new = np.arange(n0, len(t2["A"]))
    assert st["chains"] == N_X_CHAINS and st["rows"] == N_X_ROWS == len(new), st
    assert st["skipped"] == 0 and st["cut"] == 2 and st["truncated"] == 1 and st["diverged"] == 1, st
    assert st["shared_skipped"] == 2 and abs(st["k_err_per_l2re"] - 0.5) < 1e-9, st
    assert st["err_src"] == {"err": 5, "l2re*k_записи": 1, "l2re*k_сбора": 1}, st
    assert st["envs"] == {"reset": 6, "keep_safe": 1} and st["budgets"] == {"7000": 6, "1500": 1}, st
    first = new[t2["STEP"][new] == 0]
    assert len(first) == 6 and np.allclose(t2["L"][first], t2["CTX"][0, 6]), "ошибка перед первым действием — E0"
    r = new[:7]
    assert t2["A"][r].tolist() == S.parse_chain("Adam:0.01:1000,Adam:0.0001:2500," + "LBFGS:1:1000," * 3
                                                + "LBFGS:1:100,LBFGS:1:500")
    assert t2["EPA"][r].tolist() == [1000, 2500, 1000, 1000, 1000, 100, 400], "обрезанное действие: фактические эпохи"
    assert (t2["SPENT"][r] + t2["EPA"][r])[-1] == AC.BUDGET and np.allclose(t2["SPF"][r].sum(1), t2["SPENT"][r])
    assert t2["PA"][r].tolist() == [-1] + t2["A"][r][:-1].tolist() and t2["LAST"][r].tolist() == [False] * 6 + [True]
    assert np.allclose(t2["L"][r][1:], t2["LN"][r][:-1]) and np.allclose(np.diff(t2["LN"][r]), -0.3)
    # без err: ошибка шага = l2re * k (k записи по итоговым rmse/brmse либо медиана по сбору)
    assert np.allclose(t2["LN"][new[7:10]], t2["LN"][r[:3]]), "пересчёт l2re в rmse+brmse (k записи)"
    assert np.allclose(t2["LN"][new[11:13]], t2["LN"][r[:2]]), "пересчёт l2re в rmse+brmse (k сбора)"
    # концовка от общего префикса: одна строка, состояние проведено через префикс, группа владельца
    x = new[13]
    assert (t2["STEP"][x], t2["SPENT"][x], t2["PA"][x]) == (2, 3500, S.parse_chain("Adam:0.0001:2500")[0])
    assert t2["SPF"][x].tolist() == [3500, 0, 0] and np.isclose(t2["L"][x], t2["LN"][r[1]])
    assert t2["GRP"][x] == t2["CH"][r[0]] == t2["GRP"][r[0]] and t2["CH"][x] != t2["CH"][r[0]]
    assert (t2["GRP"][:n0] == t2["CH"][:n0]).all()
    # расходимость: ошибка на верхнем пределе, переход терминальный; чужой бюджет: обрезанное не берём
    dv = new[14:16]
    assert t2["LN"][dv[1]] == S.L_HI and AC.real_transitions(t2, dv)["d"].tolist() == [0, 1]
    assert t2["A"][new[16]] == S.parse_chain("Adam:0.01:1000")[0] and t2["LAST"][new[16]]
    # одна запись без единиц ошибки: пропуск; с --extra-err-scale проходит
    _, st1 = AC.extra_rows(tab, [recs[3]], 0)
    assert st1["chains"] == 0 and st1["skipped"] == 1 and st1["err_src"] == {"нет": 1}, st1
    _, st2 = AC.extra_rows(tab, [recs[3]], 0, err_scale=0.5)
    assert st2["chains"] == 1 and st2["err_src"] == {"l2re*scale": 1}, st2
    R = AC.real_transitions(t2, r)
    assert R["d"].tolist() == [0, 0, 0, 0, 0, 0, 1] and np.allclose(R["r"], 0.3, atol=1e-5)
    lc = AC.chain_finals(t2, 0)
    i = int(np.where(lc["CH"] == t2["CH"][r[0]])[0][0])
    assert lc["spent"][i] == AC.BUDGET and len(lc["chains"][i]) == 7, "обрезанное действие входит в цепочку"
    half = AC.Models(t2, work, members=2, quick=True, log=QUIET).half_of_row
    assert half[x] == half[r[0]], "записи одной группы — в одной половине"
    print(f"новые цепочки -> строки таблицы ({st['rows']} строк из {st['records']} записей): OK")
    return recs


def test_grid_and_masks():
    NB, NM = AC.neighbours(np.ones(27, bool))
    i = S.parse_chain("Adam:0.001:1000")[0]
    assert set(NB[i][NM[i]]) == set(S.parse_chain("Adam:0.01:1000,Adam:0.0001:1000,Adam:0.001:100,Adam:0.001:2500"))
    j = S.parse_chain("PSO:0:200")[0]
    assert set(NB[j][NM[j]]) == set(S.parse_chain("PSO:0:100,PSO:0:300")), "PSO с нулевым шагом: только длины"
    k = S.parse_chain("PSO:0.001:100")[0]
    assert set(NB[k][NM[k]]) == set(S.parse_chain("PSO:0.0001:100,PSO:0.001:200"))
    long = AC.make_allowed("long")
    assert long.sum() == 12 and (S.EPOCHS[long] >= 500).all()
    NB, NM = AC.neighbours(long)
    assert long[NB[NM]].all(), "соседи только среди разрешённых"
    c = AC.coarse_mask(long, 1)
    assert c.sum() == 4 and (S.LRI[c] == 1).all() and long[c].all()
    assert AC.make_allowed("mpso").sum() == 21 and AC.make_allowed("pso").sum() == 18
    print("сетка действий, соседи, маски: OK")


def test_mve(tab, work):
    models = AC.Models(tab, work, members=3, quick=True, log=QUIET)
    ens = models.get("full")
    rows = np.where((tab["PDE"] == 0) & (tab["SPENT"] + tab["EPA"] <= AC.BUDGET))[0][:40]
    R = AC.real_transitions(tab, rows)
    ag = AC.Agent(*AC.norm_stats(R["s"]), n_heads=2, seed=0)
    allowed = AC.make_allowed("long")
    B1 = AC.mve_values(ag, ens, tab, rows, 1, 0.5, 1.0, allowed)
    B2 = AC.mve_values(ag, ens, tab, rows, 2, 0.5, 1.0, allowed)
    assert B1.shape == B2.shape == (len(rows),) and np.isfinite(B1).all() and np.isfinite(B2).all()
    assert (B1[R["d"] == 1] == 0).all() and (B2[R["d"] == 1] == 0).all(), "терминальное s' — без бутстрепа"
    # K=1 вручную для одной строки: max_a [r~ + (не конец) * max Q_цели]
    i = int(np.where(R["d"] == 0)[0][0])
    st = AC.next_state(AC.model_state(tab, rows[[i]]), tab["A"][rows[[i]]], tab["EPA"][rows[[i]]], tab["LN"][rows[[i]]])
    acts = np.where(allowed)[0]
    rep = AC.take(st, np.zeros(len(acts), int))
    P, ep = AC.model_predict(ens, tab, rep, acts)
    nst = AC.next_state(rep, acts, ep, P.mean(0))
    qn = ag.q(AC.obs_of(tab, nst), allowed, target=True).max(1)
    ref = (rep["l"] - P.mean(0) - 0.5 * P.std(0) + np.where(nst["spent"] >= AC.BUDGET, 0.0, qn)).max()
    assert abs(B1[i] - ref) < 1e-6, (B1[i], ref)
    st = AC.prefix_states(ens, tab, 0, S.parse_chain(GUIDE), 7, np.random.default_rng(0))
    assert (st["spent"] == 3500).all() and (st["step"] == 2).all() and (st["pa"] == S.parse_chain(GUIDE)[-1]).all()
    assert np.allclose(st["spf"], [3500, 0, 0]) and (st["l"] < tab["CTX"][0, 6]).all()
    print("бутстреп MVE и состояния после каркаса: OK")


def check_ckpt(E, rep, guide, mask_n):
    """Чекпоинт глазами оценки: load_agent, meta, q_values на состоянии режима loss с контекстом."""
    net, mean, std, variant = E.load_agent(rep["_paths"]["ckpt"])
    meta = net._meta
    assert variant == "stat_dqn" and np.asarray(mean).shape == (1, 15, 1, 1)
    assert meta["guide"] == guide and meta["state_mode"] == "loss" and meta["ctx_err"] == "err"
    assert meta["ctx_kmax"] == 12 and meta["ctx_budget"] == 7000 and meta["pde_ctx"] and meta["err_norm"] == "init"
    assert len(meta["mask"]) == mask_n, meta["mask"]
    assert meta["guide_actions"] == (S.parse_chain(guide) if guide else [])
    g = S.parse_chain(guide) if guide else []
    maps = O.loss_state(1e-2, 5e-3, 5e-3, 1e-1)
    st = E.add_scalar_ctx(maps, len(g), 12, int(sum(S.EPOCHS[a] for a in g)), 7000, (g[-1] if g else None), 0.03)
    st = np.concatenate([st, np.stack([np.full((26, 26), v, np.float32) for v in O.pde_desc(rep["pde"])])])
    q = E.q_values(net, st, mean, std, variant)
    assert q.shape == (27,) and np.isfinite(q).all()
    st2 = st.copy()
    st2[:4] = O.loss_state(3.0, 1.0, 2.0, None)
    assert np.abs(E.q_values(net, st2, mean, std, variant) - q).max() < 1e-5, "каналы лоссов не должны влиять"
    allowed = np.ones(27, bool)
    allowed[meta["mask"]] = False
    for key, sc in rep["offline"]["final"].items():
        for spec in [sc["nominal"]] + sc["closed_chains"]:
            ch = S.parse_chain(spec)
            assert ch[:len(g)] == g[:len(ch)], f"{key}: цепочка не начинается с каркаса: {spec}"
            assert allowed[ch[len(g):]].all(), f"{key}: запрещённое действие в {spec}"
    v = rep["verify"]
    assert v and v.get("ok"), v
    assert rep["eval_cmd"] == meta["eval_cmd"] and "--no-state --state-mode loss" in rep["eval_cmd"]
    if guide:
        assert f"--policy script --script {guide} --script-tail agent --model-file" in rep["eval_cmd"]
    else:
        assert "--policy agent --model-file" in rep["eval_cmd"]
    return meta


def test_components(tab, recs, tmp):
    import online_eval_env as E
    tpath = os.path.join(tmp, "table.npz")
    S.save_table(tab, tpath)
    xpath = os.path.join(tmp, "extra.json")
    with open(xpath, "w") as f:
        json.dump(dict(records=recs, env=dict(keep_opt=False, mode=None, budget=7000)), f)
    base = ["--pde", "pb2d", "--table", tpath, "--work", os.path.join(tmp, "work"), "--out-dir", tmp,
            "--smoke", "--steps", "24", "--regen", "12", "--n-branch", "40", "--mve-n", "30", "--batch", "32",
            "--members", "3", "--tgt-every", "8"]
    off = ["--offline-eval", "none"]
    cases = [
        ("base", off, "", 15),
        ("mbpo", off + ["--mbpo-h", "2", "--mopo-lam", "0.5"], "", 15),
        ("cql", off + ["--cql-alpha", "0.2"], "", 15),
        ("iql", off + ["--iql-tau", "0.7"], "", 15),
        ("guide", off + ["--guide", GUIDE, "--mbpo-h", "2"], GUIDE, 15),
        ("redq", off + ["--ens", "3", "--ens-min", "2", "--utd", "2"], "", 15),
        ("mve", off + ["--mve", "2", "--mopo-lam", "0.5"], "", 15),
        ("dqfd", off + ["--distill-top", "30", "--distill-w", "0.5", "--mask", "all"], "", 0),
        ("pani", off + ["--pani-w", "0.2"], "", 15),
        ("sched", off + ["--mask-schedule", "--mask", "mpso", "--mbpo-h", "1"], "", 6),
        ("small", off + ["--target-only", "--target-chains", "6", "--mbpo-h", "1"], "", 15),
        ("xonly", off + ["--extra-data", xpath, "--extra-only-target", "--cql-alpha", "0.1"], "", 15),
        ("full", ["--guide", GUIDE, "--mbpo-h", "2", "--mopo-lam", "0.5", "--cql-alpha", "0.1", "--ens", "3",
                  "--ens-min", "2", "--utd", "2", "--mve", "1", "--distill-top", "30", "--distill-w", "0.5",
                  "--pani-w", "0.1", "--mask-schedule", "--extra-data", xpath, "--offline-eval", "half",
                  "--offline-steps", "12"], GUIDE, 15),
    ]
    for tag, extra, guide, mask_n in cases:
        t0 = time.time()
        rep = AC.run(_args(base + extra + ["--tag", f"t_{tag}"]), log=QUIET)
        meta = check_ckpt(E, rep, guide, mask_n)
        with open(rep["_paths"]["report"]) as f:
            disk = json.load(f)
        assert disk["kind"] == "agent_combo_train" and disk["partial"] and disk["tag"] == f"t_{tag}"
        assert disk["curves"]["td"] and all(np.isfinite(x) for x in disk["curves"]["td"])
        assert os.path.basename(rep["_paths"]["report"]) == f"poissonboltzmann2d_t_{tag}_train.json"
        assert rep["_paths"]["ckpt"].endswith(os.path.join("rl_arch", "agents_online", f"t_{tag}.pt"))
        sz, tr = rep["sizes"], rep["train"]
        if tag == "mbpo":
            assert tr["n_model"] > 0 and tr["model_unc"] is not None
        if tag == "guide":
            assert sz["real_used"] < rep_base["sizes"]["real_used"], "с каркасом — только переходы после него"
            assert meta["guide_epochs"] == 3500
        if tag == "dqfd":
            assert tr["demo"]["rows"] > 0 and disk["curves"]["dqfd"][-1] is not None
        if tag in ("cql", "iql", "pani"):
            assert disk["curves"][tag][-1] is not None, tag
        if tag == "small":
            assert sz["pdes"] == 1 and sz["target_chains"] == 6 and sz["chains"] == 6, sz
        if tag == "xonly":
            assert sz["target_chains"] == N_X_CHAINS and sz["target_rows"] == N_X_ROWS, sz
            assert rep["extra"]["chains"] == N_X_CHAINS and rep["extra"]["envs"] == {"reset": 6, "keep_safe": 1}
        if tag == "full":
            assert sz["target_chains"] == sz["target_chains_buffer"] + N_X_CHAINS
            h = rep["offline"]["halves"]["half0"]
            assert np.isfinite(h["ind"]["d_closed"]) and np.isfinite(h["own"]["d_closed"])
            assert set(rep["offline"]["final"]) == {"full", "half0", "half1"}
            assert meta["combo"]["mve"] == 1 and meta["combo"]["ens"] == 3
            assert tr["demo"]["chains"] > 0, "с каркасом лучшие цепочки должны находиться"
        if tag == "base":
            rep_base = rep
        print(f"  {tag}: {time.time() - t0:.1f} с, цепочка {rep['offline']['final']['full']['nominal'][:70]}")
    print("все приёмы по отдельности и вместе, чекпоинт через online_eval_env: OK")


def _args(argv):
    a = AC.build_parser().parse_args(argv)
    a._argv = argv
    return a


def main():
    t0 = time.time()
    tab = make_table(np.random.default_rng(0))
    test_grid_and_masks()
    with tempfile.TemporaryDirectory() as tmp:
        recs = test_extra_rows(tab, os.path.join(tmp, "work0"))
        test_mve(tab, os.path.join(tmp, "work0"))
        test_components(tab, recs, tmp)
    print(f"test_agent_combo: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
