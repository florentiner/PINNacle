#!/usr/bin/env python
"""
Проверки малого числа журналов источников для модели среды (волна 65). Без GPU и сети, на синтетических таблицах:
пять уравнений-источников с разной динамикой (изменение log10 ошибки за действие зависит от семейства
оптимизатора), цель повторяет динамику одного из них.

  * agent_combo.cap_source_chains: у источников по N цепочек, цель не трогается; выбор у задачи не зависит от
    состава остальных папок и совпадает с pick_source_chains (его берёт анализ); N = 0 — таблица как есть;
  * target_buffer.surrogate_table(src_chains=...) режет источники так же, как учитель;
  * surr_data_study: модель среды на источнике с той же динамикой предсказывает цель лучше, чем на чужом;
    жадный порядок начинается с него; порядок div — перестановка источников без повторов; точки кривой
    посчитаны, строки источников по уравнениям сходятся с n; --resume не пересчитывает точки и отказывает
    при других источниках; offline_rl.py отказывает --mb-src-chains без --target-buffer.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_surr_study.py
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HF_TOKEN_WRITE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))

import numpy as np  # noqa: E402

import agent_combo as AC  # noqa: E402
import surrogate as S  # noqa: E402
import surr_data_study as SD  # noqa: E402
import target_buffer as TB  # noqa: E402

SRC = ["burgers1d", "poisson2d_classic", "heat2d_complexgeometry", "wave1d", "grayscott"]
# изменение log10 ошибки за действие по семейству (Adam, L-BFGS, PSO) у каждого уравнения
DYN = {"burgers1d": (-0.3, -0.05, 0.1), "poisson2d_classic": (-0.05, -0.6, 0.05),
       "heat2d_complexgeometry": (-0.2, -0.2, -0.2), "wave1d": (0.05, 0.1, -0.4), "grayscott": (-0.1, 0.3, 0.0)}
TARGET = "poissonboltzmann2d"
TARGET_DYN = DYN["poisson2d_classic"]


def chains_rows(pi, dyn, n_chains, rng, ch0, file0, ctx_l0=0.0):
    cols = {k: [] for k in AC.ROWKEYS}
    ch = ch0
    for _ in range(n_chains):
        l, pa, spent, spf = ctx_l0, -1, 0, np.zeros(3)
        for i in range(int(rng.integers(4, 9))):
            a = int(rng.integers(27))
            ln = l + dyn[S.FAM[a]] * (1 + 0.3 * (S.EPOCHS[a] >= 1000)) + 0.02 * rng.standard_normal()
            for k, v in (("PDE", pi), ("FILE", file0), ("CH", ch), ("STEP", i), ("SPENT", spent), ("A", a),
                         ("PA", pa), ("L", l), ("LN", ln), ("LV", np.zeros(6)), ("LVN", np.zeros(6)),
                         ("SPF", spf.copy()), ("LAST", False), ("EPA", int(S.EPOCHS[a])), ("GRP", ch)):
                cols[k].append(v)
            spf[S.FAM[a]] += S.EPOCHS[a]
            l, pa, spent = ln, a, spent + int(S.EPOCHS[a])
        ch += 1
    return cols, ch


def make_base(n_chains=60, seed=0, names=SRC):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in AC.ROWKEYS}
    ch = 0
    for pi, name in enumerate(names):
        c, ch = chains_rows(pi, DYN[name], n_chains, rng, ch, pi)
        for k in cols:
            cols[k] += c[k]
    tab = {k: np.array(v) for k, v in cols.items()}
    tab["SPF"] = tab["SPF"].reshape(-1, 3).astype(float)
    tab["LV"] = tab["LV"].reshape(-1, 6).astype(float)
    tab["LVN"] = tab["LVN"].reshape(-1, 6).astype(float)
    tab["CTX"] = np.array([[0.2 * i, 0.3, 0.3, 0.25, 1, 0, 0.0] for i in range(len(names))])
    tab["NAMES"], tab["SUBDIRS"] = list(names), list(names)
    return tab


def with_target(base, n_chains=20, seed=1):
    """Таблица цели как у table_with: строки источников и цепочки цели в конце."""
    rng = np.random.default_rng(seed)
    tab = dict(base)
    tab["NAMES"] = list(base["NAMES"]) + [TARGET]
    tab["SUBDIRS"] = list(base["SUBDIRS"]) + ["poisson_boltzmann_2d"]
    tab["CTX"] = np.vstack([base["CTX"], [[0.5, 0.3, 0.3, 0.25, 1, 0, 0.0]]])
    pi = len(base["NAMES"])
    c, _ = chains_rows(pi, TARGET_DYN, n_chains, rng, int(base["CH"].max()) + 1, int(base["FILE"].max()) + 1)
    for k in AC.ROWKEYS:
        add = np.asarray(c[k]).astype(base[k].dtype)
        tab[k] = np.concatenate([base[k], add.reshape((len(add),) + base[k].shape[1:])])
    return tab, pi


def test_cap():
    base = make_base()
    t, info = AC.cap_source_chains(base, 7, 3, skip=("poisson2d_classic",))
    for pi, name in enumerate(base["NAMES"]):
        n = len(np.unique(t["CH"][t["PDE"] == pi]))
        assert n == (60 if name == "poisson2d_classic" else 7), (name, n)
    assert info == {nm: 7 for nm in SRC if nm != "poisson2d_classic"}
    same, inf0 = AC.cap_source_chains(base, 0, 3)
    assert same is base and inf0 == {}
    # выбор у задачи не зависит от остальных папок и совпадает с pick_source_chains
    other = make_base(names=["wave1d", "burgers1d"])
    t2, _ = AC.cap_source_chains(other, 7, 3)
    def positions(full, cut, pi):
        chs = np.unique(full["CH"][full["PDE"] == pi])
        return np.searchsorted(chs, np.unique(cut["CH"][cut["PDE"] == pi]))
    for name in ("wave1d", "burgers1d"):
        p1 = positions(base, t, SRC.index(name))
        p2 = positions(other, t2, ["wave1d", "burgers1d"].index(name))
        assert len(p1) == 7 and np.array_equal(p1, p2), (name, p1, p2)
        pi2 = ["wave1d", "burgers1d"].index(name)
        pick = AC.pick_source_chains(np.unique(other["CH"][other["PDE"] == pi2]), name, 7, 3)
        assert np.array_equal(np.unique(t2["CH"][t2["PDE"] == pi2]), pick)
    assert len(np.unique(AC.cap_source_chains(base, 500, 3)[0]["CH"])) == len(np.unique(base["CH"]))
    # surrogate_table режет источники так же
    saved = AC.get_table, TB.table_with
    try:
        AC.get_table = lambda ns, subs, log=print: make_base()
        TB.table_with = lambda tab, eps, name, meta=None: (tab, 0, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            st = TB.surrogate_table([], TARGET, "all", "", "w", src_chains=7, src_seed=3)
        assert all(len(np.unique(st["CH"][st["PDE"] == pi])) == 7 for pi in range(len(SRC)))
    finally:
        AC.get_table, TB.table_with = saved
    print("cap_source_chains и surrogate_table: OK")


def args(**kw):
    a = dict(k_max=3, sel_chains=20, sizes=[5, 20], seeds=[0, 1], hours=0.0)
    a.update(kw)
    return argparse.Namespace(**a)


def test_study():
    base = make_base()
    tab, pi = with_target(base)
    ev = SD.TargetEval(tab, pi, 4, 1, 0, True)
    assert ev.k == 4 and len(ev.chains) == 20
    good = ev.run(SD.source_rows(base, ["poisson2d_classic"], 30, 0))
    bad = ev.run(SD.source_rows(base, ["grayscott"], 30, 0))
    own = ev.run(np.zeros(0, dtype=np.int64))
    assert good["mae1"] < bad["mae1"], (good, bad)
    assert good["roll_mae"] < bad["roll_mae"] and good["n_chains"] == 20 and own["n_test"] == good["n_test"]
    saved = []
    st = SD.Study(base, {TARGET: ev}, list(SRC), args(), log=lambda *x: None, save=lambda r: saved.append(1))
    res = st.run()
    assert res["finished"] and res["greedy"][0] == "poisson2d_classic", res["greedy"]
    assert len(res["greedy"]) == 3 == len(set(res["greedy"])) and len(res["div"]) == 3 == len(set(res["div"]))
    assert set(res["div"]) <= set(SRC)
    for order in ("greedy", "div"):
        for k in (1, 2, 3):
            for n in (5, 20):
                for sd in (0, 1):
                    pt = res["points"][f"{order}|{k}|{n}|{sd}"]
                    assert pt["names"] == res[order][:k] and pt["src_rows"] == sum(pt["rows_by_pde"].values())
                    for nm in pt["names"]:
                        nch = len(np.unique(base["CH"][SD.source_rows(base, [nm], n, sd)]))
                        assert nch == n
    assert "all|0|0" in res["points"] and res["points"]["all|0|0"]["src_rows"] == len(base["A"])
    # resume: те же точки не пересчитываются
    n_pts = len(res["points"])
    calls = []
    run0 = SD.TargetEval.run
    SD.TargetEval.run = lambda self, rows: calls.append(1) or run0(self, rows)
    try:
        st2 = SD.Study(base, {TARGET: ev}, list(SRC), args(), log=lambda *x: None)
        st2.res.update({k: res[k] for k in ("points", "greedy", "div")})
        st2.run()
    finally:
        SD.TargetEval.run = run0
    assert not calls and len(st2.res["points"]) == n_pts
    print("модель среды по источникам, жадный и разнообразный порядок, кривая, продолжение: OK")


def test_main_resume_refusal():
    tmp = tempfile.mkdtemp(prefix="surr_study_")
    saved = AC.get_table, TB.load, TB.table_with
    try:
        AC.get_table = lambda ns, subs, log=print: make_base()
        TB.load = lambda spec, sub, log=print: ([], {})
        TB.table_with = lambda tab, eps, name, meta=None: with_target(tab) + (20,)
        argv = ["--tag", "t", "--target", f"{TARGET}=x", "--k-max", "2", "--sizes", "5", "--seeds", "0",
                "--folds", "3", "--quick", "--no-upload", "--out-dir", tmp, "--work", tmp]
        with contextlib.redirect_stdout(io.StringIO()):
            res = SD.main(argv)
        assert res["finished"] and os.path.exists(os.path.join(tmp, "surrstudy_t.json"))
        disk = json.load(open(os.path.join(tmp, "surrstudy_t.json")))
        assert disk["settings"]["sizes"] == [5] and set(disk["points"]) == set(res["points"])
        AC.get_table = lambda ns, subs, log=print: make_base(names=SRC[:4])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                SD.main(argv + ["--resume"])
        except SystemExit as e:
            assert "других источниках" in str(e)
        else:
            raise AssertionError("--resume на других источниках должен отказать")
    finally:
        AC.get_table, TB.load, TB.table_with = saved
        shutil.rmtree(tmp, ignore_errors=True)
    import offline_rl as O
    old = sys.argv
    sys.argv = ["offline_rl.py", "--variant", "convnext_dqn", "--subdir", "poisson_boltzmann_2d", "--mb-aux", "0.5",
                "--mb-src-chains", "5"]
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            O.main()
    except SystemExit:
        assert "--mb-src-chains" in err.getvalue(), err.getvalue()[-300:]
    else:
        raise AssertionError("--mb-src-chains без --target-buffer должен отказать")
    finally:
        sys.argv = old
    print("main, отказ --resume и --mb-src-chains без собранного буфера: OK")


def main():
    t0 = time.time()
    test_cap()
    test_study()
    test_main_resume_refusal()
    print(f"test_surr_study: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
