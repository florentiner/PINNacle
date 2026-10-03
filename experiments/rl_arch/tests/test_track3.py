#!/usr/bin/env python
"""
Проверки кода трека 3 (сеть и GPU не нужны):

  * вмешательства в состояние (apply_state_mode): blind, level, shape, shuffle;
  * скалярный контекст офлайновых переходов (chain_ctx): контекст s' равен контексту
    следующего шага и совпадает с кодированием онлайновой оценки;
  * дешёвое состояние из обучающих лоссов (loss_state);
  * награда dlog и масштаб ошибки в загрузчике цепочек;
  * выбор папок буфера для переноса (resolve_subdirs) и таблица задач.

    python experiments/rl_arch/tests/test_track3.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import offline_rl as O  # noqa: E402
from test_chain_loader import make_file  # noqa: E402

RNG = np.random.default_rng(1)


def test_state_modes():
    S = RNG.normal(size=(7, 9, 26, 26)).astype(np.float32)      # 4 карты + 5 каналов контекста
    for mode in ("blind", "level", "shape", "shuffle"):
        T = O.apply_state_mode(S, mode, seed=3)
        assert T.shape == S.shape and T is not S
        assert np.array_equal(T[:, 4:], S[:, 4:]), f"{mode}: контекст не должен меняться"
    assert np.array_equal(O.apply_state_mode(S, "full"), S)
    assert not O.apply_state_mode(S, "blind")[:, :4].any()
    L = O.apply_state_mode(S, "level")[:, :4]
    assert np.allclose(L, S[:, :4].mean(axis=(2, 3), keepdims=True), atol=1e-6)
    assert np.allclose(L.std(axis=(2, 3)), 0, atol=1e-6)
    H = O.apply_state_mode(S, "shape")[:, :4]
    assert np.allclose(H.mean(axis=(2, 3)), 0, atol=1e-4), "shape: уровень должен быть убран"
    assert np.allclose(H.std(axis=(2, 3)), 1, atol=1e-3)
    # shape не зависит от сдвига и масштаба карты
    H2 = O.apply_state_mode(S * 5.0 + 3.0, "shape")[:, :4]
    assert np.allclose(H, H2, atol=1e-3)
    # нулевая карта (старт цепочки) остаётся нулевой
    assert not O.apply_state_mode(np.zeros((4, 26, 26), np.float32), "shape").any()
    F = O.apply_state_mode(S, "shuffle", seed=5)[:, :4]
    assert np.allclose(np.sort(F.reshape(7, 4, -1), -1), np.sort(S[:, :4].reshape(7, 4, -1), -1))
    assert not np.array_equal(F, S[:, :4])
    # одиночное состояние (без оси пачки) тоже допустимо
    one = O.apply_state_mode(S[0], "level")
    assert one.shape == S[0].shape
    try:
        O.apply_state_mode(S, "loss")
        raise AssertionError("режим loss для массивов карт должен быть ошибкой")
    except ValueError:
        pass
    # нормировка по задаче: статистики из таблицы, нулевая стартовая карта остаётся нулевой
    row = O.pde_meta()["ns2d_liddriven"]
    if "map_mean" in row:
        T = O.apply_state_mode(S, "tasknorm", pde="ns2d_liddriven")
        mu, sd = np.array(row["map_mean"])[:, None, None], np.array(row["map_std"])[:, None, None]
        assert np.allclose(T[:, :3], (S[:, :3] - mu) / sd, atol=1e-5)
        assert np.array_equal(T[:, 3:], S[:, 3:])
        assert not O.apply_state_mode(np.zeros((4, 26, 26), np.float32), "tasknorm", pde="ns2d_liddriven").any()
        try:
            O.apply_state_mode(S, "tasknorm")
            raise AssertionError("tasknorm без имени УрЧП должен быть ошибкой")
        except ValueError:
            pass
    print("вмешательства в состояние: OK")


def test_ctx():
    eps = [make_file(False, [([0.4, 0.2, 0.1], -1), ([0.3, 0.008], 1)]),
           make_file(True, [([0.4, 0.3, 0.05], -1)])]
    d = O.episodes_to_arrays(eps, chain_fix=True, reward_form="delta", init_err=0.5, verbose=False)
    for with_err in (False, True):
        c, c2 = O.chain_ctx(d, k_max=10, budget=7000, with_err=with_err)
        assert c.shape == (len(d["A"]), O.SCALAR_CH)
        for i in range(len(d["A"]) - 1):
            if d["EP"][i] == d["EP"][i + 1]:
                assert np.allclose(c2[i], c[i + 1]), "контекст s' должен равняться контексту следующего шага"
        first = np.where(d["FIRST"] > 0)[0]
        assert np.allclose(c[first, 0], 0) and np.allclose(c[first, 1], 0)
        assert np.allclose(c[first, 2], -1) and np.allclose(c[first, 3], -1)
        if not with_err:
            assert not c[:, 4].any() and not c2[:, 4].any()
    # то же кодирование, что у онлайновой оценки
    import online_eval_env as E
    c, c2 = O.chain_ctx(d, k_max=10, budget=7000, with_err=True)
    i = 1                                    # второй шаг первой цепочки
    st = E.add_scalar_ctx(np.zeros((4, 26, 26), np.float32), int(d["STEP"][i]), 10,
                          int(d["SPENT"][i]), 7000, int(d["A"][i - 1]), float(d["ERR"][i - 1]))
    assert np.allclose(st[4:, 0, 0], c[i], atol=1e-6), (st[4:, 0, 0], c[i])
    S9 = O.attach_ctx(d["S"], c)
    assert S9.shape[1] == 9 and np.allclose(S9[:, 4:, 3, 7], c)
    print("скалярный контекст: OK")


def test_loss_state():
    s = O.loss_state(1e-2, 4e-3, 6e-3, prev_total=1.0)
    assert s.shape == (4, 26, 26) and s.dtype == np.float32
    assert np.allclose(s.std(axis=(1, 2)), 0)
    assert abs(s[0, 0, 0] - (np.log10(1e-2) / 6 + 1 / 3)) < 1e-6
    assert abs(s[3, 0, 0] - 1.0) < 1e-6          # падение на два порядка обрезано до 1
    assert O.loss_state(1.0, 0.5, 0.5)[3, 0, 0] == 0.0
    assert O.loss_state(1e9, 1e9, 1e9)[0, 0, 0] == 1.0 and O.loss_state(0.0, 0.0, 0.0)[0, 0, 0] == -1.0
    print("состояние из обучающих лоссов: OK")


def test_dlog_and_scale():
    eps = [make_file(False, [([0.4, 0.2, 0.1], -1), ([0.3, 0.008], 1)])]
    d = O.episodes_to_arrays(eps, chain_fix=True, reward_form="dlog", init_err=0.5, verbose=False)
    # сумма наград цепочки = log10(E0 / E_конец)
    for ep, e_end in ((0, 0.1), (1, 0.008)):
        assert abs(d["R"][d["EP"] == ep].sum() - np.log10(0.5 / e_end)) < 1e-5
    # масштаб ошибки: награда dlog от него не зависит, если E0 задан в тех же единицах
    d2 = O.episodes_to_arrays(eps, chain_fix=True, reward_form="dlog", init_err=1.0,
                              err_scale=0.5, verbose=False)
    assert np.allclose(d["R"], d2["R"], atol=1e-5)
    assert np.allclose(d2["ERR"], d["ERR"] / 0.5, atol=1e-6)
    # разность ошибок в долях E0: сумма наград = 1 - E_конец / E0
    d3 = O.episodes_to_arrays(eps, chain_fix=True, reward_form="delta", init_err=1.0,
                              err_scale=0.5, verbose=False)
    assert abs(d3["R"][d3["EP"] == 0].sum() - (1 - 0.1 / 0.5)) < 1e-5
    # без E0 первый шаг даёт ноль
    d4 = O.episodes_to_arrays(eps, chain_fix=True, reward_form="dlog", verbose=False)
    assert d4["R"][0] == 0.0
    print("награда dlog и масштаб ошибки: OK")


def test_subdirs_and_meta():
    assert len(O.ALL_SUBDIRS) == 22
    subs = O.resolve_subdirs("solvable", "ns2d_liddriven")
    assert "ns2d_liddriven" not in subs and len(subs) == 22 - len(O.UNSOLVED_SUBDIRS) - 1
    assert O.resolve_subdirs("burgers1d, wave1d") == ["burgers1d", "wave1d"]
    try:
        O.resolve_subdirs("нет_такой")
        raise AssertionError("неизвестная папка должна быть ошибкой")
    except ValueError:
        pass
    meta = O.pde_meta()
    missing = [sd for sd in O.ALL_SUBDIRS
               if not any(r.get("subdir") == sd for r in meta.values())]
    assert not missing, f"в track3/pde_meta.json нет задач для папок: {missing}"
    for sd in O.ALL_SUBDIRS:
        name = O.subdir_to_pde(sd)
        v = O.pde_desc(name)
        assert len(v) == O.PDE_DESC_CH and all(0.0 <= x <= 1.0 for x in v)
        assert meta[name]["init_err"] > 0
    print(f"папки буфера и таблица задач ({len(meta)} УрЧП): OK")


def test_multi_local():
    root = os.environ.get("RL_BUFFER_DIR")
    have = [sd for sd in O.ALL_SUBDIRS if root and os.path.isdir(os.path.join(root, sd))]
    if len(have) < 2:
        print("смесь УрЧП: пропущено (нужен RL_BUFFER_DIR с двумя и более папками)")
        return
    d = O.load_multi(have[:3], reward_form="dlog", err_norm="init", budget=7000, max_files=2,
                     verbose=False)
    assert len(d["PDE_NAMES"]) == len(have[:3])
    assert d["S"].shape[1:] == (4, 26, 26) and len(d["PDE"]) == len(d["A"])
    assert (np.diff(d["EP"]) >= 0).all(), "номера цепочек должны идти по возрастанию"
    for k in range(len(d["PDE_NAMES"])):
        eps_k = np.unique(d["EP"][d["PDE"] == k])
        other = np.unique(d["EP"][d["PDE"] != k])
        assert not np.intersect1d(eps_k, other).size, "цепочки разных УрЧП не должны пересекаться"
    assert np.isfinite(d["R"]).all() and np.abs(d["R"]).max() <= 3.0
    c, c2 = O.chain_ctx(d, k_max=10, budget=7000, with_err=False)
    desc = np.array([O.pde_desc(n) for n in d["PDE_NAMES"]], dtype=np.float32)[d["PDE"]]
    S = O.attach_ctx(d["S"], np.concatenate([c, desc], 1))
    assert S.shape[1] == 4 + O.SCALAR_CH + O.PDE_DESC_CH
    print(f"смесь УрЧП ({', '.join(d['PDE_NAMES'])}; переходов {len(d['A'])}): OK")


def test_keep_consistent():
    # две цепочки: Adam, Adam, LBFGS, LBFGS, PSO, PSO, Adam | LBFGS, LBFGS; хвост обрезанной цепочки
    A = np.array([0, 3, 9, 12, 18, 21, 5, 9, 10, 9, 9])
    EP = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    STEP = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3])
    ok = O.keep_consistent(A, EP, STEP)
    #                 первый  тот же  смена  тот же  PSO   PSO   смена  первый тот же  нет предш. тот же
    want = np.array([True, False, True, False, True, True, True, True, False, False, False])
    assert (ok == want).all(), ok
    assert O.keep_consistent(np.array([4]), np.array([0]), np.array([0])).tolist() == [True]
    # на настоящих цепочках загрузчика: один оптимизатор на всю цепочку — согласован только первый шаг
    d = O.episodes_to_arrays([make_file(False, [([0.4, 0.2, 0.1], -1), ([0.3, 0.008], 1)])],
                             chain_fix=True, reward_form="delta", init_err=0.5, verbose=False)
    ok = O.keep_consistent(d["A"], d["EP"], d["STEP"])
    assert (ok == (d["STEP"] == 0)).all()
    print("переходы, согласованные со средой без сбросов оптимизатора: OK")


if __name__ == "__main__":
    test_state_modes()
    test_keep_consistent()
    test_ctx()
    test_loss_state()
    test_dlog_and_scale()
    test_subdirs_and_meta()
    test_multi_local()
    print("ВСЕ ПРОВЕРКИ ТРЕКА 3 ПРОЙДЕНЫ")
