#!/usr/bin/env python
"""
Проверки доверия к модельным целям (offline_rl.py --mb-trust; mb_targets.trust_weights, trust_clusters,
trust_folds, trust_gate) и модельных целей по нескольким горизонтам (--mb-hs; mb_targets.lookahead,
lookahead_full, precompute с hs) без GPU, без сети и без обучения PINN, на синтетических цепочках:

  * по умолчанию цели прежние бит в бит: hs=[H] даёт ровно Q одного горизонта H (жадный и лучевой
    просмотр, с бюджетом и правилом остановки), precompute без hs — без новых полей сводки;
  * несколько горизонтов: среднее Q_h, посчитанное за один проход, совпадает со средним отдельных
    расчётов с h = 1, 2, 3, 5 (жадный) и 1, 2, 4 (луч); отказ при max(hs) != h; сводка горизонтов;
  * доверие: веса в [0, 1]; перекрёстная подгонка действительно исключает кластер цепочки строки —
    «запоминающий» ансамбль (точный ответ на виденный переход) ни разу не попадает в оцениваемую строку,
    а без исключения попал бы во все; цепочки суррогатов фолдов и цепочки оцениваемых ими строк не
    пересекаются; склейка групп с общим переходом; строки вне таблицы — полный ансамбль; число фолдов;
  * ворота: значения trust_gate, режимы state / gate / both, верный и неверный знак;
  * лосс train_variant: множитель 1 — тот же результат бит в бит, множитель 0 — добавка ноль;
  * offline_rl.main(): по умолчанию строка без новых полей; с --mb-trust both --mb-hs 1,2,3 — поля в
    строке и meta чекпоинта (настоящие ансамбли HGB фолдов на крошечной таблице), отказы разбора;
  * онлайновое дообучение: online_aux.Aux с mb_hs — те же цели, что precompute; online_train_env.py
    --mb-trust — отказ разбора.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_mb_trust.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ["HF_HUB_OFFLINE"] = "1"
for _k in ("HF_TOKEN_WRITE", "HF_TOKEN"):      # проверка ничего не выгружает
    os.environ.pop(_k, None)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "track3"), ROOT):
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import mb_targets as MB  # noqa: E402
import agent_combo as AC  # noqa: E402
import online_aux as OA  # noqa: E402
import test_map_agent as T  # noqa: E402
import test_shield_teacher as TS  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a, **k: None  # noqa: E731
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"


def must_fail(fn, exc, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except exc:
        return
    raise AssertionError(what)


class WigEns:
    """Подставной ансамбль, у которого лучшее действие меняется от шага к шагу: изменение log10 ошибки —
    функция признаков с волной, член m смещён на 0.02*m."""
    members = 3

    def predict_ln(self, X, l, member=None):
        d = -(0.05 + 0.01 * X[:, 0]) + 0.04 * np.sin(3.0 * X.sum(1))
        out = np.stack([np.clip(l + d + 0.02 * m, S.L_LO, S.L_HI) for m in range(self.members)])
        return out if member is None else out[member]


class ConstEns:
    """Изменение log10 ошибки d у любого действия; считает вызовы."""
    members = 2

    def __init__(self, d):
        self.d, self.calls = float(d), 0

    def predict_ln(self, X, l, member=None):
        self.calls += len(l)
        out = np.stack([np.clip(l + self.d, S.L_LO, S.L_HI)] * self.members)
        return out if member is None else out[member]


def xkey(x):
    return np.asarray(x, np.float64).tobytes()


class MemoEns:
    """«Запоминающий» суррогат: на переход, который был в его таблице (тот же вход ансамбля), отвечает
    точным записанным исходом, на остальные — l - 0.4. Помнит цепочки задачи-цели своей таблицы и
    входы, которые его спросили."""
    members = 2

    def __init__(self, tab, pi):
        r = np.arange(len(tab["A"]))
        X = AC.fit_features(tab, r)
        self.memo = {xkey(X[i]): float(tab["LN"][i]) for i in r}
        self.chains = set(tab["CH"][tab["PDE"] == pi].tolist())
        self.asked, self.hits = [], 0

    def predict_ln(self, X, l, member=None):
        out = np.empty(len(l))
        for i in range(len(l)):
            k = xkey(X[i])
            self.asked.append(k)
            if k in self.memo:
                self.hits += 1
                out[i] = self.memo[k]
            else:
                out[i] = l[i] - 0.4
        out = np.stack([out] * self.members)
        return out if member is None else out[member]


def trust_setup():
    bufs = TS.make_buffers()
    tab = AC.with_epa(S.build_table(episodes=bufs, verbose=False))      # две задачи, pb2d — цель
    pi = tab["NAMES"].index(PDE)
    data = O.episodes_to_arrays_chains(bufs[SUB], reward_form="delta", verbose=False)
    return bufs, tab, pi, data


# --------------------------------------------------------------------------------------------
def test_horizons():
    tab, _ = T.fake_model()
    ens = WigEns()
    d = T.make_data(seed=2, n_chains=5, length=5)
    n = len(d["A"])
    st, valid, first = MB.chain_states(d, tab, np.array([0]), np.ones(1))
    zf, sc = first.copy(), np.ones(n)
    for kw in (dict(), dict(budget=7000), dict(budget=7000, kmax=3)):
        # один горизонт через hs — ровно прежний расчёт
        q3, u3 = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, **kw)
        q3h, u3h = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, hs=[3], **kw)
        assert np.array_equal(q3, q3h) and u3 == u3h, kw
        # несколько горизонтов за один проход = среднее отдельных расчётов
        sep = {h: MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, h, 0.5, chunk=7, **kw)[0]
               for h in (1, 2, 3, 5)}
        parts = {}
        qm, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 5, 0.5, hs=[5, 1, 3, 2], parts=parts,
                             chunk=7, **kw)
        assert np.allclose(qm, np.mean([sep[h] for h in (1, 2, 3, 5)], 0), atol=1e-12, rtol=0), kw
        assert sorted(parts) == [1, 2, 3, 5] and all(np.array_equal(parts[h], sep[h]) for h in parts), kw
        assert not np.allclose(sep[1], sep[5]), "на этих данных горизонты различаются"
        # луч
        f3, _ = MB.lookahead_full(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, beam=3, **kw)
        f3h, _ = MB.lookahead_full(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, beam=3, hs=[3], **kw)
        assert np.array_equal(f3, f3h), kw
        fsep = {h: MB.lookahead_full(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, h, 0.5, beam=2, chunk=6, **kw)[0]
                for h in (1, 2, 4)}
        fm, _ = MB.lookahead_full(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 4, 0.5, beam=2, chunk=6, hs=[1, 2, 4],
                                  **kw)
        assert np.allclose(fm, np.mean([fsep[h] for h in (1, 2, 4)], 0), atol=1e-12, rtol=0), kw
    must_fail(lambda: MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, hs=[1, 5]), ValueError,
              "наибольший горизонт не равен глубине — отказ")
    must_fail(lambda: MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, hs=[0, 3]), ValueError,
              "горизонт 0 — отказ")
    # precompute: hs=[3] — бит в бит, без новых полей; hs=[1,2,3,5] — среднее и сводка
    mask = d["EP"] < 4
    args = (d, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9)
    Q0, W0, i0 = MB.precompute(*args, h=3, lam=0.5, budget=7000, model=(tab, ens), log=QUIET)
    Q1, W1, i1 = MB.precompute(*args, h=3, lam=0.5, budget=7000, model=(tab, ens), log=QUIET, hs=[3])
    assert np.array_equal(Q0, Q1) and np.array_equal(W0, W1) and "horizons" not in i0 and i1["horizons"] == "3"
    Qm, Wm, im = MB.precompute(*args, h=99, lam=0.5, budget=7000, model=(tab, ens), log=QUIET, hs=[1, 2, 3, 5])
    sep = [MB.precompute(*args, h=h, lam=0.5, budget=7000, model=(tab, ens), log=QUIET)[0] for h in (1, 2, 3, 5)]
    assert np.array_equal(Wm, W0) and np.allclose(Qm, np.mean(sep, 0), atol=1e-6)
    assert im["horizons"] == "1,2,3,5" and set(im["hs_q_best"]) == {"1", "2", "3", "5"}
    assert im["hs_q_best"]["1"] <= im["hs_q_best"]["5"] and 0 <= min(im["hs_argmax_agree"].values()) <= 1
    print("несколько горизонтов: hs=[H] бит в бит равен одному горизонту (жадный и луч, бюджет, K_max), среднее за "
          "один проход = среднее отдельных расчётов, сводка, отказы: OK")


def test_clusters():
    # группы цепочек задачи 0: у 10 и 30 общий переход (ошибка после отличается на 3e-5 порядка — в допуске),
    # у 20 и 40 тот же шаг и действие, но исход другой; задача 1 — чужая (её переход такой же, как у 10)
    tab = dict(PDE=np.array([0, 0, 0, 0, 0, 1, 0]), GRP=np.array([10, 10, 20, 30, 30, 10, 40]),
               STEP=np.array([0, 1, 0, 0, 1, 0, 0]), SPENT=np.array([0, 100, 0, 0, 100, 0, 0]),
               PA=np.array([-1, 3, -1, -1, 3, -1, -1]), A=np.array([3, 4, 3, 3, 5, 3, 3]),
               L=np.array([0.0, -0.2, 0.0, 0.0, -0.2, 0.0, 0.0]),
               LN=np.array([-0.2, -0.3, -0.1, -0.20003, -0.5, -0.2, -0.09988]))
    clu, qc, mt = MB.trust_clusters(tab, 0)
    assert clu[5] == -1 and clu[0] == clu[1] == clu[3] == clu[4] and len({clu[0], clu[2], clu[6]}) == 3, clu
    assert len(qc) == 0 and mt == []
    # запрос: строка, совпавшая с переходами групп 20 и 40 (между собой они дальше допуска), склеивает их;
    # строка без совпадения — -1
    q = ([(0, 0, -1, 3), (0, 0, -1, 3), (1, 100, 3, 4)], np.array([0.0, 0.0, -0.2]), np.array([-0.09995, -0.7, -0.3]))
    clu, qc, mt = MB.trust_clusters(tab, 0, q)
    assert sorted(mt[0]) == [2, 6] and mt[1] == [] and mt[2] == [1], mt
    assert clu[2] == clu[6] != clu[0] and qc[0] == clu[2] and qc[1] == -1 and qc[2] == clu[0], (clu, qc)
    fo, K = MB.trust_folds(clu[clu >= 0], 5, seed=0)
    assert K == 2 and sorted(fo.values()) == [0, 1]
    fo, K = MB.trust_folds(np.arange(7), 3, seed=1)
    assert K == 3 and sorted(np.bincount(list(fo.values())).tolist()) == [2, 2, 3]
    assert MB.trust_folds(np.arange(7), 0)[1] == 7 and MB.trust_folds(np.arange(4), 9)[1] == 4
    print("кластеры цепочек: склейка групп с совпадающим переходом (в допуске) и через строку тренера, чужая задача "
          "вне кластеров, строка без совпадения, фолды: OK")


def test_trust_leak():
    _, tab, pi, data = trust_setup()
    n = len(data["A"])
    rows = np.ones(n, dtype=bool)
    fitted = []

    def fit(t):
        m = MemoEns(t, pi)
        fitted.append(m)
        return m
    full = MemoEns(tab, pi)
    # контроль: суррогат, видевший всё, отвечает записанным исходом (у первых шагов с одинаковым
    # действием вход совпадает, а исход у цепочек свой — там ответ чужой цепочки)
    Tl, il = MB.trust_weights(data, rows, [PDE], np.ones(1), 7000, (tab, full), mode="state", tau=0.3,
                              fit=lambda t: full, log=QUIET)
    exact_full = int((Tl == 1).sum())
    assert full.hits == n and exact_full >= n - 2 and il["trust_err_med"] == 0.0, "запоминающий суррогат точен"
    for folds in (2, 0):
        fitted.clear()
        tr = {}
        Tw, info = MB.trust_weights(data, rows, [PDE], np.ones(1), 7000, (tab, full), mode="state", tau=0.3,
                                    folds=folds, fit=fit, log=QUIET, trace=tr)
        n_chains = len(np.unique(tab["CH"][tab["PDE"] == pi]))
        assert info["trust_k"] == (2 if folds else n_chains) == info["trust_fits"] == len(fitted), info
        assert info["trust_unmatched"] == 0 and info["trust_rows"] == n
        assert ((Tw >= 0) & (Tw <= 1)).all() and Tw.dtype == np.float32
        assert (Tw < 1).all(), "суррогат фолда ответил записанным исходом оцениваемого перехода (утечка)"
        assert sum(len(m.asked) for m in fitted) == n
        # цепочка каждой оцениваемой строки (все совпавшие с ней строки таблицы) не входит в таблицу
        # суррогата её фолда
        for j, f in enumerate(tr["fold"]):
            chs = set(tab["CH"][tr["matches"][j]].tolist())
            assert f >= 0 and len(chs) == 1 and not (chs & tr["models"][int(f)].chains), (j, f, chs)
        assert all(len(m.chains) < n_chains for m in fitted)
    # расхождение ошибок в последних знаках (1e-6 относительно) — строки всё равно находятся в таблице
    dj = dict(data, ERR=(data["ERR"].astype(np.float64) * (1 + 1e-6)).astype(np.float32))
    _, ij = MB.trust_weights(dj, rows, [PDE], np.ones(1), 7000, (tab, full), mode="state", folds=2, fit=fit,
                             log=QUIET)
    assert ij["trust_unmatched"] == 0, ij
    # строки вне таблицы (цепочку убрали из таблицы) предсказывает полный ансамбль, остальные — фолды
    c0 = int(tab["CH"][tab["PDE"] == pi][0])
    tab2 = AC.subset(tab, ~((tab["PDE"] == pi) & (tab["CH"] == c0)))
    ens = ConstEns(-0.15)
    fitted.clear()
    T2, i2 = MB.trust_weights(data, rows, [PDE], np.ones(1), 7000, (tab2, ens), mode="state", folds=5, fit=fit,
                              log=QUIET)
    lost = int((tab["CH"] == c0).sum())
    assert i2["trust_unmatched"] == lost == ens.calls and sum(m.hits for m in fitted) == 0
    # маска строк и отказы
    sub = np.zeros(n, dtype=bool)
    sub[3:7] = True
    T3, i3 = MB.trust_weights(data, sub, [PDE], np.ones(1), 7000, (tab, full), mode="state", fit=fit, log=QUIET)
    assert i3["trust_rows"] == 4 and (T3[~sub] == 1).all() and (T3[sub] < 1).all()
    must_fail(lambda: MB.trust_weights(data, rows, [PDE, "x"], np.ones(2), 0, (tab, full), fit=fit, log=QUIET),
              ValueError, "смесь УрЧП — отказ")
    must_fail(lambda: MB.trust_weights(data, rows, [PDE], np.ones(1), 0, (tab, full), mode="none", log=QUIET),
              ValueError, "режим none — отказ")
    print(f"доверие без утечки: запоминающий суррогат точен на виденном ({exact_full} из {n}), а суррогаты фолдов (2 фолда и "
          "по кластеру на фолд) не попали ни в одну оцениваемую строку, цепочки не пересекаются, веса в [0, 1], строки "
          "вне таблицы — полный ансамбль: OK")


def test_gate():
    assert MB.trust_gate(0.9, 0.7) == 1.0 and MB.trust_gate(0.7, 0.7) == 1.0
    assert abs(MB.trust_gate(0.6, 0.7) - 0.5) < 1e-12 and MB.trust_gate(0.5, 0.7) == 0.0
    assert MB.trust_gate(0.2, 0.7) == 0.0 and abs(MB.trust_gate(0.75, 1.0) - 0.5) < 1e-12
    _, tab, pi, data = trust_setup()
    n = len(data["A"])
    rows = np.ones(n, dtype=bool)
    ln = np.log10(data["ERR"].astype(np.float64))
    st, _, _ = MB.chain_states(data, tab, np.array([pi]), np.ones(1))
    dl = ln - st["l"]
    good, bad = ConstEns(-0.15), ConstEns(+0.3)
    acc_good = float(np.mean(MB._sgn(np.full(n, -0.15)) == MB._sgn(dl)))
    acc_bad = float(np.mean(MB._sgn(np.full(n, 0.3)) == MB._sgn(dl)))
    assert acc_good > 0.7 > 0.5 > acc_bad, (acc_good, acc_bad)
    out = {}
    for name, ens in (("good", good), ("bad", bad)):
        for mode in ("state", "gate", "both"):
            Tm, im = MB.trust_weights(data, rows, [PDE], np.ones(1), 7000, (tab, ens), mode=mode, tau=0.5,
                                      gate_min=0.7, fit=lambda t, e=ens: e, log=QUIET)
            out[name, mode] = (Tm, im)
            assert ((Tm >= 0) & (Tm <= 1)).all() and im["trust_mode"] == mode
    w_good = np.exp(-np.abs(-0.15 - dl) / 0.5)
    Tg, ig = out["good", "gate"]
    assert abs(ig["trust_sign_acc"] - round(acc_good, 4)) < 1e-9 and ig["trust_gate"] == 1.0 and (Tg == 1).all()
    assert np.allclose(out["good", "state"][0], w_good, atol=1e-6)
    assert np.allclose(out["good", "both"][0], w_good, atol=1e-6)
    Tb, ib = out["bad", "gate"]
    assert ib["trust_gate"] == 0.0 and (Tb == 0).all() and not out["bad", "both"][0].any()
    assert out["bad", "state"][0].min() > 0, "режим state ворота не применяет"
    print(f"ворота: доля верных знаков {acc_good:.2f} — множитель 1, {acc_bad:.2f} — 0; режимы state, gate, both: OK")


def test_loss():
    data = T.make_data(seed=1, n_chains=6, length=3)
    mask = data["EP"] < 5
    tab, ens = T.fake_model()
    QMB, MBW, _ = MB.precompute(data, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9, h=2, lam=0.5,
                                model=(tab, ens), log=QUIET)
    d2 = dict(data, QMB=QMB, MBW=MBW)
    net0, _ = T.fit("cnn_dqn", d2, mask, mb_aux=0.5)
    net1, _ = T.fit("cnn_dqn", dict(d2, MBT=np.ones(len(data["A"]), np.float32)), mask, mb_aux=0.5)
    sd0, sd1 = net0.model.state_dict(), net1.model.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0) and net0.extra["mb_loss"] == net1.extra["mb_loss"] > 0
    netz, _ = T.fit("cnn_dqn", dict(d2, MBT=np.zeros(len(data["A"]), np.float32)), mask, mb_aux=0.5)
    assert netz.extra["mb_loss"] == 0.0
    print("лосс: множитель 1 — веса сети бит в бит как без доверия, множитель 0 — добавка ноль: OK")


def test_offline_main(bufs, tmp):
    import killswitch
    root = os.path.join(tmp, "bufs", SUB)
    os.makedirs(root, exist_ok=True)
    for k, ep in enumerate(bufs[SUB]):
        torch.save(ep, os.path.join(root, f"f{k}.pt"))
    work = os.path.join(tmp, "work")
    e0 = float(O.pde_meta()[PDE]["init_err"])
    mtag = f"tmbt{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--mb-aux", "0.5", "--mb-lam", "0.5",
            "--mb-members", "2", "--mb-work", work, "--seeds", "1", "--model-tag", mtag,
            "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu", "--no-verify"]
    tab = AC.with_epa(S.build_table(episodes=bufs, verbose=False))

    def fake_build(names, subdirs, table, members, work_, quick, log):
        return tab, WigEns(), dict(surr_rows=len(tab["A"]), surr_sig="fake", surr_members=3,
                                   surr_file="/x/combo_surr_fake.pkl")

    def run(av):
        buf = io.StringIO()
        sys.argv = av
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            O.main()
        return buf.getvalue()

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        rows = []
        for extra in ([], ["--mb-trust", "both", "--mb-hs", "1,2,3", "--mb-trust-folds", "2"]):
            text = run(argv + extra)
            rows.append([json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0])
        r0, r1 = rows
        new = ("mb_trust", "mb_hs", "mb_horizons", "mb_trust_mode", "mb_trust_gate", "mb_trust_tau", "mb_hs_q_best")
        assert not any(k in r0 for k in new) and r0["mb_h"] == 3, [k for k in new if k in r0]
        assert r1["mb_trust"] == "both" and r1["mb_hs"] == "1,2,3" and r1["mb_h"] == 3 and r1["mb_horizons"] == "1,2,3"
        assert r1["mb_trust_tau"] == 0.3 and r1["mb_trust_gate_min"] == 0.7 and r1["mb_trust_folds"] == 2
        assert r1["mb_trust_k"] == 2 == r1["mb_trust_fits"] and r1["mb_trust_unmatched"] == 0
        assert 0 <= r1["mb_trust_w_min"] <= r1["mb_trust_w_mean"] <= 1 and 0 <= r1["mb_trust_gate"] <= 1
        assert r1["mb_trust_rows"] == r1["mb_rows"] > 0
        cv = os.path.join(work, "trust_cv")
        assert len([f for f in os.listdir(cv) if f.startswith("combo_surr_")]) == 2, "два настоящих ансамбля фолдов"
        import online_eval_env as E
        meta = E.load_agent(out)[0]._meta
        assert meta["mb_trust"] == "both" and meta["mb"]["trust_gate"] == r1["mb_trust_gate"]
        assert meta["mb_hs"] == "1,2,3" and meta["mb"]["horizons"] == "1,2,3"
        for bad, what in ((argv + ["--mb-hs", "0,2"], "горизонт 0"), (argv + ["--mb-hs", "a,b"], "не числа"),
                          (argv + ["--mb-h", "2", "--mb-hs", "1,3"], "--mb-h не равен наибольшему"),
                          (argv + ["--mb-trust", "both", "--mb-trust-gate-min", "0.4"], "ворота ниже 0.5"),
                          (argv + ["--mb-trust", "both", "--mb-trust-tau", "0"], "tau 0"),
                          ([a for a in argv if a not in ("--mb-aux", "0.5")] + ["--mb-lam", "0.5", "--mb-trust", "gate"],
                           "без --mb-aux"),
                          ([a for a in argv if a not in ("--mb-aux", "0.5")] + ["--mb-lam", "0.5", "--mb-hs", "1,3"],
                           "--mb-hs без --mb-aux")):
            must_fail(lambda b=bad: run(b), SystemExit, f"недопустимое сочетание флагов: {what}")
        r2 = [json.loads(ln) for ln in run(argv + ["--mb-h", "3", "--mb-hs", "1,3"]).splitlines()
              if ln.startswith('{"variant"')][0]
        assert r2["mb_hs"] == "1,3" and r2["mb_h"] == 3, "--mb-h, равный наибольшему горизонту, допустим"
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("offline_rl.main(): по умолчанию строка без новых полей; --mb-trust both --mb-hs 1,2,3 — поля в строке и meta, "
          "ансамбли фолдов в <mb-work>/trust_cv, отказы разбора: OK")


def test_online(tmp):
    import test_online_finetune as TF
    tab, ens = T.fake_model()
    d = T.make_data(seed=3, n_chains=5, length=4)
    mask = d["EP"] < 4
    aux = OA.Aux(PDE, w_mb=0.5, reward_form="delta", init_err=0.811, gamma=0.99, budget=7000, mb_h=3, mb_lam=0.5,
                 log=QUIET, mb_hs=[1, 2, 3])
    off = aux.offline(d, mask, model=(tab, ens))
    QMB, _, _ = MB.precompute(d, mask, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, h=3, lam=0.5,
                              budget=7000, model=(tab, ens), log=QUIET, hs=[1, 2, 3])
    assert np.array_equal(off["QMB"], QMB) and aux.meta()["mb_hs"] == "1,2,3" and aux.row_fields()["mb_hs"] == "1,2,3"
    aux0 = OA.Aux(PDE, w_mb=0.5, reward_form="delta", init_err=0.811, gamma=0.99, budget=7000, log=QUIET)
    assert "mb_hs" not in aux0.meta() and "mb_hs" not in aux0.row_fields()
    cwd = os.getcwd()
    try:
        TF.must_exit(lambda: TF.run_train(tmp, TF.FT + ["--mb-aux", "0.5", "--mb-trust", "both"]), "2",
                     "онлайн: --mb-trust — отказ")
        TF.must_exit(lambda: TF.run_train(tmp, TF.FT + ["--mb-hs", "1,3"]), "2", "онлайн: --mb-hs без --mb-aux")
    finally:
        os.chdir(cwd)
    print("онлайн: Aux с несколькими горизонтами — те же цели, что precompute; --mb-trust — отказ разбора: OK")


def main():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="mbtrust_")
    try:
        test_horizons()
        test_clusters()
        test_trust_leak()
        test_gate()
        test_loss()
        test_offline_main(TS.make_buffers(), tmp)
        test_online(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"все проверки доверия и горизонтов модельных целей прошли за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
