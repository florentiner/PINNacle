#!/usr/bin/env python
"""
Проверки правила остановки эпизода ОБУЧЕНИЯ (статья PINN-PELINE, формула (11): допуск по ошибке,
K_max стадий) без GPU, без сети и без обучения PINN, на синтетических цепочках:

  * offline_rl.stop_rule_cut: по умолчанию тот же объект; места обрезки по допуску (<=) и по K_max,
    терминальный конец, выброшенные переходы, пересчёт RTG и FIRST, остальные столбцы режутся так же;
    train_end / stop_hit;
  * модельные цели (mb_targets.lookahead, precompute): за допуском и за K-й стадией продолжения нет;
    по умолчанию цели те же бит в бит;
  * цели учителя (offline_rl.teacher_targets) под правилом: продолжение политики учителя в суррогате до
    остановки; на последней разрешённой стадии и при допуске после первого действия — ровно одношаговая
    модельная цель; без правила — прежние числа бит в бит; отказ без суррогата;
  * offline_rl.main(): флаги по умолчанию выключены и строка результата без новых полей; с правилом —
    обрезанные цепочки, поля в строке и meta чекпоинта, отказы разбора;
  * онлайновое дообучение (online_aux.Aux, online_train_env.main в заглушке среды): цели обеих половин
    с правилом равны целям offline_rl.py; офлайновая половина обрезана тем же правилом; онлайновые
    цепочки кончаются на допуске или на K-й стадии терминальным переходом (d = 1), конец по K_max не
    бутстрепится; отказы разбора;
  * experiments/rl_arch/epochs_to_threshold.py на синтетических строках оценки.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_stop_rule.py
"""
import contextlib
import importlib.util
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

torch.set_num_threads(1)
QUIET = lambda *a, **k: None  # noqa: E731
CPU = torch.device("cpu")
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"


def must_fail(fn, exc, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except exc:
        return
    raise AssertionError(what)


def chain_data(errs, seed=0):
    """Набор тренера (как make_data) с заданными ошибками после действий: errs — список цепочек."""
    d = T.make_data(seed=seed, n_chains=len(errs), length=len(errs[0]))
    d["ERR"] = np.concatenate([np.asarray(e, np.float32) for e in errs])
    d["X"] = np.arange(len(d["A"]))              # посторонний столбец той же длины режется вместе с набором
    d["STATS"] = dict(files=1)
    d["PDE_NAMES"] = [PDE]
    return d


# --------------------------------------------------------------------------------------------
def test_cut():
    errs = [[0.5, 0.25, 0.0625, 0.01, 0.005],    # допуск 0.0625 (точно в float32) ровно на 3-й стадии (<=)
            [0.6, 0.5, 0.4, 0.3, 0.2],           # допуск не достигнут
            [0.04, 0.03, 0.02, 0.01, 0.009]]     # допуск на 1-й стадии
    d = chain_data(errs)
    n = len(d["A"])
    same, info = O.stop_rule_cut(d, 0.0, 0)
    assert same is d and info == {}, "правило выключено — тот же объект"
    assert O.stop_hit(0.05, 0.05) and not O.stop_hit(0.0500001, 0.05) and not O.stop_hit(0.01, 0.0)
    assert not O.stop_hit(float("nan"), 0.1)
    assert O.train_end(0.05, 3, 0.05, 10) == "tolerance" and O.train_end(0.3, 10, 0.05, 10) == "kmax"
    assert O.train_end(0.3, 9, 0.05, 10) == "" and O.train_end(0.3, 99, 0.0, 0) == ""

    c, st = O.stop_rule_cut(d, 0.0625, 0, gamma=0.9)
    assert list(c["EP"]) == [0, 0, 0, 1, 1, 1, 1, 1, 2], c["EP"]
    assert list(c["D"]) == [0, 0, 1, 0, 0, 0, 0, 1, 1]
    assert st["chains_tol"] == 2 and st["chains_kmax"] == 0 and st["dropped"] == n - 9 and st["made_terminal"] == 2
    assert list(c["X"]) == [0, 1, 2, 5, 6, 7, 8, 9, 10] and c["S"].shape[0] == 9 and c["PDE_NAMES"] == [PDE]
    assert c["STATS"]["files"] == 1 and c["STATS"]["stop_rule"]["dropped"] == n - 9
    assert list(c["FIRST"]) == [1, 0, 0, 1, 0, 0, 0, 0, 1]
    R = c["R"]
    assert abs(c["RTG"][0] - (R[0] + 0.9 * R[1] + 0.81 * R[2])) < 1e-6 and abs(c["RTG"][2] - R[2]) < 1e-7
    assert abs(c["RTG"][8] - R[8]) < 1e-7
    for k in ("S", "S2", "A", "R", "ERR", "STEP", "SPENT"):
        assert np.array_equal(c[k], d[k][c["X"]]), k
    assert d["S"].shape[0] == n and list(d["D"]).count(1) == 3, "исходный набор не меняется"

    c, st = O.stop_rule_cut(d, 0.0, 2, gamma=0.9)
    assert (c["STEP"] <= 1).all() and list(c["D"]) == [0, 1] * 3 and st["chains_kmax"] == 3
    c, st = O.stop_rule_cut(d, 0.0625, 2, gamma=0.9)     # раньше сработавшее правило
    assert list(c["EP"]) == [0, 0, 1, 1, 2] and list(c["D"]) == [0, 1, 0, 1, 1]
    assert st["chains_tol"] == 1 and st["chains_kmax"] == 2
    c, _ = O.stop_rule_cut(d, 0.0, 5)                    # K не меньше длины: ничего не режется
    assert len(c["A"]) == n and np.array_equal(c["D"], d["D"])
    must_fail(lambda: O.stop_rule_cut({k: v for k, v in d.items() if k != "STEP"}, 0.1, 0), ValueError,
              "без массивов chain_fix — отказ")
    print("stop_rule_cut: по умолчанию тот же набор, обрезка по допуску (<=) и K_max, терминальный конец, RTG: OK")


def test_mb():
    tab, ens = T.fake_model()
    d = T.make_data(seed=2, n_chains=4, length=4)
    st, valid, first = MB.chain_states(d, tab, np.array([0]), np.ones(1))
    n = len(d["A"])
    sd3 = float(np.std([0.0, 0.02, 0.04]))
    r1 = 0.05 + 0.01 * np.arange(27) - 0.02 - 0.5 * sd3              # dlog, одно действие (FakeEns)
    best = r1[26]
    zf, sc = np.zeros(n, bool), np.ones(n)
    q1, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 1, 0.5)
    q3, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5)
    assert np.allclose(q3, q1 + 0.9 * best + 0.81 * best)
    q3d, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, kmax=0, tl=None)
    assert np.array_equal(q3, q3d), "по умолчанию правило выключено"
    # K_max: строка на последней разрешённой стадии — одно действие; на предпоследней — два
    for K, want in ((4, None), (5, None)):
        qk, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, kmax=K)
        left = K - d["STEP"]                               # сколько стадий цепочке ещё можно
        for i in range(n):
            ref = q1[i] + (0.9 * best if left[i] >= 2 else 0) + (0.81 * best if left[i] >= 3 else 0)
            assert np.allclose(qk[i], ref), (K, i)
    # допуск: действия a >= 13 доводят ошибку до допуска сразу (средняя убыль log10 = 0.03 + 0.01 a),
    # остальные — после второго (лучшего) действия
    one = {k: v[1:2].copy() for k, v in st.items()}
    l0 = float(one["l"][0])
    tl = MB.tol_log(tab, [0], np.ones(1), 10.0 ** (l0 - 0.155))
    qt, _ = MB.lookahead(ens, tab, one, one["l"], np.zeros(1, bool), np.ones(1), "dlog", 0.9, 3, 0.5, tl=tl)
    a = np.arange(27)
    assert np.allclose(qt[0], np.where(a >= 13, q1[1], q1[1] + 0.9 * best)), "за допуском продолжения нет"
    qf, _ = MB.lookahead_full(ens, tab, one, one["l"], np.zeros(1, bool), np.ones(1), "dlog", 0.9, 3, 0.5, tl=tl)
    assert np.allclose(qf[0, 13:], q1[1, 13:]) and (qf[0, :13] <= q3[1, :13] + 1e-12).all()
    assert MB.tol_log(tab, [0], np.ones(1), 0.0) is None
    tl2 = MB.tol_log(tab, [0], np.array([4.0]), 0.01)
    assert abs(tl2[0] - np.log10(0.04)) < 1e-12, "допуск в единицах тренера переводится в исходную ошибку"
    # precompute: по умолчанию бит в бит; с K_max строки последней стадии = одношаговая цель
    mask = np.ones(n, bool)
    Q0, W0, i0 = MB.precompute(d, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9, h=3, lam=0.5,
                               model=(tab, ens), log=QUIET)
    Qz, Wz, iz = MB.precompute(d, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9, h=3, lam=0.5,
                               model=(tab, ens), log=QUIET, tol=0.0, kmax=0)
    assert np.array_equal(Q0, Qz) and np.array_equal(W0, Wz) and "stop_kmax" not in i0
    Q1, _, _ = MB.precompute(d, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9, h=1, lam=0.5,
                             model=(tab, ens), log=QUIET)
    keep = {}
    Qk, Wk, ik = MB.precompute(d, mask, [PDE], "dlog", np.ones(1), np.array([1.0]), 0.9, h=3, lam=0.5,
                               model=(tab, ens), log=QUIET, kmax=3, keep=keep)
    last = d["STEP"] == 2
    assert np.array_equal(Qk[last], Q1[last]) and np.array_equal(Qk[d["STEP"] == 0], Q0[d["STEP"] == 0])
    assert ik["stop_kmax"] == 3 and keep["model"][1] is ens
    print("модельные цели: K_max и допуск обрывают просмотр вперёд (жадный и полный), по умолчанию бит в бит: OK")
    return d


def test_teacher(ckpt, d):
    tab, ens = T.fake_model()
    n = len(d["A"])
    mask = np.ones(n, bool)
    args = (ckpt, d, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, CPU)
    Q0, V0, i0 = O.teacher_targets(*args, budget=7000, log=QUIET)
    Qz, Vz, _ = O.teacher_targets(*args, budget=7000, log=QUIET, tol=0.0, kmax=0, model=(tab, ens))
    assert np.array_equal(Q0, Qz) and np.array_equal(V0, Vz) and i0["units"] == "dlog->delta", "без правила — прежнее"
    must_fail(lambda: O.teacher_targets(*args, budget=7000, log=QUIET, kmax=3), ValueError, "правило без суррогата")
    Q1, W1, _ = MB.precompute(d, mask, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, h=1, lam=0.5,
                              budget=7000, model=(tab, ens), log=QUIET)
    Qk, Vk, ik = O.teacher_targets(*args, budget=7000, log=QUIET, kmax=3, model=(tab, ens), lam=0.5)
    assert ik["units"] == "rollout->delta" and ik["stop_kmax"] == 3 and Vk.any()
    last = Vk & (d["STEP"] == 2)
    assert last.any() and np.allclose(Qk[last], Q1[last], atol=1e-6), "последняя стадия: только своё действие"
    early = Vk & (d["STEP"] == 0)
    assert (np.abs(Qk[early] - Q1[early]).max(1) > 1e-6).all(), "до последней стадии продолжение учителя есть"
    # допуск выше любой ошибки после первого действия: у всех строк цель одношаговая
    Qt, Vt, _ = O.teacher_targets(*args, budget=7000, log=QUIET, tol=10.0, model=(tab, ens), lam=0.5)
    assert np.allclose(Qt[Vt], Q1[Vt], atol=1e-6) and Vt.sum() == V0.sum()
    # продолжение — жадная политика учителя: K на одну стадию больше даёт ровно ещё одну награду
    teacher_act = {}
    orig = MB.policy_rollout

    def spy(ens_, tab_, st_, *a, **k):
        act = a[6]                       # l_rew, zero_first, scale, form, gamma, lam, act

        def act2(s):
            out = act(s)
            teacher_act.setdefault("n", 0)
            teacher_act["n"] += len(out)
            return out
        return orig(ens_, tab_, st_, *a[:6], act2, *a[7:], **k)
    MB.policy_rollout = spy
    try:
        Qk4, _, _ = O.teacher_targets(*args, budget=7000, log=QUIET, kmax=4, model=(tab, ens), lam=0.5)
    finally:
        MB.policy_rollout = orig
    assert teacher_act["n"] > 0 and np.allclose(Qk4[d["STEP"] == 3], Q1[d["STEP"] == 3], atol=1e-6)
    print("цели учителя под правилом: продолжение его политики в суррогате до остановки, на последней стадии и "
          "за допуском — одношаговая модельная цель, без правила бит в бит, отказ без суррогата: OK")


def test_offline_main(bufs, ckpt, tmp):
    import killswitch
    root = os.path.join(tmp, "bufs", SUB)
    os.makedirs(root, exist_ok=True)
    for k, ep in enumerate(bufs[SUB]):
        torch.save(ep, os.path.join(root, f"f{k}.pt"))
    e0 = float(O.pde_meta()[PDE]["init_err"])
    mtag = f"tsr{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
            "--teacher-from", ckpt, "--teacher-w", "0.5", "--seeds", "1", "--model-tag", mtag,
            "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu", "--no-verify"]

    def fake_build(names, subdirs, table, members, work, quick, log):
        tab, ens = T.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3, surr_file="/x/combo_surr_fake.pkl")

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
        for extra in ([], ["--train-kmax", "2"], ["--train-kmax", "2", "--mb-aux", "0", "--mb-h", "3"]):
            av = list(argv)
            if "--mb-aux" in extra:                         # без модельных целей: суррогат строится для учителя
                i = av.index("--mb-aux")
                del av[i:i + 6]
                extra = ["--train-kmax", "2"]
            text = run(av + extra)
            rows.append([json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0])
        r0, r1, r2 = rows
        assert "train_tol" not in r0 and "train_kmax" not in r0 and "stop_rule" not in r0, "по умолчанию полей нет"
        assert r0["teacher_units"] == "dlog->delta" and "mb_stop_kmax" not in r0
        assert r1["train_kmax"] == 2 and r1["train_tol"] == 0.0 and r1["stop_rule"]["chains_kmax"] > 0
        assert r1["teacher_units"] == "rollout->delta" and r1["mb_stop_kmax"] == 2
        assert r1["teacher_rows"] < r0["teacher_rows"] and r1["teacher_rows"] == 2 * 5, (r0["teacher_rows"], r1)
        assert r2["teacher_units"] == "rollout->delta" and "mb_rows" not in r2
        import online_eval_env as E
        meta = E.load_agent(out)[0]._meta
        assert meta["train_kmax"] == 2 and meta["stop_rule"]["dropped"] > 0 and meta["ctx_kmax"] == 70
        for bad in (argv + ["--train-kmax", "-1"],
                    [a for a in argv if a != "--chain-fix"] + ["--train-kmax", "2"],
                    argv + ["--train-tol", "0.1", "--distill-from", ckpt]):
            must_fail(lambda b=bad: run(b), SystemExit, f"недопустимое сочетание флагов: {bad[-3:]}")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("offline_rl.main(): по умолчанию строка без новых полей; --train-kmax режет цепочки, цели учителя через "
          "суррогат (и без --mb-aux), поля в строке и meta, отказы: OK")


def test_online(ckpt, tmp):
    import test_online_finetune as TF
    import online_train_env as OT
    tab, ens = T.fake_model()
    d = T.make_data(seed=3, n_chains=5, length=4)
    n = len(d["A"])
    mask = d["EP"] < 4
    aux = OA.Aux(PDE, w_mb=0.5, w_te=0.5, reward_form="delta", err_scale=1.0, init_err=0.811, gamma=0.99,
                 budget=7000, mb_h=3, mb_lam=0.5, teacher_from=ckpt, device=CPU, log=QUIET, kmax=3, tol=0.05)
    off = aux.offline(d, mask, model=(tab, ens))
    QMB, MBW, _ = MB.precompute(d, mask, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, h=3, lam=0.5,
                                budget=7000, model=(tab, ens), log=QUIET, kmax=3, tol=0.05)
    QTE, TEV, _ = O.teacher_targets(ckpt, d, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, CPU, budget=7000,
                                    log=QUIET, kmax=3, tol=0.05, model=(tab, ens), lam=0.5)
    assert np.array_equal(off["QMB"], QMB) and np.array_equal(off["QTE"], QTE) and (off["TEW"] == (TEV & mask)).all()
    assert aux.meta()["aux_stop_kmax"] == 3 and aux.te_info["units"] == "rollout->delta"
    for i in range(n):
        s0 = i - int(d["STEP"][i])
        r = aux.live(d["A"][s0:i], [O.action_epochs(a) for a in d["A"][s0:i]], d["ERR"][s0:i]) or {}
        if TEV[i]:
            assert np.allclose(r["te"], QTE[i], atol=1e-6), i
        if mask[i] and MBW[i]:
            assert np.allclose(r["mb"], QMB[i], atol=1e-6), i
    aux_t = OA.Aux(PDE, w_te=0.5, reward_form="delta", init_err=0.811, gamma=0.99, budget=7000,
                   teacher_from=ckpt, device=CPU, log=QUIET, kmax=3)
    off_t = aux_t.offline(d, mask, model=(tab, ens))
    QTE3, _, _ = O.teacher_targets(ckpt, d, [PDE], "delta", np.ones(1), np.array([0.811]), 0.99, CPU, budget=7000,
                                   log=QUIET, kmax=3, model=(tab, ens), lam=0.5)
    assert aux_t.model is not None and "QMB" not in off_t and np.array_equal(off_t["QTE"], QTE3), \
        "без --mb-aux суррогат строится для целей учителя"

    # ---- полный прогон тренера в заглушке среды: офлайновая половина и онлайновые эпизоды по правилу
    path = os.path.join(tmp, "agent_convnext_dqn_cfix_rdelta_vb_X_seed1.pt")
    TF.offline_ckpt(path)
    spec = TF.make_tbuf(os.path.join(tmp, "tbuf"), dict(keep_opt=True, mode="safe", budget=TF.BUDGET))
    base = [a for a in TF.FT if a not in ("--rlpd-subdir", SUB)] + [
        "--warm-start", path, "--warm-strict", "--eps-start", "0.1", "--rlpd-target-buffer", spec]
    from target_buffer import load as tb_load
    episodes, _ = tb_load(spec, SUB, log=QUIET)
    data = O.episodes_to_arrays(episodes, chain_fix=True, reward_form="delta", init_err=0.811, budget=TF.BUDGET,
                                verbose=False)
    tol, K = 0.2, 3
    cut, st = O.stop_rule_cut(data, tol, K, gamma=0.99)
    tr, _ = O.split_by_episode(cut)
    out, spy = TF.run_train(tmp, base + ["--train-tol", str(tol), "--train-kmax", str(K), "--tag", "sr_x"], n_chains=6)
    off_b = spy["off"]
    idx = np.where(tr)[0]
    assert len(off_b) == len(idx) and off_b.a == [int(a) for a in cut["A"][idx]], (len(off_b), len(idx))
    assert np.array_equal(np.float32(off_b.d), cut["D"][idx]) and st["chains_tol"] + st["chains_kmax"] > 0
    assert "правило остановки обучения" in out
    buf = torch.load(os.path.join(tmp, "buffer_sr_x.pt"), map_location="cpu", weights_only=False)
    ep_ids = np.unique(buf["EP"])
    assert len(ep_ids) >= 3
    why = []
    for e in ep_ids:
        sel = buf["EP"] == e
        D, ERR, STEP = buf["D"][sel], buf["ERR"][sel], buf["STEP"][sel]
        assert STEP.max() <= K - 1 and D[-1] == 1.0 and not D[:-1].any(), (e, D, STEP)
        hit = np.where(ERR <= tol)[0]
        spent_out = np.where(buf["SPENT"][sel] + buf["EPOCHS"][sel] >= TF.BUDGET)[0]
        end = min([K - 1] + [int(x[0]) for x in (hit, spent_out) if len(x)])
        assert len(STEP) == end + 1, (e, ERR, STEP)
        why.append("tolerance" if len(hit) and hit[0] == end else "budget" if len(spent_out) and spent_out[0] == end
                   else "kmax")
    assert {"tolerance", "kmax"} <= set(why), why
    row = [json.loads(ln) for ln in out.splitlines() if ln.startswith('{"variant"')][-1]
    assert row["train_tol"] == tol and row["train_kmax"] == K and row["stop_rule_offline"]["dropped"] == st["dropped"]
    meta = torch.load(os.path.join(tmp, "agent_sr_x.pt"), map_location="cpu", weights_only=False)["meta"]
    assert meta["train_kmax"] == K and meta["ctx_kmax"] == TF.KMAX
    # прежняя командная строка: правило выключено, полей нет
    out0, spy0 = TF.run_train(tmp, base + ["--tag", "sr_0"], n_chains=2)
    row0 = [json.loads(ln) for ln in out0.splitlines() if ln.startswith('{"variant"')][-1]
    assert "train_kmax" not in row0 and len(spy0["off"]) == int(O.split_by_episode(data)[0].sum())
    for bad, what in ((base + ["--train-kmax", "71"], "K больше --max-chain-steps"),
                      ([a if a != "0" or base[i - 1] != "--tolerance" else "0.1" for i, a in enumerate(base)]
                       + ["--train-tol", "0.2"], "вместе с --tolerance"),
                      ([a for a in base if a != "--offline-fix"] + ["--train-kmax", "3"], "без --offline-fix"),
                      (base + ["--train-kmax", "3", "--n-step", "3"], "--n-step")):
        TF.must_exit(lambda b=bad: TF.run_train(tmp, b), "2", f"разбор: {what}")
    assert OT.train_end is O.train_end
    print(f"онлайн: цели обеих половин с правилом как у offline_rl.py, офлайновая половина обрезана ({st['dropped']} "
          f"переходов), {len(ep_ids)} онлайновых эпизодов кончаются на допуске или K-й стадии с d=1 ({' '.join(why)}), отказы: OK")


def test_epochs_to_threshold(tmp):
    path = os.path.join(ROOT, "experiments", "rl_arch", "epochs_to_threshold.py")
    spec = importlib.util.spec_from_file_location("epochs_to_threshold", path)
    ett = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ett)
    rd = os.path.join(tmp, "rows")
    os.makedirs(rd)
    e0 = float(O.pde_meta()[PDE]["init_l2re"])
    # hist: [потрачено эпох нарастающим итогом, лосс, полная l2re, секунды]
    rows = {
        42: [[300, 50.0, 0.6, 10], [1300, 1.0, 0.05, 20], [2300, 0.1, 0.02, 30], [7000, 0.05, 0.0105, 90]],
        43: [[1000, 9.0, 0.3, 10], [3500, 0.2, 0.04, 30], [4000, 0.2, 0.04, 40], [7000, 0.1, 0.03, 70]],
    }
    for sd, h in rows.items():
        with open(os.path.join(rd, f"{PDE}_t3m_x_g_seed{sd}.json"), "w") as f:
            json.dump(dict(seed=sd, pde=PDE, l2re=h[-1][2], stop_reason="budget", budget=7000, hist=h), f)
    with open(os.path.join(rd, f"{PDE}_t3m_x_g_seed44.json"), "w") as f:     # промежуточная выгрузка
        json.dump(dict(seed=44, pde=PDE, l2re=0.5, stop_reason=None, partial=True, budget=7000,
                       hist=[[300, 1.0, 0.001, 1]]), f)
    with open(os.path.join(rd, f"{PDE}_t3m_smoke_seed42.json"), "w") as f:
        json.dump(dict(seed=42, pde=PDE, l2re=0.5, stop_reason="budget", budget=7000, hist=[[300, 1.0, 0.5, 1]]), f)
    assert ett.first_below(rows[42], 0.02) == (2300, 3) and ett.first_below(rows[43], 0.02) == (None, None)
    assert ett.first_below([[100, None, None, 1], [200, None, 0.01, 2]], 0.02) == (200, 2)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        summ = ett.main([rd, "--rel", "0.05,0.02", "--abs", "0.05", "--final-rel", "0.1", "--kmax", "2",
                         "--json", os.path.join(tmp, "s.json")])
    text = buf.getvalue()
    assert set(summ) == {"t3m_x_g"} and "в работе: 1" in text, text
    r = summ["t3m_x_g"]
    assert r["n"] == 2 and abs(r["final_median"] - (0.0105 + 0.03) / 2) < 1e-12 and r["init_l2re"] == e0
    a = r["thr"]["0.05"]
    assert a["reached"] == 2 and a["per_seed"] == {42: [1300, 2], 43: [3500, 2]} and a["med_epochs"] == 2400
    assert a["in_kmax"] == 2 and a["med_stages"] == 2
    t2 = r["thr"]["0.02*E0"]
    assert abs(t2["thr"] - 0.02 * e0) < 1e-12 and t2["reached"] == 1 and t2["med_epochs_all"] == float("inf")
    fin = r["thr"]["итог*(1+0.1)"]
    assert fin["thr"] is None and fin["per_seed"] == {42: [7000, 4], 43: [7000, 4]}
    assert json.load(open(os.path.join(tmp, "s.json")))["t3m_x_g"]["n"] == 2
    print("epochs_to_threshold: эпохи и стадии до порога полной l2re, недосчитанные сиды и smoke пропущены, "
          "пороги от init_l2re и от итога: OK")


def main():
    t0 = time.time()
    import test_shield_teacher as TS
    bufs = TS.make_buffers()
    with tempfile.TemporaryDirectory() as tmp:
        test_cut()
        d = test_mb()
        with contextlib.redirect_stdout(io.StringIO()):
            ckpt, _ = TS.train_teacher(bufs, tmp)
        test_teacher(ckpt, d)
        test_offline_main(bufs, ckpt, tmp)
        with AC.one_thread():
            test_online(ckpt, tmp)
        test_epochs_to_threshold(tmp)
    junk = os.path.join(ROOT, "agent_combo_work")
    if os.path.isdir(junk) and not os.listdir(junk):
        shutil.rmtree(junk)
    print(f"test_stop_rule: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
