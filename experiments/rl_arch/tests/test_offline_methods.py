#!/usr/bin/env python
"""
Проверки офлайновых приёмов тренера offline_rl.py без GPU и без сети, на крошечных синтетических
данных (данные и помощники — из test_map_agent.py и test_shield_teacher.py):

  * --hlg (HL-Gauss): голова на бинах, опора по границе --value-bound, цель TD кросс-энтропией,
    добавки --mb-aux / --teacher-w по матожиданию головы (и --hlg-aux ce), чекпоинт с полем
    hl_gauss проходит через online_eval_env.load_agent и q_values отдаёт 27 конечных чисел, равных
    Q обучавшейся сети; онлайновые чекпоинты HL-Gauss (бины в квантильной голове) читаются как раньше;
  * --mb-full: лучевой просмотр вперёд по сумме наград находит ход с убытком, который окупается
    (жадный его не видит), при H <= 3 не хуже жадного, бюджет, --mb-max-rows;
  * --aux-next: цель из S2 (surrogate.level6) и награды, голова не попадает в чекпоинт;
  * смесь УрЧП с отложенной задачей: --subdirs + --holdout + --pde-ctx + --scalar-ctx + --encoder
    resnet + --mb-aux (суррогат без отложенной задачи) + --teacher-from (учитель agent_combo.py
    --extra-only-target, без строк цели); чекпоинт оценивается на отложенной задаче;
  * значения по умолчанию: имена и формы параметров прежние (числа сверяет test_map_agent.py).

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_offline_methods.py
"""
import contextlib
import io
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import mb_targets as MB  # noqa: E402
import agent_combo as AC  # noqa: E402
import test_map_agent as T  # noqa: E402
import test_shield_teacher as TS  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a: None  # noqa: E731
PDE, SUB = TS.PDE, TS.SUB                    # отложенная задача: poissonboltzmann2d
PDE2, SUB2 = TS.PDE2, TS.SUB2
SUB3 = "wave1d"


def must_fail(fn, exc, what):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"должно было упасть: {what}")


def save_ckpt(net, stats, path, **extra):
    T.save_ckpt(net, stats, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.update(extra)
    torch.save(ck, path)
    return path


# --------------------------------------------------------------------------------------------

def test_hlg(data, mask, QMB, MBW, tmp):
    import online_eval_env as E
    from advanced_agents import HLGauss
    # опора: у delta — 1.1 ошибки на старте цепочки (R + ERR первого шага = 1.0 в make_data)
    lo, hi = O.hlg_range(data, np.where(mask)[0], T.ns(reward_form="delta"))
    assert abs(hi - 1.1) < 1e-6 and abs(lo + 0.55) < 1e-6, (lo, hi)
    assert O.hlg_range(data, np.where(mask)[0], T.ns(reward_form="delta", hlg_vmin=-2.0, hlg_vmax=3.0)) == (-2.0, 3.0)
    lo2, hi2 = O.hlg_range(data, np.where(mask)[0], T.ns(reward_form="dlog"))
    rtg = data["RTG"][mask]
    assert lo2 < np.quantile(rtg, 0.005) and hi2 > np.quantile(rtg, 0.995)
    must_fail(lambda: O.hlg_range(data, np.where(mask)[0], T.ns(hlg_vmin=1.0, hlg_vmax=1.0)), ValueError, "пустая опора")
    # цель: матожидание сглаженного распределения возвращает само значение (вдали от краёв опоры)
    h = HLGauss(v_min=lo, v_max=hi, n_bins=51)
    y = torch.tensor([-0.2, 0.0, 0.013, 0.4, 0.8])
    assert torch.allclose(h.to_scalar(h.target_probs(y).clamp_min(1e-30).log()), y, atol=2e-3)
    must_fail(lambda: O.QNet("cnx_qrdqn", T.CPU, bins=51), ValueError, "бины у квантильного варианта")
    must_fail(lambda: O.QNet("convnext_dqn", T.CPU, bins=51, ens=2), ValueError, "бины у ансамбля голов")
    base, _ = T.fit("convnext_dqn", data, mask, encoder="resnet")
    assert base.bins == 0 and base._hlg is None and "hlg_bins" not in base.extra
    x = torch.as_tensor(np.random.default_rng(5).normal(size=(3, 4, 26, 26)).astype(np.float32))
    for variant, kw in (("convnext_dqn", dict(encoder="resnet", value_bound=True, n_step=2, aug="d4", dqfd=0.2)),
                        ("cnn_cql", dict()), ("cnx_cql", dict(mask_schedule=True, epochs=3))):
        net, stats = T.fit(variant, data, mask, hlg=True, reward_form="delta", **kw)
        sd = net.model.state_dict()
        assert net.bins == 51 and sd["1.weight"].shape[0] == 27 * 51 and net.hl_gauss == (lo, hi, 51)
        assert net.extra["hlg_bins"] == 51 and abs(net.extra["hlg_vmax"] - hi) < 1e-9
        with torch.no_grad():
            assert net.q_online(x).shape == (3, 27, 51)
            q = net.q_scalar(x)
        assert q.shape == (3, 27) and torch.isfinite(q).all() and float(q.min()) >= lo and float(q.max()) <= hi
        path = save_ckpt(net, stats, os.path.join(tmp, f"hlg_{variant}.pt"), hl_gauss=net.hl_gauss)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert O.ckpt_arch(ck)["bins"] == 51
        ag, mean, std, var = E.load_agent(path)
        assert ag.bins == 51 and ag._hlg is not None and not ag.model.training
        for st in (np.zeros((4, 26, 26), np.float32), x[0].numpy()):
            qv = E.q_values(ag, st, mean, std, var)
            assert qv.shape == (27,) and np.isfinite(qv).all()
            assert np.abs(qv - E.q_values(net, st, mean, std, var)).max() < 1e-5
            assert E.pick_action(ag, st, mean, std, var) is not None
        m, _ = O.evaluate(net, stats, data, ~mask)
        assert np.isfinite(m["td_error"])
        assert O.verify_ckpt(path, net)["ok"]
    # плотные добавки на матожидании головы и кросс-энтропией; всё вместе
    d2 = dict(data, QMB=QMB, MBW=MBW)
    for mode in ("ev", "ce"):
        net, _ = T.fit("convnext_dqn", d2, mask, hlg=True, hlg_aux=mode, reward_form="delta", value_bound=True,
                       encoder="resnet", mb_aux=0.5, pani_w=0.1, pani_sigma=0.1, distill_top=50.0, distill_w=0.3,
                       epochs=2)
        assert {"mb_loss", "pani_loss", "distill_top_loss", "hlg_bins"} <= set(net.extra), net.extra
        assert all(np.isfinite(v) for v in net.extra.values()) and net.extra["mb_loss"] > 0
    # кросс-энтропия действительно учит: на одной пачке лосс TD падает (цель при gamma=0 — награда)
    old_g = O.GAMMA
    try:
        O.GAMMA = 0.0
        n1, _ = T.fit("cnn_dqn", data, mask, hlg=True, reward_form="delta", epochs=1)
        n40, s40 = T.fit("cnn_dqn", data, mask, hlg=True, reward_form="delta", epochs=25)
    finally:
        O.GAMMA = old_g
    idx = np.where(mask)[0]

    def fit_err(net, stats):
        xs = torch.as_tensor((data["S"][idx] - stats["mean"]) / stats["std"]).float()
        with torch.no_grad():
            q = net.q_scalar(xs).gather(1, torch.as_tensor(data["A"][idx])[:, None]).squeeze(1).numpy()
        return float(np.abs(q - np.clip(data["R"][idx], lo, hi)).mean())
    e1, e40 = fit_err(n1, _), fit_err(n40, s40)
    assert e40 < 0.5 * e1, (e1, e40)
    # онлайновый чекпоинт HL-Gauss (бины в квантильной голове) читается прежним путём
    qn = O.QNet("cnx_qrdqn", T.CPU, n_quantiles=51)
    path = os.path.join(tmp, "online_hlg.pt")
    torch.save({"variant": "cnx_qrdqn", "state_dict": qn.model.state_dict(), "hl_gauss": (-2.0, 6.0, 51),
                "mean": np.zeros((1, 4, 1, 1), np.float32), "std": np.ones((1, 4, 1, 1), np.float32)}, path)
    arch = O.ckpt_arch(torch.load(path, map_location="cpu", weights_only=False))
    assert "bins" not in arch and arch["n_quantiles"] == 51
    ag, mean, std, var = E.load_agent(path)
    qv = E.q_values(ag, x[0].numpy(), mean, std, var)
    assert ag.bins == 0 and qv.shape == (27,) and np.isfinite(qv).all() and -2.0 <= qv.min() and qv.max() <= 6.0
    print(f"HL-Gauss: опора [{lo:.3g}, {hi:.3g}], обучение (ошибка {e1:.3f} -> {e40:.3f}), добавки, "
          "чекпоинт через load_agent, онлайновый чекпоинт: OK")


class TrapEns:
    """Подставной ансамбль с ходом, который окупается позже: любое действие Adam поднимает log10
    ошибки на 0.05 + 0.01*a (убыток), L-BFGS сразу после Adam снимает 1.0, всё остальное — 0.01.
    Члены одинаковы (разброса нет)."""
    members = 2

    def predict_ln(self, X, l, member=None):
        a, pa = X[:, 0].astype(int), X[:, 1].astype(int)
        d = np.full(len(a), -0.01)
        d[a < 9] = 0.05 + 0.01 * a[a < 9]
        d[(a >= 9) & (a < 18) & (pa < 9)] = -1.0
        out = np.stack([np.clip(l + d, S.L_LO, S.L_HI)] * self.members)
        return out if member is None else out[member]


def test_mb_full(data, mask):
    tab, ens = T.fake_model()
    n = len(data["A"])
    st, valid, first = MB.chain_states(data, tab, np.array([0]), np.ones(1))
    zf, sc = first.copy(), np.ones(n)
    args = (st, st["l"], zf, sc, "dlog", 0.9)
    # награда зависит только от действия и положительна: полный просмотр совпадает с жадным
    for h in (1, 2, 3):
        qg, ug = MB.lookahead(ens, tab, *args, h, 0.5)
        for beam in (1, 4):
            qf, uf = MB.lookahead_full(ens, tab, *args, h, 0.5, beam=beam, chunk=5)
            assert qf.shape == (n, 27) and np.allclose(qf, qg) and abs(uf - ug) < 1e-12, (h, beam)
    qs, _ = MB.lookahead_full(ens, tab, *args, 2, 0.5, smdp_scale=100.0)
    assert np.allclose(qs, MB.lookahead(ens, tab, *args, 2, 0.5, smdp_scale=100.0)[0])
    # штраф больше награды: продолжение не лучше остановки, хвост нулевой
    qn, _ = MB.lookahead_full(ens, tab, *args, 3, 20.0)
    assert np.allclose(qn, MB.lookahead(ens, tab, *args, 1, 20.0)[0])
    # бюджет 7000. Остаток 100 эпох: любое действие последнее. Остаток 500: Adam на 2500 эпох
    # обрезается и кончает эпизод; после Adam на 100 эпох остаётся 400 — жадный берёт два PSO
    # (300 и обрезанное до 100), полный находит три действия: PSO на 200, на 100 и ещё одно на 100
    one = {k: v[:2].copy() for k, v in st.items()}
    one["spent"][:] = [6900, 6500]
    one["step"][:] = [5, 5]
    b_args = (one, one["l"], np.zeros(2, bool), np.ones(2), "dlog", 1.0)
    qb1, _ = MB.lookahead(ens, tab, *b_args, 1, 0.5, budget=7000)
    qg4, _ = MB.lookahead(ens, tab, *b_args, 4, 0.5, budget=7000)
    qb4, _ = MB.lookahead_full(ens, tab, *b_args, 4, 0.5, beam=4, budget=7000)
    rr = 0.05 + 0.01 * np.arange(27) - 0.02 - 0.5 * float(np.std([0.0, 0.02, 0.04]))
    assert np.allclose(qb4[0], qb1[0]) and np.allclose(qb4[1, 2], qb1[1, 2]) and (qb4 >= qg4 - 1e-12).all()
    assert np.allclose(qg4[1, 0], qb1[1, 0] + 2 * rr[26]) and np.allclose(qb4[1, 0], qb1[1, 0] + rr[24:27].sum())
    # ход с убытком, который окупается: после первого действия не из Adam лучший план —
    # Adam (a=0: -0.05), затем L-BFGS (+1.0); жадный берёт +0.01 и его не видит
    trap = TrapEns()
    i = int(np.where(~first)[0][0])
    s1 = {k: v[i:i + 1] for k, v in st.items()}
    t_args = (s1, s1["l"], np.zeros(1, bool), np.ones(1), "dlog", 0.9)
    qg, _ = MB.lookahead(trap, tab, *t_args, 3, 0.5)
    r1 = np.where(np.arange(27) < 9, -(0.05 + 0.01 * np.arange(27)), 0.01)
    if int(s1["pa"][0]) < 9 and int(s1["pa"][0]) >= 0:
        r1[9:18] = 1.0
    pso = np.arange(18, 27)
    assert np.allclose(qg[0, pso], r1[pso] + 0.9 * (0.01 + 0.9 * 0.01))
    for beam in (3, 4, 27):
        qf, _ = MB.lookahead_full(trap, tab, *t_args, 3, 0.5, beam=beam)
        assert np.allclose(qf[0, pso], r1[pso] + 0.9 * (-0.05 + 0.9 * 1.0)), (beam, qf[0, pso])
        assert np.allclose(qf[0, :9], r1[:9] + 0.9 * (1.0 + 0.9 * 0.01)) and (qf >= qg - 1e-12).all()
    q1b, _ = MB.lookahead_full(trap, tab, *t_args, 3, 0.5, beam=1)        # один узел: как жадный
    assert np.allclose(q1b[0, pso], qg[0, pso])
    q2f, _ = MB.lookahead_full(trap, tab, *t_args, 2, 0.5)                # глубины не хватает
    assert np.allclose(q2f[0, pso], r1[pso] + 0.9 * 0.01)
    # precompute: флаг full и предел числа строк (остальные обучающие строки — без цели)
    e0 = float(O.pde_meta()[T.PDE]["init_err"])
    kw = dict(h=3, lam=0.5, model=(tab, trap), log=QUIET)
    Qg, Wg, ig = MB.precompute(data, mask, [T.PDE], "dlog", np.ones(1), np.array([e0]), 0.9, **kw)
    Qf, Wf, inf = MB.precompute(data, mask, [T.PDE], "dlog", np.ones(1), np.array([e0]), 0.9, full=True, beam=4, **kw)
    assert (Wf == Wg).all() and (Qf >= Qg - 1e-6).all() and (Qf > Qg + 0.1).any()
    assert inf["search"] == "full" and inf["search_beam"] == 4 and "search" not in ig and "rows_cap" not in ig
    Qc, Wc, ic = MB.precompute(data, mask, [T.PDE], "dlog", np.ones(1), np.array([e0]), 0.9, full=True, beam=4, max_rows=5,
                               rows_seed=1, **kw)
    assert Wc.sum() == 5 and (Wc <= Wg).all() and ic["capped"] == int(Wg.sum()) - 5 and not Qc[~Wc].any()
    assert np.allclose(Qc[Wc], Qf[Wc])
    Qc2, Wc2, _ = MB.precompute(data, mask, [T.PDE], "dlog", np.ones(1), np.array([e0]), 0.9, max_rows=5,
                                rows_seed=1, **kw)
    assert (Wc2 == Wc).all(), "выбор строк зависит только от сида"
    print("полный просмотр вперёд: совпадение с жадным, окупающийся убыток, бюджет, предел строк: OK")


def test_aux_next(data, mask, tmp):
    import online_eval_env as E
    idx = np.where(mask)[0]
    y, w = O.aux_next_targets(data, idx)
    n = len(data["A"])
    assert y.shape == (n, 7) and w.shape == (n, 7) and y.dtype == np.float32
    term = data["D"] == 1
    assert not w[term, :6].any() and w[~term].all() and w[:, 6].all() and not y[term, :6].any()
    lv = S.level6(data["S2"])
    use = mask & ~term
    z = (lv[use] - lv[use].mean(0)) / (lv[use].std(0) + 1e-6)
    assert np.allclose(y[use, :6], z, atol=1e-4), "цель — нормированный уровень следующих карт"
    assert abs(float(y[mask, 6].mean())) < 1e-5
    base, _ = T.fit("convnext_dqn", data, mask, encoder="resnet")
    net, stats = T.fit("convnext_dqn", data, mask, encoder="resnet", aux_next=0.5, epochs=2)
    assert T.shapes_of(net) == T.shapes_of(base), "голова --aux-next не должна попадать в чекпоинт"
    assert net.extra["aux_next_loss"] > 0 and np.isfinite(net.extra["aux_next_loss"]) and net.extra["aux_next_params"] > 0
    assert "aux_next_loss" not in base.extra
    # цель учит энкодер: с весом градиент по энкодеру другой, веса расходятся с базой
    b2, _ = T.fit("convnext_dqn", data, mask, encoder="resnet", epochs=2)
    k0 = next(k for k in net.model.state_dict() if k.startswith("0.") and net.model.state_dict()[k].dim() == 4)
    assert not torch.allclose(net.model.state_dict()[k0], b2.model.state_dict()[k0])
    path = T.save_ckpt(net, stats, os.path.join(tmp, "auxnext.pt"))
    ag, mean, std, var = E.load_agent(path)
    q = E.q_values(ag, np.zeros((4, 26, 26), np.float32), mean, std, var)
    assert q.shape == (27,) and np.isfinite(q).all() and O.verify_ckpt(path, net)["ok"]
    for variant, kw in (("cnx_cql_qr", dict(ens=2, ens_min=2)), ("cnn_dqn", dict(hlg=True, reward_form="delta")),
                        ("convnext_dqn", dict(aug="d4", n_step=2))):
        nv, _ = T.fit(variant, T.make_data(in_ch=9), mask, aux_next=0.3, **kw)
        assert np.isfinite(nv.extra["aux_next_loss"])
    print("цель следующего состояния: уровень карт S2 и награда, голова вне чекпоинта: OK")


# --------------------------------------------------------------------------------------------

def run_main(argv):
    buf = io.StringIO()
    sys.argv = argv
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        O.main()
    return buf.getvalue()


def test_main(bufs, tmp):
    """Боевой рецепт одной задачи со всеми новыми флагами: строка результата, meta, загрузка."""
    import killswitch
    import online_eval_env as E
    e0 = float(O.pde_meta()[PDE]["init_err"])
    root = os.path.join(tmp, "bufs", SUB)
    mtag = f"toffm{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--mb-aux", "0.5", "--mb-h", "3", "--mb-lam", "0.5",
            "--seeds", "1", "--model-tag", mtag, "--epochs", "2", "--fqe-epochs", "1", "--batch-size", "8", "--cpu",
            "--hlg", "--mb-full", "--mb-beam", "3", "--mb-max-rows", "6", "--aux-next", "0.3"]
    seen = {}

    def fake_build(names, subdirs, table, members, work, quick, log):
        seen.update(subdirs=subdirs)
        tab, ens = T.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3)

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        text = run_main(argv)
        r = [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0]
        want = dict(hlg=True, hlg_aux="ev", hlg_bins=51, aux_next=0.3, mb_full=True, mb_beam=3, mb_max_rows=6,
                    mb_rows=6)
        assert {k: r.get(k) for k in want} == want, {k: r.get(k) for k in want}
        assert abs(r["hlg_vmax"] - 1.1 * e0) < 1e-5 and abs(r["hlg_vmin"] + 0.55 * e0) < 1e-5, (r["hlg_vmax"], e0)
        assert np.isfinite(r["mb_loss"]) and np.isfinite(r["aux_next_loss"]) and np.isfinite(r["td_error"])
        assert np.isfinite(r["fqe_value_all"]) and seen["subdirs"] == "all" and "'ok': True" in text
        ck = torch.load(out, map_location="cpu", weights_only=False)
        assert tuple(ck["hl_gauss"])[2] == 51 and abs(ck["hl_gauss"][1] - 1.1 * e0) < 1e-5
        assert not [k for k in ck["state_dict"] if not k.startswith(("0.", "1."))], "в чекпоинте только энкодер и голова"
        ag, mean, std, var = E.load_agent(out)
        meta = ag._meta
        assert meta["hlg"] is True and meta["aux_next"] == 0.3 and meta["mb_full"] is True and meta["in_ch"] == 9
        assert meta["mb"]["search"] == "full" and meta["mb"]["capped"] > 0 and meta["mb"]["rows_cap"] == 6 and "mb_holdout" not in meta
        st = E.add_scalar_ctx(np.zeros((4, 26, 26), np.float32), 0, 70, 0, 7000, None, None)
        q = E.q_values(ag, st, mean, std, var)
        assert q.shape == (27,) and np.isfinite(q).all()
        # без новых флагов в строке результата и в meta их полей нет (прежние рецепты не меняются)
        cut = argv[:argv.index("--hlg")]
        r0 = [json.loads(ln) for ln in run_main(cut).splitlines() if ln.startswith('{"variant"')][0]
        assert not [k for k in r0 if k.startswith(("hlg", "aux_next", "mb_full", "mb_beam", "mb_max", "mb_holdout"))], r0
        ck0 = torch.load(out, map_location="cpu", weights_only=False)
        assert "hl_gauss" not in ck0 and "hlg" not in ck0["meta"] and ck0["state_dict"]["1.weight"].shape[0] == 27
        i_mb = argv.index("--mb-aux")
        for bad in (argv + ["--ens", "2"], argv[:1] + ["--variant", "cnx_qrdqn"] + argv[3:],
                    argv + ["--freeze-encoder"], argv + ["--hlg-vmin", "1", "--hlg-vmax", "0"],
                    argv + ["--mb-beam", "0"], argv[:i_mb] + argv[i_mb + 2:]):
            must_fail(lambda: run_main(bad), SystemExit, f"недопустимое сочетание флагов: {bad[-4:]}")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("main(): рецепт с --hlg --mb-full --mb-max-rows --aux-next, meta и поле hl_gauss, отказы: OK")


def test_holdout(bufs, tmp):
    """Одна политика на смеси задач без отложенной: учитель без строк цели, суррогат без её буфера,
    чекпоинт оценивается на отложенной задаче (описатель задачи — из meta)."""
    import killswitch
    import online_eval_env as E
    # учитель agent_combo.py --pde <цель> --extra-only-target (без --extra-data): строки цели убраны
    table = os.path.join(tmp, "table3.npz")
    S.save_table(S.build_table(episodes=bufs, verbose=False), table)
    targv = ["--pde", PDE, "--table", table, "--extra-only-target", "--mbpo-h", "2", "--mopo-lam", "0.5",
             "--steps", "12", "--batch", "16", "--n-branch", "20", "--regen", "6", "--tgt-every", "5",
             "--offline-eval", "none", "--surr-quick", "--members", "2", "--no-upload", "--tag", "teach_ho",
             "--work", os.path.join(tmp, "work_ho"), "--out-dir", tmp]
    targs = AC.build_parser().parse_args(targv)
    targs._argv = targv
    with contextlib.redirect_stdout(io.StringIO()):
        rep = AC.run(targs, log=QUIET)
    tck = rep["_paths"]["ckpt"]
    tmeta = torch.load(tck, map_location="cpu", weights_only=False)["meta"]
    others = [O.subdir_to_pde(s) for s in (SUB2, SUB3)]
    assert rep["verify"]["ok"] and sorted(tmeta["train_pdes"]) == sorted(others) and tmeta["pde"] == PDE
    assert rep["sizes"]["target_rows"] == 0 and rep["sizes"]["rows"] > 0
    # порядок папок суррогата у ученика — как у учителя без цели: общий кэш таблицы и ансамбля
    assert O.resolve_subdirs("all", SUB) == [s for s in O.resolve_subdirs("all") if s != SUB]
    mtag = f"toffh{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_multi_{mtag}_seed1.pt"
    enc = os.path.join(tmp, "enc_resnet.pt")           # веса энкодера на 4 канала: вход ученика шире
    torch.save({"encoder": "resnet", "ssl": "mae", "state_dict": O.make_encoder("resnet", 4).state_dict(),
                "mean": np.zeros(4, np.float32), "std": np.ones(4, np.float32)}, enc)
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", os.path.join(tmp, "bufs"),
            "--subdirs", f"{SUB2},{SUB3},{SUB}", "--holdout", SUB, "--pde-ctx", "--scalar-ctx", "--err-norm", "init",
            "--reward-form", "delta", "--value-bound", "--gamma", "0.99", "--episode-budget", "7000",
            "--ctx-kmax", "70", "--save-model", "--encoder", "resnet", "--encoder-init", enc,
            "--mb-aux", "0.5", "--mb-h", "3", "--mb-lam", "0.5", "--mb-subdirs", f"{SUB},{SUB2},{SUB3}",
            "--mb-full", "--mb-max-rows", "10", "--teacher-from", tck, "--teacher-w", "0.5", "--hlg", "--aux-next", "0.2",
            "--seeds", "1", "--model-tag", mtag, "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu"]
    seen = {}

    def fake_build(names, subdirs, table, members, work, quick, log):
        seen.update(names=list(names), subdirs=subdirs)
        tab = dict(A=np.zeros(0, np.int64), CTX=np.stack([S.pde_ctx(O.pde_meta()[nm]) for nm in names]),
                   NAMES=list(names))
        return tab, T.FakeEns(), dict(surr_rows=0, surr_sig="fake", surr_members=3)

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        text = run_main(argv)
        r = [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0]
        assert seen["names"] == others and seen["subdirs"] == f"{SUB2},{SUB3}", seen     # цели в суррогате нет
        assert r["dataset"] == "multi" and r["mb_holdout"] == SUB and r["mb_rows"] == 10 and r["mb_capped"] > 0
        assert r["teacher_rows"] > 0 and np.isfinite(r["teacher_loss"]) and np.isfinite(r["mb_loss"])
        assert abs(r["hlg_vmax"] - 1.1) < 1e-5, "на смеси с --err-norm init опора — доли стартовой ошибки"
        assert "'ok': True" in text and "свежие входные каналы" in text
        ag, mean, std, var = E.load_agent(out)
        meta = ag._meta
        assert meta["pde_ctx"] is True and meta["err_norm"] == "init" and meta["in_ch"] == 15
        assert meta["train_pdes"] == others and PDE not in meta["train_pdes"] and meta["mb_holdout"] == SUB
        # состояние оценки на ОТЛОЖЕННОЙ задаче, как его собирает run_seed: контекст и её описатель
        st = E.add_scalar_ctx(np.zeros((4, 26, 26), np.float32), 0, meta["ctx_kmax"], 0, meta["ctx_budget"], None, None)
        st = np.concatenate([st, np.stack([np.full((26, 26), v, np.float32) for v in O.pde_desc(PDE)])], 0)
        q = E.q_values(ag, st, mean, std, var)
        assert st.shape[0] == 15 and q.shape == (27,) and np.isfinite(q).all()
        # --mb-table с --holdout: таблица берётся как есть, об этом сказано
        text2 = run_main(argv + ["--mb-table", "/nonexistent/table.npz"])
        assert "отложенная задача" in text2 and seen["subdirs"] == f"{SUB},{SUB2},{SUB3}"
        must_fail(lambda: run_main(argv + ["--init-err", "0.8"]), SystemExit, "--init-err на смеси")
        must_fail(lambda: run_main([a for a in argv if a != "--scalar-ctx"]), SystemExit, "--pde-ctx без --scalar-ctx")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("смесь с отложенной задачей: учитель и суррогат без цели, чекпоинт на отложенной задаче: OK")


def main():
    t0 = time.time()
    data = T.make_data()
    mask = data["EP"] < 4
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    tab, ens = T.fake_model()
    QMB, MBW, _ = MB.precompute(data, mask, [T.PDE], "delta", np.ones(1), np.ones(1), 0.9, h=2, lam=0.5,
                                model=(tab, ens), log=QUIET)
    rng = np.random.default_rng(0)
    meta = O.pde_meta()
    bufs = {sd: [TS.make_dump(rng, float(meta[O.subdir_to_pde(sd)]["init_err"]), ln) for ln in ((3, 2, 4), (4, 3))]
            for sd in (SUB2, SUB3, SUB)}
    with tempfile.TemporaryDirectory() as tmp:
        for sd, files in bufs.items():
            os.makedirs(os.path.join(tmp, "bufs", sd))
            for k, f in enumerate(files):
                torch.save(f, os.path.join(tmp, "bufs", sd, f"f{k}.pt"))
        test_hlg(data, mask, QMB, MBW, tmp); lap()
        test_mb_full(data, mask); lap()
        test_aux_next(data, mask, tmp); lap()
        test_main(bufs, tmp); lap()
        test_holdout(bufs, tmp); lap()
    print(f"test_offline_methods: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
