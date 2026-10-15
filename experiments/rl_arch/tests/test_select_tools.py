#!/usr/bin/env python
"""
Проверки инструментов выбора агентов (seq_rank.py, select_agents.py, select_report.py) без GPU, без
сети и без обучения, на синтетических данных, за секунды:

  * последовательный парный тест: на трёх агентах с известным порядком все пары решаются в верную
    сторону и раньше 10 сидов; при одинаковых агентах доля ложных решений не больше alpha; e-процесс
    ставок не растёт при нулевом сигнале; предел свидетельства n сидов;
  * достоверные лучшие k (CAS): ответ совпадает с лучшими k по полному среднему log10 на случайных
    наборах, запросов не больше полного числа, при большом разрыве — меньше; цель topk теста;
  * воспроизведение волн: незаконченные строки пропускаются, учёт сидов (потрачено + сэкономлено =
    имеется, сиды контроля не в счёт), решения сверяются с медианами;
  * select_agents на крошечном синтетическом буфере и двух случайных агентах (с контекстом и без):
    жадные действия совпадают с offline_rl.teacher_q (тот же путь сборки входа), оценка суррогата
    совпадает с ручным расчётом на ансамбле с постоянным шагом и упорядочивает «лучшие» и «худшие»
    действия; FQE через offline_rl.fqe с замороженным кодировщиком MAE (веса не меняются, фабрика
    энкодера возвращается), JSON результата; разбор аргументов и проверка очереди (lint_select);
  * select_report: корреляции и сожаление на подставном результате.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_select_tools.py
"""
import contextlib
import io
import json
import math
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
sys.path.insert(0, os.path.join(HERE, "..", "queue"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import seq_rank as SR  # noqa: E402
import select_agents as SA  # noqa: E402
import select_report as REP  # noqa: E402

torch.set_num_threads(1)
CPU = torch.device("cpu")
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"
KEYS = ["loss_total", "loss_oper", "loss_bnd"]
QUIET = contextlib.redirect_stdout


def synth(means, sd, seeds=SR.SEEDS, rng=None):
    """{тег: {сид: l2re}}: log10 ошибки агента = mean + шум сида (общий для всех: сиды парные) + свой шум."""
    rng = rng or np.random.default_rng(0)
    common = {s: rng.normal(0, 0.15) for s in seeds}
    return {t: {s: 10 ** (m + common[s] + rng.normal(0, sd)) for s in seeds} for t, m in means.items()}


# --------------------------------------------------------------------------------------------

def test_sequential():
    # e-процесс: нулевой сигнал не даёт свидетельства, одинаковый знак — растёт до предела
    b = SR.Bet()
    for y in (1, -1) * 5:
        b.update(y)
    assert b.p > 0.5, b.p
    b = SR.Bet()
    for _ in range(10):
        b.update(1.0)
    assert abs(math.exp(b.maxlogk) - SR.max_evidence(10)) < 1e-6 * SR.max_evidence(10)
    assert abs(SR.max_evidence(10) - 1.4 * 1.8 ** 9) < 1e-9 and SR.max_evidence(5) < 20 < SR.max_evidence(6)
    # известный порядок: A < B < C с шагом 0.5 порядка, шум сида 0.1
    res = synth({"A": -2.0, "B": -1.5, "C": -1.0}, 0.1)
    sr = SR.run_sequential(res, alpha=0.1, bound=0.5)
    better = sr.better_pairs()
    assert better == {("A", "B"), ("B", "C"), ("A", "C")}, better
    assert all(n <= 10 for n in sr.rejected.values()) and sum(len(v) for v in sr.x.values()) < 30, \
        (sr.rejected, {t: len(v) for t, v in sr.x.items()})
    assert sr.ranking() == ["A", "B", "C"] and not any(sr.needs(t) for t in sr.tags)
    # против контроля: решается только пара с контролем, гипотез 2 * (m - 1)
    sc = SR.run_sequential(res, alpha=0.1, bound=0.5, family="control", control="A", goal="control")
    assert sc.n_hyp == 4 and set(sc.better_pairs()) <= {("A", "B"), ("A", "C"), ("B", "C")}
    assert ("A", "C") in sc.better_pairs()
    # цель topk: лучший достоверно в лучших 1, остальные вне
    st = SR.run_sequential(res, alpha=0.1, bound=0.5, goal="topk", k=1)
    assert st.status("A") == "in" and st.status("C") == "out", {t: st.status(t) for t in st.tags}
    # одинаковые агенты: доля ложных решений (любая пара решена) не больше alpha
    rng = np.random.default_rng(1)
    false = 0
    for _ in range(300):
        r0 = synth({"a": -1.5, "b": -1.5, "c": -1.5}, 0.2, rng=rng)
        false += bool(SR.run_sequential(r0, alpha=0.1, bound=0.5).better_pairs())
    assert false / 300 <= 0.1, false
    # пропуски сидов: пара обновляется только на общих сидах
    res2 = {"A": {42: 0.01, 43: 0.01, 45: 0.01}, "B": {42: 0.1, 44: 0.1, 45: 0.1}}
    s2 = SR.SeqRank(["A", "B"])
    for t in ("A", "B"):
        for s, v in res2[t].items():
            s2.add(t, s, v)
    assert s2.hyp[("A", "B")].n == 2 and s2.common("A", "B") == 2
    print("последовательный тест: порядок, контроль, лучшие k, ложные решения при одинаковых агентах: OK")


def test_cas():
    rng = np.random.default_rng(3)
    for trial in range(60):
        m = int(rng.integers(3, 8))
        means = {f"t{i}": float(rng.normal(-1.5, 0.4)) for i in range(m)}
        res = synth(means, 0.25, rng=rng)
        # часть агентов измерена не на всех сидах
        for t in list(res)[: m // 2]:
            res[t] = {s: v for s, v in res[t].items() if s < 47}
        allx = np.log10([v for r in res.values() for v in r.values()])
        lo, hi = float(allx.min()), float(allx.max())
        k = int(rng.integers(1, m))
        c = SR.cas_select(res, k, lo, hi)
        mlog = {t: float(np.mean(np.log10(list(r.values())))) for t, r in res.items()}
        want = sorted(sorted(res), key=lambda t: (mlog[t], sorted(res).index(t)))[:k]
        assert c["certified"] and sorted(c["top"]) == sorted(want), (trial, c["top"], want)
        assert c["queries"] <= c["full"] == sum(len(r) for r in res.values())
    # большой разрыв: лучший сертифицирован раньше полного числа сидов
    res = synth({"good": -2.5, "mid": -1.0, "bad": -0.5}, 0.05)
    allx = np.log10([v for r in res.values() for v in r.values()])
    c = SR.cas_select(res, 1, float(allx.min()), float(allx.max()))
    assert c["certified"] and c["top"] == ["good"] and c["queries"] < c["full"], c["queries"]
    # живое применение: нужный сертификату сид ещё не измерен — возвращается, что запустить
    part = {t: {s: v for s, v in r.items() if s <= 43} for t, r in res.items()}
    c2 = SR.cas_select(part, 1, -3.0, 0.5, scen={t: list(SR.SEEDS) for t in part})
    assert not c2["certified"] and c2["next"] is not None and c2["next"][1] == 44, c2
    print("CAS: точное совпадение с лучшими k по среднему log10 (60 наборов), экономия при разрыве, следующий сид: OK")


def write_rows(d, pde, res, partial=()):
    for t, r in res.items():
        for s, v in r.items():
            with open(os.path.join(d, f"{pde}_{t}_seed{s}.json"), "w") as f:
                json.dump(dict(seed=s, l2re=v, stop_reason="budget"), f)
    for t, s in partial:
        with open(os.path.join(d, f"{pde}_{t}_seed{s}.json"), "w") as f:
            json.dump(dict(seed=s, l2re=0.5, stop_reason=None, partial=True), f)


def test_replay(tmp):
    pde = "ns2d_liddriven"
    rows = os.path.join(tmp, "rows")
    os.makedirs(rows)
    res = synth({"t3m_lead_g": -1.4, "t3m_a_g": -1.4, "t3m_bad_g": -0.4, "t3m_c_g": -1.2}, 0.1)
    res["t3m_c_g"] = {s: v for s, v in res["t3m_c_g"].items() if s <= 46}          # 5 сидов
    write_rows(rows, pde, res, partial=[("t3m_c_g", 47), ("t3m_a_g", 52)])
    # строка со смоук-сидом вне 42..51 и чужое уравнение не мешают
    write_rows(rows, "poissonboltzmann2d", {"t3m_lead_g": {42: 0.01}})
    assert SR.load_rows(rows, pde)["t3m_c_g"].keys() == set(range(42, 47)), "незаконченная строка прочитана"
    q = dict(jobs=[dict(id="j33", wave=33, steps=[dict(script="x", args="--tag t3m_lead_g"),
                                                  dict(script="x", args="--tag t3m_a_g --seeds 42")]),
                   dict(id="j41", wave=41, steps=[dict(script="x", args="--tag t3m_bad_g"),
                                                  dict(script="x", args="--tag t3m_c_g")]),
                   dict(id="j99", wave=99, steps=[dict(script="x", args="--tag t3m_zzz_g")])])
    qp = os.path.join(tmp, "q.json")
    with open(qp, "w") as f:
        json.dump(q, f)
    saved = dict(SR.QUEUES), dict(SR.LEADER)
    SR.QUEUES[pde] = qp
    SR.LEADER[pde] = "t3m_lead_g"
    try:
        assert SR.wave_tags(pde) == ["t3m_a_g", "t3m_bad_g", "t3m_c_g", "t3m_lead_g"]
        out = SR.replay(rows, pdes=(pde,), bounds=(0.5,), alpha=0.1, log=lambda *a: None)
    finally:
        SR.QUEUES.clear(); SR.QUEUES.update(saved[0])
        SR.LEADER.clear(); SR.LEADER.update(saved[1])
    tot = out["totals"][0]
    assert tot["avail"] == 10 + 10 + 5 and tot["used"] + tot["saved"] == tot["avail"], tot
    for r in out["waves"]:
        assert all(u <= a for u, a in r["per_tag"].values()) and r["used"] + r["saved"] == r["avail"]
        assert r["per_tag"]["t3m_lead_g"][0] <= 10                      # контроль в счёт не идёт
    w41 = [r for r in out["waves"] if r["wave"] == 41][0]
    dec = {(d["better"], d["worse"]) for d in w41["decisions"]}
    assert ("t3m_lead_g", "t3m_bad_g") in dec and all(d["med_ok"] for d in w41["decisions"]), w41["decisions"]
    assert w41["per_tag"]["t3m_bad_g"][0] < 10, "плохой агент должен остановиться раньше 10 сидов"
    assert tot["mismatch"] == 0 and tot["saved"] > 0
    pr = {p["tag"]: p for p in out["pairs"]}
    assert pr["t3m_bad_g"]["better"] == "t3m_lead_g" and pr["t3m_bad_g"]["seeds"] < 10
    assert pr["t3m_c_g"]["avail"] == 5 and pr["t3m_c_g"]["used"] <= 5
    assert out["cas"][0]["exact_ok"] and out["cas"][0]["queries"] <= out["cas"][0]["full"] == 35
    # командная строка: строки и JSON дают одно и то же
    jp = os.path.join(tmp, "res.json")
    with open(jp, "w") as f:
        json.dump({t: {str(s): v for s, v in r.items()} for t, r in res.items()}, f)
    with QUIET(io.StringIO()):
        a = SR.main(["--json", jp, "--alpha", "0.1"])
        b = SR.main(["--rows", rows, "--pde", "ns2d", "--tags", ",".join(sorted(res))])
    assert a["summary"]["better"] == b["summary"]["better"] and a["summary"]["seeds"] == b["summary"]["seeds"]
    print("воспроизведение волн: незаконченные строки, учёт сидов, контроль, сверка с медианами, CLI: OK")


# --------------------------------------------------------------------------------------------
# select_agents

def make_chain_file(rng, e0, n_chains, length):
    """Файл собранного буфера: цепочки с картами (первое состояние — нулевые карты), конец — done=1."""
    out = []
    for _ in range(n_chains):
        while True:
            acts = [int(a) for a in rng.integers(0, 27, length)]
            if S.EPOCHS[acts].sum() < 7000:
                break
        l = math.log10(e0)
        maps = [{k: np.zeros((26, 26), np.float32) for k in KEYS}]
        errs = []
        for _a in acts:
            l += rng.normal(-0.2, 0.1)
            errs.append(10 ** l)
            maps.append({k: (rng.random((26, 26)) + 1.0).astype(np.float32) for k in KEYS})
        for j, a in enumerate(acts):
            out.append(dict(state=dict(maps[j]), next_state=dict(maps[j + 1]),
                            action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                            reward=-errs[j], done=int(j == length - 1)))
    return out


def make_agent(path, seed, in_ch, encoder, two=False):
    torch.manual_seed(seed)
    kw = dict(two_nets=True) if two else {}
    net = O.QNet("convnext_dqn", CPU, in_ch=in_ch, encoder=encoder, **kw)
    meta = dict(encoder=net.enc_kind, ens=1, state_mode="full", scalar_ctx=in_ch > 4, ctx_kmax=70, ctx_budget=7000,
                ctx_err="none", in_ch=in_ch)
    if two:
        meta.update(two_nets=True, two_nets_center="max", two_nets_head="linear", two_nets_mode="additive",
                    two_nets_shared=False)
    rng = np.random.default_rng(seed)
    torch.save({"variant": "convnext_dqn", "seed": 1,
                "state_dict": {k: v.cpu() for k, v in net.model.state_dict().items()},
                "mean": rng.normal(0, 0.1, (1, in_ch, 1, 1)).astype(np.float32),
                "std": (1.0 + rng.random((1, in_ch, 1, 1))).astype(np.float32), "meta": meta}, path)
    return path


class StepEns:
    """Подставной ансамбль: изменение log10 ошибки за действие = base[a] + 0.01 * m (член m)."""

    def __init__(self, base, members=3):
        self.base, self.members = np.asarray(base, dtype=np.float64), members

    def predict_ln(self, X, l, member=None):
        a = X[:, 0].astype(np.int64)
        out = np.stack([np.clip(l + self.base[a] + 0.01 * m, S.L_LO, S.L_HI) for m in range(self.members)])
        return out if member is None else out[member]


def test_select(tmp):
    import mb_targets as MB
    meta = O.pde_meta()
    e0 = float(meta[PDE]["init_err"])
    rng = np.random.default_rng(0)
    bdir = os.path.join(tmp, "tbuf", SUB)
    os.makedirs(bdir)
    for k in range(3):
        torch.save(make_chain_file(rng, e0, 2, 4), os.path.join(bdir, f"c{k}.pt"))
    with QUIET(io.StringIO()):
        target = SA.load_target(os.path.join(tmp, "tbuf"), PDE)
    episodes, data, info = target
    n = len(data["A"])
    assert n == 24 and info["files"] == 3 and data["R"].shape == (n,)
    # награда dlog: сумма по цепочке = log10 E0_тренера - log10 итога
    e0t = SA.init_err(PDE)
    for ep in np.unique(data["EP"]):
        i = np.where(data["EP"] == ep)[0]
        assert abs(data["R"][i].sum() - (math.log10(e0t) - math.log10(data["ERR"][i[-1]]))) < 1e-4
    a1 = make_agent(os.path.join(tmp, "a1.pt"), 1, 9, "resnet")
    a2 = make_agent(os.path.join(tmp, "a2.pt"), 2, 4, "convnext")
    a3 = make_agent(os.path.join(tmp, "a3.pt"), 3, 9, "resnet", two=True)
    # вход агента и жадное действие — тот же путь, что offline_rl.teacher_q
    for p in (a1, a2, a3):
        ag = SA.load_agent(p, CPU)
        X, X2 = SA.agent_inputs(ag, data, PDE)
        q = SA.q_values(ag, X)
        qt, _ = O.teacher_q(p, data, CPU)
        assert X.shape[1] == ag["in_ch"] and np.allclose(q, qt, atol=1e-5), p
        if ag["in_ch"] > 4:     # контекст s' — следующий шаг: шаг +1
            c1, c2 = O.chain_ctx(data, k_max=70, budget=7000)
            assert np.allclose(X2[:, 4:, 0, 0], c2) and np.allclose(X[:, 4:, 0, 0], c1)
    # суррогат: постоянный шаг -0.001 у всех действий — итог считается вручную
    tab = dict(CTX=S.pde_ctx(meta[PDE])[None], NAMES=[PDE])
    st, valid, _ = MB.chain_states(data, tab, np.array([0]), np.ones(1), budget=7000)
    rows = np.where(valid)[0]
    ens0 = StepEns(np.full(27, -0.001), members=2)
    acts = data["A"][rows]
    F, _ = SA.surrogate_final(ens0, tab, st, rows, acts, budget=7000)
    for j, r in enumerate(rows):
        spent = int(data["SPENT"][r]) + int(S.EPOCHS[acts[j]])
        k = 1 + (max(0, 7000 - spent) + 99) // 100          # дальше — Adam 1e-2 на 100 эпох до бюджета
        for m in range(2):
            want = st["l"][r] + k * (-0.001 + 0.01 * m) if m == 0 else st["l"][r] + 0.009     # член 1: шаг +0.009, стоп
            assert abs(F[m, j] - want) < 1e-9, (m, j, F[m, j], want)
    # продолжение фиксированным действием (L-BFGS 1.0 на 500 эпох до бюджета): шаги идут и при ухудшении
    ci = SA.parse_cont("LBFGS:1:500")
    assert S.ACTIONS[ci] == ("LBFGS", 1.0, 500) and SA.parse_cont("greedy") is None
    for bad_spec in ("LBFGS:1:700", "nope"):
        try:
            SA.parse_cont(bad_spec)
        except ValueError:
            pass
        else:
            raise AssertionError(bad_spec)
    F2, _ = SA.surrogate_final(ens0, tab, st, rows, acts, budget=7000, cont=ci)
    for j, r in enumerate(rows):
        spent = int(data["SPENT"][r]) + int(S.EPOCHS[acts[j]])
        k = 1 + (max(0, 7000 - spent) + 499) // 500
        for m in range(2):
            assert abs(F2[m, j] - (st["l"][r] + k * (-0.001 + 0.01 * m))) < 1e-9, (m, j)
    # «лучшие» и «худшие» действия по самому суррогату: оценка упорядочивает их
    base = np.linspace(-0.08, 0.02, 27)
    ens = StepEns(base, members=3)
    Fall = np.stack([SA.surrogate_final(ens, tab, st, rows, np.full(len(rows), a), budget=7000)[0] for a in range(27)])
    good, bad = Fall.mean(1).argmin(0), Fall.mean(1).argmax(0)
    sc, inf = SA.surrogate_scores(ens, tab, st, rows, {"good": good, "bad": bad, "logged": acts}, acts)
    assert sc["good"]["surr_score"] < sc["logged"]["surr_score"] < sc["bad"]["surr_score"]
    assert abs(sum(v["surr_p_best"] for v in sc.values()) - 1.0) < 1e-12 and sc["bad"]["surr_p_best"] == 0.0
    assert abs(sc["logged"]["surr_vs_logged"]) < 1e-12 and sc["good"]["surr_vs_logged"] < 0 < sc["bad"]["surr_vs_logged"]
    assert inf["members"] == 3 and len(sc["good"]["surr_members"]) == 3
    # кодировщик MAE в формате зонда CV
    enc = O.make_encoder("resnet", in_ch=4)
    mae = os.path.join(tmp, "cvprobe_t_enc_mae_resnet.pt")
    torch.save(dict(kind="ssl_encoder", encoder="resnet", ssl="mae", in_ch=4, size=26,
                    state_dict={k: v.clone() for k, v in enc.state_dict().items()},
                    mean=np.full((1, 4, 1, 1), 0.5, np.float32), std=np.full((1, 4, 1, 1), 2.0, np.float32)), mae)
    w0 = {k: v.clone() for k, v in enc.state_dict().items()}
    out = os.path.join(tmp, "sel.json")
    args = SA.build_parser().parse_args(
        ["--pde", PDE, "--models", f"t3m_a1_g={a1},t3m_a2_g={a2}", "--extra-models",
         f"t3m_a3_g={a3},t3m_gone_g={os.path.join(tmp, 'gone.pt')}", "--method", "both", "--tag", "test",
         "--encoder", mae, "--fqe-epochs", "2", "--fqe-seeds", "2", "--batch-size", "8", "--cpu", "--no-upload",
         "--out", out, "--target-buffer", os.path.join(tmp, "tbuf")])
    mk = O.make_encoder
    with QUIET(io.StringIO()):
        row = SA.run(args, model=(tab, ens), target=target)
    assert O.make_encoder is mk, "фабрика энкодера offline_rl не возвращена"
    ag = row["agents"]
    assert set(ag) == {"t3m_a1_g", "t3m_a2_g", "t3m_a3_g"} and [m["label"] for m in row["missing"]] == ["t3m_gone_g"]
    assert ag["t3m_a1_g"]["main"] and not ag["t3m_a3_g"]["main"] and ag["t3m_a3_g"]["two_nets"]
    for key in ("rank_surrogate", "rank_fqe"):
        assert sorted(a[key] for a in ag.values()) == [1, 2, 3], key
    for a in ag.values():
        assert np.isfinite(a["surr_score"]) and np.isfinite(a["fqe_pred_log"]) and len(a["fqe_values"]) == 2
        assert abs(a["fqe_pred_log"] - (math.log10(e0t) - a["fqe_value"])) < 1e-9
    with open(out) as f:
        assert json.load(f)["agents"].keys() == ag.keys()
    # замороженный кодировщик: веса в файле и после загрузки не изменились, BatchNorm в режиме оценки
    enc2, em, es = SA.load_mae(mae, CPU)
    assert all(torch.equal(enc2.state_dict()[k], w0[k]) for k in w0) and not enc2.training
    ff = SA.frozen_features_cls()(enc2, em, es, np.zeros((1, 9, 1, 1), np.float32), np.ones((1, 9, 1, 1), np.float32), 5)
    ff.train()
    assert not ff.enc.training and ff.mlp.training
    x = torch.randn(4, 9, 26, 26)
    with torch.no_grad():
        f1 = ff.enc((x[:, :4] - torch.as_tensor(em)) / torch.as_tensor(es))
    assert ff(x).shape == (4, 256) and torch.equal(f1, ff.enc((x[:, :4] - torch.as_tensor(em)) / torch.as_tensor(es)))
    # разбор аргументов и отказ при одном агенте; проверка очереди
    import lint_queue as L
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            SA.main(["--pde", PDE, "--models", f"x={a1}"])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("один агент должен отвергаться")
    assert L.check("experiments/rl_arch/select_agents.py",
                   f"--pde {PDE} --models a={a1},b={a2} --method both --tag t") is None
    assert "отказ" in (L.check("experiments/rl_arch/select_agents.py", f"--pde {PDE} --models a={a1} --tag t") or "")
    assert "отказ" in (L.check("experiments/rl_arch/select_agents.py",
                               f"--pde {PDE} --models a={a1},b={a2} --method nope") or "")
    print("select_agents: вход агентов как у teacher_q, суррогат (ручной расчёт и порядок), FQE на замороженном "
          "MAE, JSON, отсутствующий файл, разбор аргументов и проверка очереди: OK")
    return row


def test_report(tmp, row):
    sel = os.path.join(tmp, "seljson")
    rows = os.path.join(tmp, "selrows")
    os.makedirs(sel)
    os.makedirs(rows)
    row = json.loads(json.dumps(row))
    # подставные оценки: суррогат угадывает порядок, FQE — обратный
    med = {"t3m_a1_g": 0.010, "t3m_a2_g": 0.020, "t3m_a3_g": 0.040}
    for i, (x, m) in enumerate(med.items()):
        row["agents"][x].update(surr_score=math.log10(m), fqe_pred_log=-math.log10(m))
    with open(os.path.join(sel, f"{PDE}_test.json"), "w") as f:
        json.dump(row, f)
    write_rows(rows, PDE, {x: {s: m * (1 + 0.01 * (s - 44)) for s in range(42, 47)} for x, m in med.items()})
    with QUIET(io.StringIO()):
        out = REP.main(["--tag", "test", "--pdes", PDE, "--sel", sel, "--rows", rows])
    by = {(r["method"], r["set"]): r for r in out}
    s, f = by[("surrogate", "all")], by[("fqe", "all")]
    assert s["n"] == 3 and abs(s["spearman"] - 1) < 1e-12 and abs(s["kendall"] - 1) < 1e-12 and s["regret"] == 0
    assert abs(f["spearman"] + 1) < 1e-12 and f["chosen"] == "t3m_a3_g" and abs(f["regret"] - 0.03) < 1e-9
    assert by[("surrogate", "main")]["n"] == 2 and f["overlap_top3"] == 3
    assert abs(REP.kendall([1, 2, 2, 3], [1, 2, 3, 3]) - 0.8) < 1e-9
    print("select_report: корреляции, сожаление, основные и все агенты: OK")


def main():
    t0 = time.time()
    test_sequential()
    test_cas()
    with tempfile.TemporaryDirectory() as tmp:
        test_replay(tmp)
        row = test_select(tmp)
        test_report(tmp, row)
    print(f"test_select_tools: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
