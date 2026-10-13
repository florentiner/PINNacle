#!/usr/bin/env python
"""
Проверки прогонов политики в буфер (online_eval_env.py --save-buffer, --explore-eps, --explore-stall;
запись файла — target_buffer.py, общая с collect_active.py). Без GPU и сети: исполняется НАСТОЯЩИЙ
run_seed (цикл действий, страж, разведка, чекпоинт докатки), подменены только PINN, обучение и
построение карт — модель с одним весом w (лосс = exp(w)), действие сдвигает w по таблице EFFECT
(Adam с шагом 1e-2 лосс поднимает — его откатывает страж, PSO не меняет — застой), карта однозначно
задана весом и числом обученных эпох. Агент — расписание PLAN по номеру шага из скалярного контекста.

  * --save-buffer: один файл на сид, читается target_buffer.load + episodes_to_arrays_chains и
    surrogate.build_table; состояния — те карты, что построила среда; сводка load печатает среду
    сбора и бюджет; файл и итоговая строка — один коммит;
  * страж: в файле только оставленные действия (вместо откаченного — запасное), откаченное — в поле
    rolled, эпохи — spent и guard_spent; SPENT загрузчика = spent − epochs_real − guard_spent;
  * без новых флагов прогон и строка прежние; --explore-eps 0 — тот же жадный прогон;
  * разведка меняет действия только на шагах из explore_log; после застоя вероятность — --explore-stall;
  * докатка: карты и поток разведки едут в чекпоинте (файл совпадает с непрерывным прогоном);
    чекпоинт без записи прогона — сид не записан (buffer.saved=false);
  * смешанная спецификация (файлы сбора + файлы прогонов) проходит учителя agent_combo.py и шаг
    обучения offline_rl.py.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_rollout_buffer.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.pop("HF_TOKEN", None)            # тест ничего не заливает
os.environ.pop("HF_TOKEN_WRITE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_collect_buffer as TCB  # noqa: E402
import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import mb_targets as MB  # noqa: E402
import target_buffer as TB  # noqa: E402

torch.set_num_threads(1)
E, TS = TCB.E, TCB.TS
PDE, SUB, SUB2 = TCB.PDE, TCB.SUB, TCB.SUB2
KEYS = list(TB.MAP_KEYS)
IDX = E.parse_action_spec
FB = IDX("LBFGS:1:500")
# расписание агента по шагам (по кругу): хорошее, плохое (откат), пустое (застой), ...
PLAN = [IDX(x) for x in ("LBFGS:1:1000", "Adam:0.01:1000", "PSO:0:300", "LBFGS:0.5:500", "Adam:0.001:1000",
                         "Adam:0.01:100")]
# что страж оставит от жадного прогона на бюджете 7000: [действие, фактические эпохи]
KEPT = [(PLAN[0], 1000), (FB, 500), (PLAN[2], 300), (PLAN[3], 500), (PLAN[4], 1000), (FB, 500), (PLAN[0], 1000),
        (FB, 100)]
ROLLED = {1: (PLAN[1], 1000), 5: (PLAN[5], 100), 7: (PLAN[1], 1000)}       # шаг с нуля -> откаченное
BUILT, ROWS, UPS = [], [], []
REAL = (E.load_agent, E.pick_action, E.q_values)


def effect(name, lr, it):
    """Сдвиг веса w за it эпох: Adam 1e-2 лосс поднимает, PSO не меняет, остальные снижают."""
    if name == "PSO":
        return 0.0
    if name == "Adam":
        return {1e-2: 0.3, 1e-3: -0.3, 1e-4: -0.05}[lr] * it / 1000.0
    return -0.2 * lr * it / 1000.0


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1) * 0.01)      # зависит от сида (set_random_seed)
        self.register_buffer("t", torch.zeros(1))               # обученные эпохи: едут в снимке весов


class Model:
    def __init__(self):
        self.net, self.data, self.opt = Net(), None, None
        self.train_state = types.SimpleNamespace(loss_train=np.array([0.7, 0.3]), X_train=None)
        self.losshistory = types.SimpleNamespace(steps=[], loss_train=[])

    def compile(self, optimizer, loss_weights=None):
        self.opt = optimizer

    def train(self, iterations, display_every, callbacks, model_save_path, save_model):
        name = "PSO" if isinstance(self.opt, str) else type(self.opt).__name__
        lr = None if name == "PSO" else self.opt.param_groups[0]["lr"]
        with torch.no_grad():
            self.net.w += effect(name, lr, iterations)
            self.net.t += iterations
        lv = float(torch.exp(self.net.w.detach()))
        self.train_state.loss_train = np.array([0.7 * lv, 0.3 * lv])
        tester, saver = callbacks
        tester.rmse, tester.brmse, tester.l2re, tester.bc_l2re = 0.4 * lv, 0.1 * lv, lv, 0.0
        saver.saved_models = (float(self.net.w.detach()), float(self.net.t))


class _Obj:
    def __init__(self, *a, **k):
        self.kw = k
        self.saved_models = None

    def train(self, *a, **k):
        return self


class Surface(_Obj):
    def save_equation_loss_surface(self, log_key=True):
        w, t = self.kw["solver_models"]
        m = np.random.default_rng([int(t), int(round((w + 10.0) * 1e5))]).random((3, 26, 26)).astype(np.float32) + 1
        return {k: torch.tensor(m[i]) for i, k in enumerate(E.LOSS_TYPES)}


class Agent:
    _meta = dict(in_ch=9, ctx_kmax=70, ctx_budget=7000, ctx_err="none")


def plan_action(agent, state, mean, std, variant):
    """Выбор агента: расписание по номеру шага (канал контекста 4 = шаг / ctx_kmax)."""
    return PLAN[int(round(float(state[4, 0, 0]) * 70)) % len(PLAN)]


def plan_q(agent, state, mean, std, variant):
    q = np.zeros(27, dtype=np.float32)
    q[plan_action(agent, state, mean, std, variant)] = 1.0
    return q


FAKES = {
    "experiments.chain_eval.pde_registry": dict(build_get_model=lambda *a, **k: (lambda: (Model(), None))),
    "src.utils.callbacks": dict(TesterCallback=_Obj, ModelSaverCallback=_Obj),
    "landscape_visualization._aux.visualization_model": dict(VisualizationModel=_Obj),
    "landscape_visualization._aux.plot_loss_surface": dict(PlotLossSurface=Surface),
    "landscape_visualization._aux.early_stopping_plot": dict(EarlyStopping=_Obj),
}


@contextlib.contextmanager
def patched():
    saved = (E.load_agent, E.pick_action, E.q_values, E.build_state, E.run_seed, E.upload, E.result_done,
             TB.upload_files, sys.argv, E.gc)
    E.gc = types.SimpleNamespace(collect=lambda: 0)     # сборка мусора после каждого шага — секунды на тест
    mods = {k: sys.modules.get(k) for k in FAKES}
    tmp, cwd = tempfile.mkdtemp(prefix="rollout_buf_"), os.getcwd()
    for k, v in FAKES.items():
        sys.modules[k] = types.SimpleNamespace(**v)

    def build_state(raw, prev_raw):
        BUILT.append(saved[3](raw, prev_raw))
        return BUILT[-1]

    def run_seed(seed, args, progress_cb=None):
        ROWS.append(saved[4](seed, args, progress_cb=progress_cb))
        return ROWS[-1]

    E.load_agent = lambda f: (Agent(), None, None, "convnext_dqn")
    E.pick_action, E.q_values, E.build_state, E.run_seed = plan_action, plan_q, build_state, run_seed
    E.upload = lambda row, name: UPS.append(("row", name, dict(row)))
    E.result_done = lambda name: False
    TB.upload_files = lambda pairs, msg, tries=9: (UPS.append(("files", list(pairs), msg)), True)[1]
    os.chdir(tmp)                       # чекпоинты среды пишутся в текущую папку
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        (E.load_agent, E.pick_action, E.q_values, E.build_state, E.run_seed, E.upload, E.result_done,
         TB.upload_files, sys.argv, E.gc) = saved
        for k, v in mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def rollout(tmp, tag, extra, seeds="1", guard=True):
    """Один вызов online_eval_env.main с боевыми флагами развёртывания; (строки, карты среды, заливки)."""
    del BUILT[:], ROWS[:], UPS[:]
    sys.argv = (["online_eval_env.py", "--policy", "agent", "--model-file", "fake.pt", "--keep-opt",
                 "--keep-opt-mode", "safe", "--pde", PDE, "--budget", "7000", "--seeds", seeds, "--tag", tag,
                 "--save-dir", tmp, "--ae-epochs", "1"]
                + (["--guard-rollback", "1.0", "--guard-fallback", "LBFGS:1:500"] if guard else []) + list(extra))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        E.main()
    return [dict(r) for r in ROWS], [b.copy() for b in BUILT], list(UPS), out.getvalue()


def file_of(tmp, buf, tag, seed):
    p = os.path.join(tmp, "buffers_new", buf, SUB, f"{tag}_seed{seed}.pt")
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def act(t):
    return t["action"][0] * 9 + t["action"][1]["lr"] * 3 + t["action"][1]["epochs"]


def core(row):
    """То, что в строке не зависит от часов: цепочка, ошибки, страж, история без времени."""
    return (row["chain"], row["l2re"], row["rmse"], row["spent"], row["guard_log"], row["guard_spent"],
            [h[:3] for h in row["hist"]], row["stop_reason"])


def same_file(x, y):
    return len(x) == len(y) and all(
        act(a) == act(b) and a["reward"] == b["reward"] and a["epochs_real"] == b["epochs_real"]
        and a["done"] == b["done"] and a["spent"] == b["spent"] and a["explore"] == b["explore"]
        and a["rolled"] == b["rolled"] and all(np.array_equal(a[s][k], b[s][k]) for s in ("state", "next_state")
                                               for k in KEYS) for a, b in zip(x, y))


# --------------------------------------------------------------------------------------------

def test_guard_file(tmp):
    """Жадный прогон со стражем: файл на сид, в нём только оставленные действия; загрузчики."""
    rows, built, ups, text = rollout(tmp, "rt", ["--save-buffer", "rb"], seeds="1,2")
    assert len(rows) == 2 and sorted(os.listdir(os.path.join(tmp, "buffers_new", "rb", SUB))) == [
        "rt_seed1.pt", "rt_seed2.pt"], "один файл на сид"
    n = len(KEPT)
    per_seed = len(built) // 2
    assert per_seed == n - 1, "карта после каждого оставленного действия, кроме выбравшего бюджет"
    for k, row in enumerate(rows):
        seed = row["seed"]
        tr = file_of(tmp, "rb", "rt", seed)
        maps = built[k * per_seed:(k + 1) * per_seed]
        assert [(E.ACTION_TABLE[a][0], E.ACTION_TABLE[a][1], ep) for a, ep in KEPT] == [tuple(c) for c in row["chain"]]
        assert [g[:2] for g in row["guard_log"]] == [[i + 1, a] for i, (a, _) in ROLLED.items()]
        assert row["guard_spent"] == 2100 and row["spent"] == 7000 and row["stop_reason"] == "budget"
        assert [(act(t), t["epochs_real"]) for t in tr] == KEPT, "в файле — исполненные и оставленные действия"
        assert row["buffer"] == dict(tag="rb", saved=True, why=None, n=n, file=f"rt_seed{seed}.pt",
                                     hf=f"rl_arch/buffers_new/rb/{SUB}/rt_seed{seed}.pt", n_explore=0, n_rolled=3,
                                     up=True)
        gs = 0
        for i, t in enumerate(tr):
            want = np.zeros((3, 26, 26), np.float32) if i == 0 else maps[i - 1][:3]
            assert np.array_equal(np.stack([t["state"][k] for k in KEYS]), want), "состояние перед действием"
            assert all(t["state"][k].dtype == np.float32 and t["state"][k].shape == (26, 26) for k in KEYS)
            if i < n - 1:
                assert t["done"] == 0 and np.array_equal(np.stack([t["next_state"][k] for k in KEYS]), maps[i][:3])
            rb = ROLLED.get(i)
            assert [r[:2] for r in t["rolled"]] == ([list(rb)] if rb else []) and not t["explore"]
            if rb:      # откат: лосс после выше лосса до, запасное действие — из того же состояния
                assert t["rolled"][0][3] > t["rolled"][0][2] and act(t) == FB
            gs += rb[1] if rb else 0
            assert t["guard_spent"] == gs and t["spent"] == row["hist"][i][0]
            assert abs(t["reward"] + row["hist"][i][2] * 0.5) < 1e-12, "награда = −(rmse+brmse) после действия"
            assert t["chain"] == f"rb#rt_seed{seed}" and t["group"] is None
        assert tr[-1]["done"] == 1 and tr[-1]["spent"] == 7000 and tr[-1]["guard_spent"] == 2100
        m = tr[0]["meta"]
        assert m["env"] == dict(keep_opt=True, mode="safe", budget=7000) and m["source"] == "rollout" == m["acq"]
        assert m["seed"] == seed and m["policy"] == "agent" and m["model"] == "fake.pt" and m["pde"] == PDE
        assert m["guard"] == dict(rollback=1.0, fallback="LBFGS:1:500", spent=2100, n=3) and m["explore"]["steps"] == []
    assert not same_file(file_of(tmp, "rb", "rt", 1), file_of(tmp, "rb", "rt", 2)), "сиды различаются"
    # заливка: файл и итоговая строка сида — один коммит, отдельной заливки строки нет
    assert [u[0] for u in ups] == ["files", "files"]
    for u, row in zip(ups, rows):
        (src, dst), (js, jdst) = u[1]
        assert dst == row["buffer"]["hf"] and os.path.exists(src) and jdst == f"rl_arch/online_env/{PDE}_rt_seed{row['seed']}.json"
        sent = json.loads(js)
        assert sent["buffer"]["up"] and sent["chain"] == row["chain"] and "maps" not in js.decode() and "path" not in sent["buffer"]
    assert "НЕ записан" not in text and text.count("буфер rb: rt_seed") == 2

    # загрузчики офлайновых буферов без изменений
    spec = os.path.join(tmp, "buffers_new", "rb")
    log = []
    eps, info = TB.load(spec, SUB, log=lambda *x: log.append(" ".join(map(str, x))))
    assert info["files"] == 2 == info["chains"] and info["transitions"] == 2 * n
    assert info["envs"] == {"keep_safe": 2} and info["budgets"] == {"7000": 2}, "среда сбора и бюджет в сводке"
    assert "среда сбора {'keep_safe': 2}, бюджет {'7000': 2}" in log[0]
    e0 = float(O.pde_meta()[PDE]["init_err"])
    data = O.episodes_to_arrays_chains(eps, reward_form="delta", init_err=e0, budget=7000, verbose=False)
    stt = data["STATS"]
    assert stt["chains"] == 2 == stt["ok"] and stt["copy_files"] == 0 and stt["dropped_tail"] == 0 and stt["tail"] == 0
    assert stt.get("budget_cut", 0) == 0 and stt["diverged"] == 0 and len(data["A"]) == 2 * n
    for c, ep in enumerate(eps):        # порядок файлов у загрузчика свой: сид — из meta
        row = [r for r in rows if r["seed"] == ep[0]["meta"]["seed"]][0]
        maps = built[rows.index(row) * per_seed:(rows.index(row) + 1) * per_seed]
        idx = np.where(data["EP"] == c)[0]
        assert data["A"][idx].tolist() == [a for a, _ in KEPT] and (data["STEP"][idx] == np.arange(n)).all()
        assert data["D"][idx].tolist() == [0.0] * (n - 1) + [1.0], "цепочка терминальна на последнем переходе"
        assert not data["S"][idx[0]].any() and data["FIRST"][idx[0]] == 1
        for j in range(1, n):           # состояние загрузчика = состояние, которое видел агент в прогоне
            assert np.allclose(data["S"][idx[j]], maps[j - 1], atol=1e-6), "состояние буфера = состояние прогона"
            assert np.array_equal(data["S2"][idx[j - 1]], data["S"][idx[j]])
        # счёт эпох: SPENT загрузчика — сумма номинальных эпох оставленных действий
        want = [t["spent"] - t["epochs_real"] - t["guard_spent"] for t in ep]
        assert data["SPENT"][idx].tolist() == want == [0, 1000, 1500, 1800, 2300, 3300, 3800, 4800]
        assert np.allclose(data["ERR"][idx], [h[2] * 0.5 for h in row["hist"]], rtol=1e-6)
    tab = S.build_table(episodes={SUB: eps}, verbose=False)
    assert len(tab["A"]) == 2 * n and tab["A"].tolist() == data["A"].tolist() and int(tab["LAST"].sum()) == 2
    assert len(np.unique(tab["CH"])) == 2 and tab["NAMES"] == [PDE] and tab["SPENT"].tolist() == data["SPENT"].tolist()
    new = TB.target_rows(eps, PDE)
    cut = np.where(new["EPA"] != S.EPOCHS[new["A"]])[0]
    assert len(cut) == 2 and set(new["EPA"][cut]) == {100} and set(new["A"][cut]) == {FB}, "фактические эпохи обрезанного"
    print("страж: файл на сид с оставленными действиями, откаты в rolled, один коммит, загрузчики: OK")
    return rows


def test_default_unchanged(tmp, rows_buf):
    """Без новых флагов прогон и строка прежние; --explore-eps 0 — тот же жадный прогон."""
    rows, _, ups, _ = rollout(tmp, "plain", ["--ckpt-every", "2", "--ckpt-min-interval", "0"], seeds="1,2")
    assert [core(r) for r in rows] == [core(r) for r in rows_buf], "--save-buffer не меняет сам прогон"
    new_keys = {"buffer", "explore_log", "explore_eps", "explore_stall", "stall_ratio", "n_explore",
                "explore_steps", "stall_steps"}
    assert all(not new_keys & set(r) for r in rows) and not os.path.exists(os.path.join(tmp, "buffers_new", "plain"))
    assert [u[0] for u in ups] == ["row", "row"] and all("buffer" not in u[2] for u in ups)
    ck = torch.load(f"ckpt_{PDE}_plain_seed1.pt", weights_only=False)
    assert "rollout" not in ck["t3"], "чекпоинт без флагов — прежний"
    zero, _, _, _ = rollout(tmp, "zero", ["--explore-eps", "0", "--stall-ratio", "0.5"], seeds="1,2")
    assert [core(r) for r in zero] == [core(r) for r in rows] and all(not new_keys & set(r) for r in zero)
    nog, _, _, _ = rollout(tmp, "nog", ["--save-buffer", "nog"], guard=False)
    tr = file_of(tmp, "nog", "nog", 1)
    assert nog[0]["guard_log"] == [] and [act(t) for t in tr] == [PLAN[i % len(PLAN)] for i in range(len(tr))]
    assert all(not t["rolled"] and t["guard_spent"] == 0 for t in tr), "без стража — все действия агента"
    for bad, what in ((["--save-buffer", "x", "--no-state"], "--no-state"),
                      (["--save-buffer", "x", "--state-every", "2"], "--state-every"),
                      (["--save-buffer", "a/b"], "тег"), (["--explore-eps", "1.5"], "eps > 1"),
                      (["--save-buffer", "x", "--boost-trigger", "midpoint"], "бустинг")):
        TS.must_fail(lambda b=bad: rollout(tmp, "bad", b), SystemExit, f"--save-buffer/разведка: {what}")
    sys.argv = ["online_eval_env.py", "--policy", "random", "--explore-eps", "0.1", "--tag", "x"]
    TS.must_fail(E.main, SystemExit, "разведка без агента")
    print("без флагов и с --explore-eps 0 прогон прежний, отказы на несовместимых флагах: OK")


def attempted(row, tr):
    """Что политика пыталась исполнить на каждом шаге: откаченное стражем либо оставленное."""
    return [t["rolled"][0][0] if t["rolled"] else act(t) for t in tr]


def test_explore(tmp, rows_buf):
    """Разведка меняет действия только на шагах из explore_log; после застоя — --explore-stall."""
    flags = ["--save-buffer", "ex", "--explore-eps", "0.3", "--explore-stall", "1.0", "--stall-ratio", "0.95"]
    rows, _, _, text = rollout(tmp, "ex", flags, seeds="1,2,3")
    n_ex = 0
    for row in rows:
        tr = file_of(tmp, "ex", "ex", row["seed"])
        n, log = len(tr), {e[0]: e for e in row["explore_log"]}
        assert n == row["n_steps"] and row["explore_steps"] == sorted(log) and row["n_explore"] == len(log) > 0
        att = attempted(row, tr)
        for i in range(n):
            own = PLAN[i % len(PLAN)]
            if i + 1 in log:
                step, a_own, a_rnd, p, stalled, kept = log[i + 1]
                assert a_own == own and att[i] == a_rnd and p == (1.0 if stalled else 0.3)
                assert tr[i]["explore"] == bool(kept) == (not tr[i]["rolled"])
            else:
                assert att[i] == own and not tr[i]["explore"], "вне explore_log — выбор агента"
        # застой (лосс не упал в 0.95 раза): следующее решение случайно с вероятностью 1
        losses = [None] + [h[1] for h in row["hist"]]
        stall = [i for i in range(2, n + 1) if losses[i] > 0.95 * losses[i - 1]]
        assert row["stall_steps"] == stall and stall
        assert all(log[s + 1][4] == 1 and log[s + 1][3] == 1.0 for s in stall if s < n)
        assert all(e[4] == 0 for e in row["explore_log"] if e[0] - 1 not in stall)
        assert tr[0]["meta"]["explore"] == dict(eps=0.3, stall=1.0, ratio=0.95, steps=row["explore_steps"])
        assert row["buffer"]["n_explore"] == sum(t["explore"] for t in tr)
        n_ex += len(log)
    assert text.count("разведка (p=") == n_ex
    again, _, _, _ = rollout(tmp, "ex2", flags, seeds="1,2,3")
    assert [(core(r), r["explore_log"]) for r in again] == [(core(r), r["explore_log"]) for r in rows], "тот же сид — та же разведка"
    assert [core(r) for r in rows[:2]] != [core(r) for r in rows_buf]
    # только --explore-stall: без застоя агент жадный, генератор не тратится
    st, _, _, _ = rollout(tmp, "st", ["--explore-stall", "1.0", "--stall-ratio", "1e-9"])
    assert st[0]["explore_steps"] == list(range(3, st[0]["n_steps"] + 1)), "любой шаг — застой: случайны все после второго"
    st0, _, _, _ = rollout(tmp, "st0", ["--explore-stall", "1.0", "--stall-ratio", "1e9"])
    assert st0[0]["explore_log"] == [] and core(st0[0]) == core(rows_buf[0])
    print("разведка: действия меняются только на шагах explore_log, застой включает --explore-stall: OK")
    return flags, rows


def test_resume(tmp, flags, rows_full):
    """Докатка: карты и поток разведки едут в чекпоинте; чекпоинт без записи — сид не записан."""
    ref = file_of(tmp, "ex", "ex", 1)
    part, _, ups, _ = rollout(tmp, "res", [f.replace("ex", "res") if f == "ex" else f for f in flags] + ["--hours", "1e-9"])
    assert part[0].get("unfinished") and "buffer" not in part[0] and not ups
    assert file_of(tmp, "res", "res", 1) is None, "незаконченный сид файла не пишет"
    ck = torch.load(f"ckpt_{PDE}_res_seed1.pt", weights_only=False)["t3"]["rollout"]
    assert len(ck["steps"]) == 1 and sorted(ck["maps"]) == [1] and ck["rng"] is not None
    # вторая остановка и докатка до конца: три кернела на сид
    rollout(tmp, "res", [f.replace("ex", "res") if f == "ex" else f for f in flags] + ["--hours", "1e-9", "--resume"])
    done, _, ups, _ = rollout(tmp, "res", [f.replace("ex", "res") if f == "ex" else f for f in flags] + ["--resume"])
    tr = file_of(tmp, "res", "res", 1)
    assert same_file(tr, ref), "файл после докатки совпадает с непрерывным прогоном"
    assert done[0]["explore_log"] == rows_full[0]["explore_log"] and core(done[0]) == core(rows_full[0])
    assert done[0]["buffer"]["saved"] and [u[0] for u in ups] == ["files"]
    # чекпоинт писал прогон без --save-buffer: карт прошлых шагов нет — сид помечен, файла нет
    rollout(tmp, "old", ["--hours", "1e-9"])
    done, _, ups, text = rollout(tmp, "old", ["--save-buffer", "old", "--resume"])
    b = done[0]["buffer"]
    assert b["saved"] is False and "докатка" in b["why"] and b["n"] == 0 and "up" not in b
    assert file_of(tmp, "old", "old", 1) is None and [u[0] for u in ups] == ["row"] and "НЕ записан" in text
    assert ups[0][2]["buffer"]["saved"] is False
    print("докатка: файл как у непрерывного прогона; чекпоинт без карт — сид не записан: OK")


def test_merged_training(tmp):
    """Спецификация «файлы сбора + файлы прогонов»: учитель agent_combo.py и шаг обучения offline_rl.py."""
    roll = os.path.join(tmp, "buffers_new", "ex")
    # в буфере ex — два прогона (теги ex и ex2) по три сида: имена файлов несут тег прогона
    assert len(os.listdir(os.path.join(roll, SUB))) == 6
    n_roll = sum(len(file_of(tmp, "ex", t, s)) for t in ("ex", "ex2") for s in (1, 2, 3))
    here = os.getcwd()
    with TCB.patched() as ctmp:         # сборщик с подменённой средой (test_collect_buffer)
        TCB.collect(ctmp, "w0", ["--rounds", "1", "--batch", "3", "--keep-opt", "--keep-opt-mode", "safe"])
        coll = os.path.join(tmp, "coll")
        shutil.copytree(os.path.join(ctmp, "buffers_new", "w0"), coll)
    os.chdir(here)
    n_coll = sum(len(e) for e in O.load_episodes(os.path.join(coll, SUB), SUB))
    spec = f"{coll},{roll}"
    eps, info = TB.load(spec, SUB, log=TS.QUIET)
    assert info["files"] == 9 and info["transitions"] == n_coll + n_roll and info["envs"] == {"keep_safe": 9}
    assert info["budgets"] == {"7000": 9} and len(info["sources"]) == 2
    bufs = TS.make_buffers()            # исходные буферы: вторая задача (и цель — её читать нельзя)
    root, work = os.path.join(tmp, "bufs"), os.path.join(tmp, "work")
    for sd, fs in bufs.items():
        os.makedirs(os.path.join(root, sd))
        for i, ep in enumerate(fs):
            torch.save(ep, os.path.join(root, sd, f"f{i}.pt"))
    old = (os.environ.get("RL_BUFFER_DIR"), sys.argv, MB.fit_model, O.GAMMA)
    fake = (E.load_agent, E.pick_action, E.q_values)
    E.load_agent, E.pick_action, E.q_values = REAL      # тренеры проверяют свой чекпоинт настоящим загрузчиком
    os.environ["RL_BUFFER_DIR"] = root
    MB.fit_model = lambda tab, members, work, quick, log: old[2](tab, members, work, True, log)
    mtag = f"roll{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    subs = f"{SUB2},{SUB}"
    try:
        rep = TCB.teacher(tmp, "ro_teach", ["--subdirs", subs, "--target-buffer", spec])
        sz, tb = rep["sizes"], rep["target_buffer"]
        assert sz["target_chains"] == 9 == tb["files"] and sz["target_rows"] == n_coll + n_roll == tb["transitions"]
        assert sz["target_rows_buffer"] == 0 and tb["envs"] == {"keep_safe": 9} and rep["verify"]["ok"]
        e0 = float(O.pde_meta()[PDE]["init_err"])
        text = TCB.run_main(["offline_rl.py", "--variant", "convnext_dqn", "--subdir", SUB, "--target-buffer", spec,
                             "--chain-fix", "--reward-form", "delta", "--init-err", repr(e0), "--value-bound",
                             "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70", "--scalar-ctx",
                             "--encoder", "resnet", "--save-model", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
                             "--mb-subdirs", subs, "--mb-work", work, "--mb-members", "2",
                             "--teacher-from", rep["_paths"]["ckpt"], "--teacher-w", "0.5", "--seeds", "1",
                             "--model-tag", mtag, "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu"])
        r = TCB.row_of(text)
        assert r["tb_files"] == 9 == r["tb_used_chains"] and r["tb_used_transitions"] == n_coll + n_roll == r["tb_transitions"]
        assert r["mb_surr_sig"] == rep["surrogate"]["data_sig"] and "'ok': True" in text and os.path.exists(out)
        assert np.isfinite(r["mb_loss"]) and np.isfinite(r["teacher_loss"])
    finally:
        sys.argv, MB.fit_model, O.GAMMA = old[1], old[2], old[3]
        E.load_agent, E.pick_action, E.q_values = fake
        if old[0] is None:
            os.environ.pop("RL_BUFFER_DIR", None)
        else:
            os.environ["RL_BUFFER_DIR"] = old[0]
        if os.path.exists(out):
            os.remove(out)
    print("смешанная спецификация (сбор + прогоны): учитель agent_combo.py и ученик offline_rl.py: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    with patched() as tmp:
        rows = test_guard_file(tmp); lap()
        test_default_unchanged(tmp, rows); lap()
        flags, rows_ex = test_explore(tmp, rows); lap()
        test_resume(tmp, flags, rows_ex); lap()
        test_merged_training(tmp); lap()
    print(f"test_rollout_buffer: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
