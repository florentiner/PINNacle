#!/usr/bin/env python
"""
Проверки стража застоя (online_eval_env.py --stall-guard, --stall-window, --stall-min). Без GPU и сети:
исполняется НАСТОЯЩИЙ run_seed и main (цикл действий, страж с откатом, контекст агента, чекпоинт докатки),
подменены только PINN и построение карт. Модель с одним весом w (лосс = exp(w)) обучается по эпохам, как
deepxde: на каждой эпохе вес сдвигается на RATE[оптимизатор, шаг], лосс пересчитывается раз в display_every и
в конце, колбэки получают on_train_begin / on_epoch_end / on_train_end, model.stop_training прерывает обучение.
Adam 1e-4 и L-BFGS 0.1 почти не снижают лосс (застой: меньше 1% за 250 эпох), Adam 1e-2 его поднимает.
Агент — расписание по номеру шага из скалярного контекста.

  * без флага прогон, строка и чекпоинт те же, что у кода до стража (исходник из git, если доступен), и
    StallCheck не подключается;
  * застрявшее длинное действие срезается на границе окна, остаток его эпох исполняет запасное действие
    кусками своей длины (последний по остатку), эпохи шага и бюджет прежние; оптимизатор на переходе
    новый, дальше с --keep-opt история L-BFGS переходит между кусками; тестер куска делит его длину;
  * контекст следующего решения: потраченный бюджет после всего шага, прошлое действие — запасное;
  * действие без застоя не срезается (проверки на каждой границе окна идут); действия не длиннее окна и
    само запасное действие не срезаются никогда;
  * режим tele: телеметрия срезанного шага — по последнему куску запасного (как карты);
  * журнал: stall_log, stall_spent, n_stall в строке и в чекпоинте докатки (докатка = непрерывный прогон);
  * вместе с откатом: откат судит шаг целиком (эпохи — в guard_spent), срез остаётся в stall_log;
  * отказы: --stall-guard без агента, с --save-buffer, с неверным окном или порогом.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_stall_guard.py
"""
import contextlib
import importlib.util
import io
import math
import os
import shutil
import subprocess
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
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import online_eval_env as E  # noqa: E402

torch.set_num_threads(1)
PDE = "poissonboltzmann2d"
IDX = E.parse_action_spec
FB = IDX("LBFGS:1:500")
SG = ["--stall-guard"]                       # окно 250 и порог 1% — умолчания
STALL_KEYS = {"stall_guard", "stall_window", "stall_min", "stall_fallback", "stall_log", "stall_spent", "n_stall"}
CALLS = []                                   # вызовы model.train: имя, шаг, эпохи, id оптимизатора, log_every тестера
SEEN = []                                    # состояния, по которым агент принимал решения
PLAN = []                                    # расписание агента по шагам (по кругу)


def rate(name, lr):
    """Сдвиг веса w за эпоху: Adam 1e-2 поднимает лосс, Adam 1e-4 и L-BFGS 0.1 — застой, PSO — ничего."""
    if name == "PSO":
        return 0.0
    if name == "Adam":
        return {1e-2: 0.3, 1e-3: -0.3, 1e-4: -0.004}[lr] / 1000.0
    return -0.2 * lr / 1000.0


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1) * 0.01)      # зависит от сида (set_random_seed)
        self.register_buffer("t", torch.zeros(1))               # обученные эпохи: едут в снимке весов


class StepModel:
    """deepxde.Model в миниатюре: обучение по эпохам с колбэками и прерыванием, _outputs_losses."""

    def __init__(self):
        self.net, self.data, self.opt = Net(), None, None
        self.train_state = types.SimpleNamespace(loss_train=np.array([0.7, 0.3]), X_train=None, y_train=None,
                                                 train_aux_vars=None, step=0, epoch=0)
        self.losshistory = types.SimpleNamespace(steps=[], loss_train=[])
        self.stop_training = False
        self.fresh = 0                       # вызовы _outputs_losses (проверки стража застоя)

    def compile(self, optimizer, loss_weights=None):
        self.opt = optimizer

    def _loss(self):
        lv = float(torch.exp(self.net.w.detach()))
        return np.array([0.7 * lv, 0.3 * lv])

    def _outputs_losses(self, training, inputs, targets, auxiliary_vars):
        self.fresh += 1
        return None, self._loss()

    def train(self, iterations, display_every, callbacks, model_save_path, save_model):
        name = "PSO" if isinstance(self.opt, str) else type(self.opt).__name__
        lr = None if name == "PSO" else self.opt.param_groups[0]["lr"]
        tester, saver = callbacks[0], callbacks[1]
        # сам объект оптимизатора держится в журнале: id живых объектов не повторяются
        CALLS.append(dict(name=name, lr=lr, it=0, opt=id(self.opt), obj=self.opt,
                          log_every=tester.kw.get("log_every"), n_cb=len(callbacks),
                          w0=float(self.net.w.detach())))
        for cb in callbacks:
            if hasattr(cb, "set_model"):
                cb.set_model(self)
        self.stop_training = False
        self.train_state.loss_train = self._loss()
        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin()
        for i in range(iterations):
            with torch.no_grad():
                self.net.w += rate(name, lr)
                self.net.t += 1
            self.train_state.step += 1
            self.train_state.epoch += 1
            CALLS[-1]["it"] += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self.train_state.loss_train = self._loss()
                self.losshistory.steps.append(self.train_state.step)
                self.losshistory.loss_train.append(self.train_state.loss_train)
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end()
            if self.stop_training:
                break
        CALLS[-1]["w1"] = float(self.net.w.detach())
        lv = float(torch.exp(self.net.w.detach()))
        tester.rmse, tester.brmse, tester.l2re, tester.bc_l2re = 0.4 * lv, 0.1 * lv, lv, 0.0
        saver.saved_models = (float(self.net.w.detach()), float(self.net.t))
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end()
        self.train_state.step = self.train_state.epoch = 0         # как ModelSaverCallback.on_train_end


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
    SEEN.append(np.array(state[4:9, 0, 0], dtype=np.float64))
    return PLAN[int(round(float(state[4, 0, 0]) * 70)) % len(PLAN)]


def plan_q(agent, state, mean, std, variant):
    q = np.zeros(27, dtype=np.float32)
    q[plan_action(agent, state, mean, std, variant)] = 1.0
    return q


FAKES = {
    "experiments.chain_eval.pde_registry": dict(build_get_model=lambda *a, **k: (lambda: (StepModel(), None))),
    "src.utils.callbacks": dict(TesterCallback=_Obj, ModelSaverCallback=_Obj),
    "landscape_visualization._aux.visualization_model": dict(VisualizationModel=_Obj),
    "landscape_visualization._aux.plot_loss_surface": dict(PlotLossSurface=Surface),
    "landscape_visualization._aux.early_stopping_plot": dict(EarlyStopping=_Obj),
}
PATCHED = ("load_agent", "pick_action", "q_values", "run_seed", "upload", "result_done", "gc")


@contextlib.contextmanager
def patched(mod):
    """Подмена PINN, карт, агента и заливки в модуле оценки mod (текущем или исходнике до стража)."""
    saved = {k: getattr(mod, k) for k in PATCHED}
    saved_argv = sys.argv
    mods = {k: sys.modules.get(k) for k in FAKES}
    tmp, cwd = tempfile.mkdtemp(prefix="stall_guard_"), os.getcwd()
    for k, v in FAKES.items():
        sys.modules[k] = types.SimpleNamespace(**v)
    mod.ROWS = []

    def run_seed(seed, args, progress_cb=None):
        mod.ROWS.append(saved["run_seed"](seed, args, progress_cb=progress_cb))
        return mod.ROWS[-1]

    mod.load_agent = lambda f: (Agent(), None, None, "convnext_dqn")
    mod.pick_action, mod.q_values, mod.run_seed = plan_action, plan_q, run_seed
    mod.upload = lambda row, name: None
    mod.result_done = lambda name: False
    mod.gc = types.SimpleNamespace(collect=lambda: 0)
    os.chdir(tmp)                       # чекпоинты среды пишутся в текущую папку
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            setattr(mod, k, v)
        sys.argv = saved_argv
        for k, v in mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def run(plan, extra, mod=E, tag="sg", seeds="1", guard=True, keep=True):
    """Один вызов main модуля оценки; (строки, вызовы train, состояния решений, вывод)."""
    PLAN[:] = [IDX(x) for x in plan]
    del CALLS[:], SEEN[:]
    with patched(mod) as tmp:
        sys.argv = (["online_eval_env.py", "--policy", "agent", "--model-file", "fake.pt", "--pde", PDE,
                     "--budget", "7000", "--seeds", seeds, "--tag", tag, "--save-dir", tmp, "--ae-epochs", "1"]
                    + (["--keep-opt", "--keep-opt-mode", "safe"] if keep else [])
                    + (["--guard-rollback", "1.0", "--guard-fallback", "LBFGS:1:500"] if guard else []) + list(extra))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod.main()
        rows = [dict(r) for r in mod.ROWS]
    return rows, [dict(c) for c in CALLS], [s.copy() for s in SEEN], out.getvalue()


def strip(row):
    """Строка без полей, зависящих от часов."""
    r = {k: v for k, v in row.items() if k not in ("elapsed_s", "t_ae_s", "t_surface_s", "t_opt_s")}
    r["hist"] = [h[:3] for h in row["hist"]]
    return r


def must_fail(fn, needle, what):
    """main должен выйти с сообщением, содержащим needle."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except SystemExit as e:
        assert needle in str(e.code), f"{what}: {e.code}"
        return
    raise AssertionError(what)


def base_module():
    """online_eval_env.py до стража застоя из git: родитель коммита, который ввёл --stall-guard, либо HEAD
    (страж ещё не закоммичен). None — git недоступен."""
    path = "experiments/rl_arch/online_eval_env.py"
    try:
        intro = subprocess.run(["git", "log", "--format=%H", "-S--stall-guard", "--", path], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.split()
        ref = f"{intro[-1]}^" if intro else "HEAD"
        src = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    if "--stall-guard" in src:
        return None, ref
    d = tempfile.mkdtemp(prefix="stall_base_")
    f = os.path.join(d, "online_eval_env_base.py")
    with open(f, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("online_eval_env_base", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SCRIPT_DIR = E.SCRIPT_DIR
    shutil.rmtree(d, ignore_errors=True)
    return mod, ref


# --------------------------------------------------------------------------------------------
# расписание с застоями: хорошее, долгий Adam 1e-4 (застой), L-BFGS 0.1 на 1000 (застой), короткие
MIXED = ["Adam:0.001:1000", "Adam:0.0001:2500", "LBFGS:0.1:1000", "LBFGS:0.1:100", "LBFGS:1:500",
         "Adam:0.01:100"]


def test_default_unchanged():
    """Без флага: строка как у кода до стража, StallCheck не подключается, полей стража нет."""
    rows, calls, _, _ = run(MIXED, [], seeds="1,2")
    assert all(not STALL_KEYS & set(r) for r in rows), "без флага полей стража в строке нет"
    assert all(c["n_cb"] == 2 for c in calls), "без флага колбэк стража не подключается"
    assert sum(c["it"] for c in calls) == sum(r["spent"] for r in rows) == 14000
    base, ref = base_module()
    if base is None:
        print(f"    (сверка с исходником до стража пропущена: {'git недоступен' if ref is None else ref})")
    else:
        for guard in (True, False):
            for keep in (True, False):
                new, c_new, s_new, _ = run(MIXED, [], seeds="1,2", guard=guard, keep=keep, tag="b")
                old, c_old, s_old, _ = run(MIXED, [], mod=base, seeds="1,2", guard=guard, keep=keep, tag="b")
                assert [strip(r) for r in new] == [strip(r) for r in old], f"строка не совпала (страж {guard})"
                assert [(c["name"], c["lr"], c["it"]) for c in c_new] == [(c["name"], c["lr"], c["it"]) for c in c_old]
                assert len(s_new) == len(s_old) and all(np.array_equal(a, b) for a, b in zip(s_new, s_old))
        print(f"    сверка с исходником до стража ({ref}): строки, обучение и состояния решений совпали")
    # флаг включён, но застоев нет: всё то же, кроме полей стража
    good = ["Adam:0.001:1000", "LBFGS:1:1000", "Adam:0.001:2500", "LBFGS:0.5:500"]
    off, c_off, s_off, _ = run(good, [])
    on, c_on, s_on, _ = run(good, SG)
    assert on[0]["stall_log"] == [] and on[0]["stall_spent"] == 0 and on[0]["n_stall"] == 0
    assert strip({k: v for k, v in on[0].items() if k not in STALL_KEYS}) == strip(off[0])
    assert [(c["name"], c["lr"], c["it"]) for c in c_on] == [(c["name"], c["lr"], c["it"]) for c in c_off]
    assert all(np.array_equal(a, b) for a, b in zip(s_on, s_off))
    print("без флага прогон прежний; с флагом без застоев — тот же прогон: OK")


def test_cut():
    """Застрявшее длинное действие: срез на границе окна, остаток — запасному кусками, эпохи прежние."""
    rows, calls, seen, text = run(["Adam:0.001:1000", "Adam:0.0001:2500", "Adam:0.001:1000", "LBFGS:0.5:1000"], SG)
    r = rows[0]
    a2 = IDX("Adam:0.0001:2500")
    # шаг 2 срезан после первого окна; шаг 6 (Adam 1e-4, обрезан бюджетом до 500 эпох) — тоже, остаток 250
    assert [s_[:3] + s_[4:] for s_ in r["stall_log"]] == [[2, a2, 250, 2250], [6, a2, 250, 250]], r["stall_log"]
    assert all(abs(s_[3] - math.exp(-0.001)) < 1e-4 for s_ in r["stall_log"]), "отношение лосса за окно (вес во float32)"
    assert r["stall_spent"] == 2500 and r["n_stall"] == 2 and r["stall_window"] == 250 and r["stall_min"] == 0.01
    assert r["stall_fallback"] == "LBFGS:1:500" and r["guard_log"] == [] and r["guard_spent"] == 0
    # цепочка: выбор агента с полным числом эпох шага; бюджет — 7000 эпох, как без среза
    assert [tuple(c) for c in r["chain"]] == [("Adam", 1e-3, 1000), ("Adam", 1e-4, 2500), ("Adam", 1e-3, 1000),
                                              ("LBFGS", 0.5, 1000), ("Adam", 1e-3, 1000), ("Adam", 1e-4, 500)]
    assert r["spent"] == 7000 == sum(c["it"] for c in calls) and [h[0] for h in r["hist"]] == [
        1000, 3500, 4500, 5500, 6500, 7000]
    # обучение шага 2: Adam 250 эпох (со стражем), затем L-BFGS 1.0 кусками 500 x 4 и 250
    st2 = calls[1:7]
    assert [(c["name"], c["lr"], c["it"], c["n_cb"]) for c in st2] == (
        [("Adam", 1e-4, 250, 3)] + [("LBFGS", 1.0, 500, 2)] * 4 + [("LBFGS", 1.0, 250, 2)])
    assert [c["log_every"] for c in st2[1:]] == [100] * 4 + [50], "тестер куска делит его длину"
    # оптимизатор: на переходе новый L-BFGS (сброс, как при откате), дальше с --keep-opt safe — тот же объект
    assert len({c["opt"] for c in st2[1:]}) == 1 and st2[1]["opt"] != st2[0]["opt"]
    # контекст решения 3: потрачено 3500 из 7000, прошлое действие — запасное L-BFGS 1.0 x 500
    ctx3 = seen[2]
    assert abs(ctx3[0] - 2 / 70) < 1e-6 and abs(ctx3[1] - 3500 / 7000) < 1e-6
    assert ctx3[2] == (FB // 9) / 2.0 == 0.5 and ctx3[3] == (FB % 3) / 2.0 == 0.5, ctx3
    assert seen[1][2] == 0.0 and seen[1][3] == 0.5, "до среза контекст — обычный (Adam x 1000)"
    assert "страж застоя: Adam lr=0.0001 ep=2500" in text and "срез застоя после 250" in text
    assert [(c["name"], c["it"]) for c in calls[-2:]] == [("Adam", 250), ("LBFGS", 250)]
    # без --keep-opt каждый кусок запасного — новый оптимизатор, как отдельные действия
    _, calls_nk, _, _ = run(["Adam:0.001:1000", "Adam:0.0001:2500"], SG, keep=False)
    fb = [c for c in calls_nk[1:7]]
    assert [c["name"] for c in fb] == ["Adam"] + ["LBFGS"] * 5 and len({c["opt"] for c in fb[1:]}) == 5
    # L-BFGS 0.1 застрял: запасное L-BFGS 1.0 начинает с НОВОЙ историей даже с --keep-opt
    rows3, calls3, _, _ = run(["LBFGS:0.1:1000", "LBFGS:1:1000"], SG)
    assert rows3[0]["stall_log"][0][:3] == [1, IDX("LBFGS:0.1:1000"), 250] and rows3[0]["stall_log"][0][4] == 750
    assert calls3[1]["opt"] != calls3[0]["opt"] and calls3[2]["opt"] == calls3[1]["opt"]
    print("застрявшее действие срезано, остаток — запасному кусками, эпохи и контекст согласованы: OK")


def test_not_cut():
    """Действие без застоя не срезается; не длиннее окна и само запасное — не срезаются никогда."""
    rows, calls, _, _ = run(["Adam:0.001:2500"], SG)          # третье обрезано бюджетом до 2000
    assert rows[0]["stall_log"] == [] and [c["it"] for c in calls] == [2500, 2500, 2000]
    assert all(c["n_cb"] == 3 for c in calls)
    # короткие застревающие действия (L-BFGS 0.1 x 100, PSO 0 x 100/200) — без проверки
    rows, calls, _, _ = run(["LBFGS:0.1:100", "PSO:0:200", "Adam:0.001:1000"], SG)
    assert rows[0]["stall_log"] == [] and rows[0]["spent"] == 7000
    assert all(c["n_cb"] == (3 if c["it"] > 250 else 2) for c in calls)
    # действие ровно в окно: Adam 1e-4 x 1000 при окне 1000 — не срезается
    rows, calls, _, _ = run(["Adam:0.0001:1000"], SG + ["--stall-window", "1000"])
    assert rows[0]["stall_log"] == [] and all(c["n_cb"] == 2 and c["it"] == 1000 for c in calls)
    # порог 50%: срезалось бы всё, но запасное действие (выбор агента L-BFGS 1.0 x 500) не срезается
    rows, calls, _, _ = run(["LBFGS:1:500", "Adam:0.001:1000"], SG + ["--stall-window", "100", "--stall-min", "0.5"])
    r = rows[0]
    assert all(s[1] == IDX("Adam:0.001:1000") for s in r["stall_log"]) and r["n_stall"] == 5
    assert all(c["n_cb"] == 2 for c in calls if c["name"] == "LBFGS")
    assert r["spent"] == 7000 == sum(c["it"] for c in calls)
    print("без застоя, короткие действия и запасное — не срезаются: OK")


def test_rollback_and_resume():
    """Срез и откат: откат судит шаг целиком; журнал стража едет в чекпоинте докатки."""
    # Adam 1e-2 поднимает лосс: срез на первом окне, остаток — запасному Adam 1e-2 x 100, он тоже поднимает,
    # и откат стража возвращает веса до шага и исполняет запасное с начала
    rows, calls, seen, _ = run(["Adam:0.01:1000", "Adam:0.001:1000"], SG + ["--guard-fallback", "Adam:0.01:100"])
    r = rows[0]
    a = IDX("Adam:0.01:1000")
    # шаг 1: срез на первом окне (лосс вырос), остаток 750 — запасному кусками 100 x 7 и 50; лосса до первого
    # шага нет, откат его не судит
    assert [(c["name"], c["lr"], c["it"]) for c in calls[:9]] == ([("Adam", 1e-2, 250)] + [("Adam", 1e-2, 100)] * 7
                                                                 + [("Adam", 1e-2, 50)])
    # шаги 3 и 5: срез, затем шаг целиком поднял лосс — откат и запасное с начала (без проверки стража);
    # шаг 7 обрезан бюджетом до 800 эпох: срез, остаток 550, откат уже не нужен (бюджет выбран)
    assert [g[:2] for g in r["guard_log"]] == [[3, a], [5, a]] and r["guard_spent"] == 2000
    assert [s_[0] for s_ in r["stall_log"]] == [1, 3, 5, 7] and [s_[4] for s_ in r["stall_log"]] == [750, 750, 750, 550]
    assert all(s_[1] == a and s_[2] == 250 and s_[3] > 1 for s_ in r["stall_log"]) and r["stall_spent"] == 2800
    assert [tuple(c) for c in r["chain"]] == [("Adam", 1e-2, 1000), ("Adam", 1e-3, 1000), ("Adam", 1e-2, 100),
                                              ("Adam", 1e-3, 1000), ("Adam", 1e-2, 100), ("Adam", 1e-3, 1000),
                                              ("Adam", 1e-2, 800)], "после отката в цепочке — запасное, как прежде"
    assert seen[3][2] == 0.0 and seen[3][3] == 0.0, "прошлое действие после отката — запасное Adam x 100"
    assert abs(seen[1][1] - 1000 / 7000) < 1e-6 and abs(seen[3][1] - 3100 / 7000) < 1e-6, \
        "потраченный бюджет — после всего шага"
    assert r["spent"] == 7000 == sum(c["it"] for c in calls)
    # докатка: чекпоинт после каждого шага, прогон прерван лимитом времени и продолжен с --resume
    plan = ["Adam:0.001:1000", "Adam:0.0001:2500", "Adam:0.001:1000", "LBFGS:0.5:1000"]
    full, _, _, _ = run(plan, SG + ["--ckpt-every", "1", "--ckpt-min-interval", "0"], tag="rs")
    PLAN[:] = [IDX(x) for x in plan]
    with patched(E) as tmp:
        base = ["online_eval_env.py", "--policy", "agent", "--model-file", "fake.pt", "--pde", PDE, "--budget", "7000",
                "--seeds", "1", "--tag", "rs", "--save-dir", tmp, "--ae-epochs", "1", "--keep-opt", "--keep-opt-mode",
                "safe", "--guard-rollback", "1.0", "--guard-fallback", "LBFGS:1:500", "--resume"] + SG
        parts = []
        for hours in ("0.0000001",) * 3 + ("0",):
            sys.argv = base + ["--hours", hours]
            with contextlib.redirect_stdout(io.StringIO()):
                E.main()
            parts.append(dict(E.ROWS[-1]))
        ck = torch.load(f"ckpt_{PDE}_rs_seed1.pt", weights_only=False)
    assert [p.get("unfinished", False) for p in parts] == [True, True, True, False]
    assert ck["t3"]["stall_log"] == full[0]["stall_log"][:1] and ck["t3"]["stall_spent"] == 2250
    end = parts[-1]
    for key in ("chain", "l2re", "spent", "stall_log", "stall_spent", "n_stall", "guard_log", "guard_spent"):
        assert end[key] == full[0][key], key
    assert [h[:3] for h in end["hist"]] == [h[:3] for h in full[0]["hist"]]
    print("срез вместе с откатом, журнал стража в чекпоинте докатки: OK")


def test_tele():
    """Режим tele (случайная политика без карт): телеметрия срезанного шага — по последнему куску запасного,
    как карты и прошлое действие в контексте (смещение весов — канал 9 — за этот кусок)."""
    del CALLS[:]
    with patched(E) as tmp:
        sys.argv = ["online_eval_env.py", "--policy", "random", "--state-mode", "tele", "--no-state", "--pde", PDE,
                    "--budget", "7000", "--seeds", "1,2,3", "--tag", "tl", "--save-dir", tmp, "--keep-opt",
                    "--keep-opt-mode", "safe", "--guard-rollback", "1.0", "--guard-fallback", "LBFGS:1:500"] + SG
        with contextlib.redirect_stdout(io.StringIO()):
            E.main()
        rows = [dict(r) for r in E.ROWS]
    cum, k0, n_cut = np.cumsum([c["it"] for c in CALLS]), 0, 0
    for r in rows:
        assert r["spent"] == 7000
        ends = [k0 + int(np.searchsorted(cum[k0:], cum[k0 - 1] + h[0] if k0 else h[0])) for h in r["hist"]]
        rolled = {g[0] for g in r["guard_log"]}
        for st_, a_, done, ratio, rem in r["stall_log"]:
            c = CALLS[ends[st_ - 1]]
            assert c["name"] == "LBFGS" and c["lr"] == 1.0 and c["n_cb"] == 2, "шаг кончается запасным"
            d = abs(c["w1"] - c["w0"]) / abs(c["w0"])
            want = float(np.clip((np.log10(d + 1e-12) + 4.0) / 1.5, -3.0, 3.0))
            tl = [t for t in r["tele_log"] if t[0] == st_][0]
            assert abs(tl[2 + 9] - want) < 2e-3, (st_, tl, want)
            n_cut += st_ not in rolled
        k0 = ends[-1] + 1
    assert n_cut >= 2 and k0 == len(CALLS), f"срезов без отката: {n_cut}"
    print(f"tele: телеметрия срезанного шага — по последнему куску запасного ({n_cut} срезов): OK")


def test_refusals():
    """Отказы main на несовместимых флагах."""
    for extra, what in (((["--save-buffer", "x"], "несовместим с --save-buffer"), "--save-buffer"),
                        ((["--stall-window", "0"], "--stall-window >= 1"), "окно 0"),
                        ((["--stall-min", "1.0"], "--stall-min из (0, 1)"), "порог 1"),
                        ((["--stall-min", "0"], "--stall-min из (0, 1)"), "порог 0"),
                        ((["--guard-fallback", "LBFGS:7:500"], "--guard-fallback: нет действия"), "неизвестное запасное")):
        must_fail(lambda e=extra: run(["Adam:0.001:1000"], SG + e[0], guard=False), extra[1], f"--stall-guard: {what}")

    def scripted():
        PLAN[:] = [IDX("Adam:0.001:1000")]
        with patched(E) as tmp:
            sys.argv = ["online_eval_env.py", "--policy", "script", "--script", "Adam:0.001:1000", "--tag", "x",
                        "--save-dir", tmp, "--no-state"] + SG
            E.main()
    must_fail(scripted, "только на действия агента", "--stall-guard без агента")
    print("отказы на несовместимых флагах: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    test_default_unchanged(); lap()
    test_cut(); lap()
    test_not_cut(); lap()
    test_rollback_and_resume(); lap()
    test_tele(); lap()
    test_refusals(); lap()
    print(f"test_stall_guard: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
