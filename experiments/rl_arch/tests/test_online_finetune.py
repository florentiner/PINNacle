#!/usr/bin/env python
"""
Проверки онлайнового дообучения офлайнового рецепта и режима состояния fulltele (без GPU, без
сети, PINN и автокодировщик не обучаются: среда подменена детерминированной заглушкой, буфер —
синтетические цепочки).

  * строгий тёплый старт (--warm-strict) от чекпоинта офлайнового вида (ResNet, 9 каналов,
    собран здесь через offline_rl.QNet): та же сеть и те же Q, что у online_eval_env.load_agent;
  * громкий отказ при любом расхождении (вариант, каналы, контекст, веса, энкодер, нет файла);
  * полный прогон main(): RLPD с нормировкой чекпоинта для обеих половин батча, статистики
    BatchNorm заморожены, сохранённый агент читается load_agent и даёт те же Q, что сеть тренера;
  * fulltele: раскладка каналов, расширение чекпоинта нулевыми весами (Q не меняются), нулевая
    телеметрия офлайновых переходов, состояния тренера и оценки на одной цепочке совпадают;
  * прежняя командная строка: та же сеть, прежняя нормировка по буферу, прежний eps.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_online_finetune.py
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import types
import zlib

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ["HF_HUB_OFFLINE"] = "1"
for _k in ("HF_TOKEN_WRITE", "HF_TOKEN"):      # проверка ничего не выгружает
    os.environ.pop(_k, None)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import torch  # noqa: E402
import offline_rl as O  # noqa: E402
import online_eval_env as E  # noqa: E402
import online_train_env as T  # noqa: E402
import tele_maps as TM  # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(1)        # крошечные свёртки: потоки только мешают
PDE, SUBDIR = "poissonboltzmann2d", "poisson_boltzmann_2d"
BUDGET, KMAX = 600, 70
GUIDE = "Adam:0.001:100,LBFGS:1:100,LBFGS:0.5:100,Adam:0.0001:100"
MASK = "adam:0.01,pso"


# ---------------------------------------------------------------- заглушка среды
class FakeNet(torch.nn.Module):
    """Без Linear: переинициализация среды её не трогает, веса детерминированы."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))


class FakeModel:
    """Контракт dde.Model, которым пользуются тренер и оценка. Лосс после действия — функция
    истории действий, поэтому одна и та же цепочка даёт одинаковые наблюдения в обеих средах."""

    def __init__(self):
        self.net = FakeNet()
        self.losshistory = types.SimpleNamespace(steps=[], loss_train=[])
        self.train_state = types.SimpleNamespace(loss_train=np.array([0.9, 0.1]), X_train=None,
                                                 y_train=None, train_aux_vars=None)
        self.outputs_losses_train = None
        self.pde = types.SimpleNamespace(num_pde=1)
        self.level, self.step, self.history, self.opt = 1.0, 0, [], None

    def compile(self, optimizer, loss_weights=None):
        self.opt = optimizer

    def train(self, iterations, display_every, callbacks, model_save_path=None, save_model=False):
        sig = ((self.opt, 0.0, int(iterations)) if isinstance(self.opt, str) else      # PSO — строка
               (type(self.opt).__name__, float(self.opt.param_groups[0]["lr"]), int(iterations)))
        self.history.append(sig)
        f = 0.55 + 0.4 * ((zlib.crc32(repr(sig).encode()) % 100) / 100.0)      # убыль лосса за действие
        n = max(1, iterations // display_every)
        for k in range(n + 1):
            lv = self.level * f ** (k / n)
            self.losshistory.steps.append(self.step + k * display_every)
            self.losshistory.loss_train.append(np.array([0.9 * lv, 0.1 * lv]))
        self.step += iterations
        self.level *= f
        with torch.no_grad():
            self.net.w.add_(0.01 * len(self.history))
        self.train_state.loss_train = np.array([0.9 * self.level, 0.1 * self.level])
        for cb in callbacks:
            cb.model = self
            if isinstance(cb, FakeTester):
                cb.rmse, cb.brmse = 0.5 * self.level, 0.1 * self.level
                cb.l2re, cb.bc_l2re = self.level, 0.2 * self.level


class FakeTester:
    def __init__(self, log_every=100):
        self.model = None


class FakeSaver:
    def __init__(self, total_iterations=0, n_save_models=0):
        self.model = None

    @property
    def saved_models(self):
        return self.model               # «траектория» действия: по ней строится карта


class FakeVM:
    def __init__(self, *a, **k):
        pass

    def train(self, *a, **k):
        return None


class FakeSurface:
    def __init__(self, solver_models=None, **k):
        self.m = solver_models

    def save_equation_loss_surface(self, log_key=True):
        return dict(sig=tuple(self.m.history))


def fake_build_state(raw, prev_raw):
    """Четыре карты как функция истории действий (канал delta — нули до второй карты цепочки)."""
    rng = np.random.default_rng(zlib.crc32(repr(raw["sig"]).encode()))
    maps = rng.normal(size=(4, 26, 26)).astype(np.float32)
    maps[3] = 0.0 if prev_raw is None else np.clip(maps[3], -1, 1)
    return maps


class Clock:
    """Часы тренера: лимит времени наступает, когда заглушка отдала заданное число цепочек."""

    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        pass


FAKE_MODULES = ("experiments.chain_eval.pde_registry", "src.utils.callbacks",
                "landscape_visualization._aux.visualization_model",
                "landscape_visualization._aux.plot_loss_surface",
                "landscape_visualization._aux.early_stopping_plot")


def make_eps(seed=0, n_chains=10):
    """Синтетический дамп буфера старого формата: цепочки по 3-4 действия разных семейств."""
    rng = np.random.default_rng(seed)

    def maps():
        return {k: rng.normal(size=(26, 26)).astype(np.float32)
                for k in ("loss_total", "loss_oper", "loss_bnd")}

    out = []
    for c in range(n_chains):
        n = 3 + c % 2
        states = [maps() for _ in range(n + 1)]
        err = 0.7
        for j in range(n):
            err *= float(rng.uniform(0.5, 0.95))
            out.append(dict(state=states[j], next_state=states[j + 1],
                            action=(int((c + j) % 2), {"lr": int(1 + (c + j) % 2), "epochs": 0}),
                            reward=-err, reward_model=0.0, done=(1 if j == n - 1 else 0)))
    return [out]


@contextlib.contextmanager
def fake_env(n_chains):
    """Подмена среды для online_train_env.main() и online_eval_env.run_seed()."""
    clock = Clock()
    made = [0]

    def build_get_model(pde, hidden, inverse_plain_fnn=False):
        def get_model():
            made[0] += 1
            if made[0] > n_chains:
                clock.t += 1e9          # лимит времени: следующая цепочка обрывается на первом шаге
            return FakeModel(), None
        return get_model

    mods = {FAKE_MODULES[0]: dict(build_get_model=build_get_model),
            FAKE_MODULES[1]: dict(TesterCallback=FakeTester, ModelSaverCallback=FakeSaver),
            FAKE_MODULES[2]: dict(VisualizationModel=FakeVM),
            FAKE_MODULES[3]: dict(PlotLossSurface=FakeSurface),
            FAKE_MODULES[4]: dict(EarlyStopping=lambda patience=0: None)}
    saved_mods = {k: sys.modules.get(k) for k in mods}
    saved = (T.time, T.build_state, E.build_state, O.load_episodes, T.upload, sys.argv, T.GAMMA,
             os.getcwd(), E.LBFGS_TOL, T.gc, E.gc)
    # сборка мусора после каждого шага среды (четверть секунды) в заглушке не нужна
    T.gc = E.gc = types.SimpleNamespace(collect=lambda: 0)
    for k, d in mods.items():
        m = types.ModuleType(k)
        m.__dict__.update(d)
        sys.modules[k] = m
    T.time, T.build_state, E.build_state = clock, fake_build_state, fake_build_state
    O.load_episodes = lambda data_dir=None, subdir=None, max_files=0: make_eps()
    T.upload = lambda *a, **k: None
    try:
        yield made
    finally:
        (T.time, T.build_state, E.build_state, O.load_episodes, T.upload, sys.argv, T.GAMMA,
         cwd, E.LBFGS_TOL, T.gc, E.gc) = saved
        os.chdir(cwd)
        for k, m in saved_mods.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m


def run_train(tmp, argv, n_chains=3):
    """main() тренера в заглушке среды. Возвращает (вывод, шпион: сеть и буферы)."""
    spy = {}
    orig_save, orig_mixed, orig_upd = T.save_agent, T.agent_update_mixed, T.agent_update

    def save_spy(net, *a, **k):
        spy["net"] = net
        return orig_save(net, *a, **k)

    def mixed_spy(net, off_buf, on_buf, *a, **k):
        spy["off"], spy["on"] = off_buf, on_buf
        spy["mixed"] = spy.get("mixed", 0) + 1
        return orig_mixed(net, off_buf, on_buf, *a, **k)

    def upd_spy(net, buf, *a, **k):
        # с --rlpd, пока онлайновых переходов меньше восьми, обновления идут по офлайновому буферу
        spy.setdefault("off", buf)
        return orig_upd(net, buf, *a, **k)

    out = io.StringIO()
    with fake_env(n_chains):
        os.chdir(tmp)
        T.save_agent, T.agent_update_mixed, T.agent_update = save_spy, mixed_spy, upd_spy
        sys.argv = ["online_train_env.py"] + argv + ["--save-dir", os.path.join(tmp, "runs")]
        try:
            with contextlib.redirect_stdout(out):
                T.main()
        finally:
            T.save_agent, T.agent_update_mixed, T.agent_update = orig_save, orig_mixed, orig_upd
    return out.getvalue(), spy


def must_exit(fn, text, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except (SystemExit, FileNotFoundError) as e:
        msg = str(e)
        assert text in msg, f"{what}: в сообщении нет {text!r}: {msg[:300]}"
        return msg
    raise AssertionError(f"{what}: отказа не было")


# ---------------------------------------------------------------- чекпоинт офлайнового вида
def offline_ckpt(path, in_ch=9, encoder="resnet", seed=0):
    """Чекпоинт в формате offline_rl.py --save-model: ResNet, 4 карты + 5 каналов контекста."""
    torch.manual_seed(seed)
    net = O.QNet("convnext_dqn", CPU, in_ch=in_ch, encoder=encoder)
    net.set_train(True)
    with torch.no_grad():               # статистики BatchNorm отличны от начальных
        for _ in range(3):
            net.q_scalar(torch.randn(16, in_ch, 26, 26))
    net.set_train(False)
    rng = np.random.default_rng(seed)
    mean = np.zeros((1, in_ch, 1, 1), np.float32)
    std = np.ones((1, in_ch, 1, 1), np.float32)
    mean[0, :4, 0, 0] = rng.normal(size=4) * 0.3
    std[0, :4, 0, 0] = rng.uniform(0.5, 2.0, size=4)
    meta = dict(ctx_kmax=KMAX, ctx_budget=BUDGET, ctx_err="none", state_mode="full", gamma=0.99,
                offline=True, scalar_ctx=True, episode_budget=BUDGET, init_err=0.811,
                reward_form="delta", value_bound=True, pde_ctx=False, err_norm="none", train_pdes=[],
                model_tag="X", encoder=encoder, ens=1, n_quantiles=32, in_ch=in_ch)
    ck = {"variant": "convnext_dqn", "seed": 1,
          "state_dict": {k: v.cpu().clone() for k, v in net.model.state_dict().items()},
          "mean": mean, "std": std, "meta": meta}
    torch.save(ck, path)
    return ck


def ns(ck_path, **kw):
    """Флаги тренера, которые читает warm_strict_net (значения боевой команды дообучения)."""
    base = dict(warm_start=ck_path, variant="convnext_dqn", encoder="auto", state_mode="full",
                warm_widen=False, scalar_ctx=True, max_chain_steps=KMAX, budget=BUDGET, ctx_no_err=True,
                pde_ctx=False, err_norm="none", online_reward="delta", init_err=0.811,
                episode_budget=BUDGET, value_bound=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


def rand_state(rng, mean, std):
    return (np.asarray(mean)[0] + np.asarray(std)[0] * rng.normal(size=(mean.shape[1], 26, 26))).astype(np.float32)


FT = ["--variant", "convnext_dqn", "--pde", PDE, "--hours", "1", "--save-agent", "--save-every", "1",
      "--save-buffer", "--rlpd", "--rlpd-utd", "2", "--rlpd-subdir", SUBDIR, "--offline-fix",
      "--offline-reward", "delta", "--init-err", "0.811", "--gamma", "0.99", "--episode-budget", str(BUDGET),
      "--tolerance", "0", "--max-chain-steps", str(KMAX), "--scalar-ctx", "--ctx-no-err", "--value-bound",
      "--keep-opt", "--keep-opt-mode", "safe", "--mask", MASK, "--batch-size", "8", "--seed", "42"]


# ---------------------------------------------------------------- проверки
def test_strict(tmp):
    path = os.path.join(tmp, "agent_convnext_dqn_cfix_rdelta_vb_X_seed1.pt")
    ck = offline_ckpt(path)
    ag, mean, std, variant = E.load_agent(path)
    assert ag.enc_kind == "resnet" and mean.shape == (1, 9, 1, 1)
    old_gamma, T.GAMMA = T.GAMMA, 0.99
    try:
        net, m2, s2, info = T.warm_strict_net(ck, ns(path), CPU, 4, 5)
        assert net.enc_kind == "resnet" and info["arch"]["in_ch"] == 9 and not info["widened"]
        assert info["warn"] == [] and info["tensors"] == len(ck["state_dict"])
        assert np.array_equal(m2, mean) and np.array_equal(s2, std)
        net.set_train(False)
        rng = np.random.default_rng(1)
        for k in range(4):
            st = np.zeros((9, 26, 26), np.float32) if k == 0 else rand_state(rng, mean, std)
            q_eval = E.q_values(ag, st, mean, std, variant)
            with torch.no_grad():
                q_tr = net.q_scalar(torch.as_tensor((st[None] - m2) / s2).float())[0].numpy()
            assert np.allclose(q_eval, q_tr, atol=1e-6), "строгий старт: Q не совпали с load_agent"
        for k, v in net.target.state_dict().items():
            assert torch.equal(v, net.model.state_dict()[k]), "целевая сеть не равна обучаемой"
        # расхождения в задаче — предупреждения, не отказ
        _, _, _, info = T.warm_strict_net(ck, ns(path, value_bound=False, init_err=0.5), CPU, 4, 5)
        assert len(info["warn"]) == 2 and any("value_bound" in w for w in info["warn"])

        # ---- громкие отказы
        must_exit(lambda: T.warm_strict_net(ck, ns(path, variant="cnx_qrdqn"), CPU, 4, 5), "вариант", "вариант")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, ctx_no_err=False), CPU, 4, 5), "ctx_err", "канал ошибки")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, max_chain_steps=12), CPU, 4, 5), "ctx_kmax", "K_max")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, budget=31000), CPU, 4, 5), "ctx_budget", "бюджет")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, online_reward="dlog"), CPU, 4, 5), "награда", "награда")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, scalar_ctx=False), CPU, 4, 0), "scalar_ctx", "контекст")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, encoder="convnext"), CPU, 4, 5), "энкодер", "энкодер")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, state_mode="tele"), CPU, 10, 5), "режим", "режим")
        must_exit(lambda: T.warm_strict_net(ck, ns(path, state_mode=TM.FULLTELE), CPU, 15, 5),
                  "--warm-widen", "fulltele без расширения")
        # meta без scalar_ctx (старый чекпоинт): расхождение ловится по числу каналов
        ck_nc = dict(ck, meta={k: v for k, v in ck["meta"].items() if k != "scalar_ctx"})
        must_exit(lambda: T.warm_strict_net(ck_nc, ns(path, scalar_ctx=False), CPU, 4, 0), "каналов", "каналы")
        # веса: лишний/недостающий тензор, другая форма головы, потерянный вид энкодера
        sd = dict(ck["state_dict"])
        gone = next(k for k in sd if k.startswith("0.body"))
        must_exit(lambda: T.warm_strict_net(dict(ck, state_dict={k: v for k, v in sd.items() if k != gone}),
                                            ns(path), CPU, 4, 5), "веса не легли", "нет тензора")
        hk = next(k for k in sd if k.startswith("1.") and sd[k].dim() == 2)
        must_exit(lambda: T.warm_strict_net(dict(ck, state_dict=dict(sd, **{hk: sd[hk][:, :-1].clone()})),
                                            ns(path), CPU, 4, 5), "веса не легли", "форма головы")
        must_exit(lambda: T.warm_strict_net(dict(ck, meta=dict(ck["meta"], encoder=None)), ns(path), CPU, 4, 5),
                  "веса не легли", "ResNet без meta.encoder")
        must_exit(lambda: T.warm_strict_net(dict(ck, mean=ck["mean"][:, :4]), ns(path), CPU, 4, 5),
                  "статистики", "нормировка")
    finally:
        T.GAMMA = old_gamma
    # main(): нет файла — отказ, а не старт с нуля; несовместимые флаги ловятся разбором
    must_exit(lambda: run_train(tmp, FT + ["--warm-start", "no_such_ckpt.pt", "--warm-strict"]),
              "не найден", "нет файла")
    for bad in (["--warm-strict"], ["--warm-start", path, "--warm-widen"],
                ["--warm-start", path, "--warm-strict", "--warm-widen"],
                ["--state-mode", "fulltele", "--preload", SUBDIR], ["--eps-start", "1.5"],
                ["--warm-start", path, "--warm-strict", "--hl-gauss"]):
        must_exit(lambda b=bad: run_train(tmp, [a for a in FT if a != "--rlpd"] + b), "2", f"разбор {bad}")
    print("строгий тёплый старт: сеть и Q как у load_agent, отказы при расхождениях: OK")
    return path, ck


def test_finetune_main(tmp, path, ck):
    out, spy = run_train(tmp, FT + ["--warm-start", path, "--warm-strict", "--eps-start", "0.1",
                                    "--tag", "ft_x", "--guide", GUIDE], n_chains=2)
    assert "строгий тёплый старт" in out and "статистики заморожены" in out
    assert "--keep-opt: из офлайнового буфера взяты" in out and spy.get("mixed", 0) > 0, out[-800:]
    eps = [float(x) for x in re.findall(r"eps=([0-9.]+)\)", out)]
    assert eps and abs(eps[0] - 0.1) < 0.006 and min(eps) >= 0.05, eps[:5]
    saved = os.path.join(tmp, "agent_ft_x.pt")
    ag, mean, std, variant = E.load_agent(saved)
    meta = ag._meta
    assert ag.enc_kind == "resnet" and variant == "convnext_dqn"
    assert meta["encoder"] == "resnet" and meta["in_ch"] == 9 and meta["state_mode"] == "full"
    assert (meta["ctx_kmax"], meta["ctx_budget"], meta["ctx_err"]) == (KMAX, BUDGET, "none")
    assert meta["keep_opt"] and meta["keep_opt_mode"] == "safe" and meta["scalar_ctx"]
    assert meta["warm_strict"] and meta["warm_start"] == os.path.basename(path) and meta["bn_mode"] == "frozen"
    assert meta["mask"] == [int(i) for i in np.where(~T.parse_mask(MASK))[0]] and meta["eps"][0] == 0.1
    # нормировка чекпоинта сохранена
    assert np.array_equal(np.asarray(mean), ck["mean"]) and np.array_equal(np.asarray(std), ck["std"])
    sd = torch.load(saved, map_location="cpu", weights_only=False)["state_dict"]
    moved = sum(float((sd[k].float() - ck["state_dict"][k].float()).abs().sum()) for k in sd
                if "running" not in k and "num_batches" not in k)
    assert moved > 0, "веса не изменились: обновлений не было"
    for k in sd:
        if "running" in k or "num_batches" in k:
            assert torch.equal(sd[k], ck["state_dict"][k]), f"статистика BatchNorm {k} изменилась"
    # сохранённый агент глазами оценки даёт те же Q, что сеть тренера
    rng = np.random.default_rng(5)
    for k in range(3):
        st = rand_state(rng, mean, std)
        q = E.q_values(ag, st, mean, std, variant)
        with torch.no_grad():
            q_tr = spy["net"].q_scalar(torch.as_tensor((st[None] - mean) / std).float())[0].numpy()
        assert q.shape == (27,) and np.isfinite(q).all() and np.allclose(q, q_tr, atol=1e-5)
    # офлайновая половина батча — в нормировке чекпоинта: контекст не нормирован (шаг / K_max)
    off = np.stack(spy["off"].s)
    assert off.shape[1:] == (9, 26, 26) and len(off) > 0
    step = off[:, 4, 0, 0] * KMAX
    assert np.allclose(step, np.round(step), atol=1e-4) and step.min() >= 0, step[:8]
    assert np.allclose(off[:, 8], 0.0), "канал ошибки при --ctx-no-err должен быть нулевым"
    print(f"дообучение в заглушке среды: RLPD {spy['mixed']} обновлений, нормировка чекпоинта, "
          f"агент читается load_agent: OK")


def test_tele_maps():
    assert TM.obs_channels("full") == 4 and TM.obs_channels("tele") == O.TELE_CH
    assert TM.obs_channels(TM.FULLTELE) == TM.FULLTELE_CH == 15
    assert TM.wants_tele("tele") and TM.wants_tele(TM.FULLTELE) and not TM.wants_tele("full")
    names = TM.channel_names(TM.FULLTELE, True)
    assert len(names) == 20 and names[:4] == list(TM.MAP_NAMES) and names[14] == "tele_valid"
    assert names[4] == "tele_log_loss" and names[15:] == list(TM.CTX_NAMES)
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(4, 26, 26)).astype(np.float32)
    tele = O.tele_state(1e-2, 8e-3, 2e-3, 1e-1, curve_steps=[0, 50, 100], curve_loss=[1e-1, 3e-2, 1e-2],
                        grad_norm=0.3, rel_disp=1e-2)
    st = TM.fulltele_obs(maps, tele)
    assert st.shape == (15, 26, 26) and np.array_equal(st[:4], maps) and np.array_equal(st[4:14], tele)
    assert np.all(st[14] == 1.0) and np.all(TM.fulltele_obs(maps, tele, valid=False)[14] == 0.0)
    s0 = TM.fulltele_obs(maps)
    assert np.all(s0[4:] == 0.0)
    b = TM.fulltele_batch(np.stack([maps, maps]))
    assert b.shape == (2, 15, 26, 26) and np.all(b[:, 4:] == 0.0) and np.array_equal(b[0, :4], maps)
    m, s = TM.fix_norm(np.full((1, 20, 1, 1), 0.5, np.float32), np.full((1, 20, 1, 1), 1e-6, np.float32))
    assert np.all(m[0, 4:15] == 0) and np.all(s[0, 4:15] == 1) and m[0, 0, 0, 0] == 0.5 and s[0, 19, 0, 0] < 1e-5
    # расширение: у новых каналов нулевые веса, Q в момент старта те же при любой телеметрии
    for enc in ("resnet", "convnext", "resnetpos"):
        torch.manual_seed(3)
        a = O.QNet("convnext_dqn", CPU, in_ch=9, encoder=enc)
        w = O.QNet("convnext_dqn", CPU, in_ch=20, encoder=enc)
        w.model.load_state_dict(TM.widen_state_dict(a.model.state_dict(), w.model.state_dict(), 9, 20))
        a.set_train(False); w.set_train(False)
        x9 = torch.randn(3, 9, 26, 26)
        x20 = torch.cat([x9[:, :4], torch.randn(3, 11, 26, 26) * 3, x9[:, 4:]], 1)
        with torch.no_grad():
            assert torch.allclose(a.q_scalar(x9), w.q_scalar(x20), atol=1e-5), f"расширение {enc}: Q изменились"
    m9 = np.arange(9, dtype=np.float32).reshape(1, 9, 1, 1)
    m20, s20 = TM.widen_norm(m9, m9 + 1.0, 9)
    assert list(m20[0, :, 0, 0]) == [0, 1, 2, 3] + [0] * 11 + [4, 5, 6, 7, 8]
    assert list(s20[0, :, 0, 0]) == [1, 2, 3, 4] + [1] * 11 + [5, 6, 7, 8, 9]
    print("fulltele: раскладка каналов, нормировка, расширение чекпоинта без изменения Q: OK")


def eval_states(model_file, script):
    """Состояния, по которым агент решает в online_eval_env.run_seed (сырые, до нормировки)."""
    seen = []
    orig = E.q_values

    def spy(agent, state, mean, std, variant):
        seen.append(np.array(state, copy=True))
        return orig(agent, state, mean, std, variant)

    av = ["--model-file", model_file, "--pde", PDE, "--budget", str(BUDGET), "--mask", MASK, "--keep-opt",
          "--keep-opt-mode", "safe", "--seeds", "1", "--tag", "t"]
    av += (["--policy", "script", "--script", script, "--script-tail", "agent"] if script
           else ["--policy", "agent"])
    args = E.build_parser().parse_args(av)
    with fake_env(1):
        E.q_values = spy
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                row = E.run_seed(1, args)
        finally:
            E.q_values = orig
    return seen, row


def test_fulltele_main(tmp, path, ck):
    out, spy = run_train(tmp, FT + ["--warm-start", path, "--warm-strict", "--warm-widen", "--state-mode",
                                    "fulltele", "--eps-start", "0.1", "--tag", "ft_tele", "--guide", GUIDE],
                         n_chains=2)
    assert "расширен до fulltele" in out and spy.get("mixed", 0) > 0, out[-800:]
    saved = os.path.join(tmp, "agent_ft_tele.pt")
    ag, mean, std, variant = E.load_agent(saved)
    meta = ag._meta
    assert meta["state_mode"] == TM.FULLTELE and meta["in_ch"] == 20 and meta["encoder"] == "resnet"
    assert meta["warm_widen"] and meta["channels"] == TM.channel_names(TM.FULLTELE, True)
    m20, s20 = TM.widen_norm(ck["mean"], ck["std"], 9)
    assert np.array_equal(np.asarray(mean), m20) and np.array_equal(np.asarray(std), s20)
    q = E.q_values(ag, rand_state(np.random.default_rng(2), mean, std), mean, std, variant)
    assert q.shape == (27,) and np.isfinite(q).all()
    w = torch.load(saved, map_location="cpu", weights_only=False)["state_dict"][TM.first_conv_key(ck["state_dict"])]
    assert w.shape[1] == 20 and float(w[:, 4:15].abs().sum()) > 0, "веса каналов телеметрии не учатся"
    # офлайновая половина: телеметрии нет (нули, признак 0); онлайновая: признак 1 после действия
    off, on_s, on_s2 = np.stack(spy["off"].s), np.stack(spy["on"].s), np.stack(spy["on"].s2)
    assert off.shape[1:] == (20, 26, 26) and np.all(off[:, 4:15] == 0.0)
    assert np.all(on_s2[:, 14] == 1.0) and np.abs(on_s2[:, 4:14]).sum() > 0
    assert np.all(np.ptp(on_s2[:, 4:15].reshape(len(on_s2), 11, -1), axis=2) == 0), "телеметрия — константы"
    first = on_s[:, 14, 0, 0] == 0.0                                 # старт цепочки: телеметрии ещё нет
    assert first.any() and (~first).any() and np.all(on_s[first][:, 4:15] == 0.0)

    # ---- раскладка тренера и оценки: одна цепочка (проводник) — одни и те же состояния
    buf = torch.load(os.path.join(tmp, "buffer_ft_tele.pt"), map_location="cpu", weights_only=False)
    ep1 = np.where(buf["EP"] == 1)[0]
    S_tr = buf["S"][ep1].astype(np.float32)
    g = T.parse_chain(GUIDE)
    assert list(buf["A"][ep1][:len(g)]) == g, "первая цепочка тренера должна идти по проводнику"
    toks = GUIDE.split(",")
    for j in range(len(g) + 1):
        seen, row = eval_states(saved, ",".join(toks[:j]))
        assert row["state_mode"] == TM.FULLTELE and row["keep_opt"] and len(row["tele_log"]) == row["n_steps"]
        st = seen[0]
        assert st.shape == (20, 26, 26)
        assert np.allclose(st, S_tr[j], atol=4e-3), \
            f"шаг {j}: состояние оценки и тренера расходятся, каналы {np.where(np.abs(st - S_tr[j]).max((1, 2)) > 4e-3)[0]}"
        if j:
            assert np.all(st[14] == 1.0)
            assert np.allclose(st[4:14, 0, 0], row["tele_log"][j - 1][2:], atol=1e-3)
            assert abs(st[15, 0, 0] - j / KMAX) < 1e-6 and abs(st[16, 0, 0] - 100.0 * j / BUDGET) < 1e-6
            assert st[19, 0, 0] == 0.0                                   # канал ошибки выключен
        else:
            assert np.all(st[:15] == 0.0) and st[17, 0, 0] == -1.0
    print(f"fulltele в заглушке среды: расширенный тёплый старт, нулевая телеметрия офлайна, состояния "
          f"тренера и оценки совпали на {len(g) + 1} шагах: OK")


def test_defaults(tmp):
    """Прежняя командная строка (решение BASE_NOVB) без новых флагов."""
    old = [a for a in FT if a not in ("--value-bound", "--keep-opt", "--keep-opt-mode", "safe", "--mask", MASK)]
    out, spy = run_train(tmp, old + ["--tag", "old_x"], n_chains=3)
    assert "тёплый старт" not in out and "BatchNorm" not in out and "off" in spy
    eps = [float(x) for x in re.findall(r"eps=([0-9.]+)\)", out)]
    want = [T.EPS_END + (T.EPS_START - T.EPS_END) * np.exp(-k / T.EPS_DECAY) for k in range(len(eps))]
    assert eps and np.allclose(eps, want, atol=0.006), (eps[:4], want[:4])
    ag, mean, std, variant = E.load_agent(os.path.join(tmp, "agent_old_x.pt"))
    ref = O.QNet("convnext_dqn", CPU, in_ch=9)
    assert ag.enc_kind == ref.enc_kind == "convnext"
    assert {k: tuple(v.shape) for k, v in ag.model.state_dict().items()} == \
           {k: tuple(v.shape) for k, v in ref.model.state_dict().items()}
    meta = ag._meta
    assert meta["state_mode"] == "full" and "warm_strict" not in meta and meta["keep_opt"] is False
    assert meta["mask"] == [] and meta["eps"] == [T.EPS_START, T.EPS_END, T.EPS_DECAY]
    # прежняя нормировка: статистики офлайнового буфера, включая каналы контекста
    assert np.asarray(mean).shape == (1, 9, 1, 1) and float(np.asarray(mean)[0, 4, 0, 0]) > 0
    off = np.stack(spy["off"].s)
    assert abs(float(off[:, 4].mean())) < 1e-3, "офлайновая половина нормирована не статистиками буфера"
    print("прежняя командная строка: сеть ConvNeXt, нормировка по буферу, eps авторов: OK")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path, ck = test_strict(tmp)
        test_tele_maps()
        test_finetune_main(tmp, path, ck)
        test_fulltele_main(tmp, path, ck)
        test_defaults(tmp)
    print("ВСЕ ПРОВЕРКИ ОНЛАЙНОВОГО ДООБУЧЕНИЯ И FULLTELE ПРОЙДЕНЫ")


if __name__ == "__main__":
    main()
