#!/usr/bin/env python
"""
Проверки видов карт (map_views.py, online_eval_env.py --views, render_views.py, collect_active.py --views) и
предобучения энкодера на них (cv_probe.py --views, --procedural, ssl-mv) без GPU и без сети, за секунды:

  * сложенные автокодировщики (train_view_aes) при одном виде и том же сиде совпадают с автокодировщиком
    статьи (VisualizationModel.train) до округления; несколько видов — разные плоскости; терпение останова;
  * render_views на поддельной поверхности: V карт (3, 26, 26), разные, скрытые координаты траектории сохранены;
  * run_seed: без флага прогон и строка прежние; с --views — те же цепочка, ошибки и карты среды (генераторы
    возвращены), у неплоских карт V видов, у плоских ни одного, приёмник _view_want уважается; main
    оценки флаг не принимает;
  * view_rows: повторы общего префикса убраны, канал delta по тому же виду предыдущего шага, номера состояний;
  * cv_probe --pretrain-only --views с ssl-mae и ssl-mv, --procedural: файлы энкодера формата MAE ложатся в
    агента (offline_rl.load_encoder_init, одна и две сети) и проходят offline_rl.py --encoder-init целиком;
    отказы на несовместимых флагах;
  * render_views.py: неплоские состояния исходного файла, деление по сидам, повторы не строятся, докатка,
    случайные цепочки со стопом по плоским картам;
  * collect_active.py --views: файл видов на цепочку (виды префикса у концовок), заливка одним коммитом с
    буфером; без флага — прежние записи.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_map_views.py
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
for _k in ("HF_TOKEN_WRITE", "HF_TOKEN"):      # проверка ничего не выгружает
    os.environ.pop(_k, None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import map_views as MV  # noqa: E402
import cv_probe as CV  # noqa: E402
import offline_rl as O  # noqa: E402
import online_eval_env as E  # noqa: E402
import target_buffer as TB  # noqa: E402
import render_views as RV  # noqa: E402
import collect_active as C  # noqa: E402
import killswitch  # noqa: E402
import test_cv_probe as TCV  # noqa: E402
import test_enc_pretrain as TEP  # noqa: E402
import test_collect_buffer as TCB  # noqa: E402

torch.set_num_threads(1)
CPU = torch.device("cpu")
SUB = "poisson_boltzmann_2d"
PDE = "poissonboltzmann2d"
QUIET = lambda *a: None  # noqa: E731
AEP = dict(num_of_layers=3, layers_AE=[991, 125, 15], rec_weight=10000.0)


def snapshots(n=10, seed=0, still=False):
    """n копий маленькой сети вдоль «траектории» (still — все одинаковые, как при застое L-BFGS)."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    out = []
    for _ in range(n):
        if not still:
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(0.05 * torch.randn_like(p))
        out.append(copy.deepcopy(net))
    return out


GRID = torch.stack(torch.meshgrid(torch.linspace(-1.25, 1.25, 26), torch.linspace(-1.25, 1.25, 26),
                                  indexing="ij"), -1).view(-1, 2)


def test_batched_aes():
    from landscape_visualization._aux.visualization_model import VisualizationModel
    from landscape_visualization._aux.early_stopping_plot import EarlyStopping
    snaps = snapshots()
    vm = VisualizationModel(device="cpu", path_to_plot_model=None, path_to_trajectories=None, **E.AE_MODEL_PARAMS)
    torch.manual_seed(5)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ref = vm.train(5e-4, 1200, 300, 100, 32, True, finetune_AE_model=False,
                       callbacks=[EarlyStopping(patience=4000)], solver_models=snaps)
    torch.manual_seed(5)
    one, info = MV.train_view_aes(snaps, 1, 300, 5e-4, 1200, batch_size=32, es_patience=4000, device="cpu", **AEP)
    with torch.no_grad():
        d_ref, d_one = ref.decoder(GRID), one[0].decoder(GRID)
    err = float((d_ref - d_one).abs().max() / d_ref.abs().max())
    assert err < 1e-4, f"сложенный автокодировщик разошёлся с автокодировщиком статьи: {err}"
    assert info["epochs"] == 300 and info["best_epoch"] == [200], info
    torch.manual_seed(5)
    many, info3 = MV.train_view_aes(snaps, 3, 300, 5e-4, 1200, device="cpu", **AEP)
    with torch.no_grad():
        dec = [m.decoder(GRID) for m in many]
        z = [m.encoder(torch.zeros(1, d_ref.shape[1])) for m in many]
    assert float((dec[0] - d_one).abs().max() / d_one.abs().max()) < 1e-4, "первый вид = автокодировщик того же сида"
    assert all(float((dec[0] - d).abs().mean()) > 1e-2 for d in dec[1:]), "виды — разные плоскости"
    assert all(not m.training for m in many) and len(z) == 3
    # терпение: при es_patience=1 вид перестаёт сравниваться после первой неудачи, итог — лучшая копия
    torch.manual_seed(5)
    _, info_es = MV.train_view_aes(snaps, 2, 300, 5e-2, 1200, es_patience=1, device="cpu", **AEP)
    assert all(b >= 0 for b in info_es["best_epoch"]), info_es
    print(f"сложенные автокодировщики: один вид = автокодировщик статьи (отклонение {err:.1e}), три вида — разные "
          f"плоскости, лучшая копия на эпохе {info['best_epoch'][0]}: OK")


class FakeSurface:
    """Поверхность вместо PlotLossSurface: лосс в узле сетки — средний квадрат весов, декодированных
    автокодировщиком (настоящая проверка, что вид зависит от своего автокодировщика)."""

    def __init__(self, solver_models=None, AE_model=None, dde_pde_model=None, x_range=None, **kw):
        self.ae, self.solver = AE_model, solver_models
        assert x_range == E.GRID_RANGE

    def save_equation_loss_surface(self, log_key=True):
        with torch.no_grad():
            w = self.ae.decoder(GRID)
            tr = torch.randn(len(self.solver), 2)
        self.states_dict = {"loss_total": dict(trajectory_coordinates=tr, trajectory_losses=torch.ones(len(self.solver)),
                                               original_trajectory_losses=2 * torch.ones(len(self.solver)))}
        base = (w ** 2).mean(1).view(26, 26)
        return {k: torch.log1p(base * (i + 1)) for i, k in enumerate(MV.LOSS_KEYS)}


def test_render_views():
    import landscape_visualization._aux.plot_loss_surface as PLS
    saved = PLS.PlotLossSurface
    PLS.PlotLossSurface = FakeSurface
    try:
        torch.manual_seed(3)
        maps, info = MV.render_views(snapshots(), None, 3, E.GRID_RANGE, 200, 5e-4, 1200, 32, 4000, AEP, "cpu")
    finally:
        PLS.PlotLossSurface = saved
    assert len(maps) == 3 and all(m.shape == (3, 26, 26) and m.dtype == np.float32 for m in maps)
    assert all(np.isfinite(m).all() for m in maps)
    assert not np.allclose(maps[0], maps[1]) and not np.allclose(maps[1], maps[2]), "виды различаются"
    assert info["k"] == 3 and len(info["traj"]) == 3 and info["traj"][0].shape == (10, 2)
    assert info["rec_gap"][0] is not None and info["t_ae"] >= 0 and not info["sequential"]
    raw = {k: torch.full((26, 26), 0.3) for k in MV.LOSS_KEYS}
    assert MV.flat_raw(raw) and not MV.flat_raw({k: torch.rand(26, 26) for k in MV.LOSS_KEYS})
    print("render_views: 3 вида (3, 26, 26), разные, координаты траектории и расхождение лосса сохранены: OK")


# --------------------------------------------------------------------------------------------
# run_seed с видами (среда подменена, как в test_rollout_buffer)
# --------------------------------------------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1) * 0.01)


class Model:
    def __init__(self):
        self.net, self.data, self.opt = Net(), None, None
        self.train_state = types.SimpleNamespace(loss_train=np.array([0.7, 0.3]), X_train=None)
        self.losshistory = types.SimpleNamespace(steps=[], loss_train=[])

    def compile(self, optimizer, loss_weights=None):
        self.opt = optimizer

    def train(self, iterations, display_every, callbacks, model_save_path, save_model):
        name = "PSO" if isinstance(self.opt, str) else type(self.opt).__name__
        w0 = float(self.net.w.detach())
        with torch.no_grad():
            if name != "PSO":
                self.net.w -= 0.1 * iterations / 1000.0
        lv = float(torch.exp(self.net.w.detach()))
        self.train_state.loss_train = np.array([0.7 * lv, 0.3 * lv])
        tester, saver = callbacks[:2]
        tester.rmse, tester.brmse, tester.l2re, tester.bc_l2re = 0.4 * lv, 0.1 * lv, lv, 0.0
        saver.saved_models = (w0, float(self.net.w.detach()))


class _Obj:
    def __init__(self, *a, **k):
        self.kw = k
        self.saved_models = None

    def train(self, *a, **k):
        return types.SimpleNamespace(r=float(torch.rand(1)))       # у каждого «автокодировщика» свой случай


class Surface(_Obj):
    """Карта задана весами до и после действия и случаем автокодировщика; вес не сдвинулся — плоская карта.
    Каждая поверхность тянет общий генератор torch: если виды его не вернут, карты среды разойдутся."""

    def save_equation_loss_surface(self, log_key=True):
        w0, w1 = self.kw["solver_models"]
        r = float(self.kw["AE_model"].r)
        noise = float(torch.rand(1))
        if w0 == w1:
            m = np.full((3, 26, 26), 0.25 + 0.0 * noise, np.float32)
        else:
            g = np.linspace(-1, 1, 26)
            m = np.stack([np.add.outer(g * (w1 - w0) * (i + 1), np.cos(g * 7 * r)) + noise for i in range(3)])
        return {k: torch.tensor(m[i], dtype=torch.float32) for i, k in enumerate(E.LOSS_TYPES)}


FAKES = {
    "experiments.chain_eval.pde_registry": dict(build_get_model=lambda *a, **k: (lambda: (Model(), None))),
    "src.utils.callbacks": dict(TesterCallback=_Obj, ModelSaverCallback=_Obj),
    "landscape_visualization._aux.visualization_model": dict(VisualizationModel=_Obj),
    "landscape_visualization._aux.plot_loss_surface": dict(PlotLossSurface=Surface),
    "landscape_visualization._aux.early_stopping_plot": dict(EarlyStopping=_Obj),
}
SCRIPT = "Adam:0.001:1000,PSO:0:300,LBFGS:1:500,Adam:0.0001:2500,PSO:0.001:100,LBFGS:0.5:1000"


@contextlib.contextmanager
def fake_env():
    mods = {k: sys.modules.get(k) for k in FAKES}
    saved_gc = E.gc
    E.gc = types.SimpleNamespace(collect=lambda: 0)
    for k, v in FAKES.items():
        sys.modules[k] = types.SimpleNamespace(**v)
    tmp, cwd = tempfile.mkdtemp(prefix="views_env_"), os.getcwd()
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        E.gc = saved_gc
        for k, v in mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def env_run(tmp, views=0, want=None):
    argv = ["--policy", "script", "--script", SCRIPT, "--script-tail", "stop", "--pde", PDE, "--budget", "7000",
            "--save-dir", tmp, "--tag", "v", "--ckpt-every", "0", "--progress-every", "0", "--keep-opt",
            "--keep-opt-mode", "safe"] + (["--views", str(views), "--view-sequential"] if views else [])
    args = E.build_parser().parse_args(argv)
    main, got = {}, {}
    args._map_sink = lambda s, m: main.__setitem__(int(s), np.array(m, np.float32))
    if views:
        args._view_sink = lambda s, m3, vs, info: got.__setitem__(int(s), (np.array(m3), list(vs), dict(info)))
        args._view_want = want
    with contextlib.redirect_stdout(io.StringIO()):
        row = E.run_seed(7, args)
    return row, main, got


def test_run_seed_views():
    with fake_env() as tmp:
        row0, main0, _ = env_run(tmp)
        row1, main1, got = env_run(tmp, views=3)
        row2, main2, got2 = env_run(tmp, views=2, want=lambda s: s != 3)
    core = lambda r: (r["chain"], r["l2re"], r["spent"], [h[:3] for h in r["hist"]], r["stop_reason"])  # noqa: E731
    assert core(row0) == core(row1) == core(row2), "виды не меняют прогон"
    assert sorted(main0) == sorted(main1) and all(np.array_equal(main0[s], main1[s]) for s in main0), \
        "карты среды те же (генераторы возвращены)"
    assert "views" not in row0 and "t_views_s" not in row0 and row1["views"] == 3 and "t_views_s" in row1
    steps = sorted(main0)
    assert sorted(got) == steps, "приёмник видов вызван после каждой карты среды"
    flat = {s for s in steps if np.ptp(main0[s][0]) == 0}
    assert flat == {2, 5}, flat                       # после PSO вес не сдвинулся (застой)
    for s in steps:
        m3, vs, info = got[s]
        assert np.array_equal(m3, main0[s][:3]) and info["flat"] == (s in flat)
        if s in flat:
            assert vs == [] and info["k"] == 0
        else:
            assert len(vs) == 3 and all(v.shape == (3, 26, 26) for v in vs) and info["k"] == 3
            assert not np.allclose(vs[0], vs[1]) and not any(np.allclose(v, m3) for v in vs), "виды разные"
    assert got2[3][1] == [] and len(got2[4][1]) == 2, "_view_want: шаг 3 без видов"
    # main оценки флаг не принимает
    old = sys.argv
    sys.argv = ["online_eval_env.py", "--policy", "random", "--views", "2", "--tag", "x"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            E.main()
        raise AssertionError("main должен отвергать --views")
    except SystemExit as e:
        assert "--views" in str(e.code)
    finally:
        sys.argv = old
    print(f"run_seed --views: прогон и карты среды прежние, у {len(steps) - len(flat)} неплоских карт по 3 вида, "
          f"у плоской — ни одного, _view_want уважается, main отвергает флаг: OK")


# --------------------------------------------------------------------------------------------
# Файлы видов -> строки зонда, предобучение
# --------------------------------------------------------------------------------------------

CH_A = [["Adam", 0.001, 1000], ["LBFGS", 1.0, 500], ["LBFGS", 0.5, 1000], ["Adam", 0.0001, 2500]]
CH_B = CH_A[:2] + [["Adam", 0.01, 100], ["PSO", 0.0, 300]]


def rmap(rng, flat=False):
    if flat:
        return np.full((3, 26, 26), rng.uniform(0.1, 3.0), np.float32)
    g = np.linspace(-1, 1, 26)
    a = rng.normal(size=3)
    base = np.add.outer(a[0] * g, a[1] * g ** 2) + 0.1 * rng.normal(size=(26, 26)) + 3.0
    return np.stack([base, 0.8 * base, 0.3 * base + a[2]]).astype(np.float32)


def fake_records(rng, seed=1):
    """Две цепочки одного сида с общим префиксом из двух действий: у первой виды на шагах 1-2 (по 2) и 3 (3),
    шаг 4 плоский; у второй шаги 1-2 — повтор префикса, шаг 3 с видами, шаг 4 плоский."""
    m1 = {s: rmap(rng, flat=(s == 4)) for s in (1, 2, 3, 4)}
    v1 = {1: [rmap(rng), rmap(rng)], 2: [rmap(rng), rmap(rng)], 3: [rmap(rng) for _ in range(3)]}
    m2 = {1: m1[1] + 0.001, 2: m1[2] + 0.001, 3: rmap(rng), 4: rmap(rng, flat=True)}
    v2 = {1: [rmap(rng)], 2: [rmap(rng)], 3: [rmap(rng), rmap(rng)]}
    r1 = MV.make_record(PDE, "t", "a.pt", seed, {}, CH_A, m1, v1, {s: dict(flat=False) for s in m1},
                        err={s: 0.1 / s for s in m1})
    r2 = MV.make_record(PDE, "t", "b.pt", seed, {}, CH_B, m2, v2, {s: dict(flat=False) for s in m2})
    return r1, r2


def test_view_rows():
    rng = np.random.default_rng(0)
    r1, r2 = fake_records(rng)
    R = MV.view_rows([r1, r2])
    assert R["n_states"] == 6, R["n_states"]          # r1: 1-4, r2: 3-4 (1-2 — повтор префикса)
    first = R["STEP"] == 0
    assert (R["VKEY"][first] == -1).all() and (R["VKEY"][~first] >= 0).all()
    assert R["maps"].shape[1:] == (4, 26, 26) and (R["maps"][first] == 0).all()
    # число строк: r1 — вид 0: 4, вид 1: 3, вид 2: 3, вид 3: 1; r2 — вид 0: 2, вид 1: 1, вид 2: 1
    cnt = {}
    for f, v in zip(R["FILE"][~first], R["VIEW"][~first]):
        cnt[(int(f), int(v))] = cnt.get((int(f), int(v)), 0) + 1
    assert cnt == {(0, 0): 4, (0, 1): 3, (0, 2): 3, (0, 3): 1, (1, 0): 2, (1, 1): 1, (1, 2): 1}, cnt
    # канал delta: шаг 1 — нули; вид 1 на шаге 2 — от вида 1 на шаге 1; вид 3 на шаге 3 — от карты среды шага 2
    rows = {(int(f), int(v), int(k)): i for i, (f, v, k) in enumerate(zip(R["FILE"], R["VIEW"], R["VKEY"]))}
    key = {s: MV.state_key(1, CH_A, s) for s in (1, 2, 3, 4)}
    kid = {}
    for i in np.where(~first & (R["FILE"] == 0) & (R["VIEW"] == 0))[0]:
        kid[len(kid) + 1] = int(R["VKEY"][i])
    assert len(set(key.values())) == 4
    i11, i12 = rows[(0, 1, kid[1])], rows[(0, 1, kid[2])]
    assert (R["maps"][i11, 3] == 0).all()
    assert np.allclose(R["maps"][i12, 3], O._delta_like_env(r1["views"][2][0][0], r1["views"][1][0][0]))
    assert np.allclose(R["maps"][i12, :3], r1["views"][2][0])
    i33 = rows[(0, 3, kid[3])]
    assert np.allclose(R["maps"][i33, 3], O._delta_like_env(r1["views"][3][2][0], r1["main"][1][0]))
    i04 = rows[(0, 0, kid[4])]
    assert np.allclose(R["maps"][i04, 3], O._delta_like_env(r1["main"][3][0], r1["main"][2][0]))
    # виды одного состояния — одна группа; max_views=1 — только карты среды; без dedupe префикс повторяется
    g = R["VKEY"] == kid[3]
    assert sorted(R["VIEW"][g].tolist()) == [0, 1, 2, 3]
    R1 = MV.view_rows([r1, r2], max_views=1)
    assert (R1["VIEW"] == 0).all() and int((R1["STEP"] > 0).sum()) == 6
    Rn = MV.view_rows([r1, r2], dedupe=False)
    assert Rn["n_states"] == 6 and int((Rn["STEP"] > 0).sum()) == int((R["STEP"] > 0).sum()) + 4, \
        "без удаления повторов префикс второй цепочки (2 состояния x 2 вида) идёт ещё раз с тем же номером"
    print("view_rows: повторы префикса убраны (6 состояний из 8), delta по тому же виду, группы видов: OK")


def write_views(root, recs):
    d = os.path.join(root, SUB)
    os.makedirs(d, exist_ok=True)
    for i, r in enumerate(recs):
        MV.save_record(os.path.join(d, f"t_{i}.pt"), r)
    return root


def test_pretrain(tmp):
    rng = np.random.default_rng(1)
    recs = []
    for sd in range(1, 5):
        recs += list(fake_records(rng, seed=sd))
    vroot = write_views(os.path.join(tmp, "views"), recs)
    out = os.path.join(tmp, "cv")
    argv = ["--tag", "w55t", "--subdirs", SUB, "--views", vroot, "--configs", "ssl-mae-resnet,ssl-mv-resnet",
            "--pretrain-only", "--no-upload", "--out-dir", out, "--cpu", "--batch", "16", "--no-amp",
            "--ssl-steps", "6", "--mv-dim", "32"]
    rep = CV.run(TCV._args(argv), QUIET)
    assert rep["finished"] and rep["data"]["pdes"] == [f"views:{SUB}"], rep["data"]
    files = {}
    torch.manual_seed(17)
    init = CV.make_encoder("resnet", 4, 26).state_dict()
    for kind in ("mae", "mv"):
        e = rep["encoders"][f"ssl-{kind}-resnet"]
        path = os.path.join(out, f"cvprobe_w55t_enc_{kind}_resnet.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["kind"] == "ssl_encoder" and ck["ssl"] == kind and set(ck["state_dict"]) == set(init)
        assert e["steps"] >= 6 and np.isfinite(e["loss_last"]) and ck["mean"].shape == (1, 4, 1, 1)
        assert any(not torch.equal(v, ck["state_dict"][k]) for k, v in init.items() if v.dtype.is_floating_point)
        files[kind] = path
    parts = rep["encoders"]["ssl-mv-resnet"]["loss_parts"]
    assert set(parts) == {"mae", "vic", "inv", "pairs"} and parts["pairs"] > 0 and parts["vic"] > 0, parts
    assert rep["encoders"]["ssl-mv-resnet"]["ssl_args"] == dict(mae_ratio=0.6, mv_w=1.0, mv_dim=32)
    # энкодер ложится в агента: одна и две сети, контекст — свежие каналы
    x = torch.randn(3, 9, 26, 26)
    for kind, path in files.items():
        for two in (False, True):
            net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", two_nets=two)
            info = O.load_encoder_init(net, path)
            assert info["ssl"] == kind and info["partial"] == ["stem.0.weight"]
            net.model.eval()
            with torch.no_grad():
                assert torch.isfinite(net.q_scalar(x)).all()
    # процедурные карты: ноль буферов
    pr = MV.procedural_records(30, seed=0)
    allm = np.concatenate([r["main"] for r in pr])
    flat = np.mean([np.ptp(m[0]) == 0 for m in allm])
    assert np.isfinite(allm).all() and allm.shape[1:] == (3, 26, 26) and 0.05 < flat < 0.6, flat
    assert (allm[:, 0] >= allm[:, 1] - 1e-5).all() and (allm[:, 0] >= allm[:, 2] - 1e-5).all(), "общий >= частей"
    argv = ["--tag", "w55p", "--procedural", "40", "--configs", "ssl-mae-resnet", "--pretrain-only", "--no-upload",
            "--out-dir", out, "--cpu", "--batch", "16", "--no-amp", "--ssl-steps", "4"]
    rep2 = CV.run(TCV._args(argv), QUIET)
    assert rep2["data"]["pdes"] == ["procedural"] and rep2["encoders"]["ssl-mae-resnet"]["n_maps"] > 200
    files["proc"] = os.path.join(out, "cvprobe_w55p_enc_mae_resnet.pt")
    # отказы
    for bad, text in ((["--tag", "x", "--views", vroot, "--subdirs", SUB, "--configs", "ssl-mae-resnet"], "pretrain-only"),
                      (["--tag", "x", "--procedural", "5", "--configs", "ssl-mv-resnet", "--pretrain-only", "--cpu",
                        "--no-upload", "--out-dir", out, "--ssl-steps", "2", "--batch", "8"], "ssl-mv"),
                      (["--tag", "x", "--views", vroot, "--subdirs", f"{SUB},ns2d_liddriven", "--configs",
                        "ssl-mae-resnet", "--pretrain-only", "--cpu", "--no-upload", "--out-dir", out], "одна папка")):
        try:
            CV.run(TCV._args(bad), QUIET)
            raise AssertionError(f"должен быть отказ: {text}")
        except SystemExit as e:
            assert text in str(e.code), (text, e.code)
    # умолчания: ssl-mv не в порядке по умолчанию, mae на обычных данных прежний
    assert not any(c.startswith("ssl-mv") for c in CV.DEFAULT_CONFIGS)
    tab, maps = CV.synthetic_data(200, seed=0)
    D = CV.Data(tab, maps, CPU)
    assert D.vkey is None and not D.map_only
    base = ["--tag", "d", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp"]
    a = CV.pretrain("mae", "resnet", D, TCV._args(base), CPU, log=QUIET)
    b = CV.pretrain("mae", "resnet", D, TCV._args(base + ["--mv-w", "5", "--mv-dim", "8"]), CPU, log=QUIET)
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
    print(f"cv_probe --views (ssl-mae, ssl-mv: пар {parts['pairs']:.2f} пачки) и --procedural "
          f"({rep2['encoders']['ssl-mae-resnet']['n_maps']} карт, плоских {flat:.2f}): файлы формата MAE, ложатся "
          f"в агента, отказы и умолчания: OK")
    return files


def test_offline_cli(tmp, files):
    """offline_rl.py --encoder-init с энкодером ssl-mv (две сети, контекст): main целиком."""
    root = TEP.make_bufs(os.path.join(tmp, "offl"))
    e0 = float(O.pde_meta()[O.subdir_to_pde(SUB)]["init_err"])
    for kind in ("mv", "proc"):
        argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", os.path.join(root, SUB), "--chain-fix",
                "--reward-form", "delta", "--init-err", repr(e0), "--value-bound", "--gamma", "0.99",
                "--episode-budget", "7000", "--ctx-kmax", "70", "--scalar-ctx", "--encoder", "resnet",
                "--encoder-init", files[kind], "--seeds", "1", "--epochs", "1", "--fqe-epochs", "1",
                "--batch-size", "8", "--cpu", "--model-tag", f"tw55{kind}{os.getpid()}", "--two-nets"]
        old = (killswitch.killed, O.GAMMA)
        killswitch.killed = lambda tag, remote=False, every=600: None
        try:
            out, code = TEP.run_main(argv)
        finally:
            killswitch.killed, O.GAMMA = old
        assert code is None, (code, out[-1500:])
        rows = [json.loads(ln) for ln in out.splitlines() if ln.startswith('{"variant"')]
        assert len(rows) == 1 and rows[0]["encoder_init"] == os.path.basename(files[kind]), out[-1500:]
        assert "(предобучение " in out and np.isfinite(rows[0]["td_error"])
    print("offline_rl.py --encoder-init с энкодерами ssl-mv и процедурного MAE (две сети, контекст): OK")


# --------------------------------------------------------------------------------------------
# render_views.py
# --------------------------------------------------------------------------------------------

def tr_of(acts, maps, seed, group=None):
    """Файл буфера в формате collect_active: state[j] — карта после j действий (шаг 0 нулевой)."""
    out = []
    z = {k: np.zeros((26, 26), np.float32) for k in TB.MAP_KEYS}
    for j, a in enumerate(acts):
        cur = z if j == 0 else {k: maps[j][i] for i, k in enumerate(TB.MAP_KEYS)}
        nxt = {k: maps[j + 1][i] for i, k in enumerate(TB.MAP_KEYS)} if (j + 1) in maps else cur
        out.append(dict(state=cur, next_state=nxt, action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                        reward=-0.1, done=0, epochs_real=E.ACTION_TABLE[a][2], group=group))
    out[-1]["done"] = 0 if len(acts) in maps else 1      # как chain_transitions: нет следующей карты — конец
    out[0]["meta"] = dict(seed=seed, env=dict(keep_opt=True, mode="safe", budget=7000))
    return out


RV_CALLS = []


def fake_run_seed(seed, args, progress_cb=None):
    """Контракт run_seed для render_views: скрипт, progress_cb, приёмник видов после каждой карты, мягкий выход."""
    t0 = time.time()
    script = E.parse_chain(args.script)
    RV_CALLS.append(dict(seed=seed, script=args.script, views=args.views, keep=args.keep_opt, mode=args.keep_opt_mode))
    assert args.views > 0 and not args.no_state and callable(args._view_sink)
    rng = np.random.default_rng(seed)
    chain, spent = [], 0
    for x in script:
        o, lr, ep = E.ACTION_TABLE[x]
        ep = min(ep, args.budget - spent)
        spent += ep
        chain.append([o, lr, ep])
        if progress_cb is not None and spent < args.budget:
            progress_cb(dict(n_steps=len(chain), spent=spent, l2re=0.1, rmse=0.04, brmse=0.01, elapsed_s=0.0))
        if spent >= args.budget:
            break
        s = len(chain)
        flat = o == "PSO"
        m3 = rmap(rng, flat=flat)
        want = args._view_want is None or args._view_want(s)
        vs = [m3 + 0.01 * (k + 1) for k in range(args.views)] if (want and not flat) else []
        args._view_sink(s, m3, vs, dict(flat=flat, k=len(vs), t_ae=0.0, t_srf=0.0))
        if args.hours and (time.time() - t0) / 3600.0 >= args.hours:
            return dict(seed=seed, unfinished=True, spent=spent, n_steps=len(chain))
    return dict(seed=seed, chain=chain, spent=spent, l2re=0.1, t_views_s=0.0)


def test_render_script(tmp):
    rng = np.random.default_rng(2)
    I = E.parse_action_spec
    A, L, P = I("Adam:0.001:1000"), I("LBFGS:1:500"), I("PSO:0:300")
    src = os.path.join(tmp, "src", SUB)
    os.makedirs(src)
    f1 = [A, L, A, P, P]                       # неплоские карты после 1-3 действий
    f2 = [A, L, I("Adam:0.0001:2500"), P]      # тот же сид и префикс: новое — только после 3 действий
    f3 = [L, P, P]                             # другой сид: неплоская карта после 1 действия
    for name, acts, sd, g in (("c0", f1, 11, "g0"), ("c1", f2, 11, "g0"), ("c2", f3, 22, None)):
        mp = {}
        for j in range(1, len(acts)):
            mp[j] = rmap(rng, flat=E.ACTION_TABLE[acts[j - 1]][0] == "PSO")
        torch.save(tr_of(acts, mp, sd, g), os.path.join(src, f"{name}.pt"))
    it = RV.source_item(os.path.join(src, "c0.pt"))
    assert it["steps"] == [1, 2, 3] and it["acts"] == f1[:3] and it["seed"] == 11 and it["env"]["keep_opt"]
    assert RV.source_item(os.path.join(src, "c2.pt"))["steps"] == [1]
    saved = E.run_seed
    E.run_seed = fake_run_seed
    killswitch_saved = killswitch.killed
    killswitch.killed = lambda tag, remote=False, every=600: None
    del RV_CALLS[:]
    try:
        out = os.path.join(tmp, "rv")
        base = ["--pde", PDE, "--source", os.path.join(tmp, "src"), "--views", "4", "--tag", "mvt", "--save-dir", out,
                "--no-upload"]
        with contextlib.redirect_stdout(io.StringIO()):
            st = RV.main(base + ["--worker", "0/1"])
        assert [c["script"] for c in RV_CALLS] == ["Adam:0.001:1000,LBFGS:1:500,Adam:0.001:1000",  # noqa: E127
                                                    "Adam:0.001:1000,LBFGS:1:500,Adam:0.0001:2500",
                                                    "LBFGS:1:500"], \
            ("повтор до последней неплоской карты", [c["script"] for c in RV_CALLS])
        assert all(c["views"] == 4 and c["keep"] and c["mode"] == "safe" for c in RV_CALLS), "среда из meta файла"
        d = os.path.join(out, "views", "mvt", SUB)
        recs = MV.load_records(d, SUB, QUIET)
        by = {r["_file"]: r for r in recs}
        assert sorted(by) == ["mvt_c0.pt", "mvt_c1.pt", "mvt_c2.pt"]
        assert sorted(by["mvt_c0.pt"]["views"]) == [1, 2, 3] and sorted(by["mvt_c1.pt"]["views"]) == [3], \
            "состояния общего префикса второй раз не строятся"
        assert all(v.shape == (4, 3, 26, 26) for r in recs for v in r["views"].values())
        assert st["finished"] and st["maps"] == sum(len(r["main"]) + sum(len(v) for v in r["views"].values()) for r in recs)
        R = MV.view_rows(recs)
        assert R["n_states"] == 5, R["n_states"]       # c0: 1, 2, 3; c1: 3; c2: 1
        # докатка: всё сделано — среда не вызывается; деление по сидам: группа целиком у одного работника
        del RV_CALLS[:]
        with contextlib.redirect_stdout(io.StringIO()):
            RV.main(base + ["--worker", "0/1", "--resume"])
        assert RV_CALLS == []
        items = [RV.source_item(os.path.join(src, f"c{i}.pt")) for i in range(3)]
        parts = [RV.assign(items, k, 2)[0] for k in range(2)]
        assert sorted(len(p) for p in parts) == [1, 2] and {i["seed"] for i in parts[0]}.isdisjoint({i["seed"] for i in parts[1]})
        # случайные цепочки: стоп после двух плоских карт подряд
        del RV_CALLS[:]
        with contextlib.redirect_stdout(io.StringIO()):
            st2 = RV.main(["--pde", PDE, "--random", "3", "--random-len", "8", "--views", "2", "--tag", "mvr",
                           "--save-dir", out, "--no-upload", "--flat-stop", "2", "--keep-opt", "--keep-opt-mode", "safe"])
        assert len(RV_CALLS) == 3 and len(st2["files"]) == 3
        for r in MV.load_records(os.path.join(out, "views", "mvr"), SUB, QUIET):
            fl = [s for s, i in r["info"].items() if i.get("flat")]
            run = 0
            for s in sorted(r["info"]):
                run = run + 1 if r["info"][s].get("flat") else 0
                assert run <= 2
            assert max(r["steps"]) <= len(r["chain"]) <= max(r["steps"]) + 1 and all(len(r["views"][s]) == 2 for s in r["views"])
            assert not set(fl) & set(r["views"])
        # отказы
        for bad in (["--pde", PDE, "--tag", "x"], ["--pde", PDE, "--tag", "x", "--random", "1", "--views", "0"],
                    ["--pde", PDE, "--tag", "x/y", "--random", "1"]):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    RV.main(bad)
                raise AssertionError(bad)
            except SystemExit:
                pass
    finally:
        E.run_seed, killswitch.killed = saved, killswitch_saved
    print("render_views.py: повтор до последней неплоской карты, префикс не строится дважды, среда из meta, "
          "деление по сидам, докатка, случайные цепочки со стопом по плоским картам: OK")


# --------------------------------------------------------------------------------------------
# collect_active.py --views
# --------------------------------------------------------------------------------------------

def collect_fake(seed, args, progress_cb=None):
    """Среда сбора test_collect_buffer с приёмником видов: после карты среды — 2 вида (карта + сдвиг)."""
    if getattr(args, "views", 0):
        assert callable(args._view_sink)
        orig = args._map_sink

        def sink(step, m):
            orig(step, m)
            m3 = np.array(m, np.float32)[:3]
            args._view_sink(step, m3, [m3 + 0.5 * (k + 1) for k in range(args.views)], dict(flat=False, k=args.views))
        args._map_sink = sink
    return TCB.fake_run_seed(seed, args, progress_cb)


def test_collect_views(tmp):
    with TCB.patched() as t2:
        C.run_seed = collect_fake
        ups = []
        C.upload_files = lambda pairs, msg, tries=9: (ups.append(list(pairs)), True)[1]
        C.upload = lambda st, name: None
        st = TCB.collect(t2, "cv", ["--rounds", "1", "--batch", "3", "--share-prefix", "--views", "2",
                                    "--progress-every", "1e9", "--init", f"{TCB.C1};{TCB.C_B};{TCB.C2}"], smoke=False)
        recs = st["records"]
        assert all(r.get("view_file") == r["buf_file"] for r in recs) and st["views"]["files"] == 3
        d = os.path.join(t2, "views", "cv", SUB)
        vr = {r["_file"]: r for r in MV.load_records(d, SUB, QUIET)}
        _, files = TCB.files_of(t2, "cv")
        for r in recs:
            v, tr = vr[r["view_file"]], files[r["buf_file"]]
            assert v["seed"] == r["seed"] and v["chain"] == [[c[0], float(c[1]), int(c[2])] for c in r["chain"]]
            for j in range(1, len(tr)):
                s = v["steps"].index(j)
                assert np.array_equal(v["main"][s], np.stack([tr[j]["state"][k] for k in TB.MAP_KEYS])), "вид 0 = буфер"
                assert v["views"][j].shape == (2, 3, 26, 26) and np.allclose(v["views"][j][0], v["main"][s] + 0.5)
        # концовки общего префикса несут виды префикса
        own, end = vr[recs[0]["view_file"]], vr[recs[1]["view_file"]]
        assert recs[1]["shared_prefix_len"] == 2 and np.array_equal(own["views"][1], end["views"][1])
        assert MV.view_rows(list(vr.values()))["n_states"] < sum(len(x["steps"]) for x in vr.values()), "повторы убраны"
        # заливка: файлы видов тем же коммитом, что буфер
        flat = [dst for p in ups for _, dst in p]
        assert sum(x.startswith(f"rl_arch/views/cv/{SUB}/") for x in flat) == 3 and len(ups) == 1, flat
        # без флага — прежние записи и ни одной папки видов; --views без --save-buffer — отказ
        st0 = TCB.collect(t2, "nv", ["--rounds", "1", "--batch", "2"])
        assert "views" not in st0 and not any("view_file" in r for r in st0["records"])
        assert not os.path.exists(os.path.join(t2, "views", "nv"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                C.main(["--pde", PDE, "--smoke", "--tag", "x", "--save-dir", t2, "--views", "2"])
            raise AssertionError("--views без --save-buffer")
        except SystemExit as e:
            assert "--save-buffer" in str(e.code)
    print("collect_active.py --views: файл видов на цепочку, вид 0 = карта буфера, виды префикса у концовок, "
          "заливка одним коммитом, без флага прежние записи: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            test_batched_aes(); lap()
            test_render_views(); lap()
            test_run_seed_views(); lap()
            test_view_rows(); lap()
            files = test_pretrain(tmp); lap()
            test_offline_cli(tmp, files); lap()
            test_render_script(tmp); lap()
            test_collect_views(tmp); lap()
        finally:
            os.chdir(cwd)
    print(f"test_map_views: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
