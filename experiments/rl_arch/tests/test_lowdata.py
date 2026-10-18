#!/usr/bin/env python
"""
Проверки предобучения энкодера на малом объёме данных цели (волна 61) без GPU и без сети, за секунды:

  * cv_probe.py --mix-real: процедурные карты и переходы буфера цели (--subdirs, --max-per-pde) одним набором —
    размеры частей, сдвиг цепочек и задач, обе части совпадают с отдельной загрузкой; --mix-real-frac — ровно
    доля строк буфера в каждой пачке и нормировка карт смеси; исход ssl-maeout только по строкам буфера; отказы;
  * map_views.render_random_planes: у «плоского» состояния (все копии весов совпали, карта среды постоянна) карты
    на случайных плоскостях не постоянны, форма (3, 26, 26), каналы и sign·log1p как у среды, центр плоскости —
    текущие веса, нормировка направлений по фильтрам (Li et al. 2018), направления плоскости у цепочки одни и те
    же, общие генераторы не тронуты;
  * online_eval_env.run_seed: приёмник _plane_sink вызывается после каждого действия (плоские карты и --no-state
    тоже), прогон прежний;
  * render_views.py --plane random: все состояния исходного файла (плоские тоже) с --planes картами, карта среды из
    файла, повтор без карт (--no-state), --max-per-seed, отказы; файлы видов читает cv_probe --views (предобучение
    с --ssl-d4);
  * --ssl-d4: элементы D4 — повороты и отражения карты, флаг включает D4 у mae (у con/dyn его не удваивает);
  * умолчания: без новых флагов предобучение, пачки и команды повтора прежние.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_lowdata.py
"""
import contextlib
import io
import os
import random
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

import cv_probe as CV  # noqa: E402
import map_views as MV  # noqa: E402
import offline_rl as O  # noqa: E402
import online_eval_env as E  # noqa: E402
import render_views as RV  # noqa: E402
import killswitch  # noqa: E402
import test_cv_probe as TCV  # noqa: E402
import test_map_views as TMV  # noqa: E402

torch.set_num_threads(1)
CPU = torch.device("cpu")
SUB = "poisson_boltzmann_2d"
PDE = "poissonboltzmann2d"
QUIET = lambda *a: None  # noqa: E731


@contextlib.contextmanager
def buffer_dir(root):
    old = os.environ.get("RL_BUFFER_DIR")
    os.environ["RL_BUFFER_DIR"] = root
    try:
        yield
    finally:
        os.environ.pop("RL_BUFFER_DIR") if old is None else os.environ.__setitem__("RL_BUFFER_DIR", old)


def make_buf(tmp, n_files=3, chains=5):
    rng = np.random.default_rng(0)
    root = os.path.join(tmp, "buf")
    os.makedirs(os.path.join(root, SUB), exist_ok=True)
    e0 = O.pde_meta()[O.subdir_to_pde(SUB)]["init_err"]
    for k in range(n_files):
        torch.save(TCV.make_dump(rng, chains, e0), os.path.join(root, SUB, f"f{k}.pt"))
    return root


# --------------------------------------------------------------------------------------------
# 1. Процедурные карты + свой буфер (--mix-real)
# --------------------------------------------------------------------------------------------

def test_mixed(tmp, root):
    base = ["--tag", "t", "--procedural", "20", "--subdirs", SUB, "--max-per-pde", "20"]
    with buffer_dir(root):
        tab, maps = CV.load_data(TCV._args(base + ["--mix-real"]), QUIET)
        tp, mp = CV.load_data(TCV._args(base), QUIET)                      # без --mix-real — только процедурные
        tb, mb = CV.load_data(TCV._args(["--tag", "t", "--subdirs", SUB, "--max-per-pde", "20"]), QUIET)
    nm, nr = len(mp), len(mb)
    assert len(maps) == nm + nr and maps.shape[1:] == (4, 26, 26) and 0 < nr <= 20, (nm, nr)
    assert nm == 20 * 9, nm                                                # 20 цепочек x (нулевая + 8 карт)
    assert np.array_equal(tab["REAL"], np.r_[np.zeros(nm, bool), np.ones(nr, bool)])
    assert np.array_equal(maps[:nm], mp) and np.array_equal(maps[nm:], mb), "части — те же, что по отдельности"
    assert tab["NAMES"] == ["procedural"] + list(tb["NAMES"]) and tab["SUBDIRS"][1] == SUB and set(tab["PDE"][:nm]) == {0} and set(tab["PDE"][nm:]) == {1}
    assert np.array_equal(tab["A"][nm:], tb["A"]) and np.allclose(tab["L"][nm:], tb["L"])
    assert len(set(tab["CH"][:nm]) & set(tab["CH"][nm:])) == 0 and tab["CTX"].shape == (2, 7)
    D = CV.Data(tab, maps, CPU)
    assert D.n_ch == len(np.unique(tp["CH"])) + len(np.unique(tb["CH"]))
    assert np.allclose(D.y[nm:], tb["LN"] - tb["L"]) and not D.y[:nm].any()
    # пачки: --mix-real-frac 0.5 — ровно половина строк буфера в каждой пачке
    args = TCV._args(base + ["--mix-real", "--mix-real-frac", "0.5", "--cpu", "--batch", "16", "--no-amp",
                             "--ssl-steps", "5"])
    seen = []
    fetch = D.fetch

    def spy(idx, prev=False):
        seen.append(idx.clone())
        return fetch(idx, prev)
    D.fetch = spy
    pre = CV.pretrain("mae", "resnet", D, args, CPU, log=QUIET)
    D.fetch = fetch
    fr = [float(D.real[i.numpy()].mean()) for i in seen]
    assert len(seen) == pre["steps"] >= 5 and all(f == 0.5 for f in fr) and all(len(i) == 16 for i in seen), fr
    real_rows = np.where(D.real & (D.step > 0))[0]
    used = np.unique(np.concatenate([i.numpy() for i in seen]))
    assert set(real_rows) <= set(used), "строки буфера идут по кругу и все используются"
    assert pre["n_maps_real"] == len(real_rows) and pre["real_per_batch"] == 8
    # нормировка смеси: взвешенная по доле, а не по объёму
    rows = np.where(D.step > 0)[0]
    m1, s1 = CV.map_stats(D, rows[D.real[rows]])
    m2, s2 = CV.map_stats(D, rows[~D.real[rows]])
    mean, std = pre["stats"]
    assert torch.allclose(mean, 0.5 * m1 + 0.5 * m2, atol=1e-5)
    xr, xm = (D.maps[torch.as_tensor(r)].float() for r in (rows[D.real[rows]], rows[~D.real[rows]]))
    mix_var = 0.5 * (xr.var((0, 2, 3), unbiased=True) + xr.mean((0, 2, 3)) ** 2) \
        + 0.5 * (xm.var((0, 2, 3), unbiased=True) + xm.mean((0, 2, 3)) ** 2) - mean.view(-1) ** 2
    assert torch.allclose(std.view(-1), mix_var.sqrt(), rtol=1e-4), (std.view(-1), mix_var.sqrt())
    # без доли — общий пул (прежние пачки по перестановке всех строк)
    assert CV.mix_split(D, rows, TCV._args(base + ["--mix-real"])) is None
    # целиком: предобучение-only, MAE и MAE + исход (исход только по строкам буфера)
    out = os.path.join(tmp, "cvmix")
    argv = base + ["--mix-real", "--mix-real-frac", "0.5", "--configs", "ssl-mae-resnet,ssl-maeout-resnet",
                   "--pretrain-only", "--no-upload", "--out-dir", out, "--cpu", "--batch", "16", "--no-amp",
                   "--ssl-steps", "4"]
    with buffer_dir(root):
        rep = CV.run(TCV._args(argv), QUIET)
    e1, e2 = rep["encoders"]["ssl-mae-resnet"], rep["encoders"]["ssl-maeout-resnet"]
    assert rep["finished"] and rep["data"]["pdes"] == ["procedural"] + list(tb["NAMES"])
    assert e1["ssl_args"] == dict(mae_ratio=0.6, mix_real_frac=0.5) and e1["n_maps"] == len(rows)
    assert e2["n_maps"] == len(real_rows) and np.isfinite(e2["loss_parts"]["out"]), e2
    ck = torch.load(os.path.join(out, "cvprobe_t_enc_mae_resnet.pt"), map_location="cpu", weights_only=False)
    assert ck["ssl_args"]["mix_real_frac"] == 0.5 and ck["pdes"] == ["procedural"] + list(tb["NAMES"])
    # отказы
    for bad, text in ((["--tag", "x", "--subdirs", SUB, "--mix-real", "--configs", "ssl-mae-resnet", "--pretrain-only"],
                       "--procedural или --views"),
                      (base + ["--mix-real-frac", "0.5", "--configs", "ssl-mae-resnet", "--pretrain-only"], "только с --mix-real"),
                      (base + ["--mix-real", "--mix-real-frac", "1", "--configs", "ssl-mae-resnet", "--pretrain-only"], "[0, 1)"),
                      (["--tag", "x", "--procedural", "5", "--configs", "ssl-out-resnet", "--pretrain-only", "--cpu",
                        "--no-upload", "--out-dir", out, "--ssl-steps", "2", "--batch", "8"], "нет строк с исходом")):
        try:
            with buffer_dir(root):
                CV.run(TCV._args(bad), QUIET)
            raise AssertionError(f"должен быть отказ: {text}")
        except SystemExit as e:
            assert text in str(e.code), (text, e.code)
    print(f"--mix-real: процедурных строк {nm} + буфера {nr}, части как по отдельности, в пачке ровно 0.5 буфера "
          f"({len(seen)} пачек), нормировка смеси, исход только по буферу ({e2['n_maps']} строк), отказы: OK")


# --------------------------------------------------------------------------------------------
# 2. Карты на случайных плоскостях
# --------------------------------------------------------------------------------------------

XP = torch.as_tensor(np.random.default_rng(3).uniform(-1, 1, (64, 2)), dtype=torch.float32)


class LossModel:
    """Модель лоссов в духе dde.Model для PINNLossData: сеть, обучающие точки, вектор лоссов (оператор, граница).
    Сеть строится общим генератором torch — как у настоящей фабрики (keep_rng обязан его вернуть)."""

    def __init__(self):
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh(),
                                       torch.nn.Linear(8, 1))
        self.data = types.SimpleNamespace(train_next_batch=lambda n: (XP, None, None))
        self.pde = types.SimpleNamespace(num_pde=1)
        self.losshistory = types.SimpleNamespace(loss_weights=None)

    def compile(self, optimizer, loss_weights=None):
        self.opt = optimizer

    def _outputs_losses(self, training, X, y, aux):
        with torch.no_grad():
            u = self.net(X)[:, 0]
            res = ((u - torch.sin(3 * X[:, 0]) * X[:, 1]) ** 2).mean()
            bnd = (u[:8] ** 2).mean() * 0.1
        return u.numpy(), np.array([float(res), float(bnd)])


def factory():
    return LossModel(), None


def direct_losses(net):
    m = LossModel()
    m.net.load_state_dict(net.state_dict())
    return m._outputs_losses(True, XP, None, None)[1]


def test_planes():
    torch.manual_seed(0)
    net = LossModel().net
    snaps = [net] * 10                       # «плоское» состояние: все копии весов совпали — карта среды постоянна
    assert len({id(s) for s in snaps}) == 1
    flat_env = np.full((3, 26, 26), 0.7, np.float32)
    assert TMV.MV.flat_raw({k: flat_env[i] for i, k in enumerate(MV.LOSS_KEYS)})
    g_before = torch.random.get_rng_state()
    n_before, p_before = np.random.get_state()[1].copy(), random.getstate()
    maps, info = MV.render_random_planes(net, factory, 3, E.GRID_RANGE, seed=11, scales=(1.0, 0.1))
    assert torch.equal(torch.random.get_rng_state(), g_before) and np.array_equal(np.random.get_state()[1], n_before)
    assert random.getstate() == p_before, "общие генераторы не тронуты"
    assert len(maps) == 3 and all(m.shape == (3, 26, 26) and m.dtype == np.float32 for m in maps)
    assert all(np.isfinite(m).all() and all(np.ptp(m[c]) > 1e-4 for c in range(3)) for m in maps), \
        "у плоского состояния карты на плоскостях не постоянны"
    assert not np.allclose(maps[0], maps[1]) and info["scales"] == [1.0, 0.1, 1.0]
    assert info["k"] == 3 and info["evals"] == 3 * 676 and info["t_per_map"] > 0
    # каналы: общий = оператор + граница (до log1p), преобразование sign·log1p
    raw = np.expm1(maps[0].astype(np.float64))
    assert np.allclose(raw[0], raw[1] + raw[2], rtol=1e-4, atol=1e-9)
    # центр плоскости — текущие веса: при малом масштабе карта постоянна и равна лоссу в точке
    tiny, _ = MV.render_random_planes(net, factory, 1, E.GRID_RANGE, seed=11, scales=(1e-9,))
    lv = direct_losses(net)
    want = np.log1p(np.array([lv.sum(), lv[0], lv[1]]) + 1e-12)
    assert np.allclose(tiny[0], want[:, None, None], rtol=1e-5), (tiny[0][:, 0, 0], want)
    # сетка та же, что у карт среды: 26 узлов от -1.25 до 1.25
    xs = MV.plane_coords(E.GRID_RANGE)
    assert len(xs) == 26 and abs(float(xs[0]) + 1.25) < 1e-6 and abs(float(xs[-1]) - 1.25) < 1e-5
    # направления: нормировка по фильтрам, смещения ноль; у той же плоскости — те же направления на другом шаге
    sd = net.state_dict()
    d = MV.filter_normed_direction(sd, torch.Generator().manual_seed(1))
    for k, w in sd.items():
        if w.dim() >= 2:
            assert torch.allclose(d[k].norm(dim=1), w.norm(dim=1), rtol=1e-5), k
        else:
            assert not d[k].any(), k
    sd2 = {k: 1.7 * v + 0.01 for k, v in sd.items()}           # «следующий шаг»: другие веса
    d2 = MV.filter_normed_direction(sd2, torch.Generator().manual_seed(1))
    for k, w in sd.items():
        if w.dim() >= 2:
            u1 = d[k] / w.norm(dim=1, keepdim=True)
            u2 = d2[k] / sd2[k].norm(dim=1, keepdim=True)
            assert torch.allclose(u1, u2, atol=1e-6), "направление то же, нормировка по весам шага"
    db = MV.filter_normed_direction(sd, torch.Generator().manual_seed(1), bias=True)
    assert all(torch.allclose(db[k].norm(), w.norm(), rtol=1e-5) for k, w in sd.items() if w.dim() == 1)
    again, _ = MV.render_random_planes(net, factory, 3, E.GRID_RANGE, seed=11, scales=(1.0, 0.1))
    assert all(np.array_equal(a, b) for a, b in zip(maps, again)), "сид плоскости воспроизводит карту"
    print(f"render_random_planes: у плоского состояния 3 непостоянные карты (3, 26, 26) за {info['t_srf']:.2f} с "
          f"({info['t_per_map']:.3f} с на карту, 676 оценок), каналы и центр как у среды, нормировка по фильтрам, "
          f"генераторы не тронуты: OK")
    return net


def test_env_hook():
    """run_seed с приёмником плоскостей (среда подменена, как в test_map_views): вызов после каждой карты, плоские
    тоже; с --no-state — после каждого действия; приёмник, тянущий генератор, прогон не меняет."""
    calls = {}

    def sink(step, net, get_model_rec):
        calls.setdefault(tag[0], []).append((int(step), float(net.w.detach())))
        torch.rand(5)                     # тянет общий генератор: среда обязана вернуть его состояние
        np.random.rand(3)
    tag = ["none"]
    with TMV.fake_env() as tmp:
        row0, main0, _ = TMV.env_run(tmp)
        argv = ["--policy", "script", "--script", TMV.SCRIPT, "--script-tail", "stop", "--pde", PDE, "--budget", "7000",
                "--save-dir", tmp, "--tag", "v", "--ckpt-every", "0", "--progress-every", "0", "--keep-opt",
                "--keep-opt-mode", "safe"]
        rows = {}
        for nm, extra in (("maps", []), ("nostate", ["--no-state"])):
            tag[0] = nm
            args = E.build_parser().parse_args(argv + extra)
            args._plane_sink = sink
            got = {}
            args._map_sink = lambda s, m: got.__setitem__(int(s), np.array(m, np.float32))
            with contextlib.redirect_stdout(io.StringIO()):
                rows[nm] = (E.run_seed(7, args), got)
    core = lambda r: (r["chain"], r["l2re"], r["spent"], [h[:3] for h in r["hist"]])  # noqa: E731
    assert core(row0) == core(rows["maps"][0]) == core(rows["nostate"][0]), "приёмник плоскостей не меняет прогон"
    assert all(np.array_equal(main0[s], rows["maps"][1][s]) for s in main0) and rows["nostate"][1] == {}
    steps = sorted(main0)
    flat = {s for s in steps if np.ptp(main0[s][0]) == 0}
    assert [s for s, _ in calls["maps"]] == steps and [s for s, _ in calls["nostate"]] == steps and flat, \
        (calls, steps, flat)
    print(f"run_seed: приёмник плоскостей после каждого действия ({len(steps)}, плоских {len(flat)}), и с --no-state; "
          f"прогон прежний: OK")


PLANE_CALLS = []


def fake_run_seed_planes(seed, args, progress_cb=None):
    """Контракт run_seed для render_views --plane random: скрипт, повтор без карт, приёмник плоскостей после каждого
    действия с сетью и фабрикой лоссов; PSO вес не двигает (застой)."""
    script = E.parse_chain(args.script)
    PLANE_CALLS.append(dict(seed=seed, script=args.script, no_state=args.no_state, views=args.views,
                            keep=args.keep_opt))
    assert callable(args._plane_sink) and args._view_sink is None
    torch.manual_seed(seed)
    net = LossModel().net
    chain, spent = [], 0
    for x in script:
        o, lr, ep = E.ACTION_TABLE[x]
        ep = min(ep, args.budget - spent)
        spent += ep
        chain.append([o, lr, ep])
        if o != "PSO":
            with torch.no_grad():
                for p in net.parameters():
                    p.mul_(1.0 + 0.05 * ep / 1000.0)
        if progress_cb is not None and spent < args.budget:
            progress_cb(dict(n_steps=len(chain), spent=spent, l2re=0.1, rmse=0.04, brmse=0.01, elapsed_s=0.0))
        if spent >= args.budget:
            break
        with MV.keep_rng():
            args._plane_sink(len(chain), net, factory)
    return dict(seed=seed, chain=chain, spent=spent, l2re=0.1, t_opt_s=1.0)


def test_render_planes(tmp):
    rng = np.random.default_rng(5)
    I = E.parse_action_spec
    A, L, P = I("Adam:0.001:1000"), I("LBFGS:1:500"), I("PSO:0:300")
    src = os.path.join(tmp, "src_rd", SUB)
    os.makedirs(src)
    files = (("c0", [A, L, P, P, L], 11, "g0"), ("c1", [A, L, L, P], 11, "g0"), ("c2", [L, P, P], 22, None))
    fmaps = {}
    for name, acts, sd, g in files:
        mp = {j: TMV.rmap(rng, flat=E.ACTION_TABLE[acts[j - 1]][0] == "PSO") for j in range(1, len(acts))}
        fmaps[name] = mp
        torch.save(TMV.tr_of(acts, mp, sd, g), os.path.join(src, f"{name}.pt"))
    it = RV.source_item(os.path.join(src, "c0.pt"), all_states=True)
    assert it["steps"] == [1, 2, 3, 4] and sorted(it["maps"]) == [1, 2, 3, 4] and it["acts"] == files[0][1][:4]
    assert RV.source_item(os.path.join(src, "c0.pt"))["steps"] == [1, 2], "без режима — только неплоские"
    saved, saved_k = E.run_seed, killswitch.killed
    E.run_seed = fake_run_seed_planes
    killswitch.killed = lambda tag, remote=False, every=600: None
    del PLANE_CALLS[:]
    try:
        out = os.path.join(tmp, "rd")
        base = ["--pde", PDE, "--source", os.path.join(tmp, "src_rd"), "--plane", "random", "--planes", "2",
                "--plane-scales", "1,0.1", "--tag", "rdt", "--save-dir", out, "--no-upload"]
        with contextlib.redirect_stdout(io.StringIO()):
            st = RV.main(base + ["--max-per-seed", "1"])
        assert len(PLANE_CALLS) == 2 and all(c["no_state"] and c["views"] == 0 and c["keep"] for c in PLANE_CALLS), \
            ("повтор без карт среды и без видов автокодировщика", PLANE_CALLS)
        recs = MV.load_records(os.path.join(out, "views", "rdt"), SUB, QUIET)
        by = {r["_file"]: r for r in recs}
        assert sorted(by) == ["rdt_c0.pt", "rdt_c2.pt"], "--max-per-seed 1: по одному файлу на сид"
        r0 = by["rdt_c0.pt"]
        assert sorted(r0["views"]) == [1, 2, 3, 4], "карты на плоскостях у всех состояний, плоских тоже"
        assert all(v.shape == (2, 3, 26, 26) for v in r0["views"].values())
        flat_steps = [s for s in r0["steps"] if np.ptp(r0["main"][r0["steps"].index(s)][0]) == 0]
        assert flat_steps == [3, 4] and all(np.ptp(r0["views"][s][0][0]) > 0 for s in flat_steps), \
            "у плоских состояний — непостоянные карты"
        for s in r0["steps"]:
            assert np.array_equal(r0["main"][r0["steps"].index(s)], fmaps["c0"][s]), "вид 0 — карта среды из файла"
        assert r0["plane"]["kind"] == "random" and r0["plane"]["scales"] == [1.0, 0.1] and not r0["plane"]["env_maps"]
        assert r0["info"][3]["plane"] == "random" and r0["info"][3]["flat"] and not r0["info"][1]["flat"]
        assert st["plane_maps"] == 2 * (4 + 2) and st["plane_s"] > 0 and st["finished"], st
        R = MV.view_rows(recs)
        assert R["n_states"] == 6 and R["maps"].shape[1:] == (4, 26, 26) and set(R["VIEW"]) == {0, 1, 2}
        # канал delta плоскости — по той же плоскости предыдущего шага
        k = np.where((R["VIEW"] == 1) & (R["STEP"] == 2))[0][0]
        assert np.allclose(R["maps"][k, 3], MV._delta(r0["views"][2][0][0], r0["views"][1][0][0]))
        # без --max-per-seed: c1 того же сида повторяет префикс — его состояния второй раз не строятся
        del PLANE_CALLS[:]
        with contextlib.redirect_stdout(io.StringIO()):
            RV.main(base + ["--tag", "rdt2"])
        r1 = {r["_file"]: r for r in MV.load_records(os.path.join(out, "views", "rdt2"), SUB, QUIET)}["rdt2_c1.pt"]
        assert sorted(r1["views"]) == [3] and len(PLANE_CALLS) == 3, sorted(r1["views"])
        # отказы
        for bad in (base + ["--random", "2"], ["--pde", PDE, "--random", "1", "--tag", "x", "--plane-env-maps"],
                    base + ["--planes", "0"], base + ["--plane-scales", "0"], base + ["--max-per-seed", "-1"]):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    RV.main(bad)
                raise AssertionError(bad)
            except SystemExit:
                pass
    finally:
        E.run_seed, killswitch.killed = saved, saved_k
    # умолчание: команды повтора прежние (виды автокодировщика, карты среды строятся)
    a = RV.build_parser().parse_args(["--pde", PDE, "--source", "x", "--tag", "t"])
    assert a.plane == "ae" and a.planes == 8 and a.max_per_seed == 0 and not a.plane_env_maps
    ea = RV.env_args(a, dict(acts=[A, L], env=dict(budget=7000, keep_opt=True, mode="safe")))
    assert ea.views == 12 and not ea.no_state
    ar = RV.build_parser().parse_args(["--pde", PDE, "--source", "x", "--tag", "t", "--plane", "random"])
    er = RV.env_args(ar, dict(acts=[A, L], env=dict(budget=7000, keep_opt=True, mode="safe")))
    assert er.views == 0 and er.no_state and er.keep_opt
    print("render_views.py --plane random: все состояния (плоские тоже) по 2 карты, вид 0 из файла, повтор без карт, "
          "--max-per-seed, префикс не повторяется, отказы, умолчания прежние: OK")
    return os.path.join(out, "views", "rdt")


def test_planes_pretrain(tmp, vroot):
    out = os.path.join(tmp, "cvrd")
    argv = ["--tag", "w61t", "--subdirs", SUB, "--views", vroot, "--configs", "ssl-mae-resnet", "--pretrain-only",
            "--no-upload", "--out-dir", out, "--cpu", "--batch", "8", "--no-amp", "--ssl-steps", "3", "--ssl-d4"]
    rep = CV.run(TCV._args(argv), QUIET)
    e = rep["encoders"]["ssl-mae-resnet"]
    assert rep["finished"] and e["n_maps"] == 6 * 3 and e["ssl_args"] == dict(mae_ratio=0.6, ssl_d4=True), e
    ck = torch.load(os.path.join(out, "cvprobe_w61t_enc_mae_resnet.pt"), map_location="cpu", weights_only=False)
    net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", two_nets=True)
    info = O.load_encoder_init(net, os.path.join(out, "cvprobe_w61t_enc_mae_resnet.pt"))
    assert info["ssl"] == "mae" and ck["ssl_args"]["ssl_d4"]
    print(f"cv_probe --views на картах случайных плоскостей ({e['n_maps']} карт, --ssl-d4): энкодер ложится в агента "
          "с двумя сетями: OK")


# --------------------------------------------------------------------------------------------
# 3. D4 при предобучении
# --------------------------------------------------------------------------------------------

def test_d4():
    x = torch.randn(1, 4, 26, 26)
    xn = x[0].numpy()
    group = [np.rot90(xn, r, axes=(1, 2)) for r in range(4)] + [np.rot90(xn[:, :, ::-1], r, axes=(1, 2)) for r in range(4)]
    for gi in range(8):
        assert np.array_equal(O.d4_apply(x, gi)[0].numpy(), group[gi]), gi
    assert all(not np.array_equal(group[i], group[j]) for i in range(8) for j in range(i)), "8 разных элементов"
    g = torch.Generator().manual_seed(0)
    xb = x.expand(256, -1, -1, -1).contiguous()
    a = CV.aug_d4(xb, g)
    hit = set()
    for i in range(len(a)):
        m = [gi for gi in range(8) if np.array_equal(a[i].numpy(), group[gi])]
        assert len(m) == 1, "каждая карта — поворот или отражение исходной"
        hit.add(m[0])
    assert hit == set(range(8)), hit
    # флаг: D4 включается у mae и меняет предобучение; у con (D4 уже в видах) не удваивается; без флага — прежнее
    tab, maps = CV.synthetic_data(200, seed=0)
    D = CV.Data(tab, maps, CPU)
    base = ["--tag", "d", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp"]
    cnt = [0]
    orig = CV.aug_d4

    def counting(t, gen=None):
        cnt[0] += 1
        return orig(t, gen)
    CV.aug_d4 = counting
    try:
        res = {}
        for kind in ("mae", "con"):
            for flag in (False, True):
                cnt[0] = 0
                res[kind, flag] = (CV.pretrain(kind, "resnet", D, TCV._args(base + (["--ssl-d4"] if flag else [])), CPU,
                                               log=QUIET), cnt[0])
    finally:
        CV.aug_d4 = orig
    (p0, c0), (p1, c1) = res["mae", False], res["mae", True]
    assert c0 == 0 and c1 == p1["steps"], (c0, c1)
    assert any(not torch.equal(p0["state"][k], p1["state"][k]) for k in p0["state"] if p0["state"][k].is_floating_point())
    assert res["con", False][1] == res["con", True][1] > 0
    assert all(torch.equal(res["con", False][0]["state"][k], res["con", True][0]["state"][k]) for k in p0["state"])
    assert CV.ssl_args("mae", TCV._args(base + ["--ssl-d4"])) == dict(mae_ratio=0.6, ssl_d4=True)
    assert CV.ssl_args("con", TCV._args(base + ["--ssl-d4"])) == dict(con_temp=0.2, shift=2)
    print("--ssl-d4: 8 элементов D4 — повороты и отражения, у mae по одному на пачку, у con не удваивается: OK")


# --------------------------------------------------------------------------------------------
# 4. Умолчания
# --------------------------------------------------------------------------------------------

def test_defaults():
    a = TCV._args(["--tag", "t"])
    assert not a.mix_real and a.mix_real_frac == 0.0 and not a.ssl_d4
    tab, maps = CV.synthetic_data(200, seed=0)
    D = CV.Data(tab, maps, CPU)
    assert D.real is None and D.has_y is None
    base = ["--tag", "d", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp"]
    # прежний путь пачек: перестановка всех строк общим генератором, без смеси и без D4
    seen = []
    fetch = D.fetch

    def spy(idx, prev=False):
        seen.append(idx.clone())
        return fetch(idx, prev)
    D.fetch = spy
    p = CV.pretrain("mae", "resnet", D, TCV._args(base), CPU, log=QUIET)
    D.fetch = fetch
    torch.manual_seed(0 + 17)
    _, rows = CV.pretrain_rows("mae", D)
    enc = CV.make_encoder("resnet", D.C, D.H)             # тот же порядок, что в pretrain: энкодер, голова MAE
    torch.nn.Sequential(torch.nn.Linear(enc.out_dim, 512), torch.nn.GELU(), torch.nn.Linear(512, D.C * D.H * D.W))
    rows_t = torch.as_tensor(rows)
    bs = 16
    per = len(rows) // bs
    want = []
    for _ in range(p["epochs"]):
        perm = rows_t[torch.randperm(len(rows_t))]
        want += [perm[i:i + bs] for i in range(0, len(perm) - bs + 1, bs)]
        break                                                 # первая эпоха: дальше генератор тянут маски и шаги
    assert all(torch.equal(a_, b_) for a_, b_ in zip(seen[:per], want)), "пачки первой эпохи — прежняя перестановка"
    q = CV.pretrain("mae", "resnet", D, TCV._args(base + ["--mix-real-frac", "0"]), CPU, log=QUIET)
    assert all(torch.equal(p["state"][k], q["state"][k]) for k in p["state"])
    assert set(CV.ssl_args("mae", TCV._args(base))) == {"mae_ratio"} and "n_maps_real" not in p
    print("умолчания: пачки и предобучение без новых флагов прежние, отчёт без новых полей: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            root = make_buf(tmp)
            test_mixed(tmp, root); lap()
            test_planes(); lap()
            test_env_hook(); lap()
            vroot = test_render_planes(tmp); lap()
            test_planes_pretrain(tmp, vroot); lap()
            test_d4(); lap()
            test_defaults(); lap()
        finally:
            os.chdir(cwd)
    print(f"test_lowdata: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
