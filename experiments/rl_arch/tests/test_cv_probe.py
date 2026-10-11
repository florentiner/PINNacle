#!/usr/bin/env python
"""
Проверки зонда CV на картах (cv_probe.py) без GPU и без сети, на крошечных синтетических тензорах
(несколько сотен переходов, две «задачи»; y зависит от знака наклона карты, которого не видно ни в
уровне карты, ни после перестановки пикселей):

  * построение входа: shape совпадает с offline_rl.apply_state_mode, перестановка пикселей
    сохраняет значения и теряет порядок, D4 и сдвиг сохраняют форму тензора, маска блоками;
  * загрузчик на файлах в формате дампа: карты выровнены с таблицей surrogate.build_table,
    предыдущий шаг цепочки, ограничение --max-per-pde целыми цепочками;
  * каждый энкодер строится на 4 и 8 каналах и проходит одну эпоху;
  * полный порядок конфигов (все входы, все приёмы CV, lopo): отчёт JSON с таблицей и вердиктом,
    --resume не обучает ничего повторно и досчитывает недостающее, --hours останавливает;
  * на данных с пространственным сигналом maps выигрывает у level, shuffle — нет.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_cv_probe.py
"""
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
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import cv_probe as CV  # noqa: E402

torch.set_num_threads(1)                    # крошечные тензоры: потоки только мешают
QUIET = lambda *a: None  # noqa: E731
KEYS = ["loss_total", "loss_oper", "loss_bnd"]


def _args(argv):
    a = CV.build_parser().parse_args(argv)
    a._argv = argv
    return a


def test_inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(6, 4, 26, 26, generator=g) * 3 + 1
    x[0, 1] = 2.5                                             # постоянная карта -> нули
    ref = O.apply_state_mode(x.numpy(), "shape")
    assert np.allclose(CV.shape_norm(x).numpy(), ref, atol=1e-4), "shape расходится с apply_state_mode"
    sh = CV.shuffle_pixels(x, g)
    assert torch.equal(sh.flatten(2).sort(-1)[0], x.flatten(2).sort(-1)[0]), "значения каналов сохраняются"
    assert not torch.equal(sh, x)
    d = lambda t: (t[:, 0] - t[:, 2]).flatten(1).sort(-1)[0]  # noqa: E731
    assert torch.allclose(d(sh), d(x)), "общая перестановка сохраняет пары значений каналов в пикселе"
    assert not torch.allclose(d(CV.shuffle_pixels(x, g, indep=True)), d(x))
    a = CV.aug_d4(x, g)
    assert a.shape == x.shape and torch.allclose(a.flatten(2).sort(-1)[0], x.flatten(2).sort(-1)[0])
    assert torch.equal(CV.aug_shift(x, g, 0), x)
    s = CV.aug_shift(x, g, 2)
    assert s.shape == x.shape and not torch.equal(s, x)
    col = torch.arange(26.0).view(1, 1, 1, 26).expand(4, 1, 26, 26)
    sc = CV.aug_shift(col, g, 2)                              # сдвиг вдоль x: столбцы остаются столбцами
    assert torch.equal(sc[:, :, :1].expand_as(sc), sc) and (sc[:, 0, 0, 13] - 13).abs().max() <= 2
    m = CV.block_mask(64, 26, 26, 0.6, g)
    assert m.shape == (64, 1, 26, 26) and 0.5 < m.float().mean() < 0.7
    stats = (torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1))
    assert CV.build_input("stack2", x, x * 0, stats).shape == (6, 8, 26, 26)
    assert torch.equal(CV.build_input("delta", x, x * 0, stats)[:, 4:], x)
    tab, maps = CV.synthetic_data(60, seed=0)
    f, cols = CV.features(tab)
    assert f.shape == (60, 60) and cols["lv"] == slice(54, 60) and np.isfinite(f).all()
    assert CV.parse_config("frozen-con") == ("frozen-con", None) and CV.parse_config("shape-vit") == ("shape", "vit")
    assert CV.expand_configs("all") == CV.DEFAULT_CONFIGS and "maps-vit" in CV.expand_configs("level,maps")
    try:
        CV.expand_configs("maps-foo")
        raise AssertionError("неизвестный конфиг должен отвергаться")
    except SystemExit:
        pass
    print("построение входа, аугментации, признаки, разбор конфигов: OK")


def make_dump(rng, n_chains, e0):
    """Файл в формате дампа (старый формат: next_state верный), случайные карты."""
    out = []
    for _ in range(n_chains):
        chain = [int(a) for a in rng.integers(0, 27, int(rng.integers(3, 6)))]
        maps = [{k: np.zeros((26, 26), np.float32) for k in KEYS}]
        l, errs = np.log10(e0), []
        for a in chain:
            l += rng.normal(-0.1, 0.1)
            errs.append(10 ** l)
            maps.append({k: rng.random((26, 26)).astype(np.float32) + 1.0 for k in KEYS})
        for j, a in enumerate(chain):
            done = -1 if j == len(chain) - 1 else 0
            out.append(dict(state=dict(maps[j]), next_state=dict(maps[j + 1]),
                            action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                            reward=-(errs[j] + (1.0 if done == -1 else 0.0)), done=done))
    return out


def test_loader(tmp):
    rng = np.random.default_rng(0)
    root = os.path.join(tmp, "buf")
    meta = O.pde_meta()
    for sd in ("wave1d", "ns2d_liddriven"):
        os.makedirs(os.path.join(root, sd))
        for k in range(2):
            torch.save(make_dump(rng, 4, meta[O.subdir_to_pde(sd)]["init_err"]), os.path.join(root, sd, f"f{k}.pt"))
    old = os.environ.get("RL_BUFFER_DIR")
    os.environ["RL_BUFFER_DIR"] = root
    try:
        tab, maps = CV.load_data(_args(["--tag", "t", "--subdirs", "wave1d,ns2d_liddriven"]), QUIET)
        cap_tab, cap_maps = CV.load_data(_args(["--tag", "t", "--subdirs", "wave1d,ns2d_liddriven",
                                                "--max-per-pde", "12", "--map-dtype", "float16"]), QUIET)
    finally:
        os.environ.pop("RL_BUFFER_DIR") if old is None else os.environ.__setitem__("RL_BUFFER_DIR", old)
    assert tab["NAMES"] == ["wave1d", "ns2d_liddriven"] and maps.shape[1:] == (4, 26, 26)
    assert len(maps) == len(tab["A"]) and maps.dtype == np.float32
    assert np.allclose(S.level6(maps), tab["LV"], atol=1e-5)
    assert not maps[tab["STEP"] == 0].any() and maps[tab["STEP"] > 0][:, 0].min() >= 1.0
    assert len(np.unique(tab["CH"])) == 16 and (np.diff(tab["PDE"]) >= 0).all()
    D = CV.Data(tab, maps, torch.device("cpu"))
    x, xp = D.fetch(torch.arange(D.n), prev=True)
    later = tab["STEP"] > 0
    assert torch.equal(xp[later], x[np.where(later)[0] - 1]) and not xp[~later].any()
    assert np.allclose(D.y, tab["LN"] - tab["L"]) and D.n_ch == 16
    assert cap_maps.dtype == np.float16
    for p in range(2):
        r = cap_tab["PDE"] == p
        assert 0 < r.sum() <= 12
        for c in np.unique(cap_tab["CH"][r]):                # цепочки остаются целыми и с первого шага
            st = cap_tab["STEP"][cap_tab["CH"] == c]
            assert st[0] == 0 and (np.diff(st) == 1).all()
    CV.Data(cap_tab, cap_maps, torch.device("cpu"))
    print(f"загрузчик на дампах: {len(maps)} переходов, карты выровнены, предыдущий шаг, ограничение: OK")


def test_encoders():
    tab, maps = CV.synthetic_data(40, seed=1)
    dev = torch.device("cpu")
    D = CV.Data(tab, maps, dev)
    args = _args(["--tag", "t", "--epochs", "1", "--batch", "16"])
    tr, te = np.arange(0, 30), np.arange(30, 40)
    for kind in CV.ENCODERS:
        for c in (4, 8):
            enc = CV.make_encoder(kind, c, 26)
            z = enc(torch.zeros(3, c, 26, 26))
            assert z.shape == (3, enc.out_dim), (kind, c, z.shape)
        t0 = time.time()
        pred, info = CV.fit_predict(CV.spec_of(f"maps-{kind}", None), D, tr, te, args, dev, seed=0)
        assert pred.shape == (10,) and np.isfinite(pred).all() and info["params"] > 0 and not info["skipped"]
        assert np.isfinite(info["train_r2"])
        print(f"  {kind}: параметров {info['params']}, эпоха {time.time() - t0:.2f} с")
    assert CV.make_encoder("conv", 4).enc.__class__.__name__ == "ConvEncoder", "энкодер статьи берётся из RL/"
    print("энкодеры на 4 и 8 каналах, одна эпоха: OK")


def test_run_all(tmp):
    """Весь порядок по умолчанию на 160 переходах, затем --resume и предел времени."""
    out = os.path.join(tmp, "all")
    calls = []
    saved = CV.load_data, CV.fit_predict

    def counted(sp, *a, **k):
        calls.append(sp["base"])
        return saved[1](sp, *a, **k)
    CV.load_data = lambda args, log=print: CV.synthetic_data(160, seed=2)
    CV.fit_predict = counted
    try:
        base = ["--tag", "t", "--smoke", "--out-dir", out, "--lopo", "syn_0,syn_1", "--encoder", "cnn",
                "--batch", "64", "--boot", "100"]
        t0 = time.time()
        rep = CV.run(_args(base + ["--configs", "scalars,level,maps-conv"]), QUIET)
        assert rep["finished"] and set(rep["configs"]) == {"scalars", "level", "maps-conv"}
        n1 = len(calls)
        assert n1 == 6, calls
        rep = CV.run(_args(base + ["--resume"]), QUIET)       # досчитывает остальное
        assert len(calls) > n1 and calls[n1:].count("scalars") == 0 and calls[n1:].count("level") == 2, calls   # level: только lopo
        path = rep["_paths"]["report"]
        assert path == os.path.join(out, "cvprobe_t.json") and os.path.exists(rep["_paths"]["chains"])
        with open(path) as f:
            js = json.load(f)
        assert js["finished"] and set(js["configs"]) == set(CV.DEFAULT_CONFIGS), sorted(js["configs"])
        for name, e in js["configs"].items():
            if e["kind"] == "lopo":
                assert e["done"] and set(e["pdes"]) == {"syn_0", "syn_1"}, name
                assert all(np.isfinite(v["r2_y"]) for v in e["pdes"].values())
                continue
            assert len(e["fold_r2"]) == 2 and np.isfinite(e["r2_y"]) and e["n"] == 160, name
            assert set(e["per_pde"]) == {"syn_0", "syn_1"} and e["train_s"] >= 0
            assert (e["encoder"] is None) == (name in ("scalars", "level", "level-hgb")), name
            if name != "level":
                g = e["gain_over_level"]["r2_y"]
                assert g[1] <= g[2] and np.isfinite(g).all() and set(e["gain_over_level"]["per_pde"]) == {"syn_0", "syn_1"}, name
                assert abs(g[0] - (e["r2_y"] - js["configs"]["level"]["r2_y"])) < 1e-4, name
        assert js["configs"]["shape"]["encoder"] == "cnn" and js["configs"]["maps-vit"]["encoder"] == "vit"
        assert js["configs"]["frozen"]["params_trainable"] < js["configs"]["frozen"]["params"]
        assert js["configs"]["ssl-mae"]["pretrain_s"] > 0 and js["configs"]["lopo:best"]["config"]
        sm = js["summary"]
        assert len(sm["table"]) == len(CV.DEFAULT_CONFIGS) - 2 and sm["verdict"] and sm["lopo"]
        gains = [r["gain"] for r in sm["table"]]
        assert gains == sorted(gains, reverse=True), "таблица отсортирована по выигрышу"
        assert sm["in_dist"]["config"] == sm["best_maps_config"] and sm["spatial"] is not None
        assert isinstance(sm["maps_add_information"], bool)
        n2 = len(calls)
        rep2 = CV.run(_args(base + ["--resume"]), QUIET)      # всё готово: ни одного обучения
        assert len(calls) == n2 and rep2["finished"]
        assert rep2["configs"]["maps-conv"]["r2_y"] == js["configs"]["maps-conv"]["r2_y"]
        os.remove(rep["_paths"]["chains"])                    # суммы по цепочкам потеряны: level заново
        rep3 = CV.run(_args(base + ["--resume", "--configs", "level,mapslv"]), QUIET)
        assert calls[n2:] == ["level", "level", "mapslv", "mapslv"], calls[n2:]
        assert rep3["configs"]["mapslv"]["gain_over_level"]["r2_y"][1] is not None
        rep4 = CV.run(_args(base[:1] + ["t2"] + base[2:] + ["--hours", "1e-9"]), QUIET)
        assert not rep4["finished"] and rep4["stopped_before"] == "scalars" and not rep4["configs"]
        assert os.path.exists(os.path.join(out, "cvprobe_t2.json"))
        try:                                                  # другой набор данных под тем же тегом
            CV.load_data = lambda args, log=print: CV.synthetic_data(120, seed=3)
            CV.run(_args(base + ["--resume"]), QUIET)
            raise AssertionError("--resume на других данных должен останавливаться")
        except SystemExit as e:
            assert "других данных" in str(e)
        print(f"полный порядок конфигов ({n2} обучений), отчёт, продолжение, предел времени: "
              f"OK ({time.time() - t0:.1f} с)")
    finally:
        CV.load_data, CV.fit_predict = saved


def test_signal(tmp):
    """Пространственный сигнал: карты выигрывают у уровня, перестановка пикселей — нет."""
    t0 = time.time()
    rep = CV.run(_args(["--tag", "sig", "--synthetic", "240", "--out-dir", os.path.join(tmp, "sig"),
                        "--no-upload", "--configs", "level,maps-cnn,shuffle-cnn", "--folds", "2",
                        "--epochs", "6", "--batch", "16", "--lr", "2e-3", "--boot", "200", "--seed", "1"]), QUIET)
    c, sm = rep["configs"], rep["summary"]
    gm, gs = c["maps-cnn"]["gain_over_level"]["r2_y"], sm["spatial"]["gain_over_shuffle"]
    print(f"  R² y: level {c['level']['r2_y']:.3f}, maps {c['maps-cnn']['r2_y']:.3f}, "
          f"shuffle {c['shuffle-cnn']['r2_y']:.3f}; maps над level {gm[0]:+.3f} [{gm[1]:+.3f}, {gm[2]:+.3f}], "
          f"maps над shuffle {gs[0]:+.3f} [{gs[1]:+.3f}, {gs[2]:+.3f}]")
    assert c["level"]["r2_y"] < 0.3 and c["shuffle-cnn"]["r2_y"] < 0.3, "уровень и перестановка наклона не видят"
    assert c["maps-cnn"]["r2_y"] > 0.5 and gm[0] > 0.3 and gm[1] > 0, "карты выигрывают у уровня, интервал выше нуля"
    assert gs[0] > 0.3 and gs[1] > 0 and sm["spatial"]["shuffle_config"] == "shuffle-cnn"
    assert c["maps-cnn"]["rank_within"] > c["level"]["rank_within"] + 0.2
    assert len(c["maps-cnn"]["fold_train_r2"]) == 2 and c["maps-cnn"]["fold_train_r2"][0] > 0.5
    assert sm["best_maps_config"] == "maps-cnn" and sm["in_dist"]["beats_level"]
    assert sm["spatial"]["structure_matters"] and sm["maps_add_information"] is None, "без lopo вердикт неполный"
    print(f"сигнал в форме карты: maps выигрывает, shuffle нет: OK ({time.time() - t0:.1f} с)")


def main():
    t0 = time.time()
    test_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        test_loader(tmp)
        test_encoders()
        test_run_all(tmp)
        test_signal(tmp)
    print(f"test_cv_probe: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
