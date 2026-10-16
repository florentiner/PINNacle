#!/usr/bin/env python
"""
Проверки исследования данных MAE (mae_data_study.py, mae_study_report.py; волна 56) без GPU и без сети, на
синтетике за секунды:

  * примитивы анализа: k-NN радиус и покрытие (одно распределение покрыто, сдвинутое — нет), MMD (одно
    распределение ~0 < разные), линейная CKA (поворот и масштаб — 1, независимые признаки — мало), гребневый
    зонд по цепочкам находит линейный сигнал, жадное ядро берёт разные узоры, а дубликат даёт ~0 выигрыша;
  * mae_train — та же ветка mae, что cv_probe.pretrain: при тех же строках и сиде веса энкодера совпадают;
  * режим study на пяти синтетических «задачах» (syn_a и syn_b — один узор, как два хаотических уравнения):
    анализ в пространствах raw, shape и mae (ENC_MAE — энкодер, предобученный здесь же cv_probe), пара
    syn_a~syn_b найдена избыточной, рисунки записаны, армы с прокси, повторы по строкам обучения помечены
    same_as, файлы энкодеров читаются, --resume ничего не обучает заново, --hours останавливает;
  * режим cost с подменённым подпроцессом: цена перехода из замеров, нерабочее уравнение — в missing,
    таблица действий совпадает с surrogate (эпохи и семейства);
  * отчёт mae_study_report.py по этим файлам (без сети): все разделы, приведение кернелов цены по общему
    уравнению, GPU-часы армов, жадное ядро с учётом цены;
  * lint_queue: команды study и cost волны 56 проходят разбор аргументов.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_mae_study.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ["HF_HUB_OFFLINE"] = "1"
for _k in ("HF_TOKEN_WRITE", "HF_TOKEN"):      # проверка ничего не выгружает
    os.environ.pop(_k, None)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "queue"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import torch  # noqa: E402

import cv_probe as CV  # noqa: E402
import mae_data_study as M  # noqa: E402
import mae_study_report as R  # noqa: E402

torch.set_num_threads(1)
CPU = torch.device("cpu")
QUIET = lambda *a: None  # noqa: E731
TINY = ["--synthetic", "600", "--targets", "syn_t,syn_d", "--chaotic", "syn_a,syn_b", "--ssl-steps", "3",
        "--batch", "32", "--no-upload", "--cpu", "--no-amp", "--ana-per-pde", "80", "--tsne-per-pde", "30",
        "--mmd-n", "30", "--heldout-per-pde", "20", "--eval-n", "20", "--cka-n", "50", "--probe-n", "100",
        "--core-sizes", "1,2,3", "--core-n", "60", "--single-n", "60", "--size-sweep", "10,30", "--kc-sizes", "5,20"]


def test_primitives():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(300, 5, generator=g)
    A2 = torch.randn(300, 5, generator=g)
    Bf = torch.randn(300, 5, generator=g) + 6.0
    real = {"a": A, "b": Bf}
    fake = {"a": A2, "b": torch.randn(300, 5, generator=g) + 6.0}
    Bt, valid = M.coverage_tensor(real, fake, ["a", "b"], 5, [10, 0], seed=0)
    C = np.array([[float(M.union_cov(Bt, valid, [p])[t]) for p in range(2)] for t in range(2)])
    assert C[0, 0] > 0.9 and C[1, 1] > 0.9 and C[0, 1] < 0.05 and C[1, 0] < 0.05, C
    assert float(M.union_cov(Bt, valid, [0], j=0)[0]) < C[0, 0], "10 подставных покрывают меньше, чем все"
    r = M.knn_radius(A, 5)
    assert r.shape == (300,) and (r > 0).all()
    same, diff = M.mmd2(A, A2, 2.0), M.mmd2(A, Bf, 2.0)
    assert abs(same) < 0.02 < diff, (same, diff)
    Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=g))
    assert abs(M.linear_cka(A, 3.0 * A @ Q) - 1.0) < 1e-6
    assert M.linear_cka(A, A2) < 0.2
    X = np.random.default_rng(0).normal(size=(400, 6))
    y = X[:, 0] * 2.0 - X[:, 3] + np.random.default_rng(1).normal(0, 0.1, 400)
    grp = np.arange(400) // 8
    r2, lam = M.ridge_cv(X, y, grp)
    r2n, _ = M.ridge_cv(np.random.default_rng(2).normal(size=(400, 6)), y, grp)
    assert r2 > 0.95 and r2n < 0.05 and lam is not None, (r2, r2n)
    # жадное ядро: a и b — один узор, c — свой, t (цель) — смесь a и c
    pats = {"a": 0.0, "b": 0.0, "c": 8.0, "t": None}
    real, fake = {}, {}
    for k, mu in pats.items():
        def draw(n):
            if mu is None:
                return torch.cat([torch.randn(n // 2, 3, generator=g), torch.randn(n - n // 2, 3, generator=g) + 8.0])
            return torch.randn(n, 3, generator=g) + mu
        real[k], fake[k] = draw(200), draw(200)
    names = list(pats)
    Bt, valid = M.coverage_tensor(real, fake, names, 5, [0], seed=1)
    core = M.greedy_core(Bt, valid, [0, 1, 2], [3], names)
    assert set(core["order"][:2]) in ({"a", "c"}, {"b", "c"}), core["order"]
    assert core["curve"][2]["gain"] < 0.05 and core["curve"][1]["cov_targets"] > 0.8 and core["k95"] == 2, core["curve"]
    kc = M.kcenter_order(torch.cat([fake["a"], fake["c"]]), 10)
    assert len(set(kc.tolist())) == 10
    print("покрытие, MMD, CKA, гребневый зонд, жадное ядро и k-center: OK")


def test_mae_matches_cv_probe():
    tab, maps = CV.synthetic_data(200, seed=1)
    D = CV.Data(tab, maps, CPU)
    cva = CV.build_parser().parse_args(["--tag", "t", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp"])
    ma = M.parse(["study", "--tag", "t", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp", "--synthetic", "200"])
    pre = CV.pretrain("mae", "resnet", D, cva, CPU, log=QUIET)
    enc, aux, stats, info = M.mae_train(D, CV.pretrain_rows("mae", D)[1], ma, CPU, seed=0)
    sd = enc.state_dict()
    assert all(torch.equal(pre["state"][k], sd[k].cpu()) for k in pre["state"]), "веса разошлись с cv_probe"
    assert info["steps"] == pre["steps"] and abs(info["loss_last"] - pre["loss_last"]) < 1e-9
    assert all(torch.equal(a, b) for a, b in zip(stats, pre["stats"]))
    print(f"mae_train совпадает с cv_probe.pretrain (mae): {info['steps']} шагов, лосс {info['loss_last']:.4f}: OK")


def make_ref(tmp):
    """ENC_MAE-заменитель: cv_probe-энкодер, предобученный на той же синтетике (формат --save-encoder)."""
    tab, maps = M.synthetic_pdes(600, seed=0)
    D = CV.Data(tab, maps, CPU)
    a = CV.build_parser().parse_args(["--tag", "ref", "--cpu", "--batch", "32", "--ssl-steps", "3", "--no-amp",
                                      "--no-upload", "--out-dir", tmp])
    with contextlib.redirect_stdout(io.StringIO()):
        return CV.save_encoder(CV.pretrain("mae", "resnet", D, a, CPU, log=QUIET), "mae", "resnet", D, a, log=QUIET)


def test_study(tmp):
    ref = make_ref(tmp)
    out = os.path.join(tmp, "study")
    argv = ["study", "--tag", "syn", "--enc-ref", ref, "--out-dir", out] + TINY
    with contextlib.redirect_stdout(io.StringIO()):
        rep = M.run_study(M.parse(argv), QUIET)
    assert rep["finished"] and rep["targets"] == ["syn_t", "syn_d"]
    assert set(rep["analysis"]) == {"raw", "shape", "mae"}
    raw = rep["analysis"]["raw"]
    assert [(p["a"], p["b"]) for p in raw["redundant_pairs"]] == [("syn_a", "syn_b")], raw["redundant_pairs"]
    assert raw["chaotic"]["pair"] == ["syn_a", "syn_b"] and raw["chaotic"]["a_by_b"] > 0.8
    assert len(raw["coverage"]) == 5 and len(raw["core_free"]["order"]) == 3
    assert "syn_t" not in raw["core_free"]["order"] and "syn_d" not in raw["core_free"]["order"]
    assert raw["kcenter"]["m"] == [5, 20] and raw["per_maps"]["free"][-1]["n"] == 0
    for p in ("raw_pca", "raw_tsne", "raw_coverage", "raw_mmd", "raw_dendro", "raw_coreset", "raw_permaps",
              "raw_kcenter", "mae_coverage", "proxies"):
        assert os.path.exists(os.path.join(out, f"maestudy_syn_{p}.png")), p
    arms = rep["arms"]
    for name in ("all", "rand", "enc_mae", "free", "free_n60", "core_raw_k1", "best_n10", "one_syn_a",
                 "w46_syn_t_n130", "w53_n30", "free_nochaos_n60", "rand3_s0", "free_n60_s1"):
        assert name in arms, name
    assert abs(arms["enc_mae"]["cka_ref_mean"] - 1.0) < 1e-6
    tr = [n for n, e in arms.items() if "same_as" not in e and n not in ("rand", "enc_mae")]
    for n in tr:
        e = arms[n]
        assert e["steps"] >= 3 and np.isfinite(e["recon_targets_mean"]) and e["n_maps"] > 0, n
        assert set(e["recon_targets"]) == {"syn_t", "syn_d"} and "cka_ref" in e and "probe" in e, n
        assert os.path.exists(os.path.join(out, "encoders", f"maestudy_syn_{n}.pt")), n
    assert arms["w46_syn_t_n1k"].get("same_as") == "w46_syn_t_n130", "пул меньше объёма — тот же энкодер"
    assert all(arms[n].get("cka_all_mean") is not None for n in tr if n != "all")
    assert abs(arms["rand"]["recon_targets_mean"] - 1.0) < 0.1, "нулевое предсказание формы ~1"
    with open(rep["_paths"]["report"]) as f:
        js = json.load(f)
    assert "_ref" not in js and js["best_core"]["space"] == "raw" and js["data"]["per_pde"]["syn_a"]["actions"]
    enc, stats = M.load_encoder_file(os.path.join(out, "encoders", "maestudy_syn_core_raw_k1.pt"),
                                     CV.Data(*M.synthetic_pdes(100), CPU), CPU)
    assert enc(torch.zeros(2, 4, 26, 26)).shape == (2, 256) and stats[0].shape == (1, 4, 1, 1)
    ck = torch.load(os.path.join(out, "encoders", "maestudy_syn_all.pt"), weights_only=False)
    assert ck["kind"] == "ssl_encoder" and ck["ssl"] == "mae" and ck["encoder"] == "resnet"
    # --resume: всё посчитано — ничего не обучается заново
    saved = M.mae_train

    def boom(*a, **k):
        raise AssertionError("--resume обучил арм заново")
    M.mae_train = boom
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = M.run_study(M.parse(argv + ["--resume"]), QUIET)
    finally:
        M.mae_train = saved
    assert rep2["finished"] and set(rep2["arms"]) == set(arms)
    # --hours: после анализа ни один арм не начинается
    with contextlib.redirect_stdout(io.StringIO()):
        rep3 = M.run_study(M.parse(["study", "--tag", "syn3", "--enc-ref", "none", "--spaces", "raw", "--hours", "1e-9",
                                    "--out-dir", os.path.join(tmp, "s3")] + TINY), QUIET)
    assert not rep3["finished"] and rep3["stopped_before"] == "all" and not rep3["arms"]
    print(f"study на синтетике: 3 пространства, избыточная пара syn_a~syn_b, армов {len(arms)} "
          f"(обучено {len(tr)}), энкодеры, --resume и --hours: OK")
    return out


def fake_child(pde, args, log):
    if pde == "wave1d":
        return None, "код 1: Traceback ... RuntimeError: синтетическая ошибка"
    k = 2.0 if args.tag == "cb" else 1.0
    return dict(pde=pde, device="FakeGPU", actions=[dict(opt="Adam", lr=1e-3, epochs=1000, s=10.0 * k),
                                                    dict(opt="LBFGS", lr=0.5, epochs=500, s=20.0 * k),
                                                    dict(opt="PSO", lr=1e-3, epochs=200, s=2.0 * k)],
                ae_s=[30.0 * k, 40.0 * k], surface_s=[60.0 * k, 70.0 * k]), None


def test_cost(tmp):
    assert [e for _, _, e in M.ACTIONS] == [int(x) for x in CV.S.EPOCHS[:27]]
    assert [("Adam", "LBFGS", "PSO").index(o) for o, _, _ in M.ACTIONS] == [int(x) for x in CV.S.FAM[:27]]
    assert M.cost_pdes(M.parse(["cost", "--tag", "x", "--pdes", "poisson_boltzmann_2d,wave1d"])) == \
        ["poissonboltzmann2d", "wave1d"], "папки буферов переводятся в имена задач"
    saved = M.CHILD, M.cost_pdes
    M.CHILD = fake_child
    M.cost_pdes = lambda args: list(args.pdes)          # синтетические «задачи» в pde_meta нет
    out = os.path.join(tmp, "cost")
    try:
        for tag, pdes in (("ca", "poissonboltzmann2d,wave1d,syn_a,syn_b,syn_c"), ("cb", "poissonboltzmann2d,syn_d,syn_t")):
            a = M.parse(["cost", "--tag", tag, "--pdes", "wave1d", "--no-upload", "--out-dir", out])
            a.pdes = pdes.split(",")
            with contextlib.redirect_stdout(io.StringIO()):
                rep = M.run_cost(a, QUIET)
    finally:
        M.CHILD, M.cost_pdes = saved
    d = rep["rows"]["syn_d"]["derived"]
    assert abs(d["sec_per_epoch"]["Adam"] - 0.02) < 1e-12 and abs(d["map_s"] - 200.0) < 1e-9
    rca = json.load(open(os.path.join(out, "maecost_ca.json")))
    d1 = rca["rows"]["poissonboltzmann2d"]["derived"]
    assert abs(d1["opt_s"] - (12.0 + 21.0 + 1.0 / 3.0 + 2.0) / 3.0) < 1e-9 and abs(d1["map_s"] - 100.0) < 1e-9
    assert "wave1d" in rca["missing"] and "wave1d" not in rca["rows"] and rca["finished"]
    print(f"цена перехода: {d1['per_transition_min']:.3f} мин (Adam 0.01 с/эп, LBFGS 0.04, PSO 0.01, карты 100 с), "
          f"нерабочее уравнение — в missing: OK")
    return out


def test_report(study_dir, cost_dir):
    rd = os.path.join(os.path.dirname(study_dir), "rep")
    os.makedirs(rd)
    for d in (study_dir, cost_dir):
        for f in os.listdir(d):
            if f.endswith((".json", ".npz")):
                os.link(os.path.join(d, f), os.path.join(rd, f))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = R.main(["--tag", "syn", "--cost-tags", "ca,cb", "--dir", rd, "--offline", "--space", "raw"])
    txt = buf.getvalue()
    for k in ("core", "redundancy", "proxies", "cost", "candidates", "cost_greedy"):
        assert res["sections"].get(k), k
    cost = res["cost"]
    assert cost["syn_d"]["scale"] == 0.5 and abs(cost["syn_d"]["map_s"] - 100.0) < 1e-9, "приведение к первому кернелу"
    assert cost["poisson_boltzmann_2d"]["tag"] == "ca" and "wave1d" in res["missing"]
    assert cost["syn_a"]["min_mix"] is not None, "частоты действий буфера из отчёта study"
    rows = {r["name"]: r for r in R.proxy_rows(res["study"], cost)}
    assert rows["all"]["gpu_h"] is not None and not rows["all"]["approx"]
    ones = [r["gpu_h"] for n, r in rows.items() if n.startswith("one_") and "same_as" not in r]
    assert ones and max(ones) < rows["all"]["gpu_h"], "одно уравнение дешевле всех"
    assert "syn_a~syn_b" in txt and "НЕТ ЦЕНЫ wave1d" in txt and "жадно по выигрышу покрытия" in txt
    print("отчёт: шесть разделов, приведение кернелов цены, GPU-часы армов, ядро с учётом цены: OK")


def test_lint():
    import lint_queue as LQ
    scr = "experiments/rl_arch/mae_data_study.py"
    assert LQ.check(scr, "study --tag w56 --hours 7.5 --resume --out-dir /kaggle/working") is None
    assert LQ.check(scr, "cost --tag w56a --pdes poisson_boltzmann_2d,wave1d --hours 7 --resume") is None
    assert LQ.check(scr, "study --tag w56 --spaces foo") is not None, "неизвестное пространство отвергается"
    assert LQ.check(scr, "cost --tag w56a --cost-script Adam:0.5:7") is not None, "чужое действие отвергается"
    print("lint_queue: study и cost разбираются, ошибки флагов ловятся: OK")


def main():
    t0 = time.time()
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_primitives()
            test_mae_matches_cv_probe()
            sdir = test_study(tmp)
            cdir = test_cost(tmp)
            test_report(sdir, cdir)
            test_lint()
    finally:
        os.chdir(cwd)
    print(f"ВСЕ ПРОВЕРКИ ИССЛЕДОВАНИЯ ДАННЫХ MAE ПРОЙДЕНЫ ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
