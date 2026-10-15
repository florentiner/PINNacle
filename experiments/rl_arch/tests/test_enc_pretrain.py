#!/usr/bin/env python
"""
Проверки замен MAE в предобучении энкодера (волна 49, cv_probe.py: ssl-vae, ssl-dyn, ssl-out, ssl-maeout) без
GPU и без сети, на синтетических дампах буфера (две задачи, случайные карты) и синтетическом наборе зонда:

  * индекс следующего шага цепочки (ssl-dyn): тот же номер цепочки и шаг + 1, у последнего шага -1;
  * --pretrain-only с каждым новым конфигом и --ssl-steps: несколько шагов, лосс конечен, веса энкодера
    сдвинулись от инициализации; файл cvprobe_<tag>_enc_<вид>_resnet.pt того же формата, что у MAE
    (kind, encoder, ssl, state_dict, mean, std), плюс параметры вида (ssl_args) и части лосса (loss_parts);
  * offline_rl.load_encoder_init кладёт каждый файл в ResNet агента с контекстом (9 каналов, одна сеть и две
    сети), ключи совпадают, Q считается; offline_rl.py --encoder-init целиком (main, две сети) на синтетическом
    буфере печатает вид предобучения;
  * режим фолдов: ssl-out помечен ssl_saw_labels и не идёт в «лучший конфиг с картами»;
  * умолчания: новых конфигов нет в порядке по умолчанию, флаги новых видов не меняют mae и con.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_enc_pretrain.py
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
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import torch  # noqa: E402

import cv_probe as CV  # noqa: E402
import offline_rl as O  # noqa: E402
import test_cv_probe as TC  # noqa: E402

torch.set_num_threads(1)
CPU = torch.device("cpu")
NEW = ("vae", "dyn", "out", "maeout")
SUBDIR, OTHER = "poisson_boltzmann_2d", "ns2d_liddriven"
QUIET = lambda *a: None  # noqa: E731


def test_next_index():
    tab, maps = CV.synthetic_data(200, seed=1)
    D = CV.Data(tab, maps, CPU)
    i = np.where(D.nxt >= 0)[0]
    assert len(i) and (D.ch[D.nxt[i]] == D.ch[i]).all() and (D.step[D.nxt[i]] == D.step[i] + 1).all()
    last = np.r_[D.ch[1:] != D.ch[:-1], True]
    assert (D.nxt[last] == -1).all() and (D.nxt[~last] >= 0).all()
    st, tr = CV.pretrain_rows("dyn", D)
    assert (D.step[st] > 0).all() and set(tr) <= set(st) and (D.nxt[tr] >= 0).all()
    assert np.array_equal(CV.pretrain_rows("out", D)[0], CV.pretrain_rows("mae", D)[0]), "статистики — по тем же строкам"
    print(f"следующий шаг цепочки: {len(i)} пар из {D.n} строк, у последних шагов -1: OK")


def make_bufs(tmp):
    rng = np.random.default_rng(0)
    root = os.path.join(tmp, "buf")
    meta = O.pde_meta()
    for sd in (SUBDIR, OTHER):
        os.makedirs(os.path.join(root, sd))
        for k in range(2):
            torch.save(TC.make_dump(rng, 6, meta[O.subdir_to_pde(sd)]["init_err"]), os.path.join(root, sd, f"f{k}.pt"))
    return root


def test_pretrain_and_load(tmp, root):
    old = os.environ.get("RL_BUFFER_DIR")
    os.environ["RL_BUFFER_DIR"] = root
    out = os.path.join(tmp, "cv")
    cfgs = ",".join(f"ssl-{k}-resnet" for k in NEW)
    argv = ["--tag", "w49t", "--subdirs", f"{SUBDIR},{OTHER}", "--configs", cfgs, "--pretrain-only", "--no-upload",
            "--out-dir", out, "--cpu", "--batch", "8", "--no-amp", "--ssl-steps", "6"]
    try:
        rep = CV.run(TC._args(argv), QUIET)
    finally:
        os.environ.pop("RL_BUFFER_DIR") if old is None else os.environ.__setitem__("RL_BUFFER_DIR", old)
    assert rep["kind"] == "cv_probe_pretrain" and rep["finished"] and len(rep["data"]["pdes"]) == 2
    torch.manual_seed(17)                        # инициализация энкодера в pretrain: manual_seed(seed + 17)
    init = CV.make_encoder("resnet", 4, 26).state_dict()
    files = {}
    for kind in NEW:
        e = rep["encoders"][f"ssl-{kind}-resnet"]
        assert e["file"] == f"rl_arch/encoders/cvprobe_w49t_enc_{kind}_resnet.pt", e["file"]
        assert e["steps"] >= 6 and np.isfinite(e["loss_first"]) and np.isfinite(e["loss_last"]), e
        assert all(np.isfinite(v) for v in e["loss_parts"].values()), e
        path = os.path.join(out, f"cvprobe_w49t_enc_{kind}_resnet.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["kind"] == "ssl_encoder" and ck["encoder"] == "resnet" and ck["ssl"] == kind
        assert set(ck["state_dict"]) == set(init) and ck["mean"].shape == (1, 4, 1, 1) and ck["std"].shape == (1, 4, 1, 1)
        assert ck["ssl_steps"] == e["steps"] and ck["ssl_args"] == e["ssl_args"], (ck["ssl_args"], e["ssl_args"])
        moved = [k for k, v in init.items() if v.dtype.is_floating_point and not torch.equal(v, ck["state_dict"][k])]
        assert moved, f"{kind}: веса энкодера не сдвинулись"
        files[kind] = path
    assert rep["encoders"]["ssl-dyn-resnet"]["n_maps"] < rep["encoders"]["ssl-vae-resnet"]["n_maps"], \
        "у ssl-dyn нет последних шагов цепочек"
    assert set(rep["encoders"]["ssl-dyn-resnet"]["loss_parts"]) == {"spr", "inv", "inv_acc"}
    assert set(rep["encoders"]["ssl-maeout-resnet"]["loss_parts"]) == {"mae", "out"}
    assert rep["encoders"]["ssl-vae-resnet"]["ssl_args"] == dict(vae_beta=1.0, vae_dim=32)
    # энкодер ложится в агента (offline_rl.py --encoder-init): одна сеть и две сети, контекст — свежие каналы
    x = torch.randn(3, 9, 26, 26)
    for kind, path in files.items():
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        for two in (False, True):
            net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", two_nets=two)
            info = O.load_encoder_init(net, path)
            assert info["encoder"] == "resnet" and info["ssl"] == kind and info["partial"] == ["stem.0.weight"], info
            for i in getattr(net, "_enc_idx", (0,)):
                own = net.model[i].state_dict()
                for k, v in sd.items():
                    got = own[k][:, :v.shape[1]] if k in info["partial"] else own[k]
                    assert torch.equal(got, v.to(got.dtype)), (kind, two, k)
            net.model.eval()
            with torch.no_grad():
                q = net.q_scalar(x)
            assert q.shape == (3, 27) and torch.isfinite(q).all(), (kind, two)
    print(f"--pretrain-only {', '.join(NEW)}: {rep['encoders']['ssl-vae-resnet']['steps']} шагов, файлы формата MAE, "
          f"ложатся в агента (одна и две сети), Q конечны: OK")
    return files


def run_main(argv):
    old = sys.argv
    out, code = io.StringIO(), None
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                O.main()
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = old
    return out.getvalue(), code


def test_offline_cli(root, files):
    """offline_rl.py --encoder-init с файлом нового вида: main() целиком (две сети, как лидер ns2d)."""
    import killswitch
    e0 = float(O.pde_meta()[O.subdir_to_pde(SUBDIR)]["init_err"])
    mtag = f"tw49{os.getpid()}"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", os.path.join(root, SUBDIR), "--chain-fix",
            "--reward-form", "delta", "--init-err", repr(e0), "--value-bound", "--gamma", "0.99",
            "--episode-budget", "7000", "--ctx-kmax", "70", "--scalar-ctx", "--encoder", "resnet",
            "--encoder-init", files["dyn"], "--seeds", "1", "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8",
            "--cpu", "--model-tag", mtag, "--two-nets"]
    old = (killswitch.killed, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    try:
        out, code = run_main(argv)
    finally:
        killswitch.killed, O.GAMMA = old
    assert code is None, (code, out[-1500:])
    rows = [json.loads(ln) for ln in out.splitlines() if ln.startswith('{"variant"')]
    assert len(rows) == 1 and rows[0]["encoder_init"] == os.path.basename(files["dyn"]), out[-1500:]
    assert "(предобучение dyn)" in out and "свежие входные каналы" in out, out[-1500:]
    assert np.isfinite(rows[0]["td_error"])
    print("offline_rl.py --encoder-init с энкодером ssl-dyn (две сети, контекст): обучение прошло: OK")


def test_cv_mode_and_defaults(tmp):
    # режим фолдов: ssl-out видел y отложенных фолдов — помечен и не выбирается лучшим
    out = os.path.join(tmp, "cvf")
    argv = ["--tag", "w49f", "--synthetic", "200", "--configs", "maps-resnet,ssl-out-resnet", "--folds", "2",
            "--epochs", "1", "--ssl-steps", "3", "--batch", "16", "--boot", "20", "--no-upload", "--cpu", "--no-amp",
            "--out-dir", out]
    with contextlib.redirect_stdout(io.StringIO()):
        rep = CV.run(TC._args(argv), QUIET)
    assert rep["configs"]["ssl-out-resnet"].get("ssl_saw_labels") is True
    assert "ssl_saw_labels" not in rep["configs"]["maps-resnet"]
    assert CV.best_maps_config(rep) == "maps-resnet"
    # умолчания: новые конфиги только по имени; флаги новых видов не трогают mae и con
    assert not any(c.startswith(("ssl-vae", "ssl-dyn", "ssl-out", "ssl-maeout")) for c in CV.DEFAULT_CONFIGS)
    assert CV.expand_configs("all") == list(dict.fromkeys(CV.DEFAULT_CONFIGS))
    tab, maps = CV.synthetic_data(200, seed=0)
    D = CV.Data(tab, maps, CPU)
    base = ["--tag", "d", "--cpu", "--batch", "16", "--ssl-steps", "4", "--no-amp"]
    other = ["--vae-beta", "4", "--vae-dim", "8", "--dyn-ema", "0.5", "--dyn-inv", "3", "--out-w", "7"]
    for kind in ("mae", "con"):
        a = CV.pretrain(kind, "resnet", D, TC._args(base), CPU, log=QUIET)
        b = CV.pretrain(kind, "resnet", D, TC._args(base + other), CPU, log=QUIET)
        assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]), kind
        assert a["loss_parts"] == {} and CV.ssl_args(kind, TC._args(base + other)) == CV.ssl_args(kind, TC._args(base))
    print("режим фолдов: ssl-out помечен ssl_saw_labels, лучшим не выбран; умолчания и mae/con не изменились: OK")


def main():
    t0 = time.time()
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_next_index()
            root = make_bufs(tmp)
            files = test_pretrain_and_load(tmp, root)
            test_offline_cli(root, files)
            test_cv_mode_and_defaults(tmp)
    finally:
        os.chdir(cwd)
    print(f"ВСЕ ПРОВЕРКИ ПРЕДОБУЧЕНИЯ ВОЛНЫ 49 ПРОЙДЕНЫ ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
