#!/usr/bin/env python
"""
Проверки волн 46-47 без GPU и без сети (PINN и автокодировщик не обучаются: среда — детерминированная
заглушка test_online_finetune, карты — функция истории действий, канал delta строит настоящий
online_eval_env.build_state; буферы — синтетические дампы).

Волна 46 (cv_probe.py):
  * --pretrain-only на картах одной задачи (--subdirs <папка>, --max-per-pde N целыми цепочками): в
    данных только эта задача, ни level, ни фолдов; файл энкодера пишется, в нём одна задача, число шагов
    и карт; offline_rl.load_encoder_init кладёт его в ResNet агента (9 каналов);
  * --ssl-steps задаёт число шагов (эпохи — под объём карт), без флага — прежние --ssl-epochs;
  * отказ на конфиге без предобучения.
Волна 47 (online_train_env.py):
  * --save-chains: файл на цепочку (и оборванную лимитом времени) в формате target_buffer.py, метка конца
    прогона; переходы, прочитанные --preload-online из файлов ДВУХ раундов, равны тем, что тренер положил в
    свой буфер онлайн (--save-buffer того же прогона): карты, канал delta, контекст s и s', действие,
    награда, конец, ошибка;
  * ученик --learn-only: среда не вызывается, K обновлений по загруженным переходам, агент читается
    online_eval_env.load_agent и даёт те же Q, что сеть тренера; meta с источниками;
  * работник следующего раунда: строгий тёплый старт от агента ученика, загрузка прошлых раундов,
    --pre-updates до первого действия (в журнале раньше первого шага, веса сдвинулись), затем среда;
  * прежняя командная строка без новых флагов: ни файлов цепочек, ни новых полей в meta; отказы разбора.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_scratch_online.py
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import tempfile
import time
import zlib

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
import offline_rl as O  # noqa: E402
import online_eval_env as E  # noqa: E402
import online_train_env as T  # noqa: E402
import test_online_finetune as TF  # noqa: E402

REAL_BUILD_STATE = E.build_state        # заглушка среды подменяет его; здесь нужен настоящий канал delta
CPU = torch.device("cpu")
PDE, SUBDIR = TF.PDE, TF.SUBDIR
BUDGET = 600
ONLY100 = ",".join(str(a) for a in range(27) if a % 3)     # только действия по 100 эпох: 6 шагов на цепочку
COMMON = ["--variant", "convnext_dqn", "--encoder", "resnet", "--pde", PDE, "--scalar-ctx", "--ctx-no-err",
          "--init-err", "0.811", "--value-bound", "--gamma", "0.99", "--episode-budget", str(BUDGET),
          "--tolerance", "0", "--max-chain-steps", "70", "--keep-opt", "--keep-opt-mode", "safe",
          "--batch-size", "8", "--min-buffer", "4", "--mask", ONLY100]


# ---------------------------------------------------------------- волна 46: cv_probe
def test_cv_pretrain(tmp):
    import cv_probe as CV
    import test_cv_probe as TC
    rng = np.random.default_rng(0)
    root = os.path.join(tmp, "buf")
    meta = O.pde_meta()
    for sd in (SUBDIR, "ns2d_liddriven"):
        os.makedirs(os.path.join(root, sd))
        for k in range(2):
            torch.save(TC.make_dump(rng, 5, meta[O.subdir_to_pde(sd)]["init_err"]), os.path.join(root, sd, f"f{k}.pt"))
    old = os.environ.get("RL_BUFFER_DIR")
    os.environ["RL_BUFFER_DIR"] = root
    out = os.path.join(tmp, "cv")
    base = ["--tag", "w46t", "--subdirs", SUBDIR, "--configs", "ssl-mae-resnet", "--pretrain-only", "--no-upload",
            "--out-dir", out, "--cpu", "--batch", "8", "--no-amp"]
    lines = []
    log_ = lambda *a: lines.append(" ".join(str(x) for x in a))  # noqa: E731
    try:
        rep = CV.run(TC._args(base + ["--max-per-pde", "12", "--ssl-steps", "7"]), log_)
        rep_all = CV.run(TC._args([a if a != "w46t" else "w46a" for a in base]), log_)   # все карты, --ssl-epochs 4
        for bad in (["--configs", "level"], ["--configs", "maps-resnet"], ["--configs", "lopo:level"]):
            try:
                CV.run(TC._args(base[:4] + bad + base[6:]), log_)
                raise AssertionError(f"--pretrain-only с {bad} должен отказывать")
            except SystemExit as e:
                assert "--pretrain-only" in str(e), str(e)
    finally:
        os.environ.pop("RL_BUFFER_DIR") if old is None else os.environ.__setitem__("RL_BUFFER_DIR", old)
    log = "\n".join(lines)
    assert "level" not in rep.get("configs", {}) and "фолд" not in log, "--pretrain-only посчитал конфиги"
    assert rep["kind"] == "cv_probe_pretrain" and rep["finished"]
    assert rep["data"]["subdirs"] == [SUBDIR] and len(rep["data"]["pdes"]) == 1, rep["data"]
    assert 0 < rep["data"]["n"] <= 12 and rep_all["data"]["n"] > rep["data"]["n"]
    e = rep["encoders"]["ssl-mae-resnet"]
    assert e["file"] == "rl_arch/encoders/cvprobe_w46t_enc_mae_resnet.pt", e["file"]
    per = max(1, e["n_maps"] // min(8, e["n_maps"]))
    assert e["steps"] >= 7 and e["steps"] == e["epochs"] * per and e["steps"] - per < 7, e
    e_all = rep_all["encoders"]["ssl-mae-resnet"]
    assert e_all["epochs"] == 4 and e_all["steps"] == 4 * (e_all["n_maps"] // 8), e_all   # без --ssl-steps — как было
    path = os.path.join(out, "cvprobe_w46t_enc_mae_resnet.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["kind"] == "ssl_encoder" and ck["pdes"] == rep["data"]["pdes"] and ck["ssl_steps"] == e["steps"]
    assert ck["n_maps"] == e["n_maps"] and ck["ssl_epochs"] == e["epochs"] and ck["mean"].shape == (1, 4, 1, 1)
    with open(os.path.join(out, "cvprobe_w46t.json")) as f:
        assert json.load(f)["encoders"]["ssl-mae-resnet"]["steps"] == e["steps"]
    # энкодер ложится в ResNet агента с контекстом (offline_rl.py --encoder-init)
    net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet")
    info = O.load_encoder_init(net, path)
    assert info["encoder"] == "resnet" and info["ssl"] == "mae" and info["partial"], info
    sd = net.model[0].state_dict()
    for k, v in ck["state_dict"].items():
        got = sd[k][:, :v.shape[1]] if k in info["partial"] else sd[k]
        assert torch.equal(got, v.to(got.dtype)), k
    print(f"волна 46: MAE только на картах {SUBDIR} ({rep['data']['n']} переходов из {rep_all['data']['n']}), "
          f"{e['steps']} шагов по --ssl-steps, без level и фолдов, энкодер ложится в агента: OK")


# ---------------------------------------------------------------- волна 47: заглушка среды с картами
class MapSurface:
    """Три карты потерь как функция истории действий (тензоры, как у настоящей поверхности)."""

    def __init__(self, solver_models=None, **k):
        self.m = solver_models

    def save_equation_loss_surface(self, log_key=True):
        rng = np.random.default_rng(zlib.crc32(repr(tuple(self.m.history)).encode()))
        return {k: torch.as_tensor(rng.normal(size=(26, 26)).astype(np.float32) + 2.0)
                for k in ("loss_total", "loss_oper", "loss_bnd")}


@contextlib.contextmanager
def map_env(stop_after=0):
    """Заглушка test_online_finetune с картами-тензорами и настоящим build_state; stop_after — лимит
    времени наступает после этого числа действий (цепочка в работе обрывается)."""
    with TF.fake_env(10 ** 6) as made:
        sys.modules[TF.FAKE_MODULES[3]].PlotLossSurface = MapSurface
        T.build_state = REAL_BUILD_STATE
        n = [0]
        orig = TF.FakeModel.train

        def train(self, *a, **k):
            orig(self, *a, **k)
            n[0] += 1
            if stop_after and n[0] >= stop_after:
                T.time.t += 1e9
        TF.FakeModel.train = train
        try:
            yield made, n
        finally:
            TF.FakeModel.train = orig


def run(tmp, argv, stop_after=0):
    """main() тренера; возвращает (журнал, шпион: буфер обновлений, сеть сохранённого агента, вызовы)."""
    spy = dict(updates=0)
    orig_upd, orig_save = T.agent_update, T.save_agent

    def upd(net, buf, *a, **k):
        spy["buf"] = buf
        spy["updates"] += 1
        return orig_upd(net, buf, *a, **k)

    def save(net, *a, **k):
        spy["net"] = net
        return orig_save(net, *a, **k)
    out = io.StringIO()
    with map_env(stop_after) as (made, n):
        os.chdir(tmp)
        T.agent_update, T.save_agent = upd, save
        sys.argv = ["online_train_env.py"] + argv + ["--save-dir", os.path.join(tmp, "runs")]
        try:
            with contextlib.redirect_stdout(out):
                T.main()
        finally:
            T.agent_update, T.save_agent = orig_upd, orig_save
        spy["env_models"], spy["actions"] = made[0], n[0]
    return out.getvalue(), spy


def must_fail(tmp, argv, text):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run(tmp, argv)
    except SystemExit:
        return
    raise AssertionError(f"{text}: разбор не отказал на {argv}")


def chain_files(tmp, tag):
    return sorted(glob.glob(os.path.join(tmp, "runs", "buffers_new", tag, SUBDIR, "*.pt")))


def payload_rows(path, rows):
    """Переходы, которые тренер положил в свой буфер онлайн (--save-buffer): ключ — (действие, ошибка). У цепочек
    с общим началом в заглушке среды переходы начала совпадают целиком — такие ключи сверяются между собой.
    Возвращает число переходов файла."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    for i in range(len(p["A"])):
        key = (int(p["A"][i]), round(float(p["ERR"][i]), 6))
        row = (p["S"][i].astype(np.float32), p["S2"][i].astype(np.float32), float(p["R"][i]), float(p["D"][i]))
        if key in rows:
            assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(rows[key], row)), \
                f"разные переходы с ключом {key}"
        rows[key] = row
    return len(p["A"])


def test_worker_and_preload(tmp):
    import target_buffer as TB
    # раунд 0: работник с нуля, все цепочки — файлами; лимит времени обрывает четвёртую цепочку на третьем шаге
    w0 = COMMON + ["--hours", "1", "--save-chains", "scr_r0", "--save-buffer", "--save-agent", "--save-every", "1",
                   "--eps-start", "1.0", "--eps-end", "0.5", "--eps-decay", "10", "--seed", "101", "--tag", "w0"]
    out0, spy0 = run(tmp, w0, stop_after=21)
    files0 = chain_files(tmp, "scr_r0")
    assert len(files0) == 4 and spy0["actions"] == 21, (len(files0), spy0["actions"])
    tr = [torch.load(f, weights_only=False) for f in files0]
    assert [len(t) for t in tr] == [6, 6, 6, 3], [len(t) for t in tr]
    for t in tr[:3]:
        assert [x["done"] for x in t] == [0] * 5 + [1] and t[-1]["spent"] == BUDGET
    assert [x["done"] for x in tr[3]] == [0, 0, 0] and tr[3][0]["meta"]["truncated"]
    m = tr[0][0]["meta"]
    assert m["env"] == dict(keep_opt=True, mode="safe", budget=BUDGET) and m["source"] == "online_train"
    assert m["format"] == TB.FORMAT and m["record"] == "w0_c0001" and not tr[0][0]["state"]["loss_total"].any()
    assert all(np.array_equal(a["next_state"]["loss_oper"], b["state"]["loss_oper"]) for t in tr for a, b in zip(t, t[1:]))
    assert "цепочки сохранены: файлов 4, переходов 21" in out0
    # второй раунд (другой работник, другая папка) — для загрузки двух раундов сразу
    w1 = [a for a in w0 if a not in ("scr_r0", "w0", "101")]
    w1 = w1[:w1.index("--save-chains") + 1] + ["scr_r1"] + w1[w1.index("--save-chains") + 1:]
    w1 = w1[:w1.index("--seed") + 1] + ["102"] + w1[w1.index("--seed") + 1:]
    w1 = w1[:w1.index("--tag") + 1] + ["w1b"] + w1[w1.index("--tag") + 1:]
    run(tmp, w1, stop_after=14)
    files1 = chain_files(tmp, "scr_r1")
    assert len(files1) == 3

    # ученик: оба раунда в общий буфер, K обновлений, без среды
    srcs = ",".join(os.path.join(tmp, "runs", "buffers_new", t) for t in ("scr_r0", "scr_r1"))
    lr = COMMON + ["--hours", "0", "--learn-only", "--preload-online", srcs, "--pre-updates", "12", "--seed", "1",
                   "--tag", "L0"]
    out, spy = run(tmp, lr)
    assert spy["env_models"] == 0 and spy["actions"] == 0, "ученик вызывал среду"
    assert spy["updates"] == 12 and "агент сохранён без среды" in out
    buf = spy["buf"]
    rows = {}
    n_rows = payload_rows(os.path.join(tmp, "buffer_w0.pt"), rows) + payload_rows(os.path.join(tmp, "buffer_w1b.pt"), rows)
    assert len(buf) == n_rows == 21 + 14, (len(buf), n_rows)
    ag, mean, std, variant = E.load_agent(os.path.join(tmp, "agent_L0.pt"))
    mean, std = np.asarray(mean), np.asarray(std)
    assert mean.shape == (1, 9, 1, 1) and (mean[0, 4:] == 0).all() and (std[0, 4:] == 1).all()
    n_s2 = 0
    for i in range(len(buf)):
        key = (int(buf.a[i]), round(float(buf.ub[i]), 6))
        s, s2, r, d = rows[key]
        got = buf.s[i] * std[0] + mean[0]
        assert np.allclose(got, s, atol=2e-2, rtol=2e-3), f"состояние {key}: max {np.abs(got - s).max()}"
        assert abs(buf.r[i] - r) < 1e-5 and buf.d[i] == d, (key, buf.r[i], r, buf.d[i], d)
        if d == 0:          # у терминального перехода s' в цель не входит (загрузчик ставит s)
            got2 = buf.s2[i] * std[0] + mean[0]
            assert np.allclose(got2, s2, atol=2e-2, rtol=2e-3), f"s' {key}: max {np.abs(got2 - s2).max()}"
            n_s2 += 1
    assert n_s2 == 35 - 5, n_s2          # терминальны 3 + 2 законченные цепочки
    meta = ag._meta
    assert meta["preload_online"] == srcs and meta["preload_info"]["transitions"] == 35
    assert meta["preload_info"]["envs"] == {"keep_safe": 7} and meta["pre_updates"] == 12
    assert meta["bn_mode"] == "train" and meta["encoder"] == "resnet" and meta["in_ch"] == 9
    rng = np.random.default_rng(3)
    net = spy["net"]
    net.set_train(False)
    for _ in range(3):
        st = (mean[0] + std[0] * rng.normal(size=(9, 26, 26))).astype(np.float32)
        with torch.no_grad():
            q_tr = net.q_scalar(torch.as_tensor((st[None] - mean) / std).float())[0].numpy()
        assert np.allclose(E.q_values(ag, st, mean, std, variant), q_tr, atol=1e-5)
    print(f"волна 47: файлы цепочек работника (3 законченные + оборванная), загрузка двух раундов = буфер тренера "
          f"онлайн ({len(buf)} переходов: карты, delta, контекст s и s', награда, конец), ученик без среды: OK")
    return os.path.join(tmp, "agent_L0.pt"), srcs, rows


def test_next_round(tmp, ck_path, srcs, rows):
    before = torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"]
    w = COMMON + ["--hours", "1", "--warm-start", ck_path, "--warm-strict", "--bn-mode", "train",
                  "--preload-online", srcs, "--pre-updates", "5", "--save-chains", "scr_r2", "--save-agent",
                  "--save-every", "1", "--eps-start", "0.5", "--eps-end", "0.05", "--eps-decay", "5", "--seed", "201",
                  "--tag", "w2"]
    out, spy = run(tmp, w, stop_after=7)
    assert "строгий тёплый старт" in out and "статистики пачки в обновлениях" in out
    i_pre = out.index("[до действий] обновление 5/5")
    i_step = out.index("[траектория 1] шаг 1")
    assert i_pre < i_step, "обновления до действий идут после первого шага"
    assert "--preload-online: в общем буфере 35 переходов" in out and "нормировка — чекпоинта" in out
    assert len(spy["buf"]) == 35 + 7 and spy["actions"] == 7
    files = chain_files(tmp, "scr_r2")
    assert len(files) == 2 and sum(len(torch.load(f, weights_only=False)) for f in files) == 7
    assert torch.load(files[0], weights_only=False)[0]["meta"]["warm_start"] == os.path.basename(ck_path)
    ag, mean, std, _ = E.load_agent(os.path.join(tmp, "agent_w2.pt"))
    ck0 = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert np.array_equal(np.asarray(mean), ck0["mean"]), "нормировка не из чекпоинта ученика"
    after = torch.load(os.path.join(tmp, "agent_w2.pt"), map_location="cpu", weights_only=False)["state_dict"]
    moved = sum(float((after[k].float() - before[k].float()).abs().sum()) for k in after)
    assert moved > 0 and ag._meta["pre_updates"] == 5 and ag._meta["warm_strict"]
    eps = [float(x) for x in re.findall(r"eps=([0-9.]+)\)", out)]
    assert eps and abs(eps[0] - 0.5) < 0.006, eps[:3]
    print("волна 47: работник следующего раунда — строгий старт от ученика, прошлые раунды в буфере, "
          "обновления до первого действия, затем среда и новые файлы цепочек: OK")


def test_defaults_and_refusals(tmp):
    old = [a for a in COMMON if a not in ("--encoder", "resnet")] + ["--hours", "1", "--save-agent", "--save-every",
                                                                      "1", "--seed", "42", "--tag", "dflt"]
    out, spy = run(tmp, old, stop_after=8)
    assert not glob.glob(os.path.join(tmp, "runs", "buffers_new", "*", SUBDIR, "dflt_*.pt"))
    assert "--preload-online" not in out and "[до действий]" not in out and "цепочки сохранены" not in out
    meta = torch.load(os.path.join(tmp, "agent_dflt.pt"), weights_only=False)["meta"]
    assert not {"preload_online", "preload_info", "pre_updates"} & set(meta), sorted(meta)
    src = os.path.join(tmp, "runs", "buffers_new", "scr_r0")
    for bad, why in ((["--pre-updates", "3"], "обновления без загрузки"),
                     (["--learn-only", "--preload-online", src], "ученик без обновлений"),
                     (["--learn-only", "--pre-updates", "3"], "ученик без загрузки"),
                     (["--preload-online", src, "--state-mode", "level"], "не full"),
                     (["--preload-online", src, "--n-step", "3"], "n-step"),
                     (["--preload-online", src, "--preload", SUBDIR], "две загрузки"),
                     (["--save-chains", "a/b"], "тег"),
                     (["--save-chains", "x", "--state-mode", "blind"], "без карт"),
                     (["--preload-online", src, "--pre-updates", "2", "--rlpd", "--rlpd-subdir", SUBDIR],
                      "обновления с RLPD"),
                     (["--pre-updates", "-1"], "отрицательное K")):
        must_fail(tmp, old + bad, why)
    print("прежняя командная строка без новых флагов: ни файлов цепочек, ни новых полей meta; отказы разбора: OK")


def main():
    t0 = time.time()
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_cv_pretrain(tmp)
            ck, srcs, rows = test_worker_and_preload(tmp)
            test_next_round(tmp, ck, srcs, rows)
            test_defaults_and_refusals(tmp)
    finally:
        os.chdir(cwd)
    print(f"ВСЕ ПРОВЕРКИ ВОЛН 46-47 ПРОЙДЕНЫ ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
