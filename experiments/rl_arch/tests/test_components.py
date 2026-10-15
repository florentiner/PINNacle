#!/usr/bin/env python
"""
Проверки составляющих метода волны 50 (offline_rl.py) без GPU, без сети и без обучения PINN, на синтетических
цепочках:

  * по умолчанию всё прежнее бит в бит: train_variant без новых полей и с ними в значениях по умолчанию даёт
    те же веса; строка main() без новых полей;
  * --wa: SWA с одной итерацией — веса последней итерации; SWA и EMA по эпохам 1..3 равны среднему (и
    экспоненциальному среднему) весов обучений длиной 1, 2, 3 эпохи (обучение детерминировано); у ResNet
    статистики BatchNorm пересчитаны, у замороженного энкодера — не тронуты; поля сводки;
  * --mc-lb: mc_returns — дисконтированная сумма наград до конца цепочки (оборванная — продолжение нулём);
    при G ниже любого Q штраф ноль и веса бит в бит прежние; при большом G Q записанных действий растёт;
    вид target (MCAC) поднимает цель TD;
  * ансамбль учителей: среднее целей; два одинаковых учителя — ровно цели одного, согласие пар 1;
  * main(): --train-all (учатся все переходы, модельные цели и цели учителя на всех строках, диагностики на
    прежних отложенных), --wa ema с проверкой загрузки чекпоинта через online_eval_env.load_agent,
    --mc-lb, --teacher-from из двух учителей (настоящие чекпоинты agent_combo.py с сидами 0 и 1), поля в строке
    и meta; отказы разбора.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_components.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ["HF_HUB_OFFLINE"] = "1"
for _k in ("HF_TOKEN_WRITE", "HF_TOKEN"):      # проверка ничего не выгружает
    os.environ.pop(_k, None)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "track3"), ROOT):
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import mb_targets as MB  # noqa: E402
import test_map_agent as T  # noqa: E402
import test_shield_teacher as TS  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a, **k: None  # noqa: E731
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"
CPU = torch.device("cpu")
NEW_FIELDS = ("wa", "wa_start", "wa_decay", "wa_n", "wa_agree_last", "train_all", "diag_in_sample", "mc_lb",
              "mc_lb_mode", "mc_lb_loss", "n_teachers", "teacher_n_teachers", "teacher_pair_agree_mean")


def must_fail(fn, exc, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
    except exc:
        return
    raise AssertionError(what)


def same(net_a, net_b):
    sa, sb = net_a.model.state_dict(), net_b.model.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


# --------------------------------------------------------------------------------------------

def test_defaults(data, mask):
    """Новые поля в значениях по умолчанию — те же веса, что без них."""
    for variant, kw in (("cnn_dqn", {}), ("convnext_dqn", dict(encoder="resnet", value_bound=True))):
        n0, _ = T.fit(variant, data, mask, epochs=3, **kw)
        n1, _ = T.fit(variant, data, mask, epochs=3, wa="none", wa_start=0.5, wa_decay=0.99, mc_lb=0.0,
                      mc_lb_mode="loss", train_all=False, **kw)
        assert same(n0, n1), variant
        assert not any(k in n1.extra for k in NEW_FIELDS), [k for k in NEW_FIELDS if k in n1.extra]
    print("по умолчанию: веса бит в бит прежние, новых полей в сводке нет: OK")


def test_wa(data, mask):
    kw = dict(batch_size=64)                       # вся выборка — одна пачка: шаг на эпоху
    # одна итерация в среднем (wa_start = 2/3 при трёх эпохах) — это сама последняя итерация
    n0, _ = T.fit("cnn_dqn", data, mask, epochs=3, **kw)
    n1, _ = T.fit("cnn_dqn", data, mask, epochs=3, wa="swa", wa_start=2 / 3, **kw)
    assert same(n0, n1) and n1.extra["wa_n"] == 1 and n1.extra["wa_agree_last"] == 1.0 and n1.extra["wa_bn_layers"] == 0
    # итерация e обучения в три эпохи совпадает с финалом обучения длиной e эпох
    finals = [T.fit("cnn_dqn", data, mask, epochs=e, **kw)[0].model.state_dict() for e in (1, 2, 3)]
    swa, _ = T.fit("cnn_dqn", data, mask, epochs=3, wa="swa", wa_start=0.0, **kw)
    d = 0.7
    ema, _ = T.fit("cnn_dqn", data, mask, epochs=3, wa="ema", wa_start=0.0, wa_decay=d, **kw)
    s_swa, s_ema = swa.model.state_dict(), ema.model.state_dict()
    for k in finals[0]:
        want = (finals[0][k] + finals[1][k] + finals[2][k]) / 3
        assert torch.allclose(s_swa[k], want, atol=1e-6), k
        e = d * (d * finals[0][k] + (1 - d) * finals[1][k]) + (1 - d) * finals[2][k]
        assert torch.allclose(s_ema[k], e, atol=1e-6), k
    assert swa.extra["wa_n"] == 3 == ema.extra["wa_n"] and swa.extra["wa_from_epoch"] == 0
    assert not same(swa, n0), "среднее трёх итераций отличается от последней"
    # ResNet: BatchNorm пересчитан по обучающим состояниям; у замороженного энкодера статистики не тронуты
    rkw = dict(encoder="resnet", value_bound=True, epochs=3, batch_size=64)
    r0, _ = T.fit("convnext_dqn", data, mask, **rkw)
    r1, _ = T.fit("convnext_dqn", data, mask, wa="ema", wa_start=0.0, **rkw)
    assert r1.extra["wa_bn_layers"] > 0 and 0.0 <= r1.extra["wa_agree_last"] <= 1.0
    bn = [k for k in r0.model.state_dict() if k.endswith("running_mean")]
    assert any(not torch.equal(r0.model.state_dict()[k], r1.model.state_dict()[k]) for k in bn)
    x = torch.as_tensor(np.asarray(data["S"][:5]), dtype=torch.float32)
    f0, st0 = T.fit("convnext_dqn", data, mask, freeze_encoder=True, **rkw)
    f1, _ = T.fit("convnext_dqn", data, mask, freeze_encoder=True, wa="swa", wa_start=0.0, **rkw)
    assert f1.extra["wa_bn_layers"] == 0
    enc0 = {k: v for k, v in f0.model.state_dict().items() if k.startswith("0.")}
    enc1 = {k: v for k, v in f1.model.state_dict().items() if k.startswith("0.")}
    assert all(torch.equal(enc0[k], enc1[k]) for k in enc0), "замороженный энкодер не меняется средним"
    with torch.no_grad():
        q = f1.q_scalar((x - torch.as_tensor(st0["mean"])) / torch.as_tensor(st0["std"]))
    assert torch.isfinite(q).all()
    print("--wa: одна итерация — последние веса; SWA и EMA по эпохам — среднее весов обучений длиной 1..3 эпохи; "
          "BatchNorm ResNet пересчитан, замороженный энкодер не тронут: OK")


def test_mc_lb(data, mask):
    g = O.mc_returns(data, 0.9)
    R, EP = data["R"].astype(np.float64), data["EP"]
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        for j, i in enumerate(idx):
            want = sum(0.9 ** k * R[idx[j + k]] for k in range(len(idx) - j))
            assert abs(g[i] - want) < 1e-5
    # оборванная цепочка (последний переход не терминален) продолжается нулём: формула та же
    cut = dict(data, D=np.where(data["EP"] == 0, 0.0, data["D"]).astype(np.float32))
    assert np.array_equal(O.mc_returns(cut, 0.9), g)
    # G ниже любого Q: штраф ноль, веса бит в бит без флага (у cnn нет BatchNorm, лишний проход ничего не меняет)
    low = dict(data, R=np.full_like(data["R"], -1e3))
    n0, _ = T.fit("cnn_dqn", low, mask, epochs=3)
    n1, _ = T.fit("cnn_dqn", low, mask, epochs=3, mc_lb=1.0)
    assert same(n0, n1) and n1.extra["mc_lb_loss"] == 0.0 and n1.extra["mc_lb_active"] == 0.0
    # большой G: Q записанных действий растёт
    hi = dict(data, R=np.full_like(data["R"], 5.0))
    G = O.mc_returns(hi, O.GAMMA)
    idx = np.where(mask)[0]

    def q_taken(net, st):
        S = torch.as_tensor((hi["S"][idx] - st["mean"]) / st["std"]).float()
        with torch.no_grad():
            return net.q_scalar(S).gather(1, torch.as_tensor(hi["A"][idx])[:, None]).squeeze(1).numpy()

    b0, st = T.fit("cnn_dqn", hi, mask, epochs=20)
    b1, _ = T.fit("cnn_dqn", hi, mask, epochs=20, mc_lb=1.0)
    b2, _ = T.fit("cnn_dqn", hi, mask, epochs=20, mc_lb=1.0, mc_lb_mode="target")
    gap0 = np.mean(G[idx] - q_taken(b0, st))
    gap1 = np.mean(G[idx] - q_taken(b1, st))
    gap2 = np.mean(G[idx] - q_taken(b2, st))
    assert gap1 < gap0 and gap2 < gap0, (gap0, gap1, gap2)
    assert b1.extra["mc_lb_loss"] > 0 and "mc_lb_loss" not in b2.extra
    # на всех головах: ансамбль и квантили считаются без ошибок
    for variant, kw in (("cnn_qrdqn", {}), ("convnext_dqn", dict(encoder="resnet", ens=2, ens_min=2))):
        n, _ = T.fit(variant, hi, mask, mc_lb=0.5, **kw)
        assert np.isfinite(n.extra["mc_lb_loss"])
    # две сети (лидер ns2d): штраф и среднее весов на совместном Q обеих сетей
    two, _ = T.fit("convnext_dqn", hi, mask, encoder="resnet", value_bound=True, two_nets=True, epochs=3,
                   batch_size=64, mc_lb=1.0, wa="ema", wa_start=0.0)
    assert np.isfinite(two.extra["mc_lb_loss"]) and two.extra["wa_n"] == 3 and two.extra["wa_bn_layers"] > 0
    print(f"--mc-lb: возврат до конца цепочки, ноль при низкой границе (веса прежние), разрыв G - Q записанных "
          f"действий {gap0:.2f} -> {gap1:.2f} (штраф) и {gap2:.2f} (цель MCAC): OK")


def test_teacher_ensemble(c0, c1):
    data = T.make_data()
    args = ([PDE], "delta", np.ones(1), np.array([1.0]), 0.99, CPU)
    q0, v0, _ = O.teacher_targets(c0, data, *args, log=QUIET)
    q1, v1, _ = O.teacher_targets(c1, data, *args, log=QUIET)
    qe, ve, info = O.teacher_ensemble_targets([c0, c1], data, *args, log=QUIET)
    assert np.array_equal(ve, v0 & v1) and np.allclose(qe[ve], ((q0.astype(np.float64) + q1) / 2)[ve], atol=1e-6)
    assert info["n_teachers"] == 2 and 0.0 <= info["pair_agree_mean"] <= 1.0 and info["rows"] == int(ve.sum())
    qs, vs, inf2 = O.teacher_ensemble_targets([c0, c0], data, *args, log=QUIET)
    assert np.array_equal(qs, q0) and np.array_equal(vs, v0) and inf2["pair_agree_mean"] == 1.0
    assert not np.array_equal(q0[v0 & v1], q1[v0 & v1]), "учителя с разными сидами различаются"
    print(f"ансамбль учителей: среднее целей, два одинаковых — цели одного; согласие пар разных учителей "
          f"{info['pair_agree_mean']:.2f}: OK")


def test_main(bufs, c0, c1, tmp):
    import killswitch
    import online_eval_env as E
    root = os.path.join(tmp, "bufs", SUB)
    os.makedirs(root, exist_ok=True)
    for k, ep in enumerate(bufs[SUB]):
        torch.save(ep, os.path.join(root, f"f{k}.pt"))
    e0 = float(O.pde_meta()[PDE]["init_err"])
    mtag = f"tcmp{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
            "--mb-work", os.path.join(tmp, "work"), "--teacher-from", c0, "--teacher-w", "0.5", "--seeds", "1",
            "--model-tag", mtag, "--epochs", "2", "--fqe-epochs", "1", "--batch-size", "8", "--cpu"]

    def fake_build(names, subdirs, table, members, work, quick, log):
        tab, ens = T.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3, surr_file="/x/combo_surr_fake.pkl")

    def run(av):
        buf = io.StringIO()
        sys.argv = av
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            O.main()
        text = buf.getvalue()
        return [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0], text

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        r0, _ = run(argv)
        assert not any(k in r0 for k in NEW_FIELDS), [k for k in NEW_FIELDS if k in r0]
        assert r0["teacher_from"] == os.path.basename(c0)
        n_all = r0["teacher_rows"]
        assert r0["teacher_train_rows"] < n_all, "по умолчанию часть цепочек отложена"
        # --train-all: все строки в обучении, диагностики на прежних отложенных
        r1, _ = run(argv + ["--train-all"])
        assert r1["train_all"] is True and r1["diag_in_sample"] is True
        assert r1["teacher_train_rows"] == n_all and r1["mb_rows"] > r0["mb_rows"], (r0["mb_rows"], r1["mb_rows"])
        assert np.isfinite(r1["td_error"]) and r1["teacher_agree"] is None
        # --wa ema, --mc-lb, два учителя; чекпоинт проверяется загрузкой (verify_ckpt) и читается load_agent
        r2, text = run(argv[:argv.index(c0)] + [f"{c0},{c1}"] + argv[argv.index(c0) + 1:]
                       + ["--wa", "ema", "--wa-start", "0", "--mc-lb", "0.5"])
        assert "'ok': True" in text, "чекпоинт со средними весами загружается online_eval_env"
        assert r2["wa"] == "ema" and r2["wa_start"] == 0.0 and r2["wa_n"] > 0 and r2["wa_bn_layers"] > 0
        assert r2["mc_lb"] == 0.5 and r2["mc_lb_mode"] == "loss" and np.isfinite(r2["mc_lb_loss"])
        assert r2["n_teachers"] == 2 and r2["teacher_n_teachers"] == 2 and 0 <= r2["teacher_pair_agree_mean"] <= 1
        assert r2["teacher_from"] == f"{os.path.basename(c0)},{os.path.basename(c1)}"
        meta = E.load_agent(out)[0]._meta
        assert meta["wa"] == "ema" and meta["mc_lb"] == 0.5 and meta["n_teachers"] == 2
        assert meta["teacher"]["n_teachers"] == 2
        for bad, what in ((argv + ["--wa", "swa", "--fqe-select", "--ckpt-every", "1"], "--wa с --fqe-select"),
                          (argv + ["--wa", "ema", "--wa-decay", "1.0"], "распад 1"),
                          (argv + ["--wa", "swa", "--wa-start", "1.0"], "начало 1"),
                          (argv + ["--train-all", "--plateau-patience", "2"], "--train-all с --plateau-patience"),
                          (argv + ["--mc-lb", "-1"], "отрицательный вес"),
                          (argv + ["--mc-lb", "1", "--mc-lb-mode", "target", "--ens", "2"], "MCAC с ансамблем голов"),
                          (argv[:argv.index(c0)] + [f"{c0},{c1}"] + argv[argv.index(c0) + 1:]
                           + ["--train-kmax", "3"], "ансамбль учителей с правилом остановки")):
            must_fail(lambda b=bad: run(b), SystemExit, f"недопустимое сочетание флагов: {what}")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("main(): по умолчанию строка без новых полей; --train-all, --wa ema, --mc-lb, два учителя — поля в строке "
          "и meta, чекпоинт загружается; отказы разбора: OK")


def main():
    t0 = time.time()
    data = T.make_data(seed=2, n_chains=6, length=3)
    mask = data["EP"] < 5
    tmp = tempfile.mkdtemp(prefix="components_")
    try:
        test_defaults(data, mask)
        test_wa(data, mask)
        test_mc_lb(data, mask)
        bufs = TS.make_buffers()
        c0, _ = TS.train_teacher(bufs, tmp, tag="teach_s0")
        c1, _ = TS.train_teacher(bufs, tmp, tag="teach_s1", extra=("--seed", "1"))
        test_teacher_ensemble(c0, c1)
        test_main(bufs, c0, c1, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_components: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
