#!/usr/bin/env python
"""
Проверки волны 58 (offline_rl.py: --view-aug, --view-aug-p, --view-cons, --lr-schedule, --lr-min) без GPU, без
сети и без обучения PINN, на синтетических цепочках и файлах видов, за секунды:

  * по умолчанию всё прежнее бит в бит: train_variant без новых полей и с ними в значениях по умолчанию (и с
    банком видов в наборе, но без флага) даёт те же веса; строка main() без новых полей;
  * сопоставление видов с переходами собранного буфера (сид + исполненное начало цепочки + шаг, как у
    map_views.view_rows): переходы с видами и без, общий префикс двух файлов одного сида (виды из другого файла
    видов), чужой сид и чужая цепочка, файл без сида; канал delta — по тому же виду предыдущего шага, иначе по
    карте среды; каналы контекста не меняются; первый файл видов с ключом выигрывает;
  * замена карт: только у переходов с видами, с вероятностью P, вид равновероятный, s и s' независимо; виды,
    равные картам среды, не меняют обучение (веса бит в бит);
  * согласованность видов: ноль при одинаковых видах и целевой сети, равной онлайновой; больше нуля на разных;
  * косинусный план: значения lr_at, шаг Adam на каждом шаге обучения, lr_min = lr — веса как у const, вместе с
    --wa ema;
  * main(): --target-buffer с --view-aug, --view-cons и --lr-schedule cosine — поля в строке и meta, чекпоинт
    проходит verify_ckpt и online_eval_env.load_agent, ckpt_arch тот же, что без новых полей; отказы разбора.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_train_more.py
"""
import contextlib
import io
import json
import math
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
import map_views as MV  # noqa: E402
import target_buffer as TB  # noqa: E402
import test_map_agent as T  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a, **k: None  # noqa: E731
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"
CPU = torch.device("cpu")
NEW_FIELDS = ("view_aug", "view_aug_p", "view_cons", "view_rows_s", "view_rows_s_train", "view_aug_frac_s",
              "view_cons_loss", "lr_schedule", "lr_min", "lr_steps", "lr_last")


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


def act(spec):
    o, lr, ep = spec.split(":")
    return next(i for i, (a, l, e) in enumerate(TB.S.ACTIONS) if a == o and abs(l - float(lr)) < 1e-12 and e == int(ep))


def rmap(rng, flat=False):
    if flat:
        return np.full((3, 26, 26), rng.uniform(0.1, 3.0), np.float32)
    g = np.linspace(-1, 1, 26)
    a = rng.normal(size=3)
    base = np.add.outer(a[0] * g, a[1] * g ** 2) + 0.1 * rng.normal(size=(26, 26)) + 3.0
    return np.stack([base, 0.8 * base, 0.3 * base + a[2]]).astype(np.float32)


def tr_of(acts, maps, seed, group=None):
    """Файл буфера в формате collect_active: state[j] — карта после j действий (шаг 0 нулевой), без следующей
    карты — конец цепочки. seed None — файл без сида в meta."""
    out = []
    z = {k: np.zeros((26, 26), np.float32) for k in TB.MAP_KEYS}
    for j, a in enumerate(acts):
        cur = z if j == 0 else {k: maps[j][i] for i, k in enumerate(TB.MAP_KEYS)}
        nxt = {k: maps[j + 1][i] for i, k in enumerate(TB.MAP_KEYS)} if (j + 1) in maps else cur
        out.append(dict(state=cur, next_state=nxt, action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                        reward=-0.1 / (j + 1), done=0, epochs_real=int(TB.S.ACTIONS[a][2]), group=group))
    out[-1]["done"] = 0 if len(acts) in maps else 1
    out[0]["meta"] = dict(env=dict(keep_opt=True, mode="safe", budget=7000), **({} if seed is None else dict(seed=seed)))
    return out


A1, L1, A2, P1, A3 = (act(x) for x in ("Adam:0.001:1000", "LBFGS:1:500", "Adam:0.0001:2500", "PSO:0:300",
                                        "Adam:0.01:100"))


def chain_of(acts):
    return [list(TB.S.ACTIONS[a]) for a in acts]


def scenario(seed=0):
    """Файлы буфера и файлы видов. f0 и f1 — один сид (11) с общим префиксом из двух действий (как группа
    --share-prefix), f2 — сид 22 без видов, f3 — без сида в meta. Виды: r0 (файл f0) — шаги 1, 2 по 3 вида и шаг 3
    — 4 вида; r1 (файл f1) — только шаг 3 (2 вида; префикс уже есть в r0); r0dup — повтор ключа шага 1 с другими
    картами (после r0 по имени: не выигрывает); чужой сид 5000 и сид 22 с другой цепочкой — не совпадают."""
    rng = np.random.default_rng(seed)
    f = {"f0": ([A1, L1, A2, P1], 11), "f1": ([A1, L1, A3, P1], 11), "f2": ([L1, P1, P1], 22),
         "f3": ([A1, L1, P1], None)}
    maps = {}
    eps = []
    for name, (acts, sd) in f.items():
        mp = {j: rmap(rng) for j in range(1, len(acts))}
        if name == "f1":                         # общий префикс: те же карты шагов 1-2 (другой прогон среды)
            mp[1], mp[2] = maps["f0"][1] + 0.001, maps["f0"][2] + 0.001
        maps[name] = mp
        eps.append(tr_of(acts, mp, sd))
    ch0, ch1 = chain_of(f["f0"][0]), chain_of(f["f1"][0])
    v0 = {1: [rmap(rng) for _ in range(3)], 2: [rmap(rng) for _ in range(3)], 3: [rmap(rng) for _ in range(4)]}
    r0 = MV.make_record(PDE, "t", "f0.pt", 11, {}, ch0[:3], {s: maps["f0"][s] + 0.01 for s in (1, 2, 3)}, v0, {})
    v1 = {3: [rmap(rng) for _ in range(2)]}
    r1 = MV.make_record(PDE, "t", "f1.pt", 11, {}, ch1[:3], {s: maps["f1"][s] for s in (1, 2, 3)}, v1, {})
    rdup = MV.make_record(PDE, "t", "f0b.pt", 11, {}, ch0[:1], {1: maps["f0"][1]}, {1: [rmap(rng)]}, {})
    rfor = MV.make_record(PDE, "t", "r5000", 5000, {}, ch0[:2], {1: rmap(rng)}, {1: [rmap(rng)]}, {})
    roth = MV.make_record(PDE, "t", "f2x.pt", 22, {}, chain_of([A1, P1]), {1: rmap(rng)}, {1: [rmap(rng)]}, {})
    return eps, [r0, r1, rdup, rfor, roth], maps


def load(eps, ctx=True):
    with contextlib.redirect_stdout(io.StringIO()):
        data = O.episodes_to_arrays(eps, chain_fix=True, reward_form="delta", init_err=1.0, budget=7000, verbose=False)
    if ctx:
        c1, c2 = O.chain_ctx(data, k_max=70, budget=7000)
        data["S"], data["S2"] = O.attach_ctx(data["S"], c1), O.attach_ctx(data["S2"], c2)
    return data


# --------------------------------------------------------------------------------------------

def test_defaults(data, mask):
    """Новые поля в значениях по умолчанию — те же веса, что без них; банк видов без флага не читается."""
    bank = dict(ALT=np.repeat(data["S"], 2, 0) + 1.0, OFF_S=np.arange(len(data["A"])) * 2,
                K_S=np.full(len(data["A"]), 2), OFF_S2=np.arange(len(data["A"])) * 2, K_S2=np.full(len(data["A"]), 2),
                info={})
    for variant, kw in (("cnn_dqn", {}), ("convnext_dqn", dict(encoder="resnet", value_bound=True))):
        n0, _ = T.fit(variant, data, mask, epochs=3, **kw)
        n1, _ = T.fit(variant, data, mask, epochs=3, view_aug="", view_aug_p=0.5, view_cons=0.0,
                      lr_schedule="const", lr_min=0.0, **kw)
        n2, _ = T.fit(variant, dict(data, VIEWS=bank), mask, epochs=3, view_aug_p=0.5, **kw)
        assert same(n0, n1) and same(n0, n2), variant
        assert not any(k in n1.extra for k in NEW_FIELDS), [k for k in NEW_FIELDS if k in n1.extra]
    print("по умолчанию: веса бит в бит прежние (и с банком видов в наборе без флага), новых полей нет: OK")


def test_matching():
    eps, recs, maps = scenario()
    data = load(eps)
    bank = O.view_bank(recs, eps, data)
    kk = O.view_row_keys(eps, data)
    EP, STEP = data["EP"], data["STEP"]
    assert sorted(np.unique(EP).tolist()) == [0, 1, 2, 3] and len(EP) == 4 + 4 + 3 + 3
    row = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(EP, STEP))}
    ks = {(e, p): int(bank["K_S"][row[(e, p)]]) for (e, p) in row}
    ks2 = {(e, p): int(bank["K_S2"][row[(e, p)]]) for (e, p) in row}
    # f0 (EP 0): s — шаги 1, 2 по 3 вида, шаг 3 — 4; s' — на шаг дальше, у последнего (терминального) нет
    assert [ks[(0, p)] for p in range(4)] == [0, 3, 3, 4], [ks[(0, p)] for p in range(4)]
    assert [ks2[(0, p)] for p in range(4)] == [3, 3, 4, 0], [ks2[(0, p)] for p in range(4)]
    # f1 (EP 1): шаги 1-2 — общий префикс, виды из файла r0; шаг 3 — свои (2 вида, файл r1)
    assert [ks[(1, p)] for p in range(4)] == [0, 3, 3, 2] and [ks2[(1, p)] for p in range(4)] == [3, 3, 2, 0]
    # f2 (сид 22: файл видов сида 22 с другой цепочкой не совпадает), f3 (сида нет) — без видов
    assert all(ks[(e, p)] == 0 and ks2[(e, p)] == 0 for (e, p) in row if e in (2, 3))
    assert kk["KS"][row[(3, 1)]] is None and kk["KS"][row[(2, 1)]] is not None
    assert kk["KS"][row[(0, 2)]] == MV.state_key(11, chain_of([A1, L1, A2, P1]), 2) == kk["KS"][row[(1, 2)]]
    assert kk["KS"][row[(0, 3)]] != kk["KS"][row[(1, 3)]]
    info = bank["info"]
    assert info["rows_s"] == 6 and info["rows_s2"] == 6 and info["states"] == 6, info     # r0: 3, r1: 1, сид 5000 и сид 22: по 1
    assert abs(info["mean_k_s"] - (3 + 3 + 4 + 3 + 3 + 2) / 6) < 1e-3 and info["alt"] == 2 * (3 + 3 + 4 + 3 + 3 + 2)
    r0, r1 = recs[0], recs[1]
    S, S2, ALT = data["S"], data["S2"], bank["ALT"]

    def alt(side, e, p, v):
        i = row[(e, p)]
        off = bank["OFF_S" if side == "s" else "OFF_S2"][i]
        return ALT[off + v], (S if side == "s" else S2)[i]

    # шаг 1: карты вида, delta нулевой; контекст строки тот же
    x, base = alt("s", 0, 1, 2)
    assert np.array_equal(x[:3], r0["views"][1][2]) and (x[3] == 0).all() and np.array_equal(x[4:], base[4:])
    assert base.shape[0] > 4 and not np.array_equal(x[:3], base[:3])
    # шаг 2: delta по тому же виду шага 1; первый файл видов с ключом выигрывает (r0, не r0dup)
    x, base = alt("s", 0, 2, 1)
    assert np.allclose(x[3], O._delta_like_env(r0["views"][2][1][0], r0["views"][1][1][0]))
    assert np.array_equal(x[4:], base[4:])
    # шаг 3, вид 3: у шага 2 всего 3 вида — delta по карте среды шага 2 из буфера
    x, _ = alt("s", 0, 3, 3)
    assert np.array_equal(x[:3], r0["views"][3][3])
    assert np.allclose(x[3], O._delta_like_env(r0["views"][3][3][0], maps["f0"][2][0]))
    x, _ = alt("s", 0, 3, 0)
    assert np.allclose(x[3], O._delta_like_env(r0["views"][3][0][0], r0["views"][2][0][0]))
    # f1 шаг 3 (свои виды r1): delta по виду шага 2 из r0 (общий префикс)
    x, _ = alt("s", 1, 3, 1)
    assert np.array_equal(x[:3], r1["views"][3][1])
    assert np.allclose(x[3], O._delta_like_env(r1["views"][3][1][0], r0["views"][2][1][0]))
    # s' строки шага 1 — состояние шага 2: delta по виду шага 1; контекст s' строки
    x, base = alt("s2", 0, 1, 2)
    assert np.array_equal(x[:3], r0["views"][2][2]) and np.array_equal(x[4:], base[4:])
    assert np.allclose(x[3], O._delta_like_env(r0["views"][2][2][0], r0["views"][1][2][0]))
    x, _ = alt("s2", 0, 0, 0)
    assert np.array_equal(x[:3], r0["views"][1][0]) and (x[3] == 0).all()
    # сводка совпадений и сопоставление после среза набора правилом остановки (EP и STEP сохраняются)
    cut, _ = O.stop_rule_cut(data, 0.0, 3)
    b2 = O.view_bank(recs, eps, cut)
    assert b2["info"]["rows_s"] == 4 + 0 and int((b2["K_S2"] > 0).sum()) == 4, b2["info"]
    print(f"сопоставление видов: переходы с видами s {info['rows_s']} и s' {info['rows_s2']} из {info['rows']} "
          f"(общий префикс из чужого файла видов, чужой сид, чужая цепочка, файл без сида), delta по тому же виду "
          f"или по карте среды, контекст не тронут, первый файл с ключом выигрывает: OK")
    return eps, recs, data, bank


def test_pick(data, bank):
    rows = np.arange(len(data["A"]))
    x0 = torch.as_tensor(data["S"]).float()
    alt = torch.as_tensor(bank["ALT"]).float()
    va = O.ViewAug(bank, alt, 1.0, seed=0)
    for side, X in (("s", data["S"]), ("s2", data["S2"])):
        x0 = torch.as_tensor(X).float()
        x = va.pick(x0.clone(), rows, side)
        K, OFF = bank["K_" + side.upper()], bank["OFF_" + side.upper()]
        for i in rows:
            if K[i] == 0:
                assert torch.equal(x[i], x0[i])
            else:
                assert any(torch.equal(x[i], alt[OFF[i] + v]) for v in range(K[i])), (side, i)
    # P = 0.5: доля заменённых среди строк с видами около половины, все виды встречаются, s и s' независимы
    va = O.ViewAug(bank, alt, 0.5, seed=1)
    ms, ms2 = bank["K_S"] > 0, bank["K_S2"] > 0
    xs, xs2 = torch.as_tensor(data["S"]).float(), torch.as_tensor(data["S2"]).float()
    hit, hit2, both, seen = np.zeros(len(rows)), np.zeros(len(rows)), 0, {}
    n = 1500
    for _ in range(n):
        a = va.pick(xs.clone(), rows, "s")
        b = va.pick(xs2.clone(), rows, "s2")
        ra = np.array([not torch.equal(a[i], xs[i]) for i in rows])
        rb = np.array([not torch.equal(b[i], xs2[i]) for i in rows])
        hit += ra
        hit2 += rb
        i = int(np.where(ms & ms2)[0][0])
        both += int(ra[i] and rb[i])
        if ra[i]:
            v = next(v for v in range(bank["K_S"][i]) if torch.equal(a[i], alt[bank["OFF_S"][i] + v]))
            seen[v] = seen.get(v, 0) + 1
    assert (hit[~ms] == 0).all() and (hit2[~ms2] == 0).all(), "без видов — без замены"
    fr = hit[ms] / n
    assert np.all(np.abs(fr - 0.5) < 0.06), fr
    assert abs(both / n - (hit[i] / n) * (hit2[i] / n)) < 0.04, (both / n, hit[i] / n, hit2[i] / n)
    assert sorted(seen) == list(range(bank["K_S"][i])) and min(seen.values()) > 0.6 * n * 0.5 / len(seen), seen
    sm = va.summary()
    assert abs(sm["view_aug_frac_s"] - 0.5 * ms.mean()) < 0.03 and abs(sm["view_aug_cand_s"] - ms.mean()) < 1e-4, sm
    print(f"замена карт: только у переходов с видами, доля {fr.mean():.3f} при P=0.5, s и s' независимы "
          f"(обе {both / n:.3f}), виды равновероятны {dict(sorted(seen.items()))}: OK")


def test_identity_views(data, mask):
    """Виды, равные картам среды строки: обучение бит в бит то же (у cnn нет BatchNorm, поток пачек тот же)."""
    n = len(data["A"])
    k = 3
    bank = dict(ALT=np.concatenate([np.repeat(data["S"], k, 0), np.repeat(data["S2"], k, 0)]),
                OFF_S=np.arange(n) * k, K_S=np.full(n, k), OFF_S2=n * k + np.arange(n) * k, K_S2=np.full(n, k), info={})
    d = dict(data, VIEWS=bank)
    for variant, kw in (("cnn_dqn", {}), ("cnn_dqn", dict(aug="d4"))):
        n0, _ = T.fit(variant, data, mask, epochs=3, **kw)
        n1, _ = T.fit(variant, d, mask, epochs=3, view_aug="x", view_aug_p=1.0, **kw)
        assert same(n0, n1), kw
        assert n1.extra["view_aug_frac_s"] == 1.0 and n1.extra["view_rows_s_train"] == int(mask.sum())
    # настоящие виды меняют обучение; согласованность считается и конечна; две сети с ResNet
    bank2 = dict(bank, ALT=bank["ALT"] + np.float32(0.3) * np.random.default_rng(0).normal(
        size=bank["ALT"].shape).astype(np.float32) * (np.arange(bank["ALT"].shape[1]) < 4)[None, :, None, None])
    d2 = dict(data, VIEWS=bank2)
    n0, _ = T.fit("cnn_dqn", data, mask, epochs=3)
    n2, _ = T.fit("cnn_dqn", d2, mask, epochs=3, view_aug="x", view_aug_p=0.5, view_cons=0.5)
    assert not same(n0, n2) and np.isfinite(n2.extra["view_cons_loss"]) and n2.extra["view_cons_loss"] > 0
    two, _ = T.fit("convnext_dqn", d2, mask, encoder="resnet", value_bound=True, two_nets=True, epochs=2,
                   batch_size=64, view_aug="x", view_aug_p=0.5, view_cons=0.5, wa="ema", wa_start=0.0)
    assert np.isfinite(two.extra["view_cons_loss"]) and two.extra["wa_n"] == 2
    print("виды, равные картам среды, обучение не меняют (веса бит в бит, и с D4); настоящие виды и "
          "согласованность — меняют, две сети с ResNet и --wa ema: OK")


def test_cons_zero(data):
    x = torch.as_tensor(data["S"][:6]).float()
    for variant, kw in (("cnn_dqn", {}), ("convnext_dqn", dict(encoder="resnet", two_nets=True))):
        torch.manual_seed(0)
        net = O.QNet(variant, CPU, in_ch=x.shape[1], **kw)
        net.target.load_state_dict(net.model.state_dict())
        net.set_train(False)
        net.target.eval()
        l0 = O.view_cons_loss(net, x, x.clone())
        assert l0.item() == 0.0, (variant, l0.item())
        y = x.clone()
        y[:, :3] += 0.5 * torch.randn_like(y[:, :3])
        l1 = O.view_cons_loss(net, x, y)
        assert float(l1) > 0 and l1.requires_grad
        l1.backward()
        assert all(p.grad is None for p in net.target.parameters()), "вторая сторона без градиента"
    print("согласованность видов: ноль при одинаковых видах (целевая сеть = онлайновая), больше нуля на разных, "
          "градиента в целевую сеть нет: OK")


def test_cosine(data, mask):
    lr0, lmin, tot = 1e-3, 1e-5, 40
    assert O.lr_at(0, tot, lr0, lmin, "cosine") == lr0
    assert abs(O.lr_at(tot // 2, tot, lr0, lmin, "cosine") - (lr0 + lmin) / 2) < 1e-15
    assert abs(O.lr_at(tot, tot, lr0, lmin, "cosine") - lmin) < 1e-15
    assert O.lr_at(17, tot, lr0, lmin, "const") == lr0
    vals = [O.lr_at(t, tot, lr0, 0.0, "cosine") for t in range(tot)]
    assert all(a > b for a, b in zip(vals, vals[1:])) and vals[-1] > 0
    # шаг Adam на каждом шаге обучения (пачки по 4 строки: 4 шага на эпоху)
    rec = []
    orig = torch.optim.Adam.step

    def spy(self, *a, **k):
        rec.append(self.param_groups[0]["lr"])
        return orig(self, *a, **k)
    torch.optim.Adam.step = spy
    try:
        n_tr = int(mask.sum())
        nb = math.ceil(n_tr / 4)
        net, _ = T.fit("cnn_dqn", data, mask, epochs=5, batch_size=4, lr=2e-3, lr_schedule="cosine", lr_min=1e-4)
    finally:
        torch.optim.Adam.step = orig
    total = 5 * nb
    want = [1e-4 + (2e-3 - 1e-4) * 0.5 * (1 + math.cos(math.pi * t / total)) for t in range(total)]
    assert len(rec) == total and np.allclose(rec, want, rtol=0, atol=1e-15), (len(rec), total)
    assert net.extra["lr_steps"] == total and abs(net.extra["lr_last"] - want[-1]) < 1e-15
    # lr_min = lr: косинус вырождается в постоянный шаг — веса как у const бит в бит; с --wa ema
    n0, _ = T.fit("cnn_dqn", data, mask, epochs=4, batch_size=64, wa="ema", wa_start=0.5)
    n1, _ = T.fit("cnn_dqn", data, mask, epochs=4, batch_size=64, wa="ema", wa_start=0.5, lr=1e-3,
                  lr_schedule="cosine", lr_min=1e-3)
    assert same(n0, n1) and n1.extra["wa_n"] == n0.extra["wa_n"] == 2
    n2, _ = T.fit("cnn_dqn", data, mask, epochs=4, batch_size=64, lr=1e-3, lr_schedule="cosine")
    n3, _ = T.fit("cnn_dqn", data, mask, epochs=4, batch_size=64)
    assert not same(n2, n3)
    print(f"косинусный план: lr_at (начало, середина, конец), шаг Adam на всех {total} шагах, lr_min = lr — веса "
          f"как у const (и с --wa ema): OK")


def write_dirs(tmp, eps, recs):
    tb = os.path.join(tmp, "tb")
    os.makedirs(os.path.join(tb, SUB))
    for k, ep in enumerate(eps):
        torch.save(ep, os.path.join(tb, SUB, f"c{k}.pt"))
    vd = os.path.join(tmp, "views")
    os.makedirs(os.path.join(vd, SUB))
    for k, r in enumerate(recs):
        MV.save_record(os.path.join(vd, SUB, f"v{k}.pt"), r)
    return tb, vd


def test_main(tmp, eps, recs):
    import killswitch
    import online_eval_env as E
    tb, vd = write_dirs(tmp, eps, recs)
    e0 = float(O.pde_meta()[PDE]["init_err"])
    mtag = f"ttm{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--target-buffer", tb, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--two-nets", "--seeds", "1", "--model-tag", mtag,
            "--epochs", "2", "--fqe-epochs", "1", "--batch-size", "8", "--cpu"]

    def run(av):
        buf = io.StringIO()
        sys.argv = av
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            O.main()
        text = buf.getvalue()
        return [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0], text

    old = (sys.argv, killswitch.killed, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    try:
        r0, text0 = run(argv)
        assert not any(k in r0 for k in NEW_FIELDS), [k for k in NEW_FIELDS if k in r0]
        assert "'ok': True" in text0
        arch0 = O.ckpt_arch(torch.load(out, map_location="cpu", weights_only=False))
        r1, text = run(argv + ["--view-aug", vd, "--view-aug-p", "0.5", "--view-cons", "0.2", "--epochs", "3",
                               "--lr-schedule", "cosine", "--wa", "ema", "--wa-start", "0.5"])
        assert "'ok': True" in text, "чекпоинт загружается online_eval_env"
        assert r1["view_aug"] == vd and r1["view_aug_p"] == 0.5 and r1["view_cons"] == 0.2
        assert r1["view_rows_s"] == 6 and r1["view_rows_s2"] == 6 and r1["view_records"] == len(recs)
        assert r1["view_mean_k_s"] > 2 and 0 <= r1["view_rows_s_train"] <= 6 and 0 <= r1["view_aug_frac_s"] <= 1
        assert r1["lr_schedule"] == "cosine" and r1["lr_min"] == 0.0 and r1["lr_steps"] > 0 and r1["lr_steps"] % 3 == 0
        assert r1["lr_last"] < 1e-3 and r1["wa"] == "ema" and r1["epochs"] == 3
        ck = torch.load(out, map_location="cpu", weights_only=False)
        assert ck["meta"]["view_aug"] == vd and ck["meta"]["views"]["rows_s"] == 6 and ck["meta"]["lr_schedule"] == "cosine"
        assert O.ckpt_arch(ck) == arch0, (O.ckpt_arch(ck), arch0)
        agent = E.load_agent(out)[0]
        assert agent._meta["lr_schedule"] == "cosine"
        bad_views = os.path.join(tmp, "views_none")
        os.makedirs(os.path.join(bad_views, SUB))
        MV.save_record(os.path.join(bad_views, SUB, "x.pt"), recs[3])        # только чужой сид
        for bad, what in ((argv + ["--view-aug", bad_views], "ни одного совпадения"),
                          ([a for a in argv if a not in ("--target-buffer", tb)] + ["--view-aug", vd],
                           "--view-aug без --target-buffer"),
                          (argv + ["--view-aug", vd, "--view-aug-p", "0"], "P = 0"),
                          (argv + ["--view-aug", vd, "--view-cons", "-1"], "отрицательный вес"),
                          (argv + ["--view-aug-p", "0.3"], "--view-aug-p без --view-aug"),
                          (argv + ["--view-cons", "0.3"], "--view-cons без --view-aug"),
                          (argv + ["--view-aug", vd, "--state-mode", "level"], "не full"),
                          (argv + ["--lr-min", "1e-5"], "--lr-min без cosine"),
                          (argv + ["--lr-schedule", "cosine", "--lr-min", "0.01"], "lr-min >= lr"),
                          ([a if a != "convnext_dqn" else "cnx_bbf" for a in argv if a not in ("--encoder", "resnet",
                                                                                              "--two-nets")]
                           + ["--lr-schedule", "cosine"], "cosine с cnx_bbf")):
            must_fail(lambda b=bad: run(b), SystemExit, f"недопустимое сочетание флагов: {what}")
    finally:
        sys.argv, killswitch.killed, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print(f"main(): по умолчанию строка без новых полей; --view-aug (s с видами {r1['view_rows_s']}, в среднем "
          f"{r1['view_mean_k_s']} видов), --view-cons, --lr-schedule cosine с --wa ema — поля в строке и meta, "
          f"чекпоинт загружается, ckpt_arch прежний; отказы разбора: OK")


def main():
    t0 = time.time()
    data = T.make_data(seed=2, n_chains=6, length=3)
    mask = data["EP"] < 5
    tmp = tempfile.mkdtemp(prefix="train_more_")
    try:
        test_defaults(data, mask)
        eps, recs, vdata, bank = test_matching()
        test_pick(vdata, bank)
        test_identity_views(data, mask)
        test_cons_zero(data)
        test_cosine(data, mask)
        test_main(tmp, eps, recs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_train_more: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
