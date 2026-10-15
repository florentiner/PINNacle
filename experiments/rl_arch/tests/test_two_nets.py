#!/usr/bin/env python
"""
Проверки агента из двух сетей (offline_rl.py --two-nets: сеть оптимизатора и сеть параметров, как
RL/rl_utils/DQN_classes.py статьи, наружу — совместное Q по 27 действиям) без GPU и без сети, на
крошечных синтетических данных (шесть цепочек по три шага, один-два шага градиента на проверку):

  * совместное Q: при центрировании max жадное совместное действие равно иерархическому выбору
    (оптимизатор по сети A, затем шаг и эпохи по сети B), Q_opt[o] = max по параметрам; при mean
    параметры по-прежнему выбирает сеть B, но оптимизатор — уже сумма двух сетей;
  * куда идёт градиент: плотная цель по 27 действиям учит все головы обеих сетей; редкий TD на
    жадных параметрах учит только сеть A, на нежадных — обе;
  * обучение (TD + модельные цели + цели учителя + граница ценности) двигает параметры обеих
    сетей; --encoder-init кладёт веса в оба энкодера, свежие каналы контекста у каждого свои;
  * чекпоинт через online_eval_env.load_agent и q_values отдаёт те же 27 значений, pick_action —
    иерархический выбор; verify_ckpt проходит; чекпоинт без поля two_nets в meta не читается молча;
  * значения по умолчанию — прежний путь (числа test_map_agent.REF, тот же разбор старых
    чекпоинтов); неподдержанные сочетания отвергаются; полный прогон main() с --two-nets.

Состав совместного Q (--two-nets-mode additive | joint | auto) и общий энкодер (--two-nets-shared):

  * в каждом составе, со своим и с общим энкодером, жадное совместное действие при центрировании
    max равно последовательному выбору (оптимизатор, затем параметры; у auto — оптимизатор, шаг,
    длительность); у joint и auto нет суммы по шагу и длительности;
  * случайную плотную цель по 27 действиям со взаимодействием шага и длительности joint и auto
    выучивают на крошечной пачке почти до нуля, additive упирается в остаток лучшей суммы
    «шаг + длительность» (изъян прежнего состава);
  * общий энкодер: у сети B только головы, ключей '2.*' нет; --two-nets-shared-grad stop не
    пускает градиент сети B в энкодер; обучение с плотными целями двигает обе сети;
  * чекпоинт каждого состава через load_agent и q_values; чекпоинт без состава в meta читается
    как additive; main() с новыми флагами, двумя сидами и --target-buffer.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_two_nets.py
    ... test_two_nets.py modes fit      # только названные проверки (имена — в TESTS)
"""
import contextlib
import io
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import offline_rl as O  # noqa: E402
import mb_targets as MB  # noqa: E402
import test_map_agent as TM  # noqa: E402  — те же синтетические данные и подставной суррогат
import test_shield_teacher as TS  # noqa: E402  — буферы и настоящий чекпоинт учителя на скалярах

torch.set_num_threads(1)
CPU = torch.device("cpu")
TWO = dict(two_nets=True)


def hier_np(q_o, a_lr, a_ep):
    """Иерархический выбор статьи на numpy: оптимизатор, затем его шаг и его число эпох."""
    o = q_o.argmax(1)
    r = np.arange(len(o))
    return o * 9 + a_lr[r, o].argmax(1) * 3 + a_ep[r, o].argmax(1)


def changed(net, ref, i):
    """Изменился ли хоть один параметр model[i] относительно сети с той же инициализацией."""
    return any(not torch.equal(p, q) for p, q in zip(net.model[i].parameters(), ref.model[i].parameters()))


def must_fail(fn, exc, what):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f"{what}: должно отвергаться")


# --------------------------------------------------------------------------------------------

def test_compose():
    g = torch.Generator().manual_seed(0)
    q_o, a_lr, a_ep = torch.randn(500, 3, generator=g), torch.randn(500, 3, 3, generator=g), torch.randn(500, 3, 3, generator=g)
    q = O.compose_two_nets(q_o, a_lr, a_ep, "max")
    assert q.shape == (500, 27)
    # кодировка a = opt*9 + lr*3 + ep и сама формула
    for o, l, e in ((0, 0, 0), (1, 2, 0), (2, 1, 2)):
        want = q_o[:, o] + a_lr[:, o, l] - a_lr[:, o].amax(1) + a_ep[:, o, e] - a_ep[:, o].amax(1)
        assert torch.allclose(q[:, o * 9 + l * 3 + e], want, atol=1e-6)
    assert (q.argmax(1).numpy() == hier_np(q_o.numpy(), a_lr.numpy(), a_ep.numpy())).all()
    assert torch.equal(q.view(-1, 3, 9).amax(2), q_o), "Q_opt — ценность оптимизатора с лучшими параметрами"
    # среднее: Q_opt — среднее по параметрам; параметры выбирает сеть B, оптимизатор — сумма сетей
    qm = O.compose_two_nets(q_o, a_lr, a_ep, "mean")
    assert torch.allclose(qm.view(-1, 3, 9).mean(2), q_o, atol=1e-6)
    am, r = qm.argmax(1), torch.arange(500)
    om = am // 9
    assert torch.equal((am % 9) // 3, a_lr[r, om].argmax(1)) and torch.equal(am % 3, a_ep[r, om].argmax(1))
    assert (om != q_o.argmax(1)).any(), "при mean оптимизатор не обязан совпадать с argmax сети A"
    span = q_o + (a_lr.amax(2) - a_lr.mean(2)) + (a_ep.amax(2) - a_ep.mean(2))
    assert torch.equal(om, span.argmax(1))
    must_fail(lambda: O.compose_two_nets(q_o, a_lr, a_ep, "median"), ValueError, "неизвестное центрирование")

    # случайные сети: каждый энкодер, обе головы, вход с контекстом
    rng = np.random.default_rng(1)
    for enc, head, in_ch in (("auto", "linear", 4), ("resnet", "linear", 9), ("resnet", "dueling", 9),
                             ("vit", "linear", 4), ("resnetpos", "dueling", 4)):
        torch.manual_seed(7)
        net = O.QNet("convnext_dqn", CPU, in_ch=in_ch, encoder=enc, two_head=head, **TWO)
        net.set_train(False)
        x = torch.as_tensor(rng.normal(size=(32, in_ch, 26, 26)).astype(np.float32))
        with torch.no_grad():
            q = net.q_scalar(x)
            parts = [t.numpy() for t in net.q_parts(x)]
            assert q.shape == (32, 27) and torch.isfinite(q).all()
            assert torch.equal(q.argmax(1), net.hier_action(x))
            assert (q.argmax(1).numpy() == hier_np(*parts)).all()
            assert torch.equal(net.q_online(x), q) and torch.equal(net.q_heads(x), q[None])
            # целевая сеть — та же сумма по целевым копиям обеих сетей (на старте копии равны)
            assert torch.equal(net.q_target(x), q) and torch.equal(net.q_heads(x, target=True), q[None])
            for p in net.model[3].parameters():
                p.add_(0.3)
            assert torch.equal(net.q_target(x), q), "целевая сеть не должна видеть обучаемую сеть B"
        sh = [k for k in net.model.state_dict()]
        assert {k.split(".")[0] for k in sh} == {"0", "1", "2", "3"}
        assert [k for k in sh if k.startswith("3.")][:2] == (
            ["3.lr.0.weight", "3.lr.0.bias"] if head == "linear" else ["3.lr.0.0.weight", "3.lr.0.0.bias"])
        # энкодеры раздельные: одного вида, но не один и тот же модуль и не одни и те же веса
        assert net.model[0] is not net.model[2]
        assert [tuple(p.shape) for p in net.model[0].parameters()] == [tuple(p.shape) for p in net.model[2].parameters()]
        assert not torch.equal(next(net.model[0].parameters()), next(net.model[2].parameters()))
        n_a, n_b = net.n_params_nets()
        assert n_a + n_b == net.n_params() and n_b > n_a > 0
    print("совместное Q: жадное действие = иерархический выбор (max), состав при mean: OK")


def test_gradients():
    rng = np.random.default_rng(2)
    x = torch.as_tensor(rng.normal(size=(8, 9, 26, 26)).astype(np.float32))
    for center in O.TWO_NETS_CENTER:
        torch.manual_seed(5)
        net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", two_center=center, **TWO)
        # плотная цель по всем 27 действиям (как --mb-aux и --teacher-w): градиент во все головы
        loss = F.smooth_l1_loss(net.q_online(x), torch.as_tensor(rng.normal(size=(8, 27)).astype(np.float32)))
        net.model.zero_grad(); loss.backward()
        for i in range(4):
            assert sum(float(p.grad.abs().sum()) for p in net.model[i].parameters()) > 0, (center, i)
        for h in list(net.model[3].lr) + list(net.model[3].ep):
            assert float(h.weight.grad.abs().sum()) > 0, f"{center}: голова параметра без градиента"
        assert all(p.grad is None for p in net.target.parameters())
    # редкий TD (одно взятое действие), центрирование max
    torch.manual_seed(5)
    net = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", **TWO)
    net.set_train(False)                     # статистики BatchNorm не должны связывать строки пачки
    greedy = net.hier_action(x)

    def grads(a):
        net.model.zero_grad()
        net.q_online(x).gather(1, a[:, None]).sum().backward()
        return [sum(float(p.grad.abs().sum()) for p in net.model[i].parameters() if p.grad is not None)
                for i in range(4)]
    g = grads(greedy)
    assert g[0] > 0 and g[1] > 0 and g[2] == 0 and g[3] == 0, f"жадные параметры: сеть B без градиента, {g}"
    g = grads((greedy // 9) * 9 + (greedy % 9 + 4) % 9)      # тот же оптимизатор, другие шаг и эпохи
    assert min(g) > 0, f"нежадные параметры: градиент в обе сети, {g}"
    print("градиент: плотная цель учит все головы, редкий TD на жадных параметрах — только сеть A: OK")


def test_training(tmp):
    d9 = TM.make_data(in_ch=9)
    mask = d9["EP"] < 4
    rng = np.random.default_rng(3)
    n = len(d9["A"])
    dense = dict(d9, QMB=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), MBW=mask.copy(),
                 QTE=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), TEV=np.ones(n, bool))
    for kw in (dict(), dict(two_nets_center="mean"), dict(two_nets_head="dueling"), dict(encoder="auto")):
        kw = dict(dict(encoder="resnet", value_bound=True, mb_aux=0.5, teacher_w=0.5, two_nets=True), **kw)
        net, _ = TM.fit("convnext_dqn", dense, mask, **kw)
        torch.manual_seed(3)                # та же инициализация, что внутри train_variant(seed=3)
        ref = O.QNet("convnext_dqn", CPU, in_ch=9, encoder=kw["encoder"], two_nets=True,
                     two_head=kw.get("two_nets_head", "linear"))
        assert all(changed(net, ref, i) for i in range(4)), "учатся энкодер и головы обеих сетей"
        for h, h0 in zip(list(net.model[3].lr) + list(net.model[3].ep), list(ref.model[3].lr) + list(ref.model[3].ep)):
            assert changed_mod(h, h0), "плотные цели двигают каждую голову параметра"
        assert net.two_nets and net.two_center == kw.get("two_nets_center", "max")
        assert np.isfinite(net.extra["mb_loss"]) and np.isfinite(net.extra["teacher_loss"])
        assert net.extra["n_params_opt_net"] + net.extra["n_params_par_net"] == net.n_params()
        assert 0.0 <= net.extra["teacher_agree"] <= 1.0 and not (net.has_bn and net.model.training)
        # целевые копии обеих сетей идут за обучаемыми (soft_update), статистики BatchNorm — копией
        assert all(changed_mod(net.target[i], ref.model[i]) for i in range(4))
        if net.has_bn:
            for i in (0, 2):
                for b, tb in zip(net.model[i].buffers(), net.target[i].buffers()):
                    assert torch.equal(b, tb)
    # один только TD (без плотных целей) тоже учит обе сети; cnx_cql — тот же путь с добавкой CQL
    for variant in O.TWO_NETS_VARIANTS:
        net, _ = TM.fit(variant, d9, mask, encoder="resnet", value_bound=True, two_nets=True, epochs=2)
        torch.manual_seed(3)
        ref = O.QNet(variant, CPU, in_ch=9, encoder="resnet", **TWO)
        assert all(changed(net, ref, i) for i in range(4)), variant
    # прочие добавки тренера считаются по тому же совместному Q: шум по входу, отступ на лучших
    # цепочках, грубая маска, аугментация, n шагов, DQfD — всё вместе с плотными целями
    net, _ = TM.fit("cnx_cql", dense, mask, encoder="resnet", value_bound=True, mb_aux=0.5, teacher_w=0.5,
                    pani_w=0.1, pani_sigma=0.1, distill_top=50.0, distill_w=0.3, mask_schedule=True, aug="d4",
                    n_step=2, dqfd=0.1, epochs=3, two_nets=True)
    assert {"mb_loss", "teacher_loss", "pani_loss", "distill_top_loss"} <= set(net.extra)
    assert all(np.isfinite(v) for v in net.extra.values() if isinstance(v, float))
    # --encoder-init: веса получают оба энкодера, свежие каналы контекста у каждого свои
    path = os.path.join(tmp, "enc_resnet.pt")
    torch.manual_seed(11)
    f = O.make_encoder("resnet", 4).state_dict()
    torch.save(f, path)
    q9 = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", **TWO)
    info = O.load_encoder_init(q9, path)
    assert info["partial"] == ["stem.0.weight"], info
    wa, wb = q9.model[0].stem[0].weight, q9.model[2].stem[0].weight
    assert torch.equal(wa[:, :4], f["stem.0.weight"]) and torch.equal(wb[:, :4], f["stem.0.weight"])
    assert not torch.equal(wa[:, 4:], wb[:, 4:])
    for i in (0, 2):
        sd, tsd = q9.model[i].state_dict(), q9.target[i].state_dict()
        assert all(torch.equal(sd[k], tsd[k]) for k in sd)
        assert all(torch.equal(sd[k], f[k]) for k in sd if k != "stem.0.weight")
    net, _ = TM.fit("convnext_dqn", dense, mask, encoder="resnet", encoder_init=path, mb_aux=0.5, teacher_w=0.5,
                    two_nets=True)
    for i in (0, 2):
        w = net.model[i].state_dict()["stem.0.weight"][:, :4]
        assert not torch.equal(w, f["stem.0.weight"]) and (w - f["stem.0.weight"]).abs().max() < 0.05
    # заморозка: оба энкодера стоят (кроме первой свёртки со свежими каналами), головы учатся
    net, _ = TM.fit("convnext_dqn", dense, mask, encoder="resnet", encoder_init=path, freeze_encoder=True,
                    mb_aux=0.5, two_nets=True, state_mode="full")
    for i in (0, 2):
        assert [k for k, p in net.model[i].named_parameters() if p.requires_grad] == ["stem.0.weight"]
        sd = net.model[i].state_dict()
        assert all(torch.equal(sd[k], f[k]) for k in sd if k != "stem.0.weight") and not net.model[i].training
    print("обучение: TD, модельные цели и цели учителя двигают обе сети; --encoder-init в оба энкодера: OK")


def changed_mod(m, m0):
    return any(not torch.equal(p, q) for p, q in zip(m.parameters(), m0.parameters()))


def two_meta(net):
    """Поля meta, которые offline_rl.main (--save-model) писал для двух сетей до --two-nets-mode и
    --two-nets-shared: состава и вида энкодера сети параметров в них нет."""
    return dict(encoder=net.enc_kind, ens=net.ens, two_nets=True, two_nets_center=net.two_center,
                two_nets_head=net.two_head)


def mode_meta(net):
    """Поля meta, которые offline_rl.main (--save-model) пишет для двух сетей сейчас."""
    m = dict(two_meta(net), two_nets_mode=net.two_mode, two_nets_shared=net.two_shared)
    if net.two_shared:
        m["two_nets_shared_grad"] = net.two_shared_grad
    return m


# --------------------------------------------------------------------------------------------
# состав совместного Q (--two-nets-mode) и общий энкодер (--two-nets-shared)

MODES = [(m, sh) for m in O.TWO_NETS_MODE for sh in (False, True)]
N_OUT_B = dict(additive=18, joint=27, auto=6, lowrank=36)       # выходов у сети параметров


def mk(mode, shared, **kw):
    kw = dict(dict(in_ch=9, encoder="resnet"), **kw)
    variant = kw.pop("variant", "convnext_dqn")
    return O.QNet(variant, CPU, two_mode=mode, two_shared=shared, **TWO, **kw)


def seq_np(mode, q_o, *par):
    """Последовательный выбор на numpy: оптимизатор по сети A, затем параметры по сети B (additive —
    шаг и эпохи независимо, joint — лучшая пара, auto — шаг, затем эпохи при этом шаге)."""
    o = q_o.argmax(1)
    r = np.arange(len(o))
    if mode == "joint":
        return o * 9 + par[0][r, o].argmax(1)
    if mode == "lowrank":
        t = par[0][r, o][:, :, None] + par[1][r, o][:, None, :] + np.einsum("blk,bek->ble", par[2][r, o], par[3][r, o])
        return o * 9 + t.reshape(len(o), -1).argmax(1)
    l = par[0][r, o].argmax(1)
    return o * 9 + l * 3 + (par[1][r, o, l] if mode == "auto" else par[1][r, o]).argmax(1)


def b_units(net):
    """Обучаемые части сети параметров, каждая из которых обязана получать градиент плотной цели."""
    b = net.model[3]
    return {"additive": lambda: list(b.lr) + list(b.ep), "joint": lambda: list(b.par),
            "auto": lambda: [b.emb_o, b.emb_l, b.lr, b.ep], "lowrank": lambda: list(b.lr) + list(b.ep)}[net.two_mode]()


def gsum(m):
    return sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)


def inter(q):
    """Взаимодействие шага и длительности: смешанная разность Q по (шаг, эпохи) у каждого оптимизатора."""
    t = q.view(-1, 3, 3, 3)
    return (t[:, :, 0, 0] - t[:, :, 0, 1] - t[:, :, 1, 0] + t[:, :, 1, 1]).abs()


def test_modes():
    g = torch.Generator().manual_seed(0)
    n, r = 500, torch.arange(500)
    q_o, a_lr = torch.randn(n, 3, generator=g), torch.randn(n, 3, 3, generator=g)
    a_par, a_ep = torch.randn(n, 3, 9, generator=g), torch.randn(n, 3, 3, 3, generator=g)
    # joint: одна таблица пар у оптимизатора
    q = O.compose_two_nets_joint(q_o, a_par, "max")
    assert q.shape == (n, 27)
    for o, l, e in ((0, 0, 0), (1, 2, 0), (2, 1, 2)):
        assert torch.allclose(q[:, o * 9 + l * 3 + e], q_o[:, o] + a_par[:, o, l * 3 + e] - a_par[:, o].amax(1), atol=1e-6)
    assert (q.argmax(1).numpy() == seq_np("joint", q_o.numpy(), a_par.numpy())).all()
    assert torch.equal(q.view(-1, 3, 9).amax(2), q_o), "Q_opt — ценность оптимизатора с лучшей парой"
    assert float(inter(q).median()) > 0.1, "joint: суммы по шагу и длительности нет"
    qm = O.compose_two_nets_joint(q_o, a_par, "mean")
    assert torch.allclose(qm.view(-1, 3, 9).mean(2), q_o, atol=1e-6)
    am = qm.argmax(1)
    assert torch.equal(am % 9, a_par[r, am // 9].argmax(1)) and (am // 9 != q_o.argmax(1)).any()
    # auto: длительность зависит от шага
    q = O.compose_two_nets_auto(q_o, a_lr, a_ep, "max")
    assert q.shape == (n, 27)
    for o, l, e in ((0, 0, 0), (1, 2, 0), (2, 1, 2)):
        want = q_o[:, o] + a_lr[:, o, l] - a_lr[:, o].amax(1) + a_ep[:, o, l, e] - a_ep[:, o, l].amax(1)
        assert torch.allclose(q[:, o * 9 + l * 3 + e], want, atol=1e-6)
    assert (q.argmax(1).numpy() == seq_np("auto", q_o.numpy(), a_lr.numpy(), a_ep.numpy())).all()
    assert torch.equal(q.view(-1, 3, 9).amax(2), q_o)
    # ценность шага с лучшей длительностью — Q_opt плюс преимущество шага
    assert torch.allclose(q.view(-1, 3, 3, 3).amax(3), q_o[:, :, None] + a_lr - a_lr.amax(2, keepdim=True), atol=1e-6)
    assert float(inter(q).median()) > 0.1, "auto: суммы по шагу и длительности нет"
    assert float(inter(O.compose_two_nets(q_o, a_lr, a_lr, "max")).max()) < 1e-5, "additive: только сумма"
    qm = O.compose_two_nets_auto(q_o, a_lr, a_ep, "mean")
    assert torch.allclose(qm.view(-1, 3, 9).mean(2), q_o, atol=1e-6)
    am = qm.argmax(1)
    assert torch.equal(am % 3, a_ep[r, am // 9, (am % 9) // 3].argmax(1)), "эпохи при выбранном шаге даёт сеть B"
    assert (am.numpy() != seq_np("auto", q_o.numpy(), a_lr.numpy(), a_ep.numpy())).any(), \
        "при mean совместный argmax не обязан совпадать с последовательным выбором"
    must_fail(lambda: O.compose_two_nets_joint(q_o, a_par, "median"), ValueError, "неизвестное центрирование")
    must_fail(lambda: O.compose_two_nets_auto(q_o, a_lr, a_ep, "median"), ValueError, "неизвестное центрирование")
    # lowrank: сумма additive плюс <U, V>; при нулевых U, V — ровно additive, при ненулевых взаимодействие есть
    g = torch.Generator().manual_seed(3)
    qo2, al2, ae2 = (torch.randn(64, *sh, generator=g) for sh in ((3,), (3, 3), (3, 3)))
    u0 = torch.zeros(64, 3, 3, O.TWO_NETS_RANK)
    for c in ("max", "mean"):
        qa = O.compose_two_nets(qo2, al2, ae2, c)
        assert torch.allclose(O.compose_two_nets_lowrank(qo2, al2, ae2, u0, u0, c), qa, atol=1e-5), c
    uu = torch.randn(64, 3, 3, O.TWO_NETS_RANK, generator=g)
    ql = O.compose_two_nets_lowrank(qo2, al2, ae2, uu, uu.flip(2), "max")
    assert float(inter(ql).max()) > 1e-3, "lowrank выражает взаимодействие шага и длительности"
    assert torch.allclose(ql.view(-1, 3, 9).amax(2), qo2, atol=1e-6)

    # случайные сети: каждый состав со своим и с общим энкодером
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(size=(16, 9, 26, 26)).astype(np.float32))
    one = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet")
    cases = [(m, sh, dict()) for m, sh in MODES] + [
        ("joint", True, dict(two_head="dueling", two_center="mean")), ("auto", False, dict(two_head="dueling")),
        ("auto", True, dict(two_center="mean", encoder="auto")), ("auto", True, dict(encoder="vit")),
        ("joint", False, dict(encoder="resnetpos", variant="cnx_cql"))]
    shapes = dict(additive=[(3,), (3, 3), (3, 3)], joint=[(3,), (3, 9)], auto=[(3,), (3, 3), (3, 3, 3)],
                  lowrank=[(3,), (3, 3), (3, 3), (3, 3, O.TWO_NETS_RANK), (3, 3, O.TWO_NETS_RANK)])
    for mode, shared, kw in cases:
        torch.manual_seed(7)
        net = mk(mode, shared, **kw)
        net.set_train(False)
        assert (net.two_mode, net.two_shared, net.two_shared_grad) == (mode, shared, "both")
        with torch.no_grad():
            q = net.q_scalar(x)
            parts = [t.numpy() for t in net.q_parts(x)]
            assert q.shape == (16, 27) and torch.isfinite(q).all()
            assert [p.shape[1:] for p in parts] == shapes[mode]
            if net.two_center == "max":
                assert torch.equal(q.argmax(1), net.hier_action(x)), (mode, shared)
                assert (q.argmax(1).numpy() == seq_np(mode, *parts)).all(), (mode, shared)
                assert torch.allclose(q.view(-1, 3, 9).amax(2), torch.as_tensor(parts[0]), atol=1e-6)
            else:
                assert torch.allclose(q.view(-1, 3, 9).mean(2), torch.as_tensor(parts[0]), atol=1e-5)
            assert (float(inter(q).max()) < 1e-5) == (mode == "additive"), (mode, float(inter(q).max()))
            assert torch.equal(net.q_online(x), q) and torch.equal(net.q_heads(x), q[None])
            # целевая сеть — тот же состав по целевым копиям и обучаемую сеть B не видит
            assert torch.equal(net.q_target(x), q) and torch.equal(net.q_heads(x, target=True), q[None])
            for prm in net.model[3].parameters():
                prm.mul_(1.5)
            assert torch.equal(net.q_target(x), q) and not torch.equal(net.q_online(x), q)
        keys = list(net.model.state_dict())
        assert {k.split(".")[0] for k in keys} == ({"0", "1", "3"} if shared else {"0", "1", "2", "3"})
        assert isinstance(net.model[2], torch.nn.Identity) == shared and net._enc_idx == ((0,) if shared else (0, 2))
        assert any(k.startswith(dict(additive="3.lr.0.", joint="3.par.2.", auto="3.emb_o.", lowrank="3.lr.0.")[mode]) for k in keys)
        n_a, n_b = net.n_params_nets()
        assert n_a + n_b == net.n_params() and (n_b < n_a if shared else n_b > n_a), (mode, shared, n_a, n_b)
        assert net.n_outputs_nets() == (3, N_OUT_B[mode])
        if not kw:
            print(f"  {mode:8s} энкодер {'общий' if shared else 'свой '}: выходов 3 + {N_OUT_B[mode]:2d}, параметров "
                  f"{net.n_params()} (A {n_a}, B {n_b}), к одной сети ({one.n_params()}) x{net.n_params() / one.n_params():.3f}")
    # общий энкодер и свой дают одну и ту же сеть A при одном сиде
    torch.manual_seed(1)
    a = mk("joint", False)
    torch.manual_seed(1)
    b = mk("joint", True)
    assert all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items() if k[0] in "01")

    # куда идёт градиент
    t = torch.as_tensor(rng.normal(size=(16, 27)).astype(np.float32))
    for mode, shared in MODES:
        for grad in (("both", "stop") if shared else ("both",)):
            torch.manual_seed(5)
            net = mk(mode, shared, two_shared_grad=grad)
            # плотная цель: градиент во все головы и вложения сети B и в сеть A
            net.model.zero_grad()
            F.smooth_l1_loss(net.q_online(x), t).backward()
            assert all(gsum(net.model[i]) > 0 for i in ((0, 1, 3) if shared else range(4))), (mode, shared, grad)
            assert all(gsum(u) > 0 for u in b_units(net)), f"{mode}: часть сети параметров без градиента"
            # лосс одной сети B: в общий энкодер он идёт только при both, в энкодер и голову сети A со
            # своим энкодером — никогда
            net.model.zero_grad()
            sum(a_.pow(2).sum() for a_ in net.q_parts(x)[1:]).backward()
            assert gsum(net.model[3]) > 0 and gsum(net.model[1]) == 0
            assert (gsum(net.model[0]) > 0) == (shared and grad == "both"), (mode, shared, grad, gsum(net.model[0]))
            assert shared or gsum(net.model[2]) > 0
            # редкий TD на жадном действии (max): преимущества на нём нулевые, сеть B градиента не получает
            net.set_train(False)
            net.model.zero_grad()
            net.q_online(x).gather(1, net.hier_action(x)[:, None]).sum().backward()
            assert gsum(net.model[0]) > 0 and gsum(net.model[1]) > 0 and gsum(net.model[3]) == 0, (mode, shared)
            assert shared or gsum(net.model[2]) == 0
    torch.manual_seed(5)
    a = mk("auto", True)
    torch.manual_seed(5)
    b = mk("auto", True, two_shared_grad="stop")
    with torch.no_grad():
        assert torch.equal(a.q_online(x), b.q_online(x)), "stop меняет только градиент, не значения"
    for what, bad in (("состав", lambda: mk("tree", False)), ("градиент", lambda: mk("joint", True, two_shared_grad="half")),
                      ("stop без общего энкодера", lambda: mk("joint", False, two_shared_grad="stop")),
                      ("состав без two_nets", lambda: O.QNet("convnext_dqn", CPU, two_mode="joint")),
                      ("общий энкодер без two_nets", lambda: O.QNet("convnext_dqn", CPU, two_shared=True)),
                      ("квантильный вариант", lambda: mk("auto", True, variant="cnx_cql_qr")),
                      ("ансамбль голов", lambda: mk("joint", True, ens=2))):
        must_fail(bad, ValueError, what)
    print("составы joint и auto, общий энкодер: жадное действие = последовательный выбор (max), градиенты: OK")


def test_fit(steps=250):
    """Изъян состава additive: плотную цель по 27 действиям со взаимодействием шага и длительности
    он не выучивает даже на четырёх состояниях; joint и auto выучивают."""
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(4, 9, 26, 26)).astype(np.float32))
    t = torch.as_tensor(rng.normal(size=(4, 27)).astype(np.float32))
    # нижняя граница MSE для суммы «оптимизатор + шаг + длительность»: остаток МНК внутри оптимизатора
    tt = t.view(4, 3, 3, 3)
    floor = float((tt - tt.mean(3, keepdim=True) - tt.mean(2, keepdim=True) + tt.mean((2, 3), keepdim=True)).pow(2).mean())
    assert floor > 0.2, floor
    out = {}
    for mode, shared in MODES:
        torch.manual_seed(0)
        net = mk(mode, shared)
        net.set_train(False)
        opt = torch.optim.Adam(net.params(), lr=2e-3)
        for _ in range(steps):
            loss = F.mse_loss(net.q_online(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out[mode, shared] = float(F.mse_loss(net.q_online(x), t))
        if mode == "additive":
            assert floor - 1e-5 <= out[mode, shared] < 1.2 * floor, (shared, out[mode, shared], floor)
        elif mode == "lowrank":
            # взаимодействие ранга TWO_NETS_RANK: случайную таблицу 3x3 (ранг взаимодействия до 2) целиком
            # не выразить, но остаток суммы снимается в разы
            assert out[mode, shared] < 0.1 * floor, (mode, shared, out[mode, shared], floor)
        else:
            assert out[mode, shared] < 0.01 * floor, (mode, shared, out[mode, shared], floor)
    print(f"  цель со взаимодействием, MSE после {steps} шагов (остаток лучшей суммы {floor:.4f}): "
          + "; ".join(f"{m}{'+общий' if sh else ''} {v:.2e}" for (m, sh), v in out.items()))
    print("плотная цель со взаимодействием шага и длительности: joint и auto выучивают, lowrank почти, additive нет: OK")


def test_modes_training(tmp):
    d9 = TM.make_data(in_ch=9)
    mask = d9["EP"] < 4
    rng = np.random.default_rng(3)
    n = len(d9["A"])
    dense = dict(d9, QMB=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), MBW=mask.copy(),
                 QTE=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), TEV=np.ones(n, bool))
    cases = [(m, sh, dict()) for m, sh in MODES[1:]] + [
        ("auto", True, dict(two_nets_shared_grad="stop")),
        ("joint", True, dict(two_nets_center="mean", two_nets_head="dueling")),
        ("joint", True, dict(two_nets_shared_grad="stop", encoder="auto"))]
    for mode, shared, kw in cases:
        kw = dict(dict(encoder="resnet", value_bound=True, mb_aux=0.5, teacher_w=0.5, two_nets=True,
                       two_nets_mode=mode, two_nets_shared=shared), **kw)
        net, _ = TM.fit("convnext_dqn", dense, mask, **kw)
        torch.manual_seed(3)                # та же инициализация, что внутри train_variant(seed=3)
        ref = mk(mode, shared, encoder=kw["encoder"], two_head=kw.get("two_nets_head", "linear"))
        live = (0, 1, 3) if shared else (0, 1, 2, 3)
        assert all(changed(net, ref, i) for i in live), (mode, shared, "учатся энкодер и головы обеих сетей")
        assert all(changed_mod(u, u0) for u, u0 in zip(b_units(net), b_units(ref))), f"{mode}: плотные цели двигают всю сеть B"
        assert (net.two_mode, net.two_shared, net.two_shared_grad) == (mode, shared, kw.get("two_nets_shared_grad", "both"))
        ex = net.extra
        assert ex["n_params_opt_net"] + ex["n_params_par_net"] == net.n_params()
        assert (ex["n_out_opt_net"], ex["n_out_par_net"]) == (3, N_OUT_B[mode])
        assert np.isfinite(ex["mb_loss"]) and np.isfinite(ex["teacher_loss"]) and 0.0 <= ex["teacher_agree"] <= 1.0
        assert all(changed_mod(net.target[i], ref.model[i]) for i in live) and not (net.has_bn and net.model.training)
        if net.has_bn:
            for i in net._enc_idx:
                for b, tb in zip(net.model[i].buffers(), net.target[i].buffers()):
                    assert torch.equal(b, tb)
    # один только TD и прочие добавки тренера на совместном Q
    net, _ = TM.fit("cnx_cql", d9, mask, encoder="resnet", value_bound=True, two_nets=True, two_nets_mode="auto", epochs=2)
    torch.manual_seed(3)
    ref = mk("auto", False, variant="cnx_cql")
    assert all(changed(net, ref, i) for i in range(4))
    net, _ = TM.fit("cnx_cql", dense, mask, encoder="resnet", value_bound=True, mb_aux=0.5, teacher_w=0.5,
                    pani_w=0.1, pani_sigma=0.1, distill_top=50.0, distill_w=0.3, mask_schedule=True, aug="d4",
                    n_step=2, dqfd=0.1, epochs=2, two_nets=True, two_nets_mode="joint", two_nets_shared=True)
    assert {"mb_loss", "teacher_loss", "pani_loss", "distill_top_loss"} <= set(net.extra)
    assert all(np.isfinite(v) for v in net.extra.values() if isinstance(v, float))
    # --encoder-init с общим энкодером: он один; заморозка оставляет учиться головы обеих сетей
    path = os.path.join(tmp, "enc_resnet_sh.pt")
    torch.manual_seed(11)
    f = O.make_encoder("resnet", 4).state_dict()
    torch.save(f, path)
    q9 = mk("auto", True)
    info = O.load_encoder_init(q9, path)
    assert info["partial"] == ["stem.0.weight"] and torch.equal(q9.model[0].stem[0].weight[:, :4], f["stem.0.weight"])
    assert all(torch.equal(v, q9.target[0].state_dict()[k]) for k, v in q9.model[0].state_dict().items())
    net, _ = TM.fit("convnext_dqn", dense, mask, encoder="resnet", encoder_init=path, freeze_encoder=True, mb_aux=0.5,
                    two_nets=True, two_nets_mode="auto", two_nets_shared=True, state_mode="full")
    assert [k for k, prm in net.model[0].named_parameters() if prm.requires_grad] == ["stem.0.weight"]
    sd = net.model[0].state_dict()
    assert all(torch.equal(sd[k], f[k]) for k in sd if k != "stem.0.weight") and not net.model[0].training
    torch.manual_seed(3)
    ref = mk("auto", True)
    assert changed(net, ref, 1) and all(changed_mod(u, u0) for u, u0 in zip(b_units(net), b_units(ref)))
    for what, kw in (("состав", dict(two_nets_mode="tree")), ("stop без общего энкодера", dict(two_nets_shared_grad="stop")),
                     ("--aux-next", dict(two_nets_mode="joint", aux_next=0.5))):
        must_fail(lambda: TM.fit("convnext_dqn", d9, mask, two_nets=True, **kw), ValueError, what)
    must_fail(lambda: TM.fit("convnext_dqn", d9, mask, two_nets_mode="joint"), ValueError, "состав без --two-nets")
    print("обучение составов joint и auto, общий энкодер: плотные цели двигают обе сети; --encoder-init, заморозка: OK")


def test_modes_ckpt(tmp):
    import online_eval_env as E
    torch.set_default_tensor_type(torch.FloatTensor)
    rng = np.random.default_rng(4)
    paths = {}
    cases = [(9, m, sh, dict()) for m, sh in MODES] + [
        (4, "auto", True, dict(two_nets_head="dueling", two_nets_shared_grad="stop")),
        (4, "joint", False, dict(two_nets_center="mean", encoder="vit"))]
    for in_ch, mode, shared, kw in cases:
        data = TM.make_data(in_ch=in_ch)
        mask = data["EP"] < 4
        kw = dict(dict(encoder="resnet"), **kw)
        net, stats = TM.fit("convnext_dqn", data, mask, value_bound=True, two_nets=True, two_nets_mode=mode,
                            two_nets_shared=shared, **kw)
        path = TM.save_ckpt(net, stats, os.path.join(tmp, f"m_{in_ch}_{mode}_{int(shared)}_{net.enc_kind}.pt"),
                            meta=mode_meta(net))
        paths[in_ch, mode, shared] = path
        ck = torch.load(path, weights_only=False)
        assert any(k.startswith("2.") for k in ck["state_dict"]) != shared
        assert O.ckpt_arch(ck) == dict(
            n_quantiles=32, in_ch=in_ch, encoder=net.enc_kind, ens=1, two_nets=True, two_center=net.two_center,
            two_head=net.two_head, two_mode=mode, two_shared=shared,
            **(dict(two_shared_grad=net.two_shared_grad) if shared else {}))
        ag, mean, std, var = E.load_agent(path)
        assert ag.two_nets and (ag.two_mode, ag.two_shared, ag.two_center, ag.two_head, ag.two_shared_grad) == (
            mode, shared, net.two_center, net.two_head, net.two_shared_grad)
        assert TM.shapes_of(ag) == TM.shapes_of(net) and not ag.model.training and ag._meta["two_nets_mode"] == mode
        for k in range(3):
            st = (rng.normal(size=(in_ch, 26, 26)) * (k > 0)).astype(np.float32)
            q = E.q_values(ag, st, mean, std, var)
            assert q.shape == (27,) and np.isfinite(q).all()
            assert np.abs(q - E.q_values(net, st, mean, std, var)).max() < 1e-5
            a = E.pick_action(ag, st, mean, std, var)
            assert a == int(np.argmax(q))
            if net.two_center == "max":      # оценка выбирает последовательно: оптимизатор, затем параметры
                xs = torch.as_tensor((st[None] - mean) / std).float()
                with torch.no_grad():
                    assert a == int(ag.hier_action(xs)[0]) == int(net.hier_action(xs)[0]), (mode, shared)
        v = O.verify_ckpt(path, net)
        assert v["ok"] and v["two_nets"] is True and v["max_abs_dq"] < 1e-5, v
        assert (v["two_nets_mode"], v["two_nets_shared"], v["two_nets_center"]) == (mode, shared, net.two_center), v
        # та же сеть, записанная в meta другим составом или с другим энкодером, проверку не проходит
        same = dict(in_ch=in_ch, encoder=net.enc_kind, two_head=net.two_head, two_center=net.two_center)
        for other in (mk(O.TWO_NETS_MODE[(O.TWO_NETS_MODE.index(mode) + 1) % len(O.TWO_NETS_MODE)], shared, **same), mk(mode, not shared, **same)):
            other.set_train(False)
            assert not O.verify_ckpt(path, other)["ok"]
    # чекпоинт до --two-nets-mode (в meta нет состава и общего энкодера) читается как additive со своим энкодером
    ck = torch.load(paths[9, "additive", False], weights_only=False)
    ck["meta"] = {k: v for k, v in ck["meta"].items() if k not in ("two_nets_mode", "two_nets_shared")}
    old = os.path.join(tmp, "m_old.pt")
    torch.save(ck, old)
    assert O.ckpt_arch(ck)["two_mode"] == "additive" and O.ckpt_arch(ck)["two_shared"] is False
    ag, mean, std, var = E.load_agent(old)
    ag2 = E.load_agent(paths[9, "additive", False])[0]
    assert (ag.two_mode, ag.two_shared, ag.two_shared_grad) == ("additive", False, "both")
    st = rng.normal(size=(9, 26, 26)).astype(np.float32)
    assert np.array_equal(E.q_values(ag, st, mean, std, var), E.q_values(ag2, st, mean, std, var))
    # новые чекпоинты без своих полей в meta в прежнюю сеть не ложатся — ошибка, а не тихая подмена
    for key, what in (((9, "joint", False), "joint без состава в meta"), ((9, "auto", False), "auto без состава в meta"),
                      ((9, "additive", True), "общий энкодер без поля в meta")):
        ck = torch.load(paths[key], weights_only=False)
        ck["meta"] = {k: v for k, v in ck["meta"].items() if not k.startswith(("two_nets_mode", "two_nets_shared"))}
        bad = os.path.join(tmp, "m_bad.pt")
        torch.save(ck, bad)
        must_fail(lambda: E.load_agent(bad), RuntimeError, what)
    # комитет: одна сеть, прежние две сети и два новых состава — все отдают по 27 значений
    d9 = TM.make_data(in_ch=9)
    one, st1 = TM.fit("convnext_dqn", d9, d9["EP"] < 4, encoder="resnet")
    agents = [E.load_agent(TM.save_ckpt(one, st1, os.path.join(tmp, "m_one.pt"))), E.load_agent(old),
              E.load_agent(paths[9, "joint", True]), E.load_agent(paths[9, "auto", False])]
    for how in ("mean", "vote"):
        assert 0 <= E.ensemble_action(agents, np.zeros((9, 26, 26), np.float32), np.ones(27, bool), how=how) < 27
    print("чекпоинты составов и общего энкодера: load_agent, q_values, verify_ckpt; старый читается как additive: OK")


def test_ckpt(tmp):
    import online_eval_env as E
    torch.set_default_tensor_type(torch.FloatTensor)
    rng = np.random.default_rng(4)
    for in_ch, kw in ((4, dict(encoder="resnet")), (9, dict(encoder="resnet", two_nets_center="mean")),
                      (9, dict(encoder="auto", two_nets_head="dueling")), (4, dict(encoder="vit"))):
        data = TM.make_data(in_ch=in_ch)
        mask = data["EP"] < 4
        net, stats = TM.fit("convnext_dqn", data, mask, value_bound=True, two_nets=True, **kw)
        path = TM.save_ckpt(net, stats, os.path.join(tmp, f"two_{in_ch}_{net.enc_kind}_{net.two_center}_{net.two_head}.pt"),
                            meta=two_meta(net))
        arch = O.ckpt_arch(torch.load(path, weights_only=False))
        # meta прежнего вида (two_meta): состав additive, у сети параметров свой энкодер
        assert arch == dict(n_quantiles=32, in_ch=in_ch, encoder=net.enc_kind, ens=1, two_nets=True,
                            two_center=net.two_center, two_head=net.two_head, two_mode="additive",
                            two_shared=False), arch
        ag, mean, std, var = E.load_agent(path)
        assert var == "convnext_dqn" and ag.two_nets and ag.enc_kind == net.enc_kind and not ag.model.training
        assert ag.two_mode == "additive" and not ag.two_shared
        assert (ag.two_center, ag.two_head) == (net.two_center, net.two_head)
        assert TM.shapes_of(ag) == TM.shapes_of(net)
        for k in range(3):
            st = (rng.normal(size=(in_ch, 26, 26)) * (k > 0)).astype(np.float32)
            q = E.q_values(ag, st, mean, std, var)
            assert q.shape == (27,) and np.isfinite(q).all()
            assert np.abs(q - E.q_values(net, st, mean, std, var)).max() < 1e-5
            a = E.pick_action(ag, st, mean, std, var)
            assert a == int(np.argmax(q))
            if net.two_center == "max":      # оценка выбирает как агент статьи: оптимизатор, затем параметры
                x = torch.as_tensor((st[None] - mean) / std).float()
                with torch.no_grad():
                    assert a == int(ag.hier_action(x)[0]) == int(net.hier_action(x)[0])
        v = O.verify_ckpt(path, net)
        assert v["ok"] and v["two_nets"] is True and v["two_nets_center"] == net.two_center and v["max_abs_dq"] < 1e-5, v
        # сеть с другим центрированием этой проверкой не проходит
        other = O.QNet("convnext_dqn", CPU, in_ch=in_ch, encoder=net.enc_kind, two_head=net.two_head, two_nets=True,
                       two_center=("mean" if net.two_center == "max" else "max"))
        other.model.load_state_dict(net.model.state_dict()); other.set_train(False)
        assert not O.verify_ckpt(path, other)["ok"]
    # meta без поля two_nets: веса двух сетей в одну не ложатся — ошибка, а не тихая подмена
    ck = torch.load(path, weights_only=False)
    ck["meta"] = dict(encoder=net.enc_kind, ens=1)
    bad = os.path.join(tmp, "two_nometa.pt")
    torch.save(ck, bad)
    must_fail(lambda: E.load_agent(bad), RuntimeError, "чекпоинт двух сетей без two_nets в meta")
    # комитет из агента с одной сетью и агента с двумя: оба отдают по 27 значений
    one, st1 = TM.fit("convnext_dqn", TM.make_data(), mask, encoder="resnet")
    p1 = TM.save_ckpt(one, st1, os.path.join(tmp, "one.pt"))
    agents = [E.load_agent(p1), E.load_agent(os.path.join(tmp, "two_4_resnet_max_linear.pt"))]
    assert 0 <= E.ensemble_action(agents, np.zeros((4, 26, 26), np.float32), np.ones(27, bool), how="mean") < 27
    print("чекпоинт: load_agent и q_values дают те же 27 значений, pick_action — иерархический выбор: OK")


def test_defaults_and_refusals():
    data = TM.make_data()
    mask = data["EP"] < 4
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        TM.test_defaults(data, mask)        # числа и формы, записанные до правок (REF)
    assert "OK" in buf.getvalue()
    if "пропущена" in buf.getvalue():
        print("  " + buf.getvalue().strip().splitlines()[0].strip())
    # выключенный флаг — та же сеть: те же ключи, те же веса при том же сиде, тот же разбор чекпоинта
    for variant, kw in (("convnext_dqn", dict(in_ch=9, encoder="resnet")), ("cnx_cql", dict())):
        torch.manual_seed(1)
        a = O.QNet(variant, CPU, **kw)
        torch.manual_seed(1)
        b = O.QNet(variant, CPU, two_nets=False, two_center="max", two_head="linear", **kw)
        sa, sb = a.model.state_dict(), b.model.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
        assert not a.two_nets and a._enc_idx == (0,) and {k.split(".")[0] for k in sa} == {"0", "1"}
        ck = dict(variant=variant, state_dict=sa, meta=dict(encoder=a.enc_kind, ens=1))
        assert O.ckpt_arch(ck) == dict(n_quantiles=1, in_ch=kw.get("in_ch", 4), encoder=a.enc_kind, ens=1)
        # первая сеть двух — ровно прежний энкодер при том же сиде (порядок создания модулей прежний)
        torch.manual_seed(1)
        t = O.QNet(variant, CPU, two_nets=True, **kw)
        assert all(torch.equal(v, t.model[0].state_dict()[k[2:]]) for k, v in sa.items() if k.startswith("0."))
    net, _ = TM.fit("convnext_dqn", data, mask, encoder="resnet", two_nets=False)
    assert not net.two_nets and "n_params_opt_net" not in net.extra
    for what, bad in (("квантильный вариант", lambda: O.QNet("cnx_cql_qr", CPU, **TWO)),
                      ("cnx_factored", lambda: O.QNet("cnx_factored", CPU, **TWO)),
                      ("вариант без ConvNeXt", lambda: O.QNet("cnn_dqn", CPU, **TWO)),
                      ("ансамбль голов", lambda: O.QNet("convnext_dqn", CPU, ens=2, **TWO)),
                      ("HL-Gauss", lambda: O.QNet("convnext_dqn", CPU, bins=51, **TWO)),
                      ("центрирование", lambda: O.QNet("convnext_dqn", CPU, two_center="median", **TWO)),
                      ("головы", lambda: O.QNet("convnext_dqn", CPU, two_head="mlp", **TWO)),
                      ("--aux-next", lambda: TM.fit("convnext_dqn", data, mask, two_nets=True, aux_next=0.5))):
        must_fail(bad, ValueError, what)
    print("значения по умолчанию прежние; неподдержанные сочетания отвергаются: OK")


def run_main(argv):
    """main() с подменёнными аргументами: (stdout, stderr, код выхода или None)."""
    old = sys.argv
    out, err, code = io.StringIO(), io.StringIO(), None
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                O.main()
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = old
    return out.getvalue(), err.getvalue(), code


def test_main(tmp):
    """Полный прогон main() по лучшему рецепту (контекст, граница ценности, веса энкодера, модельные
    цели, учитель на скалярах) с --two-nets: строка результата, meta чекпоинта, проверка загрузки."""
    import killswitch
    import online_eval_env as E
    rng = np.random.default_rng(0)
    e0 = float(O.pde_meta()[TS.PDE]["init_err"])
    bufs = TS.make_buffers()
    root = os.path.join(tmp, "two_bufs", TS.SUB)
    os.makedirs(root)
    for k, dump in enumerate(bufs[TS.SUB]):
        torch.save(dump, os.path.join(root, f"f{k}.pt"))
    teacher, _ = TS.train_teacher(bufs, tmp)
    enc = os.path.join(tmp, "enc_main.pt")
    torch.save(O.make_encoder("resnet", 4).state_dict(), enc)
    mtag = f"ttwo{os.getpid()}"
    ckpt = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000",
            "--ctx-kmax", "70", "--save-model", "--scalar-ctx", "--encoder", "resnet", "--encoder-init", enc,
            "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5", "--teacher-from", teacher, "--teacher-w", "0.5",
            "--seeds", "1", "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu", "--model-tag", mtag,
            "--two-nets"]

    def fake_build(names, subdirs, table, members, work, quick, log):
        tab, ens = TM.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3)

    old = (killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        out, _, code = run_main(argv)
        assert code is None, (code, out[-1500:])
        rows = [json.loads(ln) for ln in out.splitlines() if ln.startswith('{"variant"')]
        assert len(rows) == 1, out[-2000:]
        r = rows[0]
        want = dict(two_nets=True, two_nets_center="max", two_nets_head="linear", two_nets_mode="additive",
                    two_nets_shared=False, n_out_opt_net=3, n_out_par_net=18, encoder="resnet", ens=1,
                    scalar_ctx=True, value_bound=True, mb_aux=0.5, encoder_init="enc_main.pt", teacher_w=0.5,
                    teacher_from=os.path.basename(teacher))
        assert {k: r[k] for k in want} == want, {k: r.get(k) for k in want}
        assert r["n_params"] == r["n_params_opt_net"] + r["n_params_par_net"] and r["n_params_par_net"] > r["n_params_opt_net"]
        assert np.isfinite(r["mb_loss"]) and np.isfinite(r["td_error"]) and np.isfinite(r["fqe_value_all"])
        assert np.isfinite(r["teacher_loss"]) and r["teacher_train_rows"] > 0 and 0.0 <= r["teacher_agree"] <= 1.0
        assert "две сети: оптимизатор" in out and "проверка загрузки через online_eval_env" in out
        assert "'ok': True" in out and "'two_nets': True" in out, out[-1500:]
        ag, mean, std, var = E.load_agent(ckpt)
        meta = ag._meta
        assert meta["two_nets"] is True and meta["two_nets_center"] == "max" and meta["two_nets_head"] == "linear"
        assert meta["two_nets_mode"] == "additive" and meta["two_nets_shared"] is False and "two_nets_shared_grad" not in r
        assert meta["in_ch"] == 9 and meta["encoder"] == "resnet" and meta["episode_budget"] == 7000
        assert meta["teacher"]["rows"] > 0 and meta["mb"]["rows"] == r["mb_rows"]
        assert ag.two_nets and ag.two_center == "max" and ag.n_params() == r["n_params"]
        st = rng.normal(size=(9, 26, 26)).astype(np.float32)
        q = E.q_values(ag, st, mean, std, var)
        assert q.shape == (27,) and np.isfinite(q).all()
        x = torch.as_tensor((st[None] - mean) / std).float()
        with torch.no_grad():
            assert E.pick_action(ag, st, mean, std, var) == int(ag.hier_action(x)[0])
        # одна сеть того же рецепта — для сравнения числа параметров (сохранение не нужно)
        out1, _, _ = run_main([a for a in argv if a not in ("--two-nets", "--save-model")])
        r1 = [json.loads(ln) for ln in out1.splitlines() if ln.startswith('{"variant"')][0]
        assert "two_nets" not in r1 and "n_params_opt_net" not in r1 and r1["n_params"] < r["n_params"] < 2 * r1["n_params"]
        print(f"  параметров: одна сеть {r1['n_params']}, две сети {r['n_params']} "
              f"(оптимизатор {r['n_params_opt_net']}, параметры {r['n_params_par_net']})")
        # новые составы тем же рецептом: auto с общим энкодером на двух сидах, joint со своим;
        # joint с общим энкодером на собранном буфере (--target-buffer вместо --data-dir, без
        # модельных целей: их таблице нужны буферы остальных задач)
        seeds2 = [("1,2" if i and argv[i - 1] == "--seeds" else a) for i, a in enumerate(argv)]
        drop = ("--data-dir", "--mb-aux", "--mb-h", "--mb-lam")
        tb = [a for i, a in enumerate(argv) if a not in drop and not (i and argv[i - 1] in drop)]
        tb += ["--subdir", TS.SUB, "--target-buffer", root]
        runs = (("auto", True, "stop", seeds2 + ["--two-nets-mode", "auto", "--two-nets-shared", "--two-nets-shared-grad", "stop"]),
                ("joint", False, None, argv + ["--two-nets-mode", "joint", "--two-nets-center", "mean", "--two-nets-head", "dueling"]),
                ("joint", True, "both", tb + ["--two-nets-mode", "joint", "--two-nets-shared"]))
        for mode, shared, grad, av in runs:
            out2, _, code = run_main(av)
            assert code is None, (mode, code, out2[-1500:])
            rows2 = [json.loads(ln) for ln in out2.splitlines() if ln.startswith('{"variant"')]
            assert [x["seed"] for x in rows2] == ([1, 2] if "1,2" in av else [1]), out2[-1500:]
            assert out2.count("'ok': True") == len(rows2) and f"состав {mode}" in out2, out2[-1500:]
            for x in rows2:
                assert (x["two_nets"], x["two_nets_mode"], x["two_nets_shared"], x.get("two_nets_shared_grad")) == (
                    True, mode, shared, grad), x
                assert x["n_params"] == x["n_params_opt_net"] + x["n_params_par_net"] and np.isfinite(x["td_error"])
                assert (x["n_out_opt_net"], x["n_out_par_net"]) == (3, N_OUT_B[mode]) and np.isfinite(x["teacher_loss"])
                assert (x["n_params_par_net"] < x["n_params_opt_net"]) == shared
                assert ("tb_files" in x) == ("--target-buffer" in av) and ("mb_loss" in x) == ("--mb-aux" in av)
                ag, mean, std, var = E.load_agent(ckpt.replace("_seed1.pt", f"_seed{x['seed']}.pt"))
                m2 = ag._meta
                assert (m2["two_nets_mode"], m2["two_nets_shared"], m2.get("two_nets_shared_grad")) == (mode, shared, grad)
                assert (ag.two_mode, ag.two_shared) == (mode, shared) and ag.n_params() == x["n_params"]
                q = E.q_values(ag, st, mean, std, var)
                assert q.shape == (27,) and np.isfinite(q).all()
                if ag.two_center == "max":
                    with torch.no_grad():
                        assert int(np.argmax(q)) == int(ag.hier_action(torch.as_tensor((st[None] - mean) / std).float())[0])
            print(f"  {mode}{', общий энкодер' if shared else ''}: параметров {rows2[0]['n_params']} "
                  f"(оптимизатор {rows2[0]['n_params_opt_net']}, параметры {rows2[0]['n_params_par_net']}), "
                  f"выходов 3 + {rows2[0]['n_out_par_net']}, сидов {len(rows2)}")
        assert rows2[0]["tb_used_transitions"] > 0 and "исходный буфер" in out2
        # неподдержанные сочетания: ошибка разбора флагов до загрузки данных
        base = argv[:4] + ["/nonexistent"] + argv[5:]
        cases = (("--ens > 1", base + ["--ens", "2"]), ("--hlg", base + ["--hlg"]),
                 ("--aux-next", base + ["--aux-next", "0.5"]),
                 ("квантильный вариант", [("cnx_cql_qr" if a == "convnext_dqn" else a) for a in base]),
                 ("cnx_bbf", [("cnx_bbf" if a == "convnext_dqn" else a) for a in base[:-1]
                              if a not in ("--encoder-init", enc)] + ["--two-nets"]),
                 ("центрирование без --two-nets", base[:-1] + ["--two-nets-center", "mean"]),
                 ("состав без --two-nets", base[:-1] + ["--two-nets-mode", "joint"]),
                 ("общий энкодер без --two-nets", base[:-1] + ["--two-nets-shared"]),
                 ("--two-nets-shared-grad без --two-nets-shared", base + ["--two-nets-shared-grad", "stop"]),
                 ("auto с --hlg", base + ["--two-nets-mode", "auto", "--hlg"]))
        for what, av in cases:
            _, err, code = run_main(av)
            assert code == 2 and "--two-nets" in err.splitlines()[-1], (what, code, err[-300:])
    finally:
        killswitch.killed, MB.build_model, O.GAMMA = old
        for f in (ckpt, ckpt.replace("_seed1.pt", "_seed2.pt")):
            if os.path.exists(f):
                os.remove(f)
    print("main() с --two-nets, составами joint и auto, общим энкодером, двумя сидами и --target-buffer: "
          "строка результата, meta чекпоинта, проверка загрузки, отказы: OK")


# проверки по именам: без аргументов идут все, с аргументами — только названные
TESTS = dict(compose=test_compose, gradients=test_gradients, defaults=test_defaults_and_refusals,
             modes=test_modes, fit=test_fit, training=test_training, modes_training=test_modes_training,
             ckpt=test_ckpt, modes_ckpt=test_modes_ckpt, main=test_main)


def main():
    import inspect
    names = sys.argv[1:] or list(TESTS)
    if set(names) - set(TESTS):
        raise SystemExit(f"нет проверок {sorted(set(names) - set(TESTS))}: выбор {', '.join(TESTS)}")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            t1 = time.time()
            fn = TESTS[name]
            fn(tmp) if "tmp" in inspect.signature(fn).parameters else fn()
            print(f"    ({time.time() - t1:.1f} с)")
    print(f"test_two_nets: {'всё' if not sys.argv[1:] else ', '.join(names)} прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
