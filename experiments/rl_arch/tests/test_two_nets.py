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

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_two_nets.py
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
    """Поля meta, которые пишет offline_rl.main (--save-model) для двух сетей."""
    return dict(encoder=net.enc_kind, ens=net.ens, two_nets=True, two_nets_center=net.two_center,
                two_nets_head=net.two_head)


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
        assert arch == dict(n_quantiles=32, in_ch=in_ch, encoder=net.enc_kind, ens=1, two_nets=True,
                            two_center=net.two_center, two_head=net.two_head), arch
        ag, mean, std, var = E.load_agent(path)
        assert var == "convnext_dqn" and ag.two_nets and ag.enc_kind == net.enc_kind and not ag.model.training
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
        want = dict(two_nets=True, two_nets_center="max", two_nets_head="linear", encoder="resnet", ens=1,
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
        # неподдержанные сочетания: ошибка разбора флагов до загрузки данных
        base = argv[:4] + ["/nonexistent"] + argv[5:]
        cases = (("--ens > 1", base + ["--ens", "2"]), ("--hlg", base + ["--hlg"]),
                 ("--aux-next", base + ["--aux-next", "0.5"]),
                 ("квантильный вариант", [("cnx_cql_qr" if a == "convnext_dqn" else a) for a in base]),
                 ("cnx_bbf", [("cnx_bbf" if a == "convnext_dqn" else a) for a in base[:-1]
                              if a not in ("--encoder-init", enc)] + ["--two-nets"]),
                 ("центрирование без --two-nets", base[:-1] + ["--two-nets-center", "mean"]))
        for what, av in cases:
            _, err, code = run_main(av)
            assert code == 2 and "--two-nets" in err.splitlines()[-1], (what, code, err[-300:])
    finally:
        killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("main() с --two-nets: строка результата, meta чекпоинта, проверка загрузки, отказы: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    test_compose(); lap()
    test_gradients(); lap()
    test_defaults_and_refusals(); lap()
    with tempfile.TemporaryDirectory() as tmp:
        test_training(tmp); lap()
        test_ckpt(tmp); lap()
        test_main(tmp); lap()
    print(f"test_two_nets: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
