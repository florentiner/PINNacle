#!/usr/bin/env python
"""
Проверки дообучения энкодера с малым числом параметров (offline_rl.py --ft-mode bn | lora | lastblock,
--lora-rank, --lora-alpha; волна 52) без GPU и без сети, на крошечных синтетических данных
(test_map_agent.make_data: шесть цепочек по три шага, несколько шагов градиента на проверку):

  * значения по умолчанию — прежний путь: --ft-mode full (явно или без поля) даёт те же ключи, те же
    веса и те же числа после обучения, разбор чекпоинта прежний; числа test_map_agent.REF;
  * каждый режим учит только свои параметры: остальные параметры энкодера бит в бит равны старту
    (файл --encoder-init), обучаемые и головы сдвигаются; статистики BatchNorm копятся (bn, lora) или
    стоят в замороженной части (lastblock, и при --wa тоже); оптимизатору — только обучаемые;
  * lora: при создании адаптеров выход сети совпадает с предобученной сетью бит в бит (up нулевой),
    адаптеры есть и в целевой сети; свежие каналы первой свёртки учатся целиком, её адаптер стоит;
  * две сети (--two-nets, со своими энкодерами и с общим): режим действует на каждый энкодер;
  * чекпоинт: meta с ft_mode, lora_rank, lora_alpha -> ckpt_arch -> online_eval_env.load_agent
    строит ту же сеть, q_values те же; verify_ckpt проходит и ловит чужой ранг; чекпоинт lora без
    полей в meta не читается молча; полный прогон main() с --ft-mode;
  * отказы: --freeze-encoder, энкодеры, у которых режим не определён, cnx_bbf, --lora-rank без lora.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_peft.py
    ... test_peft.py modes ckpt      # только названные проверки (имена — в TESTS)
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

import offline_rl as O  # noqa: E402
import mb_targets as MB  # noqa: E402
import test_map_agent as TM  # noqa: E402  — те же синтетические данные и подставной суррогат
import test_shield_teacher as TS  # noqa: E402  — буферы и настоящий чекпоинт учителя на скалярах
import test_two_nets as TT  # noqa: E402  — run_main, must_fail

torch.set_num_threads(1)
CPU = torch.device("cpu")
MODES = ("bn", "lora", "lastblock")
# обучаемые параметры энкодера SmallResNet при входе 9 каналов (4 карты + 5 каналов контекста) и
# --encoder-init с 4 каналами: свежая первая свёртка (32 x 9 x 3 x 3 = 2592) учится в каждом режиме
N_ENC = 416096                              # параметров у энкодера SmallResNet с 9 входными каналами
N_TRAIN = dict(bn=1600 + 2592, lora=1600 + 22724 - 452 + 2592, lastblock=263168 + 2592)
N_LORA = 22724                              # адаптеры ранга 4 у SmallResNet (вход 9 каналов)


def enc_file(tmp, with_norm=True):
    """Файл весов в формате зонда CV (cv_probe.py --save-encoder): 4 канала, нормировка карт."""
    path = os.path.join(tmp, "enc_peft.pt" if with_norm else "enc_peft_bare.pt")
    if not os.path.exists(path):
        torch.manual_seed(11)
        sd = O.make_encoder("resnet", 4).state_dict()
        for k, v in sd.items():             # ненулевые статистики, как у настоящего предобучения
            if k.endswith("running_mean"):
                v.normal_(0, 0.1)
            elif k.endswith("running_var"):
                v.uniform_(0.5, 1.5)
        torch.save(dict(state_dict=sd, encoder="resnet", ssl="mae",
                        mean=np.full((1, 4, 1, 1), 0.3, np.float32), std=np.full((1, 4, 1, 1), 2.0, np.float32))
                   if with_norm else sd, path)
    return path


def start_net(variant, in_ch, path, **kw):
    """Сеть в том виде, в каком train_variant(seed=3) начинает обучение (та же инициализация)."""
    torch.manual_seed(3)
    net = O.QNet(variant, CPU, in_ch=in_ch, encoder="resnet", **kw)
    O.load_encoder_init(net, path)
    return net


def dense_data(in_ch=9):
    d = TM.make_data(in_ch=in_ch)
    mask = d["EP"] < 4
    rng = np.random.default_rng(3)
    n = len(d["A"])
    return dict(d, QMB=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), MBW=mask.copy(),
                QTE=rng.normal(0.2, 0.2, (n, 27)).astype(np.float32), TEV=np.ones(n, bool)), mask


def bn_layers(enc):
    return {mn: m for mn, m in enc.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}


# --------------------------------------------------------------------------------------------

def test_defaults(tmp):
    data = TM.make_data()
    mask = data["EP"] < 4
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        TM.test_defaults(data, mask)        # числа и формы, записанные до правок (REF)
    assert "OK" in buf.getvalue()
    if "пропущена" in buf.getvalue():
        print("  " + buf.getvalue().strip().splitlines()[0].strip())
    # --ft-mode full явно и без поля: та же сеть, те же веса после обучения, тот же оптимизатор
    d9, m9 = dense_data()
    path = enc_file(tmp)
    for kw in (dict(), dict(two_nets=True), dict(wa="ema", wa_start=0.0, wa_decay=0.9)):
        kw = dict(encoder="resnet", encoder_init=path, mb_aux=0.5, teacher_w=0.5, epochs=2, **kw)
        a, sa = TM.fit("convnext_dqn", d9, m9, **kw)
        b, sb = TM.fit("convnext_dqn", d9, m9, ft_mode="full", lora_rank=4, lora_alpha=8.0, **kw)
        for x, y in ((a.model, b.model), (a.target, b.target)):
            sx, sy = x.state_dict(), y.state_dict()
            assert list(sx) == list(sy) and all(torch.equal(sx[k], sy[k]) for k in sx), kw
        assert a.extra == b.extra and not any(k.startswith("ft_") for k in a.extra)
        assert np.array_equal(sa["mean"], sb["mean"]) and a.ft_mode == "full" and a._ft_eval == []
        assert all(p.requires_grad for p in a.model.parameters())
    for variant, kw in (("convnext_dqn", dict(in_ch=9, encoder="resnet")), ("cnx_cql", dict()),
                        ("convnext_dqn", dict(in_ch=9, encoder="resnet", two_nets=True))):
        torch.manual_seed(1)
        a = O.QNet(variant, CPU, **kw)
        torch.manual_seed(1)
        b = O.QNet(variant, CPU, ft_mode="full", **kw)
        sa, sb = a.model.state_dict(), b.model.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
        assert a.apply_ft() is None and all(p.requires_grad for p in a.model.parameters())
        ck = dict(variant=variant, state_dict=sa, meta=dict(encoder=a.enc_kind, ens=1, ft_mode="full"))
        assert "ft_mode" not in O.ckpt_arch(ck)
    print("значения по умолчанию прежние (--ft-mode full = без флага, бит в бит): OK")


def test_modes(tmp):
    """Каждый режим учит только свои параметры энкодера; головы учатся; статистики BatchNorm."""
    d9, m9 = dense_data()
    path = enc_file(tmp)
    f = torch.load(path, weights_only=False)
    for mode in MODES:
        kw = dict(encoder="resnet", encoder_init=path, mb_aux=0.5, teacher_w=0.5, value_bound=True, epochs=4,
                  ft_mode=mode)
        net, stats = TM.fit("convnext_dqn", d9, m9, **kw)
        ref = start_net("convnext_dqn", 9, path, ft_mode=mode)
        enc, enc0 = net.model[0], ref.model[0]
        train = O.ft_trainable_names(enc, mode, "resnet") | {"stem.0.weight"}
        if mode == "lora":
            train -= {"stem.0.lora_down.weight", "stem.0.lora_up.weight"}
        p0 = dict(enc0.named_parameters())
        got = sorted(n for n, p in enc.named_parameters() if p.requires_grad)
        assert got == sorted(train), (mode, set(got) ^ train)
        for n, p in enc.named_parameters():
            if n in train:
                assert not torch.equal(p, p0[n]), f"{mode}: {n} должен учиться"
            else:
                assert torch.equal(p, p0[n]), f"{mode}: {n} должен стоять"
        # стоящие параметры — ровно из файла весов (кроме свежих каналов и адаптеров)
        sd = enc.state_dict()
        frozen = [k for k in p0 if k not in train and not O.is_lora_key(k)]
        assert frozen and all(torch.equal(sd[k], f["state_dict"][k]) for k in frozen), mode
        if mode == "lora":                  # адаптер свежей первой свёртки стоит нулевым
            assert not sd["stem.0.lora_up.weight"].any() and sd["body.0.c1.lora_up.weight"].any()
        assert not torch.equal(sd["stem.0.weight"][:, :4], f["state_dict"]["stem.0.weight"])
        assert net.extra["ft_enc_trainable"] == N_TRAIN[mode] and net.extra["ft_enc_total"] == N_ENC + N_LORA * (mode == "lora")
        assert net.extra["ft_head_trainable"] == sum(p.numel() for p in net.model[1].parameters())
        assert all(not torch.equal(p, q) for p, q in zip(net.model[1].parameters(), ref.model[1].parameters()))
        # оптимизатор видел только обучаемые; у целевой сети обучаемых нет, адаптеры в ней есть
        assert list(net.target.state_dict()) == list(net.model.state_dict())
        assert not any(p.requires_grad for p in net.target.parameters())
        # статистики BatchNorm: у bn и lora копятся во всех слоях, у lastblock — только в последнем блоке
        bns, bns0 = bn_layers(enc), bn_layers(enc0)
        for mn, m in bns.items():
            moved = not torch.equal(m.running_mean, bns0[mn].running_mean)
            assert moved == (mode != "lastblock" or mn.startswith("body.3.")), (mode, mn, moved)
        assert [mn for mn, m in bns.items() if any(m is x for x in net._ft_eval)] == (
            [mn for mn in bns if not mn.startswith("body.3.")] if mode == "lastblock" else [])
        net.set_train(True)
        assert enc.training == True and all(not m.training for m in net._ft_eval)  # noqa: E712
        net.set_train(False)
        # нормировка карт: у lastblock — из файла (замороженная часть предобучена в ней), иначе своя
        from_file = np.allclose(stats["mean"][:, :4], f["mean"]) and np.allclose(stats["std"][:, :4], f["std"])
        assert from_file == (mode == "lastblock") and net.extra.get("norm_from_encoder_init", False) == from_file
        print(f"  {mode}: учатся {net.extra['ft_enc_trainable']} из {net.extra['ft_enc_total']} параметров "
              f"энкодера, головы {net.extra['ft_head_trainable']}")
    # --wa с lastblock: пересчёт BatchNorm не трогает замороженную часть
    net, _ = TM.fit("convnext_dqn", d9, m9, encoder="resnet", encoder_init=path, mb_aux=0.5, epochs=3,
                    ft_mode="lastblock", wa="ema", wa_start=0.0, wa_decay=0.9)
    ref = start_net("convnext_dqn", 9, path, ft_mode="lastblock")
    b1, b0 = bn_layers(net.model[0]), bn_layers(ref.model[0])
    n_last = sum(mn.startswith("body.3.") for mn in b1)
    assert net.extra["wa_bn_layers"] == n_last == 3, net.extra
    for mn, m in b1.items():
        # среднее весов (EMA) постоянной величины отличается от неё на округление — сравнение с допуском
        same = (torch.allclose(m.running_mean, b0[mn].running_mean, rtol=1e-5, atol=1e-7)
                and torch.allclose(m.running_var, b0[mn].running_var, rtol=1e-5, atol=1e-7))
        assert same == (not mn.startswith("body.3.")), mn
    # без --encoder-init: режим тот же, энкодер случайный, свежих каналов нет
    net, _ = TM.fit("convnext_dqn", d9, m9, encoder="resnet", mb_aux=0.5, ft_mode="bn")
    assert net.extra["ft_enc_trainable"] == 1600
    print("режимы учат только свои параметры, статистики BatchNorm как задумано: OK")


def test_lora_init(tmp):
    """Адаптеры не меняют функцию при создании; обёртка сохраняет имена прежних ключей."""
    path = enc_file(tmp)
    x = torch.randn(5, 9, 26, 26)
    for kw in (dict(), dict(two_nets=True), dict(two_nets=True, two_shared=True)):
        full = start_net("convnext_dqn", 9, path, **kw)
        lo = start_net("convnext_dqn", 9, path, ft_mode="lora", lora_rank=3, lora_alpha=6.0, **kw)
        # адаптеры создаются после всех модулей сети: при том же сиде прежние веса те же
        sd_full = full.model.state_dict()
        sd_lo = lo.model.state_dict()
        base = [k for k in sd_lo if not O.is_lora_key(k)]
        assert base == list(sd_full), "имена прежних ключей не меняются"
        assert all(torch.equal(sd_lo[k], sd_full[k]) for k in base)
        for on in (False, True):
            full.set_train(on); lo.set_train(on)
            with torch.no_grad():
                assert torch.equal(full.q_online(x), lo.q_online(x)), (kw, on)
                assert torch.equal(full.q_target(x), lo.q_target(x)), (kw, on)
        ada = [m for i in lo._enc_idx for m in lo.model[i].modules() if isinstance(m, (O.LoRAConv2d, O.LoRALinear))]
        n_conv_lin = sum(type(m) in (torch.nn.Conv2d, torch.nn.Linear) for i in full._enc_idx
                         for m in full.model[i].modules())
        assert len(ada) == n_conv_lin == 12 * len(lo._enc_idx), (len(ada), n_conv_lin)
        assert all(m.lora_scale == 2.0 and m.lora_down.weight.shape[0] == 3 and not m.lora_up.weight.any() for m in ada)
        tgt_ada = [m for i in lo._enc_idx for m in lo.target[i].modules() if isinstance(m, (O.LoRAConv2d, O.LoRALinear))]
        assert len(tgt_ada) == len(ada), "адаптеры есть и в целевой сети"
        # головы и энкодер вне адаптеров не тронуты; второй энкодер (две сети) — тоже
        assert not any(isinstance(m, (O.LoRAConv2d, O.LoRALinear)) for m in lo.model[1].modules())
    # обёртка свёртки с отступом/шагом и линейного слоя: формы и нулевая добавка
    conv = torch.nn.Conv2d(6, 8, 3, stride=2, padding=1)
    lin = torch.nn.Linear(7, 5)
    x6, x7 = torch.randn(2, 6, 9, 9), torch.randn(3, 7)
    y0, z0 = conv(x6), lin(x7)
    assert O.lora_wrap(torch.nn.Sequential(conv, lin), 2, 4.0) == 2
    assert torch.equal(conv(x6), y0) and torch.equal(lin(x7), z0) and conv.lora_down.stride == (2, 2)
    with torch.no_grad():
        conv.lora_up.weight.fill_(0.1)
        lin.lora_up.weight.fill_(0.1)
        dz = 2.0 * lin.lora_up(lin.lora_down(x7))
    assert not torch.equal(conv(x6), y0) and conv(x6).shape == y0.shape, "ненулевой up меняет выход"
    assert torch.allclose(lin(x7), z0 + dz)
    print("lora: при создании выход равен предобученной сети (в обоих режимах BatchNorm), адаптеры в целевой сети: OK")


def test_two_nets(tmp):
    d9, m9 = dense_data()
    path = enc_file(tmp)
    for shared in (False, True):
        for mode in MODES:
            kw = dict(encoder="resnet", encoder_init=path, mb_aux=0.5, teacher_w=0.5, epochs=3, ft_mode=mode,
                      two_nets=True, two_nets_shared=shared)
            net, _ = TM.fit("convnext_dqn", d9, m9, **kw)
            ref = start_net("convnext_dqn", 9, path, ft_mode=mode, two_nets=True, two_shared=shared)
            assert net._enc_idx == ((0,) if shared else (0, 2))
            for i in net._enc_idx:
                p0 = dict(ref.model[i].named_parameters())
                train = O.ft_trainable_names(net.model[i], mode, "resnet") | {"stem.0.weight"}
                if mode == "lora":
                    train -= {"stem.0.lora_down.weight", "stem.0.lora_up.weight"}
                for n, p in net.model[i].named_parameters():
                    assert p.requires_grad == (n in train) and torch.equal(p, p0[n]) == (n not in train), (mode, i, n)
            for i in (1, 3):                # головы обеих сетей учатся целиком
                assert all(p.requires_grad for p in net.model[i].parameters())
                assert TT.changed_mod(net.model[i], ref.model[i])
            assert net.extra["ft_enc_trainable"] == N_TRAIN[mode] * len(net._enc_idx), (mode, shared, net.extra)
            if mode == "lastblock":
                assert len(net._ft_eval) == 8 * len(net._enc_idx)   # stem.1, body.0-2 (b1, b2, у body.1 ещё sc.1)
            print(f"  две сети{' (общий энкодер)' if shared else ''}, {mode}: учатся {net.extra['ft_enc_trainable']} "
                  f"из {net.extra['ft_enc_total']} параметров энкодеров")
    print("две сети: режим действует на каждый энкодер: OK")


def test_ckpt(tmp):
    import online_eval_env as E
    torch.set_default_tensor_type(torch.FloatTensor)
    d9, m9 = dense_data()
    path = enc_file(tmp)
    rng = np.random.default_rng(4)
    for mode, kw in (("lora", dict()), ("lora", dict(two_nets=True)), ("bn", dict(two_nets=True)),
                     ("lastblock", dict()), ("lora", dict(lora_rank=2, lora_alpha=3.0))):
        net, stats = TM.fit("convnext_dqn", d9, m9, encoder="resnet", encoder_init=path, mb_aux=0.5, epochs=2,
                            ft_mode=mode, **kw)
        meta = dict(encoder=net.enc_kind, ens=1, ft_mode=mode)
        if mode == "lora":
            meta.update(lora_rank=net.lora_rank, lora_alpha=net.lora_alpha)
        if net.two_nets:
            meta.update(TT.mode_meta(net))
        p = TM.save_ckpt(net, stats, os.path.join(tmp, f"peft_{mode}_{len(net._enc_idx)}_{net.lora_rank}.pt"), meta=meta)
        ck = torch.load(p, weights_only=False)
        arch = O.ckpt_arch(ck)
        assert arch["ft_mode"] == mode and arch["in_ch"] == 9
        assert (arch.get("lora_rank"), arch.get("lora_alpha")) == (
            (net.lora_rank, net.lora_alpha) if mode == "lora" else (None, None)), arch
        assert any(O.is_lora_key(k) for k in ck["state_dict"]) == (mode == "lora")
        ag, mean, std, var = E.load_agent(p)
        assert ag.ft_mode == mode and TM.shapes_of(ag) == TM.shapes_of(net) and not ag.model.training
        for k in range(3):
            st = (rng.normal(size=(9, 26, 26)) * (k > 0)).astype(np.float32)
            q = E.q_values(ag, st, mean, std, var)
            assert q.shape == (27,) and np.isfinite(q).all()
            assert np.abs(q - E.q_values(net, st, mean, std, var)).max() < 1e-5
        v = O.verify_ckpt(p, net)
        assert v["ok"] and v["ft_mode"] == mode and v["max_abs_dq"] < 1e-5, v
        if mode == "lora":
            # сеть с другим рангом проверку не проходит; чекпоинт lora без полей в meta не читается молча
            other = O.QNet("convnext_dqn", CPU, in_ch=9, encoder="resnet", ft_mode="lora",
                           lora_rank=net.lora_rank, lora_alpha=net.lora_alpha * 2, two_nets=net.two_nets)
            other.model.load_state_dict(net.model.state_dict()); other.set_train(False)
            assert not O.verify_ckpt(p, other)["ok"]
            bad = os.path.join(tmp, "peft_nometa.pt")
            torch.save(dict(ck, meta={k: v for k, v in meta.items() if not k.startswith(("ft_", "lora_"))}), bad)
            TT.must_fail(lambda: E.load_agent(bad), RuntimeError, "чекпоинт lora без ft_mode в meta")
            torch.save(dict(ck, meta={k: v for k, v in meta.items() if not k.startswith("lora_")}), bad)
            TT.must_fail(lambda: E.load_agent(bad), ValueError, "чекпоинт lora без ранга в meta")
        print(f"  {mode}{' (две сети)' if net.two_nets else ''}: load_agent -> те же 27 значений, verify_ckpt ok")
    print("чекпоинт: ckpt_arch и load_agent строят ту же сеть с адаптерами, verify_ckpt проходит: OK")


def test_refusals(tmp):
    d9, m9 = dense_data()
    path = enc_file(tmp)
    cases = (("lastblock у convnext", lambda: O.QNet("convnext_dqn", CPU, ft_mode="lastblock")),
             ("lastblock у vit", lambda: O.QNet("convnext_dqn", CPU, encoder="vit", ft_mode="lastblock")),
             ("bn у convnext без BatchNorm", lambda: O.QNet("convnext_dqn", CPU, ft_mode="bn")),
             ("bn у cnn", lambda: O.QNet("cnn_dqn", CPU, ft_mode="bn")),
             ("lora у vit", lambda: O.QNet("convnext_dqn", CPU, encoder="vit", ft_mode="lora")),
             ("неизвестный режим", lambda: O.QNet("convnext_dqn", CPU, encoder="resnet", ft_mode="head")),
             ("ранг 0", lambda: O.QNet("convnext_dqn", CPU, encoder="resnet", ft_mode="lora", lora_rank=0)),
             ("--freeze-encoder", lambda: TM.fit("convnext_dqn", d9, m9, encoder="resnet", encoder_init=path,
                                                 freeze_encoder=True, ft_mode="bn")),
             ("cnx_bbf", lambda: TM.fit("cnx_bbf", d9, m9, encoder="resnet", ft_mode="lora")))
    for what, fn in cases:
        TT.must_fail(fn, ValueError, what)
    # lora определён и у энкодеров без BatchNorm (адаптеры к свёрткам и линейным слоям)
    for variant, enc in (("convnext_dqn", "auto"), ("cnn_dqn", "auto")):
        q = O.QNet(variant, CPU, encoder=enc, ft_mode="lora")
        n = q.apply_ft()
        assert 0 < n["trainable"] < n["total"] and not q.has_bn
    # флаги main(): ошибка разбора до загрузки данных
    base = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", "/nonexistent", "--chain-fix", "--cpu",
            "--encoder", "resnet", "--seeds", "1", "--epochs", "1"]
    for what, av, msg in (("--freeze-encoder", base + ["--ft-mode", "bn", "--freeze-encoder"], "--freeze-encoder"),
                          ("lastblock у convnext", [a for a in base if a not in ("--encoder", "resnet")]
                           + ["--ft-mode", "lastblock"], "lastblock"),
                          ("lora у vit", [("vit" if a == "resnet" else a) for a in base] + ["--ft-mode", "lora"], "vit"),
                          ("bn у cnn_dqn", [("cnn_dqn" if a == "convnext_dqn" else a) for a in base
                                            if a not in ("--encoder", "resnet")] + ["--ft-mode", "bn"], "BatchNorm"),
                          ("cnx_bbf", [("cnx_bbf" if a == "convnext_dqn" else a) for a in base] + ["--ft-mode", "bn"],
                           "cnx_bbf"),
                          ("--lora-rank без lora", base + ["--lora-rank", "8"], "--lora-rank"),
                          ("--lora-alpha с bn", base + ["--ft-mode", "bn", "--lora-alpha", "4"], "--lora-alpha"),
                          ("ранг 0", base + ["--ft-mode", "lora", "--lora-rank", "0"], "--lora-rank"),
                          ("режим", base + ["--ft-mode", "head"], "--ft-mode")):
        _, err, code = TT.run_main(av)
        assert code == 2 and msg in err.splitlines()[-1], (what, code, err[-400:])
    print("отказы: --freeze-encoder, режим не определён у энкодера, cnx_bbf, флаги LoRA без lora: OK")


def test_main(tmp):
    """Полный прогон main() по рецепту лидера (контекст, граница ценности, веса энкодера, модельные цели,
    учитель на скалярах) с --ft-mode: строка результата, meta чекпоинта, проверка загрузки."""
    import killswitch
    import online_eval_env as E
    e0 = float(O.pde_meta()[TS.PDE]["init_err"])
    bufs = TS.make_buffers()
    root = os.path.join(tmp, "peft_bufs", TS.SUB)
    os.makedirs(root, exist_ok=True)
    for k, dump in enumerate(bufs[TS.SUB]):
        torch.save(dump, os.path.join(root, f"f{k}.pt"))
    teacher, _ = TS.train_teacher(bufs, tmp)
    enc = enc_file(tmp)
    mtag = f"tpeft{os.getpid()}"
    ckpt = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000",
            "--ctx-kmax", "70", "--save-model", "--scalar-ctx", "--encoder", "resnet", "--encoder-init", enc,
            "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5", "--teacher-from", teacher, "--teacher-w", "0.5",
            "--seeds", "1", "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu", "--model-tag", mtag]

    def fake_build(names, subdirs, table, members, work, quick, log):
        tab, ens = TM.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3)

    old = (killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        out0, _, code = TT.run_main([a for a in argv if a != "--save-model"])
        assert code is None, out0[-1500:]
        r0 = [json.loads(ln) for ln in out0.splitlines() if ln.startswith('{"variant"')][0]
        assert not any(k.startswith(("ft_", "lora_")) for k in r0), "строка full прежняя"
        runs = (("lora", ["--ft-mode", "lora"], 1), ("lora", ["--ft-mode", "lora", "--lora-rank", "2", "--lora-alpha",
                                                              "2", "--two-nets"], 2),
                ("bn", ["--ft-mode", "bn", "--two-nets", "--two-nets-shared"], 1), ("lastblock", ["--ft-mode", "lastblock"], 1))
        for mode, extra, n_enc in runs:
            out, _, code = TT.run_main(argv + extra)
            assert code is None, (mode, code, out[-1500:])
            r = [json.loads(ln) for ln in out.splitlines() if ln.startswith('{"variant"')][0]
            rank = int(extra[extra.index("--lora-rank") + 1]) if "--lora-rank" in extra else 4
            assert r["ft_mode"] == mode and ("lora_rank" in r) == (mode == "lora")
            if mode == "lora":
                assert (r["lora_rank"], r["lora_alpha"]) == (rank, 2.0 if rank == 2 else 8.0), r
            if mode != "lora" or rank == 4:
                assert r["ft_enc_trainable"] == N_TRAIN[mode] * n_enc, (mode, r["ft_enc_trainable"])
            assert 0 < r["ft_enc_trainable"] < r["ft_enc_total"]
            if "--two-nets" not in extra:          # та же сеть, у lora — плюс адаптеры
                assert r["n_params"] == r0["n_params"] + (N_LORA if mode == "lora" else 0), (r["n_params"], r0["n_params"])
            assert np.isfinite(r["td_error"]) and np.isfinite(r["mb_loss"]) and np.isfinite(r["teacher_loss"])
            assert r.get("norm_from_encoder_init", False) == (mode == "lastblock")
            assert f"дообучение энкодера {mode}" in out and "'ok': True" in out and f"'ft_mode': '{mode}'" in out, out[-1500:]
            ag, mean, std, var = E.load_agent(ckpt)
            assert ag._meta["ft_mode"] == mode and ag.ft_mode == mode and ag.n_params() == r["n_params"]
            if mode == "lora":
                assert (ag.lora_rank, ag._meta["lora_rank"]) == (rank, rank)
            q = E.q_values(ag, np.random.default_rng(0).normal(size=(9, 26, 26)).astype(np.float32), mean, std, var)
            assert q.shape == (27,) and np.isfinite(q).all()
            print(f"  main() --ft-mode {' '.join(extra[1:])}: обучаемых в энкодере {r['ft_enc_trainable']} "
                  f"из {r['ft_enc_total']}, параметров сети {r['n_params']}")
            os.remove(ckpt)
    finally:
        killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("main() с --ft-mode: строка результата, meta чекпоинта, проверка загрузки: OK")


# проверки по именам: без аргументов идут все, с аргументами — только названные
TESTS = dict(defaults=test_defaults, modes=test_modes, lora_init=test_lora_init, two_nets=test_two_nets,
             ckpt=test_ckpt, refusals=test_refusals, main=test_main)


def main():
    names = sys.argv[1:] or list(TESTS)
    if set(names) - set(TESTS):
        raise SystemExit(f"нет проверок {sorted(set(names) - set(TESTS))}: выбор {', '.join(TESTS)}")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            t1 = time.time()
            TESTS[name](tmp)
            print(f"    ({time.time() - t1:.1f} с)")
    print(f"test_peft: {'всё' if not sys.argv[1:] else ', '.join(names)} прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
