#!/usr/bin/env python
"""
Проверки приёмов для агента на картах (offline_rl.py: --encoder, --encoder-init, --freeze-encoder,
--ens, --mb-aux, --pani-w, --distill-top, --mask-schedule; mb_targets.py; cv_probe.py --save-encoder)
без GPU и без сети, на крошечных синтетических данных (шесть цепочек по три шага, один-два шага
градиента на проверку):

  * значения по умолчанию — прежний путь: те же имена и формы параметров у cnx_cql_qr, те же числа
    после шага обучения, что давал код до правок (сверка с записанными, если версия torch та же);
  * каждый энкодер и ансамбль голов учатся, чекпоинт проходит через online_eval_env.load_agent и
    отдаёт те же 27 значений Q; чекпоинт старого формата (без полей энкодера и голов) читается;
  * цель ансамбля — минимум по целевым головам, оценка — среднее голов;
  * веса энкодера из файла зонда CV: заморозка, нормировка из файла, вход с контекстом;
  * модельные цели: состояние суррогата из массивов тренера, награда в единицах тренера, жадный
    просмотр вперёд, бюджет (на подставном ансамбле с известным ответом — без sklearn);
  * шум по входу, лучшие цепочки, грубая маска, всё вместе; полный прогон main() с сохранением и
    проверкой чекпоинта, выключатель задач.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_map_agent.py
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import zlib

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
import mb_targets as MB  # noqa: E402

torch.set_num_threads(1)                    # крошечные тензоры: потоки только мешают
QUIET = lambda *a: None  # noqa: E731
CPU = torch.device("cpu")
PDE = "poissonboltzmann2d"
KEYS = ["loss_total", "loss_oper", "loss_bnd"]
# записано кодом до правок (коммит 8aecef7) на make_data() и fit() этого файла, torch 2.12.0
REF_TORCH = "2.12.0"
REF = {
    "cnx_cql_qr": dict(n_params=466960, keys=40, shape_crc=2650789126, abs_sum=18015.83365574549,
                       q=[-0.07670588791370392, 0.03847261890769005, -0.029841596260666847, 0.03565076366066933]),
    "convnext_dqn": dict(n_params=251851, keys=40, shape_crc=2040310570, abs_sum=11275.400973377176,
                         q=[0.07166065275669098, 0.16523633897304535, 0.0553942546248436, 0.09026480466127396]),
}
REF_KW = {"cnx_cql_qr": dict(epochs=2), "convnext_dqn": dict(epochs=2, value_bound=True, n_step=2, aug="d4", dqfd=0.3)}


def make_data(seed=0, n_chains=6, length=3, in_ch=4):
    """Набор в том виде, в каком его отдаёт загрузчик с --chain-fix (награда delta, старт с нуля)."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("S", "S2", "A", "R", "D", "EP", "ERR", "STEP", "SPENT")}
    yy, xx = np.mgrid[0:26, 0:26] / 25.0
    for c in range(n_chains):
        err, spent = 1.0, 0
        maps = np.zeros((4, 26, 26), np.float32)
        for t in range(length):
            a = int(rng.integers(0, 27))
            new = err * float(np.exp(rng.normal(-0.3, 0.3)))
            nxt = np.stack([np.log1p(new * (1 + xx * rng.normal() + yy * rng.normal()) ** 2 + k)
                            for k in range(3)] + [rng.normal(0, 0.1, (26, 26))]).astype(np.float32)
            for k, v in (("S", maps), ("S2", nxt), ("A", a), ("R", err - new), ("D", float(t == length - 1)),
                         ("EP", c), ("ERR", new), ("STEP", t), ("SPENT", spent)):
                cols[k].append(v)
            spent += O.action_epochs(a)
            err, maps = new, nxt
    d = dict(S=np.stack(cols["S"]), S2=np.stack(cols["S2"]), A=np.array(cols["A"], np.int64),
             R=np.array(cols["R"], np.float32), D=np.array(cols["D"], np.float32),
             EP=np.array(cols["EP"], np.int64), ERR=np.array(cols["ERR"], np.float32),
             STEP=np.array(cols["STEP"], np.int64), SPENT=np.array(cols["SPENT"], np.int64))
    d["FIRST"] = (d["STEP"] == 0).astype(np.float32)
    rtg = np.zeros_like(d["R"])
    for i in range(len(rtg) - 1, -1, -1):
        rtg[i] = d["R"][i] + (0.0 if d["D"][i] else O.GAMMA * rtg[i + 1])
    d["RTG"] = rtg
    if in_ch > 4:                           # контекст постоянными плоскостями, как attach_ctx
        ctx = rng.uniform(-1, 1, (len(d["A"]), in_ch - 4)).astype(np.float32)
        d["S"], d["S2"] = O.attach_ctx(d["S"], ctx), O.attach_ctx(d["S2"], ctx)
    return d


def ns(**kw):
    base = dict(batch_size=12, epochs=1, cql_alpha=1.0, plateau_patience=0, cpu=True, dqfd_margin=0.8)
    base.update(kw)
    return argparse.Namespace(**base)


def fit(variant, data, mask, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        net, stats, _ = O.train_variant(variant, data, mask, ns(**kw), seed=3)
    return net, stats


def shapes_of(net):
    return [(k, list(v.shape)) for k, v in net.model.state_dict().items()]


def save_ckpt(net, stats, path, meta=None):
    """Чекпоинт в формате offline_rl.main (--save-model)."""
    meta = dict(encoder=net.enc_kind, ens=net.ens) if meta is None else meta
    torch.save({"variant": net.variant, "seed": 3,
                "state_dict": {k: v.cpu() for k, v in net.model.state_dict().items()},
                "mean": stats["mean"], "std": stats["std"], "meta": meta}, path)
    return path


# --------------------------------------------------------------------------------------------

def test_defaults(data, mask):
    x = torch.as_tensor(np.random.default_rng(5).normal(size=(3, 4, 26, 26)).astype(np.float32))
    for variant, ref in REF.items():
        net, stats = fit(variant, data, mask, **REF_KW[variant])
        sh = shapes_of(net)
        assert net.enc_kind == "convnext" and net.ens == 1 and not net.has_bn and not net.frozen
        assert net.n_params() == ref["n_params"] and len(sh) == ref["keys"], (variant, net.n_params(), len(sh))
        assert zlib.crc32(json.dumps(sh).encode()) == ref["shape_crc"], f"{variant}: имена или формы параметров изменились"
        assert not net.extra, net.extra
        if torch.__version__.split("+")[0] != REF_TORCH:
            print(f"  {variant}: torch {torch.__version__} вместо {REF_TORCH} — сверка чисел пропущена")
            continue
        with torch.no_grad():
            q = net.q_scalar(x).numpy()
        tot = float(sum(v.double().abs().sum() for v in net.model.state_dict().values()))
        assert abs(tot - ref["abs_sum"]) < 1e-6 * ref["abs_sum"], (variant, tot, ref["abs_sum"])
        assert np.allclose(q[0, :4], ref["q"], atol=1e-5), (variant, q[0, :4], ref["q"])
    sh = shapes_of(O.QNet("cnx_cql_qr", CPU))
    assert sh[0] == ("0.stem.weight", [48, 4, 2, 2]) and sh[-2:] == [("1.weight", [864, 256]), ("1.bias", [864])]
    assert O.ckpt_arch(dict(variant="cnx_cql_qr", state_dict=dict(O.QNet("cnx_cql_qr", CPU, in_ch=9).model.state_dict()))) \
        == dict(n_quantiles=32, in_ch=9, encoder="auto", ens=1)
    for bad in (lambda: O.QNet("cnn_dqn", CPU, encoder="resnet"), lambda: O.QNet("cnn_iql", CPU, ens=2),
                lambda: O.QNet("cnx_cql", CPU, encoder="lstm")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("неподдержанное сочетание должно отвергаться")
    print("значения по умолчанию: прежняя сеть и прежние числа: OK")


def test_ensemble_math():
    """Оценка — среднее голов; цель TD — минимум по целевым головам (сверка с прямым расчётом)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    s, s2 = torch.randn(5, 4, 26, 26), torch.randn(5, 4, 26, 26)
    a = torch.as_tensor(rng.integers(0, 27, 5))
    r, d, gm = torch.randn(5), torch.tensor([0., 0., 1., 0., 0.]), torch.full((5,), 0.9)
    net = O.QNet("convnext_dqn", CPU, ens=3)
    with torch.no_grad():
        for p in net.target.parameters():       # целевые головы должны отличаться от обучаемых
            p.add_(0.05 * torch.randn_like(p))
        qh = net.q_heads(s)
        assert qh.shape == (3, 5, 27) and torch.allclose(net.q_online(s), qh.mean(0), atol=1e-6)
        assert torch.allclose(net.q_scalar(s), qh.mean(0), atol=1e-6)
        a2 = net.q_heads(s2).mean(0).argmax(1)
        qt = net.q_heads(s2, target=True).gather(2, a2[None, :, None].expand(3, -1, 1)).squeeze(2)
        cand = [float(((qh.gather(2, a[None, :, None].expand(3, -1, 1)).squeeze(2)
                        - (r + gm * (1 - d) * t)[None]) ** 2).mean()) for t in (qt.min(0).values, *qt)]
        full = float(O.ens_td_loss(net, "convnext_dqn", s, a, r, d, gm, s2, None, 1.0, 3, rng))
        one = float(O.ens_td_loss(net, "convnext_dqn", s, a, r, d, gm, s2, None, 1.0, 1, rng))
        ub = float(O.ens_td_loss(net, "convnext_dqn", s, a, r, d, gm, s2, None, 1.0, 3, rng,
                                 ub=torch.zeros(5)))
    assert abs(full - cand[0]) < 1e-6, (full, cand[0])
    assert min(abs(one - c) for c in cand[1:]) < 1e-6, "при --ens-min 1 цель — одна из целевых голов"
    assert abs(ub - float(((qh.gather(2, a[None, :, None].expand(3, -1, 1)).squeeze(2) - r[None]) ** 2).mean())) < 1e-6
    with torch.no_grad():                       # штраф BCQ считается на каждой голове
        keep = torch.zeros(5, 27, dtype=torch.bool)
        bcq = float(O.ens_td_loss(net, "cnx_bcq", s, a, r, d, gm, s2, None, 1.0, 3, rng, bcq_keep=keep))
    assert abs(bcq - full - 0.5 * float((qh.clamp_min(0) ** 2).mean())) < 1e-6
    qr = O.QNet("cnx_cql_qr", CPU, ens=2, encoder="vit")
    taus = (torch.arange(32, dtype=torch.float32) + 0.5) / 32
    with torch.no_grad():
        q4 = qr.q_heads(s)
        assert q4.shape == (2, 5, 27, 32) and torch.allclose(qr.q_scalar(s), q4.mean((0, 3)), atol=1e-6)
        assert qr.q_cvar(s).shape == (5, 27) and (qr.q_cvar(s) <= qr.q_scalar(s) + 1e-6).all()
    loss = O.ens_td_loss(qr, "cnx_cql_qr", s, a, r, d, gm, s2, taus, 1.0, 2, rng,
                         allow=torch.as_tensor(O.coarse_actions(1)))
    loss.backward()
    assert torch.isfinite(loss) and all(h.weight.grad is not None for h in qr.model[1])
    print("ансамбль голов: среднее на оценке, минимум в цели: OK")


def test_encoders_and_ckpt(data, mask, tmp):
    import online_eval_env as E
    torch.set_default_tensor_type(torch.FloatTensor)
    rng = np.random.default_rng(1)
    cases = [("cnx_cql_qr", dict(encoder="resnet")), ("cnx_cql_qr", dict(encoder="vit")),
             ("cnx_cql_qr", dict(encoder="resnetpos")),
             ("cnx_cql_qr", dict(ens=3, ens_min=2)),
             ("convnext_dqn", dict(encoder="resnet", ens=2, value_bound=True))]
    for variant, kw in cases:
        t0 = time.time()
        net, stats = fit(variant, data, mask, **kw)
        assert net.enc_kind == kw.get("encoder", "convnext") and net.ens == kw.get("ens", 1)
        assert net.has_bn == (net.enc_kind in ("resnet", "resnetpos")) and not (net.has_bn and net.model.training)
        path = save_ckpt(net, stats, os.path.join(tmp, f"{variant}_{net.enc_kind}_{net.ens}.pt"))
        ag, mean, std, var = E.load_agent(path)
        assert var == variant and ag.enc_kind == net.enc_kind and ag.ens == net.ens
        assert ag.nq == net.nq or variant not in O.QUANTILE_VARIANTS     # у скалярных голов nq не используется
        assert shapes_of(ag) == shapes_of(net)
        for k in range(2):
            st = (rng.normal(size=(4, 26, 26)) * (k > 0)).astype(np.float32)
            q = E.q_values(ag, st, mean, std, var)
            assert q.shape == (27,) and np.isfinite(q).all()
            assert np.abs(q - E.q_values(net, st, mean, std, var)).max() < 1e-5
            assert E.pick_action(ag, st, mean, std, var) == int(np.argmax(q))
        v = O.verify_ckpt(path, net)
        assert v["ok"] and v["encoder"] == net.enc_kind and v["ens"] == net.ens and v["max_abs_dq"] < 1e-5, v
        assert any(k.startswith("1.0.") for k in ag.model.state_dict()) == (net.ens > 1)
        print(f"  {variant} {net.enc_kind} голов {net.ens}: параметров {net.n_params()}, {time.time() - t0:.2f} с")
    # чекпоинты старого формата: meta без полей энкодера и голов (и без meta вовсе)
    for variant, in_ch, meta in (("cnx_cql_qr", 4, {"state_mode": "full"}), ("stat_dqn", 9, None),
                                 ("convnext_dqn", 9, {})):
        net = O.QNet(variant, CPU, in_ch=in_ch)
        ck = {"variant": variant, "state_dict": net.model.state_dict(),
              "mean": np.zeros((1, in_ch, 1, 1), np.float32), "std": np.ones((1, in_ch, 1, 1), np.float32)}
        if meta is not None:
            ck["meta"] = meta
        path = os.path.join(tmp, f"old_{variant}.pt")
        torch.save(ck, path)
        ag, mean, std, var = E.load_agent(path)
        assert ag.enc_kind == net.enc_kind and ag.ens == 1 and shapes_of(ag) == shapes_of(net)
        q = E.q_values(ag, np.zeros((in_ch, 26, 26), np.float32), mean, std, var)
        assert q.shape == (27,) and np.isfinite(q).all()
    # комитет из агентов с разными энкодерами
    agents = [E.load_agent(os.path.join(tmp, f)) for f in ("cnx_cql_qr_resnet_1.pt", "cnx_cql_qr_convnext_3.pt")]
    act = E.ensemble_action(agents, np.zeros((4, 26, 26), np.float32), np.ones(27, bool), how="mean")
    assert 0 <= act < 27
    print("энкодеры и ансамбль: обучение, чекпоинт через load_agent, старые чекпоинты: OK")


def test_encoder_init(data, mask, tmp):
    import cv_probe as CV
    tab, maps = CV.synthetic_data(40, seed=1)
    D = CV.Data(tab, maps, CPU)
    a = CV.build_parser().parse_args(["--tag", "t", "--ssl-epochs", "1", "--batch", "16", "--no-upload",
                                      "--out-dir", tmp, "--save-encoder"])
    cache = dict(pre={}, z={})
    pre = CV.get_pre(dict(ssl="mae", encoder="resnet"), D, a, CPU, cache, QUIET)
    path = os.path.join(tmp, "cvprobe_t_enc_mae_resnet.pt")
    assert pre["file"] == path and os.path.exists(path)
    a.save_encoder = False                  # без флага файл не пишется (поведение зонда по умолчанию)
    assert "file" not in CV.get_pre(dict(ssl="con", encoder="vit"), D, a, CPU, cache, QUIET)
    assert not os.path.exists(os.path.join(tmp, "cvprobe_t_enc_con_vit.pt"))
    f = torch.load(path, weights_only=False)
    assert f["encoder"] == "resnet" and f["ssl"] == "mae" and f["mean"].shape == (1, 4, 1, 1)
    # заморозка: энкодер равен файлу, нормировка карт — из файла, головы учатся
    net, stats = fit("cnx_cql_qr", data, mask, encoder="resnet", encoder_init=path, freeze_encoder=True)
    sd = net.model[0].state_dict()
    assert all(torch.equal(sd[k], f["state_dict"][k]) for k in sd), "замороженный энкодер изменился"
    assert net.frozen and not net.model[0].training and not any(p.requires_grad for p in net.model[0].parameters())
    assert np.allclose(stats["mean"][:, :4], f["mean"]) and np.allclose(stats["std"][:, :4], f["std"])
    assert net.extra.get("norm_from_encoder_init") is True
    torch.manual_seed(3)                    # та же инициализация, что внутри train_variant(seed=3)
    head0 = O.QNet("cnx_cql_qr", CPU, encoder="resnet").model[1].weight
    assert net.model[1].weight.shape == head0.shape and not torch.equal(net.model[1].weight, head0), "голова должна учиться"
    # дообучение: энкодер стартует с файла и меняется, нормировка своя
    net2, stats2 = fit("cnx_cql_qr", data, mask, encoder="resnet", encoder_init=path)
    w0, w1 = f["state_dict"]["stem.0.weight"], net2.model[0].state_dict()["stem.0.weight"]
    assert not torch.equal(w0, w1) and (w0 - w1).abs().max() < 0.05
    assert not np.allclose(stats2["mean"][:, :4], f["mean"]) and "norm_from_encoder_init" not in net2.extra
    # вход с контекстом (9 каналов): первые 4 канала первой свёртки из файла, она остаётся обучаемой
    d9 = make_data(in_ch=9)
    q9 = O.QNet("cnx_cql_qr", CPU, in_ch=9, encoder="resnet")
    info = O.load_encoder_init(q9, path)
    assert info["partial"] == ["stem.0.weight"] and info["ssl"] == "mae"
    assert torch.equal(q9.model[0].stem[0].weight[:, :4], w0) and q9.model[0].stem[0].weight.shape[1] == 9
    assert torch.equal(q9.target[0].stem[0].weight, q9.model[0].stem[0].weight)
    net9, _ = fit("cnx_cql_qr", d9, mask, encoder="resnet", encoder_init=path, freeze_encoder=True,
                  state_mode="full")
    grad = [k for k, p in net9.model[0].named_parameters() if p.requires_grad]
    assert grad == ["stem.0.weight"], grad
    # энкодер другого агента как старт; несовпадение вида энкодера отвергается
    apath = save_ckpt(net2, stats2, os.path.join(tmp, "agent_resnet.pt"))
    q = O.QNet("cnx_cql", CPU, encoder="resnet")
    assert O.load_encoder_init(q, apath)["encoder"] == "resnet"
    assert torch.equal(q.model[0].stem[0].weight, w1)
    try:
        O.load_encoder_init(O.QNet("cnx_cql", CPU), path)
    except ValueError as e:
        assert "--encoder resnet" in str(e)
    else:
        raise AssertionError("веса resnet не должны грузиться в convnext")
    print("веса энкодера из зонда CV: заморозка, дообучение, вход с контекстом: OK")


class FakeEns:
    """Подставной ансамбль с известным ответом: изменение log10 ошибки зависит только от действия,
    член m смещён на 0.02*m."""
    members = 3

    def predict_ln(self, X, l, member=None):
        d = -(0.05 + 0.01 * X[:, 0])
        out = np.stack([np.clip(l + d + 0.02 * m, S.L_LO, S.L_HI) for m in range(self.members)])
        return out if member is None else out[member]


def fake_model():
    return dict(A=np.zeros(0, np.int64), CTX=S.pde_ctx(O.pde_meta()[PDE])[None], NAMES=[PDE]), FakeEns()


def test_mb(data, mask):
    tab, ens = fake_model()
    e0 = float(O.pde_meta()[PDE]["init_err"])
    n = len(data["A"])
    st, valid, first = MB.chain_states(data, tab, np.array([0]), np.ones(1))
    assert valid.all() and (first == (data["STEP"] == 0)).all()
    assert np.allclose(st["l"][first], np.log10(e0)) and np.allclose(st["l"][1:][~first[1:]], np.log10(data["ERR"][:-1][~first[1:]]))
    assert (st["pa"][first] == -1).all() and (st["pa"][1:][~first[1:]] == data["A"][:-1][~first[1:]]).all()
    assert np.allclose(st["spf"].sum(1), data["SPENT"]) and (st["spf"][first] == 0).all()
    cut = {k: (v[1:] if isinstance(v, np.ndarray) else v) for k, v in data.items()}      # цепочка без начала
    _, v2, _ = MB.chain_states(cut, tab, np.array([0]), np.ones(1))
    assert not v2[:2].any() and v2[2:].all()
    _, v3, _ = MB.chain_states(data, tab, np.array([0]), np.ones(1), budget=int(data["SPENT"].max()))
    assert v3.sum() == n - int((data["SPENT"] == data["SPENT"].max()).sum())
    sd3 = float(np.std([0.0, 0.02, 0.04]))
    r1 = 0.05 + 0.01 * np.arange(27) - 0.02 - 0.5 * sd3              # dlog, одно действие
    best = r1[26]
    zf, sc = first.copy(), np.ones(n)
    q1, unc = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 1, 0.5)
    assert q1.shape == (n, 27) and abs(unc - sd3) < 1e-9
    assert np.allclose(q1[~first], r1[None]) and not q1[first].any(), "награда первого шага без init_err — ноль"
    q2, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 2, 0.5)
    assert np.allclose(q2, q1 + 0.9 * best), "второй уровень: лучшее действие со скидкой"
    q3, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 0.5, chunk=5)
    assert np.allclose(q3, q1 + 0.9 * best + 0.81 * best)
    qs, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 2, 0.5, smdp_scale=100.0)
    assert np.allclose(qs, q1 + (0.9 ** (S.EPOCHS / 100.0))[None] * best)
    qn, _ = MB.lookahead(ens, tab, st, st["l"], zf, sc, "dlog", 0.9, 3, 20.0)  # штраф больше награды: хвоста нет
    assert best - 19.5 * sd3 < 0 and np.allclose(qn[~first], (r1 - 19.5 * sd3)[None])
    # бюджет 7000. Остаток 100 эпох: любое действие последнее. Остаток 500: Adam на 2500 эпох
    # обрезается и кончает эпизод, а после Adam на 100 эпох остаётся ещё два действия PSO (300 и 100)
    one = {k: v[:2].copy() for k, v in st.items()}
    one["spent"][:] = [6900, 6500]
    one["step"][:] = [5, 5]
    qb1, _ = MB.lookahead(ens, tab, one, one["l"], np.zeros(2, bool), np.ones(2), "dlog", 1.0, 1, 0.5, budget=7000)
    qb4, _ = MB.lookahead(ens, tab, one, one["l"], np.zeros(2, bool), np.ones(2), "dlog", 1.0, 4, 0.5, budget=7000)
    assert np.allclose(qb4[0], qb1[0]) and np.allclose(qb4[1, 2], qb1[1, 2]) \
        and np.allclose(qb4[1, 0], qb1[1, 0] + 2 * best)
    # delta: (E_до - E_после) / масштаб по членам, среднее минус разброс
    i = int(np.where(~first)[0][0])
    E0 = 10.0 ** st["l"][i]
    Rm = E0 * (1.0 - 10.0 ** (-(0.05 + 0.01 * 7) + 0.02 * np.arange(3))) / 4.0
    qd, _ = MB.lookahead(ens, tab, st, st["l"], zf, np.full(n, 4.0), "delta", 0.9, 1, 0.5)
    assert abs(qd[i, 7] - (Rm.mean() - 0.5 * Rm.std())) < 1e-12
    try:
        MB.lookahead(ens, tab, st, st["l"], zf, sc, "level", 0.9, 1, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("награда-уровень не поддержана")
    # полный расчёт: только обучающие строки; первый шаг от init_err в единицах тренера
    QMB, MBW, info = MB.precompute(data, mask, [PDE], "dlog", np.ones(1), np.array([2.0 * e0]), 0.9, h=1,
                                   lam=0.5, model=(tab, ens), log=QUIET)
    assert QMB.shape == (n, 27) and QMB.dtype == np.float32 and (MBW == mask).all() and not QMB[~mask].any()
    assert np.allclose(QMB[mask & first], (r1 + np.log10(2.0))[None], atol=1e-6) and info["rows"] == int(mask.sum())
    assert np.allclose(QMB[mask & ~first], r1[None], atol=1e-6)
    d2 = dict(data, QMB=QMB, MBW=MBW)
    for variant in ("cnn_dqn", "cnn_qrdqn"):
        net, _ = fit(variant, d2, mask, mb_aux=0.5)
        assert np.isfinite(net.extra["mb_loss"]) and net.extra["mb_loss"] > 0
    try:
        fit("cnn_dqn", data, mask, mb_aux=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("без посчитанных целей --mb-aux должен падать с понятной ошибкой")
    print("модельные цели: состояние, награда в единицах тренера, просмотр вперёд, бюджет: OK")
    return QMB, MBW


def test_aux(data, mask, QMB, MBW):
    net, _ = fit("cnn_qrdqn", data, mask, pani_w=0.2, pani_sigma=1.0)
    assert net.extra["pani_loss"] > 0 and np.isfinite(net.extra["pani_loss"])
    net, _ = fit("cnn_dqn", make_data(in_ch=9), mask, pani_w=0.2, pani_sigma=0.0)
    assert net.extra["pani_loss"] == 0.0, "нулевой шум не меняет Q"
    # лучшие цепочки: по ошибке в конце цепочки, среди обучающих, по каждой задаче отдельно
    last = np.where(data["D"] == 1)[0]
    fin = {int(data["EP"][i]): float(data["ERR"][i]) for i in last if mask[i]}
    demo, info = O.top_chain_rows(data, mask, 50.0)
    top = sorted(fin, key=fin.get)[:len(fin) // 2]
    assert info == dict(distill_chains=len(top), distill_ranked=len(fin), distill_rows=3 * len(top))
    assert (demo == (np.isin(data["EP"], top) & mask)).all()
    two = dict(data, PDE=(data["EP"] % 2).astype(np.int64))
    d_two, i_two = O.top_chain_rows(two, mask, 50.0)
    for p in (0, 1):
        f = {e: v for e, v in fin.items() if e % 2 == p}
        assert set(two["EP"][d_two & (two["PDE"] == p)]) == {min(f, key=f.get)}
    net, _ = fit("cnn_dqn", data, mask, distill_top=50.0, distill_w=0.5)
    assert net.extra["distill_rows"] == 3 * len(top) and np.isfinite(net.extra["distill_top_loss"])
    c = O.coarse_actions(1)
    assert c.sum() == 9 and sorted((a // 9, a % 3) for a in np.where(c)[0]) == [(o, e) for o in range(3) for e in range(3)]
    for variant in ("cnn_dqn", "cnn_qrdqn"):
        fit(variant, data, mask, mask_schedule=True, epochs=3)
    t0 = time.time()
    net, _ = fit("cnx_cql_qr", dict(data, QMB=QMB, MBW=MBW), mask, encoder="resnet", ens=3, ens_min=2,
                 mb_aux=0.5, pani_w=0.1, pani_sigma=0.1, distill_top=50.0, distill_w=0.3, mask_schedule=True,
                 aug="d4", n_step=2, dqfd=0.1, epochs=3)
    assert {"mb_loss", "pani_loss", "distill_top_loss", "distill_rows"} <= set(net.extra)
    assert all(np.isfinite(v) for v in net.extra.values()) and not net.model.training
    print(f"шум по входу, лучшие цепочки, грубая маска, всё вместе ({time.time() - t0:.1f} с): OK")


def make_dump(rng, n_chains, e0):
    """Файл в формате дампа (как в test_cv_probe): next_state верный, конец цепочки — done=-1."""
    out = []
    for _ in range(n_chains):
        chain = [int(a) for a in rng.integers(0, 27, 3)]
        maps = [{k: np.zeros((26, 26), np.float32) for k in KEYS}]
        l, errs = np.log10(e0), []
        for _a in chain:
            l += rng.normal(-0.1, 0.1)
            errs.append(10 ** l)
            maps.append({k: rng.random((26, 26)).astype(np.float32) + 1.0 for k in KEYS})
        for j, a in enumerate(chain):
            done = -1 if j == len(chain) - 1 else 0
            out.append(dict(state=dict(maps[j]), next_state=dict(maps[j + 1]),
                            action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                            reward=-(errs[j] + (1.0 if done == -1 else 0.0)), done=done))
    return out


def test_main(tmp):
    """Полный прогон main(): флаги -> строка результата и meta чекпоинта, проверка загрузки."""
    import killswitch
    import online_eval_env as E
    rng = np.random.default_rng(0)
    e0 = float(O.pde_meta()[PDE]["init_err"])
    root = os.path.join(tmp, "pb2d")
    os.makedirs(root)
    for k in range(3):
        torch.save(make_dump(rng, 2, e0), os.path.join(root, f"f{k}.pt"))
    mtag = f"tmap{os.getpid()}"
    ckpt = f"/tmp/agent_cnx_cql_qr_cfix_rdelta_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "cnx_cql_qr", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", str(round(e0, 3)), "--episode-budget", "7000", "--gamma", "0.99", "--seeds", "1",
            "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu", "--save-model", "--model-tag", mtag,
            "--encoder", "resnet", "--ens", "3", "--ens-min", "2", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
            "--pani-w", "0.1", "--pani-sigma", "0.05", "--distill-top", "50", "--distill-w", "0.2", "--mask-schedule"]
    seen = {}

    def fake_build(names, subdirs, table, members, work, quick, log):
        seen.update(names=list(names), subdirs=subdirs, members=members)
        tab, ens = fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3)

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: seen.setdefault("kill_tag", tag) and None
    MB.build_model = fake_build
    buf = io.StringIO()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buf):
            O.main()
        rows = [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.startswith('{"variant"')]
        assert [r["policy"] for r in rows] == ["mean", "cvar"], buf.getvalue()[-2000:]
        r = rows[0]
        want = dict(encoder="resnet", ens=3, ens_min=2, encoder_init="", freeze_encoder=False, mb_aux=0.5, mb_h=2,
                    mb_lam=0.5, mb_subdirs="all", mb_members=5, pani_w=0.1, pani_sigma=0.05, distill_top=50.0,
                    distill_w=0.2, mask_schedule=True, mask_coarse_lr=1)
        assert {k: r[k] for k in want} == want, {k: r.get(k) for k in want}
        assert r["mb_rows"] > 0 and r["mb_surr_sig"] == "fake" and np.isfinite(r["mb_loss"]) and r["distill_rows"] > 0
        assert np.isfinite(r["pani_loss"]) and np.isfinite(r["td_error"]) and np.isfinite(r["fqe_value_all"])
        assert seen["names"] == [PDE] and seen["kill_tag"] == f"offline_cnx_cql_qr_cfix_rdelta_{mtag}"
        assert "проверка загрузки через online_eval_env" in buf.getvalue() and "'ok': True" in buf.getvalue()
        ag, mean, std, var = E.load_agent(ckpt)
        meta = ag._meta
        assert {k: meta[k] for k in want} == want and meta["in_ch"] == 4 and meta["n_quantiles"] == 32
        assert meta["mb"]["rows"] == r["mb_rows"] and meta["reward_form"] == "delta" and meta["episode_budget"] == 7000
        assert ag.enc_kind == "resnet" and ag.ens == 3 and not ag.model.training
        q = E.q_values(ag, np.zeros((4, 26, 26), np.float32), mean, std, var)
        assert q.shape == (27,) and np.isfinite(q).all()
        # отменённая задача выходит до загрузки буферов
        killswitch.killed = lambda tag, remote=False, every=600: "tmap"
        sys.argv = argv[:4] + ["/nonexistent"] + argv[5:]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                O.main()
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("выключатель задач не сработал")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("main(): строка результата, meta чекпоинта, проверка загрузки, выключатель: OK")


def main():
    t0 = time.time()
    data = make_data()
    mask = data["EP"] < 4
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    test_defaults(data, mask); lap()
    test_ensemble_math(); lap()
    QMB, MBW = test_mb(data, mask); lap()
    test_aux(data, mask, QMB, MBW); lap()
    with tempfile.TemporaryDirectory() as tmp:
        test_encoder_init(data, mask, tmp); lap()
        test_encoders_and_ckpt(data, mask, tmp); lap()
        test_main(tmp); lap()
    print(f"test_map_agent: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
