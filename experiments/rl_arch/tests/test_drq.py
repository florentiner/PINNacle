#!/usr/bin/env python
"""
Проверки аугментации D4 в духе DrQ (offline_rl.py --aug drq, --aug-k). Без GPU и сети, на синтетических данных
test_map_agent.

  * d4_rows: строка с номером g — то же, что d4_apply(строка, g); вход не меняется; все g равны — как d4_apply;
  * обучение с --aug drq: у s и s' свои преобразования по строкам и независимые между собой; цель обычной
    головы (в том числе двух сетей) берёт ещё K-1 преобразований s' (вызовов d4_rows на пачку 2 + K-1), при
    --aug-k 1 — ни одного лишнего; веса конечны, обе сети учатся;
  * режимы none и d4 не меняются (сверка чисел — test_map_agent.test_defaults, его REF_KW с aug="d4");
  * разбор: --aug-k без --aug drq и --aug-k 0 — отказ.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_drq.py
"""
import contextlib
import io
import os
import sys
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import test_map_agent as TM  # noqa: E402

torch.set_num_threads(1)


def test_rows():
    x = torch.randn(16, 5, 26, 26)
    x0 = x.clone()
    g = np.arange(16) % 8
    y = O.d4_rows(x, g)
    assert torch.equal(x, x0), "вход не меняется"
    for i in range(16):
        assert torch.equal(y[i], O.d4_apply(x[i:i + 1], int(g[i]))[0])
    for k in range(8):
        assert torch.equal(O.d4_rows(x, np.full(16, k)), O.d4_apply(x, k))
    # D4 — группа: у каждого элемента есть обратный, и все 8 преобразований различны на общей карте
    outs = [O.d4_apply(x[:1], k) for k in range(8)]
    assert len({tuple(o.flatten()[:50].tolist()) for o in outs}) == 8
    print("d4_rows: OK")


def test_training():
    data = TM.make_data(in_ch=9)
    mask = np.arange(len(data["A"])) % 5 != 0
    calls = []
    real = O.d4_rows

    def spy(x, g):
        calls.append(np.asarray(g).copy())
        return real(x, g)
    O.d4_rows = spy
    try:
        for k in (1, 3):
            del calls[:]
            net, _ = TM.fit("convnext_dqn", data, mask, aug="drq", aug_k=k, value_bound=True, two_nets=True,
                            epochs=2)
            n_batch = 2 * int(np.ceil(mask.sum() / 12))
            assert len(calls) == n_batch * (2 + k - 1), (k, len(calls), n_batch)
            per = [calls[i:i + 2 + k - 1] for i in range(0, len(calls), 2 + k - 1)]
            assert any(not np.array_equal(p[0], p[1]) for p in per), "s и s' — независимые преобразования"
            assert all(len(set(p[0].tolist())) > 1 for p in per if len(p[0]) >= 8), "свои преобразования по строкам"
            for v in net.model.state_dict().values():
                assert torch.isfinite(v.float()).all()
        torch.manual_seed(3)
        ref = O.QNet("convnext_dqn", O.torch.device("cpu"), in_ch=9, two_nets=True)
        assert any(not torch.equal(a, b) for a, b in zip(net.model.state_dict().values(),
                                                           ref.model.state_dict().values()))
    finally:
        O.d4_rows = real
    print("обучение с --aug drq: преобразования по строкам, K видов цели: OK")


def test_cli():
    for extra, what in ((["--aug-k", "3"], "--aug-k задаётся только"), (["--aug", "drq", "--aug-k", "0"], "--aug-k >= 1")):
        old = sys.argv
        sys.argv = ["offline_rl.py", "--variant", "convnext_dqn", "--subdir", "poisson_boltzmann_2d"] + extra
        try:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                O.main()
        except SystemExit:
            assert what in err.getvalue(), err.getvalue()[-300:]
        else:
            raise AssertionError(f"нет отказа: {extra}")
        finally:
            sys.argv = old
    print("отказы разбора: OK")


def main():
    t0 = time.time()
    test_rows()
    test_training()
    test_cli()
    print(f"test_drq: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
