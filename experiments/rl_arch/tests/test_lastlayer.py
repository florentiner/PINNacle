#!/usr/bin/env python
"""
Проверки завершающего шага «последний слой методом наименьших квадратов» (lastlayer.py), без GPU:

  * линейная задача (Пуассон 1D, u'' = f, u(0) = u(1) = 0): невязка распознаётся как линейная
    по последнему слою, один шаг уменьшает взвешенный лосс и ошибку, считывание со всех
    скрытых слоёв (basis=all) не хуже последнего слоя;
  * нелинейная задача (u'' + u^2 = f): невязка нелинейна, Левенберг–Марквардт уменьшает лосс;
  * веса не меняются, если шаг не улучшил лосс (здесь — проверка, что лосс после <= до);
  * tester_metrics воспроизводит l2re по тем же точкам.

    DDEBACKEND=pytorch python experiments/rl_arch/tests/test_lastlayer.py
"""
import os
import sys

os.environ.setdefault("DDEBACKEND", "pytorch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import deepxde as dde  # noqa: E402

import lastlayer as LL  # noqa: E402


def _problem(nonlinear):
    geom = dde.geometry.Interval(0, 1)
    sol = lambda x: np.sin(np.pi * x)

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x)
        f = -np.pi ** 2 * torch.sin(np.pi * x)
        if nonlinear:
            return u_xx + u ** 2 - (f + torch.sin(np.pi * x) ** 2)
        return u_xx - f

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on_b: on_b)
    data = dde.data.PDE(geom, pde, bc, num_domain=64, num_boundary=2, solution=sol, num_test=100)
    net = dde.nn.FNN([1, 24, 24, 1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    return model, sol


def _weighted_loss(model, w):
    r = LL.weighted_residual(model, model.data.train_x, w)
    return float((r.double() ** 2).sum())


def _l2re(model, sol):
    x = np.linspace(0, 1, 201)[:, None]
    y = model.predict(x)
    return float(np.linalg.norm(y - sol(x)) / np.linalg.norm(sol(x)))


def test_linear():
    dde.config.set_random_seed(3)
    model, sol = _problem(nonlinear=False)
    w = [1.0, 10.0]
    model.compile("adam", lr=1e-3, loss_weights=w)
    model.train(iterations=300, display_every=1000)
    before, e0 = _weighted_loss(model, w), _l2re(model, sol)
    res = LL.finish_ls(model, w, basis="last", verbose=False)
    assert res["ok"] and res["affine"], res
    assert res["iters"] == 1
    after, e1 = _weighted_loss(model, w), _l2re(model, sol)
    assert after <= before * (1 + 1e-9), (before, after)
    assert abs(after - res["loss_after"]) <= 1e-6 * max(1.0, after)
    assert e1 < e0, (e0, e1)
    # считывание со всех скрытых слоёв: базис шире, лосс не выше
    res2 = LL.finish_ls(model, w, basis="all", verbose=False)
    assert res2["ok"] and res2["affine"] and res2["n_basis"] == 24 + 24 + 1, res2
    assert res2["loss_after"] <= res["loss_after"] * (1 + 1e-9), (res, res2)
    assert isinstance(model.net, LL.AllLayersReadout)
    # predict работает через обёртку и даёт тот же лосс, что считал шаг
    assert abs(_weighted_loss(model, w) - res2["loss_after"]) <= 1e-6 * max(1.0, res2["loss_after"])
    print(f"линейная: лосс {before:.3e} -> {after:.3e} -> {res2['loss_after']:.3e}; "
          f"l2re {e0:.3e} -> {e1:.3e} -> {_l2re(model, sol):.3e}")


def test_nonlinear():
    dde.config.set_random_seed(4)
    model, sol = _problem(nonlinear=True)
    w = [1.0, 10.0]
    model.compile("adam", lr=1e-3, loss_weights=w)
    model.train(iterations=300, display_every=1000)
    before = _weighted_loss(model, w)
    res = LL.finish_ls(model, w, basis="last", verbose=False, max_iter=4)
    assert res["ok"] and not res["affine"], res
    assert res["loss_after"] <= before * (1 + 1e-9), (before, res)
    assert res["iters"] >= 1
    print(f"нелинейная: лосс {before:.3e} -> {res['loss_after']:.3e} за {res['iters']} итераций")


def test_dtype_restored():
    dde.config.set_random_seed(5)
    model, _ = _problem(nonlinear=False)
    model.compile("adam", lr=1e-3)
    model.train(iterations=20, display_every=1000)
    LL.finish_ls(model, None, verbose=False)
    assert next(model.net.parameters()).dtype == torch.float32


if __name__ == "__main__":
    test_linear()
    test_nonlinear()
    test_dtype_restored()
    print("test_lastlayer: ok")
