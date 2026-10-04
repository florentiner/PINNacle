"""Завершающий шаг «последний слой методом наименьших квадратов» (трек 3, H12a).

Идея из 2603.04672 (last-layer retraining) и одношагового переноса (lPINN, Pi-PINN):
скрытые слои PINN задают базис φ(x), выход u = W φ(x) + b линеен по (W, b). Для линейного
УрЧП с линейными краевыми условиями (Пуассон–Больцман 2D, Пуассон, теплопроводность, волна,
Гельмгольц) невязки тоже линейны по (W, b), и лучший последний слой находится одним решением
взвешенной задачи наименьших квадратов на тех же точках коллокации и с теми же весами лоссов,
что в обучении. Для нелинейных задач (Навье–Стокс) тот же шаг повторяется как
Левенберг–Марквардт по последнему слою (линеаризация невязки, несколько итераций).

Шаг применяется к ЛЮБОМУ методу одинаково (агент, открытые цепочки, правило) после того, как
бюджет эпох исчерпан, поэтому сравнение остаётся честным; стоимость шага сообщается отдельно
(секунды, число вычислений невязки). Арифметика — в двойной точности: сеть временно
приводится к float64, после шага возвращается к прежнему типу.

Проверки без GPU: tests/test_lastlayer.py.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch


def last_linear(net):
    """Выходной линейный слой сети deepxde FNN (или None, если сеть другого вида)."""
    if isinstance(net, AllLayersReadout):
        return net.readout
    lins = getattr(net, "linears", None)
    if lins is None or len(lins) == 0 or not isinstance(lins[-1], torch.nn.Linear):
        return None
    return lins[-1]


class AllLayersReadout(torch.nn.Module):
    """Та же FNN, но выход — линейная комбинация активаций ВСЕХ скрытых слоёв (базис из
    2603.04672 «function space associated with the network»: у сети 100×5 это 500 базисных
    функций вместо 100). Инициализация воспроизводит исходную сеть точно: веса последнего
    слоя стоят на блоке последнего скрытого слоя, остальные нули. Скрытые слои заморожены."""

    def __init__(self, fnn):
        super().__init__()
        self.fnn = fnn
        hidden = [l.out_features for l in fnn.linears[:-1]]
        out_dim = fnn.linears[-1].out_features
        dtype = fnn.linears[-1].weight.dtype
        dev = fnn.linears[-1].weight.device
        self.readout = torch.nn.Linear(sum(hidden), out_dim, dtype=dtype, device=dev)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.weight[:, sum(hidden[:-1]):] = fnn.linears[-1].weight
            self.readout.bias.copy_(fnn.linears[-1].bias)
        for p in self.fnn.parameters():
            p.requires_grad_(False)

    def features(self, inputs):
        fnn = self.fnn
        x = inputs
        if fnn._input_transform is not None:
            x = fnn._input_transform(x)
        feats = []
        for j, linear in enumerate(fnn.linears[:-1]):
            x = (fnn.activation[j](linear(x)) if isinstance(fnn.activation, list)
                 else fnn.activation(linear(x)))
            feats.append(x)
        return torch.cat(feats, dim=1)

    def forward(self, inputs):
        y = self.readout(self.features(inputs))
        if self.fnn._output_transform is not None:
            y = self.fnn._output_transform(inputs, y)
        return y

    # deepxde Model обращается к этим полям у сети
    @property
    def _input_transform(self):
        return self.fnn._input_transform

    @property
    def _output_transform(self):
        return self.fnn._output_transform

    @property
    def auxiliary_vars(self):
        return getattr(self.fnn, "auxiliary_vars", None)


def _get_theta(lin):
    return torch.cat([lin.weight.detach().flatten(), lin.bias.detach().flatten()])


def _set_theta(lin, theta):
    n_w = lin.weight.numel()
    with torch.no_grad():
        lin.weight.copy_(theta[:n_w].view_as(lin.weight))
        lin.bias.copy_(theta[n_w:].view_as(lin.bias))


def weighted_residual(model, X, loss_weights=None):
    """Вектор взвешенных невязок r на точках X (сначала точки краевых условий, затем
    внутренние, как в deepxde): ||r||^2 = сумма w_i * mean(r_i^2) по слагаемым лосса, то есть
    ровно взвешенный лосс обучения. Градиенты по входу нужны для производных в УрЧП,
    поэтому без no_grad; результат отсоединён от графа."""
    import deepxde as dde

    data = model.data
    net = model.net
    dtype = next(net.parameters()).dtype
    dev = next(net.parameters()).device
    dde.grad.clear()
    x = torch.as_tensor(X, dtype=dtype, device=dev).requires_grad_(True)
    y = net(x)
    f = data.pde(x, y) if data.pde is not None else []
    if not isinstance(f, (list, tuple)):
        f = [f]
    starts = [0] + list(np.cumsum(data.num_bcs).astype(int))
    pieces = [fi[starts[-1]:] for fi in f]
    for i, bc in enumerate(data.bcs):
        pieces.append(bc.error(X, x, y, starts[i], starts[i + 1]))
    n_terms = len(pieces)
    w = list(loss_weights) if loss_weights is not None else [1.0] * n_terms
    if len(w) != n_terms:
        raise ValueError(f"loss_weights: {len(w)} весов на {n_terms} слагаемых")
    out = []
    for wi, p in zip(w, pieces):
        p = p.reshape(-1)
        out.append(p * math.sqrt(float(wi) / max(1, p.numel())))
    r = torch.cat(out).detach()
    dde.grad.clear()
    return r


def finish_ls(model, loss_weights=None, max_iter=6, fd_step=1e-4, lam0=1e-6, verbose=True,
              points=None, basis="last"):
    """Пересчитать последний слой. Возвращает словарь с лоссом до и после, числом итераций,
    признаком линейности и временем. Веса меняются только если взвешенный лосс уменьшился.
    basis="last" — последний слой той же сети; basis="all" — считывание со всех скрытых
    слоёв (сеть подменяется на AllLayersReadout, model.net после вызова — обёртка)."""
    t0 = time.time()
    if basis == "all" and not isinstance(model.net, AllLayersReadout):
        if last_linear(model.net) is None:
            return dict(ok=False, reason="нет выходного линейного слоя", time_s=0.0)
        model.net = AllLayersReadout(model.net)
    net = model.net
    lin = last_linear(net)
    if lin is None:
        return dict(ok=False, reason="нет выходного линейного слоя", time_s=0.0)
    X = points if points is not None else model.data.train_x
    if X is None:
        return dict(ok=False, reason="нет точек обучения", time_s=0.0)
    was_dtype = next(net.parameters()).dtype
    net.double()
    n_eval = 0
    try:
        theta = _get_theta(lin).double()
        P = theta.numel()

        def r_of(th):
            nonlocal n_eval
            _set_theta(lin, th)
            n_eval += 1
            return weighted_residual(model, X, loss_weights).double()

        r = r_of(theta)
        loss0 = float((r ** 2).sum())
        # линейна ли невязка по последнему слою: вторая разность вдоль случайного направления
        g = torch.randn_like(theta)
        g = g / g.norm() * max(1.0, float(theta.norm())) * 1e-2
        second = r_of(theta + g) - 2 * r + r_of(theta - g)
        affine = float(second.norm()) <= 1e-6 * max(1.0, float(r.norm()))
        best_theta, best_loss = theta.clone(), loss0
        lam = lam0
        it = 0
        for it in range(1, (1 if affine else max_iter) + 1):
            r = r_of(best_theta)
            # якобиан по столбцам: для линейной задачи одной односторонней разности достаточно
            # (точно до округления), для нелинейной — центральные разности
            J = torch.empty(r.numel(), P, dtype=torch.float64, device=r.device)
            for j in range(P):
                e = torch.zeros_like(best_theta)
                h = fd_step * max(1.0, abs(float(best_theta[j])))
                e[j] = h
                if affine:
                    J[:, j] = (r_of(best_theta + e) - r) / h
                else:
                    J[:, j] = (r_of(best_theta + e) - r_of(best_theta - e)) / (2 * h)
            improved = False
            scale = torch.sqrt((J ** 2).sum(0)).clamp_min(1e-12)
            for _ in range(6):
                # Левенберг–Марквардт как расширенная задача наименьших квадратов:
                # [J; sqrt(lam) * diag(scale)] delta = [-r; 0]
                A = torch.cat([J, math.sqrt(lam) * torch.diag(scale)], 0)
                b = torch.cat([-r, torch.zeros(P, dtype=torch.float64, device=r.device)])
                delta = torch.linalg.lstsq(A, b.unsqueeze(1), driver="gelsd").solution.squeeze(1)
                cand = best_theta + delta
                loss_c = float((r_of(cand) ** 2).sum())
                if math.isfinite(loss_c) and loss_c < best_loss:
                    best_theta, best_loss, improved = cand, loss_c, True
                    lam = max(lam / 10, 1e-12)
                    break
                lam *= 10
            if not improved:
                break
            if affine or best_loss <= 1e-30:
                break
        _set_theta(lin, best_theta)
        res = dict(ok=True, loss_before=loss0, loss_after=best_loss, iters=it, affine=bool(affine),
                   n_eval=n_eval, accepted=best_loss < loss0, basis=basis, n_basis=int(P),
                   time_s=round(time.time() - t0, 1))
    finally:
        if was_dtype != torch.float64:
            net.to(was_dtype)
    if verbose:
        print(f"[ls] последний слой ({basis}, {res['n_basis']} параметров): лосс "
              f"{res['loss_before']:.3e} -> {res['loss_after']:.3e}, "
              f"{'линейно' if res['affine'] else 'нелинейно'}, итераций {res['iters']}, "
              f"вычислений невязки {res['n_eval']}, {res['time_s']} с", flush=True)
    return res


def tester_metrics(model, tester):
    """l2re и bc_l2re текущей сети по тем же тестовым точкам, что у TesterCallback
    (после завершающего шага обучение не вызывалось, и колбэк своих чисел не обновил)."""
    if getattr(tester, "disable", False) or getattr(tester, "test_x", None) is None:
        return float("inf"), float("inf")
    with torch.no_grad():
        y = model.predict(tester.test_x)
    l2 = float(np.sqrt(((y - tester.test_y) ** 2).mean()) / tester.solution_l2)
    bc = float("nan")
    if getattr(tester, "test_x_bc", None) is not None and len(tester.test_x_bc) > 0:
        with torch.no_grad():
            yb = model.predict(tester.test_x_bc)
        bc = float(np.sqrt(((yb - tester.test_y_bc) ** 2).mean()) / (tester.solution_l2 + 1e-12))
    elif getattr(tester, "bc_mask", None) is not None and np.any(tester.bc_mask):
        with torch.no_grad():
            yb = model.predict(tester.test_x[tester.bc_mask])
        bc = float(np.sqrt(((yb - tester.test_y[tester.bc_mask]) ** 2).mean()) / (tester.solution_l2 + 1e-12))
    return l2, bc
