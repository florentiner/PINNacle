#!/usr/bin/env python
"""
Проверки режима состояния tele (сеть и GPU не нужны):

  * tele_state: форма (10, 26, 26), конечность, каналы постоянны и лежат в [-3, 3];
  * curve_stats на синтетических кривых: застой (float32-плато) даёт малый |t| и долю
    улучшений около нуля, падающая кривая — сильно отрицательный наклон и большой |t|;
  * частота записи кривой и отказ apply_state_mode для tele.

    python experiments/rl_arch/tests/test_tele.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import offline_rl as O  # noqa: E402

RNG = np.random.default_rng(3)


def _stalled(n=21, level=3.2e-4):
    """Застой L-BFGS в float32: лосс стоит, меняются только последние разряды."""
    steps = np.arange(n) * 50 + 1000
    ulp = np.float32(level) * np.float32(1.2e-7)
    loss = np.float32(level) + ulp * RNG.integers(-2, 3, size=n)
    return steps, loss.astype(np.float64)


def _falling(n=21, decades_per_100=-0.5):
    steps = np.arange(n) * 50 + 1000
    loss = 1e-1 * 10 ** (decades_per_100 * (steps - steps[0]) / 100.0) * np.exp(RNG.normal(0, 0.01, n))
    return steps, loss


def test_curve_stats():
    st = O.curve_stats(*_stalled())
    assert abs(st["t"]) < 3.0, st
    assert st["improve"] < 0.1, st
    assert abs(st["slope100"]) < 1e-5, st
    flat = O.curve_stats(np.arange(10) * 10, np.full(10, 5e-3))
    assert flat["t"] == 0.0 and flat["improve"] == 0.0 and flat["slope100"] == 0.0
    fa = O.curve_stats(*_falling())
    assert fa["slope100"] < -0.4 and fa["t"] < -50 and fa["improve"] > 0.9, fa
    # вторая половина: излом в середине виден как застой
    s1, l1 = _falling(11)
    s2 = s1[-1] + np.arange(1, 11) * 50
    st2 = O.curve_stats(np.r_[s1, s2], np.r_[l1, np.full(10, l1[-1])])
    assert abs(st2["slope100"]) < 0.05 and st2["improve"] < 0.2, st2
    assert O.curve_stats([5], [1.0])["n"] == 1 and O.curve_stats([], [])["t"] == 0.0
    print("curve_stats: застой |t| мал, улучшений нет; падение — наклон < 0, |t| велик: OK")


def test_tele_state():
    for steps, loss in (_stalled(), _falling(), ([], []), ([0], [1.0])):
        s = O.tele_state(1e-2, 4e-3, 6e-3, prev_total=1.0, curve_steps=steps, curve_loss=loss,
                         grad_norm=0.3, rel_disp=1e-3)
        assert s.shape == (O.TELE_CH, 26, 26) and s.dtype == np.float32
        assert np.isfinite(s).all() and np.abs(s).max() <= 3.0
        assert (s == s[:, :1, :1]).all()                     # постоянные каналы
    z = O.tele_state(1e-2, 4e-3, 6e-3)                     # первый шаг: нет кривой и градиента
    assert np.all(z[4:] == 0.0) and z[3, 0, 0] == 0.0
    np.testing.assert_array_equal(z[:4], O.loss_state(1e-2, 4e-3, 6e-3))
    sa = O.tele_state(1e-2, 4e-3, 6e-3, curve_steps=_stalled()[0], curve_loss=_stalled()[1],
                      grad_norm=1e-5, rel_disp=1e-9)
    fa = O.tele_state(1e-2, 4e-3, 6e-3, curve_steps=_falling()[0], curve_loss=_falling()[1],
                      grad_norm=1.0, rel_disp=1e-2)
    assert fa[4, 0, 0] < -2.0 and abs(sa[4, 0, 0]) < 0.1      # наклон: падение против застоя
    assert fa[5, 0, 0] < -1.5 and abs(sa[5, 0, 0]) < 0.7      # t-статистика
    assert sa[7, 0, 0] < -0.8 and fa[7, 0, 0] > 0.8           # доля улучшений
    assert sa[8, 0, 0] < fa[8, 0, 0] and sa[9, 0, 0] < fa[9, 0, 0]
    # плохие входы не роняют состояние
    b = O.tele_state(1e-2, 4e-3, 6e-3, curve_steps=[0, 1, 2], curve_loss=[np.nan, 1.0, 0.5],
                     grad_norm=float("nan"), rel_disp=None)
    assert np.isfinite(b).all()
    print(f"tele_state: форма {s.shape}, конечно, в [-3, 3]; застой и падение различимы: OK")


def test_modes_and_channels():
    assert "tele" in O.STATE_MODES and O.state_channels("tele") == O.TELE_CH == len(O.TELE_NAMES)
    assert O.state_channels("loss") == 4 and O.state_channels("full") == 4
    try:
        O.apply_state_mode(np.zeros((2, 4, 26, 26), np.float32), "tele")
        raise AssertionError("режим tele для массивов карт должен быть ошибкой")
    except ValueError:
        pass
    try:
        import online_eval_env as E
    except Exception as e:  # noqa: BLE001 — среда без deepxde: проверяем только offline_rl
        print(f"online_eval_env не импортирован ({type(e).__name__}) — частота записи не проверена")
        return
    assert E.tele_display_every(100, 100) == 10 and E.tele_display_every(2500, 100) == 100
    assert E.tele_display_every(1000, 100) == 50 and E.tele_display_every(1000, 20) == 20
    import torch
    net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    th = E.flat_params(net)
    assert th.shape == (2 * 3 + 3 + 3 + 1,)
    with torch.no_grad():
        net[0].weight.add_(1.0)
    assert float((E.flat_params(net) - th).norm()) > 0       # копия, а не ссылка
    print("режим tele: каналы, отказ для карт, частота записи кривой: OK")


if __name__ == "__main__":
    test_curve_stats()
    test_tele_state()
    test_modes_and_channels()
    print("ВСЕ ПРОВЕРКИ РЕЖИМА TELE ПРОЙДЕНЫ")
