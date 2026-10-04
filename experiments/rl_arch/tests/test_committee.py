#!/usr/bin/env python
"""
Комитет с проводником и воздержанием (online_eval_env.committee_with_guide), без GPU:
Q агентов подменяются заранее заданными векторами.

    DDEBACKEND=pytorch python experiments/rl_arch/tests/test_committee.py
"""
import os
import sys

os.environ.setdefault("DDEBACKEND", "pytorch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402

import online_eval_env as E  # noqa: E402


def _fake(qs):
    """agents = [(индекс,)...]; q_values возвращает заранее заданный вектор агента."""
    def q_values(ag, state, mean, std, variant):
        return np.array(qs[ag], dtype=np.float64)
    E.q_values = q_values
    return [(i, None, None, None) for i in range(len(qs))]


def test_follow_when_disagree():
    n = 27
    q0 = np.zeros(n); q0[3] = 5.0          # агент 0 за действие 3
    q1 = np.zeros(n); q1[7] = 5.0          # агент 1 за действие 7
    agents = _fake([q0, q1])
    allowed = np.ones(n, bool)
    a, dev = E.committee_with_guide(agents, None, allowed, guide_a=10, k_agree=2, bonus=0.0)
    assert (a, dev) == (10, False), (a, dev)      # нет согласия — проводник


def test_deviate_when_agree_and_margin():
    n = 27
    q0 = np.zeros(n); q0[3] = 5.0
    q1 = np.zeros(n); q1[3] = 4.0
    agents = _fake([q0, q1])
    allowed = np.ones(n, bool)
    a, dev = E.committee_with_guide(agents, None, allowed, guide_a=10, k_agree=2, bonus=1.0)
    assert (a, dev) == (3, True), (a, dev)
    # слишком большой запас — остаёмся с проводником
    a, dev = E.committee_with_guide(agents, None, allowed, guide_a=10, k_agree=2, bonus=100.0)
    assert (a, dev) == (10, False), (a, dev)


def test_mask_and_no_guide():
    n = 27
    q0 = np.zeros(n); q0[3] = 5.0; q0[5] = 4.0
    q1 = np.zeros(n); q1[3] = 5.0
    agents = _fake([q0, q1])
    allowed = np.ones(n, bool); allowed[3] = False       # лучшее действие запрещено маской
    a, dev = E.committee_with_guide(agents, None, allowed, guide_a=None, k_agree=2, bonus=0.0)
    assert dev and a != 3, (a, dev)                    # без проводника — голосование по маске
    a, dev = E.committee_with_guide(agents, None, allowed, guide_a=3, k_agree=2, bonus=0.0)
    assert dev and a != 3, (a, dev)                    # проводник вне маски — голосование


def test_fit_channels():
    st = np.zeros((9, 26, 26), np.float32)
    assert E.fit_channels(st, np.zeros((1, 4, 1, 1))).shape == (4, 26, 26)     # агент без контекста
    assert E.fit_channels(st, np.zeros((1, 9, 1, 1))).shape == (9, 26, 26)     # агент с контекстом
    assert E.fit_channels(st, 0.0).shape == (9, 26, 26)                        # скалярная нормировка


if __name__ == "__main__":
    test_fit_channels()
    test_follow_when_disagree()
    test_deviate_when_agree_and_margin()
    test_mask_and_no_guide()
    print("test_committee: ok")
