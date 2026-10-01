#!/usr/bin/env python
"""
Проверка загрузчика буфера с восстановлением цепочек (offline_rl.episodes_to_arrays_chains)
на синтетическом буфере, повторяющем оба формата дампов авторов:

  * старый: стартовое состояние — нули, next_state верный, ключа delta в state нет;
  * новый:  ключ delta есть, next_state записан КОПИЕЙ state, старт не нулевой.

Запуск (сеть и GPU не нужны):
    python experiments/rl_arch/tests/test_chain_loader.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import offline_rl as O  # noqa: E402

RNG = np.random.default_rng(0)
KEYS = ["loss_total", "loss_oper", "loss_bnd"]


def maps(v=None):
    return {k: (np.zeros((26, 26), np.float32) if v == 0 else
                RNG.normal(size=(26, 26)).astype(np.float32)) for k in KEYS}


def make_file(new_format, chains):
    """chains: список (ошибки после каждого действия, флаг конца: 1 / -1 / 0)."""
    out = []
    for errs, end in chains:
        states = [maps(0) if not new_format else maps()] + [maps() for _ in errs]
        for j, e in enumerate(errs):
            last = j == len(errs) - 1
            done = end if last else 0
            st = dict(states[j])
            nxt = dict(states[j] if new_format else states[j + 1])   # новый формат: копия
            if new_format:
                st["delta"] = np.zeros((26, 26), np.float32)
            nxt["delta"] = np.zeros((26, 26), np.float32)
            out.append(dict(state=st, next_state=nxt,
                            action=(1, {"lr": 1, "epochs": 2}),          # LBFGS 0.5 x 1000
                            reward=-(e + (1.0 if done == -1 else 0.0)),  # уровень ошибки и штраф
                            reward_model=0.123, done=done))
    return out


def main():
    old_file = make_file(False, [([0.4, 0.2, 0.1], -1), ([0.3, 0.008], 1), ([0.5, 0.25], 0)])
    new_file = make_file(True, [([0.4, 0.3, 0.05], -1), ([0.2, 0.1], 0)])
    eps = [old_file, new_file]

    legacy = O.episodes_to_arrays(eps)
    assert -1.0 in set(legacy["D"].tolist()), "в прежнем загрузчике done=-1 должен попадать в D"
    assert len(np.unique(legacy["EP"])) == 2, "прежний загрузчик: эпизод = файл"

    d = O.episodes_to_arrays_chains(eps, reward_form="delta", verbose=False)
    # 5 цепочек; хвост нового формата (2 перехода) теряет последний переход
    assert len(np.unique(d["EP"])) == 5, len(np.unique(d["EP"]))
    assert len(d["A"]) == 3 + 2 + 2 + 3 + 1, len(d["A"])
    assert set(np.unique(d["D"]).tolist()) <= {0.0, 1.0}
    assert int(d["D"].sum()) == 3, "терминальны концы трёх завершённых цепочек"
    assert np.all(d["S"][d["FIRST"] == 1] == 0), "старт цепочки — нулевое состояние"
    assert np.all(d["S"][d["STEP"] == 1][:, 3] == 0), "delta после первого действия нулевой"
    # следующее состояние = состояние следующего перехода той же цепочки
    for i in range(len(d["A"]) - 1):
        if d["D"][i] == 0 and d["EP"][i] == d["EP"][i + 1]:
            assert np.array_equal(d["S2"][i], d["S"][i + 1]), i
    # награда-разность: первый шаг 0, сумма по цепочке = E_1 - E_конец
    c0 = np.where(d["EP"] == d["EP"][0])[0]
    assert abs(d["R"][c0[0]]) < 1e-9
    assert abs(d["R"][c0].sum() - (0.4 - 0.1)) < 1e-6
    # штраф -1 за неудачу снят с ошибки
    assert abs(d["ERR"][c0[-1]] - 0.1) < 1e-6

    d2 = O.episodes_to_arrays_chains(eps, reward_form="delta", init_err=0.5, verbose=False)
    assert abs(d2["R"][c0].sum() - (0.5 - 0.1)) < 1e-6, "с init_err возврат = init_err - E_конец"

    lv = O.episodes_to_arrays_chains(eps, reward_form="level", verbose=False)
    assert abs(lv["R"][c0[-1]] + 0.1) < 1e-6
    lg = O.episodes_to_arrays_chains(eps, reward_form="logged", verbose=False)
    assert abs(lg["R"][c0[-1]] + 1.1) < 1e-6
    md = O.episodes_to_arrays_chains(eps, reward_form="model", verbose=False)
    assert abs(md["R"][0] - 0.123) < 1e-6

    # бюджет 2000 эпох при действиях по 1000: вторая цепочка-действие терминальна
    b = O.episodes_to_arrays_chains(eps, reward_form="delta", budget=2000, verbose=False)
    for c in np.unique(b["EP"]):
        idx = np.where(b["EP"] == c)[0]
        assert len(idx) <= 2, "цепочка обрезана по бюджету"
        if len(idx) == 2:
            assert b["D"][idx[-1]] == 1.0

    # RTG считается внутри цепочки
    g = O.GAMMA
    assert abs(d["RTG"][c0[0]] - (d["R"][c0[0]] + g * d["R"][c0[1]] + g * g * d["R"][c0[2]])) < 1e-6

    # прежнее поведение без chain_fix не изменилось
    same = O.episodes_to_arrays(eps, fix_next_state=False, chain_fix=False)
    for k in ("S", "A", "R", "S2", "D", "EP", "RTG", "FIRST"):
        assert np.array_equal(same[k], legacy[k]), k
    try:
        O.episodes_to_arrays(eps, reward_form="delta")
        raise AssertionError("reward_form без chain_fix должна отклоняться")
    except ValueError:
        pass
    print("test_chain_loader: все проверки пройдены")


if __name__ == "__main__":
    main()
