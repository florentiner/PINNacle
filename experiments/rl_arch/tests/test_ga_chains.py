#!/usr/bin/env python
"""
Проверки генетического поиска по цепочкам (ga_chains.py) без GPU и без сети:

  * декодирование генома в цепочку с обрезкой по бюджету, разбор обратно средой;
  * мутация, скрещивание и случайный геном детерминированы при одном сиде и уважают маску;
  * штраф за нечисловую ошибку и за исключение в прогоне;
  * сквозной дымовой прогон с подменой run_seed (сеть не обучается): файл состояния, докатка
    с сохранённого поколения и посреди поколения, учёт прогонов и top3.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_ga_chains.py
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import zlib

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402

import ga_chains as G  # noqa: E402
import online_eval_env as E  # noqa: E402

ALL = list(range(27))
NO_PSO = np.where(E.parse_mask("pso"))[0].tolist()


def test_decode():
    g = [4, 10, 4, 10]                      # Adam 1e-3 x1000, LBFGS 1 x500, ...
    assert G.effective(g, 7000) == g
    assert G.effective(g, 1500) == [4, 10]            # третий ген начинался бы при spent=1500: спит
    assert G.effective(g, 1400) == [4, 10]            # второй начинается при 1000 < 1400, среда обрежет его
    assert G.effective(g, 100) == [4]
    s = G.chain_str(g, 7000)
    assert s == "Adam:0.001:1000,LBFGS:1:500,Adam:0.001:1000,LBFGS:1:500", s
    assert E.parse_chain(s) == g                      # строка разбирается средой обратно в те же индексы
    for a in ALL:                                     # каждое действие проходит туда и обратно
        assert E.parse_chain(G.action_str(a)) == [a]
    assert G.parse_pool("1000-1003,7") == [1000, 1001, 1002, 1003, 7]
    print("декодирование: OK")


def test_operators_deterministic_and_masked():
    for allowed in (ALL, NO_PSO):
        outs = []
        for _ in range(2):
            rng = np.random.default_rng(5)
            p1 = G.random_genome(rng, allowed, 7000, 12)
            p2 = G.random_genome(rng, allowed, 7000, 12)
            kids = [G.mutate(G.crossover(p1, p2, rng, 12), rng, allowed, 12) for _ in range(40)]
            outs.append((p1, p2, kids))
        assert outs[0] == outs[1], "операторы при одном сиде должны давать одно и то же"
        p1, p2, kids = outs[0]
        assert 1 <= len(p1) <= 12
        assert sum(E.ACTION_TABLE[a][2] for a in p1) >= 7000 or len(p1) == 12, "случайный геном покрывает бюджет"
        for k in kids:
            assert 1 <= len(k) <= 12
            assert all(a in allowed for a in k), "мутация вышла за маску"
        if allowed is NO_PSO:
            assert not any(a >= 18 for k in kids for a in k)
    rng = np.random.default_rng(11)
    saved = G.MUTATIONS
    try:
        G.MUTATIONS = ("lr",)                 # другой шаг: семейство и длина те же
        for _ in range(30):
            k = G.mutate([4], rng, NO_PSO, 12)
            assert len(k) == 1 and k[0] != 4 and E.ACTION_TABLE[k[0]][0] == "Adam" and E.ACTION_TABLE[k[0]][2] == 1000
        G.MUTATIONS = ("epochs",)             # другая длина: семейство и шаг те же
        for _ in range(30):
            k = G.mutate([10], rng, NO_PSO, 12)
            assert k[0] != 10 and E.ACTION_TABLE[k[0]][:2] == ("LBFGS", 1.0)
        # под маской у Adam 1e-2 x100 нет соседей по длине: откат к замене, маска соблюдена
        al = np.where(E.parse_mask("adam:0.01:1000,adam:0.01:2500,pso"))[0].tolist()
        for _ in range(30):
            k = G.mutate([0], rng, al, 12)
            assert k[0] in al and k[0] != 0
        G.MUTATIONS = ("insert",)             # длина не растёт выше max_len
        assert len(G.mutate([4, 4], rng, ALL, 2)) == 2
        G.MUTATIONS = ("delete",)             # и не падает ниже 1
        assert G.mutate([4], rng, ALL, 12) == [4]
    finally:
        G.MUTATIONS = saved
    # следующее поколение: элита впереди без изменений, размер популяции сохранён
    pop = [[4, 13], [13, 13], [0, 4], [12]]
    fit = [0.3, 0.1, 0.5, 0.2]
    new = G.next_generation(pop, fit, np.random.default_rng(1), ALL, 12, 2, 6)
    assert new[0] == [13, 13] and new[1] == [12] and len(new) == 6
    assert len({tuple(g) for g in new}) == 6, "дубликаты геномов отсеяны"
    assert G.tournament(pop, fit, np.random.default_rng(2)) in pop
    print("операторы и маска: OK")


def fake_l2re(chain):
    """Детерминированная «ошибка» цепочки, одинаковая между процессами (hash() строк солёный)."""
    return 0.01 * (1 + zlib.crc32(chain.encode()) % 50)


def _fake_run_seed(seed, args, progress_cb=None):
    assert args.policy == "script" and args.script_tail == "stop" and args.no_state
    assert args.pde == "burgers_1d" and args.budget == 300 and not args.keep_opt
    acts = E.parse_chain(args.script)
    spent = 0
    for a in acts:
        spent += min(E.ACTION_TABLE[a][2], args.budget - spent)
    assert spent <= args.budget
    return dict(seed=seed, l2re=fake_l2re(args.script), spent=spent, stop_reason="budget",
                elapsed_s=0.0, chain=[list(E.ACTION_TABLE[a]) for a in acts])


def test_penalty():
    saved = G.run_seed
    ns = argparse.Namespace(pde="burgers_1d", budget=300, save_dir=".", tag="t", hidden_layers="100*5",
                            keep_opt_mode="all", keep_opt=False, plain_fnn=False, lbfgs_tol=None)
    try:
        G.run_seed = lambda seed, args, progress_cb=None: dict(l2re=float("nan"), spent=0)
        fit, rec = G.evaluate("Adam:0.001:100", 1000, ns)
        assert fit == G.PENALTY and rec["l2re"] is None and rec["seed"] == 1000
        G.run_seed = lambda *a, **k: 1 / 0
        with contextlib.redirect_stderr(io.StringIO()):          # ожидаемая трассировка не нужна в выводе
            fit, rec = G.evaluate("Adam:0.001:100", 1000, ns)
        assert fit == G.PENALTY and "ZeroDivisionError" in rec["error"]
        assert G.top_chains({"x": [rec], "y": [dict(seed=1, l2re=0.1)]}, 3)[0]["chain"] == "y"
        # Namespace среды: флаги прохода как у online_eval_env.py
        ea = G.env_args(argparse.Namespace(**{**vars(ns), "keep_opt": True, "lbfgs_tol": 0.0}), "LBFGS:1:100")
        assert ea.keep_opt and ea.lbfgs_tol == 0.0 and ea.no_state and ea.script_tail == "stop"
    finally:
        G.run_seed = saved
    print("штраф за неудачу: OK")


def test_smoke_and_resume():
    saved = G.run_seed
    G.run_seed = _fake_run_seed
    tmp = tempfile.mkdtemp(prefix="ga_smoke_")
    try:
        base = ["--pde", "burgers_1d", "--budget", "300", "--pop", "2", "--smoke", "--mask", "pso",
                "--tag", "t_ga", "--save-dir", tmp, "--init", "Adam:0.001:100,LBFGS:1:100"]
        G.main(base + ["--gens", "1"])
        path = os.path.join(tmp, "t_ga_ga.json")
        assert os.path.exists(path)
        d = json.load(open(path))
        assert d["done"] and d["gen"] == 0 and d["n_evals"] == 2 and len(d["history"]) == 1
        assert d["population"][0] == [3, 9], "цепочка из --init должна открывать популяцию"
        chains = [G.chain_str(g, 300) for g in d["population"]]
        assert all(f == fake_l2re(c) for f, c in zip(d["fitness"], chains))
        assert 1 <= len(d["top3"]) <= 2 and d["top3"][0]["log10_l2re"] <= d["top3"][-1]["log10_l2re"]
        assert d["history"][0]["cum_evals"] == 2 and d["stop_reason"] == "gens"
        for chain in d["evals"]:
            assert not any(a >= 18 for a in E.parse_chain(chain)), "PSO под маской"     # годится для --policy script
        # докатка с большим числом поколений: поколение 0 не пересчитывается, появляется поколение 1
        st2 = G.main(base + ["--gens", "2", "--resume"])
        d2 = json.load(open(path))
        assert d2["gen"] == 1 and d2["done"] and len(d2["history"]) == 2
        assert d2["history"][0] == d["history"][0]
        assert d2["history"][1]["seed"] != d2["history"][0]["seed"], "сид отсева меняется от поколения к поколению"
        assert 2 <= d2["n_evals"] <= 4 and d2["history"][1]["cum_evals"] == d2["n_evals"]
        assert d2["population"][0] == min(d["population"], key=lambda g: fake_l2re(G.chain_str(g, 300))), "элита"
        assert len(d2["top3"]) <= 3 and st2["top3"] == d2["top3"]
        # завершённый поиск с тем же --gens не трогается
        assert G.main(base + ["--gens", "2", "--resume"]) == d2
        # докатка посреди поколения: у второй особи нет приспособленности и нет записи прогона
        c1 = G.chain_str(d2["population"][1], 300)
        d2["fitness"][1], d2["done"] = None, False
        rest = [r for r in d2["evals"][c1] if r["seed"] != d2["gen_seed"]]
        if rest:
            d2["evals"][c1] = rest
        else:
            d2["evals"].pop(c1)
        json.dump(d2, open(path, "w"))
        n_calls = [0]

        def counting(seed, args, progress_cb=None):
            n_calls[0] += 1
            return _fake_run_seed(seed, args, progress_cb)
        G.run_seed = counting
        d3 = G.main(base + ["--gens", "2", "--resume"])
        assert n_calls[0] == 1 and d3["fitness"][1] == fake_l2re(c1) and d3["done"] and d3["gen"] == 1
    finally:
        G.run_seed = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("дымовой прогон и докатка: OK")


if __name__ == "__main__":
    test_decode()
    test_operators_deterministic_and_masked()
    test_penalty()
    test_smoke_and_resume()
    print("ВСЕ ПРОВЕРКИ ГЕНЕТИЧЕСКОГО ПОИСКА ПРОЙДЕНЫ")
