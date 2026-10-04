#!/usr/bin/env python
"""
Генетический поиск по статическим цепочкам оптимизаторов: простая альтернатива агенту при
равных GPU-часах (paper/lit_review/COVERAGE.md, «что идёт в код», п. 1).

Геном: список индексов из 27 действий агента (ACTION_TABLE, маска --mask как у среды). Среда
исполняет цепочку как --policy script --script-tail stop --no-state: действия идут по порядку,
пока не кончится бюджет эпох (последнее среда обрезает), гены за бюджетом спят. Приспособленность:
итоговая l2re на ОДНОМ отсеивающем сиде, общем для всех особей поколения (сравнение внутри
поколения честное) и новом в каждом поколении, чтобы отбор не подстраивался под шум сида;
элита копируется без изменений, но переоценивается на новом сиде. Неудачи получают штраф PENALTY.

Учёт честности: <save-dir>/<tag>_ga.json после каждого поколения хранит число прогонов среды и
секунды накопительно (с докатками), для сравнения с GPU-часами мета-обучения агента; та же строка
заливается в HF под именем <pde>_<tag>_ga (без токена только печатается). --resume продолжает с
сохранённого поколения (локальный файл, иначе HF). В конце top3 цепочек для переоценки задачей
--policy script на сидах 42-51.

    python experiments/rl_arch/ga_chains.py --pde burgers_1d --mask pso --tag t3g_ga --hours 10.5 --resume
    python experiments/rl_arch/ga_chains.py --pde burgers_1d --smoke --pop 2 --gens 1 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("DDEBACKEND", "pytorch")

from online_eval_env import (ACTION_TABLE, OUT_REPO, build_parser, parse_chain,  # noqa: E402
                             parse_mask, run_seed, upload)

N_AGENT = 27            # геном живёт в пространстве действий агента, SOAP и прочие надстройки не входят
PENALTY = 1e3           # приспособленность упавшей особи (l2re необученной сети порядка 1)
P_MUTATE = 0.8
TOURNAMENT = 3
MUTATIONS = ("replace", "lr", "epochs", "insert", "delete")


def action_str(a):
    return "%s:%g:%d" % ACTION_TABLE[a]


def effective(genome, budget):
    """Гены, которые успевают начаться до исчерпания бюджета (как цикл while spent < budget в run_seed)."""
    out, spent = [], 0
    for a in genome:
        if spent >= budget:
            break
        out.append(int(a))
        spent += ACTION_TABLE[a][2]
    return out


def chain_str(genome, budget):
    return ",".join(action_str(a) for a in effective(genome, budget))


def random_genome(rng, allowed, budget, max_len):
    """Случайные действия, пока не покрыт бюджет (поколение 0 равно случайному поиску)."""
    g, spent = [], 0
    while spent < budget and len(g) < max_len:
        g.append(int(rng.choice(allowed)))
        spent += ACTION_TABLE[g[-1]][2]
    return g


def mutate(genome, rng, allowed, max_len):
    """Одна мутация: замена действия, другой шаг или другая длина в том же семействе, вставка, удаление."""
    g = list(genome)
    kind = MUTATIONS[int(rng.integers(len(MUTATIONS)))]
    i = int(rng.integers(len(g)))
    if kind in ("lr", "epochs"):
        o, lr, ep = ACTION_TABLE[g[i]]
        cand = [a for a in allowed if a != g[i] and ACTION_TABLE[a][0] == o
                and (ACTION_TABLE[a][2] == ep if kind == "lr" else ACTION_TABLE[a][1] == lr)]
        if cand:
            g[i] = int(rng.choice(cand))
            return g
        kind = "replace"            # в семействе под маской больше нечего менять
    if kind == "replace":
        cand = [a for a in allowed if a != g[i]]
        if cand:
            g[i] = int(rng.choice(cand))
    elif kind == "insert" and len(g) < max_len:
        g.insert(int(rng.integers(len(g) + 1)), int(rng.choice(allowed)))
    elif kind == "delete" and len(g) > 1:
        del g[i]
    return g


def crossover(p1, p2, rng, max_len):
    """Одноточечное скрещивание геномов разной длины: своя точка разреза у каждого родителя."""
    c1, c2 = int(rng.integers(1, len(p1) + 1)), int(rng.integers(0, len(p2) + 1))
    return (list(p1[:c1]) + list(p2[c2:]))[:max_len]


def tournament(pop, fit, rng, k=TOURNAMENT):
    idx = [int(j) for j in rng.integers(len(pop), size=k)]
    return pop[min(idx, key=lambda j: fit[j])]


def next_generation(pop, fit, rng, allowed, max_len, elite, n):
    """Элита без изменений, остальное от турнирного отбора, скрещивания и мутации; дубликаты геномов
    отсеиваются (разные геномы с одной цепочкой ловит кэш прогонов)."""
    order = sorted(range(len(pop)), key=lambda j: fit[j])
    new = [list(pop[j]) for j in order[:elite]]
    seen = {tuple(g) for g in new}
    while len(new) < n:
        for _ in range(8):
            child = crossover(tournament(pop, fit, rng), tournament(pop, fit, rng), rng, max_len)
            if rng.random() < P_MUTATE:
                child = mutate(child, rng, allowed, max_len)
            if tuple(child) not in seen:
                break
        seen.add(tuple(child))
        new.append(child)
    return new


def genome_from_spec(spec, allowed):
    g = parse_chain(spec)
    bad = [a for a in g if a >= N_AGENT or a not in allowed]
    if bad or not g:
        sys.exit(f"--init {spec!r}: действия {bad} вне пространства агента или под маской")
    return g


def parse_pool(spec):
    out = []
    for tok in spec.split(","):
        lo, _, hi = tok.strip().partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def env_args(a, chain):
    """Namespace для run_seed: те же флаги и умолчания, что у online_eval_env.py."""
    argv = ["--policy", "script", "--script", chain, "--script-tail", "stop", "--no-state",
            "--pde", a.pde, "--budget", str(a.budget), "--save-dir", a.save_dir, "--tag", a.tag,
            "--hidden-layers", a.hidden_layers, "--keep-opt-mode", a.keep_opt_mode]
    argv += ["--keep-opt"] if a.keep_opt else []
    argv += ["--plain-fnn"] if a.plain_fnn else []
    argv += ["--lbfgs-tol", str(a.lbfgs_tol)] if a.lbfgs_tol is not None else []
    return build_parser().parse_args(argv)


def evaluate(chain, seed, a):
    """Один прогон среды: (приспособленность, запись для evals). Исключение одной особи не валит поиск."""
    t = time.time()
    try:
        row = run_seed(seed, env_args(a, chain))
    except Exception as e:
        traceback.print_exc()
        row = dict(l2re=float("nan"), error=f"{type(e).__name__}: {e}"[:200])
    l2re = float(row.get("l2re", float("nan")))
    ok = math.isfinite(l2re)
    return (l2re if ok else PENALTY), dict(seed=int(seed), l2re=(l2re if ok else None),
                                           spent=int(row.get("spent", 0)), stop=row.get("stop_reason"),
                                           error=row.get("error"), elapsed_s=round(time.time() - t, 1))


def top_chains(evals, k):
    """Лучшие цепочки по среднему log10 l2re на всех своих отсеивающих сидах (при равенстве больше сидов)."""
    rows = []
    for chain, recs in evals.items():
        vals = [math.log10(r["l2re"] if r["l2re"] is not None and r["l2re"] > 0 else PENALTY) for r in recs]
        rows.append(dict(chain=chain, log10_l2re=round(float(np.mean(vals)), 4), n=len(recs),
                         seeds=[r["seed"] for r in recs]))
    rows.sort(key=lambda r: (r["log10_l2re"], -r["n"]))
    return rows[:k]


def load_state(path, name, smoke):
    if os.path.exists(path):
        return json.load(open(path))
    if smoke:
        return None
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(OUT_REPO, f"rl_arch/online_env/{name}.json", repo_type="dataset")))
    except Exception as e:
        print(f"сохранённого поиска нет ({type(e).__name__}): старт с нуля", flush=True)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pde", required=True)
    ap.add_argument("--budget", type=int, default=7000)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="лимит GPU-часов всего поиска, накопительно через докатки: закончить особь, записать "
                         "top3 и выйти (0 без лимита)")
    ap.add_argument("--kernel-hours", type=float, default=0.0,
                    help="мягкий лимит одного процесса (сессия Kaggle): сохранить и выйти, продолжит --resume")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=6, help="число поколений, включая начальное")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--seed-pool", default="1000-1099", help="отсеивающие сиды (список или диапазон), один на поколение")
    ap.add_argument("--ga-seed", type=int, default=0, help="сид генетических операторов и выбора сидов")
    ap.add_argument("--mask", default="", help="запрещённые действия, синтаксис среды: 'pso,adam:0.01'")
    ap.add_argument("--init", action="append", default=[],
                    help="цепочки начальной популяции, несколько через ';' или повтором флага")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--keep-opt", action="store_true")
    ap.add_argument("--keep-opt-mode", default="all", choices=["all", "safe"])
    ap.add_argument("--lbfgs-tol", type=float, default=None)
    ap.add_argument("--plain-fnn", action="store_true")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=float, default=1800.0,
                    help="заливать состояние в HF не чаще, чем раз в столько секунд, кроме границ поколений")
    ap.add_argument("--smoke", action="store_true", help="бюджет 300 (если не задан), без заливки в HF")
    a = ap.parse_args(argv)
    given = {x.split("=")[0] for x in (sys.argv[1:] if argv is None else argv) if x.startswith("--")}
    if a.smoke and "--budget" not in given:
        a.budget = 300
    if a.pop < 2 or a.max_len < 1 or a.elite < 0:
        sys.exit("нужно --pop >= 2, --max-len >= 1, --elite >= 0")
    a.elite = min(a.elite, a.pop - 1)       # хотя бы один потомок в поколении
    allowed = np.where(parse_mask(a.mask))[0].tolist()
    gen_seeds = np.random.default_rng(a.ga_seed).permutation(parse_pool(a.seed_pool))
    name = f"{a.pde}_{a.tag}_ga"
    os.makedirs(a.save_dir, exist_ok=True)
    path = os.path.join(a.save_dir, f"{a.tag}_ga.json")

    st = load_state(path, name, a.smoke) if a.resume else None
    if st is None:
        rng = np.random.default_rng([a.ga_seed, 0])
        init = [genome_from_spec(s, allowed) for v in a.init for s in v.split(";") if s.strip()]
        pop = init[:a.pop] + [random_genome(rng, allowed, a.budget, a.max_len) for _ in range(a.pop - len(init))]
        st = dict(pde=a.pde, tag=a.tag, budget=a.budget, mask=a.mask, keep_opt=a.keep_opt,
                  keep_opt_mode=a.keep_opt_mode, lbfgs_tol=a.lbfgs_tol, pop=a.pop, elite=a.elite,
                  max_len=a.max_len, ga_seed=a.ga_seed, seed_pool=a.seed_pool, gen=0,
                  gen_seed=int(gen_seeds[0]), population=pop, fitness=[None] * len(pop), evals={},
                  history=[], n_evals=0, eval_s=0.0, wall_s=0.0, done=False, partial=True, smoke=a.smoke)
    elif st.get("done") and st["gen"] + 1 >= a.gens:
        print(f"поиск {name} уже завершён (поколений {st['gen'] + 1}), top3: {st.get('top3')}", flush=True)
        return st
    else:
        print(f"докатка {name}: поколение {st['gen']}, прогонов {st['n_evals']}, {st['wall_s'] / 3600:.2f} ч", flush=True)
        st["done"], st["partial"] = False, True      # завершённый поиск с большим --gens продолжается
    t0, t_proc, last_up = time.time() - st["wall_s"], time.time(), [time.time()]

    def save(up=False):
        st["wall_s"] = round(time.time() - t0, 1)
        with open(path, "w") as f:
            json.dump(st, f, indent=1)
        if not a.smoke and (up or time.time() - last_up[0] >= a.progress_every):
            last_up[0] = time.time()
            upload(st, name)

    over_total = False
    while True:
        g, seed = st["gen"], st["gen_seed"]
        for i, genome in enumerate(st["population"]):
            if st["fitness"][i] is not None:
                continue
            chain = chain_str(genome, a.budget)
            hit = next((r for r in st["evals"].get(chain, []) if r["seed"] == seed), None)
            if hit is None:
                fit, hit = evaluate(chain, seed, a)
                st["evals"].setdefault(chain, []).append(hit)
                st["n_evals"], st["eval_s"] = st["n_evals"] + 1, round(st["eval_s"] + hit["elapsed_s"], 1)
            else:
                fit = hit["l2re"] if hit["l2re"] is not None else PENALTY   # та же цепочка уже считана
            st["fitness"][i] = fit
            print(f"[ga gen {g} seed {seed}] особь {i + 1}/{len(st['population'])}: {chain} -> l2re={fit:.4e} "
                  f"({hit['elapsed_s']}s; всего прогонов {st['n_evals']}, {st['eval_s'] / 3600:.2f} ч)", flush=True)
            save()
            over_total = bool(a.hours and (time.time() - t0) / 3600.0 >= a.hours)
            if over_total:
                break
            if a.kernel_hours and (time.time() - t_proc) / 3600.0 >= a.kernel_hours:
                save(up=True)
                print(f"лимит кернела ({a.kernel_hours} ч): состояние сохранено, продолжит --resume", flush=True)
                return st
        if all(f is not None for f in st["fitness"]) and len(st["history"]) <= g:
            fit = st["fitness"]
            j = int(np.argmin(fit))
            st["history"].append(dict(gen=g, seed=seed, best=fit[j], best_chain=chain_str(st["population"][j], a.budget),
                                      mean_log10=round(float(np.mean(np.log10(np.maximum(fit, 1e-12)))), 4),
                                      cum_evals=st["n_evals"], cum_eval_s=st["eval_s"], wall_s=round(time.time() - t0, 1)))
            st["best"] = top_chains(st["evals"], 3)
            print(f"[ga gen {g}] лучшая l2re={fit[j]:.4e}: {st['history'][-1]['best_chain']}", flush=True)
        if over_total or g + 1 >= a.gens:
            break
        rng = np.random.default_rng([a.ga_seed, g + 1])
        st["population"] = next_generation(st["population"], st["fitness"], rng, allowed, a.max_len, a.elite, a.pop)
        st["fitness"] = [None] * len(st["population"])
        st["gen"], st["gen_seed"] = g + 1, int(gen_seeds[(g + 1) % len(gen_seeds)])
        save(up=True)

    st["top3"], st["done"], st["partial"] = top_chains(st["evals"], 3), True, False
    st["stop_reason"] = "hours" if over_total else "gens"
    print(f"поиск завершён ({st['stop_reason']}): прогонов {st['n_evals']}, GPU {st['eval_s'] / 3600:.2f} ч, "
          f"стена {(time.time() - t0) / 3600:.2f} ч. top3:", flush=True)
    for r in st["top3"]:
        print(f"  log10 l2re={r['log10_l2re']:+.3f} (сидов {r['n']}): {r['chain']}", flush=True)
    save(up=True)
    return st


if __name__ == "__main__":
    main()
