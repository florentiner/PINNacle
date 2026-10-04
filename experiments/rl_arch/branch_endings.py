#!/usr/bin/env python
"""
Ветвление концовок от одного снимка (трек 3, 5 октября): каркас идёт до точки решения, веса
PINN сохраняются, затем от того же снимка на том же сиде пробуются K концовок до бюджета.
Сравнение концовок парное: начальные веса, сид и префикс у всех одинаковы, поэтому шум сида
(на pb2d 0.2 декады на прогон) в разницу концовок не входит.

Зачем. По буферам 22 УрЧП итог цепочки на 53–60% решают два последних действия и не более
чем на 6% два первых (surrogate_active.py, опыт 3), а данных рядом с лучшими цепочками почти
нет. Ветвление даёт ровно эти данные: (1) запас адаптации в концовке — насколько оракул по
сиду среди концовок лучше лучшей одной концовки; (2) обучающие пары «состояние в точке
решения → лучшая концовка» для политики концовки агента.

Каждая концовка заливается отдельной строкой с тегом <tag>_e<k> (сиды те же), поэтому
обычные отчёты (posthoc.py, camp_report.py) видят их как армы с парными сидами; общая строка
<tag> хранит все концовки и префикс.

    python experiments/rl_arch/branch_endings.py --pde poissonboltzmann2d --seeds 42,43,44 \\
        --prefix Adam:0.01:1000,Adam:0.0001:2500 \\
        --endings "LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:500;LBFGS:1:500,..." \\
        --keep-opt --keep-opt-mode safe --tag t3br_sk2 --hours 4 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("DDEBACKEND", "pytorch")

import online_eval_env as E  # noqa: E402

_TOKENS = ("HF_TOKEN_WRITE", "HF_TOKEN")


def _hide_tokens():
    """Внутренние прогоны пишут чекпоинты только локально: без токена save_ckpt не заливает,
    иначе каждое действие было бы коммитом в HF (лимит 128 в час на репозиторий)."""
    saved = {k: os.environ.pop(k) for k in _TOKENS if k in os.environ}
    return saved


def _restore_tokens(saved):
    os.environ.update(saved)


def env_args(a, script, ckpt_every):
    argv = ["--policy", "script", "--script", script, "--script-tail", "stop", "--no-state",
            "--pde", a.pde, "--budget", str(a.budget), "--save-dir", a.save_dir, "--tag", a.tag,
            "--hidden-layers", a.hidden_layers, "--keep-opt-mode", a.keep_opt_mode,
            "--ckpt-every", str(ckpt_every), "--ckpt-min-interval", "0", "--resume"]
    argv += ["--keep-opt"] if a.keep_opt else []
    argv += ["--plain-fnn"] if a.plain_fnn else []
    argv += ["--lbfgs-tol", str(a.lbfgs_tol)] if a.lbfgs_tol is not None else []
    if a.smoke:
        argv += ["--smoke"]
    return E.build_parser().parse_args(argv)


def run_prefix(a, seed):
    """Префикс до точки решения; возвращает (имя локального чекпоинта, строка префикса)."""
    name = f"{a.pde}_{a.tag}_pre_seed{seed}"
    local = f"ckpt_{name}.pt"
    if os.path.exists(local):
        os.remove(local)                 # префикс всегда считается заново: чекпоинт должен быть его концом
    args = env_args(a, a.prefix, ckpt_every=1)
    args._ckpt_name = name
    saved = _hide_tokens()
    try:
        row = E.run_seed(seed, args)
    finally:
        _restore_tokens(saved)
    if not os.path.exists(local):
        raise RuntimeError(f"после префикса нет чекпоинта {local}")
    return local, row


def run_ending(a, seed, k, ending, pre_local):
    name = f"{a.pde}_{a.tag}_b{k}_seed{seed}"
    local = f"ckpt_{name}.pt"
    shutil.copyfile(pre_local, local)
    args = env_args(a, f"{a.prefix},{ending}", ckpt_every=0)
    args._ckpt_name = name
    saved = _hide_tokens()
    t = time.time()
    try:
        row = E.run_seed(seed, args)
    finally:
        _restore_tokens(saved)
        if os.path.exists(local):
            os.remove(local)
    row["elapsed_branch_s"] = round(time.time() - t, 1)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--prefix", required=True, help="цепочка до точки решения")
    ap.add_argument("--endings", required=True,
                    help="концовки через «;», каждая — цепочка через запятую; первая — концовка каркаса")
    ap.add_argument("--budget", type=int, default=7000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--keep-opt", action="store_true")
    ap.add_argument("--keep-opt-mode", default="all", choices=["all", "safe"])
    ap.add_argument("--lbfgs-tol", type=float, default=None)
    ap.add_argument("--plain-fnn", action="store_true")
    ap.add_argument("--hours", type=float, default=0.0, help="лимит процесса: следующий сид не начинать")
    ap.add_argument("--resume", action="store_true", help="пропускать сиды, у которых все концовки уже в HF")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    endings = [e.strip() for e in a.endings.split(";") if e.strip()]
    E.parse_chain(a.prefix)
    for e in endings:
        E.parse_chain(e)
    t0 = time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        names = [f"{a.pde}_{a.tag}_e{k}_seed{seed}" for k in range(len(endings))]
        if a.resume and not a.smoke and all(E.result_done(n) for n in names):
            print(f"[seed {seed}] все концовки уже в HF — пропуск", flush=True)
            continue
        if a.hours and (time.time() - t0) / 3600.0 >= a.hours:
            print(f"лимит процесса {a.hours} ч: сид {seed} и дальше — следующему кернелу с --resume", flush=True)
            break
        pre_local, pre = run_prefix(a, seed)
        print(f"[seed {seed}] префикс: spent={pre.get('spent')} l2re={pre.get('l2re'):.4e}", flush=True)
        rows = []
        for k, ending in enumerate(endings):
            r = run_ending(a, seed, k, ending, pre_local)
            r.update(branch_prefix=a.prefix, branch_ending=ending, branch_k=k,
                     branch_epoch=int(pre.get("spent", 0)), prefix_l2re=pre.get("l2re"), smoke=a.smoke)
            print(f"[seed {seed}] концовка {k}: l2re={r['l2re']:.4e} ({ending})", flush=True)
            rows.append(r)
            if not a.smoke:
                E.upload(r, names[k])
        os.remove(pre_local)
        best = min(range(len(rows)), key=lambda i: rows[i]["l2re"])
        summary = dict(seed=seed, pde=a.pde, policy="branch", prefix=a.prefix, endings=endings,
                       branch_epoch=int(pre.get("spent", 0)), prefix_l2re=pre.get("l2re"),
                       l2re=rows[0]["l2re"], best_k=best, best_l2re=rows[best]["l2re"],
                       branches=[dict(k=i, ending=endings[i], l2re=r["l2re"], loss_final=r.get("loss_final"),
                                      hist=r.get("hist"), elapsed_s=r.get("elapsed_branch_s"))
                                 for i, r in enumerate(rows)],
                       budget=a.budget, smoke=a.smoke, elapsed_s=round(time.time() - t0, 1))
        print(json.dumps({k: v for k, v in summary.items() if k != "branches"}), flush=True)
        if not a.smoke:
            E.upload(summary, f"{a.pde}_{a.tag}_seed{seed}")


if __name__ == "__main__":
    main()
