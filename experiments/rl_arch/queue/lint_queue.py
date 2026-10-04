#!/usr/bin/env python
"""
Проверка очереди без запуска: каждая команда из queue.json прогоняется через
настоящий разбор аргументов своего скрипта и через его проверки совместимости
флагов, а тяжёлая часть (данные, среда, обучение) подменена заглушками.

Ловит опечатки во флагах, недопустимые значения и несовместимые сочетания — то,
что иначе всплыло бы только в логе кернела через несколько минут GPU-времени.
Не проверяет: наличие файлов в HF, содержимое чекпоинтов, саму среду.

    DDEBACKEND=pytorch python experiments/rl_arch/queue/lint_queue.py            # вся очередь
    DDEBACKEND=pytorch python experiments/rl_arch/queue/lint_queue.py --wave 10
    DDEBACKEND=pytorch python experiments/rl_arch/queue/lint_queue.py --queue experiments/rl_arch/queue/queue_pb2d.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RL = os.path.join(ROOT, "experiments", "rl_arch")
for p in (ROOT, RL, os.path.join(RL, "tests"), os.path.join(ROOT, "experiments", "chain_eval")):
    sys.path.insert(0, p)
os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# значения подстановок только для разбора аргументов; настоящие берутся из decisions.json
DUMMY = {
    "BASE": "--variant convnext_dqn --pde ns2d_liddriven --hours 11 --save-agent --rlpd --rlpd-utd 8 "
            "--rlpd-subdir ns2d_liddriven --offline-fix --offline-reward delta --init-err 0.49 --value-bound "
            "--gamma 0.99 --episode-budget 7000 --tolerance 0 --max-chain-steps 70",
    "GUIDE": "Adam:0.001:1000,LBFGS:1:1000",
    "COMMITTEE": "rl_arch/agents_online/a.pt,rl_arch/agents_online/b.pt",
    "BEST_AGENT": "rl_arch/agents_online/a.pt",
    "BUFFERS": "q_a.pt,q_b.pt",
    "RS_CHAIN_LOSS": "Adam:0.001:1000,LBFGS:1:1000",
    "RS_CHAIN_ERR": "Adam:0.0001:2500,LBFGS:0.5:1000",
}
DUMMY["BASE_NOVB"] = DUMMY["BASE"].replace(" --value-bound", "")
DUMMY["RULE_BEST"] = ("--policy rule --rule-burst LBFGS:1:500 --rule-kick Adam:0.0001:100 --rule-max-kicks 3 "
                      "--rule-tol 0.001")
DUMMY["COMBO_A"] = DUMMY["BASE"] + " --scalar-ctx --ctx-no-err --state-mode level"
DUMMY["COMBO_B"] = DUMMY["BASE"] + " --scalar-ctx --ctx-no-err --keep-opt"
DUMMY["COMBO_C"] = DUMMY["BASE"] + " --n-step 3 --guide Adam:0.001:1000,LBFGS:1:1000 --guide-mode bonus"
DUMMY["COMBO_EVAL_A"] = ("--policy agent --model-file rl_arch/agents_online/a.pt --stop-on-noop "
                         "--guide Adam:0.001:1000,LBFGS:1:1000 --guide-bonus 0.02 --guard-rollback 1.0 "
                         "--guard-fallback LBFGS:1:500")
DUMMY["COMBO_EVAL_B"] = ("--policy agent --model-files rl_arch/agents_online/a.pt,rl_arch/agents_online/b.pt "
                         "--ensemble vote --stop-on-noop --guard-rollback 1.0 --guard-fallback LBFGS:1:500")
DUMMY["XFER_FLAGS"] = "--state-mode tasknorm"
DUMMY["AGENT_CHAIN"] = "Adam:0.001:1000,Adam:0.0001:100,LBFGS:0.5:1000"
DUMMY["GA_TOP1"] = "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000"
DUMMY["DEPLOY"] = "--guard-rollback 1.0 --guard-fallback LBFGS:1:500"
DUMMY["FINAL"] = DUMMY["BASE"] + " --scalar-ctx --ctx-no-err"
DUMMY["BASE_QR"] = DUMMY["BASE_NOVB"].replace("convnext_dqn", "cnx_qrdqn")
DUMMY["BASE_FACT"] = DUMMY["BASE"].replace("convnext_dqn", "cnx_factored")
DUMMY["BASE_STAT"] = DUMMY["BASE"].replace("convnext_dqn", "stat_dqn")


class Reached(Exception):
    """Разбор аргументов и проверки пройдены: выполнение дошло до тяжёлой части."""


def _stop(*a, **k):
    raise Reached()


def lint_offline(argv):
    import numpy as np
    import offline_rl as O
    from test_chain_loader import make_file

    def fake_episodes(data_dir=None, subdir=None, max_files=0):
        return [make_file(False, [([0.4, 0.2, 0.1], -1), ([0.3, 0.008], 1), ([0.5, 0.25], 0)]),
                make_file(True, [([0.4, 0.3, 0.05], -1), ([0.2, 0.1], 0)])]

    def fake_teacher(path, data, device):
        return np.zeros((len(data["A"]), O.N_ACTIONS), dtype=np.float32), {}

    saved = (O.load_episodes, O.train_variant, O.teacher_q, O.merge_online_buffers)
    O.load_episodes, O.train_variant, O.teacher_q = fake_episodes, _stop, fake_teacher
    O.merge_online_buffers = lambda data, paths, **k: data
    try:
        sys.argv = ["offline_rl.py"] + argv
        O.main()
    finally:
        O.load_episodes, O.train_variant, O.teacher_q, O.merge_online_buffers = saved


def lint_train(argv):
    import online_train_env as T
    saved = T.dde.config.set_default_float
    T.dde.config.set_default_float = _stop
    try:
        sys.argv = ["online_train_env.py"] + argv
        T.main()
    finally:
        T.dde.config.set_default_float = saved


def lint_eval(argv):
    import online_eval_env as E
    saved = E.run_seed, E.result_done
    E.run_seed = _stop
    E.result_done = lambda name: False        # без обращения к HF при проверке
    try:
        sys.argv = ["online_eval_env.py"] + argv
        E.main()
    finally:
        E.run_seed, E.result_done = saved
    # то, что main не проверяет: цепочки и маски должны разбираться
    ns = dict(zip(argv[::1], argv[1::1]))
    for flag in ("--script", "--guide", "--scout-set", "--scout-commit", "--bandit-arms", "--guard-fallback"):
        if flag in argv:
            E.parse_chain(argv[argv.index(flag) + 1])
    for flag in ("--rule-burst", "--rule-kick"):
        if flag in argv:
            E.parse_action_spec(argv[argv.index(flag) + 1])
    if "--mask" in argv:
        E.parse_mask(argv[argv.index("--mask") + 1])


def lint_chain(argv):
    import run_chain_pde as C
    saved = C.run_orchestrator
    C.run_orchestrator = _stop
    try:
        sys.argv = ["run_chain_pde.py"] + argv
        C.main()
    finally:
        C.run_orchestrator = saved
    if "--chain-json" in argv:
        path = os.path.join(ROOT, argv[argv.index("--chain-json") + 1])
        json.load(open(path))


def lint_ga(argv):
    """Генетический поиск по цепочкам: разбор аргументов и подготовка до первого прогона среды.
    Обычное исключение из run_seed скрипт засчитывает особи как штраф, поэтому подмена поднимает
    SystemExit (его скрипт пропускает наружу); --resume без локального файла идёт в HF, поэтому
    load_state тоже подменён; состояние пишется во временную папку, а не в runs_rl_online."""
    import ga_chains as G
    saved = G.run_seed, G.load_state

    def reached(*a, **k):
        raise SystemExit("lint: дошли до run_seed")
    G.run_seed = reached
    G.load_state = lambda path, name, smoke: None
    try:
        G.main(list(argv) + ["--save-dir", tempfile.mkdtemp(prefix="lint_ga_")])
    except SystemExit as e:
        if "lint" in str(e):
            raise Reached()
        raise
    finally:
        G.run_seed, G.load_state = saved


LINTERS = {"offline_rl.py": lint_offline, "online_train_env.py": lint_train,
           "online_eval_env.py": lint_eval, "run_chain_pde.py": lint_chain,
           "ga_chains.py": lint_ga}


def check(script, args):
    for k, v in DUMMY.items():
        args = args.replace("{" + k + "}", v)
    left = re.findall(r"\{([A-Z_]+)\}", args)
    if left:
        return f"неизвестные подстановки {left}"
    fn = LINTERS.get(os.path.basename(script))
    if fn is None:
        return f"нет проверки для {script}"
    if not os.path.exists(os.path.join(ROOT, script)):
        return f"скрипта нет: {script}"
    out, err = io.StringIO(), io.StringIO()
    argv0 = list(sys.argv)
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn(shlex.split(args))
        return "скрипт завершился, не дойдя до тяжёлой части"
    except Reached:
        return None
    except SystemExit as e:
        msg = (err.getvalue().strip().splitlines() or [str(e.code)])[-1]
        return f"отказ скрипта: {msg[:300]}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        sys.argv = argv0
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--queue", default=os.path.join(HERE, "queue.json"),
                    help="файл очереди (queue.json или queue_<префикс>.json другого УрЧП)")
    args = ap.parse_args()
    q = json.load(open(args.queue))
    # уже принятые решения проверяются настоящими значениями, остальные подстановки — образцами
    for dp in ((os.path.join(HERE, f"decisions_{q['prefix']}.json"),) if q.get("prefix")
               else (os.path.join(HERE, "decisions.json"),)):
        if dp and os.path.exists(dp):
            dec = json.load(open(dp))
            DUMMY.update({k: str(v) for k, v in dec.items()})
            print(f"подстановки из {os.path.basename(dp)}: {', '.join(sorted(dec))}")
    jobs = q["jobs"]
    if args.only:
        jobs = [j for j in jobs if j["id"] in args.only.split(",")]
    elif args.wave is not None:
        jobs = [j for j in jobs if j["wave"] == args.wave]
    bad = 0
    for j in jobs:
        for i, s in enumerate(j["steps"], 1):
            problem = check(s["script"], s["args"])
            if problem:
                bad += 1
                print(f"ОШИБКА {j['id']} шаг {i} ({os.path.basename(s['script'])}): {problem}")
                print(f"       {s['args'][:260]}")
    # готовые комбинации флагов (слоты волны 4 и {XFER_FLAGS}): те же проверки, что у задач
    combos = q.get("combos", []) if not (args.only or args.wave is not None) else []
    for c in combos:
        problem = check(c["script"], c["args"] + ("" if c["kind"] != "train" else " --seed 42 --tag combo"))
        if problem:
            bad += 1
            print(f"ОШИБКА комбинация {c['id']} ({os.path.basename(c['script'])}): {problem}")
            print(f"       {c['args'][:260]}")
    n_steps = sum(len(j["steps"]) for j in jobs)
    print(f"проверено задач {len(jobs)}, шагов {n_steps}, комбинаций {len(combos)}, ошибок {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
