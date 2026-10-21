#!/usr/bin/env python
"""Задачи очереди как bash-скрипт для своего GPU (не Kaggle): те же шаги, что у кернела (kernel_template.py).

Окружение как в кернеле: DDEBACKEND=pytorch, шаги запускаются из корня репозитория, результаты и модели
заливаются в HF-датасет результатов шагами сами (нужен HF_TOKEN_WRITE в окружении; в скрипте токенов нет).
Порядок: задачи без зависимостей от других задач списка, затем зависимые (вторые блоки сидов ждут модель,
которую заливает обучение) — по уровням графа needs. Каждая задача — функция; первым аргументом скрипта можно
дать список задач (через запятую), иначе идут все.

    python experiments/rl_arch/queue/export_wave.py --wave 69 --out run_w69.sh
    HF_TOKEN_WRITE=... bash run_w69.sh                      # все задачи по порядку
    HF_TOKEN_WRITE=... bash run_w69.sh pb2d-w69-d4-m1       # одна задача
"""
import argparse
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
QUEUES = ("queue.json", "queue_pb2d.json")


def load_jobs(queues, wave=None, only=None):
    jobs = []
    for qf in queues:
        q = json.load(open(os.path.join(HERE, qf)))
        for j in (q["jobs"] if isinstance(q, dict) else q):
            if (only and j["id"] in only) or (not only and wave is not None and j.get("wave") == wave):
                jobs.append(j)
    if only:
        missing = set(only) - {j["id"] for j in jobs}
        if missing:
            sys.exit(f"нет задач: {sorted(missing)}")
    if not jobs:
        sys.exit("пустой список задач")
    return jobs


def producer(jobs, need):
    """Задача списка, которая заливает модель need (…_<тег>_seed<m>.pt): её обучение с --model-tag <тег> --seeds <m>."""
    m = re.search(r"_([A-Za-z0-9]+)_seed(\d+)\.pt$", need)
    if not m:
        return None
    for j in jobs:
        for st in j["steps"]:
            a = " " + st["args"] + " "
            if f" --model-tag {m.group(1)} " in a and f" --seeds {m.group(2)} " in a:
                return j["id"]
    return None


def order(jobs):
    """Уровни по зависимостям внутри списка: задача, чья модель из needs заливается другой задачей, идёт после неё."""
    by = {j["id"]: j for j in jobs}
    level = {}

    def lv(jid, path=()):
        if jid in level:
            return level[jid]
        if jid in path:
            sys.exit(f"цикл зависимостей: {path + (jid,)}")
        deps = [p for p in (producer(jobs, n) for n in by[jid].get("needs", [])) if p and p != jid]
        level[jid] = 1 + max(lv(d, path + (jid,)) for d in deps) if deps else 0
        return level[jid]
    for j in jobs:
        lv(j["id"])
    return sorted(jobs, key=lambda j: (level[j["id"]], j["id"]))


def script(jobs):
    out = ["#!/usr/bin/env bash",
           "# Сгенерировано experiments/rl_arch/queue/export_wave.py — шаги задач очереди, как в кернеле Kaggle.",
           "set -uo pipefail",
           f'cd "$(dirname "$0")"/{shlex.quote(os.path.relpath(REPO, os.getcwd()))} 2>/dev/null || true',
           'cd "$(git rev-parse --show-toplevel)"',
           "export DDEBACKEND=pytorch",
           'if [ -z "${HF_TOKEN_WRITE:-}${HF_TOKEN:-}" ]; then echo "нужен HF_TOKEN_WRITE (результаты идут в HF)"; exit 2; fi',
           'PY="${PYTHON:-python}"', ""]
    names = []
    for j in jobs:
        fn = "job_" + j["id"].replace("-", "_")
        names.append((j["id"], fn))
        out.append(f"{fn}() {{  # {j.get('note', '')[:110]}")
        out.append(f'  echo "=== {j["id"]} (~{j["hours"]} GPU-ч)"')
        for s in j["steps"]:
            out.append(f'  "$PY" {shlex.quote(s["script"])} {s["args"]} || return 1')
        out.append("}")
    out += ["", 'WANT="${1:-}"', "fail=0"]
    for jid, fn in names:
        out.append(f'if [ -z "$WANT" ] || [[ ",$WANT," == *",{jid},"* ]]; then {fn} || {{ echo "ОШИБКА {jid}"; fail=1; }}; fi')
    out.append('exit $fail')
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", type=int)
    ap.add_argument("--only", default="", help="задачи через запятую (вместо --wave)")
    ap.add_argument("--queues", default=",".join(QUEUES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    if a.wave is None and not only:
        ap.error("нужен --wave или --only")
    jobs = order(load_jobs([q for q in a.queues.split(",") if q], a.wave, only))
    with open(a.out, "w") as f:
        f.write(script(jobs))
    os.chmod(a.out, 0o755)
    for j in jobs:
        print(f"{j['id']:28s} ~{j['hours']:5.2f} GPU-ч  {j.get('note', '')[:80]}")
    print(f"всего ~{sum(j['hours'] for j in jobs):.1f} GPU-ч -> {a.out}")


if __name__ == "__main__":
    main()
