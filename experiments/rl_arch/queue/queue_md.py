#!/usr/bin/env python
"""Таблицы задач для paper/lit_review/QUEUE.md из queue.json (чтобы документ не
расходился с очередью). Печатает markdown; пояснения в QUEUE.md пишутся вручную.

    python experiments/rl_arch/queue/queue_md.py --wave 11
    python experiments/rl_arch/queue/queue_md.py --sync paper/lit_review/QUEUE.md

--sync обновляет в файле строку «N задач, не более M GPU-часов» и таблицу задач в каждом
разделе «## Волна K.»; текст вокруг не трогает. Раздел волны, которого в файле нет,
называется в выводе — его надо дописать руками.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = "| Задача | Группа | Часов, не более | Что проверяет |"


def table(jobs):
    rows = [HEAD, "|---|---|---|---|"]
    for j in jobs:
        note = (j.get("note") or "").replace("|", "／")
        rows.append(f"| `{j['id']}` | {j['group']} | {j['hours']:g} | {note} |")
    return "\n".join(rows)


def count_line(q, wave):
    jobs = [j for j in q["jobs"] if j["wave"] == wave]
    return f"{len(jobs)} задач, не более {sum(j['hours'] for j in jobs):.0f} GPU-часов по лимитам сессий."


def sync(path, q):
    lines = open(path, encoding="utf-8").read().split("\n")
    out, i, seen = [], 0, set()
    wave = None
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"## Волна (\d+)\.", ln)
        if ln.startswith("## "):
            wave = int(m.group(1)) if m else None
            if wave is not None:
                seen.add(wave)
        if wave is not None and re.match(r"\d+ задач, не более \d+ GPU-часов", ln):
            out.append(count_line(q, wave))
            i += 1
            continue
        if wave is not None and ln.strip() == HEAD:
            while i < len(lines) and lines[i].startswith("|"):
                i += 1                       # прежняя таблица задач
            # задачи-оценки («оценка: ...») повторяют строку своего обучения и в таблицу не идут
            out.append(table([j for j in q["jobs"] if j["wave"] == wave and j.get("note")
                              and not j["note"].startswith("оценка")]))
            continue
        out.append(ln)
        i += 1
    open(path, "w", encoding="utf-8").write("\n".join(out))
    missing = sorted({j["wave"] for j in q["jobs"]} - seen)
    print(f"{path}: обновлены волны {sorted(seen)}" + (f"; нет разделов для волн {missing}" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="не показывать задачи-оценки с пустой заметкой")
    ap.add_argument("--sync", default="", help="обновить счётчики и таблицы волн в указанном QUEUE.md")
    args = ap.parse_args()
    q = json.load(open(os.path.join(HERE, "queue.json")))
    if args.sync:
        sync(args.sync, q)
        return
    if args.wave is None:
        ap.error("нужен --wave или --sync")
    jobs = [j for j in q["jobs"] if j["wave"] == args.wave]
    if args.no_eval:
        jobs = [j for j in jobs if j.get("note")]
    print(count_line(q, args.wave))
    print()
    print(table(jobs))


if __name__ == "__main__":
    main()
