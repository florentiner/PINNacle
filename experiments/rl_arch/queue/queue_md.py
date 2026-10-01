#!/usr/bin/env python
"""Таблицы задач для paper/lit_review/QUEUE.md из queue.json (чтобы документ не
расходился с очередью). Печатает markdown; QUEUE.md собирается из этих таблиц и
пояснений вручную.

    python experiments/rl_arch/queue/queue_md.py --wave 11
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def table(jobs, with_note=True):
    rows = ["| Задача | Группа | Часов GPU, не более | Что проверяет |", "|---|---|---|---|"]
    for j in jobs:
        note = (j.get("note") or "").replace("|", "／")
        rows.append(f"| `{j['id']}` | {j['group']} | {j['hours']:g} | {note} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--no-eval", action="store_true", help="не показывать задачи-оценки с пустой заметкой")
    args = ap.parse_args()
    q = json.load(open(os.path.join(HERE, "queue.json")))
    jobs = [j for j in q["jobs"] if j["wave"] == args.wave]
    if args.no_eval:
        jobs = [j for j in jobs if j.get("note")]
    print(f"{len([j for j in q['jobs'] if j['wave'] == args.wave])} задач, "
          f"не более {sum(j['hours'] for j in q['jobs'] if j['wave'] == args.wave):.0f} GPU-часов")
    print()
    print(table(jobs))


if __name__ == "__main__":
    main()
