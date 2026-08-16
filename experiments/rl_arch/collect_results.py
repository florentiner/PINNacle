#!/usr/bin/env python
"""Сводные таблицы по двум НЕЗАВИСИМЫМ задачам из rl_arch/online_env на HF.

  Задача 1 (статья): армы бустинга none / landscape / plateau / landscape_peak /
  midpoint — проверка идеи «вторая сеть как бустинг, момент по ландшафту».

  Задача 2 (архитектуры): ConvNeXt+DQN / +CQL / +Dueling против их бэйзлайна —
  итоговый l2re цепочек, которые порождает агент.

Использование:  python experiments/rl_arch/collect_results.py [--partial]
"""
from __future__ import annotations

import argparse
import json
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "danil-e/pinnacle-optuna-db"
ARM_RU = {"none": "без бустинга (контроль)", "landscape": "бустинг по ландшафту (P>0.5)",
          "plateau": "бустинг по плато лосса", "landscape_peak": "бустинг в пике ландшафта",
          "midpoint": "бустинг слепо в середине"}
VAR_RU = {"convnext_dqn": "ConvNeXt + DQN", "cnx_cql": "ConvNeXt + CQL",
          "cnx_dueling": "ConvNeXt + дуэлинговая голова", "their_dqn": "их бэйзлайн (ConvEncoder+Dueling)",
          "cnx_cql_qr": "ConvNeXt + CQL + квантили", "random": "случайная политика",
          "cnx_bcq": "ConvNeXt + BCQ", "cnx_bbf": "ConvNeXt + BBF",
          "cnx_smdp": "SMDP-BBF-CQL (полный стек)"}
VAR_RE = ("convnext_dqn|cnx_cql_qr|cnx_cql|cnx_dueling|cnx_bcq|cnx_bbf|cnx_smdp|"
          "their_dqn|random")


def load(partial=False):
    rows = []
    for f in list_repo_files(REPO, repo_type="dataset"):
        if not f.startswith("rl_arch/online_env/"):
            continue
        # без force_download: кеш ключуется по коммиту, изменённый файл скачается сам,
        # а неизменённый не будет дёргать API (иначе HF отдаёт 429)
        try:
            r = json.load(open(hf_hub_download(REPO, f, repo_type="dataset")))
        except Exception as e:
            print(f"  пропуск {f.split('/')[-1]}: {type(e).__name__}")
            continue
        # прогон, выбравший весь бюджет, засчитывается, даже если остался помечен
        # частичным: progress_cb пишет строку сразу после последнего шага, и её
        # l2re уже финальная — не хватает только заключительной заливки, которую
        # съедает 12-часовой срез сессии Kaggle
        r["_done"] = (not r.get("partial")) or r.get("spent", 0) >= r.get("budget", 10 ** 9)
        if not r["_done"] and not partial:
            continue
        r["_name"] = f.split("/")[-1].replace(".json", "")
        # у старых частичных строк не было ни арма, ни флага бустинга — арм читаем
        # из имени файла, а факт бустинга из самой цепочки (шаг ["BOOST", eps, 0])
        if not r.get("boost_trigger") and "_boost" in r["_name"]:
            m = re.search(r"_boost3?d?_(\w+?)_seed\d+$", r["_name"])
            if m:
                r["boost_trigger"] = m.group(1)
        if r.get("boosted") is None:
            r["boosted"] = any(isinstance(s, list) and s and s[0] == "BOOST"
                               for s in r.get("chain", []))
        rows.append(r)
    return rows


def agg(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.4f}", f"{np.median(v):.4f}", f"{v.std(ddof=1):.4f}" if len(v) > 1 else "—", len(v)


def task_paper(rows):
    print("\n" + "=" * 78)
    print("ЗАДАЧА 1 — ИДЕЯ СТАТЬИ: бустинг второй сетью, момент по ландшафту")
    print("=" * 78)
    for pde_key, pde_ru in [("poisson3d_complexgeometry", "poisson3d_complexgeometry"),
                            ("poissonboltzmann2d", "poissonboltzmann2d")]:
        # только прогоны эксперимента по статье: в их имени есть тег boost/boost3d,
        # иначе сюда попадают архитектурные прогоны (у них boost_trigger="none")
        sub = [r for r in rows if r["_name"].startswith(pde_key) and r["_done"]
               and r.get("boost_trigger") and "_boost" in r["_name"]]
        if not sub:
            continue
        print(f"\n{pde_ru}:")
        print(f"  {'арм':32s} {'сидов':>6s} {'l2re сред.':>11s} {'медиана':>10s} {'станд.откл':>11s}  бустинг срабатывал")
        for arm in ("none", "landscape", "plateau", "landscape_peak", "midpoint"):
            g = [r for r in sub if r.get("boost_trigger") == arm]
            if not g:
                continue
            m, md, sd, n = agg([r["l2re"] for r in g])
            fired = sum(1 for r in g if r.get("boosted"))
            print(f"  {ARM_RU[arm]:32s} {n:6d} {m:>11s} {md:>10s} {sd:>11s}  {fired}/{n}")


def task_arch(rows):
    print("\n" + "=" * 78)
    print("ЗАДАЧА 2 — АРХИТЕКТУРЫ: итоговый l2re цепочек, порождённых агентом")
    print("=" * 78)
    sub = [r for r in rows if "_boost" not in r["_name"] and r["_done"]]
    by_var = {}
    for r in sub:
        m = re.search(f"({VAR_RE})", r["_name"])
        if not m:
            continue
        by_var.setdefault(m.group(1), []).append(r)
    # бюджеты нельзя смешивать: прогоны на 10000 и 31000 эпох несравнимы
    for budget in sorted({r.get("budget") for g in by_var.values() for r in g}, reverse=True):
        print(f"\n  бюджет {budget} эпох:")
        print(f"  {'конфигурация':34s} {'сидов':>6s} {'l2re сред.':>11s} {'медиана':>10s} {'лучший':>10s}")
        sel = {v: [r for r in g if r.get("budget") == budget] for v, g in by_var.items()}
        sel = {v: g for v, g in sel.items() if g}
        for var, g in sorted(sel.items(), key=lambda kv: np.mean([x["l2re"] for x in kv[1]])):
            m, md, sd, n = agg([r["l2re"] for r in g])
            print(f"  {VAR_RU.get(var, var):34s} {n:6d} {m:>11s} {md:>10s} {min(r['l2re'] for r in g):10.4f}")


def task_train():
    """Онлайн-обучение: агент учится по ходу, метрика — последняя ЗАВЕРШЁННАЯ цепочка."""
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith("rl_arch/online_train/")]
    if not files:
        return
    print("\n" + "=" * 78)
    print("ЗАДАЧА 2б — ОНЛАЙН-ОБУЧЕНИЕ: агент учится в среде, 11 ч на прогон")
    print("=" * 78)
    print(f"  {'конфигурация':34s} {'цепочек':>8s} {'последняя':>10s} {'медиана5':>10s} "
          f"{'лучшая':>9s} {'часов':>6s}")
    out = []
    for f in sorted(files):
        try:
            r = json.load(open(hf_hub_download(REPO, f, repo_type="dataset")))
        except Exception:
            continue
        ls = [c["l2re"] for c in r.get("chains", [])]
        if not ls:
            continue
        name = VAR_RU.get(r["variant"], r["variant"])
        if "rlpd" in f:
            name += " + RLPD"
        out.append((name, len(ls), r["l2re_last_complete"], float(np.median(ls[-5:])),
                    r["l2re_best"], r.get("elapsed_h", 0)))
    for name, n, last, m5, best, h in sorted(out, key=lambda x: x[3]):
        print(f"  {name:34s} {n:8d} {last:10.4f} {m5:10.4f} {best:9.4f} {h:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="включить незавершённые прогоны")
    args = ap.parse_args()
    rows = load(args.partial)
    print(f"загружено прогонов: {len(rows)}" + (" (включая незавершённые)" if args.partial else ""))
    task_paper(rows)
    task_arch(rows)
    task_train()


if __name__ == "__main__":
    main()
