#!/usr/bin/env python
"""Собранный заново буфер задачи-цели вместо её исходного офлайнового буфера
(agent_combo.py --target-buffer, offline_rl.py --target-buffer).

Файлы пишет collect_active.py --save-buffer: одна цепочка с картами на файл, формат исходных
буферов, поэтому читаются они теми же загрузчиками (offline_rl.load_episodes,
episodes_to_arrays_chains, surrogate.build_table). Спецификация — через запятую префиксы в датасете
результатов HF (rl_arch/buffers_new/<тег сборщика>) и/или локальные папки; внутри ищется
<папка буфера задачи>/*.pt (у локальной папки — ещё и *.pt прямо в ней). Несколько сборщиков —
несколько префиксов: цепочки просто складываются.

С этим флагом данные цели берутся ТОЛЬКО отсюда: ученик на картах учится на этих цепочках, в
таблице суррогата (модель среды учителя, модельные цели, щит) строки цели — только из них;
остальные задачи таблицы — прежние буферы. Таблица собирается одной функцией table_with для учителя
и ученика: при одинаковых списках папок и одном --work / --mb-work отпечаток данных тот же, ансамбль
учителя переиспользуется. Кэш: таблица остальных задач — surrogate_table_<crc списка папок>.npz
(список без папки цели, с полной таблицей не совпадает), ансамбль — по отпечатку строк.

Что остаётся от исходного буфера цели: ничего из данных обучения; E0 задачи (pde_meta.json) измерен
отдельно. Веса --encoder-init сюда не относятся (чем предобучен энкодер — вопрос к его файлу).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "track3"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surrogate as S  # noqa: E402

OUT_REPO = "danil-e/pinnacle-optuna-db"


def resolve(spec, subdir):
    """Спецификация -> [(источник, папка с файлами .pt)]. Источника без файлов быть не должно:
    load_episodes на пустой папке молча ушёл бы за исходным буфером задачи."""
    out = []
    for item in (x.strip().rstrip("/") for x in spec.split(",") if x.strip()):
        if os.path.isdir(item):
            d = os.path.join(item, subdir) if os.path.isdir(os.path.join(item, subdir)) else item
        else:
            try:
                from huggingface_hub import snapshot_download
                root = snapshot_download(OUT_REPO, repo_type="dataset", allow_patterns=[f"{item}/{subdir}/*.pt"])
            except Exception as e:      # noqa: BLE001
                raise SystemExit(f"--target-buffer {item}: локальной папки нет, из HF не скачано "
                                 f"({type(e).__name__}: {str(e)[:120]})")
            d = os.path.join(root, item, subdir)
        if not glob.glob(os.path.join(d, "*.pt")):
            raise SystemExit(f"--target-buffer {item}: в {d} нет файлов .pt (папка задачи — {subdir})")
        out.append((item, d))
    if not out:
        raise SystemExit("--target-buffer: пустая спецификация")
    return out


def load(spec, subdir, log=print):
    """Файлы собранного буфера (список файлов = список списков переходов, как load_episodes) и
    сводка: sources, files, chains, transitions, envs, budgets."""
    import offline_rl as O
    eps, info = [], dict(spec=spec, subdir=subdir, sources=[], files=0, chains=0, transitions=0, envs={},
                         budgets={})
    for item, d in resolve(spec, subdir):
        n_pt = len(glob.glob(os.path.join(d, "*.pt")))
        got = O.load_episodes(d, subdir)
        if len(got) != n_pt:
            log(f"ВНИМАНИЕ: {item}: прочитано {len(got)} файлов из {n_pt}")
        info["sources"].append(f"{item} ({len(got)} файлов)")
        eps += got
    for ep in eps:
        info["transitions"] += len(ep)
        info["chains"] += sum(int(t.get("done", 0)) != 0 for t in ep) + int(int(ep[-1].get("done", 0)) == 0)
        env = (ep[0].get("meta") or {}).get("env") or {}
        label = (f"keep_{env.get('mode') or 'all'}" if env.get("keep_opt") else "reset") if env else "?"
        info["envs"][label] = info["envs"].get(label, 0) + 1
        bud = str(env.get("budget", "?"))
        info["budgets"][bud] = info["budgets"].get(bud, 0) + 1
    info["files"] = len(eps)
    log(f"собранный буфер цели ({subdir}): файлов {info['files']}, цепочек {info['chains']}, переходов "
        f"{info['transitions']}; источники: {info['sources']}; среда сбора {info['envs']}, бюджет {info['budgets']}")
    if set(info["envs"]) - {"reset"}:
        log("ВНИМАНИЕ: собранные цепочки шли в среде без сброса оптимизатора (или среда неизвестна), а буферы "
            "остальных задач и модель среды на них — со сбросом: динамика смешана")
    if set(info["budgets"]) - {str(S.BUDGET)}:
        log(f"ВНИМАНИЕ: бюджеты сбора {sorted(info['budgets'])}, обучение рассчитано на {S.BUDGET} эпох")
    return eps, info


def target_rows(episodes, name, meta=None):
    """Строки таблицы суррогата из собранных файлов задачи name (surrogate.build_table) плюс EPA —
    фактические эпохи действия (поле epochs_real: последнее действие цепочки обрезано бюджетом) и
    GRP — группа общего префикса (поле group; без него — сама цепочка)."""
    import offline_rl as O
    meta = meta if meta is not None else O.pde_meta()
    sd = meta[name].get("subdir")
    try:
        tab = S.build_table(subdirs=[sd], episodes={sd: list(episodes)}, meta=meta, verbose=False)
    except ValueError:
        raise SystemExit(f"--target-buffer: в собранных файлах задачи {name} нет ни одного перехода")
    n = len(tab["A"])
    tab["EPA"] = S.EPOCHS[tab["A"]].astype(np.int64)
    tab["GRP"] = tab["CH"].astype(np.int64)
    flat = [t for ep in episodes for t in ep]
    if len(flat) == n:                  # загрузчик не выбросил ни одного перехода: строки идут как в файлах
        ep_real = np.array([int(t.get("epochs_real") or 0) for t in flat], dtype=np.int64)
        tab["EPA"] = np.where((ep_real > 0) & (ep_real < tab["EPA"]), ep_real, tab["EPA"])
        first = {}
        for i, t in enumerate(flat):
            if t.get("group") is not None:
                tab["GRP"][i] = first.setdefault(t["group"], int(tab["CH"][i]))
    return tab


def empty_table():
    tab = dict({k: np.zeros((0,), np.int64) for k in ("PDE", "FILE", "CH", "STEP", "SPENT", "A", "PA", "EPA", "GRP")},
               CTX=np.zeros((0, 7)), NAMES=[], SUBDIRS=[])
    for k, shp, dt in (("L", (0,), float), ("LN", (0,), float), ("LV", (0, 6), float),
                       ("LVN", (0, 6), float), ("SPF", (0, 3), float), ("LAST", (0,), bool)):
        tab[k] = np.zeros(shp, dt)
    return tab


def table_with(tab, episodes, name, meta=None):
    """Таблица суррогата, в которой строки задачи name — только из собранных файлов: прежние строки
    задачи выбрасываются, новые дописываются в конец (номера цепочек и файлов — после имеющихся).
    Возвращает (таблица, номер задачи, число дописанных строк)."""
    import agent_combo as AC
    import offline_rl as O
    meta = meta if meta is not None else O.pde_meta()
    tab = AC.with_epa(dict(tab))
    pi = AC.ensure_pde(tab, name, meta)
    tab = AC.subset(tab, tab["PDE"] != pi)
    new = target_rows(episodes, name, meta)
    off_c = int(max(tab["CH"].max(), tab["GRP"].max())) + 1 if len(tab["CH"]) else 0
    off_f = int(tab["FILE"].max()) + 1 if len(tab["FILE"]) else 0
    new["PDE"] = np.full(len(new["A"]), pi, dtype=np.int64)
    new["CH"], new["GRP"], new["FILE"] = new["CH"] + off_c, new["GRP"] + off_c, new["FILE"] + off_f
    out = dict(tab)
    for k in AC.ROWKEYS:
        add = np.asarray(new[k]).astype(tab[k].dtype)
        out[k] = np.concatenate([tab[k], add.reshape((len(add),) + tab[k].shape[1:])])
    return out, pi, int(len(new["A"]))


def surrogate_table(episodes, name, subdirs="all", table="", work="agent_combo_work", log=print):
    """Таблица для ученика (offline_rl.py --target-buffer с --mb-aux): та же, что собирает
    agent_combo.py --target-buffer — буферы остальных задач (кэш в work) и собранные цепочки цели."""
    import agent_combo as AC
    import offline_rl as O
    meta = O.pde_meta()
    tsub = meta[name].get("subdir")
    others = [s for s in O.resolve_subdirs(subdirs) if s != tsub]
    if table or others:
        tab = AC.get_table(argparse.Namespace(table=table, work=work), others, log)
    else:
        tab = empty_table()
    tab, pi, n = table_with(tab, episodes, name, meta)
    log(f"таблица суррогата с собранным буфером цели: переходов {len(tab['A'])}, из них цели {n} "
        f"(цепочек {len(np.unique(tab['CH'][tab['PDE'] == pi]))}), задач {len(np.unique(tab['PDE']))}")
    return tab
