#!/usr/bin/env python
"""Собранный заново буфер задачи-цели вместо её исходного офлайнового буфера
(agent_combo.py --target-buffer, offline_rl.py --target-buffer).

Файлы пишут collect_active.py --save-buffer (направленный сбор) и online_eval_env.py --save-buffer
(прогоны самой политики в среде, см. ниже): одна цепочка с картами на файл, формат исходных
буферов, поэтому читаются они теми же загрузчиками (offline_rl.load_episodes,
episodes_to_arrays_chains, surrogate.build_table). Запись файла — общая (chain_transitions,
save_chain, upload_files в этом модуле). Спецификация — через запятую префиксы в датасете
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

Прогоны политики в буфер (итерированное офлайновое обучение): online_eval_env.py --save-buffer <тег>
пишет исполненную цепочку каждого законченного сида одним файлом
    <save-dir>/buffers_new/<тег>/<папка буфера задачи>/<тег прогона>_seed<сид>.pt
    HF: rl_arch/buffers_new/<тег>/<папка буфера задачи>/<тег прогона>_seed<сид>.pt
(файл и итоговая строка сида — один коммит). Обучение на старых и новых данных — перечислением
префиксов: --target-buffer rl_arch/buffers_new/<сбор>,rl_arch/buffers_new/<тег>.
  * в файле только исполненные и ОСТАВЛЕННЫЕ действия с их настоящими эпохами. Действие, которое
    страж (--guard-rollback) откатил, переходом не становится: веса вернулись к снимку, карта не
    строилась; переход — запасное действие стража из того же состояния. Что было откачено, лежит
    в поле rolled перехода оставленного действия: [[номер действия, эпохи, лосс до, лосс после]];
  * эпохи: поле spent — настоящий счётчик среды после действия (вместе с откаченными эпохами),
    guard_spent — сколько из них откачено к этому моменту. Загрузчики эти поля не читают: их
    SPENT — сумма номинальных эпох оставленных действий, то есть spent − epochs_real − guard_spent
    (обученные эпохи, которые несут веса). Цепочка для них кончается терминальным переходом
    (done=1) раньше бюджета — как цепочка исходного буфера, закончившаяся по допуску; досрочной
    обрезки по --episode-budget нет (сумма оставленных эпох не больше настоящего счётчика).
    Следствие: канал «потрачено» скалярного контекста у таких цепочек при обучении отстаёт от
    того, что агент видел в прогоне, на guard_spent / бюджет;
  * разведка (--explore-eps, --explore-stall): у перехода поле explore — оставленное действие было
    случайным, а не выбором агента;
  * meta первого перехода: source="rollout", политика, чекпоинт, страж, разведка; env — как у сбора
    (keep_opt, mode, budget), поэтому сводка load печатает среду и бюджет так же.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "track3"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surrogate as S  # noqa: E402

OUT_REPO = "danil-e/pinnacle-optuna-db"
FORMAT = 1
BUF_HF = "rl_arch/buffers_new"  # корень буферов с картами в датасете результатов HF
MAP_KEYS = ("loss_total", "loss_oper", "loss_bnd")


# --------------------------------------------------------------------------------------------
# Запись: цепочка среды -> файл буфера (collect_active.py и online_eval_env.py --save-buffer)
# --------------------------------------------------------------------------------------------

def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def action_index(opt, lr, ep):
    """Индекс действия из 27 по оптимизатору и шагу; эпохи — ближайшие по логарифму (последнее
    действие могло быть обрезано бюджетом), как в surrogate.load_eval_rows."""
    hit = [i for i, (o, l, _) in enumerate(S.ACTIONS) if o == opt and abs(l - float(lr)) < 1e-12]
    if not hit:
        return None
    return min(hit, key=lambda j: abs(math.log10(int(S.EPOCHS[j])) - math.log10(max(float(ep), 1.0))))


def buffer_subdir(pde):
    """Папка буфера задачи (как у исходных офлайновых буферов: pde_meta.json, поле subdir)."""
    import offline_rl as O
    return (O.pde_meta().get(pde) or {}).get("subdir") or pde


def chain_transitions(res, acts, budget):
    """Исполненная цепочка -> переходы в формате офлайнового буфера (описание — в докстринге
    collect_active.py). res — итог прогона среды (chain, hist, err, maps: {номер действия с 1:
    карты (3, 26, 26) после него}), acts — задуманные номера действий.
    Цепочка кончается на первом переходе без числовой ошибки и на первом действии, перед которым
    нет карты (упавший прогон)."""
    chain, hist, maps = res["chain"], res["hist"], res.get("maps") or {}
    n = min(len(chain), len(hist))
    errs = list(res.get("err") or []) + [None] * n
    zero = {k: np.zeros((26, 26), np.float32) for k in MAP_KEYS}
    out, has_next, bad = [], False, False
    for i in range(n):
        if i and i not in maps:
            break
        x = acts[i] if acts is not None and i < len(acts) else None
        if x is None or S.ACTIONS[x][0] != chain[i][0] or abs(S.ACTIONS[x][1] - float(chain[i][1])) > 1e-12:
            x = action_index(*chain[i])
        if x is None:
            break
        cur = {k: maps[i][j].copy() for j, k in enumerate(MAP_KEYS)} if i else {k: v.copy() for k, v in zero.items()}
        has_next = (i + 1) in maps
        nxt = {k: (maps[i + 1][j] if has_next else cur[k]).copy() for j, k in enumerate(MAP_KEYS)}
        e = num(errs[i])
        bad = e is None or e <= 0
        out.append(dict(state=cur, next_state=nxt, action=(x // 9, {"lr": (x % 9) // 3, "epochs": x % 3}),
                        reward=(0.0 if bad else -float(e)), done=0, epochs_real=int(chain[i][2]),
                        spent=int(hist[i][0]), l2re=num(hist[i][2])))
        if bad:
            break
    if out:
        out[-1]["done"] = -1 if bad else (1 if out[-1]["spent"] >= budget or not has_next else 0)
    return out


def save_chain(d, fname, tr, chain_id, group=None, meta=None):
    """Одна цепочка -> файл буфера d/fname: метки chain и group у переходов, meta у первого."""
    import torch
    for t in tr:
        t["chain"], t["group"] = chain_id, group
    tr[0]["meta"] = dict(meta or {})
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    torch.save(tr, path)
    return path


def upload_files(pairs, msg, tries=9):
    """Несколько файлов одним коммитом в датасет результатов HF: pairs — [(локальный путь или
    байты, путь в репозитории)]. Токен и повторы — как у online_eval_env.upload (лимит HF — 128
    коммитов в час на репозиторий, поэтому один коммит на заливку, а не на файл)."""
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF token — буфер только локально", flush=True)
        return False
    from huggingface_hub import CommitOperationAdd, HfApi
    for attempt in range(tries):
        try:
            HfApi().create_commit(repo_id=OUT_REPO, repo_type="dataset", token=tok, commit_message=msg,
                                  operations=[CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
                                              for src, dst in pairs])
            print(f"uploaded {len(pairs)} файлов буфера: {os.path.dirname(pairs[0][1])}", flush=True)
            return True
        except Exception as e:
            print(f"buffer upload retry {attempt}: {str(e)[:120]}", flush=True)
            time.sleep(min(600, 20 * 2 ** attempt))
    return False


def rollout_transitions(row, ro, budget):
    """Переходы прогона политики: только исполненные и оставленные действия (row — итоговая строка
    run_seed, ro — запись прогона: steps по оставленным действиям и maps). К полям сбора
    добавляются explore, rolled и guard_spent (см. докстринг модуля)."""
    steps = ro["steps"]
    res = dict(chain=row["chain"], hist=row["hist"], err=[s["err"] for s in steps], maps=ro["maps"])
    tr = chain_transitions(res, [s["a"] if s["a"] < len(S.ACTIONS) else None for s in steps], budget)
    gs = 0
    for t, s in zip(tr, steps):
        gs += sum(int(r[1]) for r in s["rolled"])
        t.update(explore=bool(s["explore"]), rolled=[list(r) for r in s["rolled"]], guard_spent=int(gs))
    return tr


def write_rollout(row, ro, args, buf_tag, run_tag):
    """Файл буфера законченного сида прогона (online_eval_env.py --save-buffer). Возвращает поле
    buffer итоговой строки: tag, saved, why (почему не записан), file, path, hf, n. Файл пишется,
    только если переходы покрывают всю исполненную цепочку с шага 0 (либо она оборвана нечисловой
    ошибкой, done=−1): докатка с чекпоинта без карт прошлых шагов или действие вне 27 — сид не записан."""
    sub = buffer_subdir(row["pde"])
    info = dict(tag=buf_tag, saved=False, why=None, n=0)
    chain = row.get("chain") or []
    if ro is None or not ro.get("complete", True) or len(ro["steps"]) != len(chain) or None in ro["steps"]:
        info["why"] = (ro or {}).get("why") or "нет записи прогона по шагам"
        return info
    tr = rollout_transitions(row, ro, int(row["budget"]))
    if not tr or (len(tr) < len(chain) and tr[-1]["done"] != -1):
        info["why"] = (f"переходы покрывают {len(tr)} шагов из {len(chain)} (нет карты перед действием или "
                       "действие вне 27)")
        return info
    seed = int(row["seed"])
    meta = dict(format=FORMAT, pde=row["pde"], tag=buf_tag, record=f"{run_tag}_seed{seed}", seed=seed,
                env=dict(keep_opt=bool(row.get("keep_opt")), mode=row.get("keep_opt_mode"), budget=int(row["budget"])),
                truncated=False, shared_prefix_len=0, stop=row.get("stop_reason"), round=None, acq="rollout",
                l2re=num(row.get("l2re")), source="rollout", policy=row.get("policy"),
                model=(getattr(args, "model_files", None) or getattr(args, "model_file", None)),
                state_mode=row.get("state_mode"),
                guard=dict(rollback=float(getattr(args, "guard_rollback", 0.0) or 0.0),
                           fallback=getattr(args, "guard_fallback", None), spent=int(row.get("guard_spent") or 0),
                           n=len(row.get("guard_log") or [])),
                explore=dict(eps=float(getattr(args, "explore_eps", 0.0) or 0.0),
                             stall=getattr(args, "explore_stall", None),
                             ratio=float(getattr(args, "stall_ratio", 0.0) or 0.0),
                             steps=[int(e[0]) for e in ro.get("explore") or []]))
    d = os.path.join(args.save_dir, "buffers_new", buf_tag, sub)
    fname = f"{run_tag}_seed{seed}.pt"
    path = save_chain(d, fname, tr, f"{buf_tag}#{run_tag}_seed{seed}", None, meta)
    info.update(saved=True, file=fname, path=path, hf=f"{BUF_HF}/{buf_tag}/{sub}/{fname}", n=len(tr),
                n_explore=sum(bool(t["explore"]) for t in tr), n_rolled=sum(len(t["rolled"]) for t in tr))
    return info


# --------------------------------------------------------------------------------------------
# Чтение: собранный буфер вместо исходного
# --------------------------------------------------------------------------------------------


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


def surrogate_table(episodes, name, subdirs="all", table="", work="agent_combo_work", log=print, src_chains=0,
                    src_seed=0):
    """Таблица для ученика (offline_rl.py --target-buffer с --mb-aux): та же, что собирает
    agent_combo.py --target-buffer — буферы остальных задач (кэш в work) и собранные цепочки цели.
    src_chains/src_seed — как agent_combo.py --src-chains/--src-seed (cap_source_chains)."""
    import agent_combo as AC
    import offline_rl as O
    meta = O.pde_meta()
    tsub = meta[name].get("subdir")
    others = [s for s in O.resolve_subdirs(subdirs) if s != tsub]
    if table or others:
        tab = AC.get_table(argparse.Namespace(table=table, work=work), others, log)
    else:
        tab = empty_table()
    if src_chains:
        tab, src_info = AC.cap_source_chains(tab, src_chains, src_seed, skip=(name,))
        log(f"источники суррогата: по {src_chains} цепочек (сид {src_seed}): {src_info}")
    tab, pi, n = table_with(tab, episodes, name, meta)
    log(f"таблица суррогата с собранным буфером цели: переходов {len(tab['A'])}, из них цели {n} "
        f"(цепочек {len(np.unique(tab['CH'][tab['PDE'] == pi]))}), задач {len(np.unique(tab['PDE']))}")
    return tab
