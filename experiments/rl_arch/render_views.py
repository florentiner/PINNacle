#!/usr/bin/env python
"""
Виды карт целевой задачи для предобучения энкодера без чужих буферов (map_views.py): повторный прогон цепочек
собранного буфера цели (и, по желанию, свежих случайных цепочек) с V видами каждой неплоской карты.

Зачем повторный прогон. Файлы буфера хранят карты, но не копии весов PINN, по которым карта строилась, а вид —
это другой автокодировщик на тех же копиях. Поэтому цепочка исполняется заново (тот же сид, те же действия, та
же среда сбора из meta файла: --keep-opt, режим, бюджет) — online_eval_env.run_seed со скриптом и --views V.
Для новой задачи повтор не нужен: collect_active.py --views строит виды прямо во время направленного сбора.

Экономия:
  * цепочка исполняется только до последней НЕПЛОСКОЙ карты исходного файла: плоская карта (все копии весов
    совпали — застой L-BFGS, PSO без шага) у любого автокодировщика одна и та же. В tbuf_pb2d_w0 плоских карт
    55% (с шестого действия почти все), в tbuf_ns2d_w0/w1 — 83% (все после второго действия), поэтому у ns2d
    исполняются два-три первых действия;
  * виды строятся только для неплоских состояний, ещё не встреченных (ключ — сид и исполненное начало
    цепочки: у цепочек одной группы общего префикса первые состояния одни и те же); файл, все неплоские
    состояния которого уже есть, не исполняется вовсе;
  * --worker k/K делит файлы по сидам (вся группа общего префикса — у одного работника), жадно по оценке
    эпох, чтобы работники шли примерно одинаково.
Свежие случайные цепочки (--random N): равновероятные действия из всех 27, длина --random-len, сид из
--random-seeds (срез [k::K]); цепочка останавливается после --flat-stop плоских карт подряд.

Выход: файл видов на цепочку <save-dir>/views/<тег>/<папка задачи>/<тег>_<имя исходного файла | r<сид>>.pt
(формат — map_views.py), в HF rl_arch/views/<тег>/<папка задачи>/ одним коммитом с файлом состояния
rl_arch/online_env/<pde>_<тег>_w<k>_views.json (сделанные файлы, ключи состояний, время). В конце (все файлы
сделаны или вышел --hours) — метка rl_arch/online_env/<pde>_<тег>_w<k>_views_done.json: на неё ждёт кернел
предобучения (needs). --resume продолжает по файлу состояния (локальному, иначе из HF).

    python experiments/rl_arch/render_views.py --pde poissonboltzmann2d --source rl_arch/buffers_new/tbuf_pb2d_w0 \\
        --views 12 --worker 0/2 --tag mv_pb2d --hours 4.0 --resume
    python experiments/rl_arch/render_views.py --pde ns2d_liddriven --random 6 --random-len 6 --views 12 \\
        --keep-opt --keep-opt-mode safe --tag mv_ns2d --worker 1/2 --hours 7.0 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
_T3 = os.path.join(SCRIPT_DIR, "track3")
if _T3 not in sys.path:
    sys.path.append(_T3)
os.environ.setdefault("DDEBACKEND", "pytorch")

import map_views as MV  # noqa: E402
import target_buffer as TB  # noqa: E402

OUT_REPO = "danil-e/pinnacle-optuna-db"
STATE_HF = "rl_arch/online_env"
N_ACT = 27


def parse_worker(spec):
    if not spec:
        return 0, 1
    try:
        k, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        sys.exit(f"--worker {spec!r}: нужен вид k/K, например 0/2")
    if not 0 <= k < n:
        sys.exit(f"--worker {spec!r}: нужно 0 <= k < K")
    return k, n


def parse_pool(spec):
    out = []
    for tok in spec.split(","):
        lo, _, hi = tok.strip().partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def action_tuple(t):
    """Переход буфера -> индекс действия из 27."""
    a = t["action"]
    return int(a[0]) * 9 + int(a[1]["lr"]) * 3 + int(a[1]["epochs"])


def _flat(m):
    x = np.asarray(m, dtype=np.float64)
    return bool(np.isfinite(x).all() and float(x.std()) < MV.FLAT_TOL)


def source_item(path, budget_default=7000):
    """Файл собранного буфера -> задание повтора: сид, среда, действия, неплоские состояния (после скольких
    действий) и скрипт до последнего из них. None — в файле нет неплоских карт."""
    import torch
    import online_eval_env as E
    tr = torch.load(path, map_location="cpu", weights_only=False)
    if not tr:
        return None
    meta = tr[0].get("meta") or {}
    acts = [action_tuple(t) for t in tr]
    n = len(tr)
    info = []
    for j in range(1, n):
        info.append((j, not _flat(tr[j]["state"]["loss_total"])))
    if int(tr[-1].get("done", 0)) == 0:
        info.append((n, not _flat(tr[-1]["next_state"]["loss_total"])))
    steps = [s for s, ok in info if ok]
    if not steps:
        return None
    env = dict(meta.get("env") or {})
    env.setdefault("budget", budget_default)
    chain = [list(E.ACTION_TABLE[a]) for a in acts]
    return dict(name=os.path.splitext(os.path.basename(path))[0], kind="source", source=os.path.basename(path),
                seed=int(meta["seed"]), env=env, acts=acts[:max(steps)], chain=chain, steps=steps,
                cost=int(sum(E.ACTION_TABLE[a][2] for a in acts[:max(steps)])), group=tr[0].get("group"))


def assign(items, k, n):
    """Работнику k из n: группы по сиду целиком, жадно по сумме эпох (самые дорогие группы — первыми)."""
    groups = {}
    for it in items:
        groups.setdefault(it["seed"], []).append(it)
    order = sorted(groups.items(), key=lambda kv: (-sum(i["cost"] for i in kv[1]), kv[0]))
    load, owner = [0] * n, {}
    for sd, its in order:
        w = min(range(n), key=lambda i: (load[i], i))
        owner[sd] = w
        load[w] += sum(i["cost"] for i in its)
    return [it for it in items if owner[it["seed"]] == k], load


def env_args(a, it):
    import online_eval_env as E
    env = it["env"]
    script = ",".join("%s:%g:%d" % E.ACTION_TABLE[x] for x in it["acts"])
    argv = ["--policy", "script", "--script", script, "--script-tail", "stop", "--pde", a.pde,
            "--budget", str(int(env.get("budget", a.budget))), "--save-dir", a.save_dir, "--tag", a.tag,
            "--hidden-layers", a.hidden_layers, "--ckpt-every", "0", "--ckpt-min-interval", "0",
            "--progress-every", "0", "--views", str(a.views)]
    if env.get("keep_opt"):
        argv += ["--keep-opt", "--keep-opt-mode", env.get("mode") or "all"]
    argv += ["--plain-fnn"] if a.plain_fnn else []
    argv += ["--lbfgs-tol", str(a.lbfgs_tol)] if a.lbfgs_tol is not None else []
    argv += ["--view-ae-epochs", str(a.view_ae_epochs)] if a.view_ae_epochs else []
    argv += ["--view-sequential"] if a.view_sequential else []
    if a.smoke:
        argv += ["--ae-epochs", "50", "--n-save-models", "3"]
    return E.build_parser().parse_args(argv)


def run_item(a, it, done_keys):
    """Один прогон среды; возвращает запись файла видов (map_views.make_record) и сводку."""
    import online_eval_env as E
    import branch_endings as B
    args = env_args(a, it)
    want = set()
    full_chain = [list(E.ACTION_TABLE[x]) for x in it["acts"]]
    for s in it["steps"]:
        if MV.state_key(it["seed"], full_chain, s) not in done_keys:
            want.add(int(s))
    main, views, vinfo, err, l2re, hist = {}, {}, {}, {}, {}, []
    flat_run = [0]

    def view_want(step):
        return it["kind"] == "random" or int(step) in want

    def sink(step, m3, vs, info):
        main[int(step)] = np.asarray(m3, dtype=np.float32)
        if vs:
            views[int(step)] = [np.asarray(v, dtype=np.float32) for v in vs]
        vinfo[int(step)] = {k: v for k, v in info.items()}
        flat_run[0] = flat_run[0] + 1 if info.get("flat") else 0
        if it["kind"] == "random" and a.flat_stop and flat_run[0] >= a.flat_stop:
            args.hours = 1e-9           # штатный мягкий выход среды после этой карты

    def cb(info):
        s = int(info["n_steps"])
        err[s] = float(info.get("rmse", float("nan")) + info.get("brmse", float("nan")))
        l2re[s] = float(info["l2re"])
        hist.append([int(info["spent"]), float(info["l2re"]), round(float(info.get("elapsed_s", 0.0)), 1)])

    args._view_sink, args._view_want, args._ckpt_name = sink, view_want, None
    saved, t0, row, error = B._hide_tokens(), time.time(), None, None
    try:
        row = E.run_seed(it["seed"], args, progress_cb=cb)
    except Exception as e:      # noqa: BLE001 — упавшая цепочка не роняет кернел
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"[:200]
    finally:
        B._restore_tokens(saved)
    n_exec = len(row.get("chain") or []) if row and not row.get("unfinished") else max(main, default=0)
    chain = full_chain[:max(n_exec, max(main, default=0))]
    rec = MV.make_record(a.pde, a.tag, it.get("source") or it["name"], it["seed"], it["env"], chain, main, views,
                         vinfo, err=err, l2re=l2re, hist=hist,
                         t=dict(elapsed_s=round(time.time() - t0, 1),
                                t_views_s=(row or {}).get("t_views_s"), t_ae_s=(row or {}).get("t_ae_s"),
                                t_surface_s=(row or {}).get("t_surface_s"), t_opt_s=(row or {}).get("t_opt_s")),
                         complete=error is None)
    rec["error"] = error
    got = sorted(s for s in views)
    summ = dict(name=it["name"], kind=it["kind"], seed=it["seed"], steps=len(main), want=sorted(want),
                views=got, n_maps=int(sum(len(v) for v in views.values()) + len(main)),
                flat=int(sum(1 for i in vinfo.values() if i.get("flat"))), elapsed_s=rec["t"]["elapsed_s"],
                t_views_s=rec["t"]["t_views_s"], error=error)
    return rec, summ


def state_paths(a, k):
    name = f"{a.pde}_{a.tag}_w{k}_views"
    return os.path.join(a.save_dir, name + ".json"), f"{STATE_HF}/{name}.json", f"{STATE_HF}/{name}_done.json"


def load_state(a, k):
    local, hf, _ = state_paths(a, k)
    if os.path.exists(local):
        return json.load(open(local))
    if a.smoke or a.no_upload:
        return None
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(OUT_REPO, hf, repo_type="dataset")))
    except Exception:           # noqa: BLE001 — файла ещё нет
        return None


def build_parser():
    ap = argparse.ArgumentParser(description="виды карт целевой задачи (map_views.py)")
    ap.add_argument("--pde", required=True)
    ap.add_argument("--source", default="", help="собранный буфер цели (спецификация как у --target-buffer)")
    ap.add_argument("--max-files", type=int, default=0, help="не больше N исходных файлов (после деления)")
    ap.add_argument("--random", type=int, default=0, help="свежих случайных цепочек")
    ap.add_argument("--random-len", type=int, default=6, help="действий в случайной цепочке (не больше)")
    ap.add_argument("--random-seeds", default="5000-5999")
    ap.add_argument("--flat-stop", type=int, default=2, help="случайная цепочка: стоп после стольких плоских карт подряд")
    ap.add_argument("--budget", type=int, default=7000, help="бюджет случайных цепочек (у исходных — из meta файла)")
    ap.add_argument("--keep-opt", action="store_true", help="случайные цепочки: среда без сброса оптимизатора")
    ap.add_argument("--keep-opt-mode", default="all", choices=["all", "safe"])
    ap.add_argument("--views", type=int, default=12, help="видов на неплоское состояние (кроме карты среды)")
    ap.add_argument("--view-ae-epochs", type=int, default=0, help="эпох автокодировщика видов (0 — как у среды)")
    ap.add_argument("--view-sequential", action="store_true", help="виды по одному кодом статьи (проверка)")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--plain-fnn", action="store_true")
    ap.add_argument("--lbfgs-tol", type=float, default=None)
    ap.add_argument("--worker", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--hours", type=float, default=0.0, help="мягкий предел: новая цепочка не начинается")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="бюджет 300, автокодировщик 50 эпох, 3 копии весов, без HF")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.views < 1:
        sys.exit("--views >= 1")
    if not a.source and not a.random:
        sys.exit("нужен --source (повтор собранного буфера) и/или --random N (свежие цепочки)")
    if a.flat_stop < 0 or a.random < 0 or a.random_len < 1:
        sys.exit("--random >= 0, --random-len >= 1, --flat-stop >= 0")
    if not all(c.isalnum() or c in "_-." for c in a.tag):
        sys.exit("--tag: буквы, цифры, '_', '-', '.' (это имя папки в rl_arch/views)")
    if a.smoke:
        a.budget, a.no_upload = min(a.budget, 300), True
    k, n = parse_worker(a.worker)
    try:
        from killswitch import exit_if_killed
        exit_if_killed(f"{a.tag}_w{k}", remote=not (a.smoke or a.no_upload), what="виды карт")
    except ImportError:
        pass
    sub = TB.buffer_subdir(a.pde)
    t_start = time.time()
    items, load = [], []
    if a.source:
        import glob
        for _item, d in TB.resolve(a.source, sub):
            for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
                it = source_item(p, a.budget)
                if it is None:
                    print(f"{os.path.basename(p)}: неплоских карт нет — пропуск", flush=True)
                    continue
                if a.smoke:
                    it["env"]["budget"] = min(int(it["env"]["budget"]), a.budget)
                items.append(it)
        items, load = assign(items, k, n)
        items.sort(key=lambda it: it["name"])
        if a.max_files:
            items = items[:a.max_files]
    if a.random:
        pool = parse_pool(a.random_seeds)[k::n]
        for sd in pool[:a.random]:
            rng = np.random.default_rng([int(sd), 20261016])
            acts = [int(x) for x in rng.integers(0, N_ACT, a.random_len)]
            items.append(dict(name=f"r{sd}", kind="random", source=f"random#{sd}", seed=int(sd),
                              env=dict(keep_opt=bool(a.keep_opt), mode=(a.keep_opt_mode if a.keep_opt else None),
                                       budget=int(a.budget)),
                              acts=acts, chain=None, steps=list(range(1, a.random_len + 1)), cost=0))
    local, hf_state, hf_done = state_paths(a, k)
    st = (load_state(a, k) if a.resume else None) or dict(
        pde=a.pde, tag=a.tag, worker=[k, n], done=[], keys=[], files=[], maps=0, gpu_s=0.0, finished=False)
    print(f"виды {a.tag}, работник {k}/{n}: заданий {len(items)} (исходных {sum(i['kind'] == 'source' for i in items)}, "
          f"случайных {sum(i['kind'] == 'random' for i in items)}), V={a.views}; эпох по работникам {load}; "
          f"уже сделано {len(st['done'])}", flush=True)
    done_keys = set(st["keys"])
    d_out = os.path.join(a.save_dir, "views", a.tag, sub)
    stop = None
    for it in items:
        if it["name"] in st["done"]:
            continue
        if it["kind"] == "source":
            chain = [list(c) for c in it["chain"]]
            if all(MV.state_key(it["seed"], chain, s) in done_keys for s in it["steps"]):
                print(f"{it['name']}: все неплоские состояния уже есть — пропуск", flush=True)
                st["done"].append(it["name"])
                continue
        if a.hours and (time.time() - t_start) / 3600.0 >= a.hours:
            stop = "hours"
            break
        print(f"{it['name']}: сид {it['seed']}, действий {len(it['acts'])}, неплоских состояний "
              f"{len(it['steps'])}", flush=True)
        rec, summ = run_item(a, it, done_keys)
        fname = f"{a.tag}_{it['name']}.pt"
        path = MV.save_record(os.path.join(d_out, fname), rec)
        for s in list(rec["views"]) + [s for s, i in rec["info"].items() if i.get("flat")]:
            done_keys.add(MV.state_key(rec["seed"], rec["chain"], int(s)))
        st["done"].append(it["name"])
        st["keys"] = sorted(done_keys)
        st["files"].append(dict(file=fname, **summ))
        st["maps"] += summ["n_maps"]
        st["gpu_s"] = round(st["gpu_s"] + summ["elapsed_s"], 1)
        st["elapsed_s"] = round(time.time() - t_start, 1)
        print(f"{it['name']}: карт {summ['n_maps']} (видов у состояний {summ['views']}, плоских {summ['flat']}), "
              f"{summ['elapsed_s']:.0f} с, из них виды {summ['t_views_s']} с; всего карт {st['maps']}, "
              f"{st['gpu_s'] / 3600:.2f} GPU-ч" + (f"; ОШИБКА {summ['error']}" if summ["error"] else ""), flush=True)
        os.makedirs(a.save_dir, exist_ok=True)
        json.dump(st, open(local, "w"), indent=1)
        if not a.no_upload:
            TB.upload_files([(path, f"{MV.VIEWS_HF}/{a.tag}/{sub}/{fname}"), (local, hf_state)],
                            f"rl_arch views {a.tag} w{k}: {fname}")
    st["finished"] = stop is None
    st["stop"] = stop
    st["elapsed_s"] = round(time.time() - t_start, 1)
    os.makedirs(a.save_dir, exist_ok=True)
    json.dump(st, open(local, "w"), indent=1)
    print(f"итог {a.tag} w{k}: файлов {len(st['files'])}, карт {st['maps']}, {st['gpu_s'] / 3600:.2f} GPU-ч"
          + (" (остановка по --hours)" if stop else ""), flush=True)
    if not a.no_upload:
        # метка конца: на неё ждёт предобучение; файл состояния — тот же коммит
        TB.upload_files([(local, hf_state), (json.dumps(st, indent=1).encode(), hf_done)],
                        f"rl_arch views {a.tag} w{k}: конец")
    return st


if __name__ == "__main__":
    main()
