#!/usr/bin/env python
"""Оркестратор задач итогового метода в офлайн-контейнере a-dgx.

Читает <work>/jobs.json (его собирает plan.py на маке): список задач
    {id, pde, cwd, env: {..}, cmds: [строки bash], needs: [id], gpu_slots: 1}
и выполняет их не больше --slots одновременно (по умолчанию RUN_SLOTS или 4), в порядке списка, когда все
needs выполнены. Каждая задача — `bash -c "cmd1 && cmd2 ..."` с логом <work>/logs/<id>.log. Состояние —
<work>/state.json (pending / running / done / failed, попытки, время), сводка — <work>/heartbeat.txt раз в минуту.
Упавшая задача повторяется до --retries раз (в командах стоит --resume: продолжение, а не заново).
После перезапуска контейнера задачи в состоянии running возвращаются в очередь. Ограничение CPU: каждой задаче
OMP_NUM_THREADS / MKL_NUM_THREADS (RUN_THREADS, по умолчанию 2). Выход, когда всё сделано или больше нечего
запускать (все оставшиеся ждут упавших задач).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/work")
    ap.add_argument("--slots", type=int, default=int(os.environ.get("RUN_SLOTS", "4")))
    ap.add_argument("--retries", type=int, default=int(os.environ.get("RUN_RETRIES", "2")))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("RUN_THREADS", "2")))
    ap.add_argument("--only", default=os.environ.get("RUN_ONLY", ""), help="только эти задачи (через запятую)")
    a = ap.parse_args()
    jobs = load_json(os.path.join(a.work, "jobs.json"), None)
    if not jobs:
        sys.exit(f"нет {a.work}/jobs.json")
    if a.only:
        want = set(a.only.split(","))
        jobs = [j for j in jobs if j["id"] in want]
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    spath = os.path.join(a.work, "state.json")
    st = load_json(spath, {})
    for j in jobs:
        s = st.setdefault(j["id"], dict(state="pending", attempts=0))
        if s["state"] == "running":          # контейнер перезапущен посреди задачи: в очередь (продолжит с --resume)
            s["state"] = "pending"
        if s["state"] == "failed" and os.environ.get("RUN_RESET_FAILED") == "1":   # ремонт: упавшие — снова в очередь
            s.update(state="pending", attempts=0)
    save_json(spath, st)
    by_id = {j["id"]: j for j in jobs}
    procs = {}
    stop = {"flag": False}

    def on_term(*_):
        stop["flag"] = True
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass
    signal.signal(signal.SIGTERM, on_term)
    last_beat = 0.0
    print(f"{now()} задач {len(jobs)}, слотов {a.slots}, потоков на задачу {a.threads}", flush=True)
    while True:
        # завершившиеся
        for jid, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[jid]
            s = st[jid]
            s["end"], s["rc"] = now(), rc
            if rc == 0:
                s["state"] = "done"
            elif s["attempts"] <= a.retries and not stop["flag"]:
                s["state"] = "pending"
            else:
                s["state"] = "failed"
            print(f"{now()} {jid}: rc={rc} -> {s['state']} (попыток {s['attempts']})", flush=True)
            save_json(spath, st)
        if stop["flag"]:
            break
        # запуск готовых
        for j in jobs:
            if len(procs) >= a.slots:
                break
            s = st[j["id"]]
            if s["state"] != "pending" or j["id"] in procs:
                continue
            if any(st.get(n, {}).get("state") != "done" for n in j.get("needs", []) if n in by_id):
                continue
            env = dict(os.environ)
            env.update({k: str(v) for k, v in (j.get("env") or {}).items()})
            env.setdefault("OMP_NUM_THREADS", str(a.threads))
            env.setdefault("MKL_NUM_THREADS", str(a.threads))
            env.setdefault("OPENBLAS_NUM_THREADS", str(a.threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            cwd = j.get("cwd") or a.work
            os.makedirs(cwd, exist_ok=True)
            if not os.path.exists(os.path.join(cwd, "ref")):
                os.symlink(os.environ.get("RUN_REF", "/app/ref"), os.path.join(cwd, "ref"))   # уравнения читают ref/*.dat от текущей папки
            log = open(os.path.join(a.work, "logs", f"{j['id']}.log"), "a")
            log.write(f"\n===== {now()} попытка {s['attempts'] + 1}\n")
            log.flush()
            cmd = " && ".join(f"( {c} )" for c in j["cmds"])
            procs[j["id"]] = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env, stdout=log,
                                              stderr=subprocess.STDOUT)
            s.update(state="running", attempts=s["attempts"] + 1, start=now())
            print(f"{now()} старт {j['id']} (попытка {s['attempts']})", flush=True)
            save_json(spath, st)
        # сводка
        if time.time() - last_beat > 60:
            cnt = {}
            for s in st.values():
                cnt[s["state"]] = cnt.get(s["state"], 0) + 1
            with open(os.path.join(a.work, "heartbeat.txt"), "w") as f:
                f.write(f"{now()} {json.dumps(cnt, ensure_ascii=False)} идут: {', '.join(sorted(procs))}\n")
            last_beat = time.time()
        pending = [j for j in jobs if st[j["id"]]["state"] == "pending"]
        if not procs and not pending:
            break
        if not procs and pending and all(
                any(st.get(n, {}).get("state") == "failed" for n in j.get("needs", [])) for j in pending):
            print(f"{now()} оставшиеся задачи ждут упавших — выход", flush=True)
            break
        time.sleep(15)
    cnt = {}
    for s in st.values():
        cnt[s["state"]] = cnt.get(s["state"], 0) + 1
    print(f"{now()} итог: {cnt}", flush=True)
    with open(os.path.join(a.work, "heartbeat.txt"), "w") as f:
        f.write(f"{now()} ЗАКОНЧЕНО {json.dumps(cnt, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
