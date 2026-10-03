#!/usr/bin/env python
"""
Запуск очереди экспериментов (queue.json) на Kaggle.

По умолчанию НИЧЕГО не запускает: печатает план. Пуш кернелов происходит только
с флагом --go.

    python experiments/rl_arch/queue/launch_queue.py list                # очередь и состояние
    python experiments/rl_arch/queue/launch_queue.py plan   --wave 0     # что и куда уйдёт
    python experiments/rl_arch/queue/launch_queue.py launch --wave 0 --go
    python experiments/rl_arch/queue/launch_queue.py launch --only w0-t1a,w0-t1b --go
    python experiments/rl_arch/queue/launch_queue.py status
    python experiments/rl_arch/queue/launch_queue.py plan --queue experiments/rl_arch/queue/queue_pb2d.json --wave 1

Перед пушем проверяется:
  * код, который склонирует кернел, совпадает с локальным: рабочее дерево
    experiments/rl_arch чистое и HEAD равен origin/<ветка>;
  * у аккаунта меньше двух живых сессий (лимит Kaggle — две);
  * файлы-зависимости задачи (чекпоинты агентов) уже лежат в HF-датасете, а обучение,
    которое пишет чекпоинт онлайнового агента, дошло до своего лимита часов (чекпоинт
    выгружается по ходу обучения, и оценка по нему измерила бы недоученного агента);
  * в аргументах не осталось незаполненных подстановок {BASE} и т.п.
    (значения берутся из decisions.json рядом с этим файлом).

Токены читаются из experiments/chain_eval/accounts.json (в git не попадает) и
уходят только в приватный кернел.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CHAIN_EVAL = os.path.join(REPO_ROOT, "experiments", "chain_eval")
sys.path.insert(0, CHAIN_EVAL)

import launch_kaggle as LK  # noqa: E402  (шаблон кернела, окружение токена, состояние)

QUEUE = os.path.join(HERE, "queue.json")          # по умолчанию; другой файл — через --queue
DECISIONS = os.path.join(HERE, "decisions.json")
HF_REPO = "danil-e/pinnacle-optuna-db"
LIVE = ("running", "queued", "new")


def load_queue(path=QUEUE):
    with open(path) as f:
        q = json.load(f)
    # решения общие, но у очереди другого УрЧП могут быть свои: decisions_<префикс>.json
    dec = {}
    for dp in (DECISIONS, os.path.join(HERE, f"decisions_{q.get('prefix', '')}.json") if q.get("prefix") else None):
        if dp and os.path.exists(dp):
            with open(dp) as f:
                dec.update(json.load(f))
    for j in q["jobs"]:
        for s in j["steps"]:
            for k, v in dec.items():
                s["args"] = s["args"].replace("{" + k + "}", str(v))
        j["unresolved"] = sorted({m for s in j["steps"] for m in re.findall(r"\{([A-Z_]+)\}", s["args"])})
    return q


def _hf_head(path):
    """Есть ли файл в публичном HF-датасете результатов (без токена)."""
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status < 400
    except Exception:
        return False


def _hf_json(path):
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return None


_TRAINERS = None


def _trainers():
    """{тег обучения: (id задачи, лимит часов)} по всем файлам очередей рядом. Чекпоинт
    rl_arch/agents_online/<тег>.pt пишет задача, у которой в шаге обучения стоит --tag <тег>."""
    global _TRAINERS
    if _TRAINERS is None:
        _TRAINERS = {}
        for f in sorted(glob.glob(os.path.join(HERE, "queue*.json"))):
            try:
                with open(f) as fh:
                    jobs = json.load(fh)["jobs"]
            except Exception:
                continue
            for j in jobs:
                for s in j["steps"]:
                    if not s["script"].endswith("online_train_env.py"):
                        continue
                    m = re.search(r"--tag (\S+)", s["args"])
                    h = re.search(r"--hours ([\d.]+)", s["args"])
                    if m:
                        _TRAINERS[m.group(1)] = (j["id"], float(h.group(1)) if h else 11.0)
    return _TRAINERS


def need_problem(path, state=None):
    """Почему зависимость ещё не готова; None — готова. Одного наличия файла в HF мало: чекпоинт
    онлайнового агента выгружается каждые несколько цепочек, то есть появляется задолго до конца
    обучения, и оценка по нему измерила бы недоученного агента. Поэтому для
    rl_arch/agents_online/<тег>.pt, который пишет задача одной из очередей, дополнительно
    требуется, чтобы обучение дошло до своего лимита часов: по времени пуша из pushed.json и по
    elapsed_h в строке обучения в HF. Переменная LQ_ALLOW_PARTIAL=1 снимает это требование."""
    m = re.match(r"rl_arch/agents_online/(.+)\.pt$", path)
    tr = _trainers().get(m.group(1)) if m else None
    if os.environ.get("LQ_ALLOW_PARTIAL"):
        tr = None
    if tr:
        # сначала локальная проверка без сети: обучение запущено отсюда и его лимит ещё не вышел
        jid, hours = tr
        state = LK.load_state() if state is None else state
        t = (state.get(jid) or {}).get("time")
        if t:
            try:
                left = time.mktime(time.strptime(t, "%Y-%m-%d %H:%M")) + hours * 3600 - time.time()
            except ValueError:
                left = 0.0
            if left > 0:
                return (f"обучение {jid} ещё идёт: запущено {t}, лимит {hours:g} ч, "
                        f"до конца не меньше {left / 3600:.1f} ч")
    if not _hf_head(path):
        return "файла ещё нет в HF"
    if not tr:
        return None
    row = _hf_json(f"rl_arch/online_train/{m.group(1)}.json")
    el = float(row.get("elapsed_h") or 0) if row else 0.0
    if el < hours - min(1.5, 0.25 * hours):
        return (f"обучение {jid} не дошло до лимита: в строке обучения {el:g} ч из {hours:g} "
                f"(идёт или оборвалось; оценить неполный чекпоинт можно с LQ_ALLOW_PARTIAL=1)")
    return None


def hf_exists(path):
    """Готова ли зависимость задачи: файл есть в HF, а для чекпоинта онлайнового агента
    закончено и обучение, которое его пишет (см. need_problem)."""
    return need_problem(path) is None


def git_preflight(branch):
    """Кернел клонирует ветку с GitHub: локальные правки туда не попадут."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)
    # --untracked-files=all: иначе целиком новая папка (track3/, queue/) показывается одной
    # строкой с косой чертой на конце и не распознаётся как незакоммиченный код
    dirty = run("status", "--porcelain", "--untracked-files=all", "--",
                "experiments/rl_arch", "experiments/chain_eval").stdout.strip()
    need = (".py", "pde_meta.json")          # без этих файлов кернел не отработает
    dirty = "\n".join(l for l in dirty.splitlines()
                      if not l.startswith("??") or l.endswith(need))
    head = run("rev-parse", "HEAD").stdout.strip()
    remote = run("ls-remote", "origin", f"refs/heads/{branch}").stdout.split()
    problems = []
    if dirty:
        problems.append("в experiments/rl_arch или experiments/chain_eval есть незакоммиченные "
                        "правки:\n" + dirty)
    if not remote:
        problems.append(f"ветка {branch} не найдена на origin (нет сети или нет ветки)")
    elif remote[0] != head:
        problems.append(f"HEAD {head[:9]} не равен origin/{branch} {remote[0][:9]}: "
                        f"сначала git push")
    return problems


def status_env(token):
    """Окружение для статусных вызовов CLI. KAGGLE_CONFIG_DIR из launch_kaggle здесь
    убираем: каталог пуст, и с ним CLI пишет «Could not find kaggle.json», игнорируя
    токен — отказ читается как «сессий нет», а пуш в аккаунт с двумя живыми
    сессиями отменяет чужой кернел."""
    env = LK.kaggle_env(token)
    env.pop("KAGGLE_CONFIG_DIR", None)
    return env


def kernel_status(token, ref):
    """Статус кернела или None, если прочитать не удалось."""
    try:
        out = subprocess.run(["kaggle", "kernels", "status", ref], env=status_env(token),
                             capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    m = re.search(r'status "?([A-Za-z.]+)', out.stdout)
    if not m:
        return None
    return m.group(1).split(".")[-1].lower()


def live_sessions(account, state):
    """Сколько кернелов этого аккаунта сейчас выполняется (по записям pushed.json).
    Статусы опрашиваются последовательно: параллельные вызовы CLI гонятся.
    Возвращает (число, список); число None, если хотя бы один статус не прочитан —
    тогда счётчик занятости неверен и пушить в аккаунт нельзя."""
    user = account["username"]
    refs = [v["ref"] for v in state.values() if v.get("ref", "").startswith(user + "/")]
    n, seen, unreadable = 0, [], False
    for ref in refs[-12:]:                      # свежие записи; старые давно завершены
        st = kernel_status(account["kaggle_token"], ref)
        seen.append((ref, st))
        if st is None:
            unreadable = True
        elif st in LIVE:
            n += 1
    return (None if unreadable else n), seen


def build_kernel(job, account, cfg):
    slug = f"pinnacle-chain-{job['id']}"
    kdir = os.path.join(LK.BUILD_DIR, slug)
    os.makedirs(kdir, exist_ok=True)
    jobs = [dict(script=s["script"], args=s["args"], csv_name=f"{job['id']}-{i + 1}",
                 chain_json=None, value_type="chain", hf_dir="csv_chain",
                 chain_key="chain_adam_lbfgs") for i, s in enumerate(job["steps"])]
    config = dict(repo=cfg.get("repo_url", LK.DEFAULT_REPO_URL),
                  branch=cfg.get("branch", LK.DEFAULT_BRANCH), jobs=jobs,
                  n_seeds=10, seed_base=42, seeds=None, no_upload=False, test_epochs=None,
                  display_every=100, workers_per_gpu=1,
                  hf_repo=cfg.get("hf_repo", HF_REPO),
                  hf_token_write=cfg.get("hf_token_write", ""),
                  hf_token_read=cfg.get("hf_token_read", ""), force=False)
    with open(LK.TEMPLATE) as f:
        body = f.read().replace("__CONFIG_JSON__", json.dumps(config))
    with open(os.path.join(kdir, "kernel_body.py"), "w") as f:
        f.write(body)
    meta = {"id": f"{account['username']}/{slug}", "title": slug, "code_file": "kernel_body.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return kdir, meta["id"]


def select(q, args):
    jobs = q["jobs"]
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        jobs = [j for j in jobs if j["id"] in want]
        missing = want - {j["id"] for j in jobs}
        if missing:
            sys.exit(f"нет задач: {sorted(missing)}")
    elif args.wave is not None:
        jobs = [j for j in jobs if j["wave"] == args.wave]
    return jobs


def cmd_list(q, args, state):
    for j in select(q, args):
        st = state.get(j["id"], {})
        flag = "запущена " + st["ref"] if st else "—"
        if j["unresolved"]:
            flag += f"  [нужны решения: {', '.join(j['unresolved'])}]"
        print(f"{j['id']:<22} волна {j['wave']}  {j['group']:<5} ≤{j['hours']:>5.1f} ч  {flag}")
        if args.verbose:
            for s in j["steps"]:
                print(f"      {os.path.basename(s['script'])} {s['args']}")
            if j.get("note"):
                print(f"      # {j['note']}")


def cmd_launch(q, args, state, go):
    jobs = [j for j in select(q, args) if args.force or j["id"] not in state]
    if not jobs:
        print("нечего запускать: все выбранные задачи уже в pushed.json (--force чтобы повторить)")
        return
    cfg = LK.load_accounts(args.accounts_json)
    branch = cfg.get("branch", LK.DEFAULT_BRANCH)
    problems = git_preflight(branch)
    if problems:
        print("ПРЕДПОЛЁТНАЯ ПРОВЕРКА НЕ ПРОЙДЕНА:")
        for p in problems:
            print("  - " + p)
        if go:
            sys.exit(2)
    accounts = []
    for a in cfg["accounts"]:
        if args.accounts and a["name"] not in args.accounts.split(","):
            continue
        try:
            a = dict(a, username=a.get("username") or LK.discover_username(a["kaggle_token"]))
        except SystemExit:
            print(f"[{a['name']}] токен не принят Kaggle — аккаунт пропущен")
            continue
        n, seen = live_sessions(a, state)
        if n is None:
            bad = [r for r, st in seen if st is None]
            print(f"[{a['name']}] {a['username']}: статус {len(bad)} кернел(ов) не читается "
                  f"(например {bad[0]}) — аккаунт исключён, занятость неизвестна")
            continue
        accounts.append(dict(a, live=n))
        print(f"[{a['name']}] {a['username']}: живых сессий {n}")
    pushed = 0
    for j in jobs:
        if args.limit and pushed >= args.limit:
            break
        if j["unresolved"]:
            print(f"{j['id']}: пропуск — не заданы {j['unresolved']} (decisions.json)")
            continue
        miss = [f"{n}: {p}" for n, p in ((n, need_problem(n, state)) for n in j["needs"]) if p]
        if miss:
            print(f"{j['id']}: пропуск — зависимости не готовы: " + "; ".join(miss))
            continue
        acc = next((a for a in accounts if a["live"] < 2), None)
        if acc is None:
            print(f"{j['id']}: нет аккаунта со свободной сессией — остановка")
            break
        kdir, ref = build_kernel(j, acc, cfg)
        print(f"{j['id']}: -> {ref}" + ("" if go else "   (план, без пуша)"))
        if not go:
            acc["live"] += 1
            continue
        r = subprocess.run([sys.executable, os.path.join(CHAIN_EVAL, "_push_with_shape.py"),
                            kdir, cfg.get("machine_shape", "NvidiaTeslaT4")],
                           env=LK.kaggle_env(acc["kaggle_token"]))
        if r.returncode != 0:
            # код 2 — Kaggle отклонил пуш (обычно исчерпана недельная квота GPU);
            # прочие коды — авторизация или окружение. В обоих случаях аккаунт снимаем,
            # но причины разные, и «нет квоты» говорим только при коде 2
            why = ("Kaggle отклонил пуш, вероятно исчерпана квота GPU" if r.returncode == 2
                   else "сбой авторизации или окружения")
            print(f"{j['id']}: пуш на {acc['name']} не прошёл (код {r.returncode}: {why}) — аккаунт снят")
            acc["live"] = 99
            continue
        acc["live"] += 1
        pushed += 1
        state[j["id"]] = dict(ref=ref, jobs=[j["id"]], smoke=False, account=acc["name"],
                              time=time.strftime("%Y-%m-%d %H:%M"))
        LK.save_state(state)
        time.sleep(2)
    print(f"запущено: {pushed}" if go else "это был план; для запуска добавьте --go")


def cmd_status(q, args, state):
    cfg = LK.load_accounts(args.accounts_json)
    tok = {a["name"]: a["kaggle_token"] for a in cfg["accounts"]}
    for j in select(q, args):
        st = state.get(j["id"])
        if not st:
            continue
        t = tok.get(st.get("account", ""))
        print(f"{j['id']:<22} {st['ref']:<50} "
              f"{(kernel_status(t, st['ref']) or 'не читается') if t else 'нет токена'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "plan", "launch", "status"])
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--only", default=None, help="id задач через запятую")
    ap.add_argument("--accounts", default=None, help="имена аккаунтов через запятую")
    ap.add_argument("--accounts-json", default=os.path.join(CHAIN_EVAL, "accounts.json"))
    ap.add_argument("--limit", type=int, default=0, help="не больше N пушей за вызов")
    ap.add_argument("--force", action="store_true", help="повторить уже запущенные задачи")
    ap.add_argument("--go", action="store_true", help="действительно пушить кернелы")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--queue", default=QUEUE,
                    help="файл очереди: queue.json (ns2d_liddriven) или queue_<префикс>.json другого УрЧП")
    args = ap.parse_args()
    q = load_queue(args.queue)
    state = LK.load_state()
    if args.command == "list":
        cmd_list(q, args, state)
    elif args.command == "plan":
        cmd_launch(q, args, state, go=False)
    elif args.command == "launch":
        cmd_launch(q, args, state, go=args.go)
    else:
        cmd_status(q, args, state)


if __name__ == "__main__":
    main()
