"""Кампания абляции на Kaggle: раскладка ячеек по аккаунтам, пуш, мониторинг.

Ячейка кампании = (уравнение, режим абляции, сид). Одна ячейка считается
цепочкой Kaggle-сессий: сессия идёт ~11 ч (лимит 12), сама останавливается по
--max-hours, выгружает агента и метрики на HF, следующая сессия той же ячейки
подхватывает чекпоинт (RESUME=auto). Аккаунтов много, у каждого своя
недельная квота GPU, поэтому ячейки раскладываются волнами: в одной волне на
аккаунт не больше --slots-per-account сессий.

Токены аккаунтов лежат ВНЕ репозитория (по умолчанию
~/.kaggle/ablation_accounts.tsv, строки "<аккаунт><TAB><токен>"):
  * KGAT_... — новый формат, уходит в KAGGLE_API_TOKEN (kaggle CLI ходит с
    Bearer; в KAGGLE_KEY такой токен даёт 401);
  * classic:<user>:<key> или голый 32-символьный ключ — старый формат,
    уходит в KAGGLE_USERNAME/KAGGLE_KEY.
Пуш-пакеты тоже собираются вне репозитория: в них зашивается HF_TOKEN.

Команды:
    accounts   проверить токены (только чтение) и показать, кто есть кто
    plan       построить ячейки и разложить по аккаунтам, записать манифест
    build      собрать пуш-пакеты волны (kernel-metadata.json + кернел)
    push       запушить собранную волну (нужен --yes)
    status     опросить статусы кернелов волны

Пример:
    python experiments/agent_ablation/kaggle/build_campaign.py plan \
        --tier solvable --seeds 1234 4321 --prefix runs_kaggle_v6
    python experiments/agent_ablation/kaggle/build_campaign.py build --wave 1
    python experiments/agent_ablation/kaggle/build_campaign.py push --wave 1 --yes
    python experiments/agent_ablation/kaggle/build_campaign.py status --wave 1
"""
import argparse
import collections
import datetime
import json
import time
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import ABLATION_MODES, PDE_SPECS, select

DEFAULT_ACCOUNTS_FILE = Path.home() / ".kaggle" / "ablation_accounts.tsv"
DEFAULT_CAMPAIGN_ROOT = Path.home() / ".rlpinn_campaign"
DEFAULT_HF_TOKEN_FILE = DEFAULT_CAMPAIGN_ROOT / "hf_token"
KERNEL_TEMPLATE = Path(__file__).with_name("campaign_kernel.py")
# Недельная квота GPU на аккаунт Kaggle (по ней считается, сколько волн
# в неделю получится прогнать).
WEEKLY_GPU_QUOTA_H = 30.0
# Шаг разноса стартов сессий внутри волны, секунды (см. build).
DEFAULT_START_SPACING_SEC = 90

MODE_SLUG = {
    "none": "full",
    "no_per": "noper",
    "no_soft_watkins": "nosw",
    "no_trust_region": "notr",
}


# --- аккаунты -------------------------------------------------------------

def load_accounts(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Файл аккаунтов не найден: {path}\n"
            "Формат — строки '<аккаунт><TAB><токен>' (KGAT_... или "
            "classic:<user>:<key>). Держите его вне репозитория."
        )
    accounts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t ]+", line, maxsplit=1)
        if len(parts) != 2:
            raise SystemExit(f"Непонятная строка в {path}: {raw!r}")
        accounts.append((parts[0], parts[1]))
    if not accounts:
        raise SystemExit(f"В {path} нет ни одного аккаунта.")
    return accounts


def kaggle_env(account, token):
    """Окружение для kaggle CLI под конкретный аккаунт."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")}
    if token.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = token
    elif token.startswith("classic:"):
        _, user, key = token.split(":", 2)
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = user, key
    else:
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = account, token
    # Свой конфиг-каталог: иначе CLI подхватит ~/.kaggle/kaggle.json другого аккаунта
    env["KAGGLE_CONFIG_DIR"] = str(DEFAULT_CAMPAIGN_ROOT / "kaggle_config" / account)
    os.makedirs(env["KAGGLE_CONFIG_DIR"], exist_ok=True)
    return env


def read_hf_token(args):
    """HF-токен для зашивки в кернел: переменная окружения или файл вне репозитория."""
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    path = Path(getattr(args, "hf_token_file", None) or DEFAULT_HF_TOKEN_FILE)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def kaggle_cli(account, token, args, timeout=180):
    # У пакета kaggle нет __main__, поэтому зовём консольный скрипт, а если его
    # нет в PATH — точку входа kaggle.cli.main тем же интерпретатором.
    executable = shutil.which("kaggle")
    cmd = ([executable] if executable
           else [sys.executable, "-c", "from kaggle.cli import main; main()"]) + args
    return subprocess.run(
        cmd, env=kaggle_env(account, token), capture_output=True, text=True, timeout=timeout,
    )


def cmd_accounts(args):
    accounts = load_accounts(args.accounts_file)
    print(f"Аккаунтов в {args.accounts_file}: {len(accounts)}\n")
    ok, bad = [], []
    for account, token in accounts:
        kind = "KGAT" if token.startswith("KGAT_") else "classic"
        res = kaggle_cli(account, token, ["kernels", "list", "--user", account, "--page-size", "1"])
        out = (res.stdout + res.stderr).strip().splitlines()
        first = out[0] if out else "<пусто>"
        if res.returncode == 0 and "401" not in first and "403" not in first:
            ok.append(account)
            print(f"  OK   {account:20s} [{kind}]")
        else:
            bad.append((account, first[:120]))
            print(f"  FAIL {account:20s} [{kind}] {first[:120]}")
    print(f"\nРабочих аккаунтов: {len(ok)}/{len(accounts)}")
    if bad:
        print("Не прошли авторизацию:")
        for account, err in bad:
            print(f"   {account}: {err}")
        return 1
    return 0


# --- план кампании --------------------------------------------------------

def slugify_cell(pde, mode, seed, prefix="rlpinn-abl"):
    # У партий оценки свой префикс (rlpinn-eval): со слагом обучения кернел оценки
    # на том же аккаунте читался бы статусом завершённого кернела обучения.
    pde_part = re.sub(r"[^a-z0-9]", "", pde.lower())[:20]
    return f"{prefix}-{pde_part}-{MODE_SLUG[mode]}-s{seed}"


def build_cells(args):
    keys = list(args.pde or [])
    for tier in args.tier or []:
        keys += [k for k in select(tiers=(tier,)) if k not in keys]
    if args.skip_done:
        keys = [k for k in keys if PDE_SPECS[k].campaign != "done"]
    if not keys:
        raise SystemExit("Не выбрано ни одного уравнения (--pde/--tier).")

    unavailable = [k for k in keys if not PDE_SPECS[k].available]
    if unavailable:
        raise SystemExit(
            "Эти уравнения в ветке не реализованы, считать их нельзя: "
            f"{', '.join(unavailable)}."
        )

    # Критерий успеха по умолчанию — L2RE против эталона (формула 11 статьи),
    # и порог берётся как EPS_FACTOR x peline_l2re. Значит нужен эталон из
    # таблицы 1, а не откалиброванный по буферу loss-порог.
    missing_eps = [k for k in keys if PDE_SPECS[k].eps_l2re is None]
    if missing_eps:
        raise SystemExit(
            "У этих уравнений нет эталонного L2RE в реестре, порог успеха взять "
            f"неоткуда: {', '.join(missing_eps)}.\n"
            "Впишите peline_l2re из таблицы 1 статьи."
        )

    # Порядок ячеек — СИД-МАЖОРНЫЙ: сначала все уравнения x все режимы на первом
    # сиде, потом второй сид и т.д. Волны режутся по этому порядку, поэтому уже
    # после первой волны таблица заполнена целиком (по одному агенту в ячейке),
    # а не «два уравнения с пятью агентами и десять пустых». Заодно ошибка в
    # конкретном уравнении всплывает в первой волне, а не через три недели.
    cells = []
    for seed in args.seeds:
        for key in keys:
            for mode in args.modes:
                cells.append({"pde": key, "mode": mode, "seed": int(seed)})

    # Добор конкретных ячеек: декартово произведение режимов на сиды не умеет
    # выбрать разрозненные пары, а «тонкие» ячейки (мало траекторий у агента)
    # как раз разрозненные. Формат: pde:mode:seed через запятую.
    if getattr(args, "cells", ""):
        нужно = set()
        for item in args.cells.split(","):
            item = item.strip()
            if not item:
                continue
            части = item.split(":")
            if len(части) != 3:
                raise SystemExit(f"--cells: ожидается pde:mode:seed, получено {item!r}")
            нужно.add((части[0], части[1], int(части[2])))
        найдено = {(c["pde"], c["mode"], c["seed"]) for c in cells}
        нет = sorted(нужно - найдено)
        if нет:
            raise SystemExit("--cells: этих ячеек нет в произведении --pde x --modes x "
                             f"--seeds (опечатка?): {нет}")
        cells = [c for c in cells if (c["pde"], c["mode"], c["seed"]) in нужно]
    return cells


def existing_load(campaign_root, prefix, skip_batch=None):
    """Сколько сессий уже разложено на каждый аккаунт в других партиях этого
    префикса. Буферы приезжают порциями, партии пушатся одна за другой, и без
    этого учёта вторая партия положила бы третью сессию на аккаунт, который в
    первой уже получил две."""
    load = collections.Counter()
    base = Path(campaign_root) / prefix
    for man_path in sorted(base.glob("*/manifest.json")) + sorted(base.glob("manifest.json")):
        batch_name = man_path.parent.name if man_path.parent != base else None
        if skip_batch is not None and batch_name == skip_batch:
            continue
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for wave in man.get("waves", []):
            for cell in wave:
                load[cell["account"]] += 1
    return load


def assign_waves(cells, accounts, slots_per_account, start_load=None, slug_prefix="rlpinn-abl"):
    """Раскладка ячеек по аккаунтам: 4 режима одной строки таблицы — на разные
    аккаунты (иначе строка считается последовательно и упирается в квоту).

    Волны режутся поровну, а не «полная волна + хвост»: иначе последняя волна
    занимает пару аккаунтов, а остальные простаивают. Внутри волны аккаунт
    выбирается по наименьшей текущей загрузке (с учётом других партий), чтобы
    ни на одном не оказалось больше slots_per_account одновременных сессий.
    """
    per_wave_cap = len(accounts) * slots_per_account
    n_waves = max(1, math.ceil(len(cells) / per_wave_cap))
    names = [a for a, _ in accounts]
    load = collections.Counter(start_load or {})

    waves, start = [], 0
    for wave_index in range(n_waves):
        size = math.ceil((len(cells) - start) / (n_waves - wave_index))
        wave = []
        for cell in cells[start:start + size]:
            # наименее загруженный аккаунт; при равенстве — по порядку в файле
            account = min(names, key=lambda a: (load[a], names.index(a)))
            load[account] += 1
            item = dict(cell)
            item["account"] = account
            item["kernel_slug"] = slugify_cell(cell["pde"], cell["mode"], cell["seed"], prefix=slug_prefix)
            item["kernel_id"] = f"{account}/{item['kernel_slug']}"
            wave.append(item)
        waves.append(wave)
        start += size
    return waves


def manifest_path(root, prefix, batch=None):
    """Манифест партии. Буферы приезжают порциями, поэтому кампания пушится
    несколькими партиями; у каждой свой манифест и своя нумерация волн, а
    папка результатов на HF (prefix) у всех общая — иначе агрегатор увидит
    только последнюю партию."""
    base = Path(root) / prefix
    return (base / batch / "manifest.json") if batch else (base / "manifest.json")


def cmd_plan(args):
    accounts = load_accounts(args.accounts_file)
    cells = build_cells(args)
    prior = existing_load(args.campaign_root, args.prefix, skip_batch=args.batch)
    if prior:
        busiest = ", ".join(f"{a}={n}" for a, n in prior.most_common(3))
        print(f"Уже разложено в других партиях {args.prefix}: {sum(prior.values())} сессий "
              f"на {len(prior)} аккаунтах (самые загруженные: {busiest})")
    waves = assign_waves(cells, accounts, args.slots_per_account, start_load=prior,
                         slug_prefix=args.slug_prefix)

    manifest = {
        "prefix": args.prefix,
        "modes": args.modes,
        "seeds": [int(s) for s in args.seeds],
        "max_hours": args.max_hours,
        "resume": args.resume,
        "commit": args.commit,
        # Произвольные флаги раннеру (например --resume-prefix для проверочного
        # запуска, который берёт чекпоинт из прошлой кампании, а пишет в свою).
        "extra_args": getattr(args, "extra_args", "") or "",
        # Каждый успешный пуш пишет в ячейку pushed_at, и кернел без этой
        # отметки считается чужим (см. effective_state). Старые манифесты без
        # поля читаются по-прежнему: их ячейки пушились до появления отметки.
        "track_pushes": True,
        "hf_results": args.hf_results,
        "hf_buffer": args.hf_buffer,
        "slots_per_account": args.slots_per_account,
        "start_spacing_sec": args.start_spacing_sec,
        "n_accounts": len(accounts),
        "waves": waves,
    }
    if args.eval_seeds:
        # Партия оценки обученных агентов как в статье (таблица 1, приложение E):
        # кернел гоняет замороженного агента из train_prefix по сидам eval_seeds
        # с бюджетом eval_budget_epochs эпох PINN на цепочку.
        if not args.train_prefix:
            raise SystemExit("--eval-seeds требует --train-prefix (откуда брать обученных агентов)")
        if args.slug_prefix == "rlpinn-abl":
            raise SystemExit("у партии оценки должен быть свой --slug-prefix (например rlpinn-eval)")
        manifest["eval_seeds"] = [int(s) for s in args.eval_seeds]
        manifest["eval_budget_epochs"] = int(args.eval_budget_epochs)
        manifest["train_prefix"] = args.train_prefix
        if args.wait_train_count:
            # Ячейка не пушится, пока у агента нет нужного числа прогонов обучения
            # новее wait_train.after с финальным чекпоинтом (см. effective_state).
            for wave in waves:
                for cell in wave:
                    cell["wait_train"] = {"prefix": args.train_prefix,
                                          "after": args.wait_train_after or "",
                                          "count": int(args.wait_train_count)}
    if args.allow_cpu:
        manifest["gpu"] = False
        manifest["allow_cpu"] = True
    path = manifest_path(args.campaign_root, args.prefix, args.batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(w) for w in waves)
    gpu_hours = total * float(args.max_hours)
    per_account_wave = args.slots_per_account * float(args.max_hours)
    waves_per_week = max(1, int(WEEKLY_GPU_QUOTA_H // per_account_wave)) \
        if per_account_wave <= WEEKLY_GPU_QUOTA_H else 0
    print(f"Кампания {args.prefix}: ячеек {total}, волн {len(waves)} "
          f"(до {len(accounts) * args.slots_per_account} сессий в волне)")
    print(f"Бюджет: ~{gpu_hours:.0f} GPU-ч (по {args.max_hours} ч на сессию), "
          f"аккаунтов {len(accounts)}")
    if waves_per_week:
        print(f"Квота Kaggle ~{WEEKLY_GPU_QUOTA_H:.0f} GPU-ч в неделю на аккаунт, "
              f"волна съедает {per_account_wave:.1f} ч на аккаунт => "
              f"{waves_per_week} волна(ы) в неделю, вся кампания ~"
              f"{math.ceil(len(waves) / waves_per_week)} нед.")
    else:
        print(f"⚠️  Волна съедает {per_account_wave:.1f} ч на аккаунт при квоте "
              f"{WEEKLY_GPU_QUOTA_H:.0f} — уменьшите --slots-per-account или --max-hours.")
    for i, wave in enumerate(waves, 1):
        pdes = sorted({c['pde'] for c in wave})
        print(f"  волна {i}: сессий {len(wave)}, уравнений {len(pdes)} ({', '.join(pdes[:6])}"
              f"{'...' if len(pdes) > 6 else ''})")
    print(f"\nМанифест: {path}")
    return 0


# --- сборка пуш-пакетов ---------------------------------------------------

def load_manifest(args):
    path = manifest_path(args.campaign_root, args.prefix, args.batch)
    if not path.exists():
        raise SystemExit(f"Нет манифеста {path} — сначала plan.")
    return json.loads(path.read_text(encoding="utf-8")), path


def wave_of(manifest, number):
    waves = manifest["waves"]
    if not 1 <= number <= len(waves):
        raise SystemExit(f"Волна {number} вне диапазона 1..{len(waves)}")
    return waves[number - 1]


def render_kernel(manifest, cell, hf_token):
    text = KERNEL_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "PDE": cell["pde"],
        "MODE": cell["mode"],
        "SEED": str(cell["seed"]),
        "MAX_HOURS": str(manifest["max_hours"]),
        "RESUME": manifest["resume"],
        "COMMIT": manifest["commit"] or "",
        "EXTRA_ARGS": manifest.get("extra_args", "") or "",
        "HF_RESULTS": manifest["hf_results"],
        "HF_BUFFER": manifest["hf_buffer"],
        "HF_PREFIX": manifest["prefix"],
        "START_DELAY_SEC": str(int(cell.get("start_delay_sec", 0))),
        # пусто — ячейка обучения; сиды через пробел — ячейка оценки
        "EVAL_SEEDS": " ".join(str(s) for s in (manifest.get("eval_seeds") or [])),
        "EVAL_BUDGET": str(manifest.get("eval_budget_epochs", 7000)),
        "TRAIN_PREFIX": manifest.get("train_prefix") or "",
        "ALLOW_CPU": "1" if manifest.get("allow_cpu") else "0",
    }
    for name, value in replacements.items():
        pattern = rf'^{name} = os\.getenv\("{name}", "[^"]*"\)$'
        new_line = f'{name} = os.getenv("{name}", "{value}")'
        text, n = re.subn(pattern, new_line, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"Не удалось подставить {name} в шаблон кернела")
    text, n = re.subn(r'^HF_TOKEN_EMBEDDED = ""',
                      f'HF_TOKEN_EMBEDDED = "{hf_token}"', text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("Не удалось подставить HF_TOKEN в шаблон кернела")
    return text


def cmd_build(args):
    manifest, _ = load_manifest(args)
    wave = wave_of(manifest, args.wave)

    hf_token = read_hf_token(args)
    if not hf_token:
        print("ВНИМАНИЕ: HF_TOKEN не задан — кернелы соберутся без токена и смогут "
              f"взять его только из Kaggle Secrets (метка HF_TOKEN у аккаунта).\n"
              f"Токен берётся из переменной HF_TOKEN или из файла "
              f"{DEFAULT_HF_TOKEN_FILE} (чтобы не светить его в истории команд).")

    out_root = (Path(args.campaign_root) / manifest["prefix"]
                / (args.batch or "") / f"wave{args.wave}")
    if project_root in str(out_root.resolve()):
        raise SystemExit(f"Пакеты нельзя собирать внутри репозитория ({out_root}): "
                         "в них зашивается HF_TOKEN.")
    out_root.mkdir(parents=True, exist_ok=True)

    # Разносим старты: ячейка i ждёт i*spacing секунд. Так пик обращений к HF
    # растягивается и лимит 1000 запросов / 5 мин не выбивается.
    spacing = int(manifest.get("start_spacing_sec", DEFAULT_START_SPACING_SEC))
    for index, cell in enumerate(wave):
        cell["start_delay_sec"] = index * spacing
        cell_dir = out_root / cell["account"] / cell["kernel_slug"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "campaign_kernel.py").write_text(
            render_kernel(manifest, cell, hf_token), encoding="utf-8")
        (cell_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": cell["kernel_id"],
            "title": cell["kernel_slug"],
            "code_file": "campaign_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": manifest.get("gpu", True),
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }, indent=2), encoding="utf-8")
        cell["package_dir"] = str(cell_dir)

    path = manifest_path(args.campaign_root, args.prefix, args.batch)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Собрано пакетов: {len(wave)} в {out_root}")
    return 0


# --- пуш и статус ---------------------------------------------------------

def cell_state(cell, token):
    """Текущее состояние ячейки на Kaggle.

    Статус разбирается по строке вида KernelWorkerStatus.RUNNING, а не поиском
    подстрок в выводе. Наивный поиск слова "error" ловил транзиентные сбои
    самого API ("500 Server Error" при опросе) и объявлял упавшими живые
    сессии: один сбой в цикле опроса помечал все 24 ячейки разом. Для
    --retry-failed это означало бы перезапуск здоровых прогонов.
    Всё неразобранное — "unknown", и упавшим не считается.
    """
    res = kaggle_cli(cell["account"], token, ["kernels", "status", cell["kernel_id"]])
    out = (res.stdout + res.stderr).strip().replace("\n", " ")
    match = re.search(r"KernelWorkerStatus\.([A-Za-z_]+)", out)
    if match:
        return match.group(1).lower()
    # Кернела ещё нет: Kaggle отвечает 403 на чужой приватный и 404 на
    # несуществующий. Второе бывает, когда пуш не прошёл (сеть, сбой API), и
    # такую ячейку надо именно допушить, а не считать упавшей.
    низ = out.lower()
    if ("403" in out and "forbidden" in низ) or ("404" in out and "not found" in низ):
        return "not_pushed"
    return "unknown"


# Kaggle знает кернел только по id аккаунт/rlpinn-abl-<pde>-<mode>-s<seed>, а в
# id нет ни партии, ни захода. Второй заход той же ячейки или перенос
# (--reassign) на аккаунт, где такой слаг уже считался, читает статус СТАРОГО
# кернела: 2026-09-14 шесть ячеек w10_h2dcg_r2 оказались на аккаунтах с
# завершёнными кернелами первого захода (версии от 09-12 07:05), читались как
# complete, и наполнитель так и не запустил бы их второй заход.
TERMINAL_STATES = ("complete", "error", "cancel_acknowledged")


def effective_state(manifest, cell, token):
    """Состояние ячейки с учётом того, пушила ли её ЭТА партия.

    hold — ячейка отложена вручную (поле "hold" с причиной в манифесте): она
    не пушится и не считается готовой, пока поле не снимут.

    В манифесте с track_pushes каждый успешный пуш пишет в ячейку pushed_at.
    Завершённый или упавший кернел у ячейки без этой отметки — чужой (прошлая
    партия, прошлый заход), и ячейка на самом деле не запускалась. Работающий
    кернел так не переписываем: старые сессии давно закончились, а живую
    сессию лучше не перепушивать даже при потерянной отметке.
    """
    if cell.get("hold"):
        return "hold"
    is_eval = bool(manifest.get("eval_seeds"))
    if is_eval and cell.get("wait_train"):
        # Оценивать можно только доученного агента (иначе раннер откажется, а
        # сессия съест квоту на клон и установку).
        ready = _train_ready(manifest, cell, cell["wait_train"])
        if ready is None:
            return "unknown"
        if not ready:
            return "wait_train"
    state = cell_state(cell, token)
    if manifest.get("track_pushes") and not cell.get("pushed_at") and state in TERMINAL_STATES:
        state = "not_pushed"
    if is_eval and state in TERMINAL_STATES + ("not_pushed",):
        # Статус Kaggle ячейки оценки ничего не говорит о сидах: сессия может
        # закончиться штатно, оставив сид следующей (не хватило времени). Готова
        # только ячейка, у которой на HF досчитаны ВСЕ сиды.
        done = _eval_seeds_done(manifest, cell)
        if done is None:
            return "unknown"
        if done:
            return "complete"
        if state in TERMINAL_STATES:
            return "not_pushed"
    return state


# --- проверки по HF для партий оценки --------------------------------------
# Факты «прогон оценки досчитан» и «у прогона обучения есть финальный чекпоинт»
# со временем не отменяются, поэтому положительные ответы кэшируются на диске:
# без этого каждый вызов наполнителя заново листал бы на HF сотни папок, а лимит
# 1000 запросов / 5 мин общий со всеми сессиями кампании.
_HF_LIST_CACHE = {}
_HF_FACTS_PATH = DEFAULT_CAMPAIGN_ROOT / "hf_positive_facts.json"
_HF_FACTS = None


def _hf_facts():
    global _HF_FACTS
    if _HF_FACTS is None:
        try:
            _HF_FACTS = set(json.loads(_HF_FACTS_PATH.read_text(encoding="utf-8")))
        except Exception:
            _HF_FACTS = set()
    return _HF_FACTS


def _hf_fact_add(key):
    facts = _hf_facts()
    if key in facts:
        return
    facts.add(key)
    tmp = _HF_FACTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(facts), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _HF_FACTS_PATH)


def _hf_list(repo, path):
    """Имена записей папки датасета без рекурсии: [] — папки нет, None — сбой чтения."""
    key = (repo, path)
    if key in _HF_LIST_CACHE:
        return _HF_LIST_CACHE[key]
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN", "").strip() or (
        DEFAULT_HF_TOKEN_FILE.read_text(encoding="utf-8").strip() if DEFAULT_HF_TOKEN_FILE.exists() else None)
    api = HfApi(token=token)
    for attempt in range(4):
        try:
            names = [e.path.split("/")[-1] for e in api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset")]
            _HF_LIST_CACHE[key] = names
            return names
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "404" in text or "EntryNotFound" in text or "not found" in text.lower():
                _HF_LIST_CACHE[key] = []
                return []
            if attempt < 3:
                time.sleep(20 * (attempt + 1))
    return None


def _eval_seeds_done(manifest, cell):
    """Все ли сиды оценки ячейки досчитаны: у каждого есть тег с results/eval_done.json.
    None — HF не прочитался (статус неизвестен, ничего не перезапускаем)."""
    repo = manifest["hf_results"]
    base = f"{manifest['prefix']}/{cell['pde']}/{cell['mode']}"
    tags = None
    for eval_seed in manifest["eval_seeds"]:
        suffix = f"_eval{eval_seed}_seed{cell['seed']}"
        if any(f.startswith(f"eval_done:{repo}:{base}/") and f.endswith(suffix) for f in _hf_facts()):
            continue
        if tags is None:
            tags = _hf_list(repo, base)
            if tags is None:
                return None
        found = False
        for tag in sorted(t for t in tags if t.endswith(suffix)):
            names = _hf_list(repo, f"{base}/{tag}/results")
            if names is None:
                return None
            if "eval_done.json" in names:
                _hf_fact_add(f"eval_done:{repo}:{base}/{tag}")
                found = True
                break
        if not found:
            return False
    return True


def _train_ready(manifest, cell, wait):
    """Доучен ли агент ячейки: не меньше wait["count"] прогонов обучения новее
    wait["after"] с финальным чекпоинтом. None — HF не прочитался."""
    repo = manifest["hf_results"]
    base = f"{wait['prefix']}/{cell['pde']}/{cell['mode']}"
    after = wait.get("after") or ""
    need = int(wait.get("count", 1))
    suffix = f"_seed{cell['seed']}"
    known = {f.split(f"final:{repo}:{base}/", 1)[1] for f in _hf_facts()
             if f.startswith(f"final:{repo}:{base}/")}
    got = sum(1 for tag in known if tag.endswith(suffix) and tag >= after)
    if got >= need:
        return True
    tags = _hf_list(repo, base)
    if tags is None:
        return None
    got = 0
    for tag in sorted(t for t in tags if t.endswith(suffix) and t >= after):
        if tag not in known:
            names = _hf_list(repo, f"{base}/{tag}/model")
            if names is None:
                return None
            if "agent_final.pt" not in names:
                continue
            _hf_fact_add(f"final:{repo}:{base}/{tag}")
        got += 1
    return got >= need


def cmd_stop(args):
    """Останавливает сессии волны.

    Программной паузы у Kaggle нет: единственный способ снять работающую
    сессию — удалить кернел (kaggle kernels delete). Результаты от этого не
    теряются, они лежат на HF; следующий push создаёт кернел заново, а
    resume=auto подхватывает последний чекпоинт.
    """
    manifest, _ = load_manifest(args)
    wave = wave_of(manifest, args.wave)
    tokens = dict(load_accounts(args.accounts_file))
    if args.only_pde:
        wave = [c for c in wave if c["pde"] in set(args.only_pde)]

    if not args.yes:
        print(f"Волна {args.wave}: будет снято {len(wave)} сессий "
              f"(kaggle kernels delete). Повторите с --yes.")
        for cell in wave[:10]:
            print(f"   {cell['kernel_id']}")
        if len(wave) > 10:
            print(f"   ... ещё {len(wave) - 10}")
        return 1

    stopped, failed = 0, []
    for cell in wave:
        token = tokens.get(cell["account"])
        if not token:
            failed.append((cell["kernel_id"], "нет токена"))
            continue
        res = kaggle_cli(cell["account"], token,
                         ["kernels", "delete", "-y", cell["kernel_id"]], timeout=300)
        out = (res.stdout + res.stderr).strip().replace("\n", " ")
        ok = res.returncode == 0
        stopped += ok
        if not ok:
            failed.append((cell["kernel_id"], out[:160]))
        print(f"{'OK  ' if ok else 'FAIL'} {cell['kernel_id']}: {out[:110]}")
    print(f"\nСнято: {stopped}/{len(wave)}")
    for kid, why in failed:
        print(f"   не снят {kid}: {why}")
    return 0 if not failed else 1


def _version_created(cell, token):
    """Время создания последней версии кернела — по листингу выходных файлов.

    Kaggle печатает у каждого файла дату создания версии («8:41 pm, Saturday 12
    September 2026 UTC»). Если выходных файлов нет (сессия умерла до старта
    раннера), вернём None, и ячейка пойдёт в перезапуск как раньше.
    """
    res = kaggle_cli(cell["account"], token, ["kernels", "files", cell["kernel_id"]], timeout=240)
    out = res.stdout + res.stderr
    m = re.search(r"(\d{1,2}:\d{2} [ap]m), \w+ (\d{1,2} \w+ \d{4}) UTC", out)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%I:%M %p %d %B %Y").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def _hf_finished_after(manifest, cell, since_utc, hf_token):
    """Тег завершённого на HF прогона ячейки, начатого не раньше since_utc, или None.

    Ограничение по времени принципиально: у ячейки продолжения (например,
    второй заход) уже есть завершённый прогон первого захода, и без сравнения
    с моментом пуша умершее на старте продолжение сочлось бы досчитанным.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token)
    base = f"{manifest['prefix']}/{cell['pde']}/{cell['mode']}"
    try:
        entries = api.list_repo_tree(manifest["hf_results"], path_in_repo=base,
                                     repo_type="dataset", recursive=False)
        tags = sorted(e.path.split("/")[-1] for e in entries
                      if e.path.split("/")[-1].endswith(f"_seed{cell['seed']}"))
    except Exception:
        return None
    for tag in reversed(tags):
        m = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", tag)
        if not m:
            continue
        started = datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=datetime.timezone.utc)
        if started < since_utc - datetime.timedelta(minutes=5):
            break
        try:
            local = hf_hub_download(manifest["hf_results"], repo_type="dataset",
                                    filename=f"{base}/{tag}/results/status.json", token=hf_token)
            state = json.loads(open(local, encoding="utf-8").read()).get("state")
        except Exception:
            continue
        if state == "finished":
            return tag
    return None


def cmd_push(args):
    manifest, _ = load_manifest(args)
    wave = wave_of(manifest, args.wave)
    tokens = dict(load_accounts(args.accounts_file))

    if manifest.get("eval_seeds") and not args.retry_failed:
        raise SystemExit("Партии оценки пушатся только с --retry-failed: ячейки ждут обучения "
                         "агента, а досчитанность сидов проверяется по HF.")

    missing = [c for c in wave if not c.get("package_dir")]
    if missing:
        raise SystemExit(f"Волна {args.wave} не собрана (build), ячеек без пакета: {len(missing)}")

    if args.retry_failed:
        # Перезапуск только упавших: запушить ячейку с тем же kernel_id — это
        # новая версия того же кернела, то есть новый запуск на том же аккаунте.
        # Работающие сессии не трогаем.
        keep, states = [], collections.Counter()
        finished_error, hf_token_cache = [], [None]
        held = []
        for cell in wave:
            token = tokens.get(cell["account"])
            if cell.get("hold"):
                # Считаем в состоянии волны: наполнитель не должен принять
                # волну с отложенной ячейкой за готовую.
                states["hold"] += 1
                held.append(cell)
                continue
            st = effective_state(manifest, cell, token) if token else "нет токена"
            if st in ("error", "cancel_acknowledged") and token:
                # Одно чтение статуса Kaggle ненадёжно (см. _hf_finished_after).
                # Ложные ERROR до сих пор ловились только на завершённых
                # кернелах, но перепуш РАБОТАЮЩЕГО кернела по такому чтению
                # оборвал бы живую сессию. Перед перезапуском читаем ещё раз.
                time.sleep(5)
                st = effective_state(manifest, cell, token)
            states[st] += 1
            # "unknown" сюда НЕ входит: это чаще всего сбой опроса, а не
            # упавшая сессия, и перезапуск убил бы живой прогон.
            if st == "error":
                # ERROR часто означает не падение, а ненулевой код выхода в
                # самом конце: 429 при финальной выгрузке на HF. Данные при этом
                # целы (2026-09-13: 4 из 4 проверенных ERROR-ячеек досчитаны).
                # Повтор такой ячейки жжёт квоту впустую, поэтому ищем на HF
                # завершённый прогон, начатый после пуша этой версии.
                since = None
                if cell.get("pushed_at"):
                    since = datetime.datetime.fromisoformat(cell["pushed_at"])
                elif token:
                    since = _version_created(cell, token)
                if since is not None:
                    if hf_token_cache[0] is None:
                        hf_token_cache[0] = read_hf_token(args)
                    tag = _hf_finished_after(manifest, cell, since, hf_token_cache[0])
                    if tag:
                        states["error"] -= 1
                        states["finished_error"] += 1
                        finished_error.append((cell, tag))
                        continue
                keep.append(cell)
            elif st in ("cancel_acknowledged", "not_pushed"):
                keep.append(cell)
        print("Состояние волны: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items()) if v))
        for cell in held:
            print(f"   отложена (hold) — не пушу и готовой не считаю: {cell['pde']}/{cell['mode']} "
                  f"seed{cell['seed']}: {cell['hold']}")
        for cell, tag in finished_error:
            print(f"   досчитана (ERROR только в коде выхода, прогон {tag} finished) — не перезапускаю: "
                  f"{cell['pde']}/{cell['mode']} seed{cell['seed']} {cell['kernel_id']}")
        if not keep:
            непрочитано = states.get("unknown", 0) + states.get("нет токена", 0)
            if непрочитано:
                # Статус не прочитан — сбой или троттлинг API Kaggle, либо нет
                # токена. Это НЕ «всё в порядке»: такие ячейки могут быть так и
                # не запущены. Раньше здесь печаталось «Перезапускать нечего», и
                # наполнитель дважды принимал незапущенные партии за готовые.
                # Формулировка сознательно без слов «Перезапускать нечего».
                print(f"Статус {непрочитано} ячеек не прочитан (unknown/нет токена) — "
                      "ничего не перезапускаю, повторите позже.")
                return 2
            print("Перезапускать нечего — упавших ячеек нет.")
            return 0
        if args.reassign:
            # Аккаунт может быть устойчиво нерабочим: на Kaggle интернет из
            # кернела включается только после подтверждения телефона, и без
            # него git clone падает с "Could not resolve host" на всех попытках.
            # Перевешиваем такую ячейку на наименее загруженный другой аккаунт.
            load = existing_load(args.campaign_root, args.prefix)
            # Кандидатов для переноса можно ограничить отдельным файлом. Токены
            # для опроса статусов нужны по ВСЕМ аккаунтам (иначе ячейки на
            # исчерпанных аккаунтах получают «нет токена» и в перенос не
            # попадают), а переносить надо только туда, где квота ещё есть.
            names = [a for a, _ in load_accounts(args.reassign_to or args.accounts_file)]
            plan = []
            for cell in keep:
                old_account = cell["account"]
                candidates = [a for a in names if a != old_account]
                if not candidates:
                    continue
                new_account = min(candidates, key=lambda a: (load[a], names.index(a)))
                load[new_account] += 1
                load[old_account] = max(0, load[old_account] - 1)
                plan.append((cell, old_account, new_account))
                print(f"   переношу {cell['pde']}/{cell['mode']}: "
                      f"{old_account} -> {new_account}")
        # Перенос применяем только вместе с реальным пушем. Иначе предпросмотр
        # (без --yes) записал бы новый аккаунт в манифест, и следующий вызов
        # увидел бы его занятым и перевесил ячейку обратно на упавший аккаунт.
        if args.reassign and args.yes:
            for cell, old_account, new_account in plan:
                cell["account"] = new_account
                cell["kernel_id"] = f"{new_account}/{cell['kernel_slug']}"
                cell["reassigned_from"] = old_account
            # пакеты надо пересобрать под новый аккаунт
            hf_token = read_hf_token(args)
            for cell, _, _ in plan:
                cell_dir = (Path(args.campaign_root) / manifest["prefix"] / (args.batch or "")
                            / f"wave{args.wave}" / cell["account"] / cell["kernel_slug"])
                cell_dir.mkdir(parents=True, exist_ok=True)
                (cell_dir / "campaign_kernel.py").write_text(
                    render_kernel(manifest, cell, hf_token), encoding="utf-8")
                (cell_dir / "kernel-metadata.json").write_text(json.dumps({
                    "id": cell["kernel_id"], "title": cell["kernel_slug"],
                    "code_file": "campaign_kernel.py", "language": "python",
                    "kernel_type": "script", "is_private": True, "enable_gpu": manifest.get("gpu", True),
                    "enable_tpu": False, "enable_internet": True, "keywords": [],
                    "dataset_sources": [], "kernel_sources": [],
                    "competition_sources": [], "model_sources": [],
                }, indent=2), encoding="utf-8")
                cell["package_dir"] = str(cell_dir)
            manifest_path(args.campaign_root, args.prefix, args.batch).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.reassign:
            print("   (перенос запишется в манифест только вместе с --yes)")

        print(f"К перезапуску {len(keep)} ячеек:")
        for cell in keep:
            print(f"   {cell['pde']}/{cell['mode']} seed{cell['seed']}  {cell['kernel_id']}")
        wave = keep
    else:
        for cell in wave:
            if cell.get("hold"):
                print(f"   отложена (hold) — не пушу: {cell['pde']}/{cell['mode']} "
                      f"seed{cell['seed']}: {cell['hold']}")
        wave = [c for c in wave if not c.get("hold")]
    if getattr(args, "max_new", None) is not None and len(wave) > args.max_new:
        print(f"Ограничение --max-new {args.max_new}: пушу {args.max_new} из {len(wave)} ячеек.")
        wave = wave[:max(0, args.max_new)]
    if not args.yes:
        print(f"Волна {args.wave}: {len(wave)} сессий на {len({c['account'] for c in wave})} "
              "аккаунтах. Это запустит счётчик GPU-квоты. Повторите с --yes.")
        for cell in wave[:10]:
            print(f"   {cell['account']:20s} {cell['kernel_id']}")
        if len(wave) > 10:
            print(f"   ... ещё {len(wave) - 10}")
        return 1

    failures = []
    for cell in wave:
        token = tokens.get(cell["account"])
        if not token:
            failures.append((cell["kernel_id"], "нет токена в файле аккаунтов"))
            continue
        res = kaggle_cli(cell["account"], token,
                         ["kernels", "push", "-p", cell["package_dir"]], timeout=600)
        out = (res.stdout + res.stderr).strip()
        ok = res.returncode == 0 and "error" not in out.lower()
        if ok:
            # Момент пуша нужен, чтобы отличать прогон ЭТОЙ версии от старых.
            cell["pushed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            manifest_path(args.campaign_root, args.prefix, args.batch).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{'OK  ' if ok else 'FAIL'} {cell['kernel_id']}: "
              f"{out.splitlines()[-1] if out else '<пусто>'}")
        if not ok:
            failures.append((cell["kernel_id"], out[:200]))
    print(f"\nЗапушено: {len(wave) - len(failures)}/{len(wave)}")
    return 1 if failures else 0


def cmd_status(args):
    manifest, _ = load_manifest(args)
    wave = wave_of(manifest, args.wave)
    tokens = dict(load_accounts(args.accounts_file))
    counts = {}
    for cell in wave:
        token = tokens.get(cell["account"])
        if not token:
            print(f"  ?  {cell['kernel_id']}: нет токена")
            continue
        # Тот же разбор, что у push. Свой поиск подстрок находил слово "Error"
        # в "404 Client Error: Not Found" и показывал незапушенные ячейки
        # упавшими (2026-09-14: 4 ячейки в w10_h2dcg_r2 и 2 в w7_h2dms).
        state = effective_state(manifest, cell, token)
        counts[state] = counts.get(state, 0) + 1
        print(f"  {state:20s} {cell['kernel_id']}")
        if state == "hold":
            print(f"      -> отложена: {cell['hold']}")
    print("\nИтого:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def main():
    # Общие флаги (--prefix, --accounts-file, ...) объявлены на родительском
    # парсере и подмешаны в main-парсер И в каждый подпарсер через parents=.
    # Без этого argparse принимает их только ДО имени подкоманды
    # ("build_campaign.py --prefix X plan"), а весь README документирует
    # обратный порядок ("build_campaign.py plan ... --prefix X") — он и был
    # тем порядком, что реально хочет напечатать пользователь.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--accounts-file", default=str(DEFAULT_ACCOUNTS_FILE))
    common.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    common.add_argument("--hf-token-file", default=str(DEFAULT_HF_TOKEN_FILE),
                        help="Файл с HF-токеном для зашивки в кернелы "
                             "(если не задана переменная HF_TOKEN).")
    common.add_argument("--prefix", default="runs_kaggle_v6",
                        help="Папка кампании на HF (общая для всех партий).")
    common.add_argument("--batch", default=None,
                        help="Имя партии: свой манифест и своя нумерация волн под "
                             "тем же --prefix. Буферы приезжают порциями, каждую "
                             "порцию пушим своей партией.")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("accounts", parents=[common], help="проверить токены аккаунтов")

    p_plan = sub.add_parser("plan", parents=[common],
                            help="построить ячейки и разложить по аккаунтам")
    p_plan.add_argument("--pde", action="append", default=None)
    p_plan.add_argument("--tier", action="append", default=None,
                        choices=["solvable", "borderline", "unsolvable"])
    p_plan.add_argument("--modes", nargs="+", default=list(ABLATION_MODES),
                        choices=list(ABLATION_MODES))
    # Пять независимо обученных агентов на ячейку: разброс между ними — то,
    # что rebuttal назвал ограничением одиночного прогона, и то, ради чего
    # претрен поднят до 500 шагов.
    p_plan.add_argument("--seeds", nargs="+",
                        default=["1234", "4321", "2718", "3141", "5772"])
    p_plan.add_argument("--slots-per-account", type=int, default=2,
                        help="Сколько сессий на аккаунт в одной волне.")
    # 10.5 при лимите сессии 12 ч. По дедлайну раннер не начинает новый чанк
    # оптимизатора, но текущий досчитывает — а чанк LBFGS на тяжёлом уравнении
    # идёт до часа. Плюс финальная выгрузка на HF с ретраями (до ~8 мин).
    # 10.5 + 1 + 0.15 = 11.65 ч, запас есть; при 10.75 его почти нет.
    p_plan.add_argument("--max-hours", type=float, default=10.5)
    p_plan.add_argument("--start-spacing-sec", type=int, default=DEFAULT_START_SPACING_SEC,
                        help="Разнос стартов сессий внутри волны, секунд на ячейку. "
                             "Защищает от лимита HF (1000 запросов / 5 мин на аккаунт, "
                             "общий для всех сессий). Пауза вычитается из бюджета сессии.")
    p_plan.add_argument("--resume", default="auto", choices=["auto", "none"])
    p_plan.add_argument("--commit", default="",
                        help="Пин коммита раннера (иначе HEAD ветки).")
    p_plan.add_argument("--extra-args", default="",
                        help="Дополнительные флаги раннеру, одной строкой "
                             "(например \"--resume-prefix runs_kaggle_v8\").")
    p_plan.add_argument("--hf-results", default="danil-e/rlpinn-ablation-runs")
    p_plan.add_argument("--hf-buffer", default="danil-e/rlpinn-ablation-buffers")
    p_plan.add_argument("--skip-done", action="store_true",
                        help="Выкинуть уравнения, уже посчитанные прошлой кампанией.")
    p_plan.add_argument("--cells", default="",
                        help="Только эти ячейки: pde:mode:seed через запятую "
                             "(добор тонких ячеек после основной партии).")
    p_plan.add_argument("--slug-prefix", default="rlpinn-abl",
                        help="Префикс слага кернела. У партий оценки свой (rlpinn-eval), "
                             "чтобы не пересекаться с кернелами обучения.")
    p_plan.add_argument("--eval-seeds", nargs="+", default=None,
                        help="Партия ОЦЕНКИ обученных агентов как в статье: сиды прогонов "
                             "(одинаковые у всех агентов — парное сравнение режимов).")
    p_plan.add_argument("--eval-budget-epochs", type=int, default=7000,
                        help="Бюджет эпох PINN на цепочку оценки (в статье 7000).")
    p_plan.add_argument("--train-prefix", default=None,
                        help="Префикс результатов обучения, откуда брать финальных агентов.")
    p_plan.add_argument("--wait-train-after", default="",
                        help="Ячейка оценки ждёт прогонов обучения с тегом не раньше этого "
                             "(YYYY-MM-DD_HH-MM-SS).")
    p_plan.add_argument("--wait-train-count", type=int, default=0,
                        help="Сколько таких прогонов с финальным чекпоинтом нужно (0 — не ждать).")
    p_plan.add_argument("--allow-cpu", action="store_true",
                        help="Кернел без GPU (проверочный прогон, GPU-квоту не тратит).")

    for name, help_text in (("build", "собрать пуш-пакеты волны"),
                            ("push", "запушить волну"),
                            ("stop", "снять работающие сессии волны"),
                            ("status", "статусы кернелов волны")):
        p = sub.add_parser(name, parents=[common], help=help_text)
        p.add_argument("--wave", type=int, default=1)
        if name == "stop":
            p.add_argument("--yes", action="store_true",
                           help="подтвердить удаление кернелов")
            p.add_argument("--only-pde", action="append", default=None,
                           help="Снять только ячейки этих уравнений.")
        if name == "push":
            p.add_argument("--yes", action="store_true", help="подтвердить запуск сессий")
            p.add_argument("--reassign-to", default=None,
                           help="Файл аккаунтов-кандидатов для --reassign (по умолчанию "
                                "--accounts-file). Нужен, когда часть флота упёрлась в "
                                "недельную квоту: статусы опрашиваются по полному файлу, "
                                "а переносить надо только на аккаунты с квотой.")
            p.add_argument("--reassign", action="store_true",
                           help="Перевесить упавшие ячейки на другой аккаунт (когда "
                                "аккаунт устойчиво нерабочий, например без интернета).")
            p.add_argument("--retry-failed", action="store_true",
                           help="Запушить только ячейки, которые сейчас упали или не "
                                "запускались; работающие сессии не трогать.")
            p.add_argument("--max-new", type=int, default=None,
                           help="Запушить не больше N ячеек за вызов: наполнитель так "
                                "ограничивает долю флота под оценку.")

    args = parser.parse_args()
    handlers = {"accounts": cmd_accounts, "plan": cmd_plan,
                "build": cmd_build, "push": cmd_push, "stop": cmd_stop,
                "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
