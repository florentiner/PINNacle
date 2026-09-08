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
import json
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

def slugify_cell(pde, mode, seed):
    pde_part = re.sub(r"[^a-z0-9]", "", pde.lower())[:20]
    return f"rlpinn-abl-{pde_part}-{MODE_SLUG[mode]}-s{seed}"


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

    missing_tol = [k for k in keys if PDE_SPECS[k].tolerance is None]
    if missing_tol:
        raise SystemExit(
            "У этих уравнений не откалиброван порог успеха, кампания на них "
            f"считаться не должна: {', '.join(missing_tol)}.\n"
            "Посчитайте calibrate_tolerance.py и впишите значения в реестр."
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


def assign_waves(cells, accounts, slots_per_account, start_load=None):
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
            item["kernel_slug"] = slugify_cell(cell["pde"], cell["mode"], cell["seed"])
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
    waves = assign_waves(cells, accounts, args.slots_per_account, start_load=prior)

    manifest = {
        "prefix": args.prefix,
        "modes": args.modes,
        "seeds": [int(s) for s in args.seeds],
        "max_hours": args.max_hours,
        "resume": args.resume,
        "commit": args.commit,
        "hf_results": args.hf_results,
        "hf_buffer": args.hf_buffer,
        "slots_per_account": args.slots_per_account,
        "start_spacing_sec": args.start_spacing_sec,
        "n_accounts": len(accounts),
        "waves": waves,
    }
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
        "HF_RESULTS": manifest["hf_results"],
        "HF_BUFFER": manifest["hf_buffer"],
        "HF_PREFIX": manifest["prefix"],
        "START_DELAY_SEC": str(int(cell.get("start_delay_sec", 0))),
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
            "enable_gpu": True,
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
    """Текущее состояние ячейки на Kaggle."""
    res = kaggle_cli(cell["account"], token, ["kernels", "status", cell["kernel_id"]])
    out = (res.stdout + res.stderr).strip().replace("\n", " ")
    if "403" in out and "forbidden" in out.lower():
        return "not_pushed"
    for candidate in ("complete", "running", "error", "cancelAcknowledged", "queued"):
        if candidate.lower() in out.lower():
            return candidate
    return "unknown"


def cmd_push(args):
    manifest, _ = load_manifest(args)
    wave = wave_of(manifest, args.wave)
    tokens = dict(load_accounts(args.accounts_file))

    missing = [c for c in wave if not c.get("package_dir")]
    if missing:
        raise SystemExit(f"Волна {args.wave} не собрана (build), ячеек без пакета: {len(missing)}")

    if args.retry_failed:
        # Перезапуск только упавших: запушить ячейку с тем же kernel_id — это
        # новая версия того же кернела, то есть новый запуск на том же аккаунте.
        # Работающие сессии не трогаем.
        keep, states = [], collections.Counter()
        for cell in wave:
            token = tokens.get(cell["account"])
            st = cell_state(cell, token) if token else "нет токена"
            states[st] += 1
            if st in ("error", "cancelAcknowledged", "not_pushed", "unknown"):
                keep.append(cell)
        print("Состояние волны: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
        if not keep:
            print("Перезапускать нечего — упавших ячеек нет.")
            return 0
        print(f"К перезапуску {len(keep)} ячеек:")
        for cell in keep:
            print(f"   {cell['pde']}/{cell['mode']} seed{cell['seed']}  {cell['kernel_id']}")
        wave = keep
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
        res = kaggle_cli(cell["account"], token, ["kernels", "status", cell["kernel_id"]])
        out = (res.stdout + res.stderr).strip().replace("\n", " ")
        # До первого push статус-запрос на несуществующий кернел даёт не 404,
        # а "403 Client Error: Forbidden" — без этой ветки такая ячейка
        # попадала бы в "error" наравне с реально упавшим запуском.
        if "403" in out and "forbidden" in out.lower():
            state = "not_pushed"
        else:
            state = "unknown"
            for candidate in ("complete", "running", "error", "cancelAcknowledged", "queued"):
                if candidate.lower() in out.lower():
                    state = candidate
                    break
        counts[state] = counts.get(state, 0) + 1
        print(f"  {state:20s} {cell['kernel_id']}")
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
    p_plan.add_argument("--hf-results", default="danil-e/rlpinn-ablation-runs")
    p_plan.add_argument("--hf-buffer", default="danil-e/rlpinn-ablation-buffers")
    p_plan.add_argument("--skip-done", action="store_true",
                        help="Выкинуть уравнения, уже посчитанные прошлой кампанией.")

    for name, help_text in (("build", "собрать пуш-пакеты волны"),
                            ("push", "запушить волну"),
                            ("status", "статусы кернелов волны")):
        p = sub.add_parser(name, parents=[common], help=help_text)
        p.add_argument("--wave", type=int, default=1)
        if name == "push":
            p.add_argument("--yes", action="store_true", help="подтвердить запуск сессий")
            p.add_argument("--retry-failed", action="store_true",
                           help="Запушить только ячейки, которые сейчас упали или не "
                                "запускались; работающие сессии не трогать.")

    args = parser.parse_args()
    handlers = {"accounts": cmd_accounts, "plan": cmd_plan,
                "build": cmd_build, "push": cmd_push, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
