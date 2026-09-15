"""Kaggle-кернел расширенной кампании абляции: ОДНА сессия = ОДНА ячейка таблицы.

Ячейка — это (уравнение, режим абляции, сид). Сессия:

  1. клонирует ветку с раннером из форка florentiner/PINNacle;
  2. ставит недостающие зависимости и чинит torch под выданную GPU
     (через API нельзя выбрать модель GPU, а P100 = sm_60 предустановленным
     torch не поддерживается);
  3. запускает experiments/agent_ablation/ablation_chain.py --pde <PDE>
     --ablation <MODE>: буфер с HF, бюджет MAX_HOURS — процесс сам
     останавливается, сохраняет агента и выгружает всё на HF с запасом до
     12-часового лимита Kaggle. RESUME=auto продолжает с последнего
     чекпоинта этой ячейки, поэтому цепочка сессий наращивает обучение;
  4. stdout дублируется в runs_.../logs/log.txt и уезжает на HF (tee внутри
     раннера) — лог кернела не единственная копия;
  5. в конце копирует runs_single в /kaggle/working.

Константы ниже переписывает генератор пуш-пакетов
(experiments/agent_ablation/kaggle/build_campaign.py). HF_TOKEN попадает
только в пушимую копию: кернелы приватные, в git токен не уезжает.

Результаты: https://huggingface.co/datasets/danil-e/rlpinn-ablation-runs
  <HF_PREFIX>/<pde>/<mode>/<run_tag>/{logs,results,rl_model_snapshots,...}
"""
import os
import shlex
import shutil
import subprocess
import sys
import time

# --- параметры ячейки (переписываются генератором) ---
PDE = os.getenv("PDE", "burgers1d")
MODE = os.getenv("MODE", "none")
SEED = os.getenv("SEED", "1234")
MAX_HOURS = os.getenv("MAX_HOURS", "10.5")
# auto: продолжить с последнего чекпоинта ячейки на HF; none: с нуля
RESUME = os.getenv("RESUME", "auto")
# Конкретный коммит вместо HEAD ветки (воспроизведение на ранней ревизии)
COMMIT = os.getenv("COMMIT", "")
EXTRA_ARGS = os.getenv("EXTRA_ARGS", "")
HF_RESULTS = os.getenv("HF_RESULTS", "danil-e/rlpinn-ablation-runs")
HF_BUFFER = os.getenv("HF_BUFFER", "danil-e/rlpinn-ablation-buffers")
HF_PREFIX = os.getenv("HF_PREFIX", "runs_kaggle_v6")
# Период выгрузки на HF. Полсотни параллельных сессий коммитят в один датасет,
# и при 900 с это ~200 коммитов в час — HF начинает отвечать 429; при 1800 с
# вдвое меньше, а теряется при обрыве всё равно не больше одного интервала.
HF_SYNC_SEC = os.getenv("HF_SYNC_SEC", "1800")
# Пауза перед стартом, секунды. Ставится генератором своя на каждую ячейку:
# скачивание буфера — это запрос на каждый файл, а лимит HF (1000 запросов /
# 5 мин) общий на все сессии кампании. Полсотни сессий, стартовавших разом,
# выбивают его за минуты — так упали 5 ячеек первой волны. Пауза вычитается
# из MAX_HOURS, чтобы сессия всё равно уложилась в 12-часовой лимит Kaggle.
START_DELAY_SEC = os.getenv("START_DELAY_SEC", "0")
# Оценка обученного агента как в статье (таблица 1, приложение E): сиды прогонов
# через пробел; пусто — обычная ячейка обучения. TRAIN_PREFIX — откуда брать
# финальный чекпоинт агента, EVAL_BUDGET — бюджет эпох PINN на цепочку.
EVAL_SEEDS = os.getenv("EVAL_SEEDS", "")
EVAL_BUDGET = os.getenv("EVAL_BUDGET", "7000")
TRAIN_PREFIX = os.getenv("TRAIN_PREFIX", "")
# 1 — разрешить CPU (проверочный запуск без GPU-квоты); в кампании всегда 0
ALLOW_CPU = os.getenv("ALLOW_CPU", "0")
HF_TOKEN_EMBEDDED = ""  # подставляется генератором в пушимую копию

REPO_URL = "https://github.com/florentiner/PINNacle.git"
BRANCH = "rlpinn_agent_ablation"
RUNNER = "experiments/agent_ablation/ablation_chain.py"
CLONE_DIR = "/kaggle/tmp/PINNacle"
OUT_DIR = "/kaggle/working"


def sh(cmd, **kwargs):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def sh_retry(cmd, attempts=5, wait=30, **kwargs):
    """Как sh(), но с повтором: сетевые шаги (клон, pip) иногда падают на
    воркере Kaggle из-за мгновенного сбоя DNS — одна такая секунда стоила бы
    всего 10-часового слота сессии."""
    for attempt in range(1, attempts + 1):
        try:
            return sh(cmd, **kwargs)
        except subprocess.CalledProcessError as exc:
            if attempt == attempts:
                raise
            pause = wait * attempt
            print(f"⚠️  шаг не прошёл (попытка {attempt}/{attempts}, код {exc.returncode}); "
                  f"повтор через {pause} с", flush=True)
            time.sleep(pause)


def ensure_torch_matches_gpu():
    """Kaggle через API не даёт выбрать модель GPU и может выдать P100 (sm_60),
    который предустановленный torch (sm_70+) не поддерживает — падение
    'no kernel image is available'. На T4 (sm_75) реинсталл не нужен."""
    import torch

    if not torch.cuda.is_available():
        # Молча считать на CPU нельзя: это 10 часов слота и GPU-квоты впустую.
        # Обычная причина — недельная квота аккаунта кончилась, и Kaggle отдал
        # сессию без ускорителя, ничего не сообщив. Лучше упасть сразу: ячейка
        # пометится error, и её видно в status/--retry-failed.
        if ALLOW_CPU != "1":
            sys.exit("❌ CUDA недоступна: у аккаунта, скорее всего, кончилась "
                     "недельная GPU-квота, либо сессия выдана без ускорителя. "
                     "Обучение на CPU не имеет смысла — выходим, не тратя слот. "
                     "Перезапустить на другом аккаунте: build_campaign.py push "
                     "--retry-failed --reassign. Осознанно считать на CPU: "
                     "ALLOW_CPU=1.")
        print("⚠️  CUDA недоступна, но ALLOW_CPU=1 — продолжаем на CPU.")
        return
    cap = torch.cuda.get_device_capability(0)
    cap_tag = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU: {torch.cuda.get_device_name(0)} ({cap_tag}); "
          f"torch {torch.__version__} поддерживает: {arch_list}")
    if cap_tag not in arch_list:
        print(f"⚠️  {cap_tag} не поддержан — ставим torch 2.5.1+cu121 (sm_50..sm_90)...")
        sh(f"{sys.executable} -m pip install -q torch==2.5.1 torchvision==0.20.1 "
           f"--index-url https://download.pytorch.org/whl/cu121")
        print("torch переустановлен; обучение пойдёт в дочернем процессе с новой сборкой.")


def get_hf_token():
    """HF_TOKEN: сначала Kaggle Secrets, потом зашитый генератором, потом окружение."""
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception as exc:
        print(f"Kaggle Secrets недоступны ({exc}).")
    return HF_TOKEN_EMBEDDED or os.getenv("HF_TOKEN")


def eval_seed_done(api, eval_seed):
    """Досчитан ли прогон оценки этого сида: есть тег с results/eval_done.json."""
    base = f"{HF_PREFIX}/{PDE}/{MODE}"
    suffix = f"_eval{eval_seed}_seed{SEED}"
    try:
        tags = [e.path.split("/")[-1] for e in api.list_repo_tree(
            HF_RESULTS, path_in_repo=base, repo_type="dataset")]
    except Exception as exc:
        # Папки ещё нет (первый запуск ячейки) или сбой чтения: считаем сид
        # недосчитанным — лишний прогон дешевле пропущенного.
        print(f"   чтение {base}: {type(exc).__name__} — сид {eval_seed} считаю недосчитанным")
        return False
    for tag in sorted(t for t in tags if t.endswith(suffix)):
        try:
            names = {e.path.split("/")[-1] for e in api.list_repo_tree(
                HF_RESULTS, path_in_repo=f"{base}/{tag}/results", repo_type="dataset")}
        except Exception:
            continue
        if "eval_done.json" in names:
            return True
    return False


def run_eval(max_hours):
    """Сессия оценки: прогоны по сидам подряд, досчитанные на HF пропускаются."""
    import socket
    from huggingface_hub import HfApi

    if not TRAIN_PREFIX:
        print("❌ TRAIN_PREFIX не задан — неоткуда брать обученного агента.")
        return 1
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    start = time.time()
    budget_s = max_hours * 3600
    last_run_s = None
    codes = []
    for eval_seed in EVAL_SEEDS.split():
        if eval_seed_done(api, eval_seed):
            print(f"\nсид оценки {eval_seed}: уже досчитан на HF — пропускаю", flush=True)
            continue
        left = budget_s - (time.time() - start)
        # Прогон идёт от получаса до многих часов. Если по длительности прошлого
        # следующий не влезет, не начинаем: оборванный прогон не засчитывается
        # (нет eval_done.json), а квоту съедает. Его возьмёт следующая сессия.
        if last_run_s is not None and left < 1.15 * last_run_s:
            print(f"\n⏭  сид оценки {eval_seed}: осталось {left / 3600:.2f} ч при прошлом прогоне "
                  f"{last_run_s / 3600:.2f} ч — оставляю следующей сессии", flush=True)
            break
        tag = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{socket.gethostname()}_eval{eval_seed}_seed{SEED}"
        cmd = [
            sys.executable, "-u", RUNNER,
            "--pde", PDE,
            "--ablation", MODE,
            "--seed", eval_seed,
            "--agent-seed", SEED,
            "--eval-only",
            "--eval-budget-epochs", EVAL_BUDGET,
            "--n-trajectories", "1",
            "--max-hours", f"{max(left, 60.0) / 3600:.4f}",
            "--hf-results", HF_RESULTS,
            "--hf-results-prefix", HF_PREFIX,
            "--hf-results-sync-sec", HF_SYNC_SEC,
            "--run-tag", tag,
            "--resume-from", "auto",
            "--resume-prefix", TRAIN_PREFIX,
        ] + shlex.split(EXTRA_ARGS)
        print("\n$ " + " ".join(cmd), flush=True)
        t0 = time.time()
        result = subprocess.run(cmd)
        last_run_s = time.time() - t0
        print(f"\nпрогон оценки {eval_seed} завершился с кодом {result.returncode} "
              f"за {last_run_s / 3600:.2f} ч", flush=True)
        codes.append(result.returncode)
    print(f"\nHF: https://huggingface.co/datasets/{HF_RESULTS}/tree/main/{HF_PREFIX}/{PDE}/{MODE}")
    return 0 if all(code == 0 for code in codes) else 1


def main():
    import socket

    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN получен: логи/результаты/модель поедут на HF.")
    else:
        print("⚠️  HF_TOKEN не найден — результаты останутся только в output кернела.")

    delay = max(0.0, float(START_DELAY_SEC))
    max_hours = float(MAX_HOURS)
    if delay > 0:
        max_hours = max(1.0, max_hours - delay / 3600.0)
        print(f"⏸  Пауза перед стартом {delay/60:.0f} мин (разносим обращения к HF); "
              f"бюджет сессии уменьшен до {max_hours:.2f} ч", flush=True)
        time.sleep(delay)

    run_tag = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{socket.gethostname()}_seed{SEED}"
    print(f"Ячейка: PDE={PDE}, MODE={MODE}, SEED={SEED}, MAX_HOURS={max_hours:.2f}, "
          f"PREFIX={HF_PREFIX}, run_tag={run_tag}", flush=True)

    os.makedirs(os.path.dirname(CLONE_DIR), exist_ok=True)
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    sh_retry(f"rm -rf {CLONE_DIR} && git clone -b {BRANCH} --single-branch {REPO_URL} {CLONE_DIR}")
    if COMMIT:
        sh(f"git -C {CLONE_DIR} checkout --quiet {COMMIT}")
    sh(f"git -C {CLONE_DIR} log -1 --format='КОММИТ КОДА: %h %cd %s' --date=iso")

    runner_path = os.path.join(CLONE_DIR, RUNNER)
    if not os.path.isfile(runner_path):
        sys.exit(f"❌ В клоне нет {RUNNER}: ветка {BRANCH} ещё не содержит раннер "
                 "расширенной кампании — сначала запушьте код, потом пушьте кернелы.")

    sh_retry(f"{sys.executable} -m pip install -q gym python-dotenv dill")
    ensure_torch_matches_gpu()

    os.chdir(CLONE_DIR)
    if EVAL_SEEDS.strip():
        sys.exit(run_eval(max_hours))
    cmd = [
        sys.executable, "-u", RUNNER,
        "--pde", PDE,
        "--ablation", MODE,
        "--seed", SEED,
        "--max-hours", f"{max_hours:.4f}",
        "--buffer-src", "hf",
        "--hf-repo", HF_BUFFER,
        "--hf-results", HF_RESULTS,
        "--hf-results-prefix", HF_PREFIX,
        "--hf-results-sync-sec", HF_SYNC_SEC,
        "--run-tag", run_tag,
        "--resume-from", RESUME,
    ] + shlex.split(EXTRA_ARGS)
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print(f"\nраннер завершился с кодом {result.returncode}", flush=True)

    src = os.path.join(CLONE_DIR, "runs_single")
    if os.path.exists(src):
        shutil.copytree(src, os.path.join(OUT_DIR, "runs_single"), dirs_exist_ok=True)
        print("скопировано в output: runs_single")

    print(f"\nHF: https://huggingface.co/datasets/{HF_RESULTS}/tree/main/"
          f"{HF_PREFIX}/{PDE}/{MODE}/{run_tag}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
