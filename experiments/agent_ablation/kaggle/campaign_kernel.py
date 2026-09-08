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
        print("CUDA недоступна — GPU-проверка пропущена.")
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
