"""Kaggle-кернел кампании: ОДНА сессия = ОДИН (PDE x режим абляции).

Кампания абляции: 3 уравнения x 4 режима = 12 сессий, раскиданных по
кагл-аккаунтам (по 2 сессии на аккаунт). Каждая сессия:

  1. клонирует ветку rlpinn_agent_ablation форка florentiner/PINNacle;
  2. ставит недостающие зависимости и чинит torch под выданную GPU;
  3. напрямую запускает <PDE>_ablation_chain.py c --ablation <MODE>:
     буфер с HF, оффлайн-претрен коллеги (50x5), MAX_HOURS=11 — процесс сам
     завершится, сохранит модель и выгрузит всё на HF до 12-часового лимита;
  4. весь stdout запуска дублируется в runs_.../logs/log.txt и уезжает на HF
     (tee внутри раннера) — kernel-лог Kaggle не единственная копия;
  5. в конце копирует runs_single в /kaggle/working (output кернела).

PDE, MODE и прочее задаются константами ниже (пуш-пакеты генерируются
скриптом на локальной машине; HF_TOKEN вписывается только в пушимую копию —
кернелы приватные, в git токен не попадает).

Результаты: https://huggingface.co/datasets/danil-e/rlpinn-ablation-runs
  runs_kaggle/<pde>/<mode>/<run_tag>/{logs,results,model,...}
"""
import os
import shutil
import subprocess
import sys
import time

# --- параметры сессии (правятся генератором пуш-пакетов) ---
PDE = os.getenv("PDE", "poisson_boltzmann_2d")
MODE = os.getenv("MODE", "none")
SEED = os.getenv("SEED", "1234")
MAX_HOURS = os.getenv("MAX_HOURS", "11")
HF_RESULTS = os.getenv("HF_RESULTS", "danil-e/rlpinn-ablation-runs")
HF_BUFFER = os.getenv("HF_BUFFER", "danil-e/rlpinn-ablation-buffers")
HF_PREFIX = os.getenv("HF_PREFIX", "runs_kaggle")

REPO_URL = "https://github.com/florentiner/PINNacle.git"
BRANCH = "rlpinn_agent_ablation"
CLONE_DIR = "/kaggle/tmp/PINNacle"
OUT_DIR = "/kaggle/working"


def sh(cmd, **kwargs):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def ensure_torch_matches_gpu():
    """Kaggle через API не даёт выбрать модель GPU и может выдать P100 (sm_60),
    который предустановленный torch (sm_70+) не поддерживает. Детектим и ставим
    совместимую сборку; на T4 (sm_75) реинсталл не нужен."""
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
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.getenv("HF_TOKEN")


def main():
    import socket

    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN получен: логи/результаты/модель поедут на HF.")
    else:
        print("⚠️  HF_TOKEN не найден — результаты останутся только в output кернела.")

    run_tag = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{socket.gethostname()}_seed{SEED}"
    print(f"Кампания: PDE={PDE}, MODE={MODE}, SEED={SEED}, MAX_HOURS={MAX_HOURS}, run_tag={run_tag}", flush=True)

    os.makedirs(os.path.dirname(CLONE_DIR), exist_ok=True)
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    sh(f"git clone -b {BRANCH} --single-branch {REPO_URL} {CLONE_DIR}")
    sh(f"{sys.executable} -m pip install -q gym python-dotenv dill")
    ensure_torch_matches_gpu()

    os.chdir(CLONE_DIR)
    cmd = [
        sys.executable, "-u",
        f"experiments/optimization_multi_pde/{PDE}_ablation_chain.py",
        "--ablation", MODE,
        "--seed", SEED,
        "--max-hours", MAX_HOURS,
        "--buffer-src", "hf",
        "--hf-repo", HF_BUFFER,
        "--hf-results", HF_RESULTS,
        "--hf-results-prefix", HF_PREFIX,
        "--run-tag", run_tag,
    ]
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print(f"\nраннер завершился с кодом {result.returncode}", flush=True)

    for name in ("runs_single",):
        src = os.path.join(CLONE_DIR, name)
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(OUT_DIR, name), dirs_exist_ok=True)
            print(f"скопировано в output: {name}")

    print(f"\nHF: https://huggingface.co/datasets/{HF_RESULTS}/tree/main/{HF_PREFIX}/{PDE}/{MODE}/{run_tag}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
