"""Kaggle-кернел: абляция DQN-стека на poisson_boltzmann_2d (все 4 режима).

Что делает:
  1. берёт HF_TOKEN из Kaggle Secrets (Add-ons -> Secrets, метка HF_TOKEN);
  2. клонирует ветку rlpinn_agent_ablation форка florentiner/PINNacle
     в /kaggle/tmp (не попадает в output кернела);
  3. ставит недостающие зависимости (gym, python-dotenv, dill) — torch и
     huggingface_hub на Kaggle уже стоят, requirements.txt не трогаем;
  4. запускает parallel-скрипт: 4 процесса абляции на одной T4, буфер с HF,
     результаты в HF-датасет danil-e/rlpinn-ablation-runs;
  5. по концу (или по бюджету MAX_HOURS) копирует runs_single и логи
     в /kaggle/working — они сохраняются как output кернела.

Лимит сессии Kaggle GPU — 12 ч, поэтому MAX_HOURS=10.5: процессы сами
останавливаются, сохраняют модель агента и делают финальную выгрузку на HF
до того, как Kaggle убьёт сессию.

Запуск: kaggle kernels push -p experiments/optimization_multi_pde/kaggle
"""
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/florentiner/PINNacle.git"
BRANCH = "rlpinn_agent_ablation"
CLONE_DIR = "/kaggle/tmp/PINNacle"
OUT_DIR = "/kaggle/working"

MAX_HOURS = os.getenv("MAX_HOURS", "10.5")
SEED = os.getenv("SEED", "1234")
MODES = os.getenv("MODES", "none no_per no_soft_watkins no_trust_region")
# Отдельная папка результатов, чтобы не мешаться с серверной кампанией в runs/
HF_PREFIX = os.getenv("HF_PREFIX", "runs_kaggle")


def sh(cmd, **kwargs):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def ensure_torch_matches_gpu():
    """Kaggle через API не даёт выбрать модель GPU и может выдать P100 (sm_60),
    который предустановленный torch (sm_70+) не поддерживает — падение
    'no kernel image is available'. Детектим и ставим совместимую сборку;
    на T4 (sm_75) реинсталл не нужен."""
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
        print("torch переустановлен; обучение пойдёт в дочерних процессах с новой сборкой.")


def get_hf_token():
    """HF_TOKEN: сначала Kaggle Secrets, потом окружение."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        print(f"Kaggle Secrets недоступны ({exc}); пробуем окружение.")
        return os.getenv("HF_TOKEN")


def main():
    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        print("HF_TOKEN получен: результаты поедут на HF.")
    else:
        print(
            "⚠️  HF_TOKEN не найден (добавьте секрет HF_TOKEN в Add-ons -> Secrets).\n"
            "    Запуск продолжится, результаты останутся только в output кернела."
        )

    os.makedirs(os.path.dirname(CLONE_DIR), exist_ok=True)
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    sh(f"git clone -b {BRANCH} --single-branch {REPO_URL} {CLONE_DIR}")
    sh(f"{sys.executable} -m pip install -q gym python-dotenv dill")
    ensure_torch_matches_gpu()

    os.chdir(CLONE_DIR)
    env = os.environ.copy()
    env.update({
        "MAX_HOURS": MAX_HOURS,
        "SEED": SEED,
        "MODES": MODES,
        "HF_PREFIX": HF_PREFIX,
        "PYTHON": sys.executable,
    })

    print(f"\n=== Запуск абляции: MODES={MODES}, MAX_HOURS={MAX_HOURS}, SEED={SEED} ===\n", flush=True)
    result = subprocess.run(
        "bash experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_parallel_run.sh",
        shell=True,
        env=env,
    )
    print(f"\nparallel_run завершился с кодом {result.returncode}")

    # Сохраняем результаты и логи как output кернела (сам клон не сохраняем).
    for name in ("runs_single", "logs"):
        src = os.path.join(CLONE_DIR, name)
        dst = os.path.join(OUT_DIR, name)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"скопировано в output: {dst}")

    print("\nГотово. Результаты также на HF: "
          "https://huggingface.co/datasets/danil-e/rlpinn-ablation-runs")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
