"""Kaggle-смок абляции: проверка окружения БЕЗ запуска обучения PINN.

Прогоняет на CPU (GPU-квота не тратится):
  1. клон ветки rlpinn_agent_ablation + установка недостающих зависимостей;
  2. smoke_test_ablation.py — все 4 режима агента на синтетическом буфере;
  3. smoke_test_hf_buffer.py — реальный буфер с HF + все 4 режима.

Если этот кернел зелёный — боевой ablation-кернел на этой же машине заведётся.
Запуск: kaggle kernels push -p experiments/optimization_multi_pde/kaggle/smoke
"""
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/florentiner/PINNacle.git"
BRANCH = "rlpinn_agent_ablation"
CLONE_DIR = "/kaggle/tmp/PINNacle"


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


def main():
    import torch

    print(f"python: {sys.version.split()[0]}, torch: {torch.__version__}, "
          f"cuda: {torch.cuda.is_available()}")

    os.makedirs(os.path.dirname(CLONE_DIR), exist_ok=True)
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    sh(f"git clone -b {BRANCH} --single-branch {REPO_URL} {CLONE_DIR}")
    sh(f"{sys.executable} -m pip install -q gym python-dotenv dill")
    ensure_torch_matches_gpu()

    os.chdir(CLONE_DIR)

    print("\n=== 1/2: смок агента (синтетический буфер, все 4 режима) ===", flush=True)
    sh(f"{sys.executable} smoke_test_ablation.py")

    print("\n=== 2/2: смок HF-буфера (реальные транзишены, все 4 режима) ===", flush=True)
    sh(f"{sys.executable} smoke_test_hf_buffer.py")

    print("\n" + "=" * 70)
    print("KAGGLE SMOKE OK: окружение готово к боевому запуску абляции")
    print("=" * 70)


if __name__ == "__main__":
    main()
