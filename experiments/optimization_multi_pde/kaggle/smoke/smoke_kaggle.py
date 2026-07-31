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


def main():
    import torch

    print(f"python: {sys.version.split()[0]}, torch: {torch.__version__}, "
          f"cuda: {torch.cuda.is_available()}")

    os.makedirs(os.path.dirname(CLONE_DIR), exist_ok=True)
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)
    sh(f"git clone -b {BRANCH} --single-branch {REPO_URL} {CLONE_DIR}")
    sh(f"{sys.executable} -m pip install -q gym python-dotenv dill")

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
