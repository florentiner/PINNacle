#!/usr/bin/env python
"""Ждать, пока в HF-датасете результатов появится нужное число файлов под заданным путём.

Нужен задачам, которые зависят от других кернелов (чередование: переобучение ждёт прогоны агента),
когда снаружи запускать их по готовности некому. По истечении --hours шаг завершается успешно, если
файлов не меньше --min (обучение пойдёт на том, что есть), иначе с ошибкой.

    python experiments/rl_arch/wait_hf.py --path rl_arch/buffers_new/pb2d_it1/poisson_boltzmann_2d --n 8 --min 4 --hours 3
"""
import argparse, os, sys, time


def count(path):
    from huggingface_hub import list_repo_tree
    try:
        return sum(1 for e in list_repo_tree("danil-e/pinnacle-optuna-db", path_in_repo=path, repo_type="dataset",
                                             token=os.environ.get("HF_TOKEN") or None)
                   if e.path.endswith((".pt", ".json")))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--n", type=int, required=True, help="сколько файлов ждать")
    ap.add_argument("--min", type=int, default=1, help="с каким числом файлов идти дальше после --hours")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=180)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        k = count(a.path)
        print(f"[wait_hf] {a.path}: файлов {k} из {a.n}, прошло {(time.time() - t0) / 60:.0f} мин", flush=True)
        if k >= a.n:
            return 0
        if time.time() - t0 > a.hours * 3600:
            return 0 if k >= a.min else 3
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
