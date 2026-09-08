"""Порог успеха траектории (tolerance) по буферу уравнения.

Зачем. В EnvRLOptimizer траектория считается успешной, когда взвешенный лосс
PINN опускается ниже tolerance (`abs(reward) < tolerance`), и этим же порогом
при загрузке буфера (`new_tol=True`) переразмечаются офлайн-цепочки. Порог
задан на уравнение и в старых экспериментах брался из tolerance-кампаний в
Comet. Для уравнений, где такой кампании не было, его надо получить из самого
буфера — иначе успех либо недостижим, либо тривиален, и success rate (главная
метрика абляции) ничего не измеряет.

Что делает скрипт: режет буфер на цепочки (цепочка кончается на done != 0),
берёт минимальный |reward| внутри цепочки — лучший лосс, которого цепочка
достигла, — и показывает, какая доля цепочек стала бы успешной при заданном
пороге. Обратная задача (--target-success-frac) выдаёт порог под нужную долю.

Порог из реестра проверяется тем же способом: для трёх уравнений кампании v5
доля успешных цепочек буфера показывает, на какую долю вообще калибровались
пороги в исходных экспериментах.

Примеры:
    python experiments/agent_ablation/calibrate_tolerance.py --pde ns2d_liddriven
    python experiments/agent_ablation/calibrate_tolerance.py --pde grayscott \
        --target-success-frac 0.5
    python experiments/agent_ablation/calibrate_tolerance.py --buffer-dir buffer_export_packed/wave1d
"""
import argparse
import math
import os
import statistics
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import get_spec

DONE_KEYS = ("abs_done", "done")


def _loss_of(tr):
    """Та же величина, которую с порогом сравнивает загрузчик буфера.

    Это НЕ поле `reward`: `set_transition_rewards_from_next_loss` перед
    обрезкой цепочек по tol перезаписывает reward на `_transition_loss_value`,
    то есть на `current_loss` (если он есть в транзишене) либо на МИНИМУМ по
    карте 26x26 `next_state["loss_total"]`. Первая версия этого скрипта мерила
    сырой reward и давала числа не из той шкалы: на heatnd она обещала 9.5%
    успешных цепочек, а реальная загрузка дала 294 успешных терминала.
    Поэтому зовём функцию самого проекта.
    """
    from RL.rl_utils.load_buffer.load_exps_from_comet import _transition_loss_value

    return _transition_loss_value(tr, loss_key="loss_total", state_loss_is_log=False)


def _field(tr, keys, default=0.0):
    for key in keys:
        if key in tr:
            return tr[key]
    return default


def load_chains(buffer_dir, max_files=None):
    """[(min|reward| по цепочке, done последнего перехода, длина)] по всему буферу."""
    import torch

    files = sorted(f for f in os.listdir(buffer_dir) if f.endswith(".pt"))
    if not files:
        raise SystemExit(f"В {buffer_dir} нет .pt-файлов (буфер не экспортирован?)")
    if max_files:
        files = files[:max_files]

    chains, n_transitions, n_no_loss = [], 0, 0
    n_current_loss = 0
    for i, fname in enumerate(files, 1):
        payload = torch.load(os.path.join(buffer_dir, fname), map_location="cpu", weights_only=False)
        transitions = payload if isinstance(payload, list) else [payload]
        current = []
        for tr in transitions:
            if not isinstance(tr, dict):
                continue
            n_transitions += 1
            if 'current_loss' in tr:
                n_current_loss += 1
            loss = _loss_of(tr)
            if loss is None or not math.isfinite(loss):
                n_no_loss += 1
                continue
            done = int(_field(tr, DONE_KEYS, 0))
            current.append(abs(float(loss)))
            if done != 0:
                chains.append((min(current), done, len(current)))
                current = []
        if current:
            chains.append((min(current), 0, len(current)))
        print(f"\r  прочитано файлов: {i}/{len(files)}, цепочек: {len(chains)}", end="", flush=True)
    print()
    return chains, n_transitions, n_no_loss, n_current_loss


def quantile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def success_frac(chains, tol):
    if not chains:
        return float("nan")
    return sum(1 for best, _, _ in chains if best <= tol) / len(chains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", type=str, default=None,
                        help="Ключ уравнения: буфер берётся с HF, порог сравнивается с реестром.")
    parser.add_argument("--buffer-dir", type=str, default=None,
                        help="Локальная папка буфера (packed-формат).")
    parser.add_argument("--hf-repo", type=str, default="danil-e/rlpinn-ablation-buffers")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Ограничить число файлов эксперимента (быстрая прикидка).")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Проверить конкретный порог (по умолчанию — из реестра).")
    parser.add_argument("--target-success-frac", type=float, default=None,
                        help="Подобрать порог под такую долю успешных цепочек буфера.")
    args = parser.parse_args()

    spec = get_spec(args.pde) if args.pde else None
    buffer_dir = args.buffer_dir
    if buffer_dir is None:
        if spec is None:
            parser.error("нужен --pde или --buffer-dir")
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                                 allow_patterns=[f"{spec.key}/*"])
        buffer_dir = os.path.join(root, spec.key)
        if not os.path.isdir(buffer_dir):
            raise SystemExit(f"В {args.hf_repo} нет папки {spec.key} — буфер не экспортирован.")

    print(f"Буфер: {buffer_dir}")
    chains, n_transitions, n_no_loss, n_current_loss = load_chains(buffer_dir, args.max_files)
    bests = sorted(best for best, _, _ in chains)
    n_done_success = sum(1 for _, done, _ in chains if done == 1)
    n_done_fail = sum(1 for _, done, _ in chains if done == -1)
    lengths = [ln for _, _, ln in chains]

    print(f"\nтранзишенов: {n_transitions} (без извлекаемого лосса: {n_no_loss}, "
          f"с полем current_loss: {n_current_loss}), цепочек: {len(chains)} "
          f"(разметка в буфере: done=1 {n_done_success}, done=-1 {n_done_fail}, "
          f"без терминала {len(chains) - n_done_success - n_done_fail})")
    print(f"длина цепочки: медиана {statistics.median(lengths):.0f}, макс {max(lengths)}")
    print("\nлучший (минимальный) |loss| внутри цепочки — квантили:")
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"   p{int(q * 100):02d}: {quantile(bests, q):.6g}")
    print(f"   min: {bests[0]:.6g}   max: {bests[-1]:.6g}")

    tol = args.tolerance if args.tolerance is not None else (spec.tolerance if spec else None)
    if tol is not None:
        frac = success_frac(chains, tol)
        source = "передан флагом" if args.tolerance is not None else "из реестра"
        print(f"\nпорог {tol:.10g} ({source}) => цепочек с достигнутым порогом {frac:.1%} "
              f"({int(round(frac * len(chains)))} из {len(chains)})")
        if frac < 0.02:
            print(f"   порог ниже всего, что видно в буфере: при new_tol=True он новых "
                  f"успехов не добавит, и агенту достанутся только уже размеченные "
                  f"done=1 ({n_done_success} цепочек, {n_done_success / len(chains):.1%}). "
                  "Если и их мало — успеху учиться не на чем.")
        elif frac > 0.9:
            print("   почти все цепочки станут успешными: success rate перестанет "
                  "различать режимы абляции (так вышло с poisson3d_complexgeometry "
                  "в кампании v5).")

    print("\nВажно: доля выше — про ОФЛАЙН-разметку буфера (сколько цепочек станет "
          "done=1 при загрузке). Онлайн порог сравнивается с текущим взвешенным train "
          "loss, а это другая величина, и офлайн-доля её не предсказывает: у "
          "ns2d_liddriven офлайн 0%, а онлайн success rate 0.84. Финальная проверка — "
          "smoke_test_hf_buffer.py --pde <ключ>: он грузит буфер настоящим загрузчиком "
          "и печатает реальное число успешных терминалов.")

    if args.target_success_frac is not None:
        proposed = quantile(bests, args.target_success_frac)
        print(f"\nпод долю успеха {args.target_success_frac:.0%} порог = {proposed:.10g}")
        print(f"   проверка: {success_frac(chains, proposed):.1%} успешных цепочек")
        if spec is not None:
            print(f"   в реестр: tolerance={proposed:.10g},  # calibrate_tolerance.py, "
                  f"доля успеха {args.target_success_frac:.0%}")


if __name__ == "__main__":
    main()
