"""Спаренный по сиду тест эффекта компонентов — по уравнению и объединённый.

Зачем отдельно от collect_results.py. Агенты одного сида различаются ТОЛЬКО
отключённым компонентом: инициализация сети, порядок буфера и предобучение у
них общие. Значит разницу надо считать попарно внутри сида, а не сравнивать
две независимые доли — иначе весь разброс между агентами (в кампании v10 у
`poissoninv` доля успеха у `no_per` гуляла 0.33..1.00 по сидам) уходит в шум и
съедает любой эффект.

Пяти сидов на одно уравнение для эффекта наблюдаемого размера мало: нужно
порядка 8-10. Поэтому главный результат здесь — ОБЪЕДИНЁННЫЙ тест, где единицей
наблюдения служит пара (уравнение, сид). На 10-14 уравнениях это 50-70 пар
вместо пяти, и мощности хватает.

Вход — файл <out>.by_agent.csv, который пишет collect_results.py (одна строка
на агента: уравнение, режим, сид, успехи, провалы). Скачивать ничего не надо.

Запуск:
    python experiments/agent_ablation/collect_results.py --prefix runs_kaggle_v10 --out v10.csv
    python experiments/agent_ablation/paired_analysis.py v10.by_agent.csv
    python experiments/agent_ablation/paired_analysis.py v10.by_agent.csv --exclude-ceiling
"""
import argparse
import collections
import csv
import math
import statistics

ABLATIONS = [("no_per", "no PER"), ("no_soft_watkins", "no soft-Watkins"),
             ("no_trust_region", "no trust region")]
FULL = "none"
# t_{0.975} по числу степеней свободы; дальше нормальное приближение.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit(df):
    if df <= 0:
        return float("nan")
    for k in sorted(T975):
        if df <= k:
            return T975[k]
    return 1.96


def ci(values):
    """Среднее и 95% интервал по t-распределению."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(values)
    if n == 1:
        return mean, float("nan"), float("nan")
    hw = t_crit(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - hw, mean + hw


def sign_test(values, eps=1e-12):
    """Двусторонний знаковый тест: сколько пар в плюс, сколько в минус, p."""
    pos = sum(1 for v in values if v > eps)
    neg = sum(1 for v in values if v < -eps)
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    # P(X <= k) + P(X >= n-k) при p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def load(path):
    rate, full_rate = {}, collections.defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n_success"]) + int(r["n_fail"])
            if n == 0:
                continue
            key = (r["pde_name"], r["ablation"], r["seed"])
            rate[key] = int(r["n_success"]) / n
            if r["ablation"] == FULL:
                full_rate[r["pde_name"]][r["seed"]] = rate[key]
    return rate, full_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("by_agent_csv", help="Файл <out>.by_agent.csv из collect_results.py.")
    parser.add_argument("--exclude-ceiling", action="store_true",
                        help="Выбросить уравнения, где ПОЛНЫЙ агент успешен во всех "
                             "сидах (доля 1.00): там разнице негде проявиться. Фильтр "
                             "смотрит только на полного агента, не на сравниваемые "
                             "режимы, поэтому не подгоняет результат.")
    args = parser.parse_args()

    rate, full_rate = load(args.by_agent_csv)
    pdes = sorted(full_rate)
    ceiling = {p for p in pdes if full_rate[p] and min(full_rate[p].values()) >= 1.0}
    used = [p for p in pdes if not (args.exclude_ceiling and p in ceiling)]

    print(f"уравнений в файле {len(pdes)}; у потолка (полный агент 1.00 на всех сидах): "
          f"{len(ceiling)}" + (f" -> {', '.join(sorted(ceiling))}" if ceiling else ""))
    print(f"в анализе {len(used)}: {', '.join(used)}\n")

    pooled = {mode: [] for mode, _ in ABLATIONS}
    print("--- по уравнениям: разница доли успеха (полный агент минус режим) ---")
    for pde in used:
        print(f"{pde}")
        for mode, label in ABLATIONS:
            diffs = [full_rate[pde][s] - rate[(pde, mode, s)]
                     for s in sorted(full_rate[pde]) if (pde, mode, s) in rate]
            if not diffs:
                print(f"   {label:16s} нет данных")
                continue
            pooled[mode] += diffs
            mean, lo, hi = ci(diffs)
            mark = "значимо" if (lo == lo and (lo > 0 or hi < 0)) else "не значимо"
            print(f"   {label:16s} сидов {len(diffs)}: среднее {mean:+.3f}, "
                  f"95% [{lo:+.3f}, {hi:+.3f}] -> {mark}")
        print()

    print("--- объединённо: единица наблюдения — пара (уравнение, сид) ---")
    for mode, label in ABLATIONS:
        d = pooled[mode]
        if not d:
            print(f"{label:16s} нет данных")
            continue
        mean, lo, hi = ci(d)
        pos, neg, p = sign_test(d)
        mark = "ЗНАЧИМО" if (lo == lo and (lo > 0 or hi < 0)) else "не значимо"
        print(f"{label:16s} пар {len(d):3d}: среднее {mean:+.3f}, 95% [{lo:+.3f}, {hi:+.3f}] "
              f"-> {mark}; знаковый тест {pos}+/{neg}-, p={p:.3f}")
    print("\nПары взяты без весов: уравнение с 30 траекториями в ячейке весит столько же,\n"
          "сколько с 8. Это осознанно — success rate уже доля, а взвешивание по числу\n"
          "траекторий отдало бы результат двум-трём самым дешёвым уравнениям.")


if __name__ == "__main__":
    main()
