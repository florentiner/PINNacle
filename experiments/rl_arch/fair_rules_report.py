#!/usr/bin/env python
"""Честное сравнение правил развёртывания (волна 70): агент статьи и итоговый метод на одних сидах и правилах.

Строки оценок (online_eval_env.py) берутся из HF-датасета результатов (rl_arch/online_env/); незаконченные
сиды (partial или без stop_reason) не считаются. Для каждого уравнения и строки таблицы — медиана, среднее и
худший сид полной l2re по общим сидам, число сидов, доля бюджета у обрезки и у отката. Строки:
  агент статьи: без правил (t3m_orig_free), только откат (t3m_orig_g), откат + мягкая обрезка (t3m_orig_ss);
  итоговый метод, обучения 1 и 2 (pb2d t3m_w67proc3n20two_m*_ss, ns2d t3m_w65src3n20two_m*_ss) и без
  обрезки там, где это есть (ns2d t3m_w65src3n20two_m*_g).
Выигрыш считается к агенту статьи с теми же правилами (_ss): это и есть ответ «сколько даёт агент, а не правила».

    python experiments/rl_arch/fair_rules_report.py                  # сиды 42-51
    python experiments/rl_arch/fair_rules_report.py --seeds 42-46
"""
import argparse
import fnmatch
import json
import os
import statistics as st

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
REPO = "danil-e/pinnacle-optuna-db"
ROWS = {
    "poissonboltzmann2d": [("агент статьи, без правил", "t3m_orig_free"), ("агент статьи, откат", "t3m_orig_g"),
                           ("агент статьи, откат + обрезка", "t3m_orig_ss"),
                           ("итоговый метод, обучение 1", "t3m_w67proc3n20two_m1_ss"),
                           ("итоговый метод, обучение 2", "t3m_w67proc3n20two_m2_ss")],
    "ns2d_liddriven": [("агент статьи, без правил", "t3m_orig_free"), ("агент статьи, откат", "t3m_orig_g"),
                       ("агент статьи, откат + обрезка", "t3m_orig_ss"),
                       ("итоговый метод, обучение 1, без обрезки", "t3m_w65src3n20two_m1_g"),
                       ("итоговый метод, обучение 2, без обрезки", "t3m_w65src3n20two_m2_g"),
                       ("итоговый метод, обучение 1", "t3m_w65src3n20two_m1_ss"),
                       ("итоговый метод, обучение 2", "t3m_w65src3n20two_m2_ss")],
}


def seeds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def load(pde, tags, token):
    from huggingface_hub import snapshot_download
    pats = [f"rl_arch/online_env/{pde}_{t}_seed*.json" for t in tags]
    root = snapshot_download(REPO, repo_type="dataset", token=token, allow_patterns=pats)
    src = os.path.join(root, "rl_arch", "online_env")
    out = {t: {} for t in tags}
    for f in (os.listdir(src) if os.path.isdir(src) else []):
        for t in tags:
            if fnmatch.fnmatch(f, f"{pde}_{t}_seed*.json") and f[len(f"{pde}_{t}_seed"):-5].isdigit():
                d = json.load(open(os.path.join(src, f)))
                if d.get("partial") or d.get("stop_reason") is None:
                    continue
                out[t][int(f[len(f"{pde}_{t}_seed"):-5])] = d
    return out


def summary(rows, seeds, budget=7000):
    got = [rows[s] for s in seeds if s in rows]
    if not got:
        return None
    v = [d["l2re"] for d in got]
    return dict(n=len(v), med=st.median(v), mean=st.mean(v), worst=max(v),
                cut=100 * st.mean((d.get("stall_spent") or 0) for d in got) / budget,
                roll=100 * st.mean((d.get("guard_spent") or 0) for d in got) / budget)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42-51")
    a = ap.parse_args()
    want = seeds_of(a.seeds)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_READ") or False
    for pde, rows in ROWS.items():
        data = load(pde, [t for _, t in rows], token)
        # общие сиды: те, что есть у всех строк с данными (иначе медианы несопоставимы)
        have = [set(data[t]) & set(want) for _, t in rows if data[t]]
        common = sorted(set.intersection(*have)) if have else []
        print(f"\n{pde}: общих сидов {len(common)} {common}")
        print(f"  {'вариант':42s} {'сидов':>5s} {'медиана':>8s} {'среднее':>8s} {'худший':>8s} {'обрезка':>8s} {'откат':>6s}")
        ref = summary(data["t3m_orig_ss"], common)
        for name, t in rows:
            s = summary(data[t], common)
            if s is None:
                print(f"  {name:42s}     —   (нет досчитанных сидов {t})")
                continue
            gain = f"  x{ref['med'] / s['med']:.2f} к статье с теми же правилами" if ref and "итоговый" in name else ""
            print(f"  {name:42s} {s['n']:5d} {s['med']:8.4f} {s['mean']:8.4f} {s['worst']:8.3f} {s['cut']:7.0f}% "
                  f"{s['roll']:5.0f}%{gain}")


if __name__ == "__main__":
    main()
