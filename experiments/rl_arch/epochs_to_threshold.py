"""Эпохи до порога ошибки по строкам оценки (без GPU): сколько эпох и стадий цепочки прошло, пока
измеренная ошибка впервые не опустилась до порога, рядом с итоговой ошибкой.

    python experiments/rl_arch/epochs_to_threshold.py <каталог строк> [--tags t3m_ffl_g,t3m_ete_two_g]
        [--rel 0.05,0.02,0.01] [--abs 0.02,0.01] [--final-rel 0.1] [--kmax 10] [--seeds] [--json out.json]

Строки — JSON-файлы online_eval_env.py (rl_arch/online_env/<pde>_<тег>_seed<сид>.json), как у
map_results.py. Строки без stop_reason (промежуточная выгрузка недосчитанного сида, partial=true)
пропускаются и считаются «в работе».

Какая ошибка. Поле hist строки — по записи на каждую ОСТАВЛЕННУЮ стадию цепочки (online_eval_env.py,
цикл run_seed):
    hist[i] = [spent, loss, l2re, t]
      spent — сколько эпох потрачено с начала прогона после стадии i, нарастающим итогом. Сюда входят
              и эпохи действий, откаченных стражем (--guard-rollback; они же в guard_spent строки), и
              эпохи проб разведки (--policy scout): это настоящая цена в эпохах, а не сумма номиналов
              оставленных действий;
      loss  — сумма слагаемых обучающего лосса PINN (train_state.loss_train) после стадии, None если
              не число;
      l2re  — ПОЛНАЯ относительная L2-ошибка относительно эталона, hypot(l2re_op, l2re_bnd) (решение и
              граница) — та же величина, что итоговое поле l2re строки и метрика протокола;
      t     — секунд от начала прогона.
Порог применяется к l2re (hist[i][2]). Это НЕ ошибка правила остановки обучения
(offline_rl.py / online_train_env.py --train-tol): тренеры меряют rmse+brmse (награда, формула (15)
статьи), а в строке оценки по стадиям записана только l2re. Отношение rmse+brmse к l2re у задач
разное (по строкам волн 33-39: pb2d около 0.8, ns2d около 0.18), поэтому пороги здесь и допуск
обучения — разные числа.
Стадии — номер первой стадии с l2re <= порога (с 1); откаченные стражем действия стадиями не
считаются. Исходной точки (до первого действия) в hist нет: порог выше ошибки после первой стадии
даёт эпохи первой стадии. Итоговое поле l2re может отличаться от hist[-1][2] только при --finish-ls.

Пороги. Статья (PINN-PELINE, раздел 2, формула (11), и раздел 3, «Stopping criterion») останавливает
эпизод обучения по e(θ_k) <= ε, где e — «MSE/RMSE относительно эталона», но численного ε не
приводит (K_max = 10 дан, L_max тоже не дан). Поэтому по умолчанию порог относительный: доли
ошибки необученной сети этой задачи init_l2re (медиана по сидам 42-46, track3/pde_meta.json; pb2d
1.10, ns2d 1.91) — 5%, 2% и 1%. Так пороги сравнимы между задачами и измеряются той же l2re.
--abs задаёт абсолютные пороги l2re; --final-rel x добавляет порог «итог сида * (1 + x)» (эпохи, к
которым сид почти вышел на свою итоговую ошибку). --kmax K — сколько сидов дошли до порога не
позже K-й стадии (K_max статьи = 10).
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATHS = (os.path.join(HERE, "track3", "pde_meta.json"),)


def load_meta(path=None):
    for p in ([path] if path else []) + list(META_PATHS):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def tag_of(fname, row):
    """Тег прогона из имени файла <pde>_<тег>_seed<сид>.json (pde берётся из строки)."""
    base = os.path.basename(fname)[:-5]
    pde = row.get("pde") or ""
    if pde and base.startswith(pde + "_"):
        base = base[len(pde) + 1:]
    return base.rsplit("_seed", 1)[0]


def first_below(hist, thr):
    """(эпохи, стадия с 1) первой записи hist с l2re <= thr, либо (None, None)."""
    for i, h in enumerate(hist or []):
        v = h[2]
        if v is not None and math.isfinite(v) and v <= thr:
            return int(h[0]), i + 1
    return None, None


def load_rows(dr, tags=(), keep_smoke=False):
    """{тег: {сид: строка}} законченных сидов и {тег: число строк в работе}."""
    rows, prog = collections.defaultdict(dict), collections.Counter()
    for f in sorted(glob.glob(os.path.join(dr, "*.json"))):
        try:
            with open(f) as fh:
                x = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(x, dict) or "hist" not in x or "seed" not in x:
            continue
        tag = tag_of(f, x)
        if ("smoke" in tag and not keep_smoke) or (tags and tag not in tags):
            continue
        if x.get("stop_reason") is None or x.get("partial"):
            prog[tag] += 1          # промежуточная выгрузка недосчитанного сида: в сводку не идёт
            continue
        rows[tag][int(x["seed"])] = x
    return rows, prog


def thresholds_for(row, meta, rel, absolute, final_rel):
    """Список (имя, порог l2re) для строки."""
    out = [(f"{a:g}", float(a)) for a in absolute]
    e0 = (meta.get(row.get("pde"), {}) or {}).get("init_l2re")
    if e0:
        out += [(f"{r:g}*E0", float(r) * float(e0)) for r in rel]
    if final_rel is not None and row.get("l2re") is not None:
        out.append((f"итог*(1+{final_rel:g})", float(row["l2re"]) * (1.0 + float(final_rel))))
    return out


def summarize(rows, meta, rel=(0.05, 0.02, 0.01), absolute=(), final_rel=0.1, kmax=10):
    """{тег: {n, final_median, budget, pde, init_l2re, thr: {имя: {thr, reached, in_kmax, med_epochs,
    med_epochs_all, med_stages, per_seed: {сид: [эпохи, стадии]}}}}}. Порог «итог*(1+x)» у каждого
    сида свой (thr = None)."""
    out = {}
    for tag, seeds in rows.items():
        any_row = next(iter(seeds.values()))
        budget = int(any_row.get("budget") or max(int(h[0]) for r in seeds.values() for h in (r["hist"] or [[0]])))
        res = dict(n=len(seeds), final_median=st.median(float(r["l2re"]) for r in seeds.values()),
                   budget=budget, pde=any_row.get("pde"),
                   init_l2re=(meta.get(any_row.get("pde"), {}) or {}).get("init_l2re"), thr={})
        for nm, thr0 in thresholds_for(any_row, meta, rel, absolute, final_rel):
            per = {}
            for sd, r in sorted(seeds.items()):
                thr = dict(thresholds_for(r, meta, rel, absolute, final_rel))[nm]
                per[sd] = list(first_below(r["hist"], thr))
            hit = [v for v in per.values() if v[0] is not None]
            # недостигшие — «больше бюджета»: медиана по всем сидам, inf если таких не меньше половины
            all_ep = sorted([v[0] for v in hit] + [math.inf] * (len(per) - len(hit)))
            res["thr"][nm] = dict(
                thr=(None if nm.startswith("итог") else thr0), reached=len(hit),
                in_kmax=sum(1 for v in hit if kmax and v[1] <= kmax),
                med_epochs=(st.median(v[0] for v in hit) if hit else None),
                med_epochs_all=(st.median(all_ep) if all_ep else None),
                med_stages=(st.median(v[1] for v in hit) if hit else None), per_seed=per)
        out[tag] = res
    return out


def fmt_ep(v, budget):
    if v is None:
        return "—"
    if math.isinf(v):
        return f">{budget}"
    return f"{v:.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="каталог JSON-строк оценки")
    ap.add_argument("--tags", default="", help="теги через запятую (по умолчанию все, кроме smoke)")
    ap.add_argument("--rel", default="0.05,0.02,0.01",
                    help="пороги — доли init_l2re задачи (pde_meta.json); пусто — без них")
    ap.add_argument("--abs", default="", help="абсолютные пороги полной l2re через запятую")
    ap.add_argument("--final-rel", type=float, default=0.1,
                    help="порог итог*(1+x) у каждого сида; отрицательное — не считать")
    ap.add_argument("--kmax", type=int, default=10,
                    help="K_max статьи: сколько сидов дошли до порога не позже K-й стадии")
    ap.add_argument("--meta", default="", help="путь к track3/pde_meta.json (по умолчанию из репозитория)")
    ap.add_argument("--seeds", action="store_true", help="печатать эпохи и стадии каждого сида")
    ap.add_argument("--json", default="", help="записать сводку в файл")
    a = ap.parse_args(argv)
    rel = [float(x) for x in a.rel.split(",") if x.strip()]
    absolute = [float(x) for x in a.abs.split(",") if x.strip()]
    meta = load_meta(a.meta or None)
    if rel and not meta:
        ap.error("нет track3/pde_meta.json (init_l2re задач): задайте --meta или пороги --abs с --rel ''")
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    rows, prog = load_rows(a.dir, tags)
    fr = a.final_rel if a.final_rel >= 0 else None
    summ = summarize(rows, meta, rel, absolute, fr, a.kmax)
    print("ошибка: полная l2re = hypot(l2re_op, l2re_bnd) после каждой оставленной стадии (hist[i][2]); "
          "эпохи — нарастающий итог spent (вместе с откаченными стражем)")
    for tag, r in sorted(summ.items(), key=lambda kv: kv[1]["final_median"]):
        e0 = f", E0={r['init_l2re']:.3g}" if r["init_l2re"] else ""
        print(f"{tag} ({r['pde']}{e0}): сидов {r['n']}, итог l2re медиана {r['final_median']:.4f}, "
              f"бюджет {r['budget']}" + (f"  [в работе: {prog[tag]}]" if prog.get(tag) else ""))
        for nm, t in r["thr"].items():
            thr = f"<= {t['thr']:.4g}" if t["thr"] is not None else ""
            stg = "—" if t["med_stages"] is None else f"{t['med_stages']:.0f}"
            print(f"    {nm:>14s} {thr:>10s}: дошли {t['reached']}/{r['n']} (за <= {a.kmax} стадий: "
                  f"{t['in_kmax']}), эпох медиана {fmt_ep(t['med_epochs'], r['budget'])} (по всем сидам "
                  f"{fmt_ep(t['med_epochs_all'], r['budget'])}), стадий медиана {stg}")
            if a.seeds:
                print("        " + " ".join(f"{sd}:{fmt_ep(ep, r['budget'])}/{'—' if sg is None else sg}"
                                            for sd, (ep, sg) in t["per_seed"].items()))
    for tag in sorted(set(prog) - set(summ)):
        print(f"{tag}: законченных сидов нет [в работе: {prog[tag]}]")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summ, f, indent=1, default=str)
    return summ


if __name__ == "__main__":
    main()
