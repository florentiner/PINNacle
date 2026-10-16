#!/usr/bin/env python
"""Отчёт по исследованию данных MAE (mae_data_study.py, волна 56): скачивает из HF rl_arch/mae_study/ отчёт
study (maestudy_<tag>.json, _cov.npz, PNG) и отчёты цены (maecost_<тег>.json) и печатает:

  1. кривую покрытия жадного ядра уравнений (без целей) в каждом пространстве и k95;
  2. избыточные уравнения: пары, покрытые друг другом, уравнения, покрытые ядром k95, пара хаотических;
  3. прокси по армам (восстановление целей и невиденных уравнений, CKA с ENC_MAE и с армом all, R² зонда),
     с оценкой GPU-часов сбора данных арма;
  4. цену перехода с картами по уравнениям (время эпохи по семействам, сборка карт, минут на переход при
     равномерной случайной политике и при частотах действий самих буферов) и цену полных буферов;
  5. кандидатов в минимальный набор: армы, у которых восстановление целей не хуже арма all на --tol (и CKA с
     ENC_MAE не ниже его на --cka-tol), по возрастанию GPU-часов сбора; жадное ядро с учётом цены (по
     индикаторам покрытия из _cov.npz: выигрыш покрытия на GPU-час).

Цены из разных кернелов приводятся к первому по общему уравнению (по умолчанию poisson_boltzmann_2d есть в
обоих). Лёгкий: только json/numpy, без torch.

    python experiments/rl_arch/mae_study_report.py --tag w56 --cost-tags w56a,w56b
    python experiments/rl_arch/mae_study_report.py --tag w56 --cost-tags w56a,w56b --pngs --dir /tmp/mae56
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DIR = "rl_arch/mae_study"
OUT_REPO = "danil-e/pinnacle-optuna-db"
ACTIONS = [(o, lr, e) for o, lrs, eps in (("Adam", (1e-2, 1e-3, 1e-4), (100, 1000, 2500)),
                                          ("LBFGS", (1.0, 0.5, 0.1), (100, 500, 1000)),
                                          ("PSO", (0.0, 1e-3, 1e-4), (100, 200, 300)))
           for lr in lrs for e in eps]
SHORT = {"poisson_boltzmann_2d": "pb2d", "ns2d_liddriven": "ns2d"}


def short(sd):
    return SHORT.get(sd, sd)


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def fetch(name, args):
    """Файл из --dir, иначе из HF (без --offline). None — нет."""
    for p in (os.path.join(args.dir, name), os.path.join(args.dir, HF_DIR, name)):
        if os.path.exists(p):
            return p
    if args.offline:
        return None
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(OUT_REPO, f"{HF_DIR}/{name}", repo_type="dataset", local_dir=args.dir)
    except Exception as e:      # noqa: BLE001
        print(f"  {name}: не скачан ({type(e).__name__})")
        return None


def load(args):
    study, costs = None, []
    p = fetch(f"maestudy_{args.tag}.json", args) if args.tag else None
    if p:
        with open(p) as f:
            study = json.load(f)
    for t in [x for x in args.cost_tags.split(",") if x.strip()]:
        p = fetch(f"maecost_{t.strip()}.json", args)
        if p:
            with open(p) as f:
                costs.append(json.load(f))
    if args.pngs and study:
        for name in study.get("plots", []):
            fetch(name, args)
    return study, costs


def pde_to_subdir():
    with open(os.path.join(HERE, "track3", "pde_meta.json")) as f:
        meta = json.load(f)
    return {k: v.get("subdir") for k, v in meta.items()}


def opt_seconds(spe, mix=None):
    """Среднее время действия: эпохи x время эпохи семейства по частотам 27 действий (None — равномерно)."""
    fill = float(np.mean(list(spe.values())))
    w = np.full(27, 1.0 / 27) if mix is None else np.asarray(mix, float) / max(1e-12, float(np.sum(mix)))
    return float(sum(wi * e * spe.get(o, fill) for wi, (o, _, e) in zip(w, ACTIONS)))


def cost_table(costs, study=None, calib="poissonboltzmann2d"):
    """{папка буфера: цена} по всем отчётам цены; кернелы приводятся к первому по общему уравнению calib."""
    to_sd = pde_to_subdir()
    per = study["data"]["per_pde"] if study else {}
    out, missing, base = {}, {}, None
    for c in costs:
        rows = c.get("rows") or {}
        scale = 1.0
        if calib in rows and (rows[calib].get("derived") or {}).get("per_transition_s"):
            ref = rows[calib]["derived"]["per_transition_s"]
            if base is None:
                base = ref
            scale = base / ref
        for pde, r in rows.items():
            d = r.get("derived")
            sd = to_sd.get(pde) or pde
            if not d:
                missing[sd] = "нет замеров"
                continue
            if sd in out:             # общее уравнение (calib): остаётся замер первого кернела
                continue
            mix = (per.get(sd) or {}).get("actions")
            opt_mix = opt_seconds(d["sec_per_epoch"], mix) if mix else None
            out[sd] = dict(pde=pde, tag=c.get("tag"), device=r.get("device"), scale=scale,
                           spe=d["sec_per_epoch"], map_s=d["map_s"] * scale, opt_s=d["opt_s"] * scale,
                           min_uniform=d["per_transition_min"] * scale,
                           min_mix=((opt_mix + d["map_s"]) / 60.0 * scale) if opt_mix is not None else None,
                           n_full=(per.get(sd) or {}).get("n_rows"))
        for pde, err in (c.get("missing") or {}).items():
            sd = to_sd.get(pde) or pde
            if sd not in out:
                missing[sd] = str(err)[-160:]
    return out, missing


def per_tr_min(cost, sd):
    c = cost.get(sd)
    if not c:
        return None
    return c["min_mix"] if c.get("min_mix") is not None else c["min_uniform"]


def gpu_hours(ntr, cost):
    """GPU-часы сбора переходов ntr ({папка: число}); уравнения без цены — по медиане остальных (флаг)."""
    known = [per_tr_min(cost, sd) for sd in cost]
    known = [v for v in known if v is not None]
    if not known or not ntr:
        return None, False
    med = float(np.median(known))
    tot, approx = 0.0, False
    for sd, n in ntr.items():
        v = per_tr_min(cost, sd)
        if v is None:
            v, approx = med, True
        tot += n * v / 60.0
    return tot, approx


def core_lines(study):
    out = []
    for sp, a in (study.get("analysis") or {}).items():
        c = a["core_free"]
        out.append(f"[{sp}] жадное ядро без целей (k95 = {c['k95']}); PCA: первые 3 компоненты "
                   f"{[round(v, 3) for v in a['pca_var'][:3]]}")
        out.append(f"  {'k':>3} {'добавлено':<28} {'покр. всех':>10} {'покр. целей':>11} {'нецелевых':>10} {'выигрыш':>8}")
        for r in c["curve"]:
            out.append(f"  {r['k']:>3} {r['added']:<28} {fmt(r['cov_all'], 3):>10} {fmt(r['cov_targets'], 3):>11} "
                       f"{fmt(r['cov_nontargets'], 3):>10} {fmt(r['gain'], 3):>8}")
        pm = a.get("per_maps") or {}
        if pm:
            out.append("  покрытие целей от числа карт на уравнение (0 — вся выборка анализа):")
            for key, rows in pm.items():
                out.append(f"    {key:<10} " + "  ".join(f"{r['n']}: {fmt(r['cov_targets'], 3)}" for r in rows))
        kc = a.get("kcenter")
        if kc:
            out.append("  отбор карт (кандидаты без целей), покрытие всех: m / k-center / случайно: " + "; ".join(
                f"{m} / {fmt(x, 3)} / {fmt(y, 3)}" for m, x, y in zip(kc["m"], kc["kcenter"]["cov_all"],
                                                                      kc["random"]["cov_all"])))
    return out


def redundancy_lines(study):
    out = []
    for sp, a in (study.get("analysis") or {}).items():
        pairs = a.get("redundant_pairs") or []
        out.append(f"[{sp}] пары, покрывающие друг друга (>= tau самопокрытия): "
                   + (", ".join(f"{p['a']}~{p['b']} ({p['a_by_b']}/{p['b_by_a']})" for p in pairs) or "нет"))
        out.append(f"[{sp}] покрыты ядром k95 без себя (лишние для сбора): "
                   + (", ".join(a.get("redundant_given_core95") or []) or "нет"))
        ch = a.get("chaotic")
        if ch:
            x, y = ch["pair"]
            out.append(f"[{sp}] хаотические {x} / {y}: покрытие друг другом {fmt(ch['a_by_b'], 3)} / "
                       f"{fmt(ch['b_by_a'], 3)} (сами собой {fmt(ch['self_a'], 3)} / {fmt(ch['self_b'], 3)}), MMD² "
                       f"{fmt(ch['mmd'], 5)} (шум {fmt(ch['mmd_self_a'], 5)} / {fmt(ch['mmd_self_b'], 5)}), выигрыш "
                       f"покрытия всех от второго: {fmt(ch['gain_all_add_b_to_a'], 3)} / {fmt(ch['gain_all_add_a_to_b'], 3)}, "
                       f"место в ядре {ch['rank_in_core']}")
        cl = a.get("clusters")
        if isinstance(cl, dict) and cl:
            k = sorted(cl, key=int)[len(cl) // 2]
            groups = {}
            for sd, c in cl[k].items():
                groups.setdefault(c, []).append(short(sd))
            out.append(f"[{sp}] кластеры по MMD (k={k}): " + " | ".join(", ".join(v) for v in groups.values()))
    return out


def proxy_rows(study, cost):
    rows = []
    for name, e in (study.get("arms") or {}).items():
        if "same_as" in e:
            rows.append(dict(name=name, group=e.get("group"), same_as=e["same_as"]))
            continue
        gh, approx = gpu_hours(e.get("n_tr_per_pde") or {}, cost) if cost else (None, False)
        rt = e.get("recon_targets") or {}
        rows.append(dict(name=name, group=e.get("group"), n_pde=len(e.get("subdirs") or []), n_per=e.get("n_per"),
                         n_maps=e.get("n_maps"), recon_t=e.get("recon_targets_mean"), recon_u=e.get("recon_unseen_mean"),
                         recon_by_t=rt, cka_ref=e.get("cka_ref_mean"), cka_all=e.get("cka_all_mean"),
                         probe=e.get("probe_mean"), gpu_h=gh, approx=approx))
    return rows


def proxy_lines(study, cost):
    tg = study.get("targets") or []
    out = [f"{'арм':<24} {'группа':<10} {'ур.':>3} {'n/ур.':>6} {'карт':>7} "
           + " ".join(f"{'восст.' + short(t):>11}" for t in tg)
           + f" {'невид.':>7} {'CKA ENC':>8} {'CKA all':>8} {'зонд R²':>8} {'GPU-ч':>8}"]
    for r in proxy_rows(study, cost):
        if "same_as" in r:
            out.append(f"{r['name']:<24} {r['group'] or '':<10} = {r['same_as']}")
            continue
        out.append(f"{r['name']:<24} {r['group'] or '':<10} {r['n_pde']:>3} {r['n_per'] or 'все':>6} "
                   f"{r['n_maps'] if r['n_maps'] is not None else '—':>7} "
                   + " ".join(f"{fmt(r['recon_by_t'].get(t)):>11}" for t in tg)
                   + f" {fmt(r['recon_u']):>7} {fmt(r['cka_ref'], 3):>8} {fmt(r['cka_all'], 3):>8} "
                     f"{fmt(r['probe'], 3):>8} {fmt(r['gpu_h'], 1) + ('≈' if r['approx'] else ''):>8}")
    seeds = [(n, n[:-3]) for n in (study.get("arms") or {}) if n.endswith("_s1")]
    for a, b in seeds:
        ea, eb = study["arms"].get(a, {}), study["arms"].get(b, {})
        if "same_as" in ea or "same_as" in eb or not eb:
            continue
        d = [abs((ea.get(k) or 0) - (eb.get(k) or 0)) if ea.get(k) is not None and eb.get(k) is not None else None
             for k in ("recon_targets_mean", "cka_ref_mean", "probe_mean")]
        out.append(f"шум сида {b}: |Δ| восст. целей {fmt(d[0])}, CKA {fmt(d[1], 3)}, зонд {fmt(d[2], 3)}")
    return out


def cost_lines(cost, missing):
    out = [f"{'уравнение':<28} {'GPU':<16} {'Adam с/эп':>9} {'LBFGS':>7} {'PSO':>7} {'карты с':>8} "
           f"{'мин/пер. (равн.)':>16} {'(буфер)':>8} {'переходов':>9} {'GPU-ч буфера':>12}"]
    tot = 0.0
    for sd in sorted(cost, key=lambda s: -(per_tr_min(cost, s) or 0)):
        c = cost[sd]
        full = c["n_full"] * per_tr_min(cost, sd) / 60.0 if c.get("n_full") else None
        tot += full or 0.0
        out.append(f"{sd:<28} {str(c.get('device') or '')[:16]:<16} {fmt(c['spe'].get('Adam'), 4):>9} "
                   f"{fmt(c['spe'].get('LBFGS'), 4):>7} {fmt(c['spe'].get('PSO'), 4):>7} {fmt(c['map_s'], 1):>8} "
                   f"{fmt(c['min_uniform'], 2):>16} {fmt(c.get('min_mix'), 2):>8} {c.get('n_full') or '—':>9} "
                   f"{fmt(full, 0):>12}" + (f"  (x{c['scale']:.2f} к первому кернелу)" if abs(c["scale"] - 1) > 1e-9 else ""))
    out.append(f"итого полных буферов с ценой: {tot:.0f} GPU-ч")
    for sd, err in missing.items():
        out.append(f"НЕТ ЦЕНЫ {sd}: {err}")
    return out


def candidate_lines(study, cost, tol=0.05, cka_tol=0.05):
    arms = study.get("arms") or {}
    ref = arms.get("all") or {}
    r0, c0 = ref.get("recon_targets_mean"), ref.get("cka_ref_mean")
    out = [f"опора — арм all: восстановление целей {fmt(r0)}, CKA с ENC_MAE {fmt(c0, 3)}; допуск {tol:.0%} и "
           f"{cka_tol} CKA"]
    if r0 is None:
        return out + ["арма all нет — кандидатов не выбрать"]
    ok = []
    for r in proxy_rows(study, cost):
        if "same_as" in r or r["recon_t"] is None or r["name"] in ("all", "rand", "enc_mae"):
            continue
        good = r["recon_t"] <= r0 * (1.0 + tol) and (c0 is None or r["cka_ref"] is None or r["cka_ref"] >= c0 - cka_tol)
        if good:
            ok.append(r)
    ok.sort(key=lambda r: (r["gpu_h"] if r["gpu_h"] is not None else float("inf"), r["n_maps"] or 0))
    for r in ok[:20]:
        out.append(f"  {r['name']:<24} {r['n_pde']} ур. x {r['n_per'] or 'все'}: карт {r['n_maps']}, восст. целей "
                   f"{fmt(r['recon_t'])}, CKA {fmt(r['cka_ref'], 3)}, GPU-ч {fmt(r['gpu_h'], 1)}"
                   + ("≈" if r["approx"] else ""))
    if not ok:
        out.append("  ни один арм не уложился в допуск")
    bc = study.get("best_core") or {}
    if bc.get("k"):
        out.append(f"лучшее ядро ({bc['space']}): k={bc['k']} — {', '.join(bc['subdirs'])}")
    return out


def load_cov(args, study, space):
    p = fetch(f"maestudy_{args.tag}_cov.npz", args)
    if not p:
        return None
    with np.load(p) as z:
        if f"{space}_B" not in z.files:
            return None
        shape = tuple(int(x) for x in z[f"{space}_Bshape"])
        B = np.unpackbits(z[f"{space}_B"])[:int(np.prod(shape))].reshape(shape).astype(bool)
        return B, z[f"{space}_valid"].astype(bool), [str(x) for x in z["names"]]


def cost_greedy_lines(B, valid, names, targets, cost, n_per):
    """Жадный выбор уравнений без целей по выигрышу покрытия всех уравнений на GPU-час сбора n_per переходов."""
    def union(S):
        if not S:
            return np.zeros(len(names))
        u = B[..., -1][:, S].any(1) & valid          # (T, |S|, n) — без смешанной индексации numpy
        return u.sum(1) / np.maximum(valid.sum(1), 1)
    tidx = [names.index(t) for t in targets if t in names]
    cand = [i for i in range(len(names)) if i not in tidx]
    chosen, out, h, prev = [], [f"жадно по выигрышу покрытия на GPU-час ({n_per} переходов на уравнение):"], 0.0, 0.0
    while len(chosen) < len(cand):
        best = None
        for c in cand:
            if c in chosen:
                continue
            g = union(chosen + [c]).mean() - prev
            ch = (per_tr_min(cost, names[c]) or np.nan) * n_per / 60.0
            score = g / ch if np.isfinite(ch) and ch > 0 else -1.0
            if best is None or score > best[0]:
                best = (score, c, ch)
        _, c, ch = best
        chosen.append(c)
        u = union(chosen)
        h += ch if np.isfinite(ch) else 0.0
        prev = u.mean()
        out.append(f"  +{names[c]:<28} GPU-ч {h:7.1f}  покрытие всех {u.mean():.3f}, целей "
                   f"{u[tidx].mean() if tidx else float('nan'):.3f}")
    return out


def report(args):
    study, costs = load(args)
    cost, missing = cost_table(costs, study, args.calib) if costs else ({}, {})
    sec = {}
    if study:
        sec["core"] = core_lines(study)
        sec["redundancy"] = redundancy_lines(study)
        sec["proxies"] = proxy_lines(study, cost)
    if costs:
        sec["cost"] = cost_lines(cost, missing)
    if study:
        sec["candidates"] = candidate_lines(study, cost, args.tol, args.cka_tol)
        space = args.space or (study.get("best_core") or {}).get("space") or "raw"
        cov = load_cov(args, study, space) if cost else None
        if cov:
            sec["cost_greedy"] = cost_greedy_lines(*cov, study.get("targets") or [], cost,
                                                   int(study["settings"].get("core_n", 2000)))
    titles = dict(core="1. Покрытие: жадное ядро уравнений", redundancy="2. Избыточные уравнения",
                  proxies="3. Прокси по армам", cost="4. Цена перехода с картами",
                  candidates="5. Кандидаты в минимальный набор", cost_greedy="5б. Ядро с учётом цены")
    for k, lines in sec.items():
        print(f"\n=== {titles[k]} ===")
        print("\n".join(lines))
    if study:
        print(f"\nисследование: {study.get('tag')}, закончено {study.get('finished')}, {study.get('elapsed_s')} с, "
              f"{study.get('device')}; рисунки: {', '.join(study.get('plots', [])[:6])}...")
    return dict(study=study, cost=cost, missing=missing, sections=sec)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="w56")
    ap.add_argument("--cost-tags", default="w56a,w56b")
    ap.add_argument("--dir", default=os.path.join(os.path.expanduser("~"), ".cache", "mae_study"))
    ap.add_argument("--offline", action="store_true", help="только файлы из --dir")
    ap.add_argument("--pngs", action="store_true", help="скачать и рисунки")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--cka-tol", type=float, default=0.05)
    ap.add_argument("--space", default="", help="пространство для жадного выбора с учётом цены")
    ap.add_argument("--calib", default="poissonboltzmann2d", help="общее уравнение кернелов цены")
    return ap


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.dir, exist_ok=True)
    return report(args)


if __name__ == "__main__":
    main()
