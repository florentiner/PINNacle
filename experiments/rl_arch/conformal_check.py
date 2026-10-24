#!/usr/bin/env python
"""Локальная проверка: даёт ли CoCoNet (Moradi et al., arXiv 2610.24246; код github.com/AI4OPT/CoCoNet) модели среды
итогового метода области неопределённости лучше, чем нынешний разброс ансамбля.

Модель среды — как в итоговом методе (волна 65): ансамбль из 5 HGB одного шага (track3/surrogate.py) на журналах
3 уравнений x 20 цепочек (burgers1d, poisson2d_classic, ns2d_backstep; agent_combo.cap_source_chains, сид 0) и
собранном буфере цели. Для каждого состояния собранной цепочки цели с H записанными действиями впереди:
прогон этих действий в ансамбле вне фолда (фолды по группам общего префикса, модель учится на источниках и
цепочках цели вне фолда) -> F (среднее по членам log10 ошибки после 1..H действий), sigma (разброс по членам),
y (настоящая log10 ошибка). Остаток r = y - F, размерность H.

Области при покрытии 1 - alpha = 0.9 (повторы случайного деления цепочек цели: score / cal / test):
  ens_raw   — F +- 1.645 sigma по каждому горизонту (как штраф разброса сейчас: без калибровки);
  ens_conf  — та же форма, масштаб по калибровке: max_h |r_h| / sigma_h;
  absolute_residual, global_ellipsoid — базовые конформные области из кода CoCoNet;
  spline, icnn — CoCoNet (код авторов, их настройки; гиперпараметры без перебора: K = 4, кодировщик [64, 32]).
Признаки x для CoCoNet: признаки состояния и первого действия (surrogate.featurize), F и sigma (стандартизованы).
Метрики на test: покрытие, размер (mean V)^(1/H), адаптивность — Spearman объёма области и max_h |r_h| по
состояниям (чем выше, тем лучше область отличает надёжные прогнозы от ненадёжных: это то, что нужно штрафу
модельных целей и выбору цепочек по LCB).

    python -I experiments/rl_arch/conformal_check.py --coconet-dir <клон CoCoNet> --out conformal_check.json
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(HERE, "track3")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DDEBACKEND", "pytorch")

import numpy as np  # noqa: E402

SRC = ("burgers1d", "poisson2d_classic", "ns2d_backstep")
TARGETS = {"poissonboltzmann2d": "rl_arch/buffers_new/tbuf_pb2d_w0",
           "ns2d_liddriven": "rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1"}


def source_table(work, n_chains, seed):
    import agent_combo as AC
    import offline_rl as O
    full = AC.get_table(argparse.Namespace(table="", work=work), list(O.ALL_SUBDIRS), print)
    idx = [i for i, s in enumerate(full["SUBDIRS"]) if s in SRC]
    tab = AC.subset(full, np.isin(full["PDE"], idx))
    tab, info = AC.cap_source_chains(tab, n_chains, seed)
    return tab, info


def oof_dataset(tab, pi, H, members, folds, seed=0):
    """Вне фолда: признаки, F, sigma, y по состояниям цели с H действиями впереди, группа цепочки."""
    import surrogate as S
    m = tab["PDE"] == pi
    rows = np.where(m)[0]
    grp = tab["GRP"][rows]
    ug = np.unique(grp)
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = {int(g): int(perm[i] % folds) for i, g in enumerate(ug)}
    # строки цели по цепочкам, по шагам
    by_ch = {}
    for r in rows:
        by_ch.setdefault(int(tab["CH"][r]), []).append(r)
    starts = []
    for ch, rs in by_ch.items():
        rs = sorted(rs, key=lambda r: tab["STEP"][r])
        for i in range(len(rs) - H + 1):
            seg = rs[i:i + H]
            if all(tab["STEP"][seg[k + 1]] == tab["STEP"][seg[k]] + 1 for k in range(H - 1)):
                starts.append(seg)
    X, F, SD, Y, G = [], [], [], [], []
    src_rows = np.where(~m)[0]
    for f in range(folds):
        tr = np.concatenate([src_rows, [r for r in rows if fold_of[int(tab["GRP"][r])] != f]]).astype(np.int64)
        te = [seg for seg in starts if fold_of[int(tab["GRP"][seg[0]])] == f]
        if not te:
            continue
        ens = S.fit_ensemble(tab, tr, members=members, seed=seed)
        r0 = np.array([seg[0] for seg in te])
        st = dict(ctx=tab["CTX"][tab["PDE"][r0]].astype(np.float64), step=tab["STEP"][r0].copy(),
                  spent=tab["SPENT"][r0].astype(np.int64), spf=tab["SPF"][r0].astype(np.float64),
                  pa=tab["PA"][r0].copy(), l=np.tile(tab["L"][r0], (members, 1)).astype(np.float64),
                  lv=np.zeros((members, len(r0), 6)))
        preds = []
        for h in range(H):
            rh = np.array([seg[h] for seg in te])
            st = S.advance(ens, st, tab["A"][rh], ep=tab["EPA"][rh])
            preds.append(st["l"].copy())                        # (members, n)
        P = np.stack(preds, -1)                                 # (members, n, H)
        X.append(S.table_features(tab, r0))
        F.append(P.mean(0))
        SD.append(P.std(0))
        Y.append(np.stack([tab["LN"][np.array([seg[h] for seg in te])] for h in range(H)], -1))
        G.append(tab["GRP"][r0])
    return (np.concatenate(X), np.concatenate(F), np.concatenate(SD), np.concatenate(Y), np.concatenate(G))


def segments(tab, rows, H):
    """Сегменты из H подряд идущих шагов одной цепочки среди строк rows."""
    by_ch = {}
    for r in rows:
        by_ch.setdefault(int(tab["CH"][r]), []).append(int(r))
    out = []
    for rs in by_ch.values():
        rs = sorted(rs, key=lambda r: tab["STEP"][r])
        for i in range(len(rs) - H + 1):
            seg = rs[i:i + H]
            if all(tab["STEP"][seg[k + 1]] == tab["STEP"][seg[k]] + 1 for k in range(H - 1)):
                out.append(seg)
    return out


def predict(tab, train_rows, segs, members, seed=0):
    """Ансамбль на train_rows; прогон записанных действий сегментов: X, F, SD, Y, группа (цепочка)."""
    import surrogate as S
    ens = S.fit_ensemble(tab, np.asarray(train_rows, dtype=np.int64), members=members, seed=seed)
    r0 = np.array([seg[0] for seg in segs])
    H = len(segs[0])
    st = dict(ctx=tab["CTX"][tab["PDE"][r0]].astype(np.float64), step=tab["STEP"][r0].copy(),
              spent=tab["SPENT"][r0].astype(np.int64), spf=tab["SPF"][r0].astype(np.float64),
              pa=tab["PA"][r0].copy(), l=np.tile(tab["L"][r0], (members, 1)).astype(np.float64),
              lv=np.zeros((members, len(r0), 6)))
    preds = []
    for h in range(H):
        rh = np.array([seg[h] for seg in segs])
        st = S.advance(ens, st, tab["A"][rh], ep=tab["EPA"][rh])
        preds.append(st["l"].copy())
    P = np.stack(preds, -1)
    Y = np.stack([tab["LN"][np.array([seg[h] for seg in segs])] for h in range(H)], -1)
    G = tab["GRP"][r0] if "GRP" in tab else tab["CH"][r0]
    return S.table_features(tab, r0), P.mean(0), P.std(0), Y, G


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


DEVICE = "cpu"


def run_split(P, cfg, X, F, SD, Y, G, rep, alpha, methods, tuned, parts=None):
    """Одно деление цепочек цели: score (с валидацией) / cal / test; все методы на одних данных. parts — готовые
    индексы (score, cal, test) вместо случайного деления по группам."""
    import torch
    rng = np.random.default_rng(1000 + rep)
    if parts is None:
        ug = rng.permutation(np.unique(G))
        n = len(ug)
        cut = [int(round(n * 0.45)), int(round(n * 0.75))]
        part = {g: (0 if i < cut[0] else 1 if i < cut[1] else 2) for i, g in enumerate(ug)}
        which = np.array([part[g] for g in G])
        sc, cal, te = (np.where(which == k)[0] for k in range(3))
    else:
        sc, cal, te = (np.asarray(p_, dtype=np.int64) for p_ in parts)
    r = Y - F
    Z = np.concatenate([X, F, np.log(SD + 1e-6)], 1)
    mu, sd = Z[sc].mean(0), Z[sc].std(0)
    Z = (Z - mu) / np.where(sd > 0, sd, 1.0)
    H = Y.shape[1]
    fcols = list(range(X.shape[1], X.shape[1] + H))

    class Select(torch.nn.Module):
        def forward(self, x):
            return x[:, fcols]
    va_n = max(2, int(round(len(sc) * 0.2)))
    perm = rng.permutation(len(sc))
    s_tr, s_va = sc[perm[va_n:]], sc[perm[:va_n]]
    f32 = lambda a: a.astype(np.float32)   # noqa: E731
    data = P.Data("pinn_env", rep, H, Z.shape[1], f32(Z[s_tr]), f32(r[s_tr]), f32(Z[s_va]), f32(r[s_va]),
                  f32(Z[cal]), f32(r[cal]), f32(Z[te]), f32(r[te]), np.ones(H), Select(),
                  0, len(sc), len(cal), len(te), "", "", 0)
    out = {}
    z = 1.6448536269514722
    rmax = np.abs(r[te]).max(1)
    # нынешний разброс ансамбля: сырой и откалиброванный
    cov_raw = np.all(np.abs(r[te]) <= z * SD[te], 1)
    vol_raw = np.prod(2 * z * SD[te], 1)
    q = float(np.quantile(np.abs(r[cal] / SD[cal]).max(1), min(1.0, math.ceil((len(cal) + 1) * (1 - alpha)) / len(cal))))
    cov_c = np.all(np.abs(r[te]) <= q * SD[te], 1)
    vol_c = np.prod(2 * q * SD[te], 1)
    for nm, cov, vol in (("ens_raw", cov_raw, vol_raw), ("ens_conf", cov_c, vol_c)):
        out[nm] = dict(coverage=float(cov.mean()), size=float(np.mean(vol) ** (1 / H)), adapt=spearman(vol, rmax))
    for m in methods:
        t0 = time.time()
        try:
            fit = P.run_method(m, data, cfg, tuned=tuned.get(m), device=DEVICE if m == "icnn" else "cpu")
            cov = fit.contains(data.x_test, data.r_test)
            vol = np.asarray(fit.volume(data.x_test), float)
            out[m] = dict(coverage=float(np.mean(cov)), size=float(np.mean(vol) ** (1 / H)), adapt=spearman(vol, rmax),
                          seconds=round(time.time() - t0, 1))
        except Exception as e:                               # noqa: BLE001
            out[m] = dict(error=f"{type(e).__name__}: {e}"[:200])
    out["_n"] = dict(score=int(len(sc)), cal=int(len(cal)), test=int(len(te)))
    return out


def pseudo(a, P, cfg, alpha, tuned, methods, log=print):
    """Каждое решаемое уравнение (кроме источников) — цель: свои = --own-chains его цепочек, тест — остальные.
    realistic: функция несоответствия учится и калибруется на своих (вне фолда, как у метода на новом уравнении);
    generous: на части остальных цепочек цели (много данных — верхняя граница пользы CoCoNet)."""
    import agent_combo as AC
    import offline_rl as O
    full = AC.get_table(argparse.Namespace(table="", work=a.work), list(O.ALL_SUBDIRS), log)
    names = list(full["SUBDIRS"])
    src_idx = [i for i, s_ in enumerate(names) if s_ in SRC]
    src_tab, info = AC.cap_source_chains(AC.subset(full, np.isin(full["PDE"], src_idx)), 20, 0)
    log(f"источники: {info}")
    targets = [s_ for s_ in names if s_ not in SRC and s_ not in O.UNSOLVED_SUBDIRS]
    if a.pseudo_targets:
        targets = [t for t in a.pseudo_targets.split(",") if t]
    res = {}
    for ti, tsub in enumerate(targets):
        pi_full = names.index(tsub)
        trow = np.where(full["PDE"] == pi_full)[0]
        chs = np.unique(full["CH"][trow])
        if len(chs) < a.own_chains + 10:
            log(f"{tsub}: цепочек {len(chs)} — мало, пропуск")
            continue
        rng = np.random.default_rng(ti)
        own_ch = rng.choice(chs, a.own_chains, replace=False)
        # таблица: источники + все строки цели (номера цепочек цели не пересекаются с источниками)
        tab = AC.subset(full, np.isin(full["PDE"], src_idx + [pi_full]) & (
            (full["PDE"] == pi_full) | np.isin(np.arange(len(full["A"])), np.where(np.isin(full["PDE"], src_idx))[0])))
        keep_src = np.isin(tab["PDE"], src_idx)
        # те же 20 цепочек у источников, что у учителя
        src_keep_ch = set(src_tab["CH"].tolist())
        tab = AC.subset(tab, ~keep_src | np.isin(tab["CH"], list(src_keep_ch)))
        tr = np.where(tab["PDE"] == pi_full)[0]
        own = tr[np.isin(tab["CH"][tr], own_ch)]
        rest = tr[~np.isin(tab["CH"][tr], own_ch)]
        srcr = np.where(tab["PDE"] != pi_full)[0]
        H = a.horizon
        seg_rest = segments(tab, rest, H)
        if len(seg_rest) > a.max_test:
            seg_rest = [seg_rest[i] for i in sorted(rng.choice(len(seg_rest), a.max_test, replace=False))]
        seg_own = segments(tab, own, H)
        if len(seg_own) < 30 or len(seg_rest) < 100:
            log(f"{tsub}: сегментов своих {len(seg_own)}, тест {len(seg_rest)} — мало, пропуск")
            continue
        # тест: модель на источниках + всех своих
        Xt, Ft, SDt, Yt, Gt = predict(tab, np.concatenate([srcr, own]), seg_rest, a.members)
        # свои вне фолда (5 фолдов по цепочкам)
        fold = {int(c): i % a.folds for i, c in enumerate(rng.permutation(own_ch))}
        parts = []
        for f in range(a.folds):
            te_s = [sg for sg in seg_own if fold[int(tab["CH"][sg[0]])] == f]
            if not te_s:
                continue
            trn = np.concatenate([srcr, [r for r in own if fold[int(tab["CH"][r])] != f]])
            parts.append(predict(tab, trn, te_s, a.members))
        Xo, Fo, SDo, Yo, Go = (np.concatenate([p_[k] for p_ in parts]) for k in range(5))
        X = np.concatenate([Xo, Xt]); F = np.concatenate([Fo, Ft]); SD = np.concatenate([SDo, SDt])
        Y = np.concatenate([Yo, Yt]); G = np.concatenate([Go, Gt])
        no = len(Yo)
        # realistic: score/cal — свои (по цепочкам 60/40), test — остальные цепочки цели
        ug = rng.permutation(np.unique(Go))
        cut = int(round(len(ug) * 0.6))
        sc = np.where(np.isin(Go, ug[:cut]))[0]
        cal = np.where(np.isin(Go, ug[cut:]))[0]
        te = np.arange(no, len(Y))
        ms_r = [m for m in methods if m != "icnn" or ti < a.icnn_targets]
        out_r = run_split(P, cfg, X, F, SD, Y, G, ti, alpha, ms_r, tuned, parts=(sc, cal, te))
        # generous: score/cal/test — части остальных цепочек цели (45/30/25 по цепочкам)
        ugt = rng.permutation(np.unique(Gt))
        c1, c2 = int(round(len(ugt) * 0.45)), int(round(len(ugt) * 0.75))
        idx = lambda gs: no + np.where(np.isin(Gt, gs))[0]   # noqa: E731
        ms_g = [m for m in methods if m != "icnn" or ti < a.icnn_targets]
        out_g = run_split(P, cfg, X, F, SD, Y, G, ti, alpha, ms_g, tuned,
                          parts=(idx(ugt[:c1]), idx(ugt[c1:c2]), idx(ugt[c2:])))
        mae = np.abs(Yt - Ft).mean(0).round(3).tolist()
        res[tsub] = dict(n_own=int(no), n_test=int(len(Yt)), mae=mae, realistic=out_r, generous=out_g)
        for nm, o in (("realistic", out_r), ("generous", out_g)):
            log(f"{tsub} [{nm}] свои {no}, тест {len(Yt)}, MAE {mae}: " + "; ".join(
                f"{k} cov {v['coverage']:.2f} size {v['size']:.3f} adapt {v['adapt']:+.2f}" if "coverage" in v
                else f"{k} {v.get('error', '')}" for k, v in o.items() if not k.startswith("_")))
        json.dump(res, open(a.out, "w"), indent=1)
    log("\nитог по псевдоцелям (среднее; адаптивность — Spearman размера области и |ошибки|):")
    for regime in ("realistic", "generous"):
        log(regime)
        keys = list(dict.fromkeys(k for r in res.values() for k in r[regime] if not k.startswith("_")))
        for k in keys:
            vals = [r[regime][k] for r in res.values() if "coverage" in r[regime].get(k, {})]
            if vals:
                log(f"  {k:18s} целей {len(vals):2d}  покрытие {np.mean([v['coverage'] for v in vals]):.3f}  "
                    f"размер {np.mean([v['size'] for v in vals]):.4f}  адаптивность "
                    f"{np.nanmean([v['adapt'] for v in vals]):+.2f}")
    return res


def upload(a):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not a.upload or not tok or not os.path.exists(a.out):
        return
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=a.out, path_in_repo=f"rl_arch/conformal/{os.path.basename(a.out)}",
                        repo_id="danil-e/pinnacle-optuna-db", repo_type="dataset", token=tok,
                        commit_message=f"conformal check {os.path.basename(a.out)}")
    print(f"залито в HF: rl_arch/conformal/{os.path.basename(a.out)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coconet-dir", required=True)
    ap.add_argument("--work", default=os.path.join(REPO, "agent_combo_work"))
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", default="absolute_residual,global_ellipsoid,spline,icnn")
    ap.add_argument("--icnn-repeats", type=int, default=2,
                    help="ICNN только в первых N повторах (на CPU ~0.5-1 ч на повтор: перебор 11 весов по 400 эпох)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--pseudo", action="store_true",
                    help="псевдоцели: решаемые уравнения полной таблицы (кроме источников) по очереди как цель")
    ap.add_argument("--pseudo-targets", default="", help="--pseudo: только эти папки через запятую")
    ap.add_argument("--own-chains", type=int, default=20, help="--pseudo: своих цепочек цели")
    ap.add_argument("--max-test", type=int, default=1500, help="--pseudo: не больше стольких тестовых состояний")
    ap.add_argument("--icnn-targets", type=int, default=4, help="--pseudo: ICNN только у первых N целей")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="устройство ICNN (cuda на Kaggle)")
    ap.add_argument("--upload", action="store_true", help="залить --out в HF rl_arch/conformal/ (нужен HF_TOKEN_WRITE)")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    sys.path.insert(0, os.path.abspath(a.coconet_dir))
    import yaml
    import coconet.pipeline as P
    import offline_rl as O
    import target_buffer as TB
    cfg = yaml.safe_load(open(os.path.join(a.coconet_dir, "configs", "experiment.yaml")))
    alpha = float(cfg["alpha"])
    tuned = {"spline": dict(num_knots=4, learning_rate=0.002, batch_size=32, encoder_hidden_dims=[64, 32]),
             "icnn": dict(learning_rate=0.002, batch_size=32, encoder_hidden_dims=[64, 32])}
    methods = [m for m in a.methods.split(",") if m]
    if a.pseudo:
        pseudo(a, P, cfg, alpha, tuned, methods)
        upload(a)
        return
    src, info = source_table(a.work, 20, 0)
    print(f"источники: {info}; строк {len(src['A'])}", flush=True)
    meta = O.pde_meta()
    res = {}
    for name in a.targets.split(","):
        eps, _ = TB.load(TARGETS[name], meta[name]["subdir"], print)
        tab, pi, n_new = TB.table_with(src, eps, name, meta)
        X, F, SD, Y, G = oof_dataset(tab, pi, a.horizon, a.members, a.folds)
        print(f"{name}: строк цели {n_new}, состояний с {a.horizon} действиями впереди {len(Y)}, цепочек "
              f"{len(np.unique(G))}; MAE прогноза по горизонтам {np.abs(Y - F).mean(0).round(3).tolist()}", flush=True)
        reps = []
        for rep in range(a.repeats):
            ms = [m for m in methods if m != "icnn" or rep < a.icnn_repeats]
            o = run_split(P, cfg, X, F, SD, Y, G, rep, alpha, ms, tuned)
            reps.append(o)
            print(f"  повтор {rep}: " + "; ".join(
                f"{k} cov {v['coverage']:.2f} size {v['size']:.3f} adapt {v['adapt']:+.2f}" if "coverage" in v
                else f"{k} {v.get('error', '')}" for k, v in o.items() if not k.startswith("_")), flush=True)
        res[name] = dict(n_states=int(len(Y)), mae=np.abs(Y - F).mean(0).tolist(), repeats=reps)
        json.dump(res, open(a.out, "w"), indent=1)
    print("\nитог (среднее по повторам):")
    for name, r in res.items():
        print(name)
        keys = list(dict.fromkeys(k for x in r["repeats"] for k in x if not k.startswith("_")))
        for k in keys:
            vals = [x[k] for x in r["repeats"] if "coverage" in x.get(k, {})]
            if not vals:
                errs = [x[k].get("error") for x in r["repeats"] if k in x]
                print(f"  {k:18s} ошибка: {errs[:1]}")
                continue
            print(f"  {k:18s} покрытие {np.mean([v['coverage'] for v in vals]):.3f}  размер "
                  f"{np.mean([v['size'] for v in vals]):.4f}  адаптивность {np.nanmean([v['adapt'] for v in vals]):+.2f}"
                  f"  (повторов {len(vals)})")


if __name__ == "__main__":
    main()
