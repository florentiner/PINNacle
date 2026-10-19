#!/usr/bin/env python
"""Сколько журналов других уравнений нужно модели среды метода (суррогат модельных целей и учитель).

Метод учит на новом уравнении-цели агента по ~130 собранным переходам цели; модель среды (ансамбль HGB одного
шага, track3/surrogate.py) для плотных целей (offline_rl.py --mb-aux) и учителя MBPO (agent_combo.py) учится на
журналах цепочек других уравнений и на этих переходах. Журналы — скаляры (действие, эпохи, ошибка), без карт.
Здесь без обучения агентов измеряется, какой минимальный набор источников (уравнения x цепочки) даёт модель
среды не хуже, чем все 20 уравнений целиком, и сколько в нём переходов.

Данные. Источники — папки буферов --pool (all | solvable | список) без папок целей. Цели — --target ИМЯ=SPEC
(SPEC — собранный буфер цели, как offline_rl.py --target-buffer), обычно pb2d и ns2d. Таблица цели — та же,
что у ученика (target_buffer.table_with): строки источников и собранные строки цели.

Качество модели среды для цели. Собранные цепочки цели делятся на --folds фолдов по группам общего префикса
(GRP). На фолде модель учится на строках источников и строках цели вне фолда (как в методе: свои переходы
есть), предсказывает строки цели в фолде. Метрики (внефолдово, по всем фолдам):
  mae1     — MAE log10 ошибки после действия (один шаг; основной критерий);
  r2d      — R^2 изменения log10 ошибки за шаг;
  roll_mae — MAE log10 итоговой ошибки при прогоне модели по записанной цепочке от старта (её действия и
             фактические эпохи; цепочки, начатые с шага 0);
  roll_sp  — Spearman предсказанной и настоящей итоговой ошибки по цепочкам (порядок цепочек — то, на что
             опираются учитель и выбор действий).
Опорные точки: own — только строки цели (источников ноль); all — все источники целиком.

Порядок уравнений-источников (до --k-max):
  greedy — жадно: на каждом шаге добавляется уравнение (по --sel-chains цепочек), сильнее всего снижающее
           среднее по целям mae1, нормированное на own цели (выбор знает цели — верхняя оценка);
  div    — без знания целей: farthest-first по сигнатуре уравнения (стандартизованные описатель задачи CTX и
           среднее изменение log10 ошибки по каждому из 27 действий в его журналах), старт — медоид.
Кривая: для префиксов k = 1..k-max каждого порядка, всех --sizes (цепочек на уравнение, выбор — тот же
agent_combo.pick_source_chains, что у учителя --src-chains и ученика --mb-src-chains) и --seeds; плюс
all x n (все источники по n цепочек). В каждой точке — число строк источников (переходов) по уравнениям:
цена сбора журналов считается по ним (отчёт).

    python experiments/rl_arch/surr_data_study.py --tag w65 \\
        --target poissonboltzmann2d=rl_arch/buffers_new/tbuf_pb2d_w0 \\
        --target ns2d_liddriven=rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 \\
        --pool all --hours 6 --resume --out-dir /kaggle/working

Результат — <out-dir>/surrstudy_<tag>.json (и в HF rl_arch/surr_study/; заливка раз в --upload-min минут и в
конце). --resume продолжает с файла (локального, иначе из HF): посчитанные точки не пересчитываются.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "track3"))

import surrogate as S  # noqa: E402

OUT_REPO = "danil-e/pinnacle-optuna-db"
HF_DIR = "rl_arch/surr_study"


# --------------------------------------------------------------------------------------------
# Метрики модели среды для одной цели и одного набора строк источников
# --------------------------------------------------------------------------------------------

def target_folds(tab, pi, n_folds, seed=0):
    """Номер фолда у строк цели pi (по группам общего префикса GRP; -1 у остальных строк)."""
    m = tab["PDE"] == pi
    grp = tab["GRP"][m] if "GRP" in tab else tab["CH"][m]
    ug = np.unique(grp)
    k = max(2, min(int(n_folds), len(ug)))
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = {int(g): int(perm[i] % k) for i, g in enumerate(ug)}
    out = np.full(len(tab["A"]), -1, dtype=np.int64)
    out[np.where(m)[0]] = [fold_of[int(g)] for g in grp]
    return out, k


def target_chains(tab, pi):
    """Цепочки цели с шага 0: номера CH, действия, фактические эпохи, итоговая log10 ошибка."""
    rows = np.where(tab["PDE"] == pi)[0]
    out = []
    for ch in np.unique(tab["CH"][rows]):
        r = rows[tab["CH"][rows] == ch]
        r = r[np.argsort(tab["STEP"][r])]
        if not len(r) or int(tab["STEP"][r[0]]) != 0 or np.any(np.diff(tab["STEP"][r]) != 1):
            continue
        ep = tab["EPA"][r] if "EPA" in tab else S.EPOCHS[tab["A"][r]]
        out.append((int(ch), [int(a) for a in tab["A"][r]], [int(e) for e in ep], float(tab["LN"][r[-1]])))
    return out


class TargetEval:
    """Всё, что для цели считается один раз: признаки, фолды, цепочки."""

    def __init__(self, tab, pi, n_folds, members, seed, quick):
        self.tab, self.pi = tab, pi
        self.members, self.seed, self.quick = members, seed, quick
        self.X = S.table_features(tab)
        self.fold, self.k = target_folds(tab, pi, n_folds, seed)
        self.chains = target_chains(tab, pi)
        self.ctx = tab["CTX"][pi]

    def run(self, src_rows):
        tab, pi = self.tab, self.pi
        n = len(tab["A"])
        pred = np.full(n, np.nan)
        roll = {}
        for f in range(self.k):
            tr = np.concatenate([np.asarray(src_rows, dtype=np.int64), np.where((self.fold >= 0) & (self.fold != f))[0]])
            te = np.where(self.fold == f)[0]
            if not len(te) or not len(tr):
                continue
            ens = S.Ensemble(members=self.members, seed=self.seed,
                             **(dict(max_iter=60, learning_rate=0.15) if self.quick else {}))
            ens.fit(self.X[tr], tab["L"][tr], tab["LN"][tr], tab["CH"][tr])
            pred[te] = ens.predict_ln(self.X[te], tab["L"][te]).mean(0)
            te_ch = set(int(c) for c in np.unique(tab["CH"][te]))
            cs = [c for c in self.chains if c[0] in te_ch]
            if cs:
                fin = S.rollout(ens, self.ctx, [c[1] for c in cs], eps=[c[2] for c in cs]).mean(0)
                for c, p in zip(cs, fin):
                    roll[c[0]] = (float(p), c[3])
        m = (self.fold >= 0) & np.isfinite(pred)
        y, l = tab["LN"][m], tab["L"][m]
        pr = np.array(list(roll.values())) if roll else np.zeros((0, 2))
        return dict(mae1=float(np.abs(y - pred[m]).mean()) if m.any() else None,
                    r2d=S.r2(y - l, pred[m] - l) if m.any() else None,
                    roll_mae=float(np.abs(pr[:, 0] - pr[:, 1]).mean()) if len(pr) else None,
                    roll_sp=S.spearman(pr[:, 0], pr[:, 1]) if len(pr) >= 3 else None,
                    n_test=int(m.sum()), n_chains=int(len(pr)))


# --------------------------------------------------------------------------------------------
# Источники: строки набора, порядки уравнений
# --------------------------------------------------------------------------------------------

def source_rows(base, names, n, seed):
    """Строки источников names: у каждого n цепочек (agent_combo.pick_source_chains; n = 0 — все)."""
    import agent_combo as AC
    rows = []
    for name in names:
        pi = list(base["NAMES"]).index(name)
        m = np.where(base["PDE"] == pi)[0]
        if n:
            pick = AC.pick_source_chains(np.unique(base["CH"][m]), name, n, seed)
            m = m[np.isin(base["CH"][m], pick)]
        rows.append(m)
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def signature(base, names):
    """Сигнатура уравнения без знания целей: CTX и среднее изменение log10 ошибки по каждому действию."""
    n_act = len(S.ACTIONS)
    d = base["LN"] - base["L"]
    glob = np.array([np.nanmean(d[base["A"] == a]) if np.any(base["A"] == a) else 0.0 for a in range(n_act)])
    out = []
    for name in names:
        pi = list(base["NAMES"]).index(name)
        m = base["PDE"] == pi
        dyn = np.array([np.nanmean(d[m & (base["A"] == a)]) if np.any(m & (base["A"] == a)) else glob[a]
                        for a in range(n_act)])
        out.append(np.concatenate([np.asarray(base["CTX"][pi], dtype=np.float64), dyn]))
    Z = np.array(out)
    sd = Z.std(0)
    return (Z - Z.mean(0)) / np.where(sd > 0, sd, 1.0)


def diversity_order(base, names, k):
    """Farthest-first по сигнатуре, старт — медоид."""
    Z = signature(base, names)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    order = [int(np.argmin(D.sum(1)))]
    while len(order) < min(k, len(names)):
        rest = [i for i in range(len(names)) if i not in order]
        order.append(max(rest, key=lambda i: D[i, order].min()))
    return [names[i] for i in order]


# --------------------------------------------------------------------------------------------
# Исследование
# --------------------------------------------------------------------------------------------

class Study:
    def __init__(self, base, evals, cands, args, log=print, save=None):
        self.base, self.evals, self.cands, self.a, self.log = base, evals, cands, args, log
        self.save = save or (lambda res: None)
        self.res = dict(points={}, greedy=[], div=[], cands=list(cands), targets=list(evals))
        self.t0 = time.time()

    def over(self):
        return bool(self.a.hours) and time.time() - self.t0 > self.a.hours * 3600

    def point(self, key, names, n, seed):
        """Метрики всех целей для набора (names, n, seed); кэш по ключу."""
        if key in self.res["points"]:
            return self.res["points"][key]
        rows = source_rows(self.base, names, n, seed)
        per = {}
        for t, ev in self.evals.items():
            per[t] = ev.run(rows)
        cnt = {nm: int((self.base["PDE"][rows] == list(self.base["NAMES"]).index(nm)).sum()) for nm in names}
        out = dict(names=list(names), n=int(n), seed=int(seed), src_rows=int(len(rows)), rows_by_pde=cnt,
                   targets=per)
        self.res["points"][key] = out
        self.save(self.res)
        return out

    def score(self, pt):
        """Среднее по целям mae1, нормированное на own цели (меньше — лучше)."""
        own = self.res["points"]["own"]["targets"]
        return float(np.mean([pt["targets"][t]["mae1"] / own[t]["mae1"] for t in self.evals]))

    def run(self):
        a = self.a
        self.point("own", [], 0, 0)
        self.log(f"own: {self._fmt(self.res['points']['own'])}")
        if not self.res.get("div"):
            self.res["div"] = diversity_order(self.base, self.cands, a.k_max)
        self.log(f"порядок div: {self.res['div']}")
        # жадный выбор
        sel = list(self.res.get("greedy") or [])
        while len(sel) < min(a.k_max, len(self.cands)) and not self.over():
            best, best_s = None, None
            for c in self.cands:
                if c in sel:
                    continue
                pt = self.point(f"g|{'+'.join(sel + [c])}|{a.sel_chains}|0", sel + [c], a.sel_chains, 0)
                s = self.score(pt)
                if best_s is None or s < best_s:
                    best, best_s = c, s
            sel.append(best)
            self.res["greedy"] = list(sel)
            self.save(self.res)
            self.log(f"жадно k={len(sel)}: +{best} (балл {best_s:.3f})")
        # кривые
        for n in [0] + list(a.sizes):
            if self.over():
                break
            self.point(f"all|{n}|0", self.cands, n, 0)
            self.log(f"all x {n or 'все'}: {self._fmt(self.res['points'][f'all|{n}|0'])}")
        for order in ("greedy", "div"):
            names = self.res[order]
            for k in range(1, len(names) + 1):
                for n in a.sizes:
                    for sd in a.seeds:
                        if self.over():
                            self.log("предел --hours: продолжение — тот же --tag и --resume")
                            return self.res
                        key = f"{order}|{k}|{n}|{sd}"
                        pt = self.point(key, names[:k], n, sd)
                        if sd == a.seeds[0]:
                            self.log(f"{order} k={k} n={n}: {self._fmt(pt)}")
        self.res["finished"] = True
        self.save(self.res)
        return self.res

    def _fmt(self, pt):
        return f"строк источников {pt['src_rows']}; " + "; ".join(
            f"{t}: mae1 {v['mae1']:.3f}, roll {v['roll_mae']:.3f}, sp {v['roll_sp'] if v['roll_sp'] is None else round(v['roll_sp'], 2)}"
            for t, v in pt["targets"].items())


# --------------------------------------------------------------------------------------------
# Загрузка и запуск
# --------------------------------------------------------------------------------------------

def upload(path, tag, log):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        log("нет токена HF — результат только локально")
        return False
    from huggingface_hub import HfApi
    for attempt in range(5):
        try:
            HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"{HF_DIR}/{os.path.basename(path)}",
                                repo_id=OUT_REPO, repo_type="dataset", token=tok,
                                commit_message=f"surr_study {tag}")
            return True
        except Exception as e:                       # noqa: BLE001
            log(f"заливка не вышла ({type(e).__name__}: {e}), попытка {attempt + 1}")
            time.sleep(30 * (attempt + 1))
    return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", action="append", required=True, help="ИМЯ=SPEC собранного буфера цели")
    ap.add_argument("--pool", default="all", help="источники: all | solvable | список папок (без целей)")
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--sel-chains", type=int, default=20, help="цепочек на уравнение при жадном выборе")
    ap.add_argument("--sizes", default="5,10,20,50,100")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--members", type=int, default=1, help="членов ансамбля на точку (метод — 5; здесь дешевле)")
    ap.add_argument("--quick", action="store_true", help="быстрые настройки HGB (проверки)")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--work", default="agent_combo_work")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--upload-min", type=float, default=30.0)
    a = ap.parse_args(argv)
    a.sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    a.seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if not a.sizes or not a.seeds or a.k_max < 1 or a.sel_chains < 1 or a.folds < 2:
        ap.error("--sizes и --seeds не пусты, --k-max >= 1, --sel-chains >= 1, --folds >= 2")
    tg = {}
    for t in a.target:
        if "=" not in t:
            ap.error(f"--target {t!r}: нужен вид ИМЯ=SPEC")
        k, v = t.split("=", 1)
        tg[k.strip()] = v.strip()
    a.targets = tg
    return a


def main(argv=None):
    a = parse_args(argv)
    import agent_combo as AC
    import offline_rl as O
    import target_buffer as TB
    log = lambda *x: print(*x, flush=True)   # noqa: E731
    meta = O.pde_meta()
    tsubs = []
    for name in a.targets:
        if name not in meta or not meta[name].get("subdir"):
            raise SystemExit(f"--target {name}: нет папки буфера в pde_meta")
        tsubs.append(meta[name]["subdir"])
    cand_subs = [s for s in O.resolve_subdirs(a.pool) if s not in tsubs]
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"surrstudy_{a.tag}.json")
    base = AC.get_table(argparse.Namespace(table="", work=a.work), cand_subs, log)
    cands = [nm for nm in base["NAMES"] if (base["PDE"] == list(base["NAMES"]).index(nm)).any()]
    log(f"источники: {len(cands)} уравнений, строк {len(base['A'])}")
    evals = {}
    for name, spec in a.targets.items():
        eps, info = TB.load(spec, meta[name]["subdir"], log)
        tab, pi, n_new = TB.table_with(base, eps, name, meta)
        evals[name] = TargetEval(tab, pi, a.folds, a.members, 0, a.quick)
        log(f"цель {name}: строк {n_new}, цепочек с шага 0 {len(evals[name].chains)}, фолдов {evals[name].k}")
    last = [time.time()]

    def save(res, force=False):
        with open(path, "w") as f:
            json.dump(dict(res, settings=dict(pool=a.pool, k_max=a.k_max, sel_chains=a.sel_chains, sizes=a.sizes,
                                              seeds=a.seeds, folds=a.folds, members=a.members, quick=a.quick,
                                              targets=a.targets)), f)
        if not a.no_upload and (force or time.time() - last[0] > a.upload_min * 60):
            upload(path, a.tag, log)
            last[0] = time.time()

    st = Study(base, evals, cands, a, log, save)
    if a.resume:
        prev = None
        if os.path.exists(path):
            prev = json.load(open(path))
        elif not a.no_upload:
            try:
                from huggingface_hub import hf_hub_download
                prev = json.load(open(hf_hub_download(OUT_REPO, f"{HF_DIR}/{os.path.basename(path)}",
                                                      repo_type="dataset")))
            except Exception:                        # noqa: BLE001
                prev = None
        if prev:
            if prev.get("cands") != list(cands) or sorted(prev.get("targets", [])) != sorted(evals):
                raise SystemExit("--resume: отчёт посчитан на других источниках или целях; возьмите другой --tag")
            st.res.update({k: prev[k] for k in ("points", "greedy", "div") if k in prev})
            log(f"продолжение: точек {len(st.res['points'])}")
    res = st.run()
    save(res, force=True)
    log(f"готово: точек {len(res['points'])}, {path}")
    return res


if __name__ == "__main__":
    main()
