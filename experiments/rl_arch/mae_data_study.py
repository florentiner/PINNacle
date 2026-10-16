#!/usr/bin/env python
"""Сколько карт и каких уравнений нужно MAE-предобучению энкодера агента (ответ рецензенту о цене сбора буферов).

ENC_MAE (rl_arch/encoders/cvprobe_w21cv_enc_mae_resnet.pt) учился на ~180 тыс. карт ландшафта исходных случайных
буферов 22 уравнений; собрать такие буферы — тысячи GPU-часов. Здесь без обучения агентов, на тех же буферах,
измеряется, какой минимальный набор (уравнения x переходы) даёт то же представление, и сколько стоит его собрать.
Три режима (первый позиционный аргумент):

study — части A и B в одном кернеле (карты всех 22 уравнений грузятся один раз, cv_probe.load_data):
  Разбиение: у каждого уравнения ~20% цепочек (не больше --heldout-per-pde карт) отложены — на них считается
  восстановление; остальное — пул обучения и анализа. Карты с шагом 0 (нулевые) и нечисловые не участвуют.
  A. Распределения карт. Выборка из пула: до --ana-per-pde карт на уравнение, цепочки поделены пополам на
     «настоящие» (запрос) и «подставные» (источник), по --ana-per-pde/2. Три пространства: raw (нормировка по
     каналам, среднее 2x2 -> 4x13x13), shape (форма без уровня, cv_probe.shape_norm, 2x2) и mae (признаки ENC_MAE
     с его нормировкой); в каждом — стандартизация и PCA до --pca-dim. Считается: доля дисперсии главных
     компонент; PCA-2, t-SNE и UMAP (если пакет есть) — по --tsne-per-pde карт, панели по уравнениям (PNG);
     MMD^2 (несмещённая, RBF, ширина — медиана расстояний; --mmd-n карт) между уравнениями и между половинами
     одного уравнения (порог шума); расстояние Фреше гауссиан в PCA; иерархическая кластеризация уравнений
     (среднее связывание по MMD, дендрограмма); покрытие (Naeem et al. 2020, «density and coverage»): доля карт
     уравнения t, в k-NN шаре которых (k = --cov-k, радиус — до k-го соседа среди карт t) есть карта
     уравнения-источника p. Матрица покрытия 22x22; покрытие набора S — та же доля для объединения S.
     Жадный выбор ядра уравнений: кандидаты — уравнения без целей (pb2d, ns2d), цель — среднее покрытие всех 22
     уравнений; кривая покрытия всех / целей / нецелевых от числа уравнений и k95 (наименьшее k с 95%
     покрытия всех кандидатов). Покрытие от числа карт на уравнение (--cov-sizes) для ядер и всех кандидатов.
     Отбор карт k-center (farthest-first, Sener & Savarese 2018) по пулу карт кандидатов против случайного
     отбора того же объёма, состав отобранных карт по уравнениям. Избыточность: пары уравнений, покрывающие друг
     друга не хуже --redundant-tau своего самопокрытия; уравнения, покрытые ядром k95 без себя; отдельно пара
     хаотических (--chaotic). Индикаторы покрытия пишутся в _cov.npz (отчёт считает по ним жадный выбор с учётом
     цены сбора).
  B. Дешёвые прогоны MAE с прокси представления. Каждый арм — тот же ResNet, цель и оптимизатор, что у ENC_MAE
     (повтор ветки mae из cv_probe.pretrain: блоки 4x4 скрыты с долей --mae-ratio, декодер из вектора признаков
     восстанавливает форму карты), одинаковое число шагов --ssl-steps (1400, как у ENC_MAE и волн 46/53/54).
     Арм — (уравнения, переходов на уравнение — цепочки пула целиком, как --max-per-pde, сид). Армы:
       ref       all (22, весь пул — повтор ENC_MAE), free (без целей, весь пул), free_n<N> (без целей по
                 --core-n), rand (случайная инициализация, без обучения), enc_mae (сам ENC_MAE, только CKA и зонд);
       core-<пр> ядра жадного выбора в пространстве --select-space размеров --core-sizes по --core-n переходов;
       size      лучшее ядро (наименьшее k, у которого восстановление целей не хуже free_n<N> на --best-tol) на
                 --size-sweep переходах на уравнение;
       single    каждое уравнение отдельно по --single-n переходов;
       calib     повторы армов агента (для проверки прокси результатами агента): волна 46 (только цель: 130,
                 1000, все), 53 (22 уравнения по 30/100/300), 54 (3 x 100, 6 x 50, 6 x 150);
       drop      без целей по --core-n: без пары хаотических, без нерешаемых, без вторых членов избыточных пар;
       core-<др> ядра в других пространствах (если отличаются); random — случайные 3 уравнения без целей (2 сида);
       seed      второй сид у free_n<N>, ядра k=3 и лучшего ядра на 100 переходах (шум прокси).
     Прокси: (1) лосс восстановления MAE на отложенных картах каждого уравнения при одних и тех же масках у
     всех армов — цели, невиденные нецелевые, виденные; у rand — нулевое предсказание (~1); (2) линейная CKA
     (Kornblith et al. 2019) признаков арма с признаками ENC_MAE и арма all на --cka-n картах каждой цели;
     (3, вторично) гребневый линейный зонд исхода (изменение log10 ошибки) на переходах цели: признаки энкодера +
     скаляры и действие (как ssl-out), GroupKFold по цепочкам; база — без признаков. Энкодер каждого арма
     сохраняется в формате cv_probe (--encoder-init читает) и заливается в HF rl_arch/mae_study/encoders/.

cost — часть C (отдельный кернел): цена перехода с картами для каждого уравнения. Буферы времени не хранят,
  поэтому на каждом уравнении (отдельный подпроцесс с пределом --timeout-min) исполняется короткая цепочка той
  же средой, что у оценки и сбора (online_eval_env.run_seed, умолчания автокодировщика и поверхности):
  --cost-script (Adam 1e-3 1000 эпох, L-BFGS 0.5 500, PSO 1e-3 200) с картами после 1-го и 3-го действия
  (--state-every 2). Время каждого model.train, обучения автокодировщика и поверхности потерь замеряется.
  Цена перехода = сборка карт + эпохи действия x время эпохи его семейства, усреднённо по 27 действиям (случайная
  политика буферов); отчёт пересчитывает и по частотам действий в самих буферах. Неподнявшееся уравнение
  записывается в missing с хвостом лога.

Выход: <out-dir>/maestudy_<tag>.json (+ _cov.npz, PNG) и maecost_<tag>.json; в HF под rl_arch/mae_study/ (одним
коммитом раз в --upload-min минут и в конце: лимит HF — 128 коммитов в час). --resume продолжает с локального
файла или HF; --hours — мягкий предел (новый арм / уравнение не начинается). Отчёт — mae_study_report.py.

    python experiments/rl_arch/mae_data_study.py study --tag w56 --hours 7.5 --resume --out-dir /kaggle/working
    python experiments/rl_arch/mae_data_study.py cost --tag w56a --pdes poisson_boltzmann_2d,wave1d --resume
    python experiments/rl_arch/mae_data_study.py study --tag syn --synthetic 600 --ssl-steps 3 --no-upload --cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(HERE, "track3"), HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DDEBACKEND", "pytorch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cv_probe as CV  # noqa: E402

HF_DIR = "rl_arch/mae_study"
OUT_REPO = "danil-e/pinnacle-optuna-db"
ENC_MAE = "rl_arch/encoders/cvprobe_w21cv_enc_mae_resnet.pt"
TARGETS = ("poisson_boltzmann_2d", "ns2d_liddriven")
CHAOTIC = ("kuramoto_sivashinsky", "ns2d_longtime")
SHORT = {"poisson_boltzmann_2d": "pb2d", "ns2d_liddriven": "ns2d"}
W54_P3 = ("burgers1d", "heat2d_complexgeometry", "poisson2d_classic")
W54_P6 = W54_P3 + ("wave1d", "grayscott", "ns2d_backstep")
SPACES = ("raw", "shape", "mae")
COST_SCRIPT = "Adam:0.001:1000,LBFGS:0.5:500,PSO:0.001:200"
# 27 действий агента (online_eval_env.ACTION_TABLE): семейство и эпохи — для пересчёта цены перехода
ACTIONS = [(o, lr, e) for o, lrs, eps in (("Adam", (1e-2, 1e-3, 1e-4), (100, 1000, 2500)),
                                          ("LBFGS", (1.0, 0.5, 0.1), (100, 500, 1000)),
                                          ("PSO", (0.0, 1e-3, 1e-4), (100, 200, 300)))
           for lr in lrs for e in eps]
SYN_NAMES = ("syn_a", "syn_b", "syn_c", "syn_d", "syn_t")


def csv(s):
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def ints(s):
    return [int(x) for x in csv(s)]


def short(sd):
    return SHORT.get(sd, sd)


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    return float(o)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jsonable)
    os.replace(path + ".tmp", path)
    return path


# --------------------------------------------------------------------------------------------
# Заливка в HF: несколько файлов одним коммитом (как target_buffer.upload_files)
# --------------------------------------------------------------------------------------------

class Uploader:
    """Копит файлы и заливает их одним коммитом не чаще раза в every секунд (лимит HF — 128 коммитов в
    час на репозиторий). Без токена или с --no-upload ничего не делает."""

    def __init__(self, args, log):
        self.off = bool(args.no_upload)
        self.every = float(args.upload_min) * 60.0
        self.pending = {}
        self.last = time.time()
        self.log = log

    def add(self, local, name=None):
        self.pending[f"{HF_DIR}/{name or os.path.basename(local)}"] = local

    def flush(self, msg, force=False, tries=6):
        if self.off or not self.pending or (not force and time.time() - self.last < self.every):
            return False
        tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
        if not tok:
            self.log(f"нет токена HF — {len(self.pending)} файлов только локально")
            self.pending.clear()
            return False
        from huggingface_hub import CommitOperationAdd, HfApi
        for attempt in range(tries):
            try:
                HfApi().create_commit(repo_id=OUT_REPO, repo_type="dataset", token=tok, commit_message=msg,
                                      operations=[CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src)
                                                  for dst, src in self.pending.items()])
                self.log(f"залито в HF {len(self.pending)} файлов ({HF_DIR}/)")
                self.pending.clear()
                self.last = time.time()
                return True
            except Exception as e:      # noqa: BLE001
                self.log(f"повтор заливки {attempt}: {str(e)[:120]}")
                time.sleep(min(600, 20 * 2 ** attempt))
        return False


def hf_get(name, out_dir, log=print):
    """Файл отчёта: локально в out_dir, иначе из HF rl_arch/mae_study/. None — нет."""
    p = os.path.join(out_dir, name)
    if os.path.exists(p):
        return p
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(OUT_REPO, f"{HF_DIR}/{name}", repo_type="dataset")
    except Exception as e:      # noqa: BLE001
        log(f"  {name}: нет ни локально, ни в HF ({type(e).__name__})")
        return None


# --------------------------------------------------------------------------------------------
# Данные и разбиение
# --------------------------------------------------------------------------------------------

def synthetic_pdes(n=600, seed=0):
    """Синтетика для проверок: пять «задач» с разными узорами карт поверх наклона cv_probe.synthetic_data.
    syn_a и syn_b — один узор (пара-дубликат, как два хаотических уравнения), syn_c и syn_d — свои узоры,
    syn_t («цель») — смесь узоров a и c."""
    tab, maps = CV.synthetic_data(n, seed=seed, n_pde=5)
    rng = np.random.default_rng(seed + 1)
    g = np.linspace(-1.0, 1.0, maps.shape[-1])
    X, Y = np.meshgrid(g, g, indexing="ij")
    pat = {0: np.exp(-((X - 0.3) ** 2 + (Y + 0.2) ** 2) / 0.1), 2: np.sin(6.0 * X),
           3: np.sign(np.sin(5.0 * X) * np.sin(5.0 * Y))}
    pat[1] = pat[0]
    later = np.where(tab["STEP"] > 0)[0]
    p = tab["PDE"][later]
    w = rng.uniform(size=len(later))[:, None, None]
    P = np.stack([pat[k] for k in (0, 0, 2, 3, 0)])[p]
    P = np.where((p == 4)[:, None, None], w * pat[0] + (1.0 - w) * pat[2], P)
    amp = rng.uniform(1.0, 3.0, len(later))[:, None, None]
    maps[later, :3] += (amp * P)[:, None] * np.array([1.0, 0.7, 0.5])[None, :, None, None]
    maps = maps.astype(np.float32)
    tab["LV"] = CV.S.level6(maps)
    tab["NAMES"] = list(SYN_NAMES)
    tab["SUBDIRS"] = list(SYN_NAMES)
    return tab, maps


def load_study_data(args, log):
    """Буферы всех --subdirs целиком (или по --load-per-pde переходов): (tab, maps) как у cv_probe.load_data."""
    if args.synthetic:
        return synthetic_pdes(args.synthetic, seed=args.seed)
    ns = argparse.Namespace(synthetic=0, subdirs=args.subdirs, max_per_pde=args.load_per_pde,
                            max_files=args.max_files, map_dtype="float32", seed=args.seed)
    return CV.load_data(ns, log)


class Split:
    """Отложенные цепочки (восстановление) и пул (обучение армов, анализ) каждого уравнения."""

    def __init__(self, D, args):
        self.D = D
        self.valid = D.step > 0
        fin = np.ones(D.n, dtype=bool)
        for i in range(0, D.n, 8192):
            m = D.maps[i:i + 8192].float()
            fin[i:i + 8192] = torch.isfinite(m.flatten(1)).all(1).cpu().numpy()
        self.n_nonfinite = int((~fin & self.valid).sum())
        self.valid &= fin
        rng = np.random.default_rng(args.seed + 101)
        self.pool, self.held, self.summary = {}, {}, {}
        for p, sd in enumerate(D.subdirs):
            r = np.where(D.pde == p)[0]
            ug, inv = np.unique(D.ch[r], return_inverse=True)
            nval = np.bincount(inv, weights=self.valid[r], minlength=len(ug))
            held, cnt = [], 0
            if len(ug) >= 2:
                for c in rng.permutation(len(ug))[:max(1, int(0.2 * len(ug)))]:
                    if cnt >= args.heldout_per_pde:
                        break
                    held.append(c)
                    cnt += nval[c]
            hmask = np.isin(inv, held)
            self.held[sd] = r[hmask & self.valid[r]]
            self.pool[sd] = r[~hmask]
            self.summary[sd] = dict(n_rows=int(len(r)), n_maps=int(self.valid[r].sum()), chains=int(len(ug)),
                                    heldout_maps=int(len(self.held[sd])), pool_rows=int((~hmask).sum()),
                                    actions=np.bincount(D.A[r], minlength=27).astype(int).tolist())

    def sample(self, sd, n, seed):
        """Строки обучения уравнения sd: случайные цепочки пула целиком, переходов не больше n (0 — весь пул), как
        cv_probe.load_data --max-per-pde. При том же сиде меньший объём вложен в больший. Возвращает (строки с
        картами, число переходов)."""
        rows = self.pool[sd]
        if n and len(rows) > n:
            ug, inv, cnt = np.unique(self.D.ch[rows], return_inverse=True, return_counts=True)
            order = np.random.default_rng([int(seed), zlib.crc32(sd.encode())]).permutation(len(ug))
            take = order[np.cumsum(cnt[order]) <= n]
            if not len(take):
                take = order[:1]
            rows = rows[np.isin(inv, take)]
        return rows[self.valid[rows]], int(len(rows))

    def analysis(self, sd, n, rng):
        """Выборка анализа: (настоящие, подставные) — карты пула из разных половин цепочек, до n/2 в каждой."""
        rows = self.pool[sd][self.valid[self.pool[sd]]]
        ch = self.D.ch[rows]
        ug = np.unique(ch)
        if len(ug) >= 2:
            half = rng.permutation(ug)[::2]
            a, b = rows[np.isin(ch, half)], rows[~np.isin(ch, half)]
        else:
            a, b = rows[::2], rows[1::2]
        k = max(1, n // 2)
        a = np.sort(rng.choice(a, min(k, len(a)), replace=False)) if len(a) else a
        b = np.sort(rng.choice(b, min(k, len(b)), replace=False)) if len(b) else b
        return a, b


# --------------------------------------------------------------------------------------------
# Признаки, PCA, расстояния
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def encode(enc, D, rows, stats, amp, bs=2048):
    """Признаки энкодера (режим eval) для строк rows с нормировкой stats."""
    enc.eval()
    out = []
    for i in range(0, len(rows), bs):
        raw, _ = D.fetch(torch.as_tensor(np.asarray(rows[i:i + bs])))
        with CV._autocast(amp):
            out.append(enc(CV.build_input("raw", raw, None, stats)).float())
    return torch.cat(out) if out else torch.zeros(0, 1, device=D.dev)


@torch.no_grad()
def space_feats(space, D, rows, stats, ref, amp, bs=2048):
    """Карты в пространстве анализа: raw — нормировка по каналам и среднее 2x2; shape — форма без уровня, 2x2;
    mae — признаки ENC_MAE (ref = (энкодер, его статистики))."""
    if space == "mae":
        return encode(ref[0], D, rows, ref[1], amp, bs)
    out = []
    for i in range(0, len(rows), bs):
        raw, _ = D.fetch(torch.as_tensor(np.asarray(rows[i:i + bs])))
        x = CV.build_input("raw", raw, None, stats).clamp(-10.0, 10.0) if space == "raw" else CV.shape_norm(raw)
        out.append(F.avg_pool2d(x, 2).flatten(1).float())
    return torch.cat(out)


def pca_fit(X, dim):
    mu = X.mean(0)
    sd = X.std(0).clamp_min(1e-6)
    _, S, V = torch.linalg.svd((X - mu) / sd, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum().clamp_min(1e-12)
    return dict(mu=mu, sd=sd, V=V[:min(dim, V.shape[0])].T, var=var)


def pca_apply(P, X):
    return ((X - P["mu"]) / P["sd"]) @ P["V"]


def knn_radius(R, k):
    """Расстояние каждой точки R до её k-го соседа в R (сама точка не считается)."""
    d = torch.cdist(R, R)
    d.fill_diagonal_(float("inf"))
    return d.kthvalue(min(k, max(1, len(R) - 1)), dim=1).values


def min_dist_curve(R, Fk, sizes, gen):
    """Для каждой точки R — расстояние до ближайшей из первых n точек Fk (Fk в случайном порядке) при каждом
    n из sizes (0 — все точки Fk): (len(R), len(sizes))."""
    perm = torch.randperm(len(Fk), generator=gen).to(Fk.device)
    cm = torch.cdist(R, Fk[perm]).cummin(1).values
    cols = [(len(Fk) if n == 0 else min(n, len(Fk))) - 1 for n in sizes]
    return cm[:, cols]


def coverage_tensor(real, fake, names, k, sizes, seed=0):
    """B[t, p, i, j] — покрыта ли i-я настоящая карта уравнения t подставными картами уравнения p (первыми
    sizes[j] из них): есть подставная внутри k-NN шара (Naeem et al. 2020). valid[t, i] — точка есть."""
    dev = real[names[0]].device
    nr = max(len(real[t]) for t in names)
    B = torch.zeros(len(names), len(names), nr, len(sizes), dtype=torch.bool, device=dev)
    valid = torch.zeros(len(names), nr, dtype=torch.bool, device=dev)
    gen = torch.Generator().manual_seed(int(seed))
    for i, t in enumerate(names):
        R = real[t]
        if len(R) < 2:
            continue
        rad = knn_radius(R, k)
        valid[i, :len(R)] = True
        for j, p in enumerate(names):
            if len(fake[p]):
                B[i, j, :len(R)] = min_dist_curve(R, fake[p], sizes, gen) <= rad[:, None]
    return B, valid


def union_cov(B, valid, S, j=-1):
    """Покрытие каждого уравнения объединением подставных карт уравнений S (индексы): (T,)."""
    if not len(S):
        return torch.zeros(B.shape[0], device=B.device)
    u = B[:, list(S), :, j].any(1) & valid
    return u.sum(1).float() / valid.sum(1).clamp_min(1).float()


def greedy_core(B, valid, cand, tidx, names, j=-1):
    """Жадный выбор уравнений из cand, максимизирующий среднее покрытие всех уравнений."""
    non = [i for i in range(len(names)) if i not in tidx]
    chosen, rows, prev = [], [], 0.0
    while len(chosen) < len(cand):
        best = None
        for c in cand:
            if c in chosen:
                continue
            u = union_cov(B, valid, chosen + [c], j)
            s = float(u.mean())
            if best is None or s > best[0] + 1e-12:
                best = (s, c, u)
        s, c, u = best
        chosen.append(c)
        rows.append(dict(k=len(chosen), added=names[c], cov_all=s,
                         cov_targets=float(u[tidx].mean()) if tidx else float("nan"),
                         cov_nontargets=float(u[non].mean()) if non else float("nan"), gain=s - prev,
                         per_pde={names[i]: round(float(u[i]), 4) for i in range(len(names))}))
        prev = s
    final = rows[-1]["cov_all"] if rows else 0.0
    k95 = next((r["k"] for r in rows if r["cov_all"] >= 0.95 * final), len(rows))
    return dict(order=[names[c] for c in chosen], curve=rows, k95=int(k95))


def kcenter_order(X, m, seed=0):
    """Порядок отбора farthest-first (k-center greedy) первых m точек X."""
    g = torch.Generator().manual_seed(int(seed))
    i = int(torch.randint(len(X), (1,), generator=g))
    sel = [i]
    dmin = torch.cdist(X, X[i:i + 1]).squeeze(1)
    for _ in range(min(m, len(X)) - 1):
        i = int(dmin.argmax())
        sel.append(i)
        dmin = torch.minimum(dmin, torch.cdist(X, X[i:i + 1]).squeeze(1))
    return torch.as_tensor(sel, device=X.device)


def covered_by_order(R, rad, X, order, ms, bs=2048):
    """Покрыта ли каждая точка R первыми m точками X[order] при каждом m из ms: (len(R), len(ms))."""
    out = []
    cols = [min(m, len(order)) - 1 for m in ms]
    for i in range(0, len(R), bs):
        cm = torch.cdist(R[i:i + bs], X[order]).cummin(1).values[:, cols]
        out.append(cm <= rad[i:i + bs, None])
    return torch.cat(out)


def mmd2(X, Y, sigma):
    """Несмещённая MMD^2 с RBF-ядром ширины sigma."""
    g = -1.0 / (2.0 * sigma ** 2)
    kxx, kyy, kxy = (torch.exp(g * torch.cdist(a, b) ** 2) for a, b in ((X, X), (Y, Y), (X, Y)))
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return float("nan")
    return float((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
                 - 2.0 * kxy.mean())


def frechet(m1, c1, m2, c2):
    from scipy.linalg import sqrtm
    cm = sqrtm(c1 @ c2)
    cm = cm.real if np.iscomplexobj(cm) else cm
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2.0 * cm))


def linear_cka(X, Y):
    """Линейная CKA (Kornblith et al. 2019) по строкам-образцам."""
    X = X.double() - X.double().mean(0)
    Y = Y.double() - Y.double().mean(0)
    num = (Y.T @ X).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return float(num / den.clamp_min(1e-30))


def ridge_cv(X, y, groups, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), k=5, seed=0):
    """Гребневая регрессия с GroupKFold по цепочкам: лучший по R² на отложенных фолдах штраф. (R², штраф)."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    ug, inv = np.unique(groups, return_inverse=True)
    k = max(2, min(k, len(ug)))
    fold = np.random.default_rng(seed).permutation(len(ug))[inv] % k
    sst = ((y - y.mean()) ** 2).sum()
    best = (float("nan"), None)
    for lam in lams:
        pred = np.zeros_like(y)
        for f in range(k):
            tr, te = fold != f, fold == f
            if not te.any() or tr.sum() < 2:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            Xt = (X[tr] - mu) / sd
            ym = y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
            pred[te] = ((X[te] - mu) / sd) @ w + ym
        r2 = 1.0 - ((y - pred) ** 2).sum() / max(sst, 1e-30)
        if not np.isfinite(best[0]) or r2 > best[0]:
            best = (float(r2), lam)
    return best


# --------------------------------------------------------------------------------------------
# Рисунки (matplotlib, без экрана): панели по уравнениям, тепловые карты одного тона, кривые с подписями
# --------------------------------------------------------------------------------------------

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b0"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b"})
    return plt


def plot_facets(Y2, lab, names, targets, title, path):
    """Малые кратные: панель на уравнение — все точки серым, точки уравнения синим (цели — оранжевым)."""
    plt = _plt()
    Y2 = np.asarray(Y2)
    nc = 6
    nr = int(math.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.0 * nr + 0.4), squeeze=False)
    for k, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= len(names):
            ax.axis("off")
            continue
        m = lab == k
        ax.scatter(Y2[~m, 0], Y2[~m, 1], s=1, c=GRAY, alpha=0.25, linewidths=0, rasterized=True)
        ax.scatter(Y2[m, 0], Y2[m, 1], s=2, c=ORANGE if names[k] in targets else BLUE, linewidths=0,
                   rasterized=True)
        ax.set_title(short(names[k]) + (" (цель)" if names[k] in targets else ""), fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_matrix(M, names, title, path, order=None, label=""):
    plt = _plt()
    M = np.asarray(M, dtype=float)
    o = list(order) if order is not None else list(range(len(names)))
    fig, ax = plt.subplots(figsize=(0.32 * len(names) + 2.6, 0.32 * len(names) + 1.6))
    im = ax.imshow(M[np.ix_(o, o)], cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(o)))
    ax.set_yticks(range(len(o)))
    ax.set_xticklabels([short(names[i]) for i in o], rotation=90, fontsize=6)
    ax.set_yticklabels([short(names[i]) for i in o], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label=label)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_dendro(Z, names, title, path):
    plt = _plt()
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(0.3 * len(names) + 3, 3.2))
    dendrogram(Z, labels=[short(n) for n in names], ax=ax, leaf_rotation=90, leaf_font_size=7,
               color_threshold=0, above_threshold_color=BLUE)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_lines(series, xlabel, ylabel, title, path, xticks=None, logx=False):
    """series — [(подпись, x, y)] (не больше четырёх линий), подписи у концов линий."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for (lab, x, y), c in zip(series, (BLUE, ORANGE, AQUA, "#4a3aa7")):
        ax.plot(x, y, color=c, lw=2, marker="o", ms=4, label=lab)
        if len(x):
            ax.annotate(lab, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#52514e",
                        va="center")
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1], rotation=60, ha="right", fontsize=6)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_proxies(arms, path):
    """Три панели (одна шкала на панель): восстановление целей, CKA с ENC_MAE, R² зонда — от числа карт арма.
    Цвет — группа (ядра, размеры, одиночные), остальные серым."""
    plt = _plt()
    col = {"size": ORANGE, "single": AQUA}
    keys = (("recon_targets_mean", "восстановление целей (MSE)"), ("cka_ref_mean", "CKA с ENC_MAE (цели)"),
            ("probe_mean", "R² зонда исхода (цели)"))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, lab) in zip(axs, keys):
        for name, e in arms.items():
            v, n = e.get(key), e.get("n_maps")
            if v is None or not n or not np.isfinite(v):
                continue
            g = e.get("group", "")
            c = BLUE if g.startswith("core") else col.get(g, GRAY)
            ax.scatter([n], [v], s=14, c=c, linewidths=0)
        ax.set_xscale("log")
        ax.set_xlabel("карт в обучении MAE")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].scatter([], [], c=BLUE, s=14, label="ядра")
    axs[0].scatter([], [], c=ORANGE, s=14, label="размеры")
    axs[0].scatter([], [], c=AQUA, s=14, label="одно уравнение")
    axs[0].scatter([], [], c=GRAY, s=14, label="прочие")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _safe(fn, *a, log=print, **k):
    try:
        return fn(*a, **k)
    except Exception as e:      # noqa: BLE001 — рисунок не должен ронять кернел
        log(f"  рисунок не построен ({fn.__name__}): {type(e).__name__}: {str(e)[:120]}")
        return None


# --------------------------------------------------------------------------------------------
# Часть A: распределения карт
# --------------------------------------------------------------------------------------------

def analyze_space(space, D, split, targets, args, stats, ref, dev, amp, prefix, log):
    """Анализ одного пространства. Возвращает (запись отчёта, {ключ: массив} для _cov.npz, [PNG])."""
    t0 = time.time()
    names = list(D.subdirs)
    rng = np.random.default_rng(args.seed + 7)
    real_rows, fake_rows = {}, {}
    for sd in names:
        real_rows[sd], fake_rows[sd] = split.analysis(sd, args.ana_per_pde, rng)
    allrows = np.concatenate([np.r_[real_rows[sd], fake_rows[sd]] for sd in names])
    lab = np.concatenate([np.full(len(real_rows[sd]) + len(fake_rows[sd]), i) for i, sd in enumerate(names)])
    X = space_feats(space, D, allrows, stats, ref, amp)
    P = pca_fit(X, args.pca_dim)
    Z = pca_apply(P, X).float()
    del X
    off = np.cumsum([0] + [len(real_rows[sd]) + len(fake_rows[sd]) for sd in names])
    real = {sd: Z[off[i]:off[i] + len(real_rows[sd])] for i, sd in enumerate(names)}
    fake = {sd: Z[off[i] + len(real_rows[sd]):off[i + 1]] for i, sd in enumerate(names)}
    tidx = [names.index(t) for t in targets if t in names]
    cand = [i for i in range(len(names)) if i not in tidx]
    ent = dict(space=space, n_points=int(len(Z)), pca_var=P["var"][:10].cpu().tolist(),
               pca_var_cum_dim=float(P["var"][:P["V"].shape[1]].sum()), pdes=names)
    pngs = []
    # --- PCA-2, t-SNE, UMAP: по --tsne-per-pde карт уравнения
    sub = np.concatenate([np.where(lab == i)[0][rng.permutation((lab == i).sum())[:args.tsne_per_pde]]
                          for i in range(len(names))])
    Zs = Z[torch.as_tensor(sub, device=Z.device)].cpu().numpy()
    pngs.append(_safe(plot_facets, Zs[:, :2], lab[sub], names, targets, f"{space}: PCA (первые 2 компоненты)",
                      f"{prefix}_{space}_pca.png", log=log))
    if args.tsne and len(Zs) > 10:
        try:
            from sklearn.manifold import TSNE
            Y2 = TSNE(n_components=2, init="pca", perplexity=min(30.0, (len(Zs) - 1) / 3.0),
                      random_state=args.seed).fit_transform(Zs)
            pngs.append(_safe(plot_facets, Y2, lab[sub], names, targets, f"{space}: t-SNE",
                              f"{prefix}_{space}_tsne.png", log=log))
        except Exception as e:      # noqa: BLE001
            ent["tsne"] = f"не посчитан: {type(e).__name__}: {str(e)[:100]}"
        try:
            import umap   # noqa: F401
            Y2 = umap.UMAP(n_components=2, random_state=args.seed,
                           n_neighbors=min(15, len(Zs) - 1)).fit_transform(Zs)
            pngs.append(_safe(plot_facets, Y2, lab[sub], names, targets, f"{space}: UMAP",
                              f"{prefix}_{space}_umap.png", log=log))
        except Exception as e:      # noqa: BLE001 — пакета нет или он не поднялся: пропуск
            ent["umap"] = f"пропущен: {type(e).__name__}: {str(e)[:100]}"
    # --- MMD (RBF, медиана расстояний) и Фреше
    both = {sd: torch.cat([real[sd], fake[sd]]) for sd in names}
    pick = {sd: both[sd][torch.randperm(len(both[sd]), generator=torch.Generator().manual_seed(args.seed + i))
                         [:args.mmd_n].to(Z.device)] for i, sd in enumerate(names)}
    pooled = torch.cat(list(pick.values()))
    pooled = pooled[torch.randperm(len(pooled), generator=torch.Generator().manual_seed(1))[:2000].to(Z.device)]
    dd = torch.cdist(pooled, pooled)
    sigma = float(dd[dd > 0].median()) if (dd > 0).any() else 1.0
    T = len(names)
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            M[i, j] = M[j, i] = mmd2(pick[names[i]], pick[names[j]], sigma)
    self_mmd = []
    for sd in names:
        h = len(pick[sd]) // 2
        self_mmd.append(mmd2(pick[sd][:h], pick[sd][h:], sigma) if h >= 2 else float("nan"))
    fr = np.zeros((T, T))
    mom = {}
    for sd in names:
        b = both[sd].double().cpu().numpy()
        mom[sd] = (b.mean(0), np.cov(b, rowvar=False) if len(b) > 1 else np.zeros((b.shape[1], b.shape[1])))
    for i in range(T):
        for j in range(i + 1, T):
            fr[i, j] = fr[j, i] = frechet(*mom[names[i]], *mom[names[j]])
    ent.update(mmd_sigma=sigma, mmd=M.tolist(), mmd_self=self_mmd, frechet=fr.tolist())
    # --- кластеризация уравнений по MMD
    order = list(range(T))
    try:
        from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
        from scipy.spatial.distance import squareform
        Dm = np.clip(np.nan_to_num(M, nan=0.0), 0.0, None)
        Lk = linkage(squareform(Dm, checks=False), "average")
        order = [int(i) for i in leaves_list(Lk)]
        ent["clusters"] = {str(k): {names[i]: int(c) for i, c in enumerate(fcluster(Lk, k, "maxclust"))}
                           for k in (3, 5, 8) if k < T}
        ent["linkage"] = Lk.tolist()
        pngs.append(_safe(plot_dendro, Lk, names, f"{space}: кластеры уравнений (MMD, среднее связывание)",
                          f"{prefix}_{space}_dendro.png", log=log))
    except Exception as e:      # noqa: BLE001
        ent["clusters"] = f"не посчитаны: {type(e).__name__}"
    ent["order"] = [names[i] for i in order]
    pngs.append(_safe(plot_matrix, M, names, f"{space}: MMD² между уравнениями", f"{prefix}_{space}_mmd.png",
                      order=order, label="MMD²", log=log))
    # --- покрытие
    sizes = sorted({n for n in args.cov_sizes if n > 0}) + [0]
    B, valid = coverage_tensor(real, fake, names, args.cov_k, sizes, seed=args.seed)
    C = np.array([[float(union_cov(B, valid, [p])[t]) for p in range(T)] for t in range(T)])
    ent.update(cov_sizes=sizes, coverage=C.tolist(), coverage_self=np.diag(C).tolist(),
               n_real={sd: int(len(real[sd])) for sd in names}, n_fake={sd: int(len(fake[sd])) for sd in names})
    pngs.append(_safe(plot_matrix, C, names, f"{space}: покрытие карт строки картами столбца (k={args.cov_k})",
                      f"{prefix}_{space}_coverage.png", order=order, label="покрытие", log=log))
    core = greedy_core(B, valid, cand, tidx, names)
    core_any = greedy_core(B, valid, list(range(T)), tidx, names)
    ent["core_free"], ent["core_any"] = core, dict(order=core_any["order"], k95=core_any["k95"],
                                                   curve=[{k: v for k, v in r.items() if k != "per_pde"}
                                                          for r in core_any["curve"]])
    cv = core["curve"]
    pngs.append(_safe(plot_lines, [("все уравнения", [r["k"] for r in cv], [r["cov_all"] for r in cv]),
                                   ("цели (не в наборе)", [r["k"] for r in cv], [r["cov_targets"] for r in cv])],
                      "уравнений в ядре (жадно, без целей)", "покрытие", f"{space}: покрытие от числа уравнений",
                      f"{prefix}_{space}_coreset.png",
                      xticks=([r["k"] for r in cv], [f"+{short(r['added'])}" for r in cv]), log=log))
    # --- покрытие от числа карт на уравнение (ядра и все кандидаты)
    idx_of = {sd: i for i, sd in enumerate(names)}
    per_maps = {}
    sets = {f"core_k{k}": [idx_of[s] for s in core["order"][:k]] for k in args.core_sizes if k <= len(cand)}
    sets["free"] = cand
    for key, S in sets.items():
        per_maps[key] = [dict(n=n, cov_all=float(union_cov(B, valid, S, j).mean()),
                              cov_targets=float(union_cov(B, valid, S, j)[tidx].mean()) if tidx else float("nan"))
                         for j, n in enumerate(sizes)]
    ent["per_maps"] = per_maps
    xs = [n if n else max(int(min(ent["n_fake"].values())), 1) for n in sizes]
    keys = [k for k in (f"core_k{min(args.core_sizes)}", f"core_k{sorted(args.core_sizes)[len(args.core_sizes) // 2]}",
                        f"core_k{max(args.core_sizes)}", "free") if k in per_maps]
    pngs.append(_safe(plot_lines, [(k, xs, [r["cov_targets"] for r in per_maps[k]]) for k in dict.fromkeys(keys)],
                      "карт на уравнение (последняя точка — вся выборка)", "покрытие целей",
                      f"{space}: покрытие целей от числа карт", f"{prefix}_{space}_permaps.png", logx=True, log=log))
    # --- k-center по картам кандидатов против случайного отбора
    if cand:
        Xc = torch.cat([fake[names[i]] for i in cand])
        lc = np.concatenate([np.full(len(fake[names[i]]), i) for i in cand])
        ms = sorted({m for m in args.kc_sizes if m < len(Xc)}) or [len(Xc)]
        kc = kcenter_order(Xc, max(ms), seed=args.seed)
        rnd = torch.randperm(len(Xc), generator=torch.Generator().manual_seed(args.seed + 3)).to(Xc.device)
        res = dict(m=ms, kcenter={}, random={})
        for key, order_ in (("kcenter", kc), ("random", rnd)):
            per = []
            for i, sd in enumerate(names):
                if len(real[sd]) < 2:
                    continue
                per.append(covered_by_order(real[sd], knn_radius(real[sd], args.cov_k), Xc, order_, ms)
                           .float().mean(0).cpu().numpy())
            per = np.array(per)
            ok = [i for i, sd in enumerate(names) if len(real[sd]) >= 2]
            tsel = [ok.index(i) for i in tidx if i in ok]
            res[key] = dict(cov_all=per.mean(0).tolist(), cov_targets=per[tsel].mean(0).tolist() if tsel else None)
        res["composition"] = {str(m): {names[i]: int(c) for i, c in zip(*np.unique(lc[kc[:m].cpu().numpy()],
                                                                                    return_counts=True))}
                              for m in ms if m <= len(kc)}
        ent["kcenter"] = res
        pngs.append(_safe(plot_lines, [("k-center", ms, res["kcenter"]["cov_all"]),
                                       ("случайно", ms, res["random"]["cov_all"])],
                          "карт отобрано (кандидаты без целей)", "покрытие всех уравнений",
                          f"{space}: отбор карт k-center", f"{prefix}_{space}_kcenter.png", logx=True, log=log))
    # --- избыточность
    tau = args.redundant_tau
    red = []
    for a in cand:
        for b in cand:
            if a < b and C[a, a] > 0 and C[b, b] > 0:
                ra, rb = C[a, b] / C[a, a], C[b, a] / C[b, b]
                if ra >= tau and rb >= tau:
                    red.append(dict(a=names[a], b=names[b], a_by_b=round(ra, 3), b_by_a=round(rb, 3),
                                    mmd=round(M[a, b], 5)))
    ent["redundant_pairs"] = red
    k95 = [idx_of[s] for s in core["order"][:core["k95"]]]
    byc = {}
    for p in cand:
        others = [q for q in k95 if q != p]
        u = float(union_cov(B, valid, others)[p]) if others else 0.0
        byc[names[p]] = dict(in_core95=p in k95, cov_by_core_wo_self=round(u, 4), self=round(C[p, p], 4),
                             ratio=round(u / C[p, p], 3) if C[p, p] > 0 else None)
    ent["covered_by_core95"] = byc
    ent["redundant_given_core95"] = [s for s, v in byc.items() if not v["in_core95"] and (v["ratio"] or 0) >= tau]
    ch = [c for c in args.chaotic if c in idx_of]
    if len(ch) == 2:
        a, b = idx_of[ch[0]], idx_of[ch[1]]
        ua, ub, uab = (union_cov(B, valid, S) for S in ([a], [b], [a, b]))
        ent["chaotic"] = dict(pair=ch, a_by_b=C[a, b], b_by_a=C[b, a], self_a=C[a, a], self_b=C[b, b],
                              mmd=M[a, b], mmd_self_a=self_mmd[a], mmd_self_b=self_mmd[b],
                              gain_all_add_b_to_a=float(uab.mean() - ua.mean()),
                              gain_all_add_a_to_b=float(uab.mean() - ub.mean()),
                              cov_targets_a=float(ua[tidx].mean()) if tidx else None,
                              cov_targets_b=float(ub[tidx].mean()) if tidx else None,
                              rank_in_core=[core["order"].index(c) + 1 if c in core["order"] else None for c in ch])
    ent["seconds"] = round(time.time() - t0, 1)
    npz = {f"{space}_B": np.packbits(B.cpu().numpy().ravel()), f"{space}_Bshape": np.array(B.shape),
           f"{space}_valid": valid.cpu().numpy()}
    return ent, npz, [p for p in pngs if p]


# --------------------------------------------------------------------------------------------
# Часть B: прогоны MAE и прокси
# --------------------------------------------------------------------------------------------

def mae_loss(out, raw, m, C):
    """Лосс MAE cv_probe.pretrain: MSE формы карты (shape_norm) в скрытых пикселях."""
    rec = out.float().view_as(raw)
    return (((rec - CV.shape_norm(raw)) ** 2) * m).sum() / (m.sum().clamp_min(1) * C)


def mae_train(D, rows, args, dev, seed, log=print):
    """Повтор ветки mae из cv_probe.pretrain на строках rows (тот же порядок случайных чисел: при rows =
    pretrain_rows('mae', D)[1] и том же сиде веса совпадают), но возвращает и декодер — для восстановления
    отложенных карт. Возвращает (энкодер, декодер, статистики, сведения)."""
    t0 = time.time()
    torch.manual_seed(seed + 17)
    stats = CV.map_stats(D, rows, seed=seed)
    enc = CV.make_encoder("resnet", D.C, D.H).to(dev)
    aux = nn.Sequential(nn.Linear(enc.out_dim, 512), nn.GELU(), nn.Linear(512, D.C * D.H * D.W)).to(dev)
    params = list(enc.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    bs = min(args.batch, len(rows))
    per = max(1, len(rows) // bs)
    n_ep = max(1, math.ceil(args.ssl_steps / per))
    steps = n_ep * per
    sched = CV._schedule(opt, steps)
    amp = CV._amp(dev, args)
    scaler = CV._scaler(amp)
    gen = CV._gen(dev, seed + 23)
    rows_t = torch.as_tensor(rows, device=dev)
    first = last = None
    for _ in range(n_ep):
        perm = rows_t[torch.randperm(len(rows_t), device=dev)]
        tot, k = 0.0, 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            raw, _ = D.fetch(idx)
            x = CV.build_input("raw", raw, None, stats)
            m = CV.block_mask(len(x), D.H, D.W, args.mae_ratio, gen, dev)
            v = x.masked_fill(m, 0.0)
            with CV._autocast(amp):
                out = aux(enc(v))
            loss = mae_loss(out, raw, m, D.C)
            if CV._step(loss, opt, sched, scaler, params):
                tot, k = tot + loss.item(), k + 1
        last = tot / max(1, k)
        first = last if first is None else first
    enc.eval()
    aux.eval()
    return enc, aux, stats, dict(seconds=round(time.time() - t0, 1), steps=int(steps), epochs=int(n_ep),
                                 loss_first=first, loss_last=last)


class Evaluator:
    """Прокси представления: восстановление на отложенных картах (одни маски у всех армов), CKA с ENC_MAE и с
    армом all на картах целей, линейный зонд исхода на переходах целей."""

    def __init__(self, D, split, targets, args, dev, ref, log=print):
        self.D, self.targets, self.args, self.dev = D, targets, args, dev
        self.amp = CV._amp(dev, args)
        rng = np.random.default_rng(args.seed + 211)
        self.rows, self.masks = {}, {}
        for i, sd in enumerate(D.subdirs):
            r = split.held[sd]
            r = np.sort(rng.choice(r, min(len(r), args.eval_n), replace=False)) if len(r) else r
            if len(r):
                self.rows[sd] = r
                g = torch.Generator().manual_seed(args.seed + 1000 + i)
                self.masks[sd] = CV.block_mask(len(r), D.H, D.W, args.mae_ratio, g, torch.device("cpu"))
        self.cka_rows, self.probe = {}, {}
        base = [i for g in ("base", "act") for i in range(D.cols[g].start, D.cols[g].stop)]
        for t in targets:
            p = D.subdirs.index(t)
            r = np.where((D.pde == p) & split.valid)[0]
            self.cka_rows[t] = np.sort(rng.choice(r, min(len(r), args.cka_n), replace=False))
            r = r[np.isfinite(D.y[r])]
            r = np.sort(rng.choice(r, min(len(r), args.probe_n), replace=False))
            S = D.feat_t[torch.as_tensor(r, device=D.feat_t.device)][:, base].cpu().numpy()
            self.probe[t] = dict(rows=r, S=S, y=D.y[r], groups=D.ch[r],
                                 base=ridge_cv(S, D.y[r], D.ch[r], seed=args.seed))
        self.ref = {t: encode(ref[0], D, self.cka_rows[t], ref[1], self.amp) for t in targets} if ref else None
        self.ref_all = None
        self.zero = {sd: self.recon(None, None, None, sd) for sd in self.rows}

    @torch.no_grad()
    def recon(self, enc, aux, stats, sd, bs=1024):
        """Лосс MAE на отложенных картах sd с фиксированными масками; enc=None — нулевое предсказание."""
        rows, mask = self.rows[sd], self.masks[sd]
        tot = cnt = 0.0
        for i in range(0, len(rows), bs):
            raw, _ = self.D.fetch(torch.as_tensor(rows[i:i + bs]))
            m = mask[i:i + bs].to(self.D.dev)
            if enc is None:
                rec = torch.zeros_like(raw)
            else:
                x = CV.build_input("raw", raw, None, stats).masked_fill(m, 0.0)
                with CV._autocast(self.amp):
                    rec = aux(enc(x)).float().view_as(raw)
            tot += float((((rec - CV.shape_norm(raw)) ** 2) * m).sum())
            cnt += float(m.sum()) * self.D.C
        return tot / max(cnt, 1.0)

    def evaluate(self, enc, aux, stats, train_subdirs):
        out = {}
        if aux is not None:
            rec = {sd: self.recon(enc, aux, stats, sd) for sd in self.rows}
        else:
            rec = dict(self.zero)
        out["recon"] = {sd: round(v, 5) for sd, v in rec.items()}
        seen = [sd for sd in rec if sd not in self.targets and sd in train_subdirs]
        unseen = [sd for sd in rec if sd not in self.targets and sd not in train_subdirs]
        out["recon_targets"] = {t: rec[t] for t in self.targets if t in rec}
        out["recon_targets_mean"] = float(np.mean(list(out["recon_targets"].values()))) if out["recon_targets"] else None
        out["recon_unseen_mean"] = float(np.mean([rec[s] for s in unseen])) if unseen else None
        out["recon_seen_mean"] = float(np.mean([rec[s] for s in seen])) if seen else None
        out["recon_all_mean"] = float(np.mean(list(rec.values()))) if rec else None
        feats = {t: encode(enc, self.D, self.cka_rows[t], stats, self.amp) for t in self.targets}
        if self.ref is not None:
            out["cka_ref"] = {t: linear_cka(feats[t], self.ref[t]) for t in self.targets}
            out["cka_ref_mean"] = float(np.mean(list(out["cka_ref"].values())))
        if self.ref_all is not None:
            out["cka_all"] = {t: linear_cka(feats[t], self.ref_all[t]) for t in self.targets}
            out["cka_all_mean"] = float(np.mean(list(out["cka_all"].values())))
        out["probe"], out["probe_base"] = {}, {}
        for t in self.targets:
            pr = self.probe[t]
            Z = encode(enc, self.D, pr["rows"], stats, self.amp).cpu().numpy()
            r2, lam = ridge_cv(np.concatenate([Z, pr["S"]], 1), pr["y"], pr["groups"], seed=self.args.seed)
            out["probe"][t], out["probe_base"][t] = r2, pr["base"][0]
            out.setdefault("probe_lam", {})[t] = lam
        out["probe_mean"] = float(np.mean(list(out["probe"].values()))) if out["probe"] else None
        return out, feats


def make_arm(name, group, subs, n, seed=0, note=""):
    return dict(name=name, group=group, subdirs=list(subs), n_per=int(n), seed=int(seed), note=note)


def best_core_k(report, space, args):
    """Наименьшее k ядра пространства space, у которого восстановление целей не хуже free_n<N> на --best-tol;
    иначе k с лучшим восстановлением. None — ядра ещё не посчитаны."""
    arms = report["arms"]
    ks = [k for k in args.core_sizes if (arms.get(f"core_{space}_k{k}") or {}).get("recon_targets_mean") is not None]
    if not ks:
        return None
    val = {k: arms[f"core_{space}_k{k}"]["recon_targets_mean"] for k in ks}
    ref = (arms.get(f"free_n{args.core_n}") or {}).get("recon_targets_mean")
    if ref is not None:
        ok = [k for k in sorted(ks) if val[k] <= ref * (1.0 + args.best_tol)]
        if ok:
            return ok[0]
    return min(ks, key=lambda k: val[k])


def plan_arms(report, D, targets, args, phase):
    """Армы фазы (1 — опорные и ядра, 2 — размеры лучшего ядра, 3 — остальное); зависят от части A."""
    subs = list(D.subdirs)
    free = [s for s in subs if s not in targets]
    cn = args.core_n
    ana = report["analysis"]
    prim = args.select_space if args.select_space in ana else next(iter(ana), None)
    arms = []
    if phase == 1:
        arms += [make_arm("all", "ref", subs, 0), make_arm("rand", "ref", [], 0)]
        if report.get("enc_ref"):
            arms.append(make_arm("enc_mae", "ref", [], 0, note="ENC_MAE: только CKA и зонд"))
        arms += [make_arm("free", "ref", free, 0), make_arm(f"free_n{cn}", "ref", free, cn)]
        if prim:
            order = ana[prim]["core_free"]["order"]
            arms += [make_arm(f"core_{prim}_k{k}", f"core-{prim}", order[:k], cn) for k in args.core_sizes
                     if k <= len(order)]
    elif phase == 2 and prim:
        k = best_core_k(report, prim, args)
        if k is not None:
            order = ana[prim]["core_free"]["order"][:k]
            arms += [make_arm(f"best_n{n}", "size", order, n, note=f"ядро {prim} k={k}") for n in args.size_sweep]
    elif phase == 3:
        arms += [make_arm(f"one_{short(sd)}", "single", [sd], args.single_n) for sd in subs]
        for t in targets:
            arms += [make_arm(f"w46_{short(t)}_{nm}", "calib", [t], n) for nm, n in (("n130", 130), ("n1k", 1000),
                                                                                       ("all", 0))]
        arms += [make_arm(f"w53_n{n}", "calib", subs, n) for n in (30, 100, 300)]
        for nm, p, n in (("p3n100", W54_P3, 100), ("p6n50", W54_P6, 50), ("p6n150", W54_P6, 150)):
            if all(s in subs for s in p):
                arms.append(make_arm(f"w54_{nm}", "calib", p, n))
        unsolved = __import__("offline_rl").UNSOLVED_SUBDIRS
        arms.append(make_arm(f"free_nochaos_n{cn}", "drop", [s for s in free if s not in args.chaotic], cn))
        arms.append(make_arm(f"free_solvable_n{cn}", "drop", [s for s in free if s not in unsolved], cn))
        if prim:
            red = {r["b"] for r in ana[prim].get("redundant_pairs", [])}
            arms.append(make_arm(f"free_dedup_n{cn}", "drop", [s for s in free if s not in red], cn,
                                 note=f"без вторых членов избыточных пар: {sorted(red)}"))
        for sp, e in ana.items():
            if sp != prim:
                arms += [make_arm(f"core_{sp}_k{k}", f"core-{sp}", e["core_free"]["order"][:k], cn)
                         for k in args.core_sizes if k <= len(e["core_free"]["order"])]
        rng = np.random.default_rng(args.seed + 5)
        for i in range(2):
            if len(free) >= 3:
                arms.append(make_arm(f"rand3_s{i}", "random", sorted(rng.choice(free, 3, replace=False)), cn))
        arms.append(make_arm(f"free_n{cn}_s1", "seed", free, cn, seed=1))
        if prim:
            order = ana[prim]["core_free"]["order"]
            arms.append(make_arm(f"core_{prim}_k3_s1", "seed", order[:3], cn, seed=1))
            k = best_core_k(report, prim, args)
            if k is not None and 100 in args.size_sweep:
                arms.append(make_arm("best_n100_s1", "seed", order[:k], 100, seed=1))
    return [a for a in arms if all(s in subs for s in a["subdirs"])]


def arm_rows(arm, split, args):
    """Строки обучения арма (цепочки пула целиком) и число переходов по уравнениям."""
    rows, ntr = [], {}
    for sd in arm["subdirs"]:
        r, n = split.sample(sd, arm["n_per"], args.seed + arm["seed"])
        rows.append(r)
        ntr[sd] = n
    return (np.sort(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)), ntr


def arm_key(arm, split, args):
    """Ключ повтора: те же строки обучения и тот же сид — тот же энкодер (например, пул меньше запрошенного
    объёма или ядро совпало с другим армом). У армов без обучения — имя."""
    if not arm["subdirs"]:
        return ("name", arm["name"])
    rows, _ = arm_rows(arm, split, args)
    return (zlib.crc32(np.asarray(rows, dtype=np.int64).tobytes()), len(rows), arm["seed"])


def save_encoder(enc, stats, arm, ent, D, args):
    """Энкодер арма в формате cv_probe.save_encoder (читает offline_rl.py --encoder-init)."""
    path = os.path.join(args.out_dir, "encoders", f"maestudy_{args.tag}_{arm['name']}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mean, std = (t.detach().float().cpu().numpy() for t in stats)
    torch.save(dict(kind="ssl_encoder", encoder="resnet", ssl="mae", in_ch=D.C, size=D.H,
                    state_dict={k: v.detach().cpu() for k, v in enc.state_dict().items()}, mean=mean, std=std,
                    tag=f"{args.tag}_{arm['name']}", ssl_steps=ent.get("steps"), n_maps=ent.get("n_maps"),
                    loss_first=ent.get("loss_first"), loss_last=ent.get("loss_last"), loss_parts={},
                    ssl_args=dict(mae_ratio=args.mae_ratio), pdes=list(arm["subdirs"]), n_per=arm["n_per"],
                    fingerprint=D.fingerprint), path)
    return path


def load_encoder_file(path, D, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = CV.make_encoder(ck.get("encoder") or "resnet", D.C, D.H)
    enc.load_state_dict(ck["state_dict"])
    enc.to(dev).eval()
    stats = tuple(torch.as_tensor(np.asarray(ck[k]), dtype=torch.float32, device=dev).view(1, -1, 1, 1)
                  for k in ("mean", "std"))
    return enc, stats


def run_arm(arm, D, split, ev, args, dev, report, log):
    """Обучение (или загрузка) энкодера арма и прокси. Возвращает (запись, энкодер, статистики, признаки целей)."""
    t0 = time.time()
    ent = dict(group=arm["group"], subdirs=arm["subdirs"], n_per=arm["n_per"], seed=arm["seed"], note=arm["note"])
    aux = None
    if arm["name"] == "enc_mae":
        enc, stats = report["_ref"]
        ent.update(n_maps=None, n_transitions=None, file=report["enc_ref"])
    elif arm["name"] == "rand":
        torch.manual_seed(args.seed + arm["seed"] + 17)
        enc = CV.make_encoder("resnet", D.C, D.H).to(dev).eval()
        stats = report["_stats"]
        ent.update(n_maps=0, n_transitions=0)
    else:
        rows, ntr = arm_rows(arm, split, args)
        if len(rows) < 2:
            raise ValueError(f"у арма {arm['name']} меньше двух карт")
        enc, aux, stats, info = mae_train(D, rows, args, dev, args.seed + arm["seed"], log)
        ent.update(info, n_maps=int(len(rows)), n_transitions=int(sum(ntr.values())), n_tr_per_pde=ntr)
    res, feats = ev.evaluate(enc, aux, stats, set(arm["subdirs"]))
    ent.update(res)
    ent["eval_s"] = round(time.time() - t0 - ent.get("seconds", 0.0), 1)
    return ent, enc, aux, stats, feats


# --------------------------------------------------------------------------------------------
# Режим study
# --------------------------------------------------------------------------------------------

def study_paths(args):
    base = os.path.join(args.out_dir, f"maestudy_{args.tag}")
    return base + ".json", base + "_cov.npz"


def load_ref(args, D, dev, log):
    """ENC_MAE (--enc-ref): локальный путь или путь в HF-датасете результатов. None — без него."""
    if not args.enc_ref or args.enc_ref == "none":
        return None
    try:
        path = args.enc_ref
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(OUT_REPO, args.enc_ref, repo_type="dataset")
        return load_encoder_file(path, D, dev)
    except Exception as e:      # noqa: BLE001
        log(f"ENC_MAE ({args.enc_ref}) не загрузился: {type(e).__name__}: {str(e)[:120]} — пространство mae и CKA "
            f"с ним пропускаются")
        return None


def run_study(args, log):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"mae_data_study {args.tag}: устройство {dev}")
    tab, maps = load_study_data(args, log)
    D = CV.Data(tab, maps, dev, args.data_device)
    del maps
    targets = [t for t in args.targets if t in D.subdirs]
    if not targets:
        raise SystemExit(f"ни одной цели из {args.targets} нет среди уравнений {D.subdirs}")
    split = Split(D, args)
    log(f"данные: переходов {D.n}, уравнений {len(D.subdirs)}, карт с шагом > 0 {int(split.valid.sum())} "
        f"(нечисловых выброшено {split.n_nonfinite}), загрузка {time.time() - t0:.0f} с")
    jpath, npath = study_paths(args)
    report = dict(kind="mae_data_study", tag=args.tag, finished=False,
                  settings={k: v for k, v in vars(args).items() if not k.startswith("_")},
                  device=str(dev) + (f" {torch.cuda.get_device_name(dev)}" if dev.type == "cuda" else ""),
                  data=dict(n=D.n, pdes=list(D.subdirs), fingerprint=D.fingerprint, per_pde=split.summary,
                            n_nonfinite=split.n_nonfinite),
                  targets=targets, analysis={}, arms={}, plots=[])
    npz = {}
    if args.resume:
        pj = hf_get(os.path.basename(jpath), args.out_dir, log)
        if pj:
            with open(pj) as f:
                prev = json.load(f)
            if (prev.get("data") or {}).get("fingerprint") != D.fingerprint:
                raise SystemExit(f"--resume: отчёт {args.tag} посчитан на других данных; возьмите другой --tag")
            for k in ("analysis", "arms", "plots"):
                report[k] = prev.get(k) or report[k]
            report["elapsed_before_s"] = float(prev.get("elapsed_s") or 0.0)
            pn = hf_get(os.path.basename(npath), args.out_dir, log)
            if pn:
                with np.load(pn) as z:
                    npz = {k: z[k] for k in z.files}
            log(f"  продолжение: анализ {sorted(report['analysis'])}, армов готово {len(report['arms'])}")
    up = Uploader(args, log)
    amp = CV._amp(dev, args)
    ref = load_ref(args, D, dev, log)
    report["enc_ref"] = args.enc_ref if ref is not None else None
    report["_ref"] = ref
    report["_stats"] = CV.map_stats(D, np.where(split.valid)[0], seed=args.seed)

    def save(force_upload=False):
        report["elapsed_s"] = round(report.get("elapsed_before_s", 0.0) + time.time() - t0, 1)
        write_json({k: v for k, v in report.items() if not k.startswith("_")}, jpath)
        up.add(jpath)
        up.flush(f"rl_arch mae_study {args.tag}", force=force_upload)

    def time_left():
        return not args.hours or (time.time() - t0) < args.hours * 3600.0

    def killed():
        pat = CV.kill_pattern(args)
        if pat:
            report["killed_by"] = pat
            log(f"задача {args.tag!r} отменена записью {pat!r} списка killed.txt")
        return bool(pat)

    prefix = os.path.join(args.out_dir, f"maestudy_{args.tag}")
    # ---------------- A
    for space in args.spaces:
        if space in report["analysis"]:
            continue
        if space == "mae" and ref is None:
            log("пространство mae пропущено: нет ENC_MAE")
            continue
        log(f"A: пространство {space}")
        ent, z, pngs = analyze_space(space, D, split, targets, args, report["_stats"], ref, dev, amp, prefix, log)
        report["analysis"][space] = ent
        npz.update(z)
        npz["names"] = np.array(D.subdirs)
        with open(npath, "wb") as f:
            np.savez_compressed(f, **npz)
        up.add(npath)
        for p in pngs:
            up.add(p)
            report["plots"].append(os.path.basename(p))
        c = ent["core_free"]
        log(f"  {space}: ядро {c['order'][:6]}..., k95={c['k95']}, избыточные пары "
            f"{[(r['a'], r['b']) for r in ent['redundant_pairs']]}, {ent['seconds']:.0f} с")
        save()
    # ---------------- B
    ev = Evaluator(D, split, targets, args, dev, ref, log)
    report["zero_recon"] = ev.zero
    if "all" in report["arms"]:          # продолжение: признаки арма all для CKA — из его файла энкодера
        allp = hf_get(f"encoders/maestudy_{args.tag}_all.pt", args.out_dir, log)
        if allp:
            e_all, s_all = load_encoder_file(allp, D, dev)
            ev.ref_all = {t: encode(e_all, D, ev.cka_rows[t], s_all, amp) for t in targets}
    seen = {}
    stopped = None
    for phase in (1, 2, 3):
        for arm in plan_arms(report, D, targets, args, phase):
            if args.groups and arm["group"].split("-")[0] not in args.groups:
                continue
            name = arm["name"]
            key = arm_key(arm, split, args)
            if name in report["arms"]:
                if "same_as" not in report["arms"][name]:
                    seen.setdefault(key, name)
                continue
            if key in seen:
                report["arms"][name] = dict(group=arm["group"], subdirs=arm["subdirs"], n_per=arm["n_per"],
                                            seed=arm["seed"], same_as=seen[key])
                continue
            if not time_left() or killed():
                stopped = name
                break
            ent, enc, aux, stats, feats = run_arm(arm, D, split, ev, args, dev, report, log)
            if aux is not None:
                ent["file"] = f"{HF_DIR}/encoders/" + os.path.basename(save_encoder(enc, stats, arm, ent, D, args))
                up.add(os.path.join(args.out_dir, "encoders", os.path.basename(ent["file"])),
                       "encoders/" + os.path.basename(ent["file"]))
            if name == "all":
                ev.ref_all = feats
                for e in report["arms"].values():      # армы до all: CKA с ним досчитывается не будет
                    e.setdefault("cka_all_note", "посчитан до арма all")
            report["arms"][name] = ent
            seen[key] = name
            f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
            log(f"B {name} [{arm['group']}]: карт {ent.get('n_maps')}, восстановление целей "
                f"{f(ent.get('recon_targets_mean'))}, невиденных {f(ent.get('recon_unseen_mean'))}, CKA ENC_MAE "
                f"{f(ent.get('cka_ref_mean'))}, CKA all {f(ent.get('cka_all_mean'))}, зонд {f(ent.get('probe_mean'))}, "
                f"{ent.get('seconds', 0):.0f}+{ent['eval_s']:.0f} с")
            save()
        if stopped:
            break
    report["finished"] = stopped is None
    report["stopped_before"] = stopped
    pk = best_core_k(report, args.select_space, args)
    report["best_core"] = dict(space=args.select_space, k=pk,
                               subdirs=(report["analysis"].get(args.select_space) or {}).get("core_free", {})
                               .get("order", [])[:pk or 0])
    p = _safe(plot_proxies, {k: v for k, v in report["arms"].items() if "same_as" not in v},
              f"{prefix}_proxies.png", log=log)
    if p:
        up.add(p)
        report["plots"].append(os.path.basename(p))
    if stopped:
        log(f"остановка перед {stopped} (предел --hours {args.hours} или отмена); продолжение — тот же --tag и --resume")
    save(force_upload=True)
    log(f"отчёт: {jpath} ({report['elapsed_s']} с)")
    report["_paths"] = dict(report=jpath, cov=npath)
    return report


# --------------------------------------------------------------------------------------------
# Часть C: цена перехода с картами
# --------------------------------------------------------------------------------------------

def cost_derive(row, mix=None):
    """Цена перехода из замеров одной цепочки: время эпохи по семействам, сборка карт, ожидание по действиям
    (mix — частоты 27 действий; по умолчанию равномерно, как у случайной политики буферов)."""
    spe = {}
    for a in row.get("actions", []):
        if a["epochs"] > 0 and a["s"] is not None:
            s, e = spe.get(a["opt"], (0.0, 0))
            spe[a["opt"]] = (s + a["s"], e + a["epochs"])
    spe = {k: s / e for k, (s, e) in spe.items()}
    builds = [a + b for a, b in zip(row.get("ae_s", []), row.get("surface_s", []))]
    if not spe or not builds:
        return None
    fill = float(np.mean(list(spe.values())))
    w = np.full(27, 1.0 / 27) if mix is None else np.asarray(mix, dtype=float) / max(1e-12, float(np.sum(mix)))
    opt_s = float(sum(wi * e * spe.get(o, fill) for wi, (o, _, e) in zip(w, ACTIONS)))
    map_s = float(np.mean(builds))
    return dict(sec_per_epoch=spe, missing_family=[o for o in ("Adam", "LBFGS", "PSO") if o not in spe],
                map_s=map_s, opt_s=opt_s, per_transition_s=opt_s + map_s,
                per_transition_min=(opt_s + map_s) / 60.0, n_builds=len(builds))


def cost_pdes(args):
    import offline_rl as O
    meta = O.pde_meta()
    out = []
    for x in (args.pdes if args.pdes else list(O.ALL_SUBDIRS)):
        out.append(x if x in meta else O.subdir_to_pde(x))
    return out


def _run_child(pde, args, log):
    """Одна цепочка среды в подпроцессе (изоляция падений и зависаний CUDA, предел --timeout-min)."""
    cmd = [sys.executable, os.path.abspath(__file__), "cost-one", "--pde", pde, "--cost-script", args.cost_script,
           "--state-every", str(args.state_every), "--hidden-layers", args.hidden_layers, "--seed", str(args.seed),
           "--out-dir", args.out_dir, "--tag", args.tag]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_min * 60.0)
    except subprocess.TimeoutExpired as e:
        tail = (e.stdout or b"")[-600:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-600:]
        return None, f"предел {args.timeout_min} мин: {tail!s}"
    for ln in p.stdout.splitlines():
        if ln.startswith("COSTROW "):
            return json.loads(ln[8:]), None
    return None, f"код {p.returncode}: " + (p.stderr or p.stdout)[-800:]


CHILD = _run_child


def run_cost(args, log):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"maecost_{args.tag}.json")
    report = dict(kind="mae_cost", tag=args.tag, finished=False, script=args.cost_script,
                  state_every=args.state_every, settings={k: v for k, v in vars(args).items() if not k.startswith("_")},
                  rows={}, missing={})
    if args.resume:
        p = hf_get(os.path.basename(path), args.out_dir, log)
        if p:
            with open(p) as f:
                prev = json.load(f)
            report["rows"] = prev.get("rows") or {}
            report["missing"] = prev.get("missing") or {}
            report["elapsed_before_s"] = float(prev.get("elapsed_s") or 0.0)
            log(f"  продолжение: готово {sorted(report['rows'])}")
    up = Uploader(args, log)

    def save(force=False):
        report["elapsed_s"] = round(report.get("elapsed_before_s", 0.0) + time.time() - t0, 1)
        write_json(report, path)
        up.add(path)
        up.flush(f"rl_arch mae_study cost {args.tag}", force=force)

    stopped = None
    for pde in cost_pdes(args):
        if pde in report["rows"]:
            continue
        if (args.hours and time.time() - t0 >= args.hours * 3600.0) or CV.kill_pattern(args):
            stopped = pde
            break
        log(f"C: {pde} ...")
        row, err = CHILD(pde, args, log)
        if row is None:
            report["missing"][pde] = err
            log(f"  {pde}: НЕ замерено — {str(err)[-300:]}")
        else:
            row["derived"] = cost_derive(row)
            report["rows"][pde] = row
            report["missing"].pop(pde, None)
            d = row["derived"] or {}
            log(f"  {pde}: сборка карт {d.get('map_s', float('nan')):.1f} с, действие в среднем "
                f"{d.get('opt_s', float('nan')):.1f} с, переход {d.get('per_transition_min', float('nan')):.2f} мин "
                f"({row.get('device')})")
        save()
    report["finished"] = stopped is None
    report["stopped_before"] = stopped
    save(force=True)
    log(f"отчёт цены: {path}")
    return report


def run_cost_one(args):
    """Подпроцесс: одна цепочка --cost-script средой online_eval_env.run_seed с замером времени каждого
    model.train, обучения автокодировщика и поверхности потерь. Печатает строку COSTROW <json>."""
    import offline_rl as O
    import online_eval_env as E
    import deepxde as dde
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    from landscape_visualization._aux.visualization_model import VisualizationModel
    rec = dict(train=[], ae=[], surf=[])

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(orig, key, with_iter=False):
        def wrap(self, *a, **k):
            sync()
            t = time.time()
            out = orig(self, *a, **k)
            sync()
            rec[key].append((k.get("iterations"), time.time() - t) if with_iter else time.time() - t)
            return out
        return wrap
    dde.Model.train = timed(dde.Model.train, "train", True)
    VisualizationModel.train = timed(VisualizationModel.train, "ae")
    PlotLossSurface.save_equation_loss_surface = timed(PlotLossSurface.save_equation_loss_surface, "surf")
    meta = O.pde_meta()[args.pde]
    argv = ["--policy", "script", "--script", args.cost_script, "--script-tail", "stop", "--pde", args.pde,
            "--budget", "31000", "--state-every", str(args.state_every), "--hidden-layers", args.hidden_layers,
            "--save-dir", os.path.join(args.out_dir, "maecost_runs"), "--tag", f"maecost_{args.tag}",
            "--seeds", str(args.seed)]
    argv += ["--plain-fnn"] if meta.get("inverse") else []
    ea = E.build_parser().parse_args(argv)
    ea._ckpt_name = None
    ea._rollout = None
    t0 = time.time()
    row = E.run_seed(args.seed, ea)
    chain = [c for c in row.get("chain", []) if c[0] != "BOOST"]
    acts = [dict(opt=c[0], lr=c[1], epochs=int(c[2]), s=(rec["train"][i][1] if i < len(rec["train"]) else None))
            for i, c in enumerate(chain)]
    out = dict(pde=args.pde, device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
               actions=acts, ae_s=rec["ae"], surface_s=rec["surf"], n_train_calls=len(rec["train"]),
               t_ae_s=row.get("t_ae_s"), t_surface_s=row.get("t_surface_s"), t_opt_s=row.get("t_opt_s"),
               elapsed_s=round(time.time() - t0, 1), l2re=row.get("l2re"), ae_epochs=ea.ae_epochs,
               n_save_models=ea.n_save_models, hidden_layers=ea.hidden_layers, plain_fnn=bool(ea.plain_fnn),
               ae_batch=ea.ae_batch)
    print("COSTROW " + json.dumps(out, default=_jsonable), flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Разбор аргументов
# --------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("study", "cost", "cost-one"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0.0, help="мягкий предел: новый арм/уравнение не начинается")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--upload-min", type=float, default=30.0, help="заливка в HF не чаще раза в N минут (и в конце)")
    g = ap.add_argument_group("данные (study)")
    g.add_argument("--subdirs", default="all")
    g.add_argument("--load-per-pde", type=int, default=0, help="грузить не больше N переходов на уравнение (0 — все)")
    g.add_argument("--max-files", type=int, default=0)
    g.add_argument("--data-device", default="auto", choices=["cpu", "cuda", "auto"])
    g.add_argument("--synthetic", type=int, default=0, help="синтетика (5 задач, N переходов) вместо буферов")
    g.add_argument("--targets", default=",".join(TARGETS))
    g.add_argument("--chaotic", default=",".join(CHAOTIC), help="пара хаотических уравнений для отдельного отчёта")
    g.add_argument("--heldout-per-pde", type=int, default=400, help="отложенных карт на уравнение (не больше 20%% цепочек)")
    g.add_argument("--enc-ref", default=ENC_MAE, help="ENC_MAE: путь в HF или локально; none — без него")
    g = ap.add_argument_group("A: анализ")
    g.add_argument("--spaces", default=",".join(SPACES))
    g.add_argument("--select-space", default="raw", help="пространство ядер для армов core и size")
    g.add_argument("--ana-per-pde", type=int, default=2000)
    g.add_argument("--pca-dim", type=int, default=50)
    g.add_argument("--tsne-per-pde", type=int, default=300)
    g.add_argument("--no-tsne", dest="tsne", action="store_false")
    g.add_argument("--mmd-n", type=int, default=500)
    g.add_argument("--cov-k", type=int, default=5)
    g.add_argument("--cov-sizes", default="10,30,100,300", help="карт на уравнение для кривой покрытия (плюс все)")
    g.add_argument("--kc-sizes", default="22,66,220,660,2200,6600")
    g.add_argument("--redundant-tau", type=float, default=0.8)
    g = ap.add_argument_group("B: прогоны MAE")
    g.add_argument("--groups", default="", help="только эти группы армов (ref,core,size,single,calib,drop,random,seed)")
    g.add_argument("--core-sizes", default="1,2,3,4,6")
    g.add_argument("--core-n", type=int, default=2000, help="переходов на уравнение у ядер и армов drop/random")
    g.add_argument("--single-n", type=int, default=2000)
    g.add_argument("--size-sweep", default="10,30,100,300,1000")
    g.add_argument("--best-tol", type=float, default=0.05)
    g.add_argument("--ssl-steps", type=int, default=1400)
    g.add_argument("--batch", type=int, default=512)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--wd", type=float, default=0.01)
    g.add_argument("--mae-ratio", type=float, default=0.6)
    g.add_argument("--eval-n", type=int, default=400)
    g.add_argument("--cka-n", type=int, default=2000)
    g.add_argument("--probe-n", type=int, default=4000)
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--cpu", action="store_true")
    g = ap.add_argument_group("C: цена перехода")
    g.add_argument("--pdes", default="", help="уравнения (имена или папки буферов); по умолчанию все 22")
    g.add_argument("--pde", default="", help="cost-one: одно уравнение")
    g.add_argument("--cost-script", default=COST_SCRIPT)
    g.add_argument("--state-every", type=int, default=2)
    g.add_argument("--hidden-layers", default="100*5")
    g.add_argument("--timeout-min", type=float, default=45.0)
    return ap


def parse(argv):
    args = build_parser().parse_args(argv)
    args.targets, args.chaotic, args.spaces = csv(args.targets), csv(args.chaotic), csv(args.spaces)
    args.pdes, args.groups = csv(args.pdes), csv(args.groups)
    args.cov_sizes, args.kc_sizes = ints(args.cov_sizes), ints(args.kc_sizes)
    args.core_sizes, args.size_sweep = ints(args.core_sizes), ints(args.size_sweep)
    bad = [s for s in args.spaces if s not in SPACES]
    if bad:
        raise SystemExit(f"--spaces: неизвестные пространства {bad}, есть {SPACES}")
    if args.select_space not in SPACES:
        raise SystemExit(f"--select-space из {SPACES}")
    known = ("ref", "core", "size", "single", "calib", "drop", "random", "seed")
    if any(gr not in known for gr in args.groups):
        raise SystemExit(f"--groups: группы из {known}")
    if args.ssl_steps < 1 or args.batch < 2 or not 0.0 < args.mae_ratio < 1.0:
        raise SystemExit("--ssl-steps >= 1, --batch >= 2, --mae-ratio из (0, 1)")
    if args.cov_k < 1 or args.ana_per_pde < 4 or not args.core_sizes or min(args.core_sizes) < 1:
        raise SystemExit("--cov-k >= 1, --ana-per-pde >= 4, --core-sizes — положительные")
    if args.mode == "cost":
        if args.state_every < 1:
            raise SystemExit("--state-every >= 1")
        import online_eval_env as E
        try:
            E.parse_chain(args.cost_script)
        except ValueError as e:
            raise SystemExit(f"--cost-script: {e}")
        cost_pdes(args)
    if args.mode == "cost-one" and not args.pde:
        raise SystemExit("cost-one: нужен --pde")
    if args.mode == "study" and not args.synthetic:
        import offline_rl as O
        O.resolve_subdirs(args.subdirs)
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    if args.mode == "cost-one":
        return run_cost_one(args)
    args._kill = True
    try:        # как cv_probe.main: отменённая задача выходит до загрузки буферов
        from killswitch import exit_if_killed
        exit_if_killed(args.tag, remote=True, what="исследование данных MAE")
    except ImportError:
        pass
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "cost":
        return run_cost(args, log)
    return run_study(args, log)


if __name__ == "__main__":
    main()
