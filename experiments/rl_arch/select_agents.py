#!/usr/bin/env python
"""Отбор членов комитета без оценки в среде: ранжирование обученных агентов на картах по собранному
буферу цели двумя способами. Запускается В КЕРНЕЛЕ Kaggle (суррогат и FQE учатся там, не на ноутбуке):

    python experiments/rl_arch/select_agents.py --pde poissonboltzmann2d \\
        --models t3m_ete_tbquart_g=rl_arch/models/convnext_dqn_cfix_rdelta_vb_ete_tbquart_seed1.pt,... \\
        [--extra-models ...] --method surrogate|fqe|both --tag w45 [--out sel.json]

Результат — JSON (локально в --out и в HF: rl_arch/select/<pde>_<tag>.json): по агенту — оценки обоих
способов, ранги, служебные числа. Метки агентов (до '=') — теги их оценок в среде (t3m_..._g):
select_report.py сверяет ранжирование с измеренными медианами по этим тегам. Чекпоинта, которого нет
в HF (HEAD-запрос) или локально, в отборе нет (пишется в missing).

Данные. Собранный буфер цели (--target-buffer; по умолчанию все сборщики направленного сбора уравнения:
pb2d tbuf_pb2d_w0..w3, ns2d tbuf_ns2d_w0..w9 — шире четверти, на которой учились агенты, чтобы
состояния не совпадали только с обучающими) загружается тем же загрузчиком, что у тренера
(target_buffer.load + offline_rl.episodes_to_arrays с --chain-fix, бюджет 7000). Вход агента
собирается по meta его чекпоинта так же, как offline_rl.main (режим состояния, скалярный контекст
ctx_kmax / ctx_budget / ctx_err) — тем же путём, что offline_rl.teacher_q. Жадное действие — argmax
скалярного Q по ВСЕМ 27 действиям (маски при развёртывании нет); страж (--guard-rollback) не
моделируется.

B1, суррогат (--method surrogate). Агенту нужны карты, а в суррогате (track3/surrogate.py: бустинг по
скалярному состоянию — задача, шаг, расход эпох по семействам, прошлое действие, log10 ошибки) карт
нет. Поэтому выбран вариант (i): агент действует только в НАСТОЯЩИХ состояниях буфера, где карты
есть, а суррогат оценивает последствия его выбора:
    F_m(s, a) = log10 итоговой ошибки (к 7000 эпохам), если в состоянии s сделать a, а дальше
                продолжать по ЧЛЕНУ m ансамбля (--cont, основное — первое): greedy — на каждом шаге
                действие с наименьшей предсказанной ошибкой, если улучшения не обещает ни одно —
                остановка; LBFGS:1:500 — повтор запасного действия стража до бюджета (это продолжение
                не эксплуатирует ошибки модели: жадное по члену может набрать до 70 коротких действий,
                которых в буфере не было, а суррогат учился на цепочках до 12 стадий);
    score_m(агент) = среднее по состояниям буфера F_m(s, a_агент(s));
    score(агент)   = среднее по членам m (выбор по среднему по ансамблю симуляторов, а не по одному
                     «подставленному» симулятору — Uncertainty-Aware Selection with Simulator
                     Ensembles, OpenReview LRJywDWhwK); разброс по членам и доля членов, у которых
                     агент лучший (p_best), — его неопределённость.
Продолжение одно и то же у всех агентов, поэтому разница оценок — только от их выбора в состояниях
буфера (одношаговое отклонение от общей политики продолжения). Почему не (ii), перегонка агента в
политику на скалярах и полный прогон в суррогате: перегонка по ~1000 состояниям теряет ровно то,
чем агенты различаются (карты), ошибка перегонки сопоставима с различиями агентов (все учились на
одних данных с одним учителем), а полный прогон накапливает ошибку суррогата за 10-20 шагов вне
распределения буфера. Оговорки (i): состояния — от политик сбора, а не от самого агента (сдвиг
распределения); суррогат без карт (если агент выигрывает тем, что видит ландшафт, суррогат этого не
заметит); продолжение жадное, а не политика агента; оценка — средняя по состояниям, а не итог эпизода
с начала. Ансамбль — mb_targets.fit_model на таблице target_buffer.surrogate_table (буферы остальных
задач + собранные цепочки цели, как у модельных целей рецепта), состояние строки — mb_targets.chain_states,
шаг — mb_targets._predict и _done (те же функции, что у модельных целей и щита).

B2, FQE (--method fqe). Fitted Q Evaluation жадной политики агента на ВСЕХ переходах буфера
(RoboFQE, arXiv 2610.18699: FQE поверх предобученного визуального кодировщика). Код — offline_rl.fqe
по импорту (тот же лосс: MSE к r + gamma (1 - d) Q'(s', pi(s')), мягкое обновление цели 0.005, Adam
1e-3, --fqe-epochs, --batch-size), без правок offline_rl.py: на время вызова его фабрика энкодера
make_encoder подменяется (FrozenFeatures): карты (каналы 0-3 входа) идут в ЗАМОРОЖЕННЫЙ кодировщик MAE
(rl_arch/encoders/cvprobe_w21cv_enc_mae_resnet.pt, нормировка карт — из его файла, BatchNorm в режиме
оценки), к 256 признакам добавляются 5 скаляров контекста, учится только MLP 261 -> 256 и линейная
голова FQE. Обучение и оценка — на всех переходах (train = test = весь буфер). Награда FQE — dlog
(убыль log10 ошибки за действие, первый шаг — от ошибки необученной сети), дисконт --fqe-gamma 1:
тогда V(s0) = log10 E0 - log10 E_итог, и предсказанная итоговая ошибка = log10 E0 - V(s0) в тех же
единицах, что метрика (меньше — лучше). Несколько сидов FQE (--fqe-seeds) — среднее и разброс.
Оговорки: FQE завышает действия вне поддержки буфера (агенты, часто выбирающие редкие в буфере
действия, выглядят лучше); дисконт 1 и короткие цепочки — бутстрэп без затухания; кодировщик заморожен
(RoboFQE дообучает свой).

Ранги: rank_surrogate, rank_fqe (1 — лучший). Сверка с измеренными медианами — select_report.py (локально).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(HERE, "track3"), HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_REPO = "danil-e/pinnacle-optuna-db"
SEL_DIR = "select"                          # rl_arch/select/<pde>_<tag>.json
ENC_MAE = "rl_arch/encoders/cvprobe_w21cv_enc_mae_resnet.pt"
BUDGET = 7000
# все сборщики направленного сбора (make_queue.py: tbuf, W28_K) — буфер оценки по умолчанию
TBUF = {"poissonboltzmann2d": ",".join(f"rl_arch/buffers_new/tbuf_pb2d_w{k}" for k in range(4)),
        "ns2d_liddriven": ",".join(f"rl_arch/buffers_new/tbuf_ns2d_w{k}" for k in range(10))}
METHODS = ("surrogate", "fqe", "both")


def log(*a):
    print(*a, flush=True)


def init_err(pde):
    """Ошибка необученной сети в единицах тренера — как --init-err рецепта (make_queue.py: INIT_ERR):
    ns2d 0.49 (медиана по буферу), остальные — init_err из pde_meta.json."""
    import offline_rl as O
    return 0.49 if pde == "ns2d_liddriven" else round(float(O.pde_meta()[pde]["init_err"]), 3)


# --------------------------------------------------------------------------------------------
# агенты
# --------------------------------------------------------------------------------------------

def parse_models(spec):
    """'метка=путь,путь,...' -> [(метка, путь)]; метка по умолчанию — имя файла без .pt."""
    out = []
    for tok in (x.strip() for x in (spec or "").split(",") if x.strip()):
        lab, _, path = tok.rpartition("=")
        path, lab = path.strip(), lab.strip()
        if not lab:
            base = os.path.basename(path)
            lab = base[:-3] if base.endswith(".pt") else base
        out.append((lab, path))
    labs = [x[0] for x in out]
    if len(set(labs)) != len(labs):
        raise ValueError(f"повтор меток агентов: {labs}")
    return out


def hf_exists(path):
    """Есть ли файл: локальный путь — на диске; иначе HEAD-запрос к датасету результатов. None — сеть
    недоступна (HF_HUB_OFFLINE или ошибка): тогда файл не отбрасывается, судьбу решит загрузка."""
    if os.path.exists(path):
        return True
    if os.path.isabs(path) or os.environ.get("HF_HUB_OFFLINE") == "1":
        return False if os.path.isabs(path) else None
    try:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_WRITE") or None
        return bool(HfApi().file_exists(OUT_REPO, path, repo_type="dataset", token=tok))
    except Exception as e:      # noqa: BLE001
        log(f"HEAD {path}: {type(e).__name__}: {str(e)[:120]}")
        return None


def fetch_file(path):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(OUT_REPO, path, repo_type="dataset")


def load_agent(path, device):
    """Чекпоинт offline_rl.py --save-model -> dict(net, mean, std, meta, variant). Сеть — как у
    online_eval_env.load_agent (offline_rl.QNet по ckpt_arch), режим оценки."""
    import torch
    import offline_rl as O
    ck = torch.load(fetch_file(path), map_location="cpu", weights_only=False)
    v = ck["variant"]
    if v in ("cnx_smdp", "stat_dqn") or ck.get("hl_gauss"):
        raise ValueError(f"{path}: вариант {v}{' с HL-Gauss' if ck.get('hl_gauss') else ''} не поддержан "
                         "(нужна Q-сеть на картах со скалярной головой)")
    net = O.QNet(v, device, **O.ckpt_arch(ck))
    net.model.load_state_dict(ck["state_dict"])
    net.model.eval()
    return dict(net=net, mean=np.asarray(ck["mean"], np.float32), std=np.asarray(ck["std"], np.float32),
                meta=ck.get("meta") or {}, variant=v, in_ch=int(O.ckpt_arch(ck)["in_ch"]))


def agent_inputs(ag, data, pde):
    """Вход агента на s и s' по meta его чекпоинта — как offline_rl.main собирал его при обучении
    (режим состояния, затем скалярный контекст chain_ctx). Возвращает (X, X2)."""
    import offline_rl as O
    meta = ag["meta"]
    mode = meta.get("state_mode", "full")
    if mode == "tasknorm":
        X, X2 = (O.apply_state_mode(data[k], "tasknorm", pde=pde) for k in ("S", "S2"))
    else:
        X, X2 = O.apply_state_mode(data["S"], mode, seed=101), O.apply_state_mode(data["S2"], mode, seed=202)
    if ag["in_ch"] > 4:
        if meta.get("pde_ctx"):
            raise ValueError("агент с описателем задачи (--pde-ctx, смесь УрЧП) не поддержан")
        c1, c2 = O.chain_ctx(data, k_max=int(meta.get("ctx_kmax", 10)), budget=int(meta.get("ctx_budget", 31000)),
                             with_err=(meta.get("ctx_err", "err") != "none"))
        X, X2 = O.attach_ctx(X, c1), O.attach_ctx(X2, c2)
    if X.shape[1] != ag["in_ch"]:
        raise ValueError(f"агент ждёт {ag['in_ch']} каналов, собрано {X.shape[1]}")
    return np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(X2, dtype=np.float32)


def q_values(ag, X, bs=512):
    """Скалярное Q (n, 27) агента: q_scalar на нормированном входе, как offline_rl.teacher_q."""
    import torch
    net = ag["net"]
    out = []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            x = torch.as_tensor((X[k:k + bs] - ag["mean"]) / ag["std"], device=net.device).float()
            out.append(net.q_scalar(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 27), np.float32)


# --------------------------------------------------------------------------------------------
# B1: суррогат
# --------------------------------------------------------------------------------------------

class MemberView:
    """Один член ансамбля под интерфейсом ансамбля (predict_ln -> (1, n)): шаг mb_targets._predict по
    одному симулятору."""
    members = 1

    def __init__(self, ens, m):
        self.ens, self.m = ens, int(m)

    def predict_ln(self, X, l, member=None):
        return np.asarray(self.ens.predict_ln(X, l, member=self.m))[None]


def parse_cont(spec):
    """Продолжение в суррогате: 'greedy' -> None, 'LBFGS:1:500' -> номер действия (повторять до бюджета)."""
    import mb_targets as MB
    if spec == "greedy":
        return None
    try:
        o, lr, ep = spec.split(":")
        lr, ep = float(lr), int(ep)
    except ValueError:
        raise ValueError(f"продолжение {spec!r}: нужно greedy или ОПТИМИЗАТОР:шаг:эпохи")
    for i, (on, l, e) in enumerate(MB.S.ACTIONS):
        if on.lower() == o.lower() and abs(l - lr) < 1e-12 and e == ep:
            return i
    raise ValueError(f"нет действия {spec!r} среди 27")


def surrogate_final(ens, tab, st, rows, acts, budget=BUDGET, max_steps=80, cont=None):
    """F (члены, len(rows)): log10 итоговой ошибки, если в состоянии rows[k] сделать acts[k], а дальше
    продолжать по члену m: cont=None — жадно (действие с наименьшей предсказанной ошибкой; без улучшения —
    стоп), cont=номер действия — повторять его до бюджета (как запасное действие стража)."""
    import agent_combo as AC
    import mb_targets as MB
    na = len(MB.S.ACTIONS)
    rows, acts = np.asarray(rows, dtype=np.int64), np.asarray(acts, dtype=np.int64)
    F = np.zeros((int(ens.members), len(rows)))
    steps = []
    for m in range(int(ens.members)):
        view = MemberView(ens, m)
        sub = AC.take(st, rows)
        P, ep = MB._predict(view, tab, sub, acts, budget)
        fin = P[0].astype(np.float64).copy()
        cur = AC.next_state(sub, acts, ep, P[0])
        idx = np.where(~MB._done(cur, budget))[0]
        cur = AC.take(cur, idx)
        n_st = 0
        for _ in range(int(max_steps)):
            k = len(idx)
            if not k:
                break
            n_st += 1
            if cont is None:
                rep = AC.take(cur, np.repeat(np.arange(k), na))
                a2 = np.tile(np.arange(na), k)
                P2, ep2 = MB._predict(view, tab, rep, a2, budget)
                P2 = P2[0].reshape(k, na)
                best = P2.argmin(1)
                lb = P2[np.arange(k), best]
                go = lb < cur["l"] - 1e-9
                flat = np.arange(k) * na + best
                rep, ep2 = AC.take(rep, flat), ep2[flat]
            else:
                rep, best = cur, np.full(k, int(cont), dtype=np.int64)
                P2, ep2 = MB._predict(view, tab, rep, best, budget)
                lb = P2[0]
                go = np.ones(k, dtype=bool)
            fin[idx[go]] = lb[go]
            nxt = AC.next_state(rep, best, ep2, lb)
            keep = np.where(go & ~MB._done(nxt, budget))[0]
            cur, idx = AC.take(nxt, keep), idx[keep]
        F[m] = fin
        steps.append(n_st)
    return F, steps


def surrogate_scores(ens, tab, st, rows, acts_by_agent, logged, budget=BUDGET, max_steps=80, cont=None):
    """Оценки B1 по агентам. acts_by_agent — {метка: действия на rows}; logged — действия буфера.
    F считается один раз на уникальные пары (строка, действие)."""
    labs = list(acts_by_agent)
    pairs = {}
    for a in [logged] + [acts_by_agent[x] for x in labs]:
        for r, x in zip(rows, a):
            pairs.setdefault((int(r), int(x)), len(pairs))
    key = np.array(list(pairs), dtype=np.int64).reshape(-1, 2)
    t0 = time.time()
    F, steps = surrogate_final(ens, tab, st, key[:, 0], key[:, 1], budget, max_steps, cont)
    log(f"суррогат: пар (состояние, действие) {len(key)} на {len(rows)} состояниях, членов {F.shape[0]}, "
        f"шагов продолжения {steps}, {time.time() - t0:.0f} с")

    def col(a):
        return np.array([pairs[(int(r), int(x))] for r, x in zip(rows, a)], dtype=np.int64)

    base = F[:, col(logged)]
    sc = {x: F[:, col(acts_by_agent[x])] for x in labs}              # (члены, строки)
    per_m = np.stack([sc[x].mean(1) for x in labs])                  # (агенты, члены)
    best_m = per_m.argmin(0)
    out = {}
    for i, x in enumerate(labs):
        out[x] = dict(surr_score=float(per_m[i].mean()), surr_sd=float(per_m[i].std()),
                      surr_members=[round(float(v), 5) for v in per_m[i]],
                      surr_p_best=float(np.mean(best_m == i)),
                      surr_vs_logged=float((sc[x] - base).mean()),
                      surr_final_err=float(10 ** per_m[i].mean()))
    return out, dict(pairs=int(len(key)), rows=int(len(rows)), members=int(F.shape[0]), steps=steps,
                     logged_score=float(base.mean()))


def build_surrogate(episodes, pde, args):
    """Таблица (буферы остальных задач + собранные цепочки цели) и ансамбль — как модельные цели рецепта."""
    import mb_targets as MB
    import target_buffer as TB
    tab = TB.surrogate_table(episodes, pde, args.surr_subdirs, args.surr_table, args.work, log=log)
    tab, ens, info = MB.fit_model(tab, members=args.members, work=args.work, log=log)
    return tab, ens, info


# --------------------------------------------------------------------------------------------
# B2: FQE на замороженном кодировщике MAE
# --------------------------------------------------------------------------------------------

def load_mae(path, device):
    """Кодировщик зонда CV (cv_probe.py --save-encoder): SmallResNet по его state_dict, нормировка карт
    из файла, режим оценки, без градиентов."""
    import torch
    import offline_rl as O
    f = torch.load(fetch_file(path), map_location="cpu", weights_only=False)
    if f.get("encoder") not in ("resnet", None):
        raise ValueError(f"{path}: кодировщик {f.get('encoder')!r}, нужен resnet")
    enc = O.make_encoder("resnet", in_ch=int(f.get("in_ch", 4)))
    enc.load_state_dict(f["state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device), np.asarray(f["mean"], np.float32), np.asarray(f["std"], np.float32)


def frozen_features_cls():
    import torch
    import torch.nn as nn

    class FrozenFeatures(nn.Module):
        """Энкодер FQE: вход — нормированный вход агента (offline_rl.fqe нормирует его статистиками
        агента); нормировка снимается, карты (0-3) идут в замороженный кодировщик MAE со своей
        нормировкой, контекст (каналы 4+, постоянные плоскости) берётся числами. Учится только MLP."""
        out_dim = 256

        def __init__(self, enc, enc_mean, enc_std, ag_mean, ag_std, n_ctx, hidden=256):
            super().__init__()
            self.enc = enc
            self.register_buffer("em", torch.as_tensor(enc_mean).float())
            self.register_buffer("es", torch.as_tensor(enc_std).float())
            self.register_buffer("am", torch.as_tensor(ag_mean).float())
            self.register_buffer("as_", torch.as_tensor(ag_std).float())
            self.n_ctx = int(n_ctx)
            self.mlp = nn.Sequential(nn.Linear(int(enc.out_dim) + self.n_ctx, hidden), nn.ReLU())
            self.out_dim = hidden

        def train(self, mode=True):
            super().train(mode)
            self.enc.eval()                 # BatchNorm кодировщика — всегда статистики предобучения
            return self

        def forward(self, x):
            raw = x * self.as_ + self.am
            with torch.no_grad():
                f = self.enc((raw[:, :4] - self.em) / self.es)
            parts = [f]
            if self.n_ctx:
                parts.append(raw[:, 4:4 + self.n_ctx, 0, 0])
            return self.mlp(torch.cat(parts, 1))

    return FrozenFeatures


@contextlib.contextmanager
def fqe_encoder(factory):
    """На время вызова offline_rl.fqe его энкодер — factory() вместо свежей CNN."""
    import offline_rl as O
    saved = O.make_encoder
    O.make_encoder = lambda kind, in_ch=4, size=26: factory()
    try:
        yield
    finally:
        O.make_encoder = saved


def fqe_score(ag, X, X2, data, mae, args, device, seeds=(0,)):
    """FQE жадной политики агента (offline_rl.fqe по импорту) на всех переходах. Возвращает dict:
    fqe_value (V(s0) в декадах), fqe_pred_log (log10 E0 - V), разброс по сидам FQE."""
    import offline_rl as O
    enc, em, es = mae
    cls = frozen_features_cls()
    d = dict(S=X, S2=X2, A=data["A"], R=data["R"], D=data["D"], FIRST=data["FIRST"])
    allm = np.ones(len(X), dtype=bool)
    ns = argparse.Namespace(fqe_epochs=int(args.fqe_epochs), batch_size=int(args.batch_size))
    vals, valls = [], []
    saved_g = O.GAMMA
    O.GAMMA = float(args.fqe_gamma)
    try:
        for s in seeds:
            with fqe_encoder(lambda: cls(enc, em, es, ag["mean"], ag["std"], X.shape[1] - 4)):
                fq = O.fqe(ag["net"], dict(mean=ag["mean"], std=ag["std"]), d, allm, allm, "mean", ns, int(s))
            vals.append(float(fq["fqe_value_init"]))
            valls.append(float(fq["fqe_value_all"]))
    finally:
        O.GAMMA = saved_g
    v = float(np.mean(vals))
    return dict(fqe_value=v, fqe_sd=float(np.std(vals)), fqe_values=[round(x, 5) for x in vals],
                fqe_value_all=float(np.mean(valls)), fqe_pred_log=float(math.log10(args.init_err) - v),
                fqe_final_err=float(10 ** (math.log10(args.init_err) - v)))


# --------------------------------------------------------------------------------------------

def ranks(scores):
    """{метка: оценка (меньше — лучше)} -> {метка: ранг с 1}."""
    order = sorted(scores, key=lambda k: (scores[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def load_target(spec, pde, reward="dlog"):
    import offline_rl as O
    import target_buffer as TB
    sub = O.pde_meta()[pde]["subdir"]
    episodes, info = TB.load(spec, sub, log=log)
    data = O.episodes_to_arrays(episodes, chain_fix=True, reward_form=reward, init_err=init_err(pde), budget=BUDGET)
    return episodes, data, info


def run(args, model=None, target=None):
    """Весь отбор. model=(tab, ens) — готовый суррогат (тесты), target=(episodes, data, info) — готовый
    буфер (тесты). Возвращает строку результата (она же пишется в --out и в HF)."""
    import torch
    import offline_rl as O
    import mb_targets as MB
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pde = args.pde
    args.init_err = init_err(pde)
    entries = [(lab, p, True) for lab, p in parse_models(args.models)] + \
              [(lab, p, False) for lab, p in parse_models(args.extra_models)]
    missing, agents = [], {}
    for lab, path, main in entries:
        ex = hf_exists(path)
        if ex is False:
            log(f"нет файла агента {lab}: {path} — не участвует")
            missing.append(dict(label=lab, path=path))
            continue
        try:
            agents[lab] = dict(load_agent(path, device), path=path, main=main)
        except Exception as e:      # noqa: BLE001 — один битый чекпоинт не роняет отбор
            log(f"агент {lab} ({path}) не загружен: {type(e).__name__}: {str(e)[:200]}")
            missing.append(dict(label=lab, path=path, error=f"{type(e).__name__}: {str(e)[:200]}"))
    if len(agents) < 2:
        raise SystemExit(f"для отбора нужно хотя бы два агента, загружено {len(agents)}")
    episodes, data, tb_info = target if target is not None else load_target(args.target_buffer, pde)
    n = len(data["A"])
    log(f"буфер оценки: переходов {n}, цепочек {len(np.unique(data['EP']))}; агентов {len(agents)} "
        f"(основных {sum(a['main'] for a in agents.values())}), нет {len(missing)}")
    res = {lab: dict(path=a["path"], main=a["main"], variant=a["variant"], encoder=a["meta"].get("encoder"),
                     two_nets=bool(a["meta"].get("two_nets")), in_ch=a["in_ch"]) for lab, a in agents.items()}
    for lab, a in agents.items():
        # входы не держатся в памяти (около 25 МБ на агента): FQE собирает их заново
        q = q_values(a, agent_inputs(a, data, pde)[0])
        a["greedy"] = q.argmax(1).astype(np.int64)
        res[lab].update(agree_logged=float(np.mean(a["greedy"] == data["A"])),
                        act_hist=np.bincount(a["greedy"], minlength=27).tolist())
    labs = list(agents)
    G = np.stack([agents[x]["greedy"] for x in labs])
    agree = (G[:, None, :] == G[None, :, :]).mean(2)
    row = dict(pde=pde, tag=args.tag, method=args.method, target_buffer=args.target_buffer, transitions=int(n),
               chains=int(len(np.unique(data["EP"]))), init_err=args.init_err, budget=BUDGET, missing=missing,
               agree_matrix=dict(labels=labs, values=np.round(agree, 4).tolist()), agents=res,
               target_info={k: v for k, v in (tb_info or {}).items() if k in ("files", "chains", "transitions", "sources")})
    if args.method in ("surrogate", "both"):
        if model is not None:
            (tab, ens), sinfo = model, dict(surr_sig="given")
        else:
            tab, ens, sinfo = build_surrogate(episodes, pde, args)
        pi = tab["NAMES"].index(pde)
        st, valid, _ = MB.chain_states(data, tab, np.array([pi]), np.ones(1), budget=BUDGET)
        rows = np.where(valid)[0]
        if args.max_rows and len(rows) > args.max_rows:
            rows = np.sort(np.random.default_rng(0).choice(rows, args.max_rows, replace=False))
        conts = [c for c in args.cont.split(",") if c]
        row["surrogate"] = dict(conts=conts, **{k: v for k, v in sinfo.items() if k in ("surr_rows", "surr_sig",
                                                                                       "surr_members")})
        for ci, cname in enumerate(conts):
            sc, info = surrogate_scores(ens, tab, st, rows, {x: agents[x]["greedy"][rows] for x in labs},
                                        data["A"][rows], BUDGET, args.max_steps, parse_cont(cname))
            rk = ranks({x: sc[x]["surr_score"] for x in labs})
            for x in labs:
                res[x].setdefault("surr_by_cont", {})[cname] = dict(sc[x], rank=rk[x])
                if ci == 0:                         # основная оценка B1 — первое продолжение
                    res[x].update(sc[x], rank_surrogate=rk[x])
            row["surrogate"][cname] = info
            log(f"ранжирование по суррогату, продолжение {cname} (меньше — лучше, log10 итоговой ошибки):")
            for x in sorted(labs, key=lambda k: rk[k]):
                log(f"  {rk[x]:2d}. {x:28s} {sc[x]['surr_score']:+.4f} ± {sc[x]['surr_sd']:.4f}  p_best "
                    f"{sc[x]['surr_p_best']:.2f}  против действий буфера {sc[x]['surr_vs_logged']:+.4f}")
    if args.method in ("fqe", "both"):
        mae = load_mae(args.encoder, device)
        t1 = time.time()
        fq = {}
        for x in labs:
            X, X2 = agent_inputs(agents[x], data, pde)
            fq[x] = fqe_score(agents[x], X, X2, data, mae, args, device, seeds=tuple(range(int(args.fqe_seeds))))
            res[x].update(fq[x])
        rk = ranks({x: fq[x]["fqe_pred_log"] for x in labs})
        for x in labs:
            res[x]["rank_fqe"] = rk[x]
        row["fqe"] = dict(encoder=args.encoder, epochs=int(args.fqe_epochs), seeds=int(args.fqe_seeds),
                          gamma=float(args.fqe_gamma), reward="dlog", sec=round(time.time() - t1, 1))
        log("ранжирование по FQE (меньше — лучше, предсказанный log10 итоговой ошибки):")
        for x in sorted(labs, key=lambda k: rk[k]):
            log(f"  {rk[x]:2d}. {x:28s} {fq[x]['fqe_pred_log']:+.4f} ± {fq[x]['fqe_sd']:.4f}")
    row["sec"] = round(time.time() - t0, 1)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=1, ensure_ascii=False)
        log(f"результат: {args.out}")
    if not args.no_upload:
        O.upload_result(row, f"{SEL_DIR}/{pde}_{args.tag}")
    return row


def build_parser():
    ap = argparse.ArgumentParser(description="отбор агентов без оценки в среде: суррогат и FQE")
    ap.add_argument("--pde", required=True, help="poissonboltzmann2d | ns2d_liddriven (имя из pde_meta.json)")
    ap.add_argument("--models", required=True, help="метка=путь через запятую (путь в HF или локальный)")
    ap.add_argument("--extra-models", default="", help="дополнительные агенты (в отчёте отдельно от основных)")
    ap.add_argument("--method", choices=METHODS, default="both")
    ap.add_argument("--tag", default="sel", help="имя результата: rl_arch/select/<pde>_<tag>.json")
    ap.add_argument("--target-buffer", default="", help="буфер оценки (по умолчанию все сборщики уравнения)")
    ap.add_argument("--encoder", default=ENC_MAE, help="кодировщик MAE для FQE")
    ap.add_argument("--members", type=int, default=5, help="членов ансамбля суррогата")
    ap.add_argument("--surr-subdirs", default="all", help="буферы остальных задач в таблице суррогата")
    ap.add_argument("--surr-table", default="", help="готовая таблица суррогата (как --mb-table)")
    ap.add_argument("--work", default="agent_combo_work", help="кэш таблицы и ансамбля")
    ap.add_argument("--max-steps", type=int, default=80, help="предел шагов продолжения в суррогате")
    ap.add_argument("--cont", default="greedy,LBFGS:1:500",
                    help="продолжения в суррогате через запятую: greedy и/или действие (повтор до бюджета); "
                         "основное — первое")
    ap.add_argument("--max-rows", type=int, default=0, help="случайная часть состояний буфера для B1 (0 — все)")
    ap.add_argument("--fqe-epochs", type=int, default=100)
    ap.add_argument("--fqe-seeds", type=int, default=3)
    ap.add_argument("--fqe-gamma", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="", help="локальный JSON с результатом")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    import offline_rl as O
    if args.pde not in O.pde_meta():
        ap.error(f"нет задачи {args.pde} в track3/pde_meta.json")
    if not args.target_buffer:
        if args.pde not in TBUF:
            ap.error(f"для {args.pde} нет буфера по умолчанию: задайте --target-buffer")
        args.target_buffer = TBUF[args.pde]
    try:
        n = len(parse_models(args.models)) + len(parse_models(args.extra_models))
    except ValueError as e:
        ap.error(str(e))
    if n < 2:
        ap.error("нужно хотя бы два агента (--models, --extra-models)")
    if args.members < 2 and args.method != "fqe":
        ap.error("--members >= 2: оценка B1 — среднее по членам ансамбля")
    if args.fqe_epochs < 1 or args.fqe_seeds < 1 or args.batch_size < 1 or not 0 < args.fqe_gamma <= 1:
        ap.error("--fqe-epochs, --fqe-seeds, --batch-size >= 1, 0 < --fqe-gamma <= 1")
    if args.max_steps < 1 or args.max_rows < 0:
        ap.error("--max-steps >= 1, --max-rows >= 0")
    try:
        conts = [parse_cont(c) for c in args.cont.split(",") if c]
    except ValueError as e:
        ap.error(f"--cont: {e}")
    if not conts and args.method != "fqe":
        ap.error("--cont: пустой список")
    return run(args)


if __name__ == "__main__":
    main()
