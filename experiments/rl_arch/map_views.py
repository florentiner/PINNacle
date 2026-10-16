#!/usr/bin/env python
"""Несколько видов одной карты ландшафта и данные для предобучения энкодера без чужих буферов.

Зачем. Карта состояния (online_eval_env.py, конвейер статьи) — поверхность лоссов в двумерной скрытой плоскости
автокодировщика, обученного заново на каждом шаге по n сохранённым копиям весов PINN. Плоскость ничем не
закреплена (веса anchor/lastzero/polars у AE_MODEL_PARAMS нулевые): другой случайный старт автокодировщика даёт
другой поворот, отражение, масштаб и изгиб той же поверхности. Это шум наблюдения, который агент видит и при
развёртывании (волна 10, вторая сборка карт: решение лидера ns2d меняется на 0-60% шагов). Значит, по одному
снимку траектории можно бесплатно для PINN получить V видов одного и того же ландшафта — данные для
самообучения энкодера на картах ОДНОЙ целевой задачи (MAE на всех видах, согласованность видов), вместо
предобучения на буферах 22 уравнений.

Что здесь:
  * train_view_aes — V автокодировщиков статьи (UniformAutoencoder, тот же RMSprop, косинусный план с
    перезапусками, лосс restore * rec_weight, лучшая копия по общему лоссу на каждой сотой эпохе, ранний останов
    по терпению) обучаются ОДНИМ проходом: параметры сложены по оси вида (torch.func.vmap). RMSprop
    поэлементный, лоссы видов складываются, поэтому каждый вид учится ровно так, как учился бы отдельно
    (VisualizationModel.train), а время — почти как у одного: автокодировщик крошечный (n = 10 копий весов,
    скрытый слой 125), его 10 000 эпох упираются в запуск ядер, а не в арифметику;
  * render_views — V видов: автокодировщики (сложенные или по одному кодом статьи, sequential) и поверхность
    лоссов PlotLossSurface статьи на каждом; карта (3, 26, 26) — те же каналы log1p, что у среды; рядом —
    скрытые координаты сохранённых копий весов (где траектория лежит на карте) и расхождение лосса
    восстановленных и настоящих копий (насколько карта верна вблизи траектории);
  * flat_raw — плоская карта: все сохранённые копии весов совпали (L-BFGS встал по допуску, PSO с нулевым
    шагом), нормировка весов даёт нулевой разброс, декодер выдаёт одну и ту же сеть во всех узлах сетки. Виды
    такой карты одинаковы, их не строим. В собранных буферах цели (tbuf_*) таких карт 55% у pb2d и 83% у ns2d
    (у ns2d — все карты после второго действия);
  * файл видов (save_record / load_records) и строки для зонда CV (view_rows): у каждого состояния карта
    среды (вид 0) и V видов; канал delta строится по предыдущему шагу того же вида, как в среде;
  * procedural_records — процедурные «ландшафты» без УрЧП (гауссовы случайные поля со степенным спектром,
    квадратичная чаша или седло, рост к краю окна, плоские карты застоя) в том же формате: предобучение на нуле
    буферов (Baradad et al., NeurIPS 2021: представления по процедурному шуму).

Формат файла видов (torch.save словаря):
    kind="map_views", format=1, pde, tag, source (откуда цепочка), seed, env {keep_opt, mode, budget},
    chain [[opt, lr, epochs], ...] исполненные действия, steps [s, ...] — после скольких действий построена карта,
    main (len(steps), 3, 26, 26) карта среды, views {s: (k, 3, 26, 26)}, info {s: {flat, k, t_ae, t_srf,
    traj (k, n, 2), rec_gap (k,), best_epoch (k,)}}, err {s: rmse+brmse}, l2re {s: l2re}, hist, t {...}.
Ключ состояния для удаления повторов — сид и исполненное начало цепочки (state_key): у цепочек с общим префиксом
(collect_active.py --share-prefix) и при повторном прогоне одной цепочки состояния префикса совпадают.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np

FORMAT = 1
VIEWS_HF = "rl_arch/views"          # корень файлов видов в датасете результатов HF
LOSS_KEYS = ("loss_total", "loss_oper", "loss_bnd")
FLAT_TOL = 1e-7


# --------------------------------------------------------------------------------------------
# Автокодировщики видов
# --------------------------------------------------------------------------------------------

def _trajectory_data(solver_models, batch_size, device):
    """Нормированные плоские веса копий (как VisualizationModel: get_trajectory_dataloader) — (n, D) и
    нормировка. Тот же загрузчик, что у статьи, поэтому нормировка и порядок параметров те же."""
    import torch
    from landscape_visualization._aux.trajectories_data import get_trajectory_dataloader
    sds = [m.state_dict() for m in solver_models]
    dl, transform = get_trajectory_dataloader(batch_size, models=sds, device=device)
    X = torch.stack([dl.dataset[i] for i in range(len(dl.dataset))]).to(device).float()
    return X, transform


def train_view_aes(solver_models, n_views, epochs, lr, cosine_patience, batch_size=32, es_patience=4000,
                   device="cpu", num_of_layers=3, layers_AE=(991, 125, 15), rec_weight=10000.0, every=100,
                   seed=None):
    """n_views автокодировщиков статьи одним проходом (параметры сложены по оси вида). Семантика каждого вида —
    VisualizationModel.train с solver_models: инициализация UniformAutoencoder (вид v создаётся v-м по счёту из
    общего генератора torch — при одном виде и том же сиде совпадает с автокодировщиком статьи), RMSprop(lr),
    CosineAnnealingWarmRestarts(cosine_patience) с шагом по (эпоха + пачка / пачек), лосс — сумма квадратов
    восстановления * rec_weight, на эпохах, кратных every, — сравнение среднего по пачкам лосса с лучшим:
    лучше — копия весов, иначе счётчик терпения; вид, исчерпавший терпение, дальше не сравнивается (его
    итог — лучшая копия, как у статьи после останова). Возвращает (список UniformAutoencoder в режиме eval,
    info: seconds, epochs, best_epoch, best_loss)."""
    import torch
    from torch.func import functional_call, stack_module_state, vmap
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from landscape_visualization._aux.AEmodel import UniformAutoencoder

    t0 = time.time()
    X, _ = _trajectory_data(solver_models, batch_size, device)
    n, dim = X.shape
    if seed is not None:
        torch.manual_seed(int(seed))
    models = [UniformAutoencoder(dim, num_of_layers, 2, h=list(layers_AE)).to(device) for _ in range(n_views)]
    stacked, _ = stack_module_state(models)
    params = {k: v.detach().clone().requires_grad_(True) for k, v in stacked.items()}
    base = copy.deepcopy(models[0]).to("meta")

    def one(p, x):
        return functional_call(base, p, (x,))

    fwd = vmap(one, in_dims=(0, None))
    opt = torch.optim.RMSprop(list(params.values()), lr=lr)
    sched = CosineAnnealingWarmRestarts(opt, cosine_patience)
    nb = max(1, math.ceil(n / batch_size))
    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
    best = [math.inf] * n_views
    best_ep = [-1] * n_views
    best_p = [None] * n_views
    wait = [0] * n_views
    live = [True] * n_views
    ep_run = 0
    for ep in range(epochs):
        if not any(live):
            break
        order = torch.randperm(n, generator=gen, device=X.device) if nb > 1 else None
        tot = torch.zeros(n_views, device=X.device)
        for b in range(nb):
            xb = X if order is None else X[order[b * batch_size:(b + 1) * batch_size]]
            opt.zero_grad()
            rec, _z = fwd(params, xb)
            per = ((rec.float() - xb) ** 2).sum(dim=(1, 2)) * float(rec_weight)
            per.sum().backward()
            opt.step()
            sched.step(ep + b / nb)
            tot = tot + per.detach()
        ep_run = ep + 1
        if ep % every == 0:
            cur = (tot / nb).cpu().tolist()
            for v in range(n_views):
                if not live[v]:
                    continue
                if cur[v] < best[v]:
                    best[v], best_ep[v], wait[v] = cur[v], ep, 0
                    best_p[v] = {k: p[v].detach().clone() for k, p in params.items()}
                else:
                    wait[v] += 1
                    if wait[v] >= es_patience:
                        live[v] = False
    out = []
    for v in range(n_views):
        m = models[v]
        if best_p[v] is not None:
            m.load_state_dict(best_p[v])
        else:                                   # эпох меньше одной проверки: как у статьи — последние веса
            m.load_state_dict({k: p[v].detach().clone() for k, p in params.items()})
        m.eval()
        out.append(m)
    return out, dict(seconds=round(time.time() - t0, 2), epochs=ep_run, best_epoch=best_ep,
                     best_loss=[None if not math.isfinite(b) else float(b) for b in best])


def flat_raw(raw, tol=FLAT_TOL):
    """Плоская ли карта среды (raw: {канал: тензор или массив}) — нулевой разброс общего лосса по сетке."""
    x = np.asarray(raw["loss_total"].detach().cpu() if hasattr(raw["loss_total"], "detach") else raw["loss_total"],
                   dtype=np.float64)
    return bool(np.isfinite(x).all() and float(x.std()) < tol)


def raw_to_maps(raw):
    """{канал: (26, 26)} -> (3, 26, 26) float32 в порядке LOSS_KEYS (как build_state без канала delta)."""
    return np.stack([np.asarray(raw[k].detach().float().cpu() if hasattr(raw[k], "detach") else raw[k],
                                dtype=np.float32) for k in LOSS_KEYS])


def render_views(solver_models, get_model_rec, n_views, x_range, ae_epochs, ae_lr, ae_cosine_patience, ae_batch,
                 ae_es_patience, ae_params, device, sequential=False, vm=None, early_stopping=None):
    """n_views видов одного снимка траектории. Автокодировщики — train_view_aes (sequential=False) или по одному
    VisualizationModel.train статьи (sequential=True, нужен vm и класс early_stopping); поверхность —
    PlotLossSurface статьи на каждом. Случайность — общий генератор torch (вызывающий сохраняет и
    восстанавливает его состояние). Возвращает (список карт (3, 26, 26), info)."""
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    t0 = time.time()
    if sequential:
        aes, best_ep = [], []
        for _ in range(n_views):
            aes.append(vm.train(ae_lr, ae_cosine_patience, ae_epochs, 100, ae_batch, True, finetune_AE_model=False,
                                callbacks=[early_stopping(patience=ae_es_patience)], solver_models=solver_models))
            best_ep.append(None)
        ae_info = dict(seconds=round(time.time() - t0, 2), epochs=ae_epochs, best_epoch=best_ep, best_loss=None)
    else:
        aes, ae_info = train_view_aes(solver_models, n_views, ae_epochs, ae_lr, ae_cosine_patience,
                                      batch_size=ae_batch, es_patience=ae_es_patience, device=device,
                                      num_of_layers=ae_params["num_of_layers"], layers_AE=ae_params["layers_AE"],
                                      rec_weight=ae_params["rec_weight"])
    t_ae = time.time() - t0
    maps, traj, gap = [], [], []
    t1 = time.time()
    for ae in aes:
        pls = PlotLossSurface(solver_models=solver_models, AE_model=ae, dde_pde_model=get_model_rec, x_range=x_range,
                              batch_size=ae_batch, loss_types=list(LOSS_KEYS), loss_name="loss_total",
                              path_to_plot_model=None, path_to_trajectories=None, img_dir="")
        raw = pls.save_equation_loss_surface(log_key=True)
        maps.append(raw_to_maps(raw))
        st = (getattr(pls, "states_dict", None) or {}).get("loss_total") or {}
        tc = st.get("trajectory_coordinates")
        traj.append(None if tc is None else np.asarray(tc.detach().cpu() if hasattr(tc, "detach") else tc,
                                                       dtype=np.float32))
        tl, ol = st.get("trajectory_losses"), st.get("original_trajectory_losses")
        if tl is not None and ol is not None:
            tl = np.asarray(tl.detach().cpu() if hasattr(tl, "detach") else tl, dtype=np.float64)
            ol = np.asarray(ol.detach().cpu() if hasattr(ol, "detach") else ol, dtype=np.float64)
            gap.append(float(np.mean(np.abs(np.log1p(np.abs(tl)) - np.log1p(np.abs(ol))))))
        else:
            gap.append(None)
        del pls
    info = dict(k=len(maps), t_ae=round(t_ae, 2), t_srf=round(time.time() - t1, 2),
                best_epoch=ae_info.get("best_epoch"), traj=traj, rec_gap=gap, sequential=bool(sequential))
    return maps, info


# --------------------------------------------------------------------------------------------
# Файл видов
# --------------------------------------------------------------------------------------------

def a2s(c):
    return "%s:%g:%d" % (c[0], float(c[1]), int(c[2]))


def state_key(seed, chain, s):
    """Состояние после s действий цепочки chain на сиде seed."""
    return f"{int(seed)}|" + ",".join(a2s(c) for c in chain[:s])


def make_record(pde, tag, source, seed, env, chain, main, views, info, err=None, l2re=None, hist=None, t=None,
                complete=True):
    """main {s: (3, 26, 26)}, views {s: [(3, 26, 26), ...]} -> словарь файла видов (см. описание модуля)."""
    steps = sorted(int(s) for s in main)
    return dict(kind="map_views", format=FORMAT, pde=pde, tag=tag, source=source, seed=int(seed), env=dict(env or {}),
                chain=[[c[0], float(c[1]), int(c[2])] for c in chain], steps=steps,
                main=(np.stack([np.asarray(main[s], dtype=np.float32) for s in steps]) if steps
                      else np.zeros((0, 3, 26, 26), np.float32)),
                views={int(s): np.stack([np.asarray(m, dtype=np.float32) for m in v]) for s, v in views.items() if len(v)},
                info={int(s): dict(i) for s, i in (info or {}).items()}, err={int(k): v for k, v in (err or {}).items()},
                l2re={int(k): v for k, v in (l2re or {}).items()}, hist=list(hist or []), t=dict(t or {}),
                complete=bool(complete))


def save_record(path, rec):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(rec, path)
    return path


def n_views_of(rec):
    return {int(s): int(len(v)) for s, v in (rec.get("views") or {}).items()}


def load_records(spec, subdir, log=print):
    """Файлы видов по спецификации (через запятую: префиксы в датасете результатов HF, например
    rl_arch/views/<тег>, или локальные папки; внутри — <папка задачи>/*.pt), отсортированы по имени."""
    import glob
    import torch
    out = []
    for item in (x.strip().rstrip("/") for x in spec.split(",") if x.strip()):
        if os.path.isdir(item):
            d = os.path.join(item, subdir) if os.path.isdir(os.path.join(item, subdir)) else item
        else:
            try:
                from huggingface_hub import snapshot_download
                root = snapshot_download("danil-e/pinnacle-optuna-db", repo_type="dataset",
                                         allow_patterns=[f"{item}/{subdir}/*.pt"])
            except Exception as e:      # noqa: BLE001
                raise SystemExit(f"виды {item}: локальной папки нет, из HF не скачано ({type(e).__name__}: {str(e)[:120]})")
            d = os.path.join(root, item, subdir)
        files = sorted(glob.glob(os.path.join(d, "*.pt")))
        if not files:
            raise SystemExit(f"виды {item}: в {d} нет файлов .pt (папка задачи — {subdir})")
        got = 0
        for p in files:
            rec = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(rec, dict) or rec.get("kind") != "map_views":
                log(f"  {os.path.basename(p)}: не файл видов — пропущен")
                continue
            rec["_file"] = os.path.basename(p)
            out.append(rec)
            got += 1
        log(f"  виды {item}: файлов {got}")
    return out


def _delta(tot, prev_tot):
    """Канал delta как у среды (online_eval_env.build_state = offline_rl._delta_like_env)."""
    if prev_tot is None:
        return np.zeros_like(tot, dtype=np.float32)
    d = tot - prev_tot
    dl = np.sign(d) * np.log1p(np.abs(d))
    dl = dl / (np.abs(dl).max() + 1e-6)
    return np.clip(dl, -1.0, 1.0).astype(np.float32)


def view_rows(records, max_views=0, dedupe=True):
    """Строки для зонда CV из файлов видов. Цепочка строк — (файл, вид): нулевая карта на шаге 0 (как старт
    цепочки у загрузчика буферов), затем карты этого вида по порядку шагов. Вид 0 — карта среды (есть на каждом
    шаге), виды 1..k — построенные рядом (только у неплоских состояний). Канал delta: шаг 1 — нули (как у среды
    после первого действия), дальше — разность с картой того же вида на предыдущем шаге, а если её нет — с картой
    среды на предыдущем шаге (у среды соседние карты тоже от разных автокодировщиков). max_views > 0 — только
    виды 0..max_views-1 (1 — одна карта среды). dedupe — состояние с уже встреченным ключом (сид + начало
    цепочки) пропускается во всех видах. Возвращает dict: maps (N, 4, 26, 26), CH, STEP, VKEY (номер состояния,
    общий у его видов; −1 у стартовых нулевых строк), VIEW, FILE, A (следующее действие или −1), ERR, n_states."""
    import target_buffer as TB
    seen, keys = set(), {}
    maps, ch, step, vkey, view, fil, act, err = ([] for _ in range(8))
    cid = 0
    for fi, rec in enumerate(records):
        steps = [int(s) for s in rec["steps"]]
        main = {s: rec["main"][i] for i, s in enumerate(steps)}
        views = {int(s): v for s, v in (rec.get("views") or {}).items()}
        chain = rec["chain"]
        own = []
        for s in steps:
            if s < 1:
                continue
            k = state_key(rec["seed"], chain, s)
            if dedupe and k in seen:
                continue
            own.append((s, k))
        for s, k in own:
            seen.add(k)
        if not own:
            continue
        nv = 1 + max((len(views.get(s, ())) for s, _ in own), default=0)
        if max_views:
            nv = min(nv, int(max_views))
        for v in range(nv):
            rows = []
            for s, k in own:
                if v == 0:
                    m = main[s]
                elif s in views and v - 1 < len(views[s]):
                    m = views[s][v - 1]
                else:
                    continue
                if s - 1 >= 1 and (s - 1) in main:
                    pv = views.get(s - 1)
                    prev = pv[v - 1] if (v > 0 and pv is not None and v - 1 < len(pv)) else main[s - 1]
                    d = _delta(m[0], prev[0])
                else:
                    d = _delta(m[0], None)
                a = -1
                if s < len(chain):
                    a = TB.action_index(chain[s][0], chain[s][1], chain[s][2])
                    a = -1 if a is None else int(a)
                rows.append((np.concatenate([m, d[None]], 0), s, k, a, (rec.get("err") or {}).get(s)))
            if not rows:
                continue
            maps.append(np.zeros((4,) + rows[0][0].shape[1:], np.float32))
            ch.append(cid); step.append(0); vkey.append(-1); view.append(v); fil.append(fi); act.append(-1)
            err.append(None)
            for j, (m, s, k, a, e) in enumerate(rows, 1):
                maps.append(m.astype(np.float32))
                ch.append(cid); step.append(j); vkey.append(keys.setdefault(k, len(keys))); view.append(v)
                fil.append(fi); act.append(a); err.append(e)
            cid += 1
    if not maps:
        raise SystemExit("файлы видов: ни одного состояния (все повторы или пустые файлы)")
    return dict(maps=np.stack(maps).astype(np.float32), CH=np.asarray(ch, np.int64), STEP=np.asarray(step, np.int64),
                VKEY=np.asarray(vkey, np.int64), VIEW=np.asarray(view, np.int64), FILE=np.asarray(fil, np.int64),
                A=np.asarray(act, np.int64), ERR=err, n_states=len(keys))


# --------------------------------------------------------------------------------------------
# Процедурные ландшафты (ноль буферов)
# --------------------------------------------------------------------------------------------

def _grf(rng, size, beta):
    """Гауссово случайное поле со спектром мощности ~ k^-beta, нулевое среднее, единичный разброс."""
    kx = np.fft.fftfreq(size)[:, None]
    ky = np.fft.fftfreq(size)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    f = np.real(np.fft.ifft2((rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) * amp))
    return (f - f.mean()) / (f.std() + 1e-12)


def procedural_chain(rng, size=26, chain_len=8, half=1.25):
    """Одна цепочка процедурных карт (3, size, size) — log1p(общий, оператор, граница), как каналы среды.
    Лосс в узле u скрытой плоскости: exp(уровень + чаша или седло вокруг текущей точки c + рост к краю окна
    (декодер вне облака копий весов выдаёт негодную сеть) + гладкое случайное поле); граничная часть — свой
    уровень, своя доля чаши и своё поле; общий = оператор + граница. По шагам цепочки уровень падает, поле и
    геометрия меняются частично; с ростом шага чаще плоская карта (застой: все копии весов совпали) с
    уровнем в текущей точке, застой обычно держится."""
    g = np.linspace(-half, half, size)
    U, V = np.meshgrid(g, g, indexing="ij")
    c = np.clip(rng.normal(0.0, 0.35, 2), -0.8, 0.8)
    lvl_op = rng.uniform(-3.0, 2.0) * math.log(10.0)
    rho = rng.uniform(-2.0, 1.0) * math.log(10.0)
    f_op, f_bn = _grf(rng, size, rng.uniform(1.5, 4.0)), _grf(rng, size, rng.uniform(1.5, 4.0))
    flat = False
    out = []
    for t in range(chain_len):
        th = rng.uniform(0, np.pi)
        l1 = math.exp(rng.uniform(math.log(0.1), math.log(10.0)))
        l2 = l1 * math.exp(rng.uniform(math.log(0.01), 0.0)) * (-1.0 if rng.random() < 0.2 else 1.0)
        a = (U - c[0]) * math.cos(th) + (V - c[1]) * math.sin(th)
        b = -(U - c[0]) * math.sin(th) + (V - c[1]) * math.cos(th)
        q = 0.5 * (l1 * a ** 2 + l2 * b ** 2)
        r = rng.uniform(0.6, 1.2)
        edge = math.exp(rng.uniform(math.log(0.5), math.log(20.0))) * np.maximum(np.maximum(abs(U), abs(V)) - r, 0.0) ** 2
        s_op, s_bn, kap = rng.uniform(0.0, 2.0), rng.uniform(0.0, 2.0), rng.uniform(0.0, 1.0)
        lo = np.clip(lvl_op + q + edge + s_op * f_op, -30.0, 12.0)
        lb = np.clip(lvl_op + rho + kap * (q + edge) + s_bn * f_bn, -30.0, 12.0)
        op, bn = np.exp(lo), np.exp(lb)
        m = np.stack([np.log1p(op + bn), np.log1p(op), np.log1p(bn)]).astype(np.float32)
        p_flat = min(0.6, 0.12 * t)
        flat = (flat and rng.random() < 0.8) or (t > 0 and rng.random() < p_flat)
        if flat:
            i, j = int(np.argmin(np.abs(g - c[0]))), int(np.argmin(np.abs(g - c[1])))
            m = np.broadcast_to(m[:, i, j][:, None, None], m.shape).astype(np.float32).copy()
        out.append(m)
        # следующий шаг: уровень ниже, поле и точка частично новые
        lvl_op -= rng.uniform(0.0, 1.0) * math.log(10.0)
        keep = rng.uniform(0.3, 0.9)
        f_op = keep * f_op + math.sqrt(1 - keep ** 2) * _grf(rng, size, rng.uniform(1.5, 4.0))
        f_bn = keep * f_bn + math.sqrt(1 - keep ** 2) * _grf(rng, size, rng.uniform(1.5, 4.0))
        c = np.clip(c + rng.normal(0.0, 0.1, 2), -0.8, 0.8)
    return out


def procedural_records(n_chains, seed=0, size=26, chain_len=8):
    """n_chains процедурных цепочек в формате файла видов (одна карта среды на шаг, видов нет): строки
    для зонда CV — тем же view_rows. Действия цепочки — фиктивные (номер цепочки в ключе), сид — номер."""
    rng = np.random.default_rng([int(seed), 20261016])
    recs = []
    for i in range(int(n_chains)):
        ms = procedural_chain(rng, size, chain_len)
        chain = [["Adam", 1e-3, 100 + i * chain_len + j] for j in range(len(ms))]
        recs.append(make_record("procedural", "procedural", f"procedural#{i}", i, {}, chain,
                                {j + 1: m for j, m in enumerate(ms)}, {}, {}))
    return recs
