#!/usr/bin/env python
"""
Проверки щита оценки (online_eval_env.py --shield; mb_targets.live_state, shield, shield_step,
load_shield) и дистилляции учителя на скалярах (offline_rl.py --teacher-from, --teacher-w;
teacher_scalar_q, teacher_targets) без GPU, без сети и без обучения PINN, на крошечных
синтетических буферах двух задач (по два файла, цепочки по 2-4 шага):

  * вход ансамбля суррогата одной и той же строки буфера совпадает бит в бит на трёх путях:
    таблица (на ней ансамбль учится: agent_combo.fit_features), массивы тренера (модельные цели:
    chain_states + step_features) и живой прогон (щит: live_state + step_features), в том числе у
    новых цепочек с обрезанным бюджетом последним действием (agent_combo.extra_rows);
  * правило вето: знак (улучшение — отрицательное изменение log10 ошибки), пессимизм, маска,
    «не меньше K действий», обрезка действия остатком бюджета; журнал и сводка в цикле решений со
    стражем и действием вне 27; настоящий ансамбль HGB из кэша через load_shield (файл, папка, auto);
  * учитель: настоящий чекпоинт agent_combo.run (несколько шагов обучения); Q учителя на строках
    буфера, посчитанные тренером ученика, совпадают с Q того же чекпоинта на состоянии, которое
    собирает оценка (online_eval_env.add_scalar_ctx + описатель) из строк ТАБЛИЦЫ учителя — и у
    одного УрЧП, и на смеси с нормировкой ошибки на E0; перевод единиц dlog -> delta / dlog, каркас
    учителя, отказы; добавка к TD в train_variant и полный прогон main() по боевому рецепту;
  * онлайновое дообучение (online_aux.py): цепочка живого прогона строками набора тренера даёт то
    же состояние суррогата, что строка буфера и live_state, а цели онлайнового перехода (модельные
    и учителя) равны целям строки буфера с той же цепочкой; слагаемое лосса — как в train_variant.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_shield_teacher.py
"""
import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("HF_TOKEN", None)            # тест ничего не заливает
os.environ.pop("HF_TOKEN_WRITE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import mb_targets as MB  # noqa: E402
import agent_combo as AC  # noqa: E402
import online_aux as OA  # noqa: E402
import test_map_agent as T  # noqa: E402

torch.set_num_threads(1)
QUIET = lambda *a: None  # noqa: E731
CPU = torch.device("cpu")
PDE, SUB = "poissonboltzmann2d", "poisson_boltzmann_2d"
PDE2, SUB2 = "ns2d_liddriven", "ns2d_liddriven"
KEYS = ["loss_total", "loss_oper", "loss_bnd"]
SD3 = float(np.std([0.0, 0.02, 0.04]))


def make_dump(rng, e0, lengths):
    """Файл в формате дампа: цепочки заданной длины, расход каждой меньше 7000 эпох."""
    out = []
    for n in lengths:
        while True:
            chain = [int(a) for a in rng.integers(0, 27, n)]
            if S.EPOCHS[chain].sum() < 7000:
                break
        maps = [{k: np.zeros((26, 26), np.float32) for k in KEYS}]
        l, errs = np.log10(e0), []
        for _a in chain:
            l += rng.normal(-0.15, 0.1)
            errs.append(10 ** l)
            maps.append({k: rng.random((26, 26)).astype(np.float32) + 1.0 for k in KEYS})
        for j, a in enumerate(chain):
            done = -1 if j == n - 1 else 0
            out.append(dict(state=dict(maps[j]), next_state=dict(maps[j + 1]),
                            action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                            reward=-(errs[j] + (1.0 if done == -1 else 0.0)), done=done))
    return out


def make_buffers(seed=0):
    rng = np.random.default_rng(seed)
    meta = O.pde_meta()
    return {sd: [make_dump(rng, float(meta[nm]["init_err"]), ln) for ln in ((3, 2, 4), (4, 3))]
            for sd, nm in ((SUB2, PDE2), (SUB, PDE))}


def chain_of(data, i):
    """Исполненные до строки i действия её цепочки в записи online_eval_env и ошибка после них."""
    s = i - int(data["STEP"][i])
    chain = [[S.ACTIONS[a][0], S.ACTIONS[a][1], int(S.EPOCHS[a])] for a in data["A"][s:i]]
    return chain, (float(data["ERR"][i - 1]) if i > s else float("inf"))


class LinEns:
    """Подставной ансамбль: изменение log10 ошибки задано функцией от признаков, член m смещён на
    0.02*m (среднее по членам f + 0.02, разброс SD3)."""
    members = 3

    def __init__(self, f):
        self.f = f

    def predict_ln(self, X, l, member=None):
        out = np.stack([np.clip(l + self.f(X) + 0.02 * m, S.L_LO, S.L_HI) for m in range(self.members)])
        return out if member is None else out[member]


def must_fail(fn, exc, what):
    try:
        fn()
    except exc:
        return
    raise AssertionError(what)


# --------------------------------------------------------------------------------------------

def test_features(bufs, tmp):
    """Одна строка буфера — один и тот же вход ансамбля на пути обучения и на пути оценки."""
    tab = AC.with_epa(S.build_table(episodes=bufs, verbose=False))      # две задачи: pb2d в таблице вторая
    pi = tab["NAMES"].index(PDE)
    rows = np.where(tab["PDE"] == pi)[0]
    X_fit = AC.fit_features(tab, rows)                                  # на этом ансамбль учится
    data = O.episodes_to_arrays_chains(bufs[SUB], reward_form="delta", verbose=False)
    n = len(data["A"])
    assert n == len(rows) == 16 and (tab["A"][rows] == data["A"]).all() and (tab["STEP"][rows] == data["STEP"]).all()
    assert np.allclose(tab["LN"][rows], np.log10(data["ERR"].astype(np.float64)))
    # путь тренера (модельные цели)
    st, valid, _ = MB.chain_states(data, tab, np.array([pi]), np.ones(1))
    X_tr, ep = MB.step_features(tab, st, data["A"])
    assert valid.all() and np.array_equal(X_tr, X_fit) and (ep == S.EPOCHS[data["A"]]).all()
    # путь оценки (щит): своя таблица из одной строки описателя, состояние из исполненной цепочки
    ens_file = os.path.join(tmp, "fake.pkl")
    with open(ens_file, "wb") as f:
        pickle.dump(LinEns(None), f)
    tab_l, _, _ = MB.load_shield(ens_file, PDE)
    assert np.array_equal(tab_l["CTX"][0], tab["CTX"][pi])
    for i in range(n):
        chain, err = chain_of(data, i)
        st_l = MB.live_state(tab_l, chain, err)
        left = 7000 - int(st_l["spent"][0])
        X_l, ep_l = MB.step_features(tab_l, st_l, data["A"][i:i + 1], budget=int(st_l["spent"][0]) + left)
        assert np.array_equal(X_l[0], X_fit[i]), (i, np.where(X_l[0] != X_fit[i])[0])
        assert int(st_l["step"][0]) == len(chain) and int(st_l["spent"][0]) == sum(c[2] for c in chain)
        # четвёртый путь — онлайновое дообучение (online_aux.live_data): цепочка живого прогона строками
        # набора тренера, состояние — тем же chain_states; совпадает и с live_state, и со строкой таблицы
        s0 = i - int(data["STEP"][i])
        ld = OA.live_data(data["A"][s0:i], S.EPOCHS[data["A"][s0:i]], data["ERR"][s0:i])
        st_o, ok_o, _ = MB.chain_states(ld, tab_l, np.array([0]), np.ones(1), budget=7000)
        st_o = {k: v[i - s0:i - s0 + 1] for k, v in st_o.items()}
        assert ok_o[i - s0] and all(np.array_equal(st_o[k], st_l[k]) for k in st_l), i
        assert np.array_equal(MB.step_features(tab_l, st_o, data["A"][i:i + 1], budget=7000)[0][0], X_fit[i]), i
    # новые цепочки (строки оценки / collect_active): последнее действие обрезано бюджетом 7000
    full = [["Adam", 0.01, 2500], ["Adam", 0.0001, 2500], ["LBFGS", 1.0, 1000], ["LBFGS", 0.5, 500],
            ["Adam", 0.001, 500]]
    spent = np.cumsum([c[2] for c in full]).tolist()
    e0 = float(10 ** tab["CTX"][pi, 6])
    errs = [e0 * 10 ** (-0.4 * (k + 1)) for k in range(len(full))]
    rec = dict(chain=full, hist=[[s, 1e-3, e * 2, 1.0] for s, e in zip(spent, errs)], err=errs, seed=1,
               env=dict(keep_opt=False, mode=None, budget=7000), pde=PDE)
    tab2, info = AC.extra_rows(tab, [rec], pi)
    new = np.arange(len(tab["A"]), len(tab2["A"]))
    assert info["rows"] == 5 and info["truncated"] == 1 and tab2["EPA"][new[-1]] == 500
    X_new = AC.fit_features(tab2, new)
    for k in range(len(full)):
        st_l = MB.live_state(tab_l, full[:k], errs[k - 1] if k else float("inf"))
        X_l, ep_l = MB.step_features(tab_l, st_l, tab2["A"][new[k]:new[k] + 1],
                                     budget=int(st_l["spent"][0]) + 7000 - (spent[k - 1] if k else 0))
        assert np.array_equal(X_l[0], X_new[k]), k
        # то же через строки набора (онлайновое дообучение): действие обрезается остатком бюджета 7000
        ld = OA.live_data(tab2["A"][new[:k]], [c[2] for c in full[:k]], errs[:k])
        st_o, ok_o, _ = MB.chain_states(ld, tab_l, np.array([0]), np.ones(1), budget=7000)
        X_o, ep_o = MB.step_features(tab_l, {key: v[k:k + 1] for key, v in st_o.items()},
                                     tab2["A"][new[k]:new[k] + 1], budget=7000)
        # ошибка в наборе тренера хранится в float32 (как в дампе буфера): log10 ошибки совпадает до 1e-7
        assert ok_o[k] and np.allclose(X_o[0], X_new[k], rtol=0, atol=1e-6) and int(ep_o[0]) == full[k][2], k
        assert all(np.array_equal(st_o[key][k:k + 1], st_l[key]) for key in st_l if key != "l"), k
    assert int(ep_l[0]) == 500 and S.EPOCHS[tab2["A"][new[-1]]] == 1000
    ld = OA.live_data(tab2["A"][new], [c[2] for c in full], errs)       # бюджет выбран: состояния для цели нет
    assert not MB.chain_states(ld, tab_l, np.array([0]), np.ones(1), budget=7000)[1][len(full)]
    # состояние после обрезанного действия: эпохи по семействам считаются фактические
    st_l = MB.live_state(tab_l, full, errs[-1])
    assert int(st_l["spent"][0]) == 7000 and st_l["spf"][0].tolist() == [5500.0, 1500.0, 0.0]
    assert MB.live_state(tab_l, full[:1] + [["SOAP", 0.003, 1000]], 0.1) is None      # действие вне 27
    assert MB.live_state(tab_l, full[:1] + [["BOOST", 0.01, 0]], 0.1) is None
    assert MB.live_state(tab_l, full[:1], float("nan")) is None
    # ошибка по самой модели (--shield-err model): открытый прогон среднего ансамбля от E0
    flat = LinEns(lambda X: np.full(len(X), -0.12))
    assert abs(np.log10(MB.model_err(flat, tab_l, full[:3])) - (np.log10(e0) + 3 * (-0.12 + 0.02))) < 1e-9
    assert MB.model_err(flat, tab_l, []) == e0 and MB.model_err(flat, tab_l, [["SOAP", 0.003, 100]]) is None
    seen = []
    MB.model_err(LinEns(lambda X: seen.append(X.copy()) or np.zeros(len(X))), tab_l, full)
    assert np.array_equal(np.concatenate(seen[::3])[:, [0, 6, 14, 15]], X_new[:, [0, 6, 14, 15]]), \
        "действие, фактические эпохи, шаг и расход — как у строк таблицы"
    print("щит: вход ансамбля на путях таблицы, тренера и живого прогона совпадает бит в бит: OK")
    return tab, tab_l, data


def test_shield_rule(tab_l):
    """Правило вето на подставных ансамблях с известным ответом."""
    full = np.ones(27, bool)
    st = MB.live_state(tab_l, [["Adam", 0.001, 1000]], 0.05)
    # улучшение (отрицательное изменение) растёт с номером действия: среднее 0.12 - 0.02a
    ens = LinEns(lambda X: 0.10 - 0.02 * X[:, 0])
    ok, score, mu, sd = MB.shield(ens, tab_l, st, full, lam=0.5, keep=3, left=6000)
    assert np.allclose(mu, 0.12 - 0.02 * np.arange(27)) and np.allclose(sd, SD3)
    assert np.allclose(score, mu + 0.5 * SD3) and ok.tolist() == [a >= 7 for a in range(27)]
    ok5, _, _, _ = MB.shield(ens, tab_l, st, full, lam=5.0, keep=3, left=6000)      # пессимизм строже
    assert ok5.tolist() == [a >= 11 for a in range(27)]
    mask = S.parse_mask("pso,adam:0.01")                                            # разрешены 3..17
    okm, _, _, _ = MB.shield(ens, tab_l, st, mask, lam=0.5, keep=3, left=6000)
    assert np.where(okm)[0].tolist() == list(range(7, 18)) and not (okm & ~mask).any()
    # ни одно действие не обещает улучшения: остаются K лучших по прогнозу среди разрешённых
    bad = LinEns(lambda X: 0.05 + 0.01 * X[:, 0])
    okb, sb, _, _ = MB.shield(bad, tab_l, st, full, lam=0.5, keep=3, left=6000)
    assert np.where(okb)[0].tolist() == [0, 1, 2] and (sb >= 0).all()
    okb, _, _, _ = MB.shield(bad, tab_l, st, mask, lam=0.5, keep=2, left=6000)
    assert np.where(okb)[0].tolist() == [3, 4]
    # улучшает одно действие: оно остаётся всегда, до K добираются лучшие из остальных
    one = LinEns(lambda X: np.where(X[:, 0] == 5, -0.5, 0.1))
    assert np.where(MB.shield(one, tab_l, st, full, 0.5, 3, 6000)[0])[0].tolist() == [0, 1, 5]
    assert np.where(MB.shield(one, tab_l, st, full, 0.5, 1, 6000)[0])[0].tolist() == [5]
    # обрезка остатком бюджета: «помогают только действия от 317 эпох» (признак 6 — log10 эпох)
    lng = LinEns(lambda X: np.where(X[:, 6] > 2.5, -0.3, 0.1))
    assert np.where(MB.shield(lng, tab_l, st, full, 0.5, 3, 6000)[0])[0].tolist() == [a for a in range(27)
                                                                                      if S.EPOCHS[a] >= 500]
    okl, sl, _, _ = MB.shield(lng, tab_l, st, full, 0.5, 3, 100)                     # осталось 100 эпох
    assert okl.sum() == 3 and (sl >= 0).all()
    print("щит: знак, пессимизм, маска, не меньше K действий, остаток бюджета: OK")


def test_shield_loop(tab, tab_l, data, tmp):
    """Цикл решений как в run_seed: выбор агента по своим Q среди оставленных, страж, сводка."""
    ens = LinEns(lambda X: 0.10 - 0.02 * X[:, 0])                # улучшение обещают действия 7..26
    allowed = S.parse_mask("pso")
    q = -np.arange(27, dtype=np.float64)                         # агент без щита всегда берёт действие 0
    chain, err, spent, log = [], float("inf"), 0, []
    for step in range(4):
        a_free = int(np.argmax(np.where(allowed, q, -np.inf)))
        ok, sh = MB.shield_step(ens, tab_l, chain, err, allowed, a_free, 0.5, 3, left=7000 - spent, spent=spent)
        a = int(np.argmax(np.where(ok, q, -np.inf)))
        log.append(MB.shield_chosen(sh, a))
        assert a == 7 and sh["free_vetoed"] and sh["vetoed"] == 7 and sh["n_allowed"] == 18 and not sh["forced"]
        assert sh["step"] == len(chain) + 1 and sh["spent"] == spent and abs(sh["pred"] - (0.12 - 0.14)) < 1e-4
        if step == 1:
            spent += 500            # страж откатил действие: эпохи потрачены, цепочка и состояние прежние
            continue
        chain.append([S.ACTIONS[a][0], S.ACTIONS[a][1], int(S.EPOCHS[a])])
        spent += int(S.EPOCHS[a])
        err = 0.1 / (step + 1)
    assert [x["step"] for x in log] == [1, 2, 2, 3] and [x["spent"] for x in log] == [0, 1000, 1500, 2500]
    q2 = np.arange(27, dtype=np.float64)                         # другой агент: его выбор (17) щит не трогает
    ok, sh = MB.shield_step(ens, tab_l, chain, err, allowed, 17, 0.5, 3, left=100, spent=6900)
    log.append(MB.shield_chosen(sh, int(np.argmax(np.where(ok, q2, -np.inf)))))
    assert not sh["free_vetoed"] and sh["a"] == 17
    ok, sh = MB.shield_step(ens, tab_l, chain + [["SOAP", 0.003, 100]], err, allowed, 17, 0.5, 3, left=100)
    log.append(MB.shield_chosen(sh, 17))
    assert sh["skip"] and (ok == allowed).all() and "a" not in sh
    summ = MB.shield_summary(log)
    assert summ == dict(shield_steps=5, shield_skips=1, shield_veto_rate=0.8, shield_vetoed_frac=round(7 / 18, 4),
                        shield_forced=0), summ
    assert MB.shield_summary([])["shield_veto_rate"] is None
    json.dumps(dict(shield_log=log, **summ))                     # строка результата сериализуется
    # настоящий ансамбль из кэша шага обучения: имя файла, папка, путь из meta чекпоинта
    work = os.path.join(tmp, "work")
    models = AC.Models(tab, work, members=2, quick=True, seed=0, log=QUIET)
    real = models.get("full")
    path = models.path("full")
    assert os.path.exists(path) and os.path.basename(path) == f"combo_surr_{models.sig}_full_m2_q_s0.pkl"
    for spec, meta in ((path, None), (work, None), ("auto", dict(mb=dict(surr_file=path)))):
        _, ens_l, p = MB.load_shield(spec, PDE, meta)
        assert p == path and ens_l.members == 2
    pi = tab["NAMES"].index(PDE)
    i = int(np.where((tab["PDE"] == pi) & (tab["STEP"] == 2))[0][0])
    j = i - int(np.where(tab["PDE"] == pi)[0][0])
    ch, e = chain_of(data, j)
    st = MB.live_state(tab_l, ch, e)
    _, _, mu, sd = MB.shield(ens_l, tab_l, st, np.ones(27, bool), 0.5, 3, left=7000 - int(st["spent"][0]))
    ref = real.predict_ln(AC.fit_features(tab, np.array([i])), tab["L"][[i]])[:, 0] - tab["L"][i]
    assert abs(mu[tab["A"][i]] - ref.mean()) < 1e-12 and abs(sd[tab["A"][i]] - ref.std()) < 1e-12
    with open(models.path("half0"), "wb") as f:
        pickle.dump(real, f)
    assert MB.load_shield(work, PDE)[2] == path, "половинные ансамбли в папке не мешают"
    with open(os.path.join(work, "combo_surr_1_x_full_m5_s0.pkl"), "wb") as f:
        pickle.dump(real, f)
    must_fail(lambda: MB.load_shield(work, PDE), SystemExit, "два полных ансамбля в папке — отказ")
    must_fail(lambda: MB.load_shield("auto", PDE, {}), SystemExit, "auto без пути в meta — отказ")
    must_fail(lambda: MB.load_shield(os.path.join(tmp, "нет.pkl"), PDE), SystemExit, "нет файла — отказ")
    must_fail(lambda: MB.load_shield(path, "нет_такой_задачи"), SystemExit, "неизвестная задача — отказ")
    print("щит: цикл решений, журнал и сводка, настоящий ансамбль из кэша (файл, папка, auto): OK")


# --------------------------------------------------------------------------------------------

def train_teacher(bufs, tmp, tag="teach", extra=()):
    """Настоящий чекпоинт учителя: agent_combo.run на синтетической таблице, несколько шагов."""
    table = os.path.join(tmp, "table.npz")
    if not os.path.exists(table):
        S.save_table(S.build_table(episodes=bufs, verbose=False), table)
    argv = ["--pde", PDE, "--table", table, "--mbpo-h", "2", "--mopo-lam", "0.5", "--mask", "long",
            "--steps", "12", "--batch", "16", "--n-branch", "20", "--regen", "6", "--tgt-every", "5",
            "--offline-eval", "none", "--surr-quick", "--members", "2", "--no-upload", "--tag", tag,
            "--work", os.path.join(tmp, "work"), "--out-dir", tmp] + list(extra)      # кэш ансамбля общий
    args = AC.build_parser().parse_args(argv)
    args._argv = argv
    with contextlib.redirect_stdout(io.StringIO()):
        rep = AC.run(args, log=QUIET)
    assert rep["verify"]["ok"], rep["verify"]       # Q обучавшейся сети = Q чекпоинта на состоянии оценки
    return rep["_paths"]["ckpt"], S.load_table(table)


def eval_path_q(ckpt, tab, rows):
    """Q чекпоинта на состоянии, которое собирает цикл оценки (как agent_combo.verify_ckpt), для
    строк таблицы учителя."""
    import online_eval_env as E
    torch.set_default_tensor_type(torch.FloatTensor)
    net, mean, std, variant = E.load_agent(ckpt)
    meta = net._meta
    out = []
    for i in rows:
        name = tab["NAMES"][tab["PDE"][i]]
        e0 = float(O.pde_meta()[name]["init_err"])
        pa = int(tab["PA"][i])
        st = E.add_scalar_ctx(np.zeros((4, 26, 26), np.float32), int(tab["STEP"][i]), meta["ctx_kmax"],
                              int(tab["SPENT"][i]), meta["ctx_budget"], None if pa < 0 else pa,
                              1.0 if tab["STEP"][i] == 0 else 10.0 ** tab["L"][i] / e0)
        st = np.concatenate([st, np.stack([np.full(st.shape[1:], v, dtype=np.float32) for v in O.pde_desc(name)])])
        out.append(E.q_values(net, st, mean, std, variant))
    return np.stack(out), meta


def test_teacher(bufs, tmp):
    ckpt, tab = train_teacher(bufs, tmp)
    e0 = {n: float(O.pde_meta()[n]["init_err"]) for n in (PDE, PDE2)}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # --- один УрЧП, ошибки как есть (боевой рецепт) ---
    data = O.episodes_to_arrays_chains(bufs[SUB], reward_form="delta", verbose=False, init_err=e0[PDE], budget=7000)
    n = len(data["A"])
    rows = np.where(tab["PDE"] == tab["NAMES"].index(PDE))[0]
    assert n == len(rows) and (tab["A"][rows] == data["A"]).all() and (tab["SPENT"][rows] == data["SPENT"]).all()
    q_ref, meta = eval_path_q(ckpt, tab, rows)
    assert meta["state_mode"] == "loss" and meta["ctx_err"] == "err" and meta["guide_epochs"] == 0
    q, valid, first, l = O.teacher_scalar_q(ck, data, CPU, [PDE], np.ones(1))
    assert valid.all() and (first == (data["STEP"] == 0)).all() and np.allclose(l, tab["L"][rows])
    assert np.abs(q - q_ref).max() < 1e-4, np.abs(q - q_ref).max()
    assert np.abs(q).max() > 1e-3 and q.std(1).min() > 0, "Q учителя не вырождены"
    # --- смесь двух УрЧП, ошибки в долях E0 (load_multi) ---
    for sd in (SUB2, SUB):
        os.makedirs(os.path.join(tmp, "bufs", sd))
        for k, ep in enumerate(bufs[sd]):
            torch.save(ep, os.path.join(tmp, "bufs", sd, f"f{k}.pt"))
    multi = O.load_multi([SUB2, SUB], reward_form="dlog", err_norm="init", budget=7000,
                         data_dir=os.path.join(tmp, "bufs"), verbose=False)
    names = multi["PDE_NAMES"]
    assert names == tab["NAMES"] and (multi["A"] == tab["A"]).all() and (multi["PDE"] == tab["PDE"]).all()
    scale = np.array([e0[x] for x in names])
    q_all, meta = eval_path_q(ckpt, tab, np.arange(len(tab["A"])))
    qm, vm, fm, lm = O.teacher_scalar_q(ck, multi, CPU, names, scale)
    assert vm.all() and np.abs(qm - q_all).max() < 1e-4, np.abs(qm - q_all).max()
    # --- единицы ---
    prev = np.r_[1.0, data["ERR"][:-1].astype(np.float64)]
    QTE, TEV, info = O.teacher_targets(ckpt, data, [PDE], "delta", np.ones(1), np.array([e0[PDE]]), 0.99, CPU, log=QUIET)
    assert QTE.shape == (n, 27) and QTE.dtype == np.float32 and TEV.all() and info["units"] == "dlog->delta"
    assert info["mask"] == [int(i) for i in np.where(~AC.make_allowed("long"))[0]] and info["gamma"] == 1.0
    nf = ~first
    assert np.allclose(QTE[nf], prev[nf, None] * (1.0 - 10.0 ** (-q[nf].astype(np.float64))), atol=1e-5)
    assert np.allclose(QTE[first], e0[PDE] * (1.0 - 10.0 ** (-q[first].astype(np.float64))), atol=1e-5)
    assert (QTE <= np.where(first, e0[PDE], prev)[:, None] + 1e-6).all(), "граница ценности: Q не больше текущей ошибки"
    Qd, _, i2 = O.teacher_targets(ckpt, data, [PDE], "dlog", np.ones(1), np.array([2.0 * e0[PDE]]), 1.0, CPU, log=QUIET)
    assert np.allclose(Qd[nf], q[nf], atol=1e-5) and np.allclose(Qd[first], q[first] + np.log10(2.0), atol=1e-5)
    # смесь: награда тренера в долях E0, первый шаг от 1.0 — те же формулы в его единицах
    Qm, Vm, _ = O.teacher_targets(ckpt, multi, names, "delta", scale, np.ones(2), 0.99, CPU, log=QUIET)
    pm = np.where(fm, 1.0, np.r_[1.0, multi["ERR"][:-1].astype(np.float64)])
    assert Vm.all() and np.allclose(Qm, pm[:, None] * (1.0 - 10.0 ** (-qm.astype(np.float64))), atol=1e-5)
    # расходящееся действие: Q учителя -5 декад обрезается до TEACHER_QMIN, цель -9 текущих ошибок
    ck2 = dict(ck, state_dict=dict(ck["state_dict"]))
    ck2["state_dict"]["1.bias"] = ck["state_dict"]["1.bias"].clone()
    ck2["state_dict"]["1.bias"][3] -= 8.0
    low = os.path.join(tmp, "teach_low.pt")
    torch.save(ck2, low)
    Ql, _, _ = O.teacher_targets(low, data, [PDE], "delta", np.ones(1), np.array([e0[PDE]]), 0.99, CPU, log=QUIET)
    assert O.TEACHER_QMIN == -1.0 and np.allclose(Ql[nf, 3], -9.0 * prev[nf], rtol=1e-5)
    assert np.allclose(np.delete(Ql, 3, 1), np.delete(QTE, 3, 1))
    # каркас учителя: он решает только после каркаса — строки до него без цели
    ck3 = dict(ck, meta=dict(ck["meta"], guide_epochs=1000, train_pdes=[PDE]))
    gfile = os.path.join(tmp, "teach_guide.pt")
    torch.save(ck3, gfile)
    _, Vg, ig = O.teacher_targets(gfile, data, [PDE], "delta", np.ones(1), np.array([e0[PDE]]), 0.99, CPU, log=QUIET)
    assert (Vg == (data["SPENT"] >= 1000)).all() and ig["rows"] == int(Vg.sum()) and ig["guide_epochs"] == 1000
    _, Vp, _ = O.teacher_targets(gfile, multi, names, "delta", scale, np.ones(2), 0.99, CPU, log=QUIET)
    assert (Vp == ((multi["SPENT"] >= 1000) & (multi["PDE"] == names.index(PDE)))).all(), "чужая задача без цели"
    # цепочка без начала: прошлая ошибка неизвестна
    cut = {k: (v[1:] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    _, Vc, _ = O.teacher_targets(ckpt, cut, [PDE], "delta", np.ones(1), np.array([e0[PDE]]), 0.99, CPU, log=QUIET)
    assert not Vc[:2].any() and Vc[2:].all()
    # отказы с понятным текстом
    args = (data, [PDE])
    must_fail(lambda: O.teacher_targets(ckpt, *args, "level", np.ones(1), np.array([1.0]), 0.99, CPU, log=QUIET),
              ValueError, "награда-уровень не переводится")
    must_fail(lambda: O.teacher_targets(ckpt, *args, "delta", np.ones(1), None, 0.99, CPU, log=QUIET),
              ValueError, "без init_err — отказ")
    ck4 = dict(ck, state_dict=dict(ck["state_dict"]))
    ck4["state_dict"]["0.mlp.0.weight"] = ck["state_dict"]["0.mlp.0.weight"].clone()
    ck4["state_dict"]["0.mlp.0.weight"][0, 2 * 15 + 1] = 0.1          # вес на канале лосса
    lfile = os.path.join(tmp, "teach_loss.pt")
    torch.save(ck4, lfile)
    must_fail(lambda: O.teacher_targets(lfile, *args, "delta", np.ones(1), np.array([1.0]), 0.99, CPU, log=QUIET),
              ValueError, "учитель, который смотрит на лоссы, не годится")
    for key, val in (("ctx_err", "none"), ("state_mode", "full"), ("ctx_budget", 31000)):
        bad = os.path.join(tmp, "teach_bad.pt")
        torch.save(dict(ck, meta=dict(ck["meta"], **{key: val})), bad)
        must_fail(lambda: O.teacher_targets(bad, *args, "delta", np.ones(1), np.array([1.0]), 0.99, CPU, log=QUIET),
                  ValueError, f"учитель с {key}={val} — отказ")
    print("учитель: Q на строках буфера = Q чекпоинта на состоянии оценки (один УрЧП и смесь), единицы, "
          "каркас, отказы: OK")
    return ckpt


def test_teacher_loss(ckpt):
    """Добавка к TD в train_variant: вместе с модельными целями, ученик приближается к учителю."""
    data = T.make_data()
    mask = data["EP"] < 4
    n = len(data["A"])
    QTE, TEV, _ = O.teacher_targets(ckpt, data, [PDE], "delta", np.ones(1), np.array([1.0]), 0.99, CPU, log=QUIET)
    TEV = TEV & (np.arange(n) != 1)                              # одна строка без цели
    d2 = dict(data, QTE=QTE, TEV=TEV)
    net1, _ = T.fit("cnn_dqn", d2, mask, teacher_w=1.0)
    assert net1.extra["teacher_loss"] > 0 and net1.extra["teacher_agree_rows"] == int((~mask & TEV).sum())
    assert 0.0 <= net1.extra["teacher_agree"] <= 1.0
    net2, _ = T.fit("cnn_dqn", d2, mask, teacher_w=1.0, epochs=25)
    assert net2.extra["teacher_loss"] < 0.5 * net1.extra["teacher_loss"], (net1.extra, net2.extra)
    net0, _ = T.fit("cnn_dqn", d2, mask)                         # вес 0: добавки нет, цели не читаются
    assert "teacher_loss" not in net0.extra
    tab, ens = T.fake_model()
    QMB, MBW, _ = MB.precompute(data, mask, [PDE], "delta", np.ones(1), np.array([1.0]), 0.99, h=1, lam=0.5,
                                model=(tab, ens), log=QUIET)
    for variant, kw in (("cnn_qrdqn", {}), ("convnext_dqn", dict(encoder="resnet", ens=2, ens_min=2, value_bound=True))):
        net, _ = T.fit(variant, dict(d2, QMB=QMB, MBW=MBW), mask, teacher_w=0.5, mb_aux=0.5, **kw)
        assert np.isfinite(net.extra["teacher_loss"]) and np.isfinite(net.extra["mb_loss"])
    must_fail(lambda: T.fit("cnn_dqn", data, mask, teacher_w=0.5), ValueError, "без посчитанных целей — отказ")
    print("учитель: добавка к TD и к модельным целям, ученик приближается к учителю: OK")


def test_online_targets(ckpt):
    """Онлайновое дообучение (online_aux.Aux): цели строк офлайновой половины — те же числа, что считает
    offline_rl.py; цели онлайнового перехода равны целям строки набора с той же цепочкой."""
    import torch.nn.functional as F
    data = T.make_data()
    n = len(data["A"])
    every = np.ones(n, bool)
    mask = data["EP"] < 4
    for form in ("delta", "dlog"):
        aux = OA.Aux(PDE, w_mb=0.5, w_te=0.5, reward_form=form, err_scale=1.0, init_err=1.0, gamma=0.99,
                     budget=7000, mb_h=3, mb_lam=0.5, teacher_from=ckpt, device=CPU, log=QUIET)
        off = aux.offline(data, mask, model=T.fake_model())
        QMB, MBW, _ = MB.precompute(data, every, [PDE], form, np.ones(1), np.array([1.0]), 0.99, h=3, lam=0.5,
                                    budget=7000, model=T.fake_model(), log=QUIET)
        QTE, TEV, _ = O.teacher_targets(ckpt, data, [PDE], form, np.ones(1), np.array([1.0]), 0.99, CPU,
                                        budget=7000, log=QUIET)
        assert (off["MBW"] == (MBW & mask)).all() and (off["TEW"] == (TEV & mask)).all() and MBW.any() and TEV.any()
        assert np.array_equal(off["QMB"][mask], QMB[mask]) and np.array_equal(off["QTE"], QTE)
        assert not off["QMB"][~mask].any(), "строки вне обучающей части без модельной цели"
        assert OA.Aux.row(off, int(np.where(~mask)[0][0])) is None
        r0 = OA.Aux.row(off, 0)
        assert np.array_equal(r0["mb"], QMB[0]) and np.array_equal(r0["te"], QTE[0])
        assert aux.meta()["teacher"]["rows"] == int(TEV.sum()) and aux.meta()["mb"]["rows"] == int((MBW & mask).sum())
        for i in range(n):
            s0 = i - int(data["STEP"][i])
            r = aux.live(data["A"][s0:i], [O.action_epochs(a) for a in data["A"][s0:i]], data["ERR"][s0:i]) or {}
            assert ("mb" in r) == bool(MBW[i]) and ("te" in r) == bool(TEV[i]), (form, i)
            if MBW[i]:
                assert r["mb"].shape == (27,) and np.allclose(r["mb"], QMB[i], atol=1e-6), (form, i)
            if TEV[i]:
                assert r["te"].shape == (27,) and np.allclose(r["te"], QTE[i], atol=1e-6), (form, i)
        assert aux.live_n["calls"] == n and aux.live_n["mb"] == int(MBW.sum()) and aux.live_n["te"] == int(TEV.sum())
    # остаток бюджета: действие в модели обрезается им, как в среде тренера; бюджет выбран — цели нет
    acts, errs = [2, 2], [0.5, 0.3]                                 # Adam 0.01 на 2500 эпох дважды
    r_big = aux.live(acts, [2500, 2500], errs)
    aux_small = OA.Aux(PDE, w_mb=0.5, reward_form="dlog", init_err=1.0, budget=5200, mb_h=1, log=QUIET)
    aux_small.build_model(model=T.fake_model())
    aux_wide = OA.Aux(PDE, w_mb=0.5, reward_form="dlog", init_err=1.0, budget=7000, mb_h=1, log=QUIET)
    aux_wide.build_model(model=T.fake_model())
    seen = []
    orig = MB.step_features

    def spy(tab, st, a, budget=0):
        X, ep = orig(tab, st, a, budget)
        seen.append(ep.copy())
        return X, ep
    MB.step_features = spy
    try:
        r_s, r_w = aux_small.live(acts, [2500, 2500], errs), aux_wide.live(acts, [2500, 2500], errs)
    finally:
        MB.step_features = orig
    assert r_big is not None and r_s["mb"].shape == (27,) and r_w["mb"].shape == (27,)
    assert seen[0].max() == 200 and (seen[0] == np.minimum(S.EPOCHS, 200)).all(), "обрезка остатком бюджета 200"
    assert (seen[1] == np.minimum(S.EPOCHS, 2000)).all()
    assert aux_small.live(acts + [1], [2500, 2500, 200], errs + [0.2]) is None, "бюджет выбран: цели нет"
    # слагаемое лосса — выражение train_variant: Huber по 27 действиям, среднее по строкам с целью
    torch.manual_seed(0)
    net = O.QNet("convnext_dqn", CPU, in_ch=4, encoder="resnet", ens=2)
    s = torch.randn(5, 4, 26, 26)
    rows = [dict(mb=QMB[0], te=QTE[0]), None, dict(mb=QMB[1]), dict(te=QTE[2]), dict(mb=QMB[3], te=QTE[3])]
    batch = OA.cat_aux([OA.stack_aux(rows[:2], CPU), OA.stack_aux(rows[2:], CPU)])
    assert batch["mb"][1].tolist() == [1, 0, 1, 0, 1] and batch["te"][1].tolist() == [1, 0, 0, 1, 1]
    term, parts = OA.aux_loss(net, s, batch, 0.5, 0.25)
    qv = net.q_heads(s)
    ref = 0.0
    for key, w in (("mb", 0.5), ("te", 0.25)):
        tgt, wt = batch[key]
        hub = F.smooth_l1_loss(qv, tgt[None].expand_as(qv), reduction="none").mean((0, 2))
        ref = ref + w * (hub * wt).sum() / wt.sum()
        assert abs(parts[key] - float(((hub * wt).sum() / wt.sum()).detach())) < 1e-7
    assert torch.allclose(term, ref) and term.requires_grad
    assert OA.aux_loss(net, s, OA.stack_aux([None] * 5, CPU), 0.5, 0.25) == (None, {}), "нет целей — нет слагаемого"
    assert OA.aux_loss(net, s, batch, 0.0, 0.0) == (None, {})
    print("онлайновое дообучение: цели офлайновых строк как у offline_rl.py, цели онлайнового перехода равны "
          "целям строки буфера с той же цепочкой, обрезка бюджетом, слагаемое лосса: OK")


def test_main(bufs, ckpt, tmp):
    """Полный прогон main() по боевому рецепту (convnext_dqn, delta, граница ценности, контекст)."""
    import killswitch
    import online_eval_env as E
    e0 = float(O.pde_meta()[PDE]["init_err"])
    root = os.path.join(tmp, "bufs", SUB)
    mtag = f"tsht{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    argv = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000", "--ctx-kmax", "70",
            "--save-model", "--scalar-ctx", "--encoder", "resnet", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
            "--teacher-from", ckpt, "--teacher-w", "0.5", "--seeds", "1", "--model-tag", mtag,
            "--epochs", "1", "--fqe-epochs", "1", "--batch-size", "8", "--cpu"]

    def fake_build(names, subdirs, table, members, work, quick, log):
        tab, ens = T.fake_model()
        return tab, ens, dict(surr_rows=0, surr_sig="fake", surr_members=3, surr_file="/x/combo_surr_fake.pkl")

    def run(av):
        buf = io.StringIO()
        sys.argv = av
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            O.main()
        return buf.getvalue()

    old = (sys.argv, killswitch.killed, MB.build_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    MB.build_model = fake_build
    try:
        text = run(argv)
        r = [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0]
        assert r["teacher_w"] == 0.5 and r["teacher_from"] == os.path.basename(ckpt), r
        assert r["teacher_units"] == "dlog->delta" and r["teacher_rows"] == 16 and 0 < r["teacher_train_rows"] <= 16
        assert np.isfinite(r["teacher_loss"]) and np.isfinite(r["mb_loss"]) and np.isfinite(r["td_error"])
        assert r["mb_surr_file"] == "/x/combo_surr_fake.pkl" and "'ok': True" in text
        ag, mean, std, var = E.load_agent(out)
        meta = ag._meta
        assert meta["teacher"]["rows"] == 16 and meta["teacher"]["mask"] and "mask" not in meta, \
            "маску учителя ученик не наследует"
        assert meta["mb"]["surr_file"] == "/x/combo_surr_fake.pkl" and meta["in_ch"] == 9 and meta["ctx_err"] == "none"
        for bad in (argv + ["--distill-from", ckpt], argv[:argv.index("--teacher-from")] + argv[argv.index("--teacher-w"):],
                    [a for a in argv if a not in ("--init-err", repr(e0), "--value-bound")], argv + ["--pbrs", "0.5"]):
            must_fail(lambda: run(bad), SystemExit, f"недопустимое сочетание флагов: {bad[-4:]}")
    finally:
        sys.argv, killswitch.killed, MB.build_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("main(): --teacher-from и --teacher-w с модельными целями, meta чекпоинта, отказы: OK")


def test_eval_flags():
    """Флаги оценки: по умолчанию щит выключен."""
    import online_eval_env as E
    a = E.build_parser().parse_args(["--policy", "agent"])
    assert a.shield == "" and a.shield_lam == 0.5 and a.shield_keep == 3 and a.shield_err == "true"
    a = E.build_parser().parse_args(["--policy", "agent", "--shield", "auto", "--shield-lam", "1", "--shield-keep", "5"])
    assert a.shield == "auto" and a.shield_lam == 1.0 and a.shield_keep == 5
    print("оценка: флаги щита, по умолчанию выключен: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    bufs = make_buffers()
    with tempfile.TemporaryDirectory() as tmp:
        tab, tab_l, data = test_features(bufs, tmp); lap()
        test_shield_rule(tab_l); lap()
        test_shield_loop(tab, tab_l, data, tmp); lap()
        ckpt = test_teacher(bufs, tmp); lap()
        test_teacher_loss(ckpt); lap()
        test_online_targets(ckpt); lap()
        test_main(bufs, ckpt, tmp); lap()
        test_eval_flags(); lap()
    print(f"test_shield_teacher: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
