#!/usr/bin/env python
"""
Проверки урезанного буфера цели (offline_rl.py --target-chains N --data-seed K; agent_combo.py с теми
же флагами; mb_targets.target_subset, match_chains) без GPU, без сети и без обучения PINN, на
крошечном синтетическом буфере задачи-цели (5 файлов, 14 записанных цепочек) и второй задачи:

  * цепочки, оставленные тренеру ученика, — ровно те, что agent_combo.target_chain_rows оставляет в
    таблице суррогата, при любых N и K. Сверка независимая: у тренера EP — номер записанной
    цепочки в порядке чтения, у таблицы CH — её порядковый номер среди непустых; сопоставление в
    коде идёт по содержимому и этих номеров не видит. В буфере есть цепочка, обрезанная бюджетом
    (в таблице она целая), незаконченные хвосты файлов с копией next_state (теряют последний
    переход; пустой; хвост, чей последний переход выбирает бюджет, — у тренера он остаётся),
    файл, начатый с середины цепочки (без записанного начала), повтор цепочки и две цепочки с
    общим первым переходом;
  * цепочка тренера без начала или с чужим содержимым отбрасывается и считается; при малой доле
    найденных цепочек — отказ; отказы на несовместимых флагах;
  * настоящий agent_combo.run с теми же N и K учит суррогат на той же таблице (тот же отпечаток
    данных в имени кэша), отпечаток урезанной таблицы отличается от полной;
  * полный прогон main() по боевому рецепту с --target-chains: строка результата, meta чекпоинта,
    отложенная часть только из оставленных цепочек; без флага — прежний путь.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_target_chains.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")   # бустинг на десятках строк: потоки только мешают
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
import test_shield_teacher as TS  # noqa: E402

torch.set_num_threads(1)
QUIET = TS.QUIET
PDE, SUB, PDE2, SUB2 = TS.PDE, TS.SUB, TS.PDE2, TS.SUB2
KEYS = TS.KEYS
BUDGET = 7000
# Записанные цепочки цели по файлам: (копия next_state в дампе, [(действия, флаг конца)]).
# Действия 2, 5, 8 — Adam на 2500 эпох: третье подряд выбирает бюджет 7000.
FILES = [
    (False, [([10, 19, 3], -1),             # 0
             ([2, 5, 8, 2, 5], 1),          # 1: обрезается бюджетом на третьем шаге, в таблице 5 строк
             ([12, 20], 0)]),               # 2: хвост старого формата — остаётся целиком
    (True, [([4, 13, 22, 9], -1),           # 3
            ([1, 11, 21], 1),               # 4
            ([14, 23, 6], 0)]),             # 5: хвост с копией next_state теряет последний переход
    (True, [([16, 25], -1),                 # 6
            ([2, 5, 8], 0)]),               # 7: хвост, чей последний переход выбирает бюджет
    (True, [([7, 17, 26], -1),              # 8
            ([15], 0)]),                    # 9: хвост из одного перехода — пустая цепочка
    (False, [([18, 0], 1),                  # 10: файл начат с середины цепочки (начало не записано)
             ([10, 19, 3], -1),             # 11: повтор цепочки 0 (те же ошибки)
             ([10, 24, 3, 13], -1),         # 12: первый переход как у цепочки 0, дальше своё
             ([2, 2, 2, 2], -1)]),          # 13: обрезается бюджетом, в таблице 4 строки
]
ROWS_TAB = {0: 3, 1: 5, 2: 2, 3: 4, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 10: 2, 11: 3, 12: 4, 13: 4}     # строк в таблице
ROWS_TR = {**ROWS_TAB, 1: 3, 7: 3, 13: 3}                                                      # у тренера
RAW = sorted(ROWS_TAB)                      # записанные цепочки, от которых остались строки


def make_target(seed=1):
    rng = np.random.default_rng(seed)
    errs, files = {}, []
    cid = 0
    for copy, chains in FILES:
        out = []
        for j, (acts, end) in enumerate(chains):
            e = [float(x) for x in 10.0 ** rng.uniform(-3, -0.5, len(acts))]
            if cid == 11:
                e = list(errs[0])
            if cid == 12:
                e[0] = errs[0][0]
            errs[cid] = e
            start_zero = not (cid == 10 or copy)
            maps = [{k: (np.zeros((26, 26), np.float32) if start_zero and i == 0 else
                         rng.random((26, 26)).astype(np.float32) + 1.0) for k in KEYS} for i in range(len(acts) + 1)]
            for i, a in enumerate(acts):
                done = end if i == len(acts) - 1 else 0
                st, nxt = dict(maps[i]), dict(maps[i] if copy else maps[i + 1])
                if copy:
                    st["delta"] = nxt["delta"] = np.zeros((26, 26), np.float32)
                out.append(dict(state=st, next_state=nxt, action=(a // 9, {"lr": (a % 9) // 3, "epochs": a % 3}),
                                reward=-(e[i] + (1.0 if done == -1 else 0.0)), done=done))
            cid += 1
        files.append(out)
    return files


def rows_of(data, keep):
    n = len(data["A"])
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in data.items()}


def must_exit(fn, text, what):
    try:
        fn()
    except SystemExit as e:
        assert text in str(e), (what, str(e))
        return
    raise AssertionError(what)


# --------------------------------------------------------------------------------------------

def test_match(bufs, tmp):
    e0 = float(O.pde_meta()[PDE]["init_err"])
    table = os.path.join(tmp, "table.npz")
    S.save_table(S.build_table(episodes=bufs, verbose=False), table)
    work = os.path.join(tmp, "work")
    full = MB.load_tab([PDE], table=table, work=work, log=QUIET)
    pi = full["NAMES"].index(PDE)
    data = O.episodes_to_arrays(bufs[SUB], chain_fix=True, reward_form="delta", init_err=e0, budget=BUDGET,
                                verbose=False)
    # буфер устроен как задумано: нумерация тренера считает пустую цепочку 9, таблица — нет
    assert pi == 1 and sorted(np.unique(data["EP"]).tolist()) == RAW
    assert {c: int((data["EP"] == c).sum()) for c in RAW} == ROWS_TR and len(data["A"]) == 37
    chs = np.unique(full["CH"][full["PDE"] == pi])
    off = int(chs[0])
    assert off > 0 and (chs == off + np.arange(len(RAW))).all()
    assert [int((full["CH"] == c).sum()) for c in chs] == [ROWS_TAB[c] for c in RAW]
    assert data["STATS"]["budget_cut"] == 3 and data["STATS"]["dropped_tail"] == 2
    _, keys = MB.table_chain_keys(full, pi, BUDGET)
    assert keys[RAW.index(0)] == keys[RAW.index(11)] and len(set(keys)) == len(RAW) - 1
    # выбор зависит только от числа цепочек цели: таблица из одной задачи выбирает те же по порядку
    alone = AC.with_epa(S.build_table(episodes={SUB: bufs[SUB]}, verbose=False))
    for n, k in ((1, 0), (4, 3), (7, 1)):
        a, b = AC.target_chain_rows(full, pi, n, k)[1], AC.target_chain_rows(alone, 0, n, k)[1]
        assert (a - off == b).all() and len(a) == n
        # и это прежний выбор agent_combo.py (до выноса в функцию): старые учителя воспроизводимы
        assert (a == np.random.default_rng(k).choice(chs, min(n, len(chs)), replace=False)).all()
    sig_full = AC.Models(full, work, members=2, quick=True, log=QUIET).sig
    sig_of, seen_fix = {}, set()
    for n in (1, 2, 3, 5, 8, 12, 13, 99):
        for k in (0, 1, 2, 3):
            logs = []
            out, small, info = MB.target_subset(data, PDE, n, k, budget=BUDGET, table=table, work=work,
                                                log=logs.append)
            keep_tab, pick = AC.target_chain_rows(full, pi, n, k)             # путь agent_combo.py
            want = sorted(RAW[int(c) - off] for c in pick)                      # записанные цепочки по номеру CH
            got = sorted(np.unique(out["EP"]).tolist())                        # у тренера EP — номер записанной
            assert got == want, (n, k, got, want)
            m = min(n, len(RAW))
            assert info["requested"] == n and info["data_seed"] == k and info["buffer"] == len(RAW)
            assert info["table"] == info["data"] == m and info["unmatched"] == info["missing"] == 0
            assert info["no_start"] == 0 and info["not_picked"] == len(RAW) - m and info["found"] == 1.0
            assert info["unmatched_action"] == "dropped" and info["data_before"] == len(RAW)
            assert info["rows"] == len(out["A"]) == sum(ROWS_TR[c] for c in want) and info["rows_before"] == 37
            assert info["table_rows"] == sum(ROWS_TAB[c] for c in want) == int((small["PDE"] == pi).sum())
            assert info["tail_fix"] == int(7 in want)
            seen_fix.add(info["tail_fix"])
            # таблица: чужая задача цела, у цели — выбранные цепочки целиком (без обрезки по бюджету)
            assert (small["PDE"] != pi).sum() == (full["PDE"] != pi).sum() == info["surr_rows"] - info["table_rows"]
            assert sorted(np.unique(small["CH"][small["PDE"] == pi]).tolist()) == sorted(int(c) for c in pick)
            assert all(np.array_equal(small[c], full[c][keep_tab], equal_nan=(c == "LVN")) for c in AC.ROWKEYS)
            # набор тренера: строки оставленных цепочек без изменений, остальные столбцы на месте
            sel = np.isin(data["EP"], want)
            for c in ("S", "S2", "A", "R", "D", "EP", "RTG", "FIRST", "ERR", "STEP", "SPENT"):
                assert np.array_equal(out[c], data[c][sel]), c
            assert out["STATS"]["target_chains"] == info and "target_chains" not in data["STATS"]
            tr, te = O.split_by_episode(out)
            assert tr.sum() + te.sum() == info["rows"] and set(np.unique(out["EP"][te])) <= set(want)
            # отпечаток данных в имени кэша ансамбля: урезанная таблица не совпадает с полной
            sig = AC.Models(small, work, members=2, quick=True, log=QUIET).sig
            assert (sig == sig_full) == (m == len(RAW)), (n, k)
            assert sig_of.setdefault(tuple(want), sig) == sig
            assert ("ВНИМАНИЕ" in " ".join(logs)) == (n > len(RAW)) and "отброшены 0" in logs[-1]
    assert len(set(sig_of.values())) == len(sig_of) > 20, "разный набор цепочек — разный файл кэша"
    assert seen_fix == {0, 1}
    # без бюджета тренер не режет цепочки: строк столько же, сколько в таблице
    nob = O.episodes_to_arrays(bufs[SUB], chain_fix=True, reward_form="dlog", verbose=False)
    out, small, info = MB.target_subset(nob, PDE, 6, 2, table=table, work=work, log=QUIET)
    assert info["data"] == 6 and info["rows"] == info["table_rows"] and info["tail_fix"] == 0
    print("урезанный буфер: тренер и agent_combo оставляют одни цепочки (бюджет, хвосты, повтор, общий "
          "первый переход), отпечаток кэша свой: OK")
    return table, work, data, full, off


def test_unmatched(table, work, data, full, off):
    """Цепочки тренера, которых нет в таблице, отбрасываются; порог найденных; отказы."""
    all_n = len(RAW)
    kw = dict(budget=BUDGET, table=table, work=work, log=QUIET)
    # цепочка без начала: у цепочки 3 нет первой строки
    cut = rows_of(data, ~((data["EP"] == 3) & (data["STEP"] == 0)))
    out, _, info = MB.target_subset(cut, PDE, all_n, 0, **kw)
    assert 3 not in out["EP"] and info["unmatched"] == info["no_start"] == info["missing"] == 1
    assert info["data"] == all_n - 1 and info["rows"] == 37 - ROWS_TR[3] and info["found"] == round(12 / 13, 4)
    # чужая цепочка: ошибка одного шага другая (таблица собрана из другого файла)
    alien = dict(data, ERR=data["ERR"].copy())
    alien["ERR"][np.where(data["EP"] == 4)[0][1]] *= 1.0001
    out, _, info = MB.target_subset(alien, PDE, all_n, 0, **kw)
    assert 4 not in out["EP"] and info["unmatched"] == 1 and info["no_start"] == 0 and info["missing"] == 1
    # не выбранная и не сопоставленная цепочки считаются отдельно
    pick = AC.target_chain_rows(full, 1, 5, 1)[1]
    lost = RAW[int(pick[0]) - off]
    out, _, info = MB.target_subset(rows_of(data, data["EP"] != lost), PDE, 5, 1, **kw)
    assert info["data"] == 4 and info["missing"] == 1 and info["unmatched"] == 0 and info["not_picked"] == all_n - 5
    # повтор цепочки: лишняя копия у тренера (в таблице их две) не сопоставляется
    dup = np.r_[np.arange(len(data["A"])), np.where(data["EP"] == 11)[0]]
    more = rows_of(data, dup)
    more["EP"] = more["EP"].copy()
    more["EP"][-ROWS_TR[11]:] = 99
    out, _, info = MB.target_subset(more, PDE, all_n, 0, **kw)
    assert info["unmatched"] == 1 and info["data"] == all_n and 99 not in out["EP"]
    # порог: тренер прочитал только два файла из пяти
    assert MB.TARGET_MIN_FOUND == 0.8
    part = rows_of(data, data["EP"] <= 5)
    must_exit(lambda: MB.target_subset(part, PDE, all_n, 0, **kw), "меньше 80%", "мало найденных цепочек — отказ")
    # задачи нет в таблице суррогата; набор без границ цепочек; смесь УрЧП
    must_exit(lambda: MB.target_subset(data, "heat2d_multiscale", 3, 0, **kw), "нет цепочек задачи",
              "цели нет в таблице — отказ")
    legacy = {k: v for k, v in data.items() if k not in ("STEP", "SPENT", "ERR")}
    TS.must_fail(lambda: MB.target_subset(legacy, PDE, 3, 0, **kw), ValueError, "без --chain-fix — отказ")
    TS.must_fail(lambda: MB.target_subset(dict(data, PDE=np.zeros(len(data["A"]), np.int64)), PDE, 3, 0, **kw),
                 ValueError, "смесь УрЧП — отказ")
    print("урезанный буфер: цепочки без начала и чужие отброшены и посчитаны, порог найденных, отказы: OK")


def test_teacher_and_main(bufs, table, work, tmp):
    """Настоящий учитель agent_combo.run и полный прогон main() ученика с теми же N и K."""
    import killswitch
    import online_eval_env as E
    n, k = 5, 1
    e0 = float(O.pde_meta()[PDE]["init_err"])
    assert os.path.exists(table)                                   # учитель и ученик читают одну таблицу
    ckpt, _ = TS.train_teacher(bufs, tmp, tag="teach_tc", extra=["--target-chains", str(n), "--data-seed", str(k)])
    with open(os.path.join(tmp, "rl_arch", "online_env", f"{PDE}_teach_tc_train.json")) as f:
        rep = json.load(f)
    full = MB.load_tab([PDE], table=table, work=work, log=QUIET)
    keep_tab, pick = AC.target_chain_rows(full, 1, n, k)
    want = sorted(RAW[int(c) - int(full["CH"][full["PDE"] == 1].min())] for c in pick)
    sz = rep["sizes"]
    assert sz["target_chains_buffer"] == len(RAW) and sz["target_chains"] == n and sz["rows"] == int(keep_tab.sum())
    assert sz["target_rows"] == sum(ROWS_TAB[c] for c in want)
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["meta"]["combo"]["data_seed"] == k
    root = os.path.join(tmp, "bufs", SUB)
    os.makedirs(root)
    for i, ep in enumerate(bufs[SUB]):
        torch.save(ep, os.path.join(root, f"f{i}.pt"))
    mtag = f"ttc{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    head = ["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--chain-fix", "--reward-form", "delta",
            "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", str(BUDGET),
            "--ctx-kmax", "70", "--scalar-ctx", "--encoder", "resnet", "--mb-table", table, "--mb-work", work,
            "--mb-members", "2", "--seeds", "1", "--model-tag", mtag, "--epochs", "1", "--fqe-epochs", "1",
            "--batch-size", "8", "--cpu"]
    base = head + ["--save-model", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5", "--teacher-from", ckpt,
                   "--teacher-w", "0.5"]
    tc = ["--target-chains", str(n), "--data-seed", str(k)]
    argv = base + tc

    def run(av):
        buf = io.StringIO()
        sys.argv = av
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            O.main()
        return buf.getvalue()

    def refuses(av, text):
        err = io.StringIO()
        sys.argv = av
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                O.main()
        except SystemExit:
            assert text in err.getvalue(), (text, err.getvalue()[-300:])
            return
        raise AssertionError(f"недопустимое сочетание флагов должно отвергаться: {text}")

    def row_of(text):
        return [json.loads(ln) for ln in text.splitlines() if ln.startswith('{"variant"')][0]

    old = (sys.argv, killswitch.killed, MB.target_subset, MB.fit_model, O.GAMMA)
    killswitch.killed = lambda tag, remote=False, every=600: None
    # быстрые настройки бустинга, как у учителя (--surr-quick): тогда и имя файла в кэше то же
    MB.fit_model = lambda tab, members, work, quick, log: old[3](tab, members, work, True, log)
    surr = os.path.join(work, f"combo_surr_{rep['surrogate']['data_sig']}_full_m2_q_s0.pkl")
    stamp = os.stat(surr).st_mtime_ns
    try:
        text = run(argv)
        r = row_of(text)
        rows = sum(ROWS_TR[c] for c in want)
        assert r["target_chains"] == n and r["data_seed"] == k and r["tc_requested"] == n and r["tc_buffer"] == len(RAW)
        assert r["tc_table"] == r["tc_data"] == n and r["tc_unmatched"] == 0 and r["tc_missing"] == 0
        assert r["tc_unmatched_action"] == "dropped" and r["tc_rows"] == rows and r["tc_rows_before"] == 37
        assert r["tc_table_rows"] == sz["target_rows"] and r["tc_surr_rows"] == sz["rows"] == r["mb_surr_rows"]
        # суррогат ученика учится на той же таблице, что суррогат учителя: отпечаток данных тот же
        assert r["mb_surr_sig"] == rep["surrogate"]["data_sig"], (r["mb_surr_sig"], rep["surrogate"]["data_sig"])
        # ...и потому берёт из кэша файл ансамбля, который записал учитель, а не учит свой
        assert os.path.samefile(r["mb_surr_file"], surr) and os.stat(surr).st_mtime_ns == stamp
        # TD, модельные цели, цели учителя и отложенная часть — только на оставленных цепочках
        assert f"transitions={rows} " in text and "урезанный буфер цели" in text and "'ok': True" in text
        tr, te = [int(x.split("=")[1]) for x in text[text.index(f"transitions={rows} "):].split()[1:3]]
        assert tr + te == rows and te > 0 and r["teacher_rows"] <= rows and r["mb_rows"] <= tr
        assert 0 < r["teacher_train_rows"] <= tr and np.isfinite(r["mb_loss"]) and np.isfinite(r["teacher_loss"])
        ag, _, _, _ = E.load_agent(out)
        meta = ag._meta
        assert meta["target_chains"] == n and meta["data_seed"] == k
        assert meta["target_data"] == {key[3:]: v for key, v in r.items() if key.startswith("tc_")}
        assert meta["mb"]["surr_file"] == r["mb_surr_file"] and meta["mb"]["surr_sig"] == r["mb_surr_sig"]
        # без --mb-aux таблица нужна только для выбора: данные ученика урезаны так же
        r2 = row_of(run(head + tc))
        assert r2["tc_rows"] == rows and r2["tc_data"] == n and "mb_surr_sig" not in r2 and r2["mb_aux"] == 0.0
        # отказы на несовместимых флагах
        refuses(argv + ["--subdirs", f"{SUB2},{SUB}"], "--target-chains несовместим с --subdirs")
        refuses(argv + ["--per-pde-files", "2"], "--target-chains несовместим с --per-pde-files")
        refuses(head + tc + ["--extra-buffers", "q_a.pt"], "--target-chains несовместим с --extra-buffers")
        refuses(["offline_rl.py", "--variant", "convnext_dqn", "--data-dir", root, "--seeds", "1", "--cpu"] + tc,
                "--target-chains требует --chain-fix")
        refuses(base + ["--target-chains", "-1"], "--target-chains >= 0")
        # по умолчанию прежний путь: отбор не вызывается, в данных все цепочки
        def never(*a, **kw):
            raise AssertionError("без --target-chains отбор цепочек не вызывается")
        MB.target_subset = never
        text0 = run(base + ["--data-seed", "7"])
        r0 = row_of(text0)
        assert r0["target_chains"] == 0 and not [key for key in r0 if key.startswith("tc_")]
        assert "transitions=37 " in text0 and "урезанный буфер" not in text0 and r0["mb_surr_rows"] == len(full["A"])
        assert r0["mb_surr_sig"] != r["mb_surr_sig"], "полная и урезанная таблицы — разные файлы кэша"
        assert "target_data" not in E.load_agent(out)[0]._meta
    finally:
        sys.argv, killswitch.killed, MB.target_subset, MB.fit_model, O.GAMMA = old
        if os.path.exists(out):
            os.remove(out)
    print("учитель agent_combo.run и main() ученика с теми же N и K: одна таблица суррогата, строка результата, "
          "meta чекпоинта, отказы, прежний путь по умолчанию: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    rng = np.random.default_rng(0)
    e2 = float(O.pde_meta()[PDE2]["init_err"])
    bufs = {SUB2: [TS.make_dump(rng, e2, ln) for ln in ((3, 2, 4), (4, 3))], SUB: make_target()}
    with tempfile.TemporaryDirectory() as tmp:
        table, work, data, full, off = test_match(bufs, tmp); lap()
        test_unmatched(table, work, data, full, off); lap()
        test_teacher_and_main(bufs, table, work, tmp); lap()
    print(f"test_target_chains: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
