#!/usr/bin/env python
"""
Проверки буфера с картами от направленного сбора (collect_active.py --save-buffer) и обучения на нём
вместо исходного буфера цели (agent_combo.py / offline_rl.py --target-buffer, target_buffer.py). Без
GPU, без сети, PINN и автокодировщик не обучаются: среда подменена копией контракта run_seed с
картами (порядок как в online_eval_env: действие, progress_cb, карта через args._map_sink, чекпоинт,
мягкий выход), содержимое карты однозначно задано сидом и исполненным началом цепочки.

  * файл на цепочку в формате офлайнового буфера: читается offline_rl.load_episodes +
    episodes_to_arrays_chains и surrogate.build_table без изменений; состояние загрузчика (канал
    delta) совпадает с online_eval_env.build_state на тех же картах;
  * цепочка с общим префиксом записана целиком с шага 0, переходы префикса — те же, что у владельца;
    остановленная приёмом 3 — короче, её последний переход не терминальный, с настоящей картой;
    выбравшая бюджет — терминальная, обрезанное действие хранит задуманный номер и фактические эпохи;
  * заливка: один коммит на раунд со всеми новыми файлами, до заливки состояния;
  * сборщики --worker k/K: цепочки и сиды не пересекаются, выходы складываются;
  * --target-buffer: учитель и ученик не читают исходный буфер цели (счёт цепочек и содержимое
    таблицы), таблица суррогата у них одна (тот же отпечаток), кэш не совпадает со стандартным;
    отказы на несовместимых флагах; без флага — прежний путь (--no-state, прежние поля записи);
  * размер смоука (2 цепочки по 2 действия) проходит оба тренера.

    DDEBACKEND=pytorch HF_HUB_OFFLINE=1 python experiments/rl_arch/tests/test_collect_buffer.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.pop("HF_TOKEN", None)            # тест ничего не заливает
os.environ.pop("HF_TOKEN_WRITE", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "track3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import collect_active as C  # noqa: E402
import offline_rl as O  # noqa: E402
import surrogate as S  # noqa: E402
import mb_targets as MB  # noqa: E402
import agent_combo as AC  # noqa: E402
import target_buffer as TB  # noqa: E402
import killswitch  # noqa: E402
import test_collect_active as TC  # noqa: E402
import test_shield_teacher as TS  # noqa: E402

torch.set_num_threads(1)
E = C.E
PDE, SUB, PDE2, SUB2 = TS.PDE, TS.SUB, TS.PDE2, TS.SUB2
KEYS = list(C.MAP_KEYS)
SK, C1, C2, PSO4 = TC.SK, TC.C1, TC.C2, TC.PSO4
C_B, C_C = SK + ",LBFGS:0.5:1000,LBFGS:1:500", SK + ",PSO:0:100"
BAD = PSO4 + ",PSO:0.001:300,PSO:0.001:300"
FULL = "Adam:0.01:2500,Adam:0.001:2500,Adam:0.0001:2500"       # третье действие обрезается до 2000
CALLS = []


def map_of(seed, acts, step):
    """Карта (4, 26, 26) после действия step цепочки acts на сиде seed (что отдала бы среда)."""
    rng = np.random.default_rng([int(seed), int(step)] + [int(x) for x in acts[:step]])
    return (rng.random((4, 26, 26)) + 1.0).astype(np.float32)


def fake_run_seed(seed, args, progress_cb=None):
    """Контракт online_eval_env.run_seed для --policy script с картами: тот же порядок шагов."""
    assert args.policy == "script" and args.script_tail == "stop" and not args.no_state
    assert args.state_every == 1 and args.ae_epochs == 10000, "карты — с умолчаниями оценки"
    assert callable(args._map_sink)
    t0 = time.time()
    script = E.parse_chain(args.script)
    name = getattr(args, "_ckpt_name", None)
    CALLS.append(dict(script=args.script, seed=seed, resume=bool(args.resume)))
    spent, chain, hist, l, pa = 0, [], [], 0.01 * (seed % 5 - 2), -1
    if args.resume and name and os.path.exists(f"ckpt_{name}.pt"):
        ck = torch.load(f"ckpt_{name}.pt", weights_only=False)
        spent, chain, hist, l, pa = ck["spent"], ck["chain"], ck["t3"]["hist"], ck["l"], ck["pa"]
    start = spent

    def dump():
        torch.save(dict(spent=spent, chain=chain, t3=dict(hist=hist), l=l, pa=pa), f"ckpt_{name}.pt")

    stop = "budget"
    while spent < args.budget:
        if len(chain) >= len(script):
            stop = "script_end"
            break
        x = script[len(chain)]
        o, lr, ep = E.ACTION_TABLE[x]
        ep = min(ep, args.budget - spent)
        l += float(TC.delta(x, pa, ep))
        pa, spent = x, spent + ep
        chain.append([o, lr, ep])
        l2 = 10 ** l
        hist.append([spent, 3 * l2, l2, 0.01 * (spent - start)])
        if progress_cb is not None and spent < args.budget:
            progress_cb(dict(seed=seed, policy="script", pde=args.pde, partial=True, l2re=l2, rmse=0.4 * l2,
                             brmse=0.1 * l2, spent=spent, budget=args.budget, n_steps=len(chain),
                             chain=chain, elapsed_s=0.01 * (spent - start)))
        if spent >= args.budget:
            break
        args._map_sink(len(chain), map_of(seed, script, len(chain)))
        if name and args.ckpt_every and len(chain) % args.ckpt_every == 0:
            dump()
        if args.hours and (time.time() - t0) / 3600.0 >= args.hours:
            if name:
                dump()
            return dict(seed=seed, policy="script", pde=args.pde, unfinished=True, l2re=l2, spent=spent,
                        budget=args.budget, n_steps=len(chain), elapsed_s=0.01 * (spent - start))
    l2 = 10 ** l
    return dict(seed=seed, policy="script", pde=args.pde, l2re=l2, stop_reason=stop, spent=spent,
                rmse=0.4 * l2, brmse=0.1 * l2, budget=args.budget, n_steps=len(chain), chain=chain,
                hist=hist, keep_opt=bool(args.keep_opt), elapsed_s=0.01 * (spent - start))


@contextlib.contextmanager
def patched():
    saved = (C.run_seed, C.upload_files, C.upload, killswitch.killed)
    tmp, cwd = tempfile.mkdtemp(prefix="collect_buf_"), os.getcwd()
    del CALLS[:]
    C.run_seed = fake_run_seed
    killswitch.killed = lambda tag, remote=False, every=600: None
    os.chdir(tmp)                       # чекпоинты среды пишутся в текущую папку
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        C.run_seed, C.upload_files, C.upload, killswitch.killed = saved
        shutil.rmtree(tmp, ignore_errors=True)


def collect(tmp, tag, extra, smoke=True, budget="7000"):
    argv = ["--pde", PDE, "--budget", budget, "--save-dir", tmp, "--tag", tag, "--save-buffer",
            "--acq", "random"] + (["--smoke"] if smoke else []) + extra
    with contextlib.redirect_stdout(io.StringIO()):
        return C.main(argv)


def files_of(tmp, tag):
    d = os.path.join(tmp, "buffers_new", tag, SUB)
    return d, {f: torch.load(os.path.join(d, f), weights_only=False) for f in sorted(os.listdir(d))}


def same_tr(x, y):
    return (x["action"] == y["action"] and x["reward"] == y["reward"] and x["epochs_real"] == y["epochs_real"]
            and all(np.array_equal(x[s][k], y[s][k]) for s in ("state", "next_state") for k in KEYS))


# --------------------------------------------------------------------------------------------

def test_files(tmp):
    """Сбор с общим префиксом, ранней остановкой и цепочкой до конца бюджета -> файлы буфера."""
    st = collect(tmp, "w0", ["--rounds", "1", "--batch", "6", "--share-prefix", "--keep-opt", "--keep-opt-mode",
                             "safe", "--truncate", "3", "--truncate-min", "300", "--truncate-min-ref", "2",
                             "--init", f"{C1};{C_B};{C_C};{C2};{FULL};{BAD}"])
    recs = st["records"]
    assert [r["planned"] for r in recs] == [C1, C_B, C_C, C2, FULL, BAD]
    assert [r["shared_prefix_len"] for r in recs] == [2, 2, 2, 0, 0, 0] and [r["truncated"] for r in recs][5]
    assert [c["script"] for c in CALLS] == [SK, C1, C_B, C_C, C2, FULL, BAD], "префикс исполнен один раз"
    d, files = files_of(tmp, "w0")
    assert list(files) == [f"w0_c{i:04d}.pt" for i in range(6)] == [r["buf_file"] for r in recs]
    assert st["buffer"]["files"] == 6 and st["buffer"]["transitions"] == sum(r["buf_n"] for r in recs)
    assert st["buffer"]["hf"] == f"rl_arch/buffers_new/w0/{SUB}" and st["buffer"]["uploaded"] == 0
    assert "maps" not in json.dumps(st), "карты в JSON состояния не попадают"
    for r, tr in zip(recs, files.values()):
        acts = r["actions"]
        assert len(tr) == len(r["chain"]) == r["buf_n"] and all(isinstance(t, dict) for t in tr)
        assert tr[0]["meta"]["seed"] == r["seed"] and tr[0]["meta"]["env"] == r["env"]
        for i, t in enumerate(tr):
            a = t["action"]
            assert a[0] * 9 + a[1]["lr"] * 3 + a[1]["epochs"] == acts[i], "задуманный номер действия"
            assert t["reward"] == -r["err"][i] and t["epochs_real"] == r["chain"][i][2]
            want = np.zeros((3, 26, 26), np.float32) if i == 0 else map_of(r["seed"], acts, i)[:3]
            assert all(t["state"][k].shape == (26, 26) and t["state"][k].dtype == np.float32 for k in KEYS)
            assert np.array_equal(np.stack([t["state"][k] for k in KEYS]), want), "карта перед действием"
            if i < len(tr) - 1:
                assert t["done"] == 0
                assert np.array_equal(np.stack([t["next_state"][k] for k in KEYS]), map_of(r["seed"], acts, i + 1)[:3])
    # общий префикс: у концовок файл с шага 0, первые два перехода — те же, что у владельца
    own = files["w0_c0000.pt"]
    for f in ("w0_c0001.pt", "w0_c0002.pt"):
        assert len(files[f]) > 2 and same_tr(files[f][0], own[0]) and same_tr(files[f][1], own[1])
        assert files[f][0]["group"] == own[0]["group"] == "w0#g0" and not same_tr(files[f][2], own[2])
    assert files["w0_c0003.pt"][0]["group"] is None
    # концы цепочек: раньше бюджета — не терминал, настоящая следующая карта; бюджет выбран — терминал
    for i in (0, 1, 2, 3):
        t = files[f"w0_c{i:04d}.pt"][-1]
        assert t["done"] == 0 and not np.array_equal(t["next_state"]["loss_total"], t["state"]["loss_total"])
        assert np.array_equal(t["next_state"]["loss_total"], map_of(recs[i]["seed"], recs[i]["actions"], recs[i]["buf_n"])[0])
    full = files["w0_c0004.pt"]
    assert full[-1]["done"] == 1 and full[-1]["epochs_real"] == 2000 and full[-1]["spent"] == 7000
    assert full[-1]["action"] == (0, {"lr": 2, "epochs": 2}) and len(full) == 3
    bad = files["w0_c0005.pt"]
    assert recs[5]["stop"] == "truncated" and len(bad) == 4 == len(recs[5]["chain"]) and bad[-1]["done"] == 0
    assert np.array_equal(bad[-1]["next_state"]["loss_total"], map_of(recs[5]["seed"], recs[5]["actions"], 4)[0])
    print("файлы буфера: формат, общий префикс целиком, остановленная и законченная цепочки: OK")
    return d, recs


def test_loaders(d, recs):
    """Файлы читают загрузчики офлайновых буферов без изменений; состояние — как у среды оценки."""
    e0 = float(O.pde_meta()[PDE]["init_err"])
    eps = O.load_episodes(d, SUB)
    assert len(eps) == 6
    data = O.episodes_to_arrays_chains(eps, reward_form="delta", init_err=e0, budget=7000, verbose=False)
    stt = data["STATS"]
    assert stt["chains"] == 6 and stt["copy_files"] == 0 and stt["dropped_tail"] == 0 and stt["diverged"] == 0
    assert len(data["A"]) == sum(r["buf_n"] for r in recs) and stt["tail"] == 5 and stt["budget_cut"] == 1
    for c, r in enumerate(recs):
        rows = np.where(data["EP"] == c)[0]
        n = len(rows)
        assert n == len(r["chain"]) and (data["STEP"][rows] == np.arange(n)).all(), "цепочка с шага 0"
        assert data["A"][rows].tolist() == r["actions"][:n]
        assert np.allclose(data["ERR"][rows], r["err"], rtol=1e-6)
        assert data["SPENT"][rows].tolist() == [0] + [h[0] for h in r["hist"][:-1]]
        assert not data["S"][rows[0]].any() and data["FIRST"][rows[0]] == 1
        assert abs(data["R"][rows[0]] - (e0 - r["err"][0])) < 1e-6
        assert data["D"][rows].tolist() == [0.0] * (n - 1) + [1.0 if r["planned"] == FULL else 0.0]
        prev = None
        for j in range(1, n + 1):       # состояние после действия j: загрузчик против build_state оценки
            m = map_of(r["seed"], r["actions"], j)
            raw = {k: torch.tensor(m[i]) for i, k in enumerate(KEYS)}
            want = E.build_state(raw, prev)
            prev = raw
            if j < n:
                assert np.allclose(data["S"][rows[j]], want, atol=1e-6), "состояние буфера = состояние оценки"
                assert np.array_equal(data["S2"][rows[j - 1]], data["S"][rows[j]])
            elif data["D"][rows[-1]] == 0:
                assert np.allclose(data["S2"][rows[-1]], want, atol=1e-6), "следующее состояние хвоста — настоящее"
    tab = S.build_table(episodes={SUB: eps}, verbose=False)
    assert len(tab["A"]) == len(data["A"]) and len(np.unique(tab["CH"])) == 6 and tab["NAMES"] == [PDE]
    assert tab["A"].tolist() == data["A"].tolist() and int(tab["LAST"].sum()) == 6
    assert np.allclose(tab["LN"], np.log10(data["ERR"].astype(np.float64)), atol=1e-6)
    assert np.isfinite(tab["LVN"][~tab["LAST"]]).all() and np.isnan(tab["LVN"][tab["LAST"]]).sum() == 6
    new = TB.target_rows(eps, PDE)
    cut = np.where(new["EPA"] != S.EPOCHS[new["A"]])[0]
    assert len(cut) == 1 and new["EPA"][cut[0]] == 2000 and new["A"][cut[0]] == 8, "фактические эпохи обрезанного действия"
    grp = [int(new["GRP"][new["CH"] == c][0]) for c in range(6)]
    assert grp[0] == grp[1] == grp[2] == 0 and grp[3:] == [3, 4, 5], "цепочки общего префикса — одна группа"
    print("загрузчики: episodes_to_arrays_chains, build_state оценки, surrogate.build_table, EPA и GRP: OK")
    return eps


def test_upload_and_workers(tmp):
    """Один коммит на раунд со всеми новыми файлами; сборщики k/K не пересекаются."""
    ups, order = [], []
    C.upload_files = lambda pairs, msg, tries=9: (ups.append(list(pairs)), order.append("buf"), True)[2]
    C.upload = lambda st, name: order.append("state")
    st = collect(tmp, "up", ["--rounds", "2", "--batch", "2", "--progress-every", "1e9"], smoke=False)
    assert len(st["records"]) == 4 and [len(p) for p in ups] == [2, 2], "по одному коммиту на раунд"
    assert order == ["buf", "state", "buf", "state", "state"], "буфер заливается до состояния"
    assert all(dst == f"rl_arch/buffers_new/up/{SUB}/{os.path.basename(src)}" and os.path.exists(src)
               for p in ups for src, dst in p)
    assert all(r["buf_up"] for r in st["records"]) and st["buffer"]["uploaded"] == 4 == st["buffer"]["files"]
    # заливка не удалась: файлы остаются на следующий раз; локальный файл пропал — цепочка помечена
    C.upload_files = lambda pairs, msg, tries=9: False
    st = collect(tmp, "fail", ["--rounds", "1", "--batch", "2"], smoke=False)
    assert not any(r["buf_up"] for r in st["records"]) and st["buffer"]["uploaded"] == 0
    os.remove(os.path.join(st["buffer"]["dir"], st["records"][0]["buf_file"]))
    with contextlib.redirect_stdout(io.StringIO()):
        C.flush_buffer(st, C.argparse.Namespace(buffer_dir="", save_dir=tmp, tag="fail", pde=PDE))
    assert st["records"][0].get("buf_lost") and st["buffer"]["lost"] == 1 and not st["records"][1].get("buf_lost")
    # сборщики
    w = [collect(tmp, f"k{k}", ["--rounds", "2", "--batch", "3", "--worker", f"{k}/3"]) for k in range(3)]
    keys = [{C.key_of(*C.eff(r["actions"], 7000, 12)) for r in s["records"]} for s in w]
    seeds = [{r["seed"] for r in s["records"]} for s in w]
    for i in range(3):
        assert len(keys[i]) == 6 and w[i]["settings"]["collect_seed"] == i and w[i]["settings"]["worker"] == f"{i}/3"
        assert all(s % 3 == (1000 + i) % 3 for s in seeds[i]), "свой срез пула сидов"
        for j in range(i):
            assert not keys[i] & keys[j] and not seeds[i] & seeds[j], "сборщики не пересекаются"
    names = [f for k in range(3) for f in files_of(tmp, f"k{k}")[1]]
    assert len(set(names)) == 18, "имена файлов разных сборщиков не совпадают"
    for bad in ("3/3", "x", "1"):
        TS.must_fail(lambda: C.parse_worker(bad), SystemExit, f"--worker {bad}")
    # без флага — прежний путь: среда без карт, новых полей в записях нет
    saved = C.run_seed
    C.run_seed = TC.fake_run_seed       # он проверяет args.no_state
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            st = C.main(["--pde", PDE, "--smoke", "--budget", "7000", "--save-dir", tmp, "--tag", "plain",
                         "--acq", "random", "--rounds", "1", "--batch", "2"])
    finally:
        C.run_seed = saved
    assert "buffer" not in st and all(set(r) == TC.REC_KEYS for r in st["records"])
    assert not os.path.exists(os.path.join(tmp, "buffers_new", "plain"))
    print("заливка одним коммитом на раунд, сборщики k/K, прежний путь без флага: OK")


def run_main(av):
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


def teacher(tmp, tag, extra, steps="12"):
    argv = ["--pde", PDE, "--mbpo-h", "2", "--mopo-lam", "0.5", "--mask", "long", "--steps", steps, "--batch", "16",
            "--n-branch", "20", "--regen", "6", "--tgt-every", "5", "--offline-eval", "none", "--surr-quick",
            "--members", "2", "--no-upload", "--tag", tag, "--work", os.path.join(tmp, "work"),
            "--out-dir", tmp] + list(extra)
    args = AC.build_parser().parse_args(argv)
    args._argv = argv
    with contextlib.redirect_stdout(io.StringIO()):
        return AC.run(args, log=TS.QUIET)


def test_trainers(tmp, d0):
    """--target-buffer: исходный буфер цели не читается, таблица у учителя и ученика одна."""
    collect(tmp, "w1", ["--rounds", "2", "--batch", "3", "--worker", "1/2"])
    d1, f1 = files_of(tmp, "w1")
    spec = f"{os.path.join(tmp, 'buffers_new', 'w0')},{d1}"         # корень сборщика и сама папка задачи
    eps = O.load_episodes(d0, SUB) + O.load_episodes(d1, SUB)
    n_ch, n_tr = len(eps), sum(len(e) for e in eps)
    assert n_ch == 12
    # исходные буферы: цель и вторая задача; чтение буфера цели — ошибка теста
    bufs = TS.make_buffers()
    root = os.path.join(tmp, "bufs")
    for sd, fs in bufs.items():
        os.makedirs(os.path.join(root, sd))
        for i, ep in enumerate(fs):
            torch.save(ep, os.path.join(root, sd, f"f{i}.pt"))
    n_orig = sum(len(e) for e in bufs[SUB])
    work = os.path.join(tmp, "work")
    seen = []
    old = (O.load_episodes, os.environ.get("RL_BUFFER_DIR"), sys.argv, MB.fit_model, O.GAMMA)

    def spy(data_dir, subdir=O.SUBDIR, max_files=0):
        seen.append((data_dir, subdir))
        assert not (subdir == SUB and (data_dir is None or os.path.abspath(data_dir).startswith(root))), \
            "исходный буфер цели читать нельзя"
        return old[0](data_dir, subdir, max_files=max_files)

    O.load_episodes = spy
    os.environ["RL_BUFFER_DIR"] = root
    MB.fit_model = lambda tab, members, work, quick, log: old[3](tab, members, work, True, log)
    e0 = float(O.pde_meta()[PDE]["init_err"])
    mtag = f"tbuf{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    try:
        subs = f"{SUB2},{SUB}"
        rep = teacher(tmp, "tb_teach", ["--subdirs", subs, "--target-buffer", spec])
        sz, tb = rep["sizes"], rep["target_buffer"]
        assert sz["target_chains"] == n_ch == tb["table_chains"] == tb["chains"] == tb["files"]
        assert sz["target_rows"] == n_tr == tb["table_rows"] == tb["transitions"] and sz["target_rows_buffer"] == 0
        assert len(tb["sources"]) == 2 and tb["envs"] == {"keep_safe": 6, "reset": 6} and tb["budgets"] == {"7000": 12}
        ck = torch.load(rep["_paths"]["ckpt"], map_location="cpu", weights_only=False)
        assert ck["meta"]["target_buffer"]["chains"] == n_ch and ck["meta"]["combo"]["target_buffer"] == spec
        assert rep["verify"]["ok"]
        assert (None, SUB2) in seen and all(dd in (d0, d1) for dd, sd in seen if sd == SUB)
        # кэш: таблица остальных задач под своим именем, без строк цели; стандартного имени нет
        cache = [f for f in os.listdir(work) if f.startswith("surrogate_table")]
        assert len(cache) == 1 and cache[0] != "surrogate_table.npz"
        assert S.load_table(os.path.join(work, cache[0]))["NAMES"] == [PDE2]
        # таблица учителя — та же, что соберёт ученик; строки цели в ней — только собранные
        tab = TB.surrogate_table(eps, PDE, subs, "", work, log=TS.QUIET)
        assert AC.Models(tab, work, members=2, quick=True, log=TS.QUIET).sig == rep["surrogate"]["data_sig"]
        pi = tab["NAMES"].index(PDE)
        tm = tab["PDE"] == pi
        assert int(tm.sum()) == n_tr and tab["A"][tm].tolist() == [
            t["action"][0] * 9 + t["action"][1]["lr"] * 3 + t["action"][1]["epochs"] for e in eps for t in e]
        assert np.allclose(10.0 ** tab["LN"][tm], [-t["reward"] for e in eps for t in e], rtol=1e-5)
        assert len(np.unique(tab["CH"])) == len(np.unique(tab["CH"][~tm])) + n_ch and tab["CH"][tm].min() > tab["CH"][~tm].max()
        # стандартный учитель на тех же папках: другой отпечаток, в таблице исходный буфер цели
        O.load_episodes = old[0]
        rep0 = teacher(tmp, "tb_std", ["--subdirs", subs])
        O.load_episodes = spy
        assert rep0["sizes"]["target_rows"] == n_orig != n_tr and rep0["target_buffer"] is None
        assert rep0["surrogate"]["data_sig"] != rep["surrogate"]["data_sig"]
        assert "target_buffer" not in torch.load(rep0["_paths"]["ckpt"], map_location="cpu", weights_only=False)["meta"]
        # готовая таблица с исходным буфером цели: его строки выбрасываются
        full = S.build_table(episodes=bufs, verbose=False)
        t2, pi2, n2 = TB.table_with(full, eps, PDE)
        assert n2 == n_tr == int((t2["PDE"] == pi2).sum()) and len(t2["A"]) == len(full["A"]) - n_orig + n_tr
        for flag, val in (("--target-chains", ["3"]), ("--extra-only-target", []), ("--target-only", []),
                          ("--extra-data", ["x.json"])):
            TC_must = lambda: teacher(tmp, "tb_bad", ["--subdirs", subs, "--target-buffer", spec, flag] + val)  # noqa: E731
            TS.must_fail(TC_must, SystemExit, f"agent_combo --target-buffer с {flag}")
        TS.must_fail(lambda: TB.load(os.path.join(tmp, "work"), SUB, TS.QUIET), SystemExit, "папка без файлов .pt")

        # ученик на картах: боевой рецепт с --target-buffer
        head = ["offline_rl.py", "--variant", "convnext_dqn", "--subdir", SUB, "--chain-fix", "--reward-form", "delta",
                "--init-err", repr(e0), "--value-bound", "--gamma", "0.99", "--episode-budget", "7000",
                "--ctx-kmax", "70", "--scalar-ctx", "--encoder", "resnet", "--mb-subdirs", subs, "--mb-work", work,
                "--mb-members", "2", "--seeds", "1", "--model-tag", mtag, "--epochs", "1", "--fqe-epochs", "1",
                "--batch-size", "8", "--cpu"]
        base = head + ["--save-model", "--mb-aux", "0.5", "--mb-h", "2", "--mb-lam", "0.5",
                       "--teacher-from", rep["_paths"]["ckpt"], "--teacher-w", "0.5"]
        del seen[:]
        text = run_main(base + ["--target-buffer", spec])
        r = row_of(text)
        assert r["tb_files"] == n_ch == r["tb_used_chains"] and r["tb_used_transitions"] == n_tr == r["tb_transitions"]
        assert f"transitions={n_tr} " in text and "исходный буфер" in text and "'ok': True" in text
        assert r["mb_surr_sig"] == rep["surrogate"]["data_sig"], "суррогат ученика — на таблице учителя"
        assert r["mb_surr_rows"] == sz["rows"] and np.isfinite(r["mb_loss"]) and np.isfinite(r["teacher_loss"])
        assert all(dd in (d0, d1) for dd, sd in seen if sd == SUB) and seen
        meta = E.load_agent(out)[0]._meta
        assert meta["target_buffer"]["used_chains"] == n_ch and len(meta["target_buffer"]["sources"]) == 2
        refuses(base + ["--target-buffer", spec, "--subdirs", subs], "--target-buffer несовместим с --subdirs")
        refuses(base + ["--target-buffer", spec, "--target-chains", "3"], "--target-buffer несовместим с --target-chains")
        refuses(head + ["--target-buffer", spec, "--extra-buffers", "q.pt"], "--target-buffer несовместим с --extra-buffers")
        refuses(head + ["--target-buffer", spec, "--per-pde-files", "2"], "--target-buffer несовместим с --per-pde-files")
        refuses(["offline_rl.py", "--variant", "convnext_dqn", "--subdir", SUB, "--seeds", "1", "--cpu",
                 "--target-buffer", spec], "--target-buffer требует --chain-fix")
        # без флага ученик читает исходный буфер, как раньше
        O.load_episodes = old[0]
        text0 = run_main(head + ["--save-model"])
        r0 = row_of(text0)
        n0 = len(O.episodes_to_arrays(bufs[SUB], chain_fix=True, reward_form="delta", init_err=e0, budget=7000,
                                      verbose=False)["A"])
        assert not [k for k in r0 if k.startswith("tb_")] and "target_buffer" not in E.load_agent(out)[0]._meta
        assert f"transitions={n0} " in text0 and "собранный буфер" not in text0
    finally:
        O.load_episodes, sys.argv, MB.fit_model, O.GAMMA = old[0], old[2], old[3], old[4]
        if old[1] is None:
            os.environ.pop("RL_BUFFER_DIR", None)
        else:
            os.environ["RL_BUFFER_DIR"] = old[1]
        if os.path.exists(out):
            os.remove(out)
    print("--target-buffer: учитель и ученик без исходного буфера цели, общая таблица, отказы, прежний путь: OK")
    return root, work


def test_smoke_size(tmp, root, work):
    """Размер смоука кернела: 2 цепочки по 2 действия с общим первым -> оба тренера проходят."""
    st = collect(tmp, "sm", ["--rounds", "1", "--batch", "2", "--share-prefix", "--share-min", "100", "--keep-opt",
                             "--keep-opt-mode", "safe", "--init", "Adam:0.001:100,LBFGS:1:100;Adam:0.001:100,Adam:0.0001:100"],
                 budget="300")
    assert [r["buf_n"] for r in st["records"]] == [2, 2] and [c["script"] for c in CALLS[-3:]][0] == "Adam:0.001:100"
    spec = os.path.join(tmp, "buffers_new", "sm")
    old = (os.environ.get("RL_BUFFER_DIR"), sys.argv, O.GAMMA)
    os.environ["RL_BUFFER_DIR"] = root
    mtag = f"tbsm{os.getpid()}"
    out = f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_{mtag}_seed1.pt"
    try:
        rep = teacher(tmp, "sm_teach", ["--subdirs", f"{SUB2},{SUB}", "--target-buffer", spec, "--smoke"], steps="30")
        assert rep["sizes"]["target_chains"] == 2 and rep["sizes"]["target_rows"] == 4 and rep["verify"]["ok"]
        text = run_main(["offline_rl.py", "--variant", "convnext_dqn", "--subdir", SUB, "--target-buffer", spec,
                         "--chain-fix", "--reward-form", "delta", "--init-err", "0.81", "--value-bound", "--gamma", "0.99",
                         "--episode-budget", "7000", "--ctx-kmax", "70", "--save-model", "--scalar-ctx", "--encoder",
                         "resnet", "--mb-aux", "0.5", "--mb-h", "3", "--mb-lam", "0.5", "--mb-subdirs", f"{SUB2},{SUB}",
                         "--mb-work", work, "--mb-members", "2", "--teacher-from", rep["_paths"]["ckpt"],
                         "--teacher-w", "0.5", "--seeds", "1", "--model-tag", mtag, "--epochs", "2",
                         "--fqe-epochs", "2", "--batch-size", "8", "--cpu"])
        r = row_of(text)
        assert r["tb_used_chains"] == 2 and r["tb_used_transitions"] == 4 and "'ok': True" in text
        assert os.path.exists(out)
    finally:
        sys.argv, O.GAMMA = old[1], old[2]
        if old[0] is None:
            os.environ.pop("RL_BUFFER_DIR", None)
        else:
            os.environ["RL_BUFFER_DIR"] = old[0]
        if os.path.exists(out):
            os.remove(out)
    print("размер смоука (2 цепочки, 4 перехода): сбор, учитель, ученик с чекпоинтом: OK")


def main():
    t0 = time.time()
    marks = [t0]

    def lap():
        marks.append(time.time())
        print(f"    ({marks[-1] - marks[-2]:.1f} с)")

    with patched() as tmp:
        d0, recs = test_files(tmp); lap()
        test_loaders(d0, recs); lap()
        test_upload_and_workers(tmp); lap()
        root, work = test_trainers(tmp, d0); lap()
        test_smoke_size(tmp, root, work); lap()
    print(f"test_collect_buffer: всё прошло за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
