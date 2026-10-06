#!/usr/bin/env python
"""
Генератор очереди экспериментов (queue.json) для треков ns2d_liddriven.
Волны 0-3 — треки 1 и 2 (качество агента), волны 10-14 — трек 3 (ответ рецензентам).

Очередь разбита на волны. Каждая задача — один кернел Kaggle: список шагов
(script + args), оценка времени и зависимости (файлы, которые должны уже лежать
в HF-датасете результатов). Секретов здесь нет: токены берёт launch_queue.py из
experiments/chain_eval/accounts.json в момент запуска.

    python experiments/rl_arch/queue/make_queue.py                               # queue.json для ns2d_liddriven
    python experiments/rl_arch/queue/make_queue.py --pde poissonboltzmann2d --prefix pb2d   # queue_pb2d.json

Очередь для другого УрЧП (отборочная задача из PLAN.md) содержит те же армы с той же
нумерацией волн; задачи, привязанные к уже обученным агентам ns2d (лидеры, волны 10 и 14,
дистилляция, страж на лидерах), в неё не входят. Идентификаторы задач и теги обучения
получают префикс, чтобы не пересекаться с ns2d в pushed.json и в HF.

Обоснование каждой задачи — в paper/lit_review/QUEUE.md, порядок запуска — в PLAN.md.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--pde", default="ns2d_liddriven")
_ap.add_argument("--prefix", default="", help="префикс идентификаторов и тегов для не-ns2d очереди")
_ap.add_argument("--out", default=None, help="файл очереди (по умолчанию queue.json или queue_<prefix>.json)")
_ap.add_argument("--mined-files", type=int, default=0,
                 help="сколько файлов буфера читать для лучших цепочек (0 = все; у больших буферов 40)")
_args = _ap.parse_args()
PDE = _args.pde
NS2D = PDE == "ns2d_liddriven"
PREFIX = _args.prefix or ("" if NS2D else PDE.replace("_", "")[:6])
if not NS2D and not PREFIX:
    sys.exit("для не-ns2d очереди нужен --prefix")
OUT_FILE = _args.out or os.path.join(HERE, "queue.json" if NS2D else f"queue_{PREFIX}.json")
with open(os.path.join(HERE, "..", "track3", "pde_meta.json")) as _f:
    PDE_META = json.load(_f)
if PDE not in PDE_META:
    sys.exit(f"нет задачи {PDE} в track3/pde_meta.json")
SUBDIR = PDE_META[PDE]["subdir"]                    # папка буфера этого УрЧП
if not SUBDIR:
    sys.exit(f"у {PDE} нет офлайнового буфера в HF")
BUDGET = 7000
# ошибка необученной сети: для ns2d медиана по буферу (n=121, прежние армы), для остальных —
# медиана по сидам 42-46 из таблицы задач
INIT_ERR = 0.49 if NS2D else round(float(PDE_META[PDE]["init_err"]), 3)
INIT_ERR_P3D = 0.25      # poisson3d_complexgeometry (выборка 28 файлов), тёплый старт трека 2
EVAL = "experiments/rl_arch/online_eval_env.py"
TRAIN = "experiments/rl_arch/online_train_env.py"
OFFLINE = "experiments/rl_arch/offline_rl.py"
Q = "q" if NS2D else f"q{PREFIX}"                   # префикс тегов обучения (имена файлов агентов в HF)
# суффикс имени чекпоинта офлайнового агента — так же, как его строит offline_rl.py
_OFF_DEFAULT_SUBDIR = "poisson_boltzmann_2d"
SUBSFX = "" if SUBDIR == _OFF_DEFAULT_SUBDIR else "_" + SUBDIR.split("_")[0]
PLAIN = " --plain-fnn" if PDE_META[PDE].get("inverse") else ""   # обратные задачи: обычная сеть вместо PFNN

jobs = []


def add(jid, wave, group, steps, hours, needs=(), note=""):
    if PREFIX:
        jid = f"{PREFIX}-{jid}"
    assert len(jid) <= 30 and jid == jid.lower(), jid
    assert all(j["id"] != jid for j in jobs), f"дубликат {jid}"
    jobs.append(dict(id=jid, wave=wave, group=group, hours=hours, needs=list(needs),
                     note=note, steps=[dict(script=s, args=" ".join(a.split())) for s, a in steps]))


def off_model(variant, mtag="", m=1, vb=True, reward="delta"):
    """Путь к чекпоинту офлайнового агента в кернеле (см. offline_rl.py, --save-model)."""
    return (f"/tmp/agent_{variant}_cfix_r{reward}" + ("_vb" if vb else "") + SUBSFX
            + (f"_{mtag}" if mtag else "") + f"_seed{m}.pt")


def mined_chains(k=5):
    """Лучшие цепочки офлайнового буфера этого УрЧП (по итоговой ошибке одного прогона,
    расход до бюджета). Считаются один раз и кэшируются рядом в mined_<папка>.json."""
    cache = os.path.join(HERE, f"mined_{SUBDIR}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)[:k]
    sys.path.insert(0, os.path.join(HERE, ".."))
    import numpy as np
    import offline_rl as O
    import online_eval_env as E
    d = O.episodes_to_arrays(O.load_episodes(None, SUBDIR, max_files=_args.mined_files), chain_fix=True,
                             reward_form="delta",
                             init_err=1.0, budget=BUDGET, verbose=False)
    rows = []
    for e in np.unique(d["EP"]):
        i = np.where(d["EP"] == e)[0]
        acts = [int(a) for a in d["A"][i]]
        rows.append(dict(err=float(d["ERR"][i[-1]]),
                         epochs=int(d["SPENT"][i[-1]] + E.ACTION_TABLE[acts[-1]][2]),
                         script=",".join("%s:%g:%d" % E.ACTION_TABLE[a] for a in acts)))
    rows.sort(key=lambda r: r["err"])
    with open(cache, "w") as f:
        json.dump(rows[:20], f, indent=1)
    return rows[:k]


def seeds(lo, hi):
    return ",".join(str(s) for s in range(lo, hi + 1))


def ev_common(tag, sd, hours=10.5):
    return (f"--pde {PDE}{PLAIN} --budget {BUDGET} --hours {hours} --resume --ckpt-every 10 "
            f"--seeds {sd} --tag {tag}")


# ---------------------------------------------------------------- волна 0
# Опорные уровни среды без обучения агента. Карты ландшафта не строятся
# (--no-state), поэтому сид стоит ~10 минут GPU вместо 1-3 часов.
def baseline(name, policy_args, n_seeds, group="G26", note="", extra="", hours=None):
    chunks = [(42, 51)] if n_seeds == 10 else [(42, 51), (52, 61)]
    for i, (lo, hi) in enumerate(chunks):
        suf = "" if len(chunks) == 1 else "ab"[i]
        add(f"w0-{name}{suf}", 0, group,
            [(EVAL, f"{policy_args} --no-state {extra} " + ev_common(f"bl_{name}", seeds(lo, hi)))],
            hours or 2.5, note=note)


T1_CHAIN = "PSO:0:300,LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500"
if NS2D:
    baseline("t1", f"--policy script --script {T1_CHAIN} --script-tail stop", 20,
             note="цепочка, которую фактически исполняет лидер трека 1; сверка с 0.0632 и 20 сидов")
    baseline("l2k", "--policy script --script LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500 --script-tail stop", 20,
             note="то же без пустого первого шага PSO")
baseline("l1k", "--policy script --script LBFGS:1:1000 --script-tail repeat", 20,
         note="цепочка лидера трека 2: L-BFGS 1.0 весь бюджет")
baseline("l05", "--policy script --script LBFGS:0.5:500 --script-tail repeat", 10)
baseline("l100", "--policy script --script LBFGS:1:100 --script-tail repeat", 10,
         note="короткие всплески L-BFGS: каждое действие заново создаёт оптимизатор")
baseline("a3l", "--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat", 10,
         note="классическая эвристика Adam -> L-BFGS")
baseline("a4l", "--policy script --script Adam:0.0001:2500,LBFGS:0.5:1000 --script-tail repeat", 10)
baseline("rule", "--policy rule --rule-burst LBFGS:1:100 --rule-kick Adam:0.0001:100 --rule-max-kicks 3", 20,
         note="правило «всплеск L-BFGS, при застое лосса толчок Adam» — структура DP-политики")
baseline("rule2", "--policy rule --rule-burst LBFGS:0.5:500 --rule-kick Adam:0.001:100 --rule-max-kicks 3", 10)
baseline("f64", "--policy script --script LBFGS:1:1000 --script-tail repeat", 10, group="G25",
         extra="--float64", hours=6.0,
         note="диагностика уровня среды: снимает ли двойная точность застой L-BFGS")
baseline("f64a", "--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat", 10,
         group="G25", extra="--float64", hours=6.0)
# двойная точность по одному сиду на задачу (те же теги bl_f64, bl_f64a): сид в двойной точности
# на T4 стоит 3.5-6 часов, и задача на 10 сидов успевала за сессию Kaggle (12 ч) только 2-3 сида.
# На pb2d двойная точность сняла застой L-BFGS: 0.0061 против 0.0250 (3 сида, 3 октября)
for sd_ in range(42, 52):
    for name, sc in (("f64", "LBFGS:1:1000"), ("f64a", "Adam:0.001:1000,LBFGS:1:1000")):
        add(f"w0-{name}-s{sd_}", 0, "G25",
            [(EVAL, f"--policy script --script {sc} --script-tail repeat --no-state --float64 "
                    + ev_common(f"bl_{name}", str(sd_)))],
            7.0, note=f"двойная точность, сид {sd_}: {sc}")

# среда без сбросов оптимизатора (--keep-opt): действие продолжает моменты Adam и историю
# кривизны L-BFGS прошлого действия того же семейства. Прецедент — перенос состояния при
# переключении в AOS (arXiv 2608.01997). Без флага каждое действие создаёт оптимизатор заново,
# поэтому «L-BFGS весь бюджет» и «Adam, затем L-BFGS» до сих пор шли со сбросом каждые 1000
# эпох; эти армы — те же цепочки в обычной практике PINN, какой её ждёт рецензент
KEEP = "--keep-opt"
# часы больше, чем у пар со сбросом: если L-BFGS без сброса не встаёт, все 7000 эпох идут в
# полную цену, а у пар со сбросом больше половины бюджета — быстрые пустые эпохи
baseline("l1kk", "--policy script --script LBFGS:1:1000 --script-tail repeat", 10, group="H12", extra=KEEP,
         hours=8.0, note="L-BFGS 1.0 весь бюджет без сбросов истории кривизны; пара к w0-l1k")
baseline("a3lk", "--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat", 10,
         group="H12", extra=KEEP, hours=8.0, note="классическая пара Adam -> L-BFGS без сбросов; пара к w0-a3l")
baseline("rulek", "--policy rule --rule-burst LBFGS:1:100 --rule-kick Adam:0.0001:100 --rule-max-kicks 3", 10,
         group="H12", extra=KEEP, hours=8.0,
         note="правило «всплеск L-BFGS, при застое толчок Adam» без сбросов между всплесками; пара к w0-rule")
if NS2D:
    baseline("t1k", f"--policy script --script {T1_CHAIN} --script-tail stop", 10, group="H12", extra=KEEP,
             hours=4.0, note="цепочка лидера трека 1 в среде без сбросов; пара к w0-t1")

# замороженная оценка двух обучающих лидеров (прежний протокол, 5 сидов); агенты обучены на ns2d
for short, tag in ((("lead1", "zero_q1_rean_distill_pbrs_seed42"), ("lead2", "comb_q1_distill_pbrs_seed42"))
                   if NS2D else ()):
    for suf, sd in (("a", "42,43,44"), ("b", "45,46")):
        add(f"w0-{short}{suf}", 0, "G01",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"ev_{short}", sd))],
            10.5, needs=[f"rl_arch/agents_online/{tag}.pt"],
            note="лидер по обучающей медиане, замороженная политика ещё не оценена")
# лидеры треков на сидах 47-51: база для парных сравнений на 10 сидах
if NS2D:
  add("w0-t1more", 0, "G22",
    [(EVAL, "--policy agent --model-file rl_arch/agents_online/convnext_dqn_ns2d_rlpdtol_seed42.pt "
            "--stop-on-noop " + ev_common("ns2drlpdtol_sn", seeds(42, 51)))],
    8.0, needs=["rl_arch/agents_online/convnext_dqn_ns2d_rlpdtol_seed42.pt"],
    note="лидер трека 1 на 10 сидах со «стоп на пустом действии»; сиды 42-46 сверяются со старыми")
  add("w0-t2more", 0, "G22",
    [(EVAL, "--policy agent --model-file rl_arch/agents_online/convnext_dqn_ns2d_wsrlreset_seed42.pt "
            + ev_common("onreset", seeds(47, 51)))],
    6.0, needs=["rl_arch/agents_online/convnext_dqn_ns2d_wsrlreset_seed42.pt"],
    note="лидер трека 2: досчитать сиды 47-51")

# ---------------------------------------------------------------- волна 1
# Исправленный конвейер против прежнего. Два обучающих сида на арм: до сих пор
# все прогоны шли с одним сидом 42, разброс между обучениями неизвестен.
T1 = (f"--variant convnext_dqn --pde {PDE}{PLAIN} --hours 11 --save-agent --save-every 5 --save-buffer "
      f"--rlpd --rlpd-utd 8 --rlpd-subdir {SUBDIR}")
FIXD_NB = f"--offline-fix --offline-reward delta --init-err {INIT_ERR}"
# границы ценности входят в основной пакет: без них офлайновый DQN на награде-разности
# завышал Q(s0) втрое выше физического предела (локальная проверка на буфере)
FIXD = f"{FIXD_NB} --value-bound"
ALIGN = f"{FIXD} --gamma 0.99 --episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70"
ALIGN_NB = f"{FIXD_NB} --gamma 0.99 --episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70"
MASK = "--mask adam:0.01,pso:0.001,pso:0.0001"
T2 = (f"--variant convnext_dqn --pde {PDE}{PLAIN} --hours 11 --save-agent --save-every 5 --save-buffer "
      f"--wsrl-warmup 20 --rlpd --rlpd-utd 8 --self-prior 60 --reset-every 200")
P3D_MODEL = "/tmp/agent_convnext_dqn_cfix_rdelta_vb_poisson3d_seed1.pt"
P3D_STEP = (OFFLINE, f"--variant convnext_dqn --subdir poisson3d_complexgeometry --seeds 1 --chain-fix "
                     f"--reward-form delta --init-err {INIT_ERR_P3D} --value-bound --gamma 0.99 --save-model")

arms = {
    # арм: (группа, шаги до обучения, аргументы обучения, пояснение)
    "t1ref": ("base", [], f"{T1} --max-chain-steps 10 --tolerance 0.01",
              "контроль: конфигурация лидера трека 1 без изменений"),
    "t1fixd": ("G04", [], f"{T1} --max-chain-steps 10 --tolerance 0.01 {FIXD}",
               "исправленный офлайн-буфер, единая награда-разность, награда первого шага"),
    "t1al": ("G03", [], f"{T1} {ALIGN}",
             "то же + эпизод по бюджету 7000 эпох как на оценке, без допуска, дисконт 0.99"),
    "t1alnb": ("G27", [], f"{T1} {ALIGN_NB}",
               "абляция: согласованный конвейер без границ ценности"),
    "t1alc": ("G06", [], f"{T1} {ALIGN} --scalar-ctx --ctx-no-err",
              "то же + контекст без истинной ошибки: шаг, доля бюджета, прошлое действие"),
    "t1alm": ("G02", [], f"{T1} {ALIGN} {MASK}",
              "то же + запрет заведомо вредных и пустых действий (Adam 1e-2, PSO с шагом)"),
    "t2ref": ("base", [], f"{T2} --warm-start convnext_dqn_poisson3d_seed1.pt "
                          f"--max-chain-steps 10 --tolerance 0.01",
              "контроль: конфигурация лидера трека 2 без изменений"),
    "t2al": ("G03", [P3D_STEP], f"{T2} --warm-start {P3D_MODEL} --init-err {INIT_ERR} --value-bound "
                                f"--gamma 0.99 --episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70",
             "трек 2: тёплый старт переобучен на исправленном буфере, эпизод по бюджету"),
}
train_seeds = {"t1ref": (42, 43), "t1fixd": (42, 43), "t1al": (42, 43), "t1alnb": (42,), "t1alc": (42,),
               "t1alm": (42,), "t2ref": (42, 43), "t2al": (42, 43)}
for arm, (group, pre, targs, note) in arms.items():
    for s in train_seeds[arm]:
        tag = f"{Q}_{arm}_s{s}"
        add(f"w1-{arm}-s{s}", 1, group, pre + [(TRAIN, f"{targs} --seed {s} --tag {tag}")],
            11.8, note=note)
        for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
            add(f"w1-{arm}-s{s}-e{suf}", 1, group,
                [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt --stop-on-noop "
                        + ev_common(f"e_{tag}", sd))],
                10.5, needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка: " + note)
        for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
            add(f"w1-{arm}-s{s}-n{suf}", 1, group,
                [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                        + ev_common(f"n_{tag}", sd))],
                10.5, needs=[f"rl_arch/agents_online/{tag}.pt"],
                note="оценка без остановки на пустом действии: " + note)

# Оценка без остановки на пустом действии (без --stop-on-noop). Агенты этого проекта учились на
# эпизоде по бюджету: в обучении PSO с нулевым шагом не заканчивал эпизод, а это случайный поиск
# вокруг весов, после которого агент продолжал. Остановка на нём на оценке обрывала цепочку
# (pb2d, 3 октября: t3on_shuf — 1.02 на всех сидах после одного действия на 100 эпох). Прежние
# задачи с остановкой оставлены как были (их строки уже сняты); остановка сохранена только у
# лидеров треков 1-2 и агента qr, чтобы совпасть с их прежними оценками и вмешательствами.
# чисто офлайновый режим на исправленном буфере: обучение минуты, затем оценка
OF_FIX = (f"--variant convnext_dqn --subdir {SUBDIR} --chain-fix --reward-form delta "
          f"--init-err {INIT_ERR} --value-bound --gamma 0.99 --episode-budget {BUDGET} --save-model")
OF_MODEL = off_model("convnext_dqn", m="{m}")
for m in (1, 2, 3):
    add(f"w1-offix-m{m}", 1, "G04",
        [(OFFLINE, f"{OF_FIX} --seeds {m}"),
         (EVAL, f"--policy agent --model-file {OF_MODEL.format(m=m)} --stop-on-noop "
                + ev_common(f"of_fixd_m{m}", seeds(42, 51) if m == 1 else seeds(42, 46)))],
        11.0, note="чистый офлайн на исправленном буфере (было 0.0774 на прежнем)")
add("w1-ofstat", 1, "G06",
    [(OFFLINE, OF_FIX.replace("--variant convnext_dqn", "--variant stat_dqn") + " --seeds 1"),
     (EVAL, f"--policy agent --model-file {off_model('stat_dqn')} --stop-on-noop "
            + ev_common("of_stat_m1", seeds(42, 51)))],
    11.0, note="офлайн: кодировщик на статистиках карт (24 тыс. параметров вместо 250 тыс.)")
add("w1-offix-mask", 1, "G02",
    [(OFFLINE, f"{OF_FIX} --seeds 1"),
     (EVAL, f"--policy agent --model-file {OF_MODEL.format(m=1)} --stop-on-noop "
            "--mask adam:0.01,pso:0.001,pso:0.0001 " + ev_common("of_fixd_m1_mask", seeds(42, 51)))],
    11.0, note="тот же офлайновый агент, на оценке запрещены Adam 1e-2 и PSO с шагом")
add("w1-offixn-m1", 1, "G04",
    [(OFFLINE, f"{OF_FIX} --seeds 1 --model-tag n"),
     (EVAL, f"--policy agent --model-file {off_model('convnext_dqn', 'n')} "
            + ev_common("ofn_fixd_m1", seeds(42, 51)))],
    11.0, note="чистый офлайн на исправленном буфере, оценка без остановки на пустом действии")
add("w1-offixn-mask", 1, "G02",
    [(OFFLINE, f"{OF_FIX} --seeds 1 --model-tag n"),
     (EVAL, f"--policy agent --model-file {off_model('convnext_dqn', 'n')} "
            "--mask adam:0.01,pso:0.001,pso:0.0001 " + ev_common("ofn_fixd_m1_mask", seeds(42, 51)))],
    11.0, note="то же с маской на оценке, без остановки на пустом действии")
add("w1-ofstatn", 1, "G06",
    [(OFFLINE, OF_FIX.replace("--variant convnext_dqn", "--variant stat_dqn") + " --seeds 1 --model-tag n"),
     (EVAL, f"--policy agent --model-file {off_model('stat_dqn', 'n')} "
            + ev_common("ofn_stat_m1", seeds(42, 51)))],
    11.0, note="кодировщик на статистиках карт, без остановки на пустом действии")
if NS2D:
  add("w1-ofref-more", 1, "base",
    [(EVAL, "--policy agent --model-file rl_arch/models/convnext_dqn_ns2d_seed1.pt "
            + ev_common("ofdqn", seeds(47, 51)))],
    8.0, needs=["rl_arch/models/convnext_dqn_ns2d_seed1.pt"],
    note="прежний офлайновый агент: досчитать сиды 47-51 для парного сравнения")

# ---------------------------------------------------------------- волна 2
# Методы из статей поверх лучшей конфигурации волны 1. {BASE} и {GUIDE}
# подставляет launch_queue.py из decisions.json после разбора волн 0 и 1.
W2 = {
    "jsrl": ("G02", "{BASE} --guide {GUIDE} --guide-mode jsrl --guide-chains 12",
             "Jump-Start RL: начало цепочки ведёт лучшая открытая цепочка волны 0"),
    "bonus": ("G02", "{BASE} --guide {GUIDE} --guide-mode bonus --guide-bonus 0.05 --guide-chains 20",
              "отход от эвристики только при выигрыше по Q (аналог ограничения HEPO)"),
    "geps": ("G14", "{BASE} --guide {GUIDE} --guide-mode eps",
             "разведка по проводнику вместо равномерной (BEQ)"),
    "nstep": ("G15", "{BASE} --n-step 3", "трёхшаговые цели"),
    "utd2": ("G09", "{BASE} --rlpd-utd 2", "UTD 2 вместо 8"),
    "utd20": ("G09", "{BASE} --rlpd-utd 20", "UTD 20 вместо 8"),
    "g1": ("G04", "{BASE} --gamma 1.0", "дисконт 1.0: возврат равен итоговому улучшению"),
    # бутстреп-головы и ансамбль критиков несовместимы с границами ценности (цель считается
    # по другой формуле), поэтому оба арма идут от базы без --value-bound
    "boot": ("G14", "{BASE_NOVB} --boot-heads 8", "бутстреп-головы для разведки"),
    "full": ("G12", "{BASE_NOVB} --rlpd-full", "ансамбль из 10 критиков с LayerNorm (полный RLPD)"),
    "hlg": ("G18", "{BASE_QR} --hl-gauss", "HL-Gauss поверх исправленного конвейера"),
    "fact": ("G16", "{BASE_FACT}", "голова Q = V + q_o + q_ol + q_oe: слагаемые по оптимизатору, шагу, длительности"),
    "stat": ("G06", "{BASE_STAT}", "кодировщик на сводных статистиках карт вместо ConvNeXt"),
    "frz": ("G17", "{BASE} --freeze-encoder", "замороженный кодировщик, учится только голова"),
    # офлайновая половина батча при этом берётся только из переходов, согласованных со средой
    # без сбросов (первый шаг цепочки, смена семейства оптимизатора, PSO); оценка включает
    # --keep-opt сама — по записи в чекпоинте
    "keep": ("H12", "{BASE} --keep-opt", "среда без сбросов оптимизатора между действиями одного семейства"),
}
for name, (group, targs, note) in W2.items():
    tag = f"{Q}2_{name}_s42"
    add(f"w2-{name}-s42", 2, group, [(TRAIN, f"{targs} --seed 42 --tag {tag}")], 11.8, note=note)
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w2-{name}-s42-e{suf}", 2, group,
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"e_{tag}", sd))],
            10.5, needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка: " + note)
# без нового обучения: комитет, откат к эвристике при оценке, офлайновое переобучение
add("w2-vote-ea", 2, "G12",
    [(EVAL, "--policy agent --model-files {COMMITTEE} --ensemble vote "
            + ev_common("e_vote", seeds(42, 46)))], 10.5,
    note="комитет замороженных агентов (голосование)")
add("w2-vote-eb", 2, "G12",
    [(EVAL, "--policy agent --model-files {COMMITTEE} --ensemble vote "
            + ev_common("e_vote", seeds(47, 51)))], 10.5)
add("w2-gbon-ea", 2, "G22",
    [(EVAL, "--policy agent --model-file {BEST_AGENT} --guide {GUIDE} --guide-bonus 0.02 "
            "" + ev_common("e_gbon", seeds(42, 46)))], 10.5,
    note="при оценке агент отходит от эвристики только при выигрыше по Q больше 0.02")
add("w2-gbon-eb", 2, "G22",
    [(EVAL, "--policy agent --model-file {BEST_AGENT} --guide {GUIDE} --guide-bonus 0.02 "
            "" + ev_common("e_gbon", seeds(47, 51)))], 10.5)
add("w2-ooo", 2, "G12",
    [(OFFLINE, f"{OF_FIX} --seeds 1 --extra-buffers {{BUFFERS}} --model-tag ooo"),
     (EVAL, f"--policy agent --model-file {off_model('convnext_dqn', 'ooo')} "
            "" + ev_common("e_ooo", seeds(42, 51)))], 11.0,
    note="OOO: итоговая политика переобучается офлайн на буфере + всех онлайновых данных волны 1")

# ---------------------------------------------------------------- волна 3
# Методы, уже проверенные на прежнем конвейере: их вердикты получены на буфере с
# копиями состояний и смешанной наградой и после исправления не обязаны сохраниться.
W3 = {
    "distpbrs": ("G02", "{BASE_NOVB} --distill 1 --distill-top 3 --pbrs 1.0",
                 "дистилляция лучших цепочек + PBRS (лидер по обучающей медиане)"),
    "rean": ("G23", "{BASE} --vem --reanalyse 5", "модель по ценности + Reanalyse"),
    "per": ("G13", "{BASE} --per", "приоритетный реплей"),
    "sil": ("G27", "{BASE} --sil", "self-imitation"),
    "goexp": ("G10", "{BASE} --go-explore", "Go-Explore"),
    "reset": ("G24", "{BASE} --reset-every 200", "периодические сбросы"),
    "redo": ("G24", "{BASE} --redo-every 200", "ReDo"),
    "aug": ("G29", "{BASE} --aug", "диэдральные аугментации"),
    "spr": ("G28", "{BASE} --spr 1", "SPR"),
    "munch": ("G18", "{BASE_NOVB} --munchausen", "Munchausen"),
}
for name, (group, targs, note) in W3.items():
    tag = f"{Q}3_{name}_s42"
    add(f"w3-{name}-s42", 3, group, [(TRAIN, f"{targs} --seed 42 --tag {tag}")], 11.8,
        note="повтор на исправленном конвейере: " + note)
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w3-{name}-s42-e{suf}", 3, group,
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"e_{tag}", sd))],
            10.5, needs=[f"rl_arch/agents_online/{tag}.pt"])

# ---------------------------------------------------------------- волны 4 и 5
# Сборка победителей. Волна 4 — подтверждение объединённой базы после волны 1: все флаги,
# давшие выигрыш по отдельности, собираются в {BASE}; второй обучающий сид и сиды 52-61
# для 20-сидовой оценки. Волна 5 — итоговая конфигурация {FINAL} = {BASE} плюс победители
# волн 2, 3, 12 и 13; два обучающих сида, 20 сидов оценки, оценка со стражем.
for s_ in (42, 43):
    tag = f"{Q}4_base_s{s_}"
    add(f"w4-base-s{s_}", 4, "base", [(TRAIN, f"{{BASE}} --seed {s_} --tag {tag}")], 11.8,
        note="объединённая база волны 1: все флаги с индивидуальным выигрышем вместе")
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w4-base-s{s_}-e{suf}", 4, "base",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"e_{tag}", sd))], 10.5,
            needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка объединённой базы")
add("w4-base-s42-ec", 4, "base",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{Q}4_base_s42.pt "
            + ev_common(f"e_{Q}4_base_s42", seeds(52, 61)))], 10.5,
    needs=[f"rl_arch/agents_online/{Q}4_base_s42.pt"], note="сиды 52-61: двадцать сидов у финалиста волны 1")
# слоты комбинаций: до трёх наборов флагов обучения и до двух наборов флагов оценки, собранных
# из победителей по правилам PLAN.md (раздел 3б). Готовые наборы, прошедшие проверку
# совместимости, лежат в ключе combos очереди; победитель слота входит в {FINAL}
for c_ in "abc":
    tag = f"{Q}4_c{c_}_s42"
    add(f"w4-c{c_}-s42", 4, "base", [(TRAIN, f"{{COMBO_{c_.upper()}}} --seed 42 --tag {tag}")], 11.8,
        note=f"комбинация {c_.upper()}: набор флагов обучения из decisions.json (COMBO_{c_.upper()})")
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w4-c{c_}-s42-e{suf}", 4, "base",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"e_{tag}", sd))], 10.5,
            needs=[f"rl_arch/agents_online/{tag}.pt"], note=f"оценка комбинации {c_.upper()}")
for c_ in "ab":
    add(f"w4-ce{c_}", 4, "base",
        [(EVAL, f"{{COMBO_EVAL_{c_.upper()}}} " + ev_common(f"e_{Q}4_ce{c_}", seeds(42, 51)))], 10.5,
        note=f"комбинация оценки {c_.upper()}: политика и флаги развёртывания из decisions.json "
             f"(COMBO_EVAL_{c_.upper()}), без нового обучения")
for s_ in (42, 43):
    tag = f"{Q}5_final_s{s_}"
    add(f"w5-final-s{s_}", 5, "base", [(TRAIN, f"{{FINAL}} --seed {s_} --tag {tag}")], 11.8,
        note="итоговая конфигурация: база плюс победители волн 2, 3, 12, 13")
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51)), ("c", seeds(52, 56)), ("d", seeds(57, 61))):
        add(f"w5-final-s{s_}-e{suf}", 5, "base",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
                    + ev_common(f"e_{tag}", sd))], 10.5,
            needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка итоговой конфигурации, 20 сидов")
add("w5-final-guard", 5, "H16",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{Q}5_final_s42.pt "
            "--guard-rollback 1.0 --guard-fallback LBFGS:1:500 " + ev_common(f"e_{Q}5_final_guard", seeds(42, 51)))],
    10.5, needs=[f"rl_arch/agents_online/{Q}5_final_s42.pt"], note="итоговая конфигурация со стражем")
add("w5-final-gbon", 5, "H16",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{Q}5_final_s42.pt --guide {{GUIDE}} --guide-bonus 0.02 "
            "" + ev_common(f"e_{Q}5_final_gbon", seeds(42, 51)))],
    10.5, needs=[f"rl_arch/agents_online/{Q}5_final_s42.pt"], note="итоговая конфигурация с порогом по Q для отхода от проводника")
add("w5-final-gg", 5, "H16",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{Q}5_final_s42.pt --guide {{GUIDE}} --guide-bonus 0.02 "
            "--guard-rollback 1.0 --guard-fallback LBFGS:1:500 "
            + ev_common(f"e_{Q}5_final_gg", seeds(42, 51)))],
    10.5, needs=[f"rl_arch/agents_online/{Q}5_final_s42.pt"],
    note="итоговая конфигурация: порог по Q и страж вместе (гарантия «не хуже проводника» с двух сторон)")
add("w5-final-ev3", 5, "H04",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{Q}5_final_s42.pt --state-every 3 "
            + ev_common(f"e_{Q}5_final_ev3", seeds(42, 51)))],
    10.5, needs=[f"rl_arch/agents_online/{Q}5_final_s42.pt"],
    note="итоговая конфигурация с картами после каждого третьего действия; не нужна, если состояние без карт")

# ======================================================================= трек 3
# Ответ на три претензии рецензентов: цена мета-обучения (R1), вклад ландшафтного
# состояния и RL-машинерии не изолирован (R2), выигрыш и перенос малы относительно
# простых альтернатив (R3). Обоснование и ссылки: paper/lit_review/TRACK3.md.
# Волны 10-14 не зависят от решений волн 0-3, кроме отмеченных подстановок.
CHAIN = "experiments/chain_eval/run_chain_pde.py"
LEAD1 = "rl_arch/agents_online/convnext_dqn_ns2d_rlpdtol_seed42.pt"     # трек 1, 0.0632
LEAD2 = "rl_arch/agents_online/convnext_dqn_ns2d_wsrlreset_seed42.pt"   # трек 2, 0.0712
OFQR = "rl_arch/models/cnx_qrdqn_ns2d_seed1.pt"    # лучший агент, чьи действия зависят от карт, 0.0700
CTX = "--scalar-ctx --ctx-no-err"
S5, S10 = seeds(42, 46), seeds(42, 51)


def plain_of(pde):
    return " --plain-fnn" if PDE_META[pde].get("inverse") else ""   # обратные задачи: обычная сеть


def t3ev(tag, sd, extra="", hours=10.5, pde=PDE, budget=BUDGET):
    plain = plain_of(pde)
    return (f"{extra} --pde {pde}{plain} --budget {budget} --hours {hours} --resume --ckpt-every 10 "
            f"--seeds {sd} --tag {tag}")


# ---------------------------------------------------------------- волна 10
# Вмешательства в состояние у готовых агентов: обучение не нужно. Агент получает
# искажённые карты, истинные строятся рядом, в строку пишется доля совпадения действий.
INTERV = {
    "full": ("", "контроль: тот же протокол и новый код (учёт времени, история лосса)"),
    "shuf": ("--state-mode shuffle --state-compare", "пиксели карт перемешаны: пространственной структуры нет"),
    "lvl": ("--state-mode level --state-compare", "каждая карта заменена своим средним: только уровень потерь"),
    "shape": ("--state-mode shape --state-compare", "из карт убран уровень: только форма"),
    "blind": ("--state-mode blind --state-compare", "карты обнулены"),
    "reb": ("--state-mode rebuild --state-compare",
            "вторая сборка карт с другой случайностью автокодировщика: воспроизводимо ли наблюдение"),
    "ev3": ("--state-every 3", "карты строятся после каждого третьего действия: цена наблюдения втрое ниже"),
}
for short, model, which in ((("l1", LEAD1, list(INTERV)), ("qr", OFQR, list(INTERV)),
                             ("l2", LEAD2, ["full", "shuf", "blind"])) if NS2D else ()):
    for name in which:
        flags, note = INTERV[name]
        add(f"w10-{short}-{name}", 10, "H06",
            [(EVAL, f"--policy agent --model-file {model} --stop-on-noop {flags} "
                    + t3ev(f"t3a_{short}_{name}", S5))],
            10.5, needs=[model], note=note)

# ---------------------------------------------------------------- волна 11
# Простые альтернативы без обучения агента. Карты не строятся.
def t3script(jid, tag, script, tail="stop", note="", sd=S10, group="H15", hours=6.0, extra=""):
    add(jid, 11, group,
        [(EVAL, f"--policy script --script {script} --script-tail {tail} --no-state {extra} "
                + t3ev(tag, sd, hours=hours))], hours, note=note)


# случайный поиск статических цепочек при равном бюджете: 24 случайные цепочки без PSO,
# затем победитель переоценивается на сидах оценки
for i, (lo, hi) in enumerate(((1000, 1007), (1008, 1015), (1016, 1023))):
    add(f"w11-rs-{'abc'[i]}", 11, "H15",
        [(EVAL, "--policy random --mask pso --no-state " + t3ev("t3m_rs", seeds(lo, hi), hours=8.0))],
        8.0, note="случайный поиск: 8 случайных цепочек из действий Adam и L-BFGS, сид задаёт цепочку")
t3script("w11-rsl", "t3b_rsl", "{RS_CHAIN_LOSS}", note="победитель случайного поиска по обучающему лоссу")
t3script("w11-rse", "t3b_rse", "{RS_CHAIN_ERR}",
         note="победитель случайного поиска по истинной ошибке (тот же сигнал, что награда агента)")
# цепочки, добытые из офлайнового буфера агента: нулевая добавочная цена мета-обучения
MINED = mined_chains(5)
for i, r in enumerate(MINED, 1):
    t3script(f"w11-log{i}", f"t3b_log{i}", r["script"], group="H07",
             note=f"цепочка №{i} из лучших в офлайновом буфере {SUBDIR} (ошибка {r['err']:.4f} по одному прогону): "
                  f"проверка на 10 сидах")
# каркасы, которые подсказывают лучшие цепочки буфера: Adam с большим шагом, Adam с малым, L-BFGS
t3script("w11-sk1", "t3b_sk1", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000", tail="repeat", group="H07",
         note="каркас лучших цепочек буфера: Adam 1e-2, Adam 1e-4, затем L-BFGS до конца бюджета")
t3script("w11-sk2", "t3b_sk2", "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000", tail="repeat", group="H07",
         note="тот же каркас с коротким первым этапом")
t3script("w11-sk3", "t3b_sk3", "Adam:0.001:2500,LBFGS:1:1000", tail="repeat", group="H07",
         note="Adam 1e-3 на 2500 эпох, затем L-BFGS")
# универсальная цепочка, лучшая по остальным УрЧП (rand6 из csv_random)
RAND6 = ("PSO:0:200,LBFGS:1:100,Adam:0.0001:2500,LBFGS:0.1:500,LBFGS:1:500,LBFGS:1:1000,LBFGS:0.1:1000,"
         "PSO:0:100,PSO:0.0001:300,PSO:0.001:300")
RAND6N = "LBFGS:1:100,Adam:0.0001:2500,LBFGS:0.1:500,LBFGS:1:500,LBFGS:1:1000,LBFGS:0.1:1000"
t3script("w11-rand6", "t3b_rand6", RAND6, group="H14",
         note="универсальная цепочка rand6 в действиях среды: лучшая по остальным 14 УрЧП, выбрана без ns2d")
t3script("w11-rand6n", "t3b_rand6n", RAND6N, tail="repeat", group="H14", note="то же без шагов PSO")
# политики без предобучения: разведка и бандит внутри одного прогона
add("w11-scout", 11, "H07",
    [(EVAL, "--policy scout --no-state " + t3ev("t3b_scout", S10, hours=6.0))], 6.0,
    note="разведка: три коротких пробы из общей точки, победившая продолжается (ROR)")
add("w11-scout2", 11, "H07",
    [(EVAL, "--policy scout --scout-set Adam:0.0001:100,LBFGS:1:100,LBFGS:0.5:100 "
            "--scout-commit Adam:0.0001:1000,LBFGS:1:1000,LBFGS:0.5:1000 --no-state "
            + t3ev("t3b_scout2", S10, hours=6.0))], 6.0,
    note="разведка с длинными исполняемыми действиями")
add("w11-bandit", 11, "H08",
    [(EVAL, "--policy bandit --no-state " + t3ev("t3b_bandit", S10, hours=6.0))], 6.0,
    note="UCB-бандит по пяти действиям, учится внутри прогона по убыли обучающего лосса")
add("w11-bandit2", 11, "H08",
    [(EVAL, "--policy bandit --bandit-arms Adam:0.001:1000,Adam:0.0001:1000,LBFGS:1:500,LBFGS:0.5:500,"
            "LBFGS:1:1000 --bandit-c 0.5 --bandit-discount 0.9 --no-state "
            + t3ev("t3b_bandit2", S10, hours=6.0))], 6.0,
    note="бандит с длинными действиями и слабым забыванием")
# правило с подобранными параметрами: длина всплеска, толчок и порог застоя перебираются по
# малой сетке (прецедент: обучаемые параметры простой эвристики, arXiv 2608.27975). Лучший
# вариант выбирается по обучающему лоссу и переоценивается на сидах 52-61
for name, burst, kick, tol in (("rule3", "LBFGS:1:500", "Adam:0.0001:100", 1e-3),
                               ("rule4", "LBFGS:0.5:1000", "Adam:0.001:100", 1e-3),
                               ("rule5", "LBFGS:1:100", "Adam:0.001:1000", 1e-2),
                               ("rule6", "LBFGS:1:1000", "Adam:0.0001:1000", 1e-2)):
    add(f"w11-{name}", 11, "H07",
        [(EVAL, f"--policy rule --rule-burst {burst} --rule-kick {kick} --rule-max-kicks 3 "
                f"--rule-tol {tol} --no-state " + t3ev(f"t3b_{name}", S10, hours=6.0))], 6.0,
        note=f"правило: всплеск {burst}, толчок {kick}, порог застоя {tol}")
add("w11-rulebest", 11, "H07",
    [(EVAL, "{RULE_BEST} --no-state " + t3ev("t3b_rulebest", seeds(52, 61), hours=6.0))], 6.0,
    note="лучший вариант правила (выбран по обучающему лоссу на сидах 42-51) на новых сидах 52-61: "
         "поправка на проклятие победителя")
# те же простые политики в среде без сбросов оптимизатора (пары к армам выше)
t3script("w11-sk1k", "t3b_sk1k", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000", tail="repeat", group="H12",
         extra=KEEP, hours=8.0, note="каркас лучших цепочек буфера без сбросов: моменты Adam переходят с шага 1e-2 на "
                          "1e-4, история L-BFGS копится до конца бюджета; пара к w11-sk1")
add("w11-scoutk", 11, "H12",
    [(EVAL, f"--policy scout {KEEP} --no-state " + t3ev("t3b_scoutk", S10, hours=8.0))], 8.0,
    note="разведка без сбросов: победившее действие продолжает оптимизатор прошлого победителя того же "
         "семейства (ROR с переносом состояния, как в AOS); пара к w11-scout")
# пары без сбросов для лучших открытых цепочек pb2d (3 октября): без них нельзя посчитать планку
# BEST_SIMPLE среды без сбросов (решение 7 фазы A). На pb2d первые пары дали выигрыш 15-63%:
# второе действие L-BFGS без сброса истории продолжает снижать ошибку, а со сбросом встаёт
for nm, sc in (("sk2", "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000"), ("sk3", "Adam:0.001:2500,LBFGS:1:1000")):
    t3script(f"w11-{nm}k", f"t3b_{nm}k", sc, tail="repeat", group="H12", extra=KEEP, hours=8.0,
             note=f"каркас {nm} без сбросов оптимизатора; пара к w11-{nm}")
for i, r in enumerate(MINED, 1):
    t3script(f"w11-log{i}k", f"t3b_log{i}k", r["script"], group="H12", extra=KEEP, hours=8.0,
             note=f"цепочка №{i} из буфера без сбросов оптимизатора; пара к w11-log{i}")
# безопасный режим (--keep-opt-mode safe): история L-BFGS сохраняется, Adam при смене шага новый.
# На ns2d перенос моментов Adam при смене шага вредил (каркасы хуже, цепочки буфера расходились)
KEEPS = f"{KEEP} --keep-opt-mode safe"
for nm, sc in (("sk1", "Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000"),
               ("sk2", "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000"), ("sk3", "Adam:0.001:2500,LBFGS:1:1000")):
    t3script(f"w11-{nm}s", f"t3b_{nm}s", sc, tail="repeat", group="H12", extra=KEEPS, hours=8.0,
             note=f"каркас {nm} в безопасном режиме сохранения оптимизатора; пара к w11-{nm} и w11-{nm}k")
for i, r in enumerate(MINED, 1):
    t3script(f"w11-log{i}s", f"t3b_log{i}s", r["script"], group="H12", extra=KEEPS, hours=8.0,
             note=f"цепочка №{i} из буфера в безопасном режиме сохранения оптимизатора")
# цепочка, которую исполняет лучший агент (подстановка AGENT_CHAIN из decisions_<префикс>.json):
# если она без агента даёт то же, вклад RL — найденное расписание, а не обратная связь
for nm, ex, note in (("agch", "", "цепочка лучшего агента как открытая"),
                     ("agchk", KEEP, "цепочка лучшего агента как открытая, без сбросов оптимизатора")):
    t3script(f"w11-{nm}", f"t3b_{nm}", "{AGENT_CHAIN}", tail="repeat", group="H02", extra=ex, hours=8.0,
             note=note)
# потолок среды: оптимизатор SOAP в тех же эпохах, сидах и метрике
t3script("w11-soap", "t3b_soap", "SOAP:0.003:1000", tail="repeat", group="H12",
         note="SOAP весь бюджет 7000 эпох")
t3script("w11-asoap", "t3b_asoap", "Adam:0.001:1000,SOAP:0.003:1000", tail="repeat", group="H12",
         note="Adam 1000 эпох, затем SOAP")
t3script("w11-soapl", "t3b_soapl", "SOAP:0.003:2500,SOAP:0.003:2500,LBFGS:1:1000", tail="repeat", group="H12",
         note="SOAP 5000 эпох, затем L-BFGS: взаимодополняемость оптимизаторов")
add("w11-scouts", 11, "H12",
    [(EVAL, "--policy scout --scout-set Adam:0.001:100,LBFGS:1:100,SOAP:0.003:100 "
            "--scout-commit Adam:0.001:1000,LBFGS:1:500,SOAP:0.003:1000 --no-state "
            + t3ev("t3b_scouts", S10, hours=6.0))], 6.0,
    note="разведка с SOAP среди кандидатов: выбор между цепочкой и SOAP без обучения")
t3script("w11-soapk", "t3b_soapk", "SOAP:0.003:1000", tail="repeat", group="H12", extra=KEEP,
         note="SOAP весь бюджет без сброса предобусловливателя: штатный SOAP при равных эпохах; пара к w11-soap")
add("w11-scoutsk", 11, "H12",
    [(EVAL, "--policy scout --scout-set Adam:0.001:100,LBFGS:1:100,SOAP:0.003:100 "
            f"--scout-commit Adam:0.001:1000,LBFGS:1:500,SOAP:0.003:1000 {KEEP} --no-state "
            + t3ev("t3b_scoutsk", S10, hours=8.0))], 8.0,
    note="разведка с SOAP без сбросов: самая сильная политика без обучения, собранная из трёх приёмов "
         "(пробы ROR, портфель с SOAP, перенос состояния AOS)")
add("w11-soap31k", 11, "H12",
    [(CHAIN, f"--pde-name {PDE} --chain-json experiments/chain_eval/chain_soap.json "
             f"--chain-key soap --value-type soap --hf-dir csv_soap --csv-name {PDE} "
             f"--n-seeds 10 --seed-base 42 --save-dir runs_chain_eval/t3_soap")], 11.0,
    note="SOAP в штатном режиме chain_eval (31 000 эпох): сравнение при равном времени, а не эпохах")

# ---------------------------------------------------------------- волна 12
# Лестница состояния с обучением. Офлайн: обучение минуты, затем оценка на 10 сидах.
T3_OF = (f"--subdir {SUBDIR} --chain-fix --reward-form delta --init-err {INIT_ERR} --value-bound "
         f"--gamma 0.99 --episode-budget {BUDGET} --ctx-kmax 70 --save-model")


def t3off(jid, variant, flags, mtag, note, wave=12, group="H06", model_seeds=(1,), sd=S10, hours=11.0,
          stop=True):
    for m in model_seeds:
        suf = "" if len(model_seeds) == 1 else f"-m{m}"
        model = off_model(variant, mtag, m)
        add(f"{jid}{suf}", wave, group,
            [(OFFLINE, f"--variant {variant} {T3_OF} {flags} --seeds {m} --model-tag {mtag}"),
             (EVAL, f"--policy agent --model-file {model} {'--stop-on-noop ' if stop else ''}"
                    + t3ev(f"{'t3s' if stop else 't3sn'}_{mtag}_m{m}", sd if m == 1 else S5))],
            hours, note=note)


def t3offn(jid, variant, flags, mtag, note, **kw):
    """Прежний арм лестницы и его пара без остановки на пустом действии (jid -> jid с 'ofn')."""
    t3off(jid, variant, flags, mtag, note, **kw)
    t3off(jid.replace("-of-", "-ofn-"), variant, flags, mtag + "n",
          note + "; оценка без остановки на пустом действии", stop=False, **kw)


t3offn("w12-of-full", "convnext_dqn", "", "full", "контроль: полные карты, без контекста", model_seeds=(1, 2))
t3offn("w12-of-ctx", "convnext_dqn", "--scalar-ctx", "ctx", "полные карты и контекст времени", model_seeds=(1, 2))
t3offn("w12-of-lvl", "convnext_dqn", "--scalar-ctx --state-mode level", "lvl",
      "только уровень потерь и контекст времени", model_seeds=(1, 2))
t3offn("w12-of-shape", "convnext_dqn", "--scalar-ctx --state-mode shape", "shape",
      "только форма карт (уровень убран) и контекст времени")
t3offn("w12-of-shuf", "convnext_dqn", "--scalar-ctx --state-mode shuffle", "shuf",
      "перемешанные пиксели и контекст времени", model_seeds=(1, 2))
t3offn("w12-of-blind", "convnext_dqn", "--scalar-ctx --state-mode blind", "blind",
      "без карт, только время: политика без обратной связи (выученная статическая цепочка)",
      model_seeds=(1, 2), hours=6.0)
t3offn("w12-of-blinde", "convnext_dqn", "--scalar-ctx --ctx-err --state-mode blind", "blinde",
      "без карт, время и истинная ошибка (привилегированный верхний ориентир дешёвого состояния)", hours=6.0)
t3offn("w12-of-slvl", "stat_dqn", "--scalar-ctx --state-mode level", "slvl",
      "малая сеть на скалярах: уровень потерь и время")
t3offn("w12-of-sblind", "stat_dqn", "--scalar-ctx --state-mode blind", "sblind",
      "малая сеть на скалярах: только время", hours=6.0)
# дистилляция лидера в дешёвое состояние: нужен ли ему ландшафт при развёртывании
for name, variant, flags, note in ((
        ("dfull", "convnext_dqn", "", "контроль процедуры: ученик с полными картами"),
        ("dlvl", "stat_dqn", "--state-mode level", "ученик видит только уровень потерь"),
        ("dshuf", "convnext_dqn", "--state-mode shuffle", "ученик видит перемешанные карты"),
        ("dblind", "stat_dqn", "--scalar-ctx --state-mode blind", "ученик видит только время")) if NS2D else ()):
    model = off_model(variant, name, vb=False)
    add(f"w12-{name}", 12, "H11",
        [(OFFLINE, f"--variant {variant} --subdir {SUBDIR} --chain-fix --reward-form delta --init-err {INIT_ERR} "
                   f"--episode-budget {BUDGET} --ctx-kmax 70 --distill-from {LEAD1} {flags} --seeds 1 "
                   f"--model-tag {name} --save-model"),
         (EVAL, f"--policy agent --model-file {model} --stop-on-noop " + t3ev(f"t3s_{name}", S10))],
        11.0 if "blind" not in name else 6.0, needs=[LEAD1],
        note="дистилляция лидера трека 1: " + note)
# онлайн по протоколу трека 1 (RLPD с офлайн-буфером); опорный арм — w1-t1alc
T3_ON = f"{T1} {ALIGN} {CTX}"
for name, flags, note in (("blind", "--state-mode blind", "без карт: только время"),
                          ("lvl", "--state-mode level", "только уровень потерь"),
                          ("shuf", "--state-mode shuffle", "перемешанные пиксели")):
    tag = f"{Q}_t3on_{name}_s42"
    add(f"w12-on-{name}", 12, "H06", [(TRAIN, f"{T3_ON} {flags} --seed 42 --tag {tag}")], 11.8,
        note="онлайн, протокол трека 1: " + note)
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w12-on-{name}-e{suf}", 12, "H06",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt --stop-on-noop "
                    + t3ev(f"e_{tag}", sd))],
            10.5 if name != "blind" else 4.0, needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка: " + note)
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w12-on-{name}-n{suf}", 12, "H06",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt " + t3ev(f"n_{tag}", sd))],
            10.5, needs=[f"rl_arch/agents_online/{tag}.pt"],
            note="оценка без остановки на пустом действии: " + note)
# онлайн без офлайн-данных и без тёплого старта: дешёвое состояние из обучающих лоссов
T3_ON2 = (f"--pde {PDE}{PLAIN} --hours 11 --save-agent --save-every 5 --save-buffer --rlpd --rlpd-utd 8 "
          f"--self-prior 60 --reset-every 200 --init-err {INIT_ERR} --value-bound --gamma 0.99 "
          f"--episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70 {CTX}")
# мелкий шаг решений: разрешены только действия по 100 эпох (Adam 1e-3 и 1e-4, L-BFGS 1, 0.5, 0.1),
# 70 решений на эпизод. Имеет смысл только без сбросов оптимизатора (иначе короткий L-BFGS теряет
# историю) и только с дешёвым состоянием (70 карт на эпизод стоили бы больше часа)
FINE_MASK = ("adam:0.01,pso,adam:0.001:1000,adam:0.001:2500,adam:0.0001:1000,adam:0.0001:2500,"
             "lbfgs:1:500,lbfgs:1:1000,lbfgs:0.5:500,lbfgs:0.5:1000,lbfgs:0.1:500,lbfgs:0.1:1000")
for name, variant, flags, note in (
        ("full", "convnext_dqn", "", "контроль: полные карты"),
        ("loss", "stat_dqn", "--state-mode loss", "обучающие лоссы вместо карт, карты не строятся"),
        ("blind", "stat_dqn", "--state-mode blind", "только время, карты не строятся"),
        ("fullk", "convnext_dqn", KEEP, "полные карты, среда без сбросов оптимизатора"),
        ("lossk", "stat_dqn", f"--state-mode loss {KEEP}", "обучающие лоссы, среда без сбросов оптимизатора"),
        ("fine", "stat_dqn", f"--state-mode loss {KEEP} --mask {FINE_MASK}",
         "мелкий шаг решений: 70 действий по 100 эпох, лоссы вместо карт, без сбросов оптимизатора "
         "(ближайший аналог политики из arXiv 2609.01811)")):
    tag = f"{Q}_t3on2_{name}_s42"
    add(f"w12-on2-{name}", 12, "H04", [(TRAIN, f"--variant {variant} {T3_ON2} {flags} --seed 42 --tag {tag}")],
        11.8, note="чистый онлайн с нуля: " + note)
    add(f"w12-on2-{name}-e", 12, "H04",
        [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt --stop-on-noop "
                + t3ev(f"e_{tag}", S10))],
        10.5 if name.startswith("full") else 4.0, needs=[f"rl_arch/agents_online/{tag}.pt"],
        note="оценка: " + note)
    add(f"w12-on2-{name}-ka", 12, "H12",
        [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt {KEEP} "
                + t3ev(f"k_{tag}", seeds(42, 46)))],
        10.5, needs=[f"rl_arch/agents_online/{tag}.pt"],
        note="оценка без остановки и без сбросов оптимизатора (агент учился со сбросами): " + note)
    for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
        add(f"w12-on2-{name}-n{suf}", 12, "H04",
            [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt " + t3ev(f"n_{tag}", sd))],
            10.5, needs=[f"rl_arch/agents_online/{tag}.pt"],
            note="оценка без остановки на пустом действии: " + note)

# состояние tele (6 октября; AOS 2608.01997, ExpTest 2411.16975, SWATS 1712.07628): лоссы,
# наклон и t-статистика кривой лосса внутри последнего действия, доля улучшающих эпох, норма
# градиента, смещение весов за действие — сигнал застоя, которого нет в картах. Двойники армов
# loss/lossk/fine, два сида обучения; оценки по 5+5 сидов без остановки на пустом действии.
for name, flags, note in (
        ("tele", "--state-mode tele", "состояние застоя вместо карт"),
        ("telek", f"--state-mode tele {KEEP}", "состояние застоя, среда без сбросов оптимизатора"),
        ("telef", f"--state-mode tele {KEEP} --mask {FINE_MASK}",
         "состояние застоя, мелкий шаг решений по 100 эпох, без сбросов оптимизатора")):
    for s_ in (42, 43):
        tag = f"{Q}_t3on2_{name}_s{s_}"
        add(f"w18-{name}-s{s_}", 18, "H04",
            [(TRAIN, f"--variant stat_dqn {T3_ON2} {flags} --seed {s_} --tag {tag}")],
            11.8, note="чистый онлайн с нуля: " + note)
        for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
            add(f"w18-{name}-s{s_}-n{suf}", 18, "H04",
                [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt --no-state "
                        + t3ev(f"n_{tag}", sd, hours=4.0))],
                4.0, needs=[f"rl_arch/agents_online/{tag}.pt"], note="оценка: " + note)

# модельный агент MBPO (7 октября, work/track3/mbpo_agent.py): stat_dqn на скалярном контексте и
# описателе задачи, обучен на реальных переходах буферов 22 УрЧП и модельных продолжениях длины 5
# со штрафом за неуверенность ансамбля (0.5); на независимой половине данных на 0.087 декады лучше
# лучшей цепочки, без модели — на 0.175 хуже. Только pb2d; проверка 5 сидов в обеих средах
if PDE == "poissonboltzmann2d":
    _mb = "rl_arch/agents_online/poissonboltzmann2d_mbpo.pt"
    for nm, ex in (("mbpo", ""), ("mbpok", KEEPS)):
        add(f"w18-{nm}", 18, "H03",
            [(EVAL, f"--policy agent --model-file {_mb} --no-state --state-mode loss {ex} "
                    + t3ev(f"n_{nm}_h5l05", seeds(42, 46), hours=3.0))],
            3.0, needs=[_mb], note=f"модельный агент MBPO, 5 сидов {'без сбросов' if ex else 'со сбросами'}")

# ---------------------------------------------------------------- развёртывание лидеров
# Лидеры треков 1-2 на ns2d в среде без сбросов оптимизатора и со стражем, без нового обучения:
# в их цепочках идут подряд действия L-BFGS, а на pb2d и ns2d L-BFGS без сброса истории на 20-30%
# лучше. Протокол оценки лидера прежний (с остановкой на пустом действии), чтобы сравнение с
# w0-t1more было парным; простые цепочки той же среды — w0-l1kk, w0-a3lk, w0-t1k
if NS2D:
    for short, model, tagm in (("l1", LEAD1, "ns2drlpdtol"), ("l2", LEAD2, "onreset")):
        for nm, ex, note in (("keep", KEEP, "без сбросов оптимизатора"),
                             ("kg", f"{KEEP} --guard-rollback 1.0 --guard-fallback LBFGS:1:500",
                              "без сбросов и со стражем")):
            add(f"w16-{short}-{nm}", 16, "H16",
                [(EVAL, f"--policy agent --model-file {model} --stop-on-noop {ex} "
                        + ev_common(f"t3d_{short}_{nm}", seeds(42, 51)))],
                10.5, needs=[model], note=f"лидер трека {short[1]} на ns2d {note}")

# ---------------------------------------------------------------- волна 13
# Лестница машинерии, кривая стоимости, страж и сдвиги условий.
t3off("w13-of-g0", "convnext_dqn", "--gamma 0", "g0", stop=False, note=
      "близорукий офлайновый агент (дисконт 0): регрессия награды шага, как классический выбор алгоритма",
      wave=13, group="H08")
# дисконт 0 идёт последним флагом: у argparse побеждает последнее значение --gamma
tag = f"{Q}_t3g0_s42"
add("w13-on-g0", 13, "H08", [(TRAIN, f"{T1} {ALIGN} --gamma 0 --seed 42 --tag {tag}")], 11.8,
    note="близорукий онлайновый агент: контекстный бандит вместо RL")
add("w13-on-g0-e", 13, "H08",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
            + t3ev(f"e_{tag}", S10))], 10.5, needs=[f"rl_arch/agents_online/{tag}.pt"])
# кривая «качество — часы мета-обучения»: 0 (офлайн), 2, 5 и 11 часов (11 — арм w1-t1al)
for h in (2, 5):
    tag = f"{Q}_t3h{h}_s42"
    add(f"w13-h{h}", 13, "H01",
        [(TRAIN, f"{T1.replace('--hours 11', f'--hours {h}')} {ALIGN} --seed 42 --tag {tag}"),
         (EVAL, f"--policy agent --model-file agent_{tag}.pt "
                + t3ev(f"e_{tag}", S5, hours=max(1.0, 10.5 - h)))],
        11.5, note=f"онлайновое обучение {h} ч вместо 11, оценка в том же кернеле")
# кривая «качество — объём офлайнового буфера»: 9 и 28 файлов из 93
for k in (9, 28):
    t3off(f"w13-buf{k}", "convnext_dqn", f"--per-pde-files {k}", f"buf{k}",
          f"офлайновый агент на {k} файлах буфера из 93", wave=13, group="H01", stop=False)
# страж: откат действия агента, поднявшего обучающий лосс (агенты ns2d)
if NS2D:
  add("w13-guard-l1", 13, "H16",
    [(EVAL, f"--policy agent --model-file {LEAD1} --stop-on-noop --guard-rollback 1.0 "
            "--guard-fallback LBFGS:1:500 " + t3ev("t3g_l1", S10))], 10.5, needs=[LEAD1],
    note="лидер трека 1 со стражем: при росте лосса откат и L-BFGS 1×500")
  add("w13-guard-qr", 13, "H16",
    [(EVAL, f"--policy agent --model-file {OFQR} --stop-on-noop --guard-rollback 1.0 "
            "--guard-fallback LBFGS:1:500 " + t3ev("t3g_qr", S10))], 10.5, needs=[OFQR],
    note="агент, зависящий от карт, со стражем")
# сдвиг бюджета и архитектуры: агент против своей модальной цепочки и против L-BFGS
for short, flags in (("b35", "--budget 3500"), ("b140", "--budget 14000"),
                     ("w64", f"--budget {BUDGET} --hidden-layers 64*3"),
                     ("w200", f"--budget {BUDGET} --hidden-layers 200*5")):
    common = f"--pde {PDE}{PLAIN} {flags} --hours 10.5 --resume --ckpt-every 10 --seeds {S5}"
    if NS2D:
        add(f"w13-{short}-ag", 13, "H13",
            [(EVAL, f"--policy agent --model-file {LEAD1} --stop-on-noop {common} --tag t3x_{short}_l1")],
            10.5, needs=[LEAD1], note="лидер трека 1 при сдвиге условий: " + flags)
        add(f"w13-{short}-qr", 13, "H13",
            [(EVAL, f"--policy agent --model-file {OFQR} --stop-on-noop {common} --tag t3x_{short}_qr")],
            10.5, needs=[OFQR], note="агент, зависящий от карт, при сдвиге условий: " + flags)
    add(f"w13-{short}-ch", 13, "H13",
        [(EVAL, f"--policy script --script LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500 --script-tail repeat "
                f"--no-state {common} --tag t3x_{short}_t1c"),
         (EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat --no-state {common} "
                f"--tag t3x_{short}_l1k"),
         (EVAL, f"--policy script --script Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000 --script-tail repeat "
                f"--no-state {common} --tag t3x_{short}_sk1"),
         (EVAL, f"--policy scout --no-state {common} --tag t3x_{short}_scout")],
        8.0, note="статические цепочки и разведка при том же сдвиге: " + flags)
# дешёвая среда: обучение на половинном бюджете эпизода, оценка на полном
tag = f"{Q}_t3lofi_s42"
add("w13-lofi", 13, "H03",
    [(TRAIN, f"{T1} {ALIGN.replace(f'--episode-budget {BUDGET}', '--episode-budget 3500')} {CTX} "
             f"--budget {BUDGET} --seed 42 --tag {tag}")], 11.8,
    note="обучение на эпизодах 3500 эпох (вдвое дешевле), контекст нормирован на 7000")
add("w13-lofi-e", 13, "H03",
    [(EVAL, f"--policy agent --model-file rl_arch/agents_online/{tag}.pt "
            + t3ev(f"e_{tag}", S10))], 10.5, needs=[f"rl_arch/agents_online/{tag}.pt"])

# ---------------------------------------------------------------- волна 14
# Перенос между УрЧП: обучение на смеси без целевой задачи, проверка на ней.
XFER = {"ns2d": ("ns2d_liddriven", "ns2d_liddriven"), "pb2d": ("poissonboltzmann2d", "poisson_boltzmann_2d"),
        "h2ms": ("heat2d_multiscale", "heat2d_multiscale"), "wave": ("wave1d", "wave1d")}
T3_MULTI = (f"--subdirs solvable --per-pde-files 30 --gamma 0.99 --episode-budget {BUDGET} "
            f"--ctx-kmax 70 --scalar-ctx --epochs 60 --save-model")
XVAR = {"full": ("", "полные карты"), "shape": ("--state-mode shape", "карты без уровня (безразмерное состояние)"),
        "norm": ("--state-mode tasknorm",
                 "карты, нормированные статистиками своей задачи: одномерный оптимальный перенос уровня между УрЧП"),
        "blind": ("--state-mode blind", "без карт: универсальная выученная цепочка"),
        "desc": ("--pde-ctx", "полные карты и описатель задачи")}
for tk, (pde, sub) in (XFER.items() if NS2D else ()):
    for vk, (flags, note) in XVAR.items():
        mtag = f"x{tk}{vk}"
        model = f"/tmp/agent_convnext_dqn_cfix_rdlog_multi_{mtag}_seed1.pt"
        add(f"w14-{tk}-{vk}", 14, "H14",
            [(OFFLINE, f"--variant convnext_dqn {T3_MULTI} --holdout {sub} {flags} --seeds 1 --model-tag {mtag}"),
             (EVAL, f"--policy agent --model-file {model} "
                    + t3ev(f"t3x_{mtag}", S5, pde=pde))],
            11.5, note=f"перенос на {pde} без дообучения, обучение на остальных решаемых УрЧП: {note}")
    # простые альтернативы на той же задаче, тех же сидах и бюджете
    common = f"--pde {pde} --budget {BUDGET} --hours 10.5 --resume --ckpt-every 10 --seeds {S5} --no-state"
    add(f"w14-{tk}-base", 14, "H15",
        [(EVAL, f"--policy script --script {RAND6} --script-tail stop {common} --tag t3x_{tk}_rand6"),
         (EVAL, f"--policy script --script LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500 --script-tail repeat "
                f"{common} --tag t3x_{tk}_t1c"),
         (EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat {common} --tag t3x_{tk}_l1k"),
         (EVAL, f"--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat {common} "
                f"--tag t3x_{tk}_a3l"),
         (EVAL, f"--policy script --script SOAP:0.003:1000 --script-tail repeat {common} --tag t3x_{tk}_soap"),
         (EVAL, f"--policy scout {common} --tag t3x_{tk}_scout"),
         (EVAL, f"--policy scout --scout-set Adam:0.001:100,LBFGS:1:100,SOAP:0.003:100 "
                f"--scout-commit Adam:0.001:1000,LBFGS:1:500,SOAP:0.003:1000 {common} --tag t3x_{tk}_scouts"),
         (EVAL, f"--policy bandit {common} --tag t3x_{tk}_bandit")],
        11.5, note=f"{pde}: универсальная цепочка, цепочки лидеров, Adam и L-BFGS, SOAP, разведка, бандит")
    add(f"w14-{tk}-keep", 14, "H12",
        [(EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat {KEEP} {common} --tag t3x_{tk}_l1kk"),
         (EVAL, f"--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat {KEEP} {common} "
                f"--tag t3x_{tk}_a3lk"),
         (EVAL, f"--policy scout --scout-set Adam:0.001:100,LBFGS:1:100,SOAP:0.003:100 "
                f"--scout-commit Adam:0.001:1000,LBFGS:1:500,SOAP:0.003:1000 {KEEP} {common} "
                f"--tag t3x_{tk}_scoutsk")],
        8.0, note=f"{pde}: L-BFGS, Adam -> L-BFGS и разведка с SOAP в среде без сбросов оптимизатора")
    if tk != "ns2d":
        add(f"w14-{tk}-l1", 14, "H14",
            [(EVAL, f"--policy agent --model-file {LEAD1} --stop-on-noop " + t3ev(f"t3x_{tk}_l1", S5, pde=pde))],
            10.5, needs=[LEAD1], note=f"агент, обученный только на ns2d_liddriven, на {pde} без дообучения")
    # дообучение на целевой задаче 3 часа от агента смеси (протокол трека 2)
    ftag = f"q_t3ft_{tk}_s42"
    warm = f"convnext_dqn_cfix_rdlog_multi_x{tk}full_seed1.pt"
    add(f"w14-{tk}-ft", 14, "H14",
        [(TRAIN, f"--variant convnext_dqn --pde {pde} --hours 3 --save-agent --save-every 2 --wsrl-warmup 5 "
                 f"--rlpd --rlpd-utd 8 --self-prior 30 --warm-start {warm} --online-reward dlog --err-norm init "
                 f"--gamma 0.99 --episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70 {CTX} "
                 f"--seed 42 --tag {ftag}"),
         (EVAL, f"--policy agent --model-file agent_{ftag}.pt "
                + t3ev(f"e_{ftag}", S5, pde=pde, hours=7.5))],
        11.5, needs=[f"rl_arch/models/{warm}"],
        note=f"{pde}: три часа дообучения агента смеси на целевой задаче, затем оценка")

# ---------------------------------------------------------------- волна 15
# Итоговая таблица веток «метод» и «политика для новых уравнений» на всех 15 решаемых УрЧП:
# работы июля-сентября 2026 (PINNMorph, PINNForge) отчитываются на 13-25 задачах. Агент смеси
# учится без целевой задачи в варианте состояния, победившем в волне 14 ({XFER_FLAGS}), и
# оценивается с флагами развёртывания {DEPLOY} (страж или пусто). Для 11 уравнений, которых нет
# в волне 14, добавлены простые альтернативы, их пары без сбросов и трёхчасовое дообучение
REST11 = {"burg": ("burgers_1d", "burgers1d"), "gs": ("grayscott", "grayscott"),
          "h2cg": ("heat2d_complexgeometry", "heat2d_complexgeometry"),
          "h2vc": ("heat2d_varyingcoef", "heat2d_varyingcoef"), "hinv": ("heatinv", "heatinv"),
          "hnd": ("heatnd", "heatnd"), "nsbs": ("ns2d_backstep", "ns2d_backstep"),
          "p2c": ("poisson2d_classic", "poisson2d_classic"),
          "p3d": ("poisson3d_complexgeometry", "poisson3d_complexgeometry"),
          "pinv": ("poissoninv", "poissoninv"), "pnd": ("poissonnd", "poissonnd")}
SCOUT_SOAP = ("--policy scout --scout-set Adam:0.001:100,LBFGS:1:100,SOAP:0.003:100 "
              "--scout-commit Adam:0.001:1000,LBFGS:1:500,SOAP:0.003:1000")
for k, (pde, sub) in ((dict(XFER, **REST11)).items() if NS2D else ()):
    mtag = f"xf{k}"
    model = f"/tmp/agent_convnext_dqn_cfix_rdlog_multi_{mtag}_seed1.pt"
    add(f"w15-{k}-ag", 15, "H14",
        [(OFFLINE, f"--variant convnext_dqn {T3_MULTI} --holdout {sub} {{XFER_FLAGS}} --seeds 1 --model-tag {mtag}"),
         (EVAL, f"--policy agent --model-file {model} {{DEPLOY}} "
                + t3ev(f"t3f_{k}_ag", S5, pde=pde))],
        11.5, note=f"итоговая таблица, {pde}: агент смеси остальных решаемых УрЧП без дообучения, "
                   f"состояние и развёртывание по решениям фаз C и E")
    if k not in REST11:
        continue
    common = (f"--pde {pde}{plain_of(pde)} --budget {BUDGET} --hours 10.5 --resume --ckpt-every 10 "
              f"--seeds {S5} --no-state")
    add(f"w15-{k}-base", 15, "H15",
        [(EVAL, f"--policy script --script {RAND6} --script-tail stop {common} --tag t3f_{k}_rand6"),
         (EVAL, f"--policy script --script LBFGS:0.5:1000,LBFGS:0.5:500,LBFGS:1:500 --script-tail repeat "
                f"{common} --tag t3f_{k}_t1c"),
         (EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat {common} --tag t3f_{k}_l1k"),
         (EVAL, f"--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat {common} "
                f"--tag t3f_{k}_a3l"),
         (EVAL, f"--policy script --script SOAP:0.003:1000 --script-tail repeat {common} --tag t3f_{k}_soap"),
         (EVAL, f"--policy scout {common} --tag t3f_{k}_scout"),
         (EVAL, f"{SCOUT_SOAP} {common} --tag t3f_{k}_scouts"),
         (EVAL, f"--policy bandit {common} --tag t3f_{k}_bandit")],
        11.5, note=f"итоговая таблица, {pde}: универсальная цепочка, цепочки лидеров, Adam и L-BFGS, SOAP, "
                   f"разведка, бандит")
    add(f"w15-{k}-keep", 15, "H12",
        [(EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat {KEEP} {common} --tag t3f_{k}_l1kk"),
         (EVAL, f"--policy script --script Adam:0.001:1000,LBFGS:1:1000 --script-tail repeat {KEEP} {common} "
                f"--tag t3f_{k}_a3lk"),
         (EVAL, f"{SCOUT_SOAP} {KEEP} {common} --tag t3f_{k}_scoutsk")],
        8.0, note=f"итоговая таблица, {pde}: L-BFGS, Adam -> L-BFGS и разведка с SOAP без сбросов оптимизатора")
    ftag = f"q_t3f_{k}_ft_s42"
    warm = f"convnext_dqn_cfix_rdlog_multi_{mtag}_seed1.pt"
    add(f"w15-{k}-ft", 15, "H14",
        [(TRAIN, f"--variant convnext_dqn --pde {pde}{plain_of(pde)} --hours 3 --save-agent --save-every 2 "
                 f"--wsrl-warmup 5 --rlpd --rlpd-utd 8 --self-prior 30 --warm-start {warm} {{XFER_FLAGS}} "
                 f"--online-reward dlog --err-norm init --gamma 0.99 --episode-budget {BUDGET} --tolerance 0 "
                 f"--max-chain-steps 70 {CTX} --seed 42 --tag {ftag}"),
         (EVAL, f"--policy agent --model-file agent_{ftag}.pt {{DEPLOY}} "
                + t3ev(f"e_{ftag}", S5, pde=pde, hours=7.5))],
        11.5, needs=[f"rl_arch/models/{warm}"],
        note=f"итоговая таблица, {pde}: три часа дообучения агента смеси на целевой задаче")

# ---------------------------------------------------------------- готовые комбинации
# ---------------------------------------------------------------- волна 17
# Рычаги среды из полного обзора областей 4 октября (COVERAGE.md), оба применяются к любой политике
# одинаково и проверяются ПАРАМИ с прежними армами на тех же сидах. Локальные проверки на CPU
# (pb2d, сид 42) оказались слабыми, поэтому волна минимальная: по одной паре на рычаг.
#  * --lbfgs-tol 0: допуски torch.optim.LBFGS по умолчанию (1e-7, 1e-9) обрывают внутренний цикл
#    «по сходимости» (2505.10949; PINNacle через deepxde ставит ftol=0). Локально: L-BFGS 1.0 по 1000
#    эпох без сбросов, бюджет 4000 — с нулевыми допусками 0.0350 / 0.0266 / 0.0261 после 1000 / 2000
#    / 3000 эпох, затем застой на 0.0261, как и с прежними допусками (0.0261 после 3000). То есть
#    застой в одинарной точности не от допусков; рычаг проверяется на 10 сидах только парой к l1k;
#  * --finish-ls last|all: после бюджета последний слой пересчитывается методом наименьших квадратов
#    на точках и весах обучения (2603.04672; pb2d линейно, один шаг). Локально после Adam 1500 +
#    L-BFGS 300 (ошибка 0.0409): по 101 параметру лосс меняется на 0.1%, по 501 (все скрытые слои)
#    на 2%; на нелинейном burgers_1d (ЛМ, 6 итераций) лосс обучения ниже в 3.6 раза, ошибка хуже на
#    10%. Пара только к лучшей простой цепочке, чтобы иметь число на 10 сидах для ответа рецензенту.
TOL0 = "--lbfgs-tol 0"
add("w17-l1k-t0", 17, "H12",
    [(EVAL, f"--policy script --script LBFGS:1:1000 --script-tail repeat --no-state {KEEP} {TOL0} "
            + t3ev("t3e_l1k_t0", S10, hours=8.0))],
    8.0, note="пара к w0-l1kk (L-BFGS без сбросов): нулевые допуски L-BFGS")
_sk, _skname = (("Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000", "sk1s") if NS2D
                else ("Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000", "sk2s"))
_ls = "--finish-ls last" if NS2D else "--finish-ls all"      # у ns2d три выхода: базис all слишком дорог для ЛМ
add(f"w17-{_skname}-ls", 17, "H12",
    [(EVAL, f"--policy script --script {_sk} --script-tail repeat --no-state {KEEPS} {_ls} "
            + t3ev(f"t3e_{_skname}_ls", S10, hours=7.0))],
    7.0, note=f"пара к w11-{_skname}: завершающий шаг «последний слой МНК» ({_ls})")

# комитет с проводником и воздержанием (UBRL, 2305.07487; область E обзора): от действия проводника
# комитет отходит только при согласии двух агентов и запасе среднего стандартизованного Q в 0.5
# z-единицы; проводник на pb2d сильнее любого агента, поэтому воздержание защищает медиану.
# Пара к развёртыванию {COMBO_EVAL_B} (каркас, затем лидер) и к комитету E2
CMTG = "--policy agent --model-files {COMMITTEE} --guide {GUIDE} --guide-bonus 0.5 --guide-agree 2"
add("w17-cmtg", 17, "H16",
    [(EVAL, f"{CMTG} {KEEPS} " + ev_common("n_w17_cmtg", S10))],
    10.5, needs=["{COMMITTEE}"], note="комитет с проводником и воздержанием, безопасная среда без сбросов")

# генетический поиск по цепочкам при GPU-часах, равных одному обучению агента (область C9 обзора,
# 2011.11062): одна сессия, популяция 8, до 6 поколений, отсеивающие сиды 1000-1099 (как у
# случайного поиска w11-rs), в безопасной среде без сбросов; затем top3 переоцениваются на сидах
# 42-51 задачей --policy script (подстановка {GA_TOP1} из decisions после чтения строки *_ga в HF)
GA = "experiments/rl_arch/ga_chains.py"
_ga_init = ("Adam:0.01:2500,Adam:0.0001:2500,LBFGS:1:1000" if NS2D
            else "Adam:0.01:1000,Adam:0.0001:2500,LBFGS:1:1000")
add("w17-ga", 17, "H15",
    [(GA, f"--pde {PDE}{PLAIN} --budget {BUDGET} --hours 10.5 --kernel-hours 11 --resume --mask pso "
          f"--pop 8 --gens 6 {KEEPS} --init {_ga_init} --tag t3g_ga --save-dir runs_rl_online")],
    11.5, note="генетический поиск по цепочкам при стоимости одного обучения агента; каркас в начальной популяции")
add("w17-ga-top", 17, "H15",
    [(EVAL, f"--policy script --script {{GA_TOP1}} --script-tail stop --no-state {KEEPS} "
            + t3ev("t3e_ga_top1", S10, hours=7.0))],
    7.0, note="лучшая цепочка генетического поиска, переоценка на сидах 42-51 (подстановка GA_TOP1)")

# те же задачи волны 17 кусками по 2-5 сидов (теги те же, строки сливаются по сидам): до
# восполнения квоты Kaggle 10 октября у аккаунтов остаётся по 0.6-5.6 GPU-часа, и задача на
# 10 сидов не помещается ни на один аккаунт; первый кусок даёт первый взгляд на 3-5 сидах
_W17_CHUNKS = ((42, 44), (45, 47), (48, 49), (50, 51)) if NS2D else ((42, 46), (47, 51))
for _i, (_lo, _hi) in enumerate(_W17_CHUNKS, 1):
    add(f"w17-cmtg-c{_i}", 17, "H16",
        [(EVAL, f"{CMTG} {KEEPS} " + ev_common("n_w17_cmtg", seeds(_lo, _hi), hours=4.5))],
        4.5, needs=["{COMMITTEE}"], note=f"комитет с проводником и воздержанием, сиды {_lo}-{_hi}")
    add(f"w17-ga-top-c{_i}", 17, "H15",
        [(EVAL, f"--policy script --script {{GA_TOP1}} --script-tail stop --no-state {KEEPS} "
                + t3ev("t3e_ga_top1", seeds(_lo, _hi), hours=4.5))],
        4.5, note=f"лучшая цепочка генетического поиска, сиды {_lo}-{_hi} (подстановка GA_TOP1)")

# цепочки планировщика по суррогатной среде из логов (track3/surrogate.py, 4 октября: ансамбль
# градиентного бустинга по 114 728 переходам буферов 22 УрЧП; итоговая ошибка цепочек на
# отложенных эпизодах предсказывается с ранговой корреляцией 0.85-0.91, среди лучших 25%
# только 0.31-0.35, поэтому планы проверяются на GPU): A/B — лучший каркас с одной правкой,
# которую все варианты модели ставят на 0.04-0.13 декады выше; robust — план, устойчивый к
# проверке половиной данных. Среда со сбросами, как буферы: контроль t3b_sk1 / t3b_sk2 / t3b_log3
# на тех же сидах; правило: медиана лучше на 15% и не меньше 4 побед из 5, затем 10 сидов
if PDE in ("ns2d_liddriven", "poissonboltzmann2d"):
    _sur = ({"ab": "Adam:0.01:2500,Adam:0.001:2500,LBFGS:1:1000,LBFGS:1:1000",
             "rob": "PSO:0:300,Adam:0.01:2500,Adam:0.001:2500,LBFGS:1:1000,PSO:0.001:100,LBFGS:0.5:100,"
                    "LBFGS:0.5:100,LBFGS:0.5:100,LBFGS:0.5:100,LBFGS:0.5:100,LBFGS:1:100"} if NS2D else
            {"ab": "Adam:0.01:1000,Adam:0.001:2500,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:500",
             "rob": "Adam:0.01:100,Adam:0.001:2500,PSO:0.0001:200,LBFGS:1:1000,PSO:0.001:100,LBFGS:1:1000,"
                    "LBFGS:0.5:100,LBFGS:0.5:100,LBFGS:1:500,LBFGS:0.5:100,LBFGS:0.5:100,LBFGS:1:1000"})
    _sur_chunks = ((42, 44), (45, 46)) if NS2D else ((42, 46),)
    for _nm, _sc in _sur.items():
        for _i, (_lo, _hi) in enumerate(_sur_chunks, 1):
            add(f"w17-sur-{_nm}-c{_i}", 17, "H03",
                [(EVAL, f"--policy script --script {_sc} --script-tail stop --no-state "
                        + t3ev(f"t3e_sur_{_nm}", seeds(_lo, _hi), hours=4.5))],
                4.5, note=f"цепочка планировщика по суррогатной среде ({_nm}), сиды {_lo}-{_hi}, среда со сбросами")
    if not NS2D:
        add("w17-sur-abs", 17, "H03",
            [(EVAL, f"--policy script --script {_sur['ab']} --script-tail stop --no-state {KEEPS} "
                    + t3ev("t3e_sur_abs", seeds(42, 46), hours=4.5))],
            4.5, note="цепочка A/B планировщика в безопасной среде без сбросов; контроль t3b_sk2s")

# ветвление концовок от одного снимка (branch_endings.py, 5 октября): каркас до конца фаз Adam,
# затем шесть концовок L-BFGS от тех же весов на том же сиде — парное сравнение без шума сида.
# Итог цепочки на 53-60% решают два последних действия (surrogate_active.py, опыт 3), данных
# рядом с лучшими цепочками в буферах нет; ветвление меряет запас адаптации в концовке (оракул
# по сиду против лучшей концовки) и даёт пары «состояние в точке решения -> лучшая концовка»
BRANCH = "experiments/rl_arch/branch_endings.py"
if PDE in ("ns2d_liddriven", "poissonboltzmann2d"):
    if NS2D:
        _bp = "Adam:0.01:2500,Adam:0.0001:2500"
        _be = ["LBFGS:1:1000,LBFGS:1:1000",
               "LBFGS:1:500,LBFGS:1:500,LBFGS:1:500,LBFGS:1:500",
               "LBFGS:1:1000,Adam:0.001:100,LBFGS:1:500,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100",
               "SOAP:0.003:1000,LBFGS:1:1000",
               "LBFGS:0.5:500,LBFGS:1:500,LBFGS:0.1:500,LBFGS:1:500",
               "LBFGS:1:1000,PSO:0.001:100,LBFGS:1:500,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100"]
        _bchunks = [(s_, s_) for s_ in range(42, 47)]
    else:
        _bp = "Adam:0.01:1000,Adam:0.0001:2500"
        _be = ["LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:1000,LBFGS:1:500",
               "LBFGS:1:500,LBFGS:1:500,LBFGS:1:500,LBFGS:1:500,LBFGS:1:500,LBFGS:1:500,LBFGS:1:500",
               "LBFGS:1:1000,Adam:0.001:100,LBFGS:1:1000,Adam:0.001:100,LBFGS:1:1000,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100",
               "SOAP:0.003:1000,SOAP:0.003:1000,LBFGS:1:1000,LBFGS:1:500",
               "LBFGS:0.1:500,LBFGS:1:100,LBFGS:1:500,LBFGS:0.1:1000,LBFGS:0.5:500,LBFGS:1:500,"
               "LBFGS:1:100,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100",
               "LBFGS:1:1000,PSO:0.001:100,LBFGS:1:1000,PSO:0.001:100,LBFGS:1:1000,LBFGS:1:100,LBFGS:1:100,LBFGS:1:100"]
        _bchunks = [(42, 43), (44, 45), (46, 47), (48, 49), (50, 51)]
    for _i, (_lo, _hi) in enumerate(_bchunks, 1):
        add(f"w17-br-c{_i}", 17, "H07",
            [(BRANCH, f"--pde {PDE}{PLAIN} --seeds {seeds(_lo, _hi)} --prefix {_bp} --endings {';'.join(_be)} "
                      f"--budget {BUDGET} {KEEPS} --tag t3br_sk --hours 4.5 --resume")],
            4.5, note=f"ветвление шести концовок от снимка каркаса, сиды {_lo}-{_hi}")

# концовка с SOAP на ns2d (ветвление 6 октября: 0.0328 против 0.0356 у каркаса, 5 побед из 5 на
# сидах 42-46): досчёт до 10 сидов той же цепочкой без ветвления, тег тот же — строки сливаются
if NS2D:
    for _i, (_lo, _hi) in enumerate(((47, 48), (49, 50), (51, 51)), 1):
        add(f"w17-soapend-c{_i}", 17, "H12",
            [(EVAL, f"--policy script --script Adam:0.01:2500,Adam:0.0001:2500,SOAP:0.003:1000,LBFGS:1:1000 "
                    f"--script-tail stop --no-state {KEEPS} " + t3ev("t3br_sk_e3", seeds(_lo, _hi), hours=3.0))],
            3.0, note=f"каркас, SOAP, L-BFGS: досчёт концовки с SOAP, сиды {_lo}-{_hi}")

# Наборы флагов для слотов волны 4 ({COMBO_A..C}, {COMBO_EVAL_A..B}) и для {XFER_FLAGS}. Каждый
# набор lint_queue.py прогоняет через разбор аргументов и проверки совместимости своего скрипта,
# поэтому в таблицу комбинаций PLAN.md попадают только исполнимые сочетания
GUIDE_B = "--guide {GUIDE} --guide-mode bonus --guide-bonus 0.05 --guide-chains 20"
GUARD = "--guard-rollback 1.0 --guard-fallback LBFGS:1:500"
AG = "--policy agent --model-file {BEST_AGENT}"
EVC = ev_common("combo", S10)
MULTI_NS = f"--variant convnext_dqn {T3_MULTI} --holdout ns2d_liddriven --seeds 1 --model-tag xcombo"
combos = [
    dict(id="K1", kind="train", script=TRAIN, args=f"{{BASE}} {CTX} {MASK} --state-mode level",
         note="дешёвая по данным база: контекст времени, маска и только уровень карт"),
    dict(id="K2", kind="train", script=TRAIN, args=f"{{BASE}} {CTX} {KEEP}",
         note="среда без сбросов и контекст (прошлый оптимизатор виден агенту)"),
    # маска только для PSO с шагом: лучшие каркасы-проводники начинаются с Adam 1e-2
    dict(id="K3", kind="train", script=TRAIN, args=f"{{BASE}} {CTX} --mask pso:0.001,pso:0.0001 {GUIDE_B}",
         note="проводник с порогом по Q, контекст и маска PSO с шагом: всё, что сужает разведку"),
    dict(id="K4", kind="train", script=TRAIN, args=f"{{BASE}} {CTX} {KEEP} {GUIDE_B}",
         note="среда без сбросов и проводник: проводник задаёт каркас, агент решает, где менять семейство"),
    dict(id="K5", kind="train", script=TRAIN, args=f"{{BASE}} {CTX} --n-step 3 {GUIDE_B}",
         note="трёхшаговые цели и проводник: длинные эпизоды по 70 шагов"),
    dict(id="K6", kind="train", script=TRAIN,
         args=f"--variant stat_dqn {T3_ON2} --state-mode loss {KEEP} --mask {FINE_MASK}",
         note="только онлайн: лоссы вместо карт, без сбросов, 70 решений по 100 эпох (арм w12-on2-fine)"),
    dict(id="K7", kind="train", script=TRAIN,
         args=f"--variant stat_dqn {T3_ON2} --state-mode loss {KEEP} --n-step 3",
         note="только онлайн: лоссы вместо карт, без сбросов, трёхшаговые цели"),
    dict(id="E1", kind="eval", script=EVAL, args=f"{AG} --guide {{GUIDE}} --guide-bonus 0.02 {GUARD} {EVC}",
         note="порог по Q и страж вместе"),
    dict(id="E2", kind="eval", script=EVAL,
         args=f"--policy agent --model-files {{COMMITTEE}} --ensemble vote {GUARD} {EVC}",
         note="комитет агентов со стражем"),
    dict(id="E3", kind="eval", script=EVAL, args=f"{AG} --state-every 3 {GUARD} {EVC}",
         note="карты раз в три действия и страж: дешёвое развёртывание с защитой"),
    dict(id="E4", kind="eval", script=EVAL,
         args=f"--policy script --script {{GUIDE}} --script-tail agent --model-file {{BEST_AGENT}} {EVC}",
         note="проводник как начало цепочки, агент как продолжение (без обучения)"),
    dict(id="E5", kind="eval", script=EVAL, args=f"{AG} {KEEP} {EVC}",
         note="готовый агент в среде без сбросов: дешёвая проба до обучения K2"),
    dict(id="E6", kind="eval", script=EVAL, args=f"{CMTG} {KEEPS} {GUARD} {EVC}",
         note="комитет с проводником и воздержанием плюс страж: отход от проводника только при согласии агентов"),
    dict(id="X1", kind="xfer", script=OFFLINE, args=f"{MULTI_NS} --state-mode tasknorm --pde-ctx",
         note="перенос: нормировка по задаче и описатель задачи вместе"),
    dict(id="X2", kind="xfer", script=OFFLINE, args=f"{MULTI_NS} --state-mode shape --pde-ctx",
         note="перенос: карты без уровня и описатель задачи"),
]

out = dict(pde=PDE, prefix=PREFIX, subdir=SUBDIR, budget=BUDGET, init_err=INIT_ERR,
           placeholders=["BASE", "BASE_NOVB", "BASE_QR", "BASE_FACT", "BASE_STAT", "GUIDE",
                         "COMMITTEE", "BEST_AGENT", "BUFFERS", "RS_CHAIN_LOSS", "RS_CHAIN_ERR", "FINAL",
                         "RULE_BEST", "COMBO_A", "COMBO_B", "COMBO_C", "COMBO_EVAL_A", "COMBO_EVAL_B",
                         "XFER_FLAGS", "DEPLOY", "AGENT_CHAIN", "GA_TOP1"],
           combos=combos, jobs=jobs)
with open(OUT_FILE, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"{OUT_FILE}: УрЧП {PDE}, буфер {SUBDIR}, E0 {INIT_ERR}, задач {len(jobs)}")
by = {}
for j in jobs:
    by.setdefault(j["wave"], [0, 0.0])
    by[j["wave"]][0] += 1
    by[j["wave"]][1] += j["hours"]
for w in sorted(by):
    print(f"волна {w}: задач {by[w][0]}, GPU-часов не более {by[w][1]:.0f}")
