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

# ---------------------------------------------------------------- волна 19
# Полная комбинация методов агента из обзора (agent_combo.py, 11 октября): офлайн-обучение в
# GPU-кернеле на буферах и модельных продолжениях, затем оценка чекпоинта. Компоненты: MBPO со
# штрафом MOPO, CQL, проводник-каркас, ансамбль Q (REDQ), упреждающие цели через модель (MVE),
# дистилляция лучших цепочек (DQfD), сглаживание по сетке действий (PANI), расписание маски.
# Плюс удешевлённый сбор данных (collect_active.py, только pb2d): направленный выбор цепочек (LCB),
# общие префиксы, ранний обрыв; контроль — случайный сбор того же размера; вариант половинной длины.
COMBO = "experiments/rl_arch/agent_combo.py"
COLLECT = "experiments/rl_arch/collect_active.py"
if PDE in ("ns2d_liddriven", "poissonboltzmann2d"):
    _cg = "Adam:0.01:2500,Adam:0.0001:2500" if NS2D else "Adam:0.01:1000,Adam:0.0001:2500"
    _cfull = ("--mbpo-h 5 --mopo-lam 0.5 --cql-alpha 1.0 --ens 5 --ens-min 2 --utd 4 --mve 2 "
              "--distill-top 15 --distill-w 1.0 --pani-w 0.3 --mask-schedule")

    def combo_jobs(jid, tag, extra, note, needs=()):
        ck = f"rl_arch/agents_online/{tag}.pt"
        add(f"{jid}-tr", 19, "H03",
            [(COMBO, f"--pde {PDE} --guide {_cg} {_cfull} {extra} --offline-eval half --tag {tag}")],
            2.0, needs=list(needs), note="офлайн-обучение: " + note)
        for suf, sd in (("a", seeds(42, 46)), ("b", seeds(47, 51))):
            add(f"{jid}-n{suf}", 19, "H03",
                [(EVAL, f"--policy script --script {_cg} --script-tail agent --model-file {ck} "
                        f"--no-state --state-mode loss {KEEPS} " + t3ev(f"n_{tag}", sd, hours=6.0))],
                6.0, needs=[ck], note="оценка: " + note)

    # смоуки в кернеле перед настоящими запусками: скачивание буферов, таблица, заливка чекпоинта,
    # мягкая остановка цепочки и докатка концовок в среде; на HF итог не пишут (кроме чекпоинта смоука)
    if not NS2D:
        add("w19-smoke-combo", 19, "H03",
            [(COMBO, f"--pde {PDE} --guide {_cg} {_cfull} --smoke --tag {Q}_combo_smoke"),
             (EVAL, f"--policy script --script {_cg} --script-tail agent "
                    f"--model-file rl_arch/agents_online/{Q}_combo_smoke.pt --no-state --state-mode loss {KEEPS} "
                    f"--pde {PDE} --budget 300 --seeds 42 --smoke --tag smoke_combo")],
            1.0, note="смоук: обучение комбинации и оценка чекпоинта в одном кернеле")
        add("w19-smoke-ca", 19, "H03",
            [(COLLECT, f"--pde {PDE} --acq random --rounds 2 --batch 3 --share-prefix --truncate 3 {KEEPS} "
                       f"--smoke --tag ca_smoke --save-dir runs_rl_online"),
             (COLLECT, f"--pde {PDE} --acq lcb --rounds 1 --batch 2 --share-prefix --truncate 3 {KEEPS} "
                       f"--smoke --quick --tag ca_smoke_lcb --save-dir runs_rl_online")],
            1.5, note="смоук: сбор с общими префиксами и обрывом; направленный сбор с таблицей буферов")
    combo_jobs("w19-combo", f"{Q}_combo_s42", "", "полная комбинация на существующих буферах")
    # лестница по одному компоненту (11 октября): полная комбинация сразу оказалась хуже цепочки
    # (ns2d на GPU 0.356 против 0.0356; офлайн-отчёт показывал это заранее), поэтому каждый компонент
    # проверяется отдельно поверх базы «каркас + MBPO». Только офлайн-обучение и офлайн-оценка на
    # независимой половине данных, PINN не обучается; десять обучений подряд в одном кернеле на общей
    # таблице, два сида обучения двумя задачами. На GPU-оценку идут варианты не хуже лучшей цепочки
    _lb = "--mbpo-h 5 --mopo-lam 0.5"
    LADDER = [("c0", f"--guide {_cg} {_lb}", "база: каркас и MBPO"),
              ("c1", f"--guide {_cg} {_lb} --cql-alpha 1.0", "база и CQL"),
              ("c2", f"--guide {_cg} {_lb} --ens 5 --ens-min 2 --utd 4", "база и ансамбль голов Q"),
              ("c3", f"--guide {_cg} {_lb} --mve 2", "база и упреждающие цели через модель"),
              ("c4", f"--guide {_cg} {_lb} --distill-top 15 --distill-w 1.0", "база и дистилляция лучших цепочек"),
              ("c5", f"--guide {_cg} {_lb} --pani-w 0.3", "база и сглаживание по сетке действий"),
              ("c6", f"--guide {_cg} {_lb} --mask-schedule", "база и расписание маски"),
              ("c7", _lb, "MBPO без каркаса"),
              ("c8", f"--guide {_cg} --cql-alpha 1.0", "каркас и CQL без модельных данных"),
              ("c9", f"--guide {_cg}", "каркас и обычный DQN")]
    for s_ in (0, 1):
        add(f"w19-lad-s{s_}", 19, "H03",
            [(COMBO, f"--pde {PDE} {fl} --seed {s_} --offline-eval half --tag {Q}_lad_{nm}_s{s_}")
             for nm, fl, _n in LADDER],
            4.0, note=f"лестница по одному компоненту, офлайн, сид обучения {s_}: " + "; ".join(n for _a, _b, n in LADDER))
    if not NS2D:
        _ca = ("--pde {p} --budget {b} --rounds 8 --batch 5 --share-prefix --prefix-bonus 0.1 --truncate 3 "
               "{k} --hours 11 --kernel-hours 11 --resume --save-dir runs_rl_online")
        for nm, acq, bud, extra, note in (
                ("lcb", "--acq lcb --lcb-k 1", BUDGET, "", "направленный сбор, общие префиксы, ранний обрыв"),
                ("rnd", "--acq random", BUDGET, "", "контроль: случайный сбор того же размера"),
                ("lcbh", "--acq lcb --lcb-k 1", BUDGET // 2, "--no-clip", "направленный сбор на эпизодах половинной длины")):
            ctag = f"ca_{nm}"
            add(f"w19-ca-{nm}", 19, "H03",
                [(COLLECT, _ca.format(p=PDE, b=bud, k=KEEPS) + f" {acq} {extra} --tag {ctag}")],
                11.5, note="сбор буфера заново: " + note)
            cfile = f"rl_arch/online_env/{PDE}_{ctag}_collect.json"
            combo_jobs(f"w19-combo-{nm}", f"{Q}_combo_{nm}_s42", f"--extra-data {cfile} --extra-only-target",
                       "полная комбинация на малом буфере (" + note + ")", needs=[cfile])

# ---------------------------------------------------------------- волна 20
# CV-проба (11 октября): агент должен работать по картам ландшафта как по изображению, а методы
# CV толком не проверялись. Задача с учителем на 195 тысячах состояний с картами из буферов 22
# УрЧП (без RL и без обучения PINN): предсказать изменение ошибки после действия. Сравниваются
# входы (только уровень, только форма, перемешанные пиксели, стопка и разность карт), кодировщики
# (исходный свёрточный, ConvNeXt, ResNet, ViT), самообучение (маскирование, контрастное),
# аугментации, замороженный кодировщик, перенос на уравнение вне обучения. Итог: какие методы
# CV извлекают из карт пользу сверх уровня ошибки; победитель идёт в агента.
CVPROBE = "experiments/rl_arch/cv_probe.py"
if NS2D:
    add("w20-cvprobe", 20, "H06",
        [(CVPROBE, "--tag w20cv_smoke --smoke"),
         (CVPROBE, "--tag w20cv --hours 4.5 --resume --out-dir /kaggle/working")],
        5.5, note="CV-проба на картах: смоук, затем полная сетка входов, кодировщиков и методов CV")

# ---------------------------------------------------------------- волна 21
# Методы из обзора на агенте с картами, по одному. Агент обязан читать карты кодировщиком CV:
# агент на скалярах (agent_combo.py, волна 19) остаётся только абляцией. Обучение офлайн на буфере
# с картами (минуты), оценка на 5 сидах в среде без сбросов. База — ConvNeXt, полные карты и время.
# Победители по медиане идут в комбинации (волна 22), три лучших — на второе уравнение.
ENC_MAE = "rl_arch/encoders/cvprobe_w21cv_enc_mae_resnet.pt"
ENC_CON = "rl_arch/encoders/cvprobe_w21cv_enc_con_resnet.pt"
if NS2D:
    add("w21-cvprobe2", 21, "H06",
        [(CVPROBE, "--tag w21cv --configs level,level-hgb,maps-resnet,maps-resnetpos,shuffle-resnetpos,"
                   "ssl-mae-resnet,ssl-con-resnet --save-encoder --hours 1.5 --out-dir /kaggle/working")],
        2.0, note="CV-проба, второй запуск: считывание с сохранением положения на карте (каналы координат, "
                  "без усреднения) и веса предобученных кодировщиков для агента")
MAP_BASE = f"{T3_OF} --scalar-ctx"
MAP_ARMS = (
    ("base", "convnext_dqn", "", (), "база: ConvNeXt, полные карты и время, без добавок"),
    ("cqlqr", "cnx_cql_qr", "", (), "консервативное квантильное обучение (CQL + QR), лучший вариант статьи"),
    ("res", "convnext_dqn", "--encoder resnet", (), "кодировщик ResNet"),
    # считывание с сохранением положения (resnetpos) в лестницу не идёт: во втором запуске пробы оно
    # хуже усреднения (R² 0.8963 против 0.9001) и равно своему варианту с перемешанными пикселями
    ("mae", "convnext_dqn", f"--encoder resnet --encoder-init {ENC_MAE}", (ENC_MAE,),
     "ResNet, предобученный восстановлением закрытых блоков на картах 22 уравнений"),
    ("frz", "convnext_dqn", f"--encoder resnet --encoder-init {ENC_CON} --freeze-encoder", (ENC_CON,),
     "замороженный контрастно предобученный ResNet: учится только голова (дешёвое обучение)"),
    ("d4", "convnext_dqn", "--aug d4", (), "аугментация поворотами и отражениями"),
    ("ens", "convnext_dqn", "--ens 5 --ens-min 2", (), "ансамбль из 5 голов, цель по минимуму двух (REDQ)"),
    ("mb", "convnext_dqn", "--mb-aux 0.5 --mb-h 3 --mb-lam 0.5", (),
     "модельные цели: суррогат среды даёт пессимистичную оценку всех 27 действий в реальных состояниях"),
    ("pani", "convnext_dqn", "--pani-w 0.5 --pani-sigma 0.1", (), "сглаживание Q по шуму на картах"),
    ("dtop", "convnext_dqn", "--distill-top 10 --distill-w 1.0", (), "отступ в пользу действий лучших 10% цепочек"),
    ("msk", "convnext_dqn", "--mask-schedule", (), "расписание маски: сначала грубая сетка шагов"),
)


def map_job(jid, variant, flags, mtag, note, needs=(), wave=21, sd=S5, hours=3.5):
    model = off_model(variant, mtag)
    add(jid, wave, "H06",
        [(OFFLINE, f"--variant {variant} {MAP_BASE} {flags} --seeds 1 --model-tag {mtag}"),
         (EVAL, f"--policy agent --model-file {model} --stop-on-noop {KEEPS} " + t3ev(f"t3m_{mtag}", sd, hours=hours))],
        hours, needs=needs, note=note)


add("w21-smoke2", 21, "H06",
    [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} --epochs 2 --fqe-epochs 2 --encoder resnetpos --ens 3 --ens-min 2 --mb-aux 0.5 "
               "--mb-h 2 --pani-w 0.5 --pani-sigma 0.1 --distill-top 10 --distill-w 1.0 --mask-schedule --aug d4 "
               "--seeds 1 --model-tag w21smoke"),
     (EVAL, f"--policy agent --model-file {off_model('convnext_dqn', 'w21smoke')} --stop-on-noop {KEEPS} "
            + t3ev("t3m_smoke", "42", hours=0.5, budget=1500))],
    1.0, note="смоук нового кода агента на картах: все приёмы разом, 2 эпохи (не --smoke: с ним чекпоинт "
              "не сохраняется), один короткий сид оценки")
for nm, variant, flags, needs, note in MAP_ARMS:
    map_job(f"w21-m-{nm}", variant, flags, f"m{nm}", note, needs=needs)

# ---------------------------------------------------------------- волна 22
# Комбинации победителей волны 21 на pb2d (медианы 5 сидов: модельные цели 0.045, замороженный
# предобученный кодировщик 0.051, предобучение восстановлением блоков 0.073, база 0.28; ансамбль,
# отступ к лучшим цепочкам, CQL+QR и ResNet без предобучения базу не улучшили). У каждого чекпоинта
# две оценки: как в волне 21 и с общими для всех уравнений правилами развёртывания — маска лучшего
# агента с картами (без Adam 1e-2 и PSO с ненулевым шагом) и страж с откатом при росте лосса.
MB = "--mb-aux 0.5 --mb-h 3 --mb-lam 0.5"
DEPLOY = f"{MASK} --guard-rollback 1.0 --guard-fallback LBFGS:1:500"
MAP_COMBOS = (
    ("mbfrz", f"--encoder resnet --encoder-init {ENC_CON} --freeze-encoder {MB}", (ENC_CON,), True,
     "модельные цели + замороженный контрастно предобученный ResNet"),
    ("mbmae", f"--encoder resnet --encoder-init {ENC_MAE} {MB}", (ENC_MAE,), True,
     "модельные цели + ResNet, предобученный восстановлением блоков"),
    ("mb2", "--mb-aux 2.0 --mb-h 3 --mb-lam 0.5", (), True, "модельные цели с весом 2 вместо 0.5"),
    ("mbd", MB, (), False, "модельные цели волны 21: только оценка с правилами развёртывания"),
    ("frzd", f"--encoder resnet --encoder-init {ENC_CON} --freeze-encoder", (ENC_CON,), False,
     "замороженный кодировщик волны 21: только оценка с правилами развёртывания"),
)
MAP_EH = 3.5 if NS2D else 2.0        # мягкий предел одной оценки на 5 сидах: сид ns2d около 30 минут


def combo_job(nm, flags, needs, plain, note, wave=22):
    model = off_model("convnext_dqn", f"c{nm}")
    steps = [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {flags} --seeds 1 --model-tag c{nm}")]
    if plain:
        steps.append((EVAL, f"--policy agent --model-file {model} --stop-on-noop {KEEPS} "
                            + t3ev(f"t3m_c{nm}", S5, hours=MAP_EH)))
    steps.append((EVAL, f"--policy agent --model-file {model} --stop-on-noop {DEPLOY} {KEEPS} "
                        + t3ev(f"t3m_c{nm}_dep", S5, hours=MAP_EH)))
    add(f"w{wave}-c-{nm}", wave, "H06", steps, (1.0 + MAP_EH) * (2 if plain else 1), needs=needs, note=note)


for nm, flags, needs, plain, note in MAP_COMBOS:
    combo_job(nm, flags, needs, plain, note)

# ---------------------------------------------------------------- волна 23
# Итог волны 22 на pb2d (медианы 5 сидов): замороженный кодировщик + правила развёртывания 0.0090,
# модельные цели + правила 0.0163, модельные цели + предобучение 0.0179 без правил (все сиды в
# 0.013-0.027); модельные цели с замороженным кодировщиком 0.178 и с весом 2 — 0.15 (хуже частей).
# У лидера после третьего шага почти все действия исполняет страж (откат и LBFGS:1:500), поэтому
# нужны контроли: те же правила у базы и у случайной политики, и самая частая цепочка лидера без
# агента. Второй круг комбинаций: к победителям добавлены сглаживание и аугментация.
PANI_D4 = "--pani-w 0.5 --pani-sigma 0.1 --aug d4"
for nm, flags, needs, plain, note in (
        ("based", "", (), False, "контроль: база волны 21 с правилами развёртывания"),
        ("mbmaepd", f"--encoder resnet --encoder-init {ENC_MAE} {MB} {PANI_D4}", (ENC_MAE,), True,
         "модельные цели + предобучение + сглаживание + аугментация"),
        ("frzpd", f"--encoder resnet --encoder-init {ENC_CON} --freeze-encoder {PANI_D4}", (ENC_CON,), False,
         "замороженный кодировщик + сглаживание + аугментация, оценка с правилами развёртывания")):
    combo_job(nm, flags, needs, plain, note, wave=23)
add("w23-rnd-dep", 23, "H06",
    [(EVAL, f"--policy random --no-state --mask adam:0.01,pso --guard-rollback 1.0 --guard-fallback LBFGS:1:500 "
            f"{KEEPS} " + t3ev("t3m_rnd_dep", S5, hours=MAP_EH))],
    1.0 + MAP_EH, note="контроль: случайная политика со стражем и маской (без Adam 1e-2 и без PSO: пустого "
                       "действия-остановки у случайной политики нет), без агента и без карт")
# ---------------------------------------------------------------- волна 24
# На ns2d победители pb2d проваливаются (0.4-0.7 при 0.05 у случайной политики с правилами): агент
# останавливается пустым действием через 200-2800 эпох или идёт одним Adam. Правила развёртывания,
# вторая версия: пустое действие и весь PSO запрещены, поэтому бюджет тратится целиком; страж прежний.
DEPLOY2 = "--mask adam:0.01,pso --guard-rollback 1.0 --guard-fallback LBFGS:1:500"
for nm, flags, needs, note in (
        ("mb", MB, (), "модельные цели"),
        ("mbmae", f"--encoder resnet --encoder-init {ENC_MAE} {MB}", (ENC_MAE,), "модельные цели + предобучение")):
    model = off_model("convnext_dqn", f"d{nm}")
    add(f"w24-d2-{nm}", 24, "H06",
        [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {flags} --seeds 1 --model-tag d{nm}"),
         (EVAL, f"--policy agent --model-file {model} {DEPLOY2} {KEEPS} " + t3ev(f"t3m_d{nm}_dep2", S5, hours=MAP_EH))],
        1.0 + MAP_EH, needs=needs, note=note + ": оценка без пустого действия и PSO, со стражем")

# ---------------------------------------------------------------- волна 25
# Два метода поверх рецепта волны 24 (предобученный ResNet + модельные цели + правила без пустого
# действия). Щит: модель среды до выбора действия запрещает те, что по её пессимистичному прогнозу
# поднимут ошибку (на ns2d страж тратит на такие действия 2600-5100 эпох из 7000). Текущую ошибку
# щит берёт из собственного прогноза по исполненной цепочке (--shield-err model); вариант с
# измеренной ошибкой привилегирован и идёт только как верхняя оценка. Учитель: агент на скалярах с
# полными прогонами MBPO, его Q добавлены к целям сети на картах; сам ученик ошибки не видит.
RECIPE = f"--encoder resnet --encoder-init {ENC_MAE} {MB}"
SHIELD = "--shield auto --shield-lam 0.5 --shield-keep 3"
TEACH = f"{Q}_teach_mbpo"
TEACH_CK = f"rl_arch/agents_online/{TEACH}.pt"
TEACH_TR = f"--pde {PDE} --mbpo-h 5 --mopo-lam 0.5 --seed 0 --offline-eval none"


def w25_eval(model, tag, extra, sd=S5, hours=None, budget=BUDGET):
    return (EVAL, f"--policy agent --model-file {model} {DEPLOY2} {extra} {KEEPS} "
                  + t3ev(tag, sd, hours=MAP_EH if hours is None else hours, budget=budget))


_m = off_model("convnext_dqn", "esmoke")
add("w25-smoke", 25, "H06",
    [(COMBO, f"{TEACH_TR} --smoke --tag {TEACH}_smoke"),
     (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --epochs 2 --fqe-epochs 2 "
               f"--teacher-from rl_arch/agents_online/{TEACH}_smoke.pt --teacher-w 0.5 --seeds 1 --model-tag esmoke"),
     w25_eval(_m, "t3m_esmoke", f"{SHIELD} --shield-err model", sd="42", hours=0.5, budget=1500)],
    1.5, needs=[ENC_MAE], note="смоук: учитель MBPO, ученик на картах с его целями, оценка со щитом")
_m = off_model("convnext_dqn", "esh")
add("w25-sh", 25, "H06",
    [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --seeds 1 --model-tag esh"),
     w25_eval(_m, "t3m_esh_shm", f"{SHIELD} --shield-err model"),
     w25_eval(_m, "t3m_esh_sht", f"{SHIELD} --shield-err true")],
    1.0 + 2 * MAP_EH, needs=[ENC_MAE],
    note="рецепт волны 24 со щитом: по прогнозу модели и (верхняя оценка) по измеренной ошибке")
_m = off_model("convnext_dqn", "ete")
add("w25-te", 25, "H06",
    [(COMBO, f"{TEACH_TR} --tag {TEACH}"),
     (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --teacher-from {TEACH_CK} --teacher-w 0.5 "
               "--seeds 1 --model-tag ete"),
     w25_eval(_m, "t3m_ete_dep2", ""),
     w25_eval(_m, "t3m_ete_shm", f"{SHIELD} --shield-err model")],
    1.5 + 2 * MAP_EH, needs=[ENC_MAE],
    note="рецепт волны 24 с целями учителя MBPO: без щита и со щитом по прогнозу модели")

# те же прогоны по одной оценке на кернел: остаток недельной квоты раздроблен по аккаунтам, и задача с
# двумя оценками ns2d (до 11 часов по оценке планировщика) не помещается ни в один
_st = (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --teacher-from {TEACH_CK} --teacher-w 0.5 "
                "--seeds 1 --model-tag ete")
add("w25-shm", 25, "H06",
    [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --seeds 1 --model-tag esh"),
     w25_eval(off_model("convnext_dqn", "esh"), "t3m_esh_shm", f"{SHIELD} --shield-err model")],
    1.0 + MAP_EH, needs=[ENC_MAE], note="рецепт волны 24 со щитом по прогнозу модели (одна оценка)")
add("w25-te1", 25, "H06",
    [(COMBO, f"{TEACH_TR} --tag {TEACH}"), _st, w25_eval(off_model("convnext_dqn", "ete"), "t3m_ete_dep2", "")],
    1.5 + MAP_EH, needs=[ENC_MAE], note="рецепт волны 24 с целями учителя MBPO, без щита (одна оценка)")
add("w25-te2", 25, "H06",
    [(COMBO, f"{TEACH_TR} --tag {TEACH}b"),
     (_st[0], _st[1].replace(TEACH_CK, f"rl_arch/agents_online/{TEACH}b.pt")),
     w25_eval(off_model("convnext_dqn", "ete"), "t3m_ete_shm", f"{SHIELD} --shield-err model")],
    1.5 + MAP_EH, needs=[ENC_MAE], note="рецепт волны 24 с целями учителя MBPO и щитом (одна оценка)")

# ---------------------------------------------------------------- волна 26
# Подтверждение на новых сидах 47-51 (вместе с 42-46 — 10 сидов; отбор по 5 сидам завышает победителя).
# Итог волны 25: цели учителя — pb2d 0.0225, ns2d 0.0506; учитель и щит — pb2d 0.0126, ns2d 0.068;
# щит — pb2d 0.0315, ns2d 0.0508; рецепт без добавок — 0.0174 и 0.077.
S5B = seeds(47, 51)
add("w26-cf-te", 26, "H06",
    [(COMBO, f"{TEACH_TR} --tag {TEACH}"), _st,
     w25_eval(off_model("convnext_dqn", "ete"), "t3m_ete_dep2", "", sd=S5B)],
    1.5 + MAP_EH, needs=[ENC_MAE], note="подтверждение: рецепт с целями учителя, сиды 47-51")
add("w26-cf-tes", 26, "H06",
    [(COMBO, f"{TEACH_TR} --tag {TEACH}b"),
     (_st[0], _st[1].replace(TEACH_CK, f"rl_arch/agents_online/{TEACH}b.pt")),
     w25_eval(off_model("convnext_dqn", "ete"), "t3m_ete_shm", f"{SHIELD} --shield-err model", sd=S5B)],
    1.5 + MAP_EH, needs=[ENC_MAE], note="подтверждение: рецепт с целями учителя и щитом, сиды 47-51")

# ---------------------------------------------------------------- волна 27
# Кривая по размеру буфера: рецепт с целями учителя на половине, четверти, восьмой и шестнадцатой
# части цепочек своего уравнения. Урезание согласовано: те же цепочки (--target-chains, --data-seed)
# видят суррогат среды, учитель и ученик на картах; буферы остальных 21 уравнения не трогаются.
# Не урезан только предобученный кодировщик: он учился без разметки на картах всех буферов.
N_CHAINS = {"poissonboltzmann2d": 528, "ns2d_liddriven": 1075}.get(PDE, 0)
for _div in ((2, 4, 8, 16) if N_CHAINS else ()):
    _n = N_CHAINS // _div
    _tt = f"{TEACH}_c{_n}"
    _mt = f"ete_c{_n}"
    add(f"w27-buf-{_div}", 27, "H06",
        [(COMBO, f"{TEACH_TR} --target-chains {_n} --data-seed 0 --tag {_tt}"),
         (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --teacher-from rl_arch/agents_online/{_tt}.pt "
                   f"--teacher-w 0.5 --target-chains {_n} --data-seed 0 --seeds 1 --model-tag {_mt}"),
         w25_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}_dep2", "")],
        1.5 + MAP_EH, needs=[ENC_MAE], note=f"кривая буфера: {_n} цепочек цели из {N_CHAINS} (1/{_div})")

# ---------------------------------------------------------------- волна 28
# Направленный сбор С КАРТАМИ для нового уравнения: модель среды по буферам остальных 21 уравнения
# и собственным прогонам выбирает цепочки (LCB), общие префиксы и ранняя остановка экономят эпохи,
# каждая цепочка пишется файлом буфера в исходном формате. Затем рецепт учится только на собранном
# (--target-buffer): исходный случайный буфер своего уравнения не читается нигде.
# Сравнение — с тем же числом случайных цепочек исходного буфера (волна 27).
W28_K = {"poissonboltzmann2d": 4, "ns2d_liddriven": 10}.get(PDE, 0)
W28_SHORT = "ns2d" if NS2D else PREFIX
_cbatch = 3 if NS2D else 5


def tbuf(k):
    return f"rl_arch/buffers_new/tbuf_{W28_SHORT}_w{k}"


if PREFIX == "pb2d":
    add("w28-smoke", 28, "H06",
        [(COLLECT, f"--pde {PDE} --acq random --rounds 1 --batch 2 --share-prefix --share-min 100 {KEEPS} --smoke "
                   "--save-buffer --init Adam:0.001:100,LBFGS:1:100 --init Adam:0.001:100,Adam:0.0001:100 "
                   "--tag tb_smoke --save-dir runs_rl_online"),
         (COMBO, f"{TEACH_TR} --smoke --target-buffer runs_rl_online/buffers_new/tb_smoke --tag tb_smoke_teach"),
         (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer runs_rl_online/buffers_new/tb_smoke "
                   "--teacher-from rl_arch/agents_online/tb_smoke_teach.pt --teacher-w 0.5 --seeds 1 "
                   "--model-tag tb_smoke --epochs 2 --fqe-epochs 2"),
         w25_eval(off_model("convnext_dqn", "tb_smoke"), "t3m_tb_smoke", "", sd="42", hours=0.5, budget=1500)],
        1.5, needs=[ENC_MAE], note="смоук: сбор двух цепочек с картами, учитель и ученик на собранном, оценка")
for _k in range(W28_K):
    add(f"w28-col-{_k}", 28, "H06",
        [(COLLECT, f"--pde {PDE} --budget {BUDGET} --rounds 99 --batch {_cbatch} --share-prefix --prefix-bonus 0.1 "
                   f"--truncate 3 {KEEPS} --acq lcb --lcb-k 1 --save-buffer --worker {_k}/{W28_K} --hours 8.0 "
                   f"--kernel-hours 8.0 --resume --save-dir runs_rl_online --tag tbuf_{W28_SHORT}_w{_k}")],
        9.0, note=f"направленный сбор с картами, сборщик {_k} из {W28_K}, 8 часов")
# обучение рецепта на собранном: все сборщики, половина, четверть
for _nm, _ks in ((("all", range(W28_K)), ("half", range(W28_K // 2)), ("quart", range(max(1, W28_K // 4))))
                 if W28_K else ()):
    _src = ",".join(tbuf(k) for k in _ks)
    _tt, _mt = f"{TEACH}_tb{_nm}", f"ete_tb{_nm}"
    add(f"w29-tb-{_nm}", 29, "H06",
        [(COMBO, f"{TEACH_TR} --target-buffer {_src} --tag {_tt}"),
         (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {_src} "
                   f"--teacher-from rl_arch/agents_online/{_tt}.pt --teacher-w 0.5 --seeds 1 --model-tag {_mt}"),
         w25_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}_dep2", "")],
        1.5 + MAP_EH, needs=[ENC_MAE],
        note=f"рецепт на направленно собранном буфере: сборщиков {len(list(_ks))} из {W28_K}")

# ---------------------------------------------------------------- волна 30
# Оставшиеся офлайновые методы обзора поверх рецепта с целями учителя, по одному: HL-Gauss (цель Q
# как распределение, 2403.03950), полный просмотр вперёд в суррогате вместо близорукого,
# вспомогательное предсказание уровня следующих карт. Отдельно — одна политика на все уравнения:
# ученик, суррогат и учитель не видят своё уравнение совсем, оценка на нём без дообучения.
def w30_job(nm, extra, note):
    _tt, _mt = f"{TEACH}_{nm}", f"ete_{nm}"
    add(f"w30-{nm}", 30, "H06",
        [(COMBO, f"{TEACH_TR} --tag {_tt}"),
         (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --teacher-from rl_arch/agents_online/{_tt}.pt "
                   f"--teacher-w 0.5 {extra} --seeds 1 --model-tag {_mt}"),
         w25_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}_dep2", "")],
        1.5 + MAP_EH, needs=[ENC_MAE], note=note)


w30_job("hlg", "--hlg", "HL-Gauss: модельные цели и цели учителя по ожидаемому значению")
w30_job("hlgce", "--hlg --hlg-aux ce", "HL-Gauss: модельные цели и цели учителя тоже как распределение")
w30_job("mbfull", "--mb-full --mb-beam 3", "полный просмотр вперёд в суррогате (луч 3, глубина 3)")
w30_job("auxn", "--aux-next 0.5", "вспомогательное предсказание уровня следующих карт и награды")
_tt, _mt = f"{TEACH}_ho", "ete_ho"
add("w30-ho", 30, "H06",
    [(COMBO, f"{TEACH_TR} --extra-only-target --tag {_tt}"),
     (OFFLINE, f"--variant convnext_dqn --subdirs solvable --holdout {SUBDIR} --per-pde-files 40 --pde-ctx --scalar-ctx "
               f"--err-norm init --reward-form delta --value-bound --gamma 0.99 --episode-budget {BUDGET} --ctx-kmax 70 "
               f"--save-model {RECIPE} --mb-max-rows 12000 --epochs 60 --teacher-from rl_arch/agents_online/{_tt}.pt "
               f"--teacher-w 0.5 --seeds 1 --model-tag {_mt}"),
     w25_eval(f"/tmp/agent_convnext_dqn_cfix_rdelta_vb_multi_{_mt}_seed1.pt", f"t3m_{_mt}_dep2", "")],
    3.0 + MAP_EH, needs=[ENC_MAE],
    note="одна политика на все уравнения: своё уравнение отложено у ученика, суррогата и учителя")

# ---------------------------------------------------------------- волна 31
# Два онлайновых метода поверх чекпоинта рецепта с целями учителя (волна 25, метка ete): дообучение
# RLPD со строгим тёплым стартом и то же с сигналами обучения рядом с картами (fulltele: новые каналы
# стартуют с нулевых весов, начальные Q равны офлайновым). 9 часов обучения, оценка отдельным кернелом.
FT_CK = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_seed1.pt"
FT_TR = (f"--variant convnext_dqn --pde {PDE} --hours 9 --save-agent --save-every 5 --save-buffer --rlpd --rlpd-utd 8 "
         f"--rlpd-subdir {SUBDIR} --offline-fix --offline-reward delta --init-err {INIT_ERR} --value-bound --gamma 0.99 "
         f"--episode-budget {BUDGET} --tolerance 0 --max-chain-steps 70 --scalar-ctx --ctx-no-err {KEEPS} "
         f"--mask adam:0.01,pso --warm-start {FT_CK} --warm-strict --eps-start 0.1 --lr 3e-4 --seed 42")
for _nm, _extra, _note in (("ft", "", "дообучение онлайн (RLPD) рецепта с целями учителя"),
                           ("ftt", "--state-mode fulltele --warm-widen",
                            "то же с сигналами обучения рядом с картами (fulltele)")):
    _tag = f"{Q}_{_nm}_ete_s42"
    _ag = f"rl_arch/agents_online/{_tag}.pt"
    add(f"w31-{_nm}-tr", 31, "H06", [(TRAIN, f"{FT_TR} {_extra} --tag {_tag}")], 9.8, needs=[FT_CK],
        note="обучение 9 часов: " + _note)
    add(f"w31-{_nm}-ev", 31, "H06", [w25_eval(_ag, f"t3m_{_nm}_dep2", "")], 0.5 + MAP_EH, needs=[_ag],
        note="оценка: " + _note)

# ---------------------------------------------------------------- волна 32
# Повтор лучшего результата волны 29 на других сборщиках. Четверть направленно собранного буфера
# (pb2d: сборщик 0, 19 цепочек; ns2d: сборщики 0-1, 17 цепочек) дала 0.0105 и 0.0456 — лучше полного
# собранного (0.045 и 0.068) и исходного буфера (0.0227 и 0.0631). У каждой точки одно обучение,
# поэтому проверяется, держится ли результат на таких же по размеру наборах от других сборщиков.
for _i, _ks in enumerate(((1,), (2,), (3,)) if PREFIX == "pb2d" else
                         ((2, 3), (4, 5), (6, 7), (8, 9)) if NS2D else ()):
    _src = ",".join(tbuf(k) for k in _ks)
    _tt, _mt = f"{TEACH}_tbq{_i}", f"ete_tbq{_i}"
    add(f"w32-tbq-{_i}", 32, "H06",
        [(COMBO, f"{TEACH_TR} --target-buffer {_src} --tag {_tt}"),
         (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {_src} "
                   f"--teacher-from rl_arch/agents_online/{_tt}.pt --teacher-w 0.5 --seeds 1 --model-tag {_mt}"),
         w25_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}_dep2", "")],
        1.5 + MAP_EH, needs=[ENC_MAE], note=f"рецепт на направленно собранном: сборщики {list(_ks)} (повтор четверти)")

# ---------------------------------------------------------------- волна 33
# Требование 13 октября: у агента остаются все 27 действий, маска при развёртывании не допускается
# (Adam 1e-2 и PSO должны быть доступны). Поэтому оценки этой волны идут БЕЗ маски: агент работает до
# конца бюджета, страж с откатом остаётся (тег ..._g) либо выключен (тег ..._free, протокол статьи).
# 1) Уже обученные лучшие чекпоинты без маски: рецепт на исходном буфере (ete) и на четверти
#    направленно собранного (ete_tbquart).
# 2) Лучший вариант с ConvNeXt вместо ResNet: без предобучения, с предобучением восстановлением
#    блоков (кодировщик предобучается в том же кернеле зондом CV), шире и глубже (convnextw), без
#    раннего сжатия карты (convnexthr).
# 3) Лучший вариант без скалярного контекста (шаг, потраченный бюджет, прошлое действие): его вклад.
GUARD = "--guard-rollback 1.0 --guard-fallback LBFGS:1:500"
W33_SRC = ",".join(tbuf(k) for k in range(max(1, W28_K // 4))) if W28_K else ""
W33_CV = "/kaggle/working"
# целиком, двумя кернелами и по одному сиду на кернел: остаток недельной квоты раздроблен по аккаунтам
W33_SPLIT = ((("", S5), ("-a", "42,43"), ("-b", "44,45,46")) + tuple((f"-s{x}", str(x)) for x in range(42, 52))
             + (("-c", seeds(47, 51)),))        # -c: сиды 47-51, вместе с 42-46 — подтверждение на 10 сидах
# исходный агент статьи (исходный свёрточный кодировщик + DQN, буфер и дообучение онлайн) в нашем
# протоколе: 7000 эпох, все 27 действий, без стража — до сих пор он оценивался только при 31 000 эпох
ORIG_CK = ("rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt" if NS2D
           else "rl_arch/agents_online/their_dqn_pb2d_verify_seed42.pt")


def free_eval(model, tag, sd, guard=True):
    return (EVAL, f"--policy agent --model-file {model} {GUARD if guard else ''} {KEEPS} "
                  + t3ev(tag + ("_g" if guard else "_free"), sd, hours=MAP_EH))


def w33_job(nm, enc, pre, sd=S5, sfx="", ctx=True, extra="", src=None, tr_extra="", st_extra=""):
    """src — источники собранного буфера (по умолчанию четверть сборщиков волны 28); tr_extra и
    st_extra — добавки к командам учителя и ученика (например, свой набор чужих буферов)."""
    _tt, _mt = f"{TEACH}_{nm}{sfx}", f"ete_{nm}"
    src = src or W33_SRC
    steps = []
    init = ""
    if pre == "probe":
        steps.append((CVPROBE, f"--tag w33cv --configs level,ssl-mae-{enc} --save-encoder --hours 0.7 "
                               f"--out-dir {W33_CV}"))
        init = f" --encoder-init {W33_CV}/cvprobe_w33cv_enc_mae_{enc}.pt"
    elif pre:
        init = f" --encoder-init {pre}"
    base = MAP_BASE if ctx else MAP_BASE.replace(" --scalar-ctx", "")
    steps += [(COMBO, f"{TEACH_TR} --target-buffer {src} {tr_extra} --tag {_tt}"),
              (OFFLINE, f"--variant convnext_dqn {base} --encoder {enc}{init} {MB} --target-buffer {src} "
                        f"--teacher-from rl_arch/agents_online/{_tt}.pt --teacher-w 0.5 {extra} {st_extra} --seeds 1 "
                        f"--model-tag {_mt}"),
              free_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}", sd)]
    add(f"w33-{nm}{sfx}", 33, "H06", steps, 2.0 + MAP_EH, needs=[pre] if pre and pre != "probe" else [],
        note=f"лучший вариант, кодировщик {enc}, " + ("с предобучением" if pre else "без предобучения")
             + ("" if ctx else ", без скалярного контекста") + f", без маски, сиды {sd}")


if W28_K:
    for _sfx, _sd in W33_SPLIT:
        for _ck, _nm in ((f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbquart_seed1.pt", "tbquart"),
                         (f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_seed1.pt", "ete")):
            for _g in (True, False):
                add(f"w33-nm-{_nm}-{'g' if _g else 'f'}{_sfx}", 33, "H06",
                    [free_eval(_ck, f"t3m_ete_{_nm}" if _nm != "ete" else "t3m_ete", _sd, guard=_g)],
                    0.5 + MAP_EH, needs=[_ck],
                    note=f"без маски, {'со стражем' if _g else 'без стража'}: готовый чекпоинт {_nm}, сиды {_sd}")
        for _nm, _enc, _pre in (("cnx", "convnext", ""), ("cnxm", "convnext", "probe"),
                                ("cnxwm", "convnextw", "probe"), ("cnxhrm", "convnexthr", "probe")):
            # на pb2d кодировщик предобучается в кернеле; на ns2d берётся уже сохранённый оттуда файл
            if _pre and PREFIX != "pb2d":
                _pre = f"rl_arch/encoders/cvprobe_w33cv_enc_mae_{_enc}.pt"
            w33_job(_nm, _enc, _pre, sd=_sd, sfx=_sfx)
        for _g in (False, True):
            add(f"w33-orig-{'g' if _g else 'f'}{_sfx}", 33, "H06", [free_eval(ORIG_CK, "t3m_orig", _sd, guard=_g)],
                0.5 + MAP_EH, needs=[ORIG_CK],
                note=f"исходный агент статьи в нашем протоколе, {'со стражем' if _g else 'без стража'}, сиды {_sd}")
        if NS2D:
            # второй чекпоинт исходной архитектуры на ns2d: буфер случайных переходов загружен до обучения,
            # как прогрев буфера в статье (первый — с симметричной выборкой RLPD)
            _o2 = "rl_arch/agents_online/their_dqn_ns2d_preload_seed42.pt"
            add(f"w33-orig2-f{_sfx}", 33, "H06", [free_eval(_o2, "t3m_orig2", _sd, guard=False)], 0.5 + MAP_EH,
                needs=[_o2], note=f"исходный агент статьи (прогрев буфера), наш протокол, без стража, сиды {_sd}")
        w33_job("noctx", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, ctx=False)
        # только контекст, без карт (карты обнулены): не сводится ли агент к расписанию по шагу и бюджету
        w33_job("blind", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, st_extra="--state-mode blind")
        # вклад предобучения кодировщика в лучшем варианте: тот же ResNet со случайных весов
        w33_job("resnp", "resnet", "", sd=_sd, sfx=_sfx)
        # две сети, как в исходном агенте статьи: одна выбирает оптимизатор, вторая (свой кодировщик)
        # — его шаг и длительность; выбор в два этапа. two — головы как у лучшего варианта, twod —
        # dueling-головы со скрытым слоем, как в статье
        # ещё два набора по два сборщика на ns2d (обучение и оценка без маски): разброс между наборами
        for _i, _ks in enumerate(((6, 7), (8, 9)) if NS2D else ()):
            w33_job(f"set{_i + 2}", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, src=",".join(tbuf(k) for k in _ks))
        w33_job("two", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, extra="--two-nets")
        w33_job("twod", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, extra="--two-nets --two-nets-head dueling")
        # повторы четверти собранного буфера (волна 32) без маски: держится ли лучший результат на
        # других наборах сборщиков при всех 27 действиях
        for _i in range(3 if PREFIX == "pb2d" else 2):
            _ck = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbq{_i}_seed1.pt"
            add(f"w33-nm-tbq{_i}-g{_sfx}", 33, "H06", [free_eval(_ck, f"t3m_ete_tbq{_i}", _sd)],
                0.5 + MAP_EH, needs=[_ck], note=f"без маски, со стражем: повтор {_i} четверти собранного, сиды {_sd}")

# ---------------------------------------------------------------- волна 34
# Сколько чужих данных нужно методу. Тот же направленный сбор и рецепт, но модель среды (у сборщика,
# учителя и ученика) видит журналы только ОДНОГО другого уравнения вместо 21: для pb2d это ns2d, для
# ns2d — pb2d. Предобученный кодировщик прежний. Оценка без маски.
W34_OTHER = {"poissonboltzmann2d": "ns2d_liddriven", "ns2d_liddriven": "poisson_boltzmann_2d"}.get(PDE, "")
W34_K = {"poissonboltzmann2d": 1, "ns2d_liddriven": 2}.get(PDE, 0)
for _k in range(W34_K):
    add(f"w34-lcol-{_k}", 34, "H06",
        [(COLLECT, f"--pde {PDE} --budget {BUDGET} --rounds 99 --batch {_cbatch} --share-prefix --prefix-bonus 0.1 "
                   f"--truncate 3 {KEEPS} --acq lcb --lcb-k 1 --save-buffer --worker {_k}/{W34_K} --hours 8.0 "
                   f"--kernel-hours 8.0 --resume --subdirs {W34_OTHER} --save-dir runs_rl_online "
                   f"--tag tbufl_{W28_SHORT}_w{_k}")],
        9.0, note=f"направленный сбор с картами, модель среды по одному чужому уравнению ({W34_OTHER})")
if W34_K:
    _lsrc = ",".join(f"rl_arch/buffers_new/tbufl_{W28_SHORT}_w{k}" for k in range(W34_K))
    for _sfx, _sd in W33_SPLIT[:3]:
        w33_job("lean", "resnet", ENC_MAE, sd=_sd, sfx=_sfx, src=_lsrc, tr_extra=f"--subdirs {W34_OTHER}",
                st_extra=f"--mb-subdirs {W34_OTHER}")

# ---------------------------------------------------------------- волна 35
# Только метрика (цена сбора и обучения не ограничена): кандидаты, которые могут обойти лучший метод
# таблицы статьи. 1) Комитет агентов, обученных на разных наборах сборщиков (усреднение Q и
# голосование). 2) Дообучение онлайн с сигналами обучения рядом с картами (fulltele), старт с лучшего
# агента на собранном буфере; исследование и оценка со всеми 27 действиями.
_cm = [f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_{n}_seed1.pt"
       for n in (("tbquart", "tbq0", "tbq1", "tbq2") if PREFIX == "pb2d" else ("tbquart", "tbq0", "tbq1"))]
for _how in (("mean", "vote") if W28_K else ()):
    add(f"w35-cmt-{_how}", 35, "H06",
        [(EVAL, f"--policy agent --model-files {','.join(_cm)} --ensemble {_how} {GUARD} {KEEPS} "
                + t3ev(f"t3m_cmt_{_how}_g", S5, hours=MAP_EH))],
        0.5 + MAP_EH, needs=_cm, note=f"комитет из {len(_cm)} агентов на разных наборах сборщиков, {_how}, без маски")
if W28_K:
    _wk = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbquart_seed1.pt"
    _tag = f"{Q}_ftt2_tbq_s42"
    _ag = f"rl_arch/agents_online/{_tag}.pt"
    add("w35-ftt2-tr", 35, "H06",
        [(TRAIN, FT_TR.replace(FT_CK, _wk).replace(" --mask adam:0.01,pso", "")
                 + f" --state-mode fulltele --warm-widen --tag {_tag}")],
        9.8, needs=[_wk], note="дообучение онлайн с сигналами обучения, старт с лучшего агента, все 27 действий")
    add("w35-ftt2-ev", 35, "H06", [free_eval(_ag, "t3m_ftt2", S5)], 0.5 + MAP_EH, needs=[_ag],
        note="оценка: дообучение онлайн с сигналами обучения от лучшего агента, без маски")
    # то же без сигналов обучения: чистое дообучение онлайн (RLPD) от лучшего офлайнового агента
    _tag2 = f"{Q}_ft2_tbq_s42"
    _ag2 = f"rl_arch/agents_online/{_tag2}.pt"
    add("w35-ft2-tr", 35, "H06",
        [(TRAIN, FT_TR.replace(FT_CK, _wk).replace(" --mask adam:0.01,pso", "") + f" --tag {_tag2}")],
        9.8, needs=[_wk], note="дообучение онлайн (RLPD) от лучшего офлайнового агента, все 27 действий")
    add("w35-ft2-ev", 35, "H06", [free_eval(_ag2, "t3m_ft2", S5)], 0.5 + MAP_EH, needs=[_ag2],
        note="оценка: дообучение онлайн от лучшего офлайнового агента, без маски")
    # обучение и оценка одним кернелом (внешний запуск оценки не нужен): 8 часов обучения, 3 или 2 сида
    _asd = "42,43,44" if PREFIX == "pb2d" else "42,43"
    for _nm2, _tg, _ex in (("ftt2", _tag, " --state-mode fulltele --warm-widen"), ("ft2", _tag2, "")):
        add(f"w35-{_nm2}-auto", 35, "H06",
            [(TRAIN, FT_TR.replace(FT_CK, _wk).replace(" --mask adam:0.01,pso", "").replace("--hours 9", "--hours 8")
                     + f"{_ex} --tag {_tg}a"),
             free_eval(f"agent_{_tg}a.pt", f"t3m_{_nm2}", _asd)],
            11.5, needs=[_wk], note=f"дообучение онлайн ({_nm2}) и оценка одним кернелом")
    # те же оценки по одному сиду на кернел: считаются одновременно, результат на 4 часа раньше
    for _x in range(42, 47):
        add(f"w35-ftt2-ev-s{_x}", 35, "H06", [free_eval(_ag, "t3m_ftt2", str(_x))], 1.5, needs=[_ag],
            note=f"оценка, сид {_x}: дообучение онлайн с сигналами обучения от лучшего агента")
        add(f"w35-ft2-ev-s{_x}", 35, "H06", [free_eval(_ag2, "t3m_ft2", str(_x))], 1.5, needs=[_ag2],
            note=f"оценка, сид {_x}: дообучение онлайн от лучшего агента")

# ---------------------------------------------------------------- волна 36
# Как лучше соединить две сети (сеть оптимизатора и сеть его параметров, как в исходном агенте): три
# способа против одной сети. Чтобы сравнению можно было верить, каждый вариант учится дважды (сид
# обучения 1 и 2) и оценивается на трёх сидах среды; учитель общий (готовый чекпоинт), данные —
# четверть направленно собранного буфера, оценка без маски.
#   one   — одна сеть на 27 действий (контроль);
#   addsh — шаг и длительность складываются, кодировщик общий (как в коде статьи);
#   jntsh — у каждого оптимизатора одна голова на все его пары (шаг, длительность), кодировщик общий;
#   autsh — последовательный выбор: оптимизатор, шаг, длительность; кодировщик общий.
W36_TEACH = f"rl_arch/agents_online/{TEACH}_tbquart.pt"
for _nm, _fl in ((("one", ""),
                  ("addsh", "--two-nets --two-nets-mode additive --two-nets-shared"),
                  ("jntsh", "--two-nets --two-nets-mode joint --two-nets-shared"),
                  ("autsh", "--two-nets --two-nets-mode auto --two-nets-shared")) if W28_K else ()):
    for _m in (1, 2):
        _mt = f"n2{_nm}"
        add(f"w36-{_nm}-m{_m}", 36, "H06",
            [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {W33_SRC} --teacher-from {W36_TEACH} "
                       f"--teacher-w 0.5 {_fl} --seeds {_m} --model-tag {_mt}"),
             free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", "42,43,44")],
            1.0 + 0.6 * MAP_EH, needs=[ENC_MAE, W36_TEACH],
            note=f"две сети, вариант {_nm}, сид обучения {_m}, три сида среды, без маски")

# ---------------------------------------------------------------- волна 37
# Чередование: лучший агент сам решает уравнение на новых сидах с долей пробных действий (больше проб
# там, где лосс перестал падать), все состояния с картами пишутся в буфер; затем учитель, модель среды
# и агент заново учатся офлайн полным рецептом на старых и новых данных. Половина прогонов идёт без
# стража: при страже действие, поднявшее лосс, откатывается и в данные не попадает, а агенту полезно
# видеть и такие исходы. Прогоны по одному сиду на кернел — все одновременно.
if W28_K:
    _it_old = W33_SRC
    _it_ck = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbquart_seed1.pt"
    for _r in (1, 2):
        _buf = f"{W28_SHORT}_it{_r}"
        _prev = _it_ck if _r == 1 else f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_it{_r - 1}_seed1.pt"
        for _i in range(8):
            _sd = 100 * _r + _i
            _gd = GUARD if _i % 2 == 0 else ""
            add(f"w37-it{_r}-roll-{_i}", 37, "H06",
                [(EVAL, f"--policy agent --model-file {_prev} {_gd} {KEEPS} --save-buffer {_buf} --explore-eps 0.1 "
                        f"--explore-stall 0.5 --stall-ratio 0.95 "
                        + t3ev(f"{_buf}_roll", str(_sd), hours=MAP_EH))],
                1.5, needs=[_prev], note=f"круг {_r}: прогон агента с пробными действиями, сид {_sd}, "
                                         + ("со стражем" if _gd else "без стража"))
        _src = ",".join([_it_old] + [f"rl_arch/buffers_new/{W28_SHORT}_it{k}" for k in range(1, _r + 1)])
        _tt, _mt = f"{TEACH}_it{_r}", f"ete_it{_r}"
        add(f"w37-it{_r}-train", 37, "H06",
            [(COMBO, f"{TEACH_TR} --target-buffer {_src} --tag {_tt}"),
             (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {_src} "
                       f"--teacher-from rl_arch/agents_online/{_tt}.pt --teacher-w 0.5 --seeds 1 --model-tag {_mt}")],
            1.5, needs=[ENC_MAE], note=f"круг {_r}: учитель и агент заново на старых и новых данных")
        # тот же круг одним кернелом без внешнего запуска: ждёт файлы прогонов в HF, учит, оценивает
        if _r == 1:
            add(f"w37-it{_r}-auto", 37, "H06",
                [("experiments/rl_arch/wait_hf.py", f"--path rl_arch/buffers_new/{_buf}/{SUBDIR} --n 8 --min 4 --hours 3.5"),
                 (COMBO, f"{TEACH_TR} --target-buffer {_src} --tag {_tt}a"),
                 (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {_src} "
                           f"--teacher-from rl_arch/agents_online/{_tt}a.pt --teacher-w 0.5 --seeds 1 --model-tag {_mt}"),
                 free_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}", S5)],
                5.0 + MAP_EH, needs=[ENC_MAE], note=f"круг {_r} одним кернелом: ждёт прогоны, учит, оценивает на 5 сидах")
        _new = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{_mt}_seed1.pt"
        for _x in range(42, 47):
            add(f"w37-it{_r}-ev-s{_x}", 37, "H06", [free_eval(_new, f"t3m_{_mt}", str(_x))], 1.5, needs=[_new],
                note=f"круг {_r}: оценка агента после переобучения, сид {_x}, без маски")

# ---------------------------------------------------------------- волна 38
# Дообучение онлайн лоссом офлайнового рецепта: офлайновая половина батча — направленно собранный
# буфер (а не исходный), на обеих половинах плотные цели модели среды и учителя. Старт с лучшего
# офлайнового агента, все 27 действий. Учитель готовый (тот же, что учил стартового агента).
if W28_K:
    _wk = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbquart_seed1.pt"
    _ffl = (FT_TR.replace(FT_CK, _wk).replace(" --mask adam:0.01,pso", "")
            .replace(f"--rlpd-subdir {SUBDIR}", f"--rlpd-target-buffer {W33_SRC}")
            + f" {MB} --teacher-from {W36_TEACH} --teacher-w 0.5")
    _tag = f"{Q}_ffl_tbq_s42"
    _ag = f"rl_arch/agents_online/{_tag}.pt"
    add("w38-ffl-tr", 38, "H06", [(TRAIN, f"{_ffl} --tag {_tag}")], 9.8, needs=[_wk, W36_TEACH],
        note="дообучение онлайн лоссом офлайнового рецепта на собранном буфере, все 27 действий")
    # обучение и оценка одним кернелом (внешний запуск оценки не нужен). На pb2d обучение короче: эпизод
    # там идёт около 12 минут, 5 часов дают около 25 эпизодов, а задача помещается в остаток квоты
    _fh = 5 if PREFIX == "pb2d" else 8
    add("w38-ffl-auto", 38, "H06",
        [(TRAIN, _ffl.replace("--hours 9", f"--hours {_fh}") + f" --tag {_tag}a"),
         free_eval(f"agent_{_tag}a.pt", "t3m_ffl", "42,43,44" if PREFIX == "pb2d" else "42,43")],
        _fh + 3.5, needs=[_wk, W36_TEACH], note="дообучение онлайн лоссом офлайнового рецепта и оценка одним кернелом")
    for _x in range(42, 47):
        add(f"w38-ffl-ev-s{_x}", 38, "H06", [free_eval(_ag, "t3m_ffl", str(_x))], 1.5, needs=[_ag],
            note=f"оценка, сид {_x}: дообучение онлайн лоссом офлайнового рецепта")

# ---------------------------------------------------------------- волна 39
# Добор сидов для двух кандидатов в лучший метод (14 октября): на pb2d дообучение онлайн лоссом
# офлайнового рецепта дало 0.0099 на сидах 42-44, на ns2d две сети с линейной головой 0.0444 на
# сидах 42-46. Оценка уже обученных агентов, без нового обучения; остаток квоты недели кусками
# по 1-3 часа, поэтому кернелы короткие.
if PREFIX == "pb2d":
    _ag = f"rl_arch/agents_online/{Q}_ffl_tbq_s42a.pt"
    for _nm, _sd in (("a", "45,46,47,48"), ("b", "49,50,51")):
        add(f"w39-ffl-ev-{_nm}", 39, "H06", [free_eval(_ag, "t3m_ffl", _sd)], 1.0 + MAP_EH, needs=[_ag],
            note=f"дообучение онлайн лоссом рецепта: оценка на сидах {_sd}")
if NS2D:
    _m2 = "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_ete_two_seed1.pt"
    for _x in range(47, 52):
        add(f"w39-two-ev-s{_x}", 39, "H06", [free_eval(_m2, "t3m_ete_two", str(_x))], 1.5, needs=[_m2],
            note=f"две сети (линейная голова), обученный агент волны 33, сид {_x}")

# ---------------------------------------------------------------- волна 40
# Правило остановки эпизода ОБУЧЕНИЯ из статьи (раздел 2, формула (11); раздел 3, «Stopping criterion»):
# эпизод кончается, когда ошибка относительно эталона дошла до допуска, после K_max = 10 стадий или при
# расходимости; оценка, как и в статье, идёт до конца бюджета 7000 эпох. Флаги --train-tol и --train-kmax
# (offline_rl.py, online_train_env.py): конец терминален, дальнейшие переходы цепочки выбрасываются,
# модельные цели и цели учителя за остановку не смотрят. Числа допуска в статье нет; берётся одно правило
# для всех задач — 1% ошибки необученной сети в единицах тренера (rmse+brmse, --init-err): pb2d 0.0081,
# ns2d 0.0049. На четверти собранного буфера (по сводкам сборщиков) допуск обрезает на pb2d 4 цепочки из 19
# (медиана — 7-я стадия), K_max — все 19 (35 переходов из 225); на ns2d допуск ни одной (ошибка цепочек не
# ниже 0.011), работает только K_max (14 цепочек из 17, 25 переходов из 195). Онлайновые эпизоды pb2d
# (итог агента 0.006-0.010) допуска достигают чаще. Сравнение — с тем же методом без правила (pb2d: t3m_ffl_g, волны 38-39;
# ns2d: t3m_ete_two_g, волны 33 и 39), рядом с итоговой ошибкой — эпохи до порога
# (experiments/rl_arch/epochs_to_threshold.py). Все оценки без маски.
W40_TOL = round(0.01 * INIT_ERR, 5)
W40_RULE = f"--train-tol {W40_TOL} --train-kmax 10"
if PREFIX == "pb2d":
    # дообучение онлайн лоссом рецепта (как w38-ffl-auto) с правилом остановки на обеих половинах батча.
    # Обучение то же по длительности, что у контроля (5 часов), иначе сравнение не парное; одним кернелом в
    # 2.5 часа оно не помещается (онлайновый буфер между кернелами не переносится). Оценки — отдельными
    # кернелами по 3-4 сида, со стражем (пара к t3m_ffl_g на сидах 42-51) и без стража (сиды 42-46; пара —
    # тот же агент волны 38 без стража, w40-ffl-free-*)
    _wk = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_tbquart_seed1.pt"
    _ffl = (FT_TR.replace(FT_CK, _wk).replace(" --mask adam:0.01,pso", "")
            .replace(f"--rlpd-subdir {SUBDIR}", f"--rlpd-target-buffer {W33_SRC}")
            + f" {MB} --teacher-from {W36_TEACH} --teacher-w 0.5")
    _tag = f"{Q}_fflsr_tbq_s42"
    _ag = f"rl_arch/agents_online/{_tag}.pt"
    add("w40-fflsr-tr", 40, "H06", [(TRAIN, _ffl.replace("--hours 9", "--hours 5") + f" {W40_RULE} --tag {_tag}")],
        5.8, needs=[_wk, W36_TEACH],
        note="дообучение онлайн лоссом рецепта с правилом остановки обучения (допуск 1% E0, K_max 10), 5 часов как "
             "у w38-ffl-auto; в 2.5 часа не помещается")
    for _nm, _sd, _g in (("a", "42,43,44", True), ("b", "45,46,47", True), ("c", "48,49,50,51", True),
                         ("fa", "42,43,44", False), ("fb", "45,46", False)):
        add(f"w40-fflsr-ev-{_nm}", 40, "H06", [free_eval(_ag, "t3m_fflsr", _sd, guard=_g)],
            0.5 + 0.5 * len(_sd.split(",")), needs=[_ag],
            note=f"оценка дообучения с правилом остановки, {'со стражем' if _g else 'без стража'}, сиды {_sd}")
    _ag0 = f"rl_arch/agents_online/{Q}_ffl_tbq_s42a.pt"
    for _nm, _sd in (("a", "42,43,44"), ("b", "45,46")):
        add(f"w40-ffl-free-{_nm}", 40, "H06", [free_eval(_ag0, "t3m_ffl", _sd, guard=False)],
            0.5 + 0.5 * len(_sd.split(",")), needs=[_ag0],
            note=f"контроль без стража: дообучение волны 38 без правила остановки, сиды {_sd}")
if NS2D:
    # офлайновый рецепт с двумя сетями (как w33_job("two", ...)) и правилом остановки. Учитель готовый
    # (W36_TEACH: тот же рецепт на той же четверти собранного буфера), иначе обучение с оценкой одного сида не
    # помещается в 2.5 часа. Первый кернел учит и оценивает сид 42 (чекпоинт уходит в HF), остальные сиды —
    # по одному на кернел, со стражем (пара к t3m_ete_two_g на сидах 42-51). Без стража не оцениваем: сид ns2d
    # стоит 1.0-1.3 часа, а контроля без стража у двух сетей нет
    _mt = "ete_twosr"
    _m40 = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{_mt}_seed1.pt"
    add("w40-twosr-s42", 40, "H06",
        [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {W33_SRC} --teacher-from {W36_TEACH} "
                   f"--teacher-w 0.5 --two-nets {W40_RULE} --seeds 1 --model-tag {_mt}"),
         free_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}", "42")],
        2.4, needs=[ENC_MAE, W36_TEACH],
        note="две сети (линейная голова) с правилом остановки обучения (допуск 1% E0, K_max 10): обучение и сид 42")
    for _x in range(43, 52):
        add(f"w40-twosr-ev-s{_x}", 40, "H06", [free_eval(_m40, f"t3m_{_mt}", str(_x))], 1.5, needs=[_m40],
            note=f"две сети с правилом остановки обучения, сид {_x}" + (" (добор до 10 сидов)" if _x > 46 else ""))
    # остаток квоты недели — куски по 1-2 часа: обучение отдельным кернелом, сид 42 — тоже отдельным
    add("w40-twosr-tr", 40, "H06",
        [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {W33_SRC} --teacher-from {W36_TEACH} "
                   f"--teacher-w 0.5 --two-nets {W40_RULE} --seeds 1 --model-tag {_mt}")],
        1.0, needs=[ENC_MAE, W36_TEACH], note="две сети с правилом остановки обучения: только обучение")
    add("w40-twosr-ev-s42", 40, "H06", [free_eval(_m40, f"t3m_{_mt}", "42")], 1.5, needs=[_m40],
        note="две сети с правилом остановки обучения, сид 42")
if PREFIX == "pb2d":
    # то же правило на офлайновом рецепте pb2d (одна сеть): обучение 0.5 часа помещается в остаток квоты недели,
    # онлайновое (w40-fflsr-tr, 5 часов) — нет. Пара — t3m_ete_tbquart_g (тот же рецепт и учитель без правила)
    _mt = "ete_sr"
    _m40 = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{_mt}_seed1.pt"
    add("w40-sr-tr", 40, "H06",
        [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {W33_SRC} --teacher-from {W36_TEACH} "
                   f"--teacher-w 0.5 {W40_RULE} --seeds 1 --model-tag {_mt}")],
        1.0, needs=[ENC_MAE, W36_TEACH], note="офлайновый рецепт с правилом остановки обучения: только обучение")
    for _nm, _sd in (("a", "42,43,44"), ("b", "45,46")):
        add(f"w40-sr-ev-{_nm}", 40, "H06", [free_eval(_m40, f"t3m_{_mt}", _sd)], 0.5 + 0.5 * len(_sd.split(",")),
            needs=[_m40], note=f"офлайновый рецепт с правилом остановки обучения, сиды {_sd}")

# ---------------------------------------------------------------- волна 41
# Подтверждение лидеров на двух уравнениях (14 октября, три новых аккаунта). Лидеры на 10 сидах: pb2d —
# офлайновый рецепт с одной сетью (t3m_ete_tbquart_g, 0.0118), ns2d — он же с двумя сетями (t3m_ete_two_g,
# 0.0473). У каждого одно обучение, а разброс между обучениями того же рецепта велик (0.0147 и 0.0198 у
# повторов волны 36), поэтому:
#   1) лидер учится заново ещё дважды (сид обучения 2 и 3, тот же учитель и данные), оценка на 10 сидах;
#   2) вторая архитектура (pb2d — две сети, ns2d — одна) дважды на 5 сидах: одна ли архитектура на оба
#      уравнения;
#   3) кодировщик предобучается без карт целевого уравнения (21 задача вместо 22), затем лидер учится с ним
#      и оценивается на 10 сидах: перенос предобучения на новое уравнение.
# Сид ns2d стоит 1.0-1.3 часа, поэтому оценка на 10 сидах делится на кернел обучения с сидами 42-46 и
# кернел оценки сидов 47-51 по чекпоинту из HF.
# папки буферов всех 22 задач (как offline_rl.ALL_SUBDIRS; на них предобучен ENC_MAE)
ALL_SUBDIRS22 = ("burgers1d", "burgers2d", "grayscott", "heat2d_complexgeometry", "heat2d_longtime",
                 "heat2d_multiscale", "heat2d_varyingcoef", "heatinv", "heatnd", "kuramoto_sivashinsky",
                 "ns2d_backstep", "ns2d_liddriven", "ns2d_longtime", "poisson2d_classic", "poisson2d_manyarea",
                 "poisson3d_complexgeometry", "poisson_boltzmann_2d", "poissoninv", "poissonnd", "wave1d",
                 "wave2d_heterogeneous", "wave2d_longtime")
W41_TEACH = {"one": W36_TEACH, "two": f"rl_arch/agents_online/{TEACH}_two.pt"}
W41_LEAD = "two" if NS2D else "one"
W41_OTHER = "one" if NS2D else "two"


def w41_train(arch, m, mt, init=ENC_MAE):
    return (OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE.replace(ENC_MAE, init)} --target-buffer {W33_SRC} "
                     f"--teacher-from {W41_TEACH[arch]} --teacher-w 0.5 {'--two-nets' if arch == 'two' else ''} "
                     f"--seeds {m} --model-tag {mt}")


def w41_hf_model(mt, m):
    return f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{mt}_seed{m}.pt"


if W28_K:
    _halves = (("42,43,44,45,46", "47,48,49,50,51") if NS2D else ("42,43,44,45,46,47,48,49,50,51", ""))
    _eh = (lambda sd: 0.4 + 1.2 * len(sd.split(","))) if NS2D else (lambda sd: 0.3 + 0.3 * len(sd.split(",")))
    for _m in (2, 3):
        _mt = f"w41{W41_LEAD}"
        add(f"w41-lead-m{_m}", 41, "H06",
            [w41_train(W41_LEAD, _m, _mt), free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _halves[0])],
            1.0 + _eh(_halves[0]), needs=[ENC_MAE, W41_TEACH[W41_LEAD]],
            note=f"лидер ({W41_LEAD}) заново, сид обучения {_m}, сиды {_halves[0]}")
        if _halves[1]:
            add(f"w41-lead-m{_m}-b", 41, "H06",
                [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _halves[1])], _eh(_halves[1]),
                needs=[w41_hf_model(_mt, _m)], note=f"лидер ({W41_LEAD}) заново, сид обучения {_m}, сиды {_halves[1]}")
        _mo = f"w41{W41_OTHER}"
        add(f"w41-other-m{_m}", 41, "H06",
            [w41_train(W41_OTHER, _m, _mo), free_eval(off_model("convnext_dqn", _mo, _m), f"t3m_{_mo}_m{_m}", S5)],
            1.0 + _eh(S5), needs=[ENC_MAE, W41_TEACH[W41_OTHER]],
            note=f"вторая архитектура ({W41_OTHER}), сид обучения {_m}, 5 сидов")
    # предобучение без карт целевого уравнения
    _cvtag = f"w41ho{W28_SHORT}"
    _subs = ",".join(x for x in ALL_SUBDIRS22 if x != SUBDIR)
    _enc = f"{W33_CV}/cvprobe_{_cvtag}_enc_mae_resnet.pt"
    _mt = f"w41ho{W41_LEAD}"
    add("w41-holdpre", 41, "H06",
        [(CVPROBE, f"--tag {_cvtag} --subdirs {_subs} --configs level,ssl-mae-resnet --save-encoder --hours 1.0 "
                   f"--out-dir {W33_CV}"),
         w41_train(W41_LEAD, 1, _mt, init=_enc),
         free_eval(off_model("convnext_dqn", _mt), f"t3m_{_mt}", _halves[0])],
        2.5 + _eh(_halves[0]), needs=[W41_TEACH[W41_LEAD]],
        note=f"кодировщик предобучен без карт {SUBDIR} (21 задача), лидер ({W41_LEAD}), сиды {_halves[0]}")
    if _halves[1]:
        add("w41-holdpre-b", 41, "H06", [free_eval(w41_hf_model(_mt, 1), f"t3m_{_mt}", _halves[1])], _eh(_halves[1]),
            needs=[w41_hf_model(_mt, 1)], note=f"кодировщик без карт {SUBDIR}: сиды {_halves[1]}")

# ---------------------------------------------------------------- волна 43
# Повторные обучения волны 41 показали, что разброс между обучениями — главный источник ошибки: pb2d
# 0.0118, 0.0123 и 0.0319 у трёх обучений одного рецепта, ns2d 0.0473, 0.0616 и 0.0603. Комитет из этих
# трёх обучений (среднее Q, тот же рецепт и данные, в отличие от комитета волны 35 на разных наборах):
# снимает ли усреднение зависимость от удачного обучения. Оценка по сиду на кернел — остаток квоты кусками.
if W28_K:
    _arch = "two" if NS2D else "one"
    _first = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{'ete_two' if NS2D else 'ete_tbquart'}_seed1.pt"
    _cm3 = [_first] + [w41_hf_model(f"w41{_arch}", m) for m in (2, 3)]
    for _x in range(42, 52):
        add(f"w43-cmt3-s{_x}", 43, "H06",
            [(EVAL, f"--policy agent --model-files {','.join(_cm3)} --ensemble mean {GUARD} {KEEPS} "
                    + t3ev(f"t3m_cmt3{_arch}_g", str(_x), hours=MAP_EH))],
            1.0 if not NS2D else 1.9, needs=_cm3,
            note=f"комитет трёх обучений лидера ({_arch}), среднее Q, сид {_x}")
        # комитет из двух: второе и третье обучение (без первого, самого удачного на ns2d) — держится ли
        # выигрыш без лучшего члена и сколько добавляет третий
        add(f"w43-cmt2-s{_x}", 43, "H06",
            [(EVAL, f"--policy agent --model-files {','.join(_cm3[1:])} --ensemble mean {GUARD} {KEEPS} "
                    + t3ev(f"t3m_cmt2{_arch}_g", str(_x), hours=MAP_EH))],
            1.0 if not NS2D else 1.9, needs=_cm3[1:],
            note=f"комитет двух обучений лидера ({_arch}, обучения 2 и 3), среднее Q, сид {_x}")

# ---------------------------------------------------------------- волна 44
# Две правки модельных целей лидера, по одной (обе выключены по умолчанию, offline_rl.py):
#   trust — доверие к суррогату (RG-TAIL, ICLR 2027 gspGh6aGPl): вес модельной цели строки
#           exp(-|ошибка предсказания записанного исхода| / 0.3) и ворота по доле верных знаков (0.7);
#           предсказание — суррогатом без кластера цепочек строки (5 фолдов, mb_targets.trust_weights);
#   hs    — цели по горизонтам 1, 2, 3, 5 вместо одного H=3, среднее (SHARP, ICLR 2027 su2U0XecdI;
#           жадный путь общий, просмотр вдвое дороже H=3).
# Каждая — три обучения лидера (сиды 1-3; pb2d одна сеть, ns2d --two-nets с учителем волны 41) и комитет
# трёх (среднее Q, страж) на сидах 42-51. Контроль — комитет трёх обучений прежнего рецепта волны 43
# (t3m_cmt3one_g на pb2d, t3m_cmt3two_g на ns2d), нового контроля нет. Кернелы обучения выгружают модель
# в HF, оценка — блоками сидов (сид pb2d около 0.85 часа, ns2d около 1.6), не больше 9 часов на кернел.
W44_ARMS = (("trust", lambda a: a + " --mb-trust both", 1.0, "доверие к модельным целям (state и ворота)"),
            ("hs", lambda a: a.replace("--mb-h 3", "--mb-hs 1,2,3,5"), 0.8, "модельные цели по горизонтам 1,2,3,5"))
W44_BLOCKS = ((42, 45), (46, 48), (49, 51)) if NS2D else ((42, 46), (47, 51))
W44_SEED_H = 1.6 if NS2D else 0.85
if W28_K:
    _arch = W41_LEAD
    for _arm, _mod, _th, _what in W44_ARMS:
        _mt = f"w44{_arm}{_arch}"
        _models = [w41_hf_model(_mt, m) for m in (1, 2, 3)]
        for _m in (1, 2, 3):
            _scr, _a = w41_train(_arch, _m, _mt)
            assert "--mb-h 3" in _a
            add(f"w44-{_arm}-tr-m{_m}", 44, "H06", [(_scr, _mod(_a))], _th, needs=[ENC_MAE, W41_TEACH[_arch]],
                note=f"{_what}: обучение лидера ({_arch}), сид обучения {_m}, модель в HF")
        for _lo, _hi in W44_BLOCKS:
            _sd = seeds(_lo, _hi)
            add(f"w44-{_arm}-ev{_lo}-{_hi}", 44, "H06",
                [(EVAL, f"--policy agent --model-files {','.join(_models)} --ensemble mean {GUARD} {KEEPS} "
                        + t3ev(f"t3m_cmt3{_arch}_{_arm}_g", _sd, hours=MAP_EH))],
                round((0.4 if NS2D else 0.3) + W44_SEED_H * (_hi - _lo + 1), 2), needs=_models,
                note=f"{_what}: комитет трёх обучений ({_arch}), среднее Q, страж, сиды {_sd}")

# ---------------------------------------------------------------- волна 45
# Отбор членов комитета без оценки в среде (experiments/rl_arch/select_agents.py): разброс между обучениями
# одного рецепта велик (волна 43), а оценка на 10 сидах стоит 8.5 (pb2d) и 16 (ns2d) GPU-часов на агента.
# Один кернел на уравнение ранжирует уже обученных и измеренных агентов двумя способами по собранному буферу
# цели (все сборщики): B1 — жадное действие агента в настоящих состояниях буфера, последствия — суррогат
# (ансамбль модельных целей, продолжение жадно по каждому члену, среднее по членам); B2 — FQE жадной политики
# на замороженном кодировщике MAE. Метки агентов — теги их оценок: select_report.py сверяет ранжирование с
# измеренными медианами (Spearman, Kendall, сожаление выбора). Основные — агенты с 10 (pb2d: и 3-6) сидами,
# дополнительные — варианты волн 33, 36, 37, 40 с 2-5 сидами (больше точек для корреляции, дёшево: оценка
# агента — секунды суррогата и минута FQE). Обучения и среды нет; оценка времени — сборка таблицы суррогата
# (буферы 22 задач) и ансамбль, как в обучении рецепта, плюс прогоны суррогата и FQE.
SELECT = "experiments/rl_arch/select_agents.py"
if W28_K:
    def _w45(mt, m=1):
        return f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_{mt}_seed{m}.pt"

    if NS2D:
        _main = (("t3m_ete_two_g", _w45("ete_two")), ("t3m_w41two_m2_g", _w45("w41two", 2)),
                 ("t3m_w41two_m3_g", _w45("w41two", 3)), ("t3m_ete_tbquart_g", _w45("ete_tbquart")),
                 ("t3m_w41hotwo_g", _w45("w41hotwo")))
        _extra = tuple((f"t3m_ete_{nm}_g", _w45(f"ete_{nm}"))
                       for nm in ("tbq0", "tbq1", "twod", "noctx", "resnp", "blind", "set2", "set3", "cnxwm",
                                  "cnxhrm", "twosr")) \
            + (("t3m_ete_g", _w45("ete")), ("t3m_w41one_m2_g", _w45("w41one", 2)), ("t3m_w41one_m3_g", _w45("w41one", 3)))
    else:
        _main = (("t3m_ete_tbquart_g", _w45("ete_tbquart")), ("t3m_w41one_m2_g", _w45("w41one", 2)),
                 ("t3m_w41one_m3_g", _w45("w41one", 3)), ("t3m_n2one_m1_g", _w45("n2one", 1)),
                 ("t3m_n2one_m2_g", _w45("n2one", 2)), ("t3m_w41two_m2_g", _w45("w41two", 2)),
                 ("t3m_w41two_m3_g", _w45("w41two", 3)), ("t3m_ete_sr_g", _w45("ete_sr")),
                 ("t3m_w41hoone_g", _w45("w41hoone")))
        _extra = tuple((f"t3m_ete_{nm}_g", _w45(f"ete_{nm}"))
                       for nm in ("two", "twod", "tbq0", "tbq1", "tbq2", "cnx", "cnxhrm", "cnxm", "cnxwm", "noctx",
                                  "resnp", "blind", "it1")) \
            + (("t3m_ete_g", _w45("ete")),) \
            + tuple((f"t3m_n2{nm}_m{m}_g", _w45(f"n2{nm}", m)) for nm in ("addsh", "jntsh", "autsh") for m in (1, 2))
    add("w45-select", 45, "H06",
        [(SELECT, f"--pde {PDE} --models {','.join(f'{a}={b}' for a, b in _main)} "
                  f"--extra-models {','.join(f'{a}={b}' for a, b in _extra)} --method both --tag w45")],
        1.2, needs=[b for _, b in _main + _extra] + [ENC_MAE],
        note=f"отбор агентов без оценки в среде: суррогат и FQE, основных {len(_main)}, дополнительных {len(_extra)}")

# ---------------------------------------------------------------- волна 46
# Предобучение MAE только на картах самой задачи-цели (вопрос рецензента: нужны ли карты 22 задач). Энкодер
# предобучается зондом CV (cv_probe.py --pretrain-only: без level, фолдов и lopo) на картах исходного (случайного)
# буфера цели трёх объёмов: 130 переходов (как направленно собранный буфер рецепта), 1000 и все (pb2d ~5 841,
# ns2d ~12 214); --max-per-pde берёт случайные цепочки целиком. Число шагов предобучения у всех трёх одно
# (--ssl-steps 1400 — как у ENC_MAE: 4 эпохи по ~350 пачек из 512 на ~180 тыс. карт), меняется только объём
# карт. Отдельный кернел на (задачу, объём) выгружает энкодер в HF rl_arch/encoders/; затем лидер (pb2d — одна
# сеть, ns2d — две) учится с ним дважды (сиды обучения 1 и 2, тот же учитель и собранный буфер, что в волне 41) и
# оценивается на сидах 42-46 (страж, без сбросов, бюджет 7000). Контроль — лидер с ENC_MAE (22 задачи): волны 41/43
# (t3m_ete_tbquart_g, t3m_w41one_m2/m3_g на pb2d; t3m_ete_two_g, t3m_w41two_m2/m3_g на ns2d) и без карт цели —
# t3m_w41ho<арх>_g. На ns2d оценка делится на два кернела (сиды 42-44 вместе с обучением, 45-46 по модели из HF):
# сид ns2d ~1.4 GPU-ч, кернел не больше 9 часов.
W46_SIZES = (("n130", 130), ("n1k", 1000), ("all", 0))
W46_SSL_STEPS = 1400
W46_SEED_H = 1.4 if NS2D else 0.7
W46_PRE_H = 0.8 if NS2D else 0.6
if W28_K:
    _arch = W41_LEAD
    for _sz, _n in W46_SIZES:
        _cvtag = f"w46{_sz}{W28_SHORT}"
        _enc = f"rl_arch/encoders/cvprobe_{_cvtag}_enc_mae_resnet.pt"
        add(f"w46-pre-{_sz}", 46, "H06",
            [(CVPROBE, f"--tag {_cvtag} --subdirs {SUBDIR} --max-per-pde {_n} --configs ssl-mae-resnet --pretrain-only "
                       f"--ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
            W46_PRE_H, note=f"MAE только на картах {SUBDIR}: " + (f"{_n} переходов" if _n else "весь исходный буфер")
                             + f", {W46_SSL_STEPS} шагов, энкодер в HF")
        _mt = f"w46{_sz}{_arch}"
        for _m in (1, 2):
            _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
            _eh = lambda sd: (0.4 if NS2D else 0.3) + W46_SEED_H * len(sd.split(","))   # noqa: E731
            add(f"w46-{_sz}-m{_m}", 46, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"лидер ({_arch}) с MAE только на картах цели ({_sz}), сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w46-{_sz}-m{_m}-b", 46, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"лидер ({_arch}) с MAE только на картах цели ({_sz}), сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 47
# Агент с нуля только на своих онлайновых данных: без офлайнового буфера, без предобучения MAE, без учителя и
# суррогата. DQN на картах (тот же ResNet, что у рецепта, случайная инициализация), скалярный контекст, все 27
# действий без маски, среда без сбросов (safe), бюджет 7000, награда-разность с границами ценности. Один кернел
# за 10 часов собирает ~14 (pb2d) / ~7 (ns2d) цепочек, поэтому раунды из W = 8 параллельных работников:
#   работник раунда r (online_train_env.py, 10 ч): r = 0 — сеть с нуля (у каждого свой сид среды и сети: при
#     eps >= 0.5 и выброшенной после раунда сети инициализация работника на результат не влияет); r >= 1 —
#     строгий тёплый старт от агента ученика прошлого раунда, в общем буфере все переходы раундов 0..r-1
#     (--preload-online), затем обычный онлайновый DQN тренера (5 обновлений по 32 на шаг среды) с eps по
#     расписанию раунда. Цепочки — файлами буфера с картами (--save-chains) в rl_arch/buffers_new/<буфер раунда>/,
#     в конце — метка прогона в <буфер раунда>_done/;
#   ученик раунда r (кернел после 6 из 8 меток раунда; ждёт остальные до 0.5 ч): тёплый старт от ученика r-1
#     (r = 0 — с нуля, сид 1), все переходы раундов 0..r в общем буфере, K обновлений собственным лоссом тренера
#     (K = 5 обновлений на собранный переход раунда: та же доля, что у работника), агент в HF
#     rl_arch/agents_online/<Q>_scratch_r<r>.pt (--learn-only), затем оценка этого агента на сидах 42-46 (страж,
#     без сбросов) — тег t3m_scratch_r<r>_g. На ns2d сиды 45-46 — отдельным кернелом.
# Кривая обучения: ошибка t3m_scratch_r<r>_g против GPU-часов сбора (8 x 10.5 на раунд). Нормировка карт — по
# переходам раунда 0 у ученика r = 0, дальше наследуется чекпоинтом. eps: раунд 0 — 1.0 -> 0.5, раунд 1 —
# 0.5 -> 0.05, раунд 2 — 0.1 -> 0.05 (постоянная затухания — доля шагов работника за раунд: ~150 на pb2d, ~75 на ns2d).
W47_W = 8
W47_ROUNDS = 3
W47_STEPS = 75 if NS2D else 150                   # шагов среды у работника за 10 часов (~11 действий на цепочку)
W47_EPS = {0: (1.0, 0.5, W47_STEPS // 2), 1: (0.5, 0.05, W47_STEPS // 3), 2: (0.1, 0.05, W47_STEPS // 3)}
W47_K = 5 * W47_W * W47_STEPS                     # обновлений ученика за раунд: 6000 на pb2d, 3000 на ns2d
W47_BASE = (f"--variant convnext_dqn --encoder resnet --pde {PDE}{PLAIN} --scalar-ctx --ctx-no-err "
            f"--init-err {INIT_ERR} --value-bound --gamma 0.99 --episode-budget {BUDGET} --tolerance 0 "
            f"--max-chain-steps 70 {KEEPS}")
W47_SEED_H = 1.4 if NS2D else 0.7


def w47_buf(r):
    return f"rl_arch/buffers_new/online_scratch_{W28_SHORT}_r{r}"


def w47_agent(r):
    return f"rl_arch/agents_online/{Q}_scratch_r{r}.pt"


if W28_K:
    for _r in range(W47_ROUNDS):
        _e0, _e1, _ed = W47_EPS[_r]
        _prev = ",".join(w47_buf(k) for k in range(_r))
        _warm = f"--warm-start {w47_agent(_r - 1)} --warm-strict --bn-mode train" if _r else ""
        _wtags = [f"{Q}_scr_r{_r}_w{_i}" for _i in range(W47_W)]
        for _i, _wt in enumerate(_wtags):
            add(f"w47-r{_r}-w{_i}", 47, "H06",
                [(TRAIN, f"{W47_BASE} --hours 10 --save-agent --save-every 5 "
                         f"--save-chains {w47_buf(_r).split('/')[-1]} {_warm} "
                         + (f"--preload-online {_prev} " if _prev else "")
                         + f"--eps-start {_e0} --eps-end {_e1} --eps-decay {_ed} --seed {100 * (_r + 1) + _i} "
                           f"--tag {_wt}")],
                10.5, needs=[w47_agent(_r - 1)] if _r else [],
                note=f"с нуля онлайн, раунд {_r}, работник {_i}: "
                     + (f"старт от ученика раунда {_r - 1}, в буфере раунды 0..{_r - 1}" if _r else "сеть с нуля")
                     + f", eps {_e0} -> {_e1}")
        _done = f"{w47_buf(_r)}_done"
        _src = ",".join(w47_buf(k) for k in range(_r + 1))
        _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
        _eh = lambda sd: (0.4 if NS2D else 0.3) + W47_SEED_H * len(sd.split(","))   # noqa: E731
        add(f"w47-r{_r}-learn", 47, "H06",
            [("experiments/rl_arch/wait_hf.py", f"--path {_done} --n {W47_W} --min {W47_W - 2} --hours 0.5"),
             (TRAIN, f"{W47_BASE} --hours 0 --learn-only --preload-online {_src} --pre-updates {W47_K} "
                     f"{_warm} --seed 1 "
                     f"--tag {Q}_scratch_r{_r}"),
             free_eval(w47_agent(_r), f"t3m_scratch_r{_r}", _blocks[0])],
            round(0.5 + 0.3 + _eh(_blocks[0]), 2),
            needs=[f"{_done}/{t}.json" for t in _wtags[:W47_W - 2]] + ([w47_agent(_r - 1)] if _r else []),
            note=f"с нуля онлайн, ученик раунда {_r}: {W47_K} обновлений по раундам 0..{_r}, агент в HF, "
                 f"оценка на сидах {_blocks[0]}")
        if len(_blocks) > 1:
            add(f"w47-r{_r}-ev-b", 47, "H06", [free_eval(w47_agent(_r), f"t3m_scratch_r{_r}", _blocks[1])],
                round(_eh(_blocks[1]), 2), needs=[w47_agent(_r)],
                note=f"с нуля онлайн, агент ученика раунда {_r}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 48
# Две сети без потери на pb2d. Сумма additive (лидер ns2d) на pb2d хуже одной сети в 1.5-2 раза: она не
# выражает взаимодействие шага и длительности. Три варианта, у которых число выходов растёт как сумма,
# а не произведение числа значений параметров: дуэлинговая голова (одно обучение волны 33 дало 0.0169),
# последовательный выбор со своим энкодером (с общим энкодером волна 36 дала 0.031 и 0.148) и сумма плюс
# взаимодействие ранга 1 (--two-nets-mode lowrank). По три обучения (сид обучения 1-3, учитель двух сетей
# волны 33), каждое на сидах 42-46. Сначала pb2d; на ns2d — победитель.
W48_ARMS = (("duel", "--two-nets --two-nets-head dueling"),
            ("auto", "--two-nets --two-nets-mode auto"),
            ("lowrank", "--two-nets --two-nets-mode lowrank"))
if PREFIX == "pb2d":
    for _nm, _fl in W48_ARMS:
        for _m in (1, 2, 3):
            _mt = f"w48{_nm}"
            add(f"w48-{_nm}-m{_m}", 48, "H06",
                [(OFFLINE, f"--variant convnext_dqn {MAP_BASE} {RECIPE} --target-buffer {W33_SRC} "
                           f"--teacher-from {W41_TEACH['two']} --teacher-w 0.5 {_fl} --seeds {_m} --model-tag {_mt}"),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", S5)],
                1.0 + 0.6 * MAP_EH * 2, needs=[ENC_MAE, W41_TEACH["two"]],
                note=f"две сети ({_nm}), сид обучения {_m}, сиды 42-46")

# ---------------------------------------------------------------- волна 50
# Составляющие метода (обзор 15 октября, paper/lit_review/work/track3/areas_verdicts/components_2026-10-15.tsv):
# четыре правки обучения лидера, по одной, против разброса между обучениями и узкого места данных (~130 переходов
# цели). Все флаги offline_rl.py, по умолчанию выключены:
#   wa   — развёртывается экспоненциальное среднее весов второй половины обучения (--wa ema --wa-start 0.5,
#          распад 0.99; SWA, Izmailov et al. UAI 2018; EMA весов, Morales-Brotons et al. TMLR 2024): меньше
#          дрожания жадной политики к концу обучения (policy churn, Schaul et al. NeurIPS 2022);
#   all  — учиться на всех переходах цели (--train-all): 20% цепочек сейчас отложены только ради диагностик,
#          а отложенный TD для отбора бесполезен (Paine et al. 2020; Kumar et al. CoRL 2021);
#   lb   — нижняя граница Q записанного действия по исходу его цепочки (--mc-lb 1.0: штраф relu(G - Q)^2 / 2;
#          SIL, Oh et al. ICML 2018; optimality tightening, He et al. ICLR 2017; MCAC, Wilcox et al. NeurIPS 2022):
#          пессимизм MOPO в модельных целях занижает Q хороших записанных цепочек, настоящий исход его поправляет;
#   tens — ансамбль из трёх учителей (сид 0 — учитель лидера, сиды 1 и 2 — новые, тот же рецепт учителя на том
#          же собранном буфере), цель учителя — среднее их Q (дистилляция ансамбля, Hinton et al. 2015;
#          Allen-Zhu, Li ICLR 2023).
# У каждой правки два обучения (сиды обучения 1 и 2) лидера (pb2d — одна сеть, ns2d — две), оценка на сидах 42-46
# (страж, без сбросов, бюджет 7000, все 27 действий). На ns2d сиды 45-46 — отдельным кернелом по модели из HF.
# Контроль — лидер с MAE волн 41/43: pb2d t3m_ete_tbquart_g, t3m_w41one_m2_g, t3m_w41one_m3_g; ns2d t3m_ete_two_g,
# t3m_w41two_m2_g, t3m_w41two_m3_g. Новый сбор (Thompson вместо LCB) сюда не входит: 8.4 / 17.1 GPU-ч на буфер.
W50_ARMS = (("wa", "--wa ema --wa-start 0.5 --wa-decay 0.99", "среднее весов второй половины обучения (EMA 0.99)"),
            ("all", "--train-all", "обучение на всех переходах цели (без отложенных 20% цепочек)"),
            ("lb", "--mc-lb 1.0", "нижняя граница Q по исходу записанной цепочки (штраф, вес 1)"),
            ("tens", "", "ансамбль трёх учителей (сиды 0, 1, 2), среднее Q"))
W50_SEED_H = 1.4 if NS2D else 0.7
W50_TEACH = [f"rl_arch/agents_online/{TEACH}_w50s{_k}.pt" for _k in (1, 2)]
if W28_K:
    _arch = W41_LEAD
    add("w50-teach", 50, "H06",
        [(COMBO, f"{TEACH_TR.replace('--seed 0', f'--seed {_k}')} --target-buffer {W33_SRC} --tag {TEACH}_w50s{_k}")
         for _k in (1, 2)],
        1.5, note="два учителя MBPO на скалярах с сидами 1 и 2 (рецепт и собранный буфер учителя лидера), в HF")
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W50_SEED_H * len(sd.split(","))   # noqa: E731
    for _nm, _fl, _what in W50_ARMS:
        _mt = f"w50{_nm}{_arch}"
        _needs = [ENC_MAE, W41_TEACH[_arch]]
        for _m in (1, 2):
            _scr, _a = w41_train(_arch, _m, _mt)
            if _nm == "tens":
                assert f"--teacher-from {W41_TEACH[_arch]} " in _a
                _a = _a.replace(f"--teacher-from {W41_TEACH[_arch]} ",
                                f"--teacher-from {','.join([W41_TEACH[_arch]] + W50_TEACH)} ")
                _needs_m = _needs + W50_TEACH
            else:
                _a = f"{_a} {_fl}"
                _needs_m = _needs
            add(f"w50-{_nm}-m{_m}", 50, "H06",
                [(_scr, _a), free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=_needs_m,
                note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w50-{_nm}-m{_m}-b", 50, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 49
# Замены MAE в предобучении энкодера лидера (обзор 15 октября: что лучше MAE при ~130 переходах цели и большом
# многозадачном пуле с действиями и исходами). Кандидаты (cv_probe.py, конфиги ssl-<вид>-resnet):
#   vae    — β-VAE, восстановление всей нормированной карты (уровень и форма) через латент 32, β = 1. Слабая
#            сторона по литературе (SAC-AE, ATC: VAE хуже детерминированного автокодировщика и ATC) — проверка
#            по просьбе: у нас карты гладкие, размытие VAE им почти не вредит;
#   dyn    — самопредсказание с действием, SPR + обратная динамика (SGI, NeurIPS 2021: обратная динамика —
#            главный член, без неё латентный прогноз схлопывается): по картам и действию предсказывается
#            проекция следующих карт цепочки от EMA-энкодера, по паре карт — действие; общий элемент D4;
#   out    — исход с учителем по 22 задачам: изменение log10 ошибки после действия по картам, действию и
#            скалярам (без текущей ошибки — у агента её нет); ближе всего к тому, что учит голова агента;
#   maeout — MAE + та же голова исхода (как MAE + предсказание награды в Masked World Models, CoRL 2022):
#            сохраняет то, что уже даёт MAE, и добавляет признаки исхода.
# Плюс контрастный энкодер, который уже лежит в HF (ENC_CON: SimCLR на том же пуле волны 21, те же 4 эпохи, что
# у ENC_MAE): в агента он шёл только замороженным (волны 21-22), с дообучением в рецепте лидера — ни разу.
# Предобучение — тот же пул, что у ENC_MAE (карты всех 22 задач, --subdirs all, цель не исключена), то же число
# шагов (--ssl-steps 1400 = 4 эпохи по ~350 пачек из 512 у ENC_MAE), тот же ResNet, нормировка и оптимизатор;
# кернел на (кандидат, задача) выгружает энкодер в HF rl_arch/encoders/. Затем лидер (pb2d — одна сеть, ns2d —
# две) учится с ним дважды (сиды обучения 1 и 2, тот же учитель и собранный буфер, что в волнах 41/46) и
# оценивается на сидах 42-46 (страж, без сбросов, бюджет 7000); на ns2d сиды 45-46 — отдельным кернелом по
# модели из HF. Контроль — лидер с ENC_MAE на тех же сидах (pb2d t3m_ete_tbquart_g, t3m_w41one_m2/m3_g; ns2d
# t3m_ete_two_g, t3m_w41two_m2/m3_g), нового контроля нет. Разброс между обучениями до x2.6: по двум обучениям
# видны только большие эффекты (около x1.3 и больше); кандидат с выигрышем идёт на подтверждение.
W49_ARMS = ("vae", "dyn", "out", "maeout")
W49_PRE_H = 1.5                          # загрузка буферов 22 задач (~195 тыс. наборов карт) и 1400 шагов
if W28_K:
    _arch = W41_LEAD
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W46_SEED_H * len(sd.split(","))   # noqa: E731
    for _nm in W49_ARMS + ("con",):
        if _nm == "con":
            _enc = ENC_CON
        else:
            _cvtag = f"w49{_nm}{W28_SHORT}"
            _enc = f"rl_arch/encoders/cvprobe_{_cvtag}_enc_{_nm}_resnet.pt"
            add(f"w49-pre-{_nm}", 49, "H06",
                [(CVPROBE, f"--tag {_cvtag} --subdirs all --configs ssl-{_nm}-resnet --pretrain-only "
                           f"--ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
                W49_PRE_H, note=f"предобучение {_nm} вместо MAE на картах 22 задач, {W46_SSL_STEPS} шагов, энкодер в HF")
        _mt = f"w49{_nm}{_arch}"
        for _m in (1, 2):
            add(f"w49-{_nm}-m{_m}", 49, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"лидер ({_arch}) с энкодером {_nm} вместо MAE, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w49-{_nm}-m{_m}-b", 49, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"лидер ({_arch}) с энкодером {_nm} вместо MAE, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 51
# Страж застоя (online_eval_env.py --stall-guard). На pb2d у ConvNeXt без MAE (t3m_ete_cnx_g) сиды 43 и 44
# тратят конец бюджета на долгий Adam 1e-4 (2500 и 1000 эпох), ошибка почти стоит (0.122 -> 0.116), а страж с
# откатом молчит: лосс чуть падает. Страж застоя проверяет действие длиннее окна (250 эпох) на каждой границе
# окна и, если обучающий лосс за окно упал меньше чем на 1%, отдаёт остаток эпох действия запасному L-BFGS 1.0
# x 500 (кусками). Эпохи шага и бюджет прежние, агенту доступны все 27 действий, только обучающий лосс. Все оценки
# — со стражем с откатом и без сбросов (safe), как у лидеров, плюс страж застоя (тег _sg); сиды 42-46.
#   pb2d: готовый ConvNeXt без MAE (пара к t3m_ete_cnx_g) и готовый лидер (пара к t3m_ete_tbquart_g);
#         два новых обучения ConvNeXt без предобучения (сиды обучения 2 и 3, одна сеть, учитель лидера).
#   ns2d: готовый лидер (две сети, пара к t3m_ete_two_g); два обучения лидера с ConvNeXt без предобучения
#         (две сети, сиды обучения 1 и 2) — сиды 42-44 вместе с обучением, 45-46 по модели из HF.
# Сид pb2d ~0.75 GPU-ч, ns2d ~1.5 GPU-ч; кернел не больше 9 часов.
STALLG = "--stall-guard --stall-window 250 --stall-min 0.01"
W51_SEED_H = 1.5 if NS2D else 0.75


def sg_eval(model, tag, sd):
    """Оценка со стражем с откатом, без сбросов (safe) и со стражем застоя; тег с суффиксом _sg."""
    return (EVAL, f"--policy agent --model-file {model} {GUARD} {STALLG} {KEEPS} "
                  + t3ev(tag + "_sg", sd, hours=MAP_EH))


def w51_train(arch, m, mt):
    """Рецепт лидера (w41_train) с ConvNeXt без предобучения: --encoder convnext без --encoder-init."""
    scr, a = w41_train(arch, m, mt)
    old = f"--encoder resnet --encoder-init {ENC_MAE}"
    assert a.count(old) == 1, a
    return scr, a.replace(old, "--encoder convnext")


if W28_K and (NS2D or PREFIX == "pb2d"):
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W51_SEED_H * len(sd.split(","))   # noqa: E731
    _ready = ((("lead", f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_two_seed1.pt", "лидер (две сети)"),)
              if NS2D else
              (("cnx", "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ete_cnx_seed1.pt", "ConvNeXt без MAE (ete_cnx)"),
               ("lead", "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ete_tbquart_seed1.pt", "лидер (ete_tbquart)")))
    for _nm, _ck, _what in _ready:
        for _i, _sd in enumerate(_blocks):
            _sfx = "" if len(_blocks) == 1 else "-" + "ab"[_i]
            add(f"w51-{_nm}{_sfx}", 51, "H06", [sg_eval(_ck, f"t3m_w51{_nm}", _sd)], round(_eh(_sd), 2),
                needs=[_ck], note=f"страж застоя: готовый {_what}, сиды {_sd}")
    _arch = "two" if NS2D else "one"
    _mt = f"w51cnx{_arch}"
    for _m in ((1, 2) if NS2D else (2, 3)):
        add(f"w51-cnx{_arch}-m{_m}", 51, "H06",
            [w51_train(_arch, _m, _mt),
             sg_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
            round(1.0 + _eh(_blocks[0]), 2), needs=[W41_TEACH[_arch]],
            note=f"рецепт ({_arch}) с ConvNeXt без предобучения, сид обучения {_m}, страж застоя, сиды {_blocks[0]}")
        if len(_blocks) > 1:
            add(f"w51-cnx{_arch}-m{_m}-b", 51, "H06",
                [sg_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                note=f"рецепт ({_arch}) с ConvNeXt без предобучения, сид обучения {_m}, страж застоя, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 52
# Дообучение энкодера с малым числом параметров (offline_rl.py --ft-mode). Лидер дообучает весь ResNet с MAE
# (~416 тыс. параметров энкодера, у двух сетей — вдвое) на ~130 переходах цели, и одно обучение из трёх выходит
# плохим (pb2d 0.0118, 0.0123, 0.0319): похоже на переобучение энкодера. Три режима вместо полного дообучения:
#   bn        — учатся только аффинные параметры BatchNorm (BitFit для норм; Frankle et al. ICLR 2021), 1.6 тыс.;
#   lora      — веса энкодера стоят, адаптеры LoRA ранга 4 (alpha 8) к каждой свёртке и линейному слою плюс
#               аффинные параметры BatchNorm (Hu et al. ICLR 2022), 24 тыс.;
#   lastblock — учится последний остаточный блок (стадия 7x7) и проекция 128 -> 256, 263 тыс.; остальное —
#               неизменная функция (статистики BatchNorm стоят, нормировка карт из файла MAE).
# У всех режимов головы учатся целиком, первая свёртка (свежие каналы контекста) — тоже (+2.6 тыс.); статистики
# BatchNorm в bn и lora копятся, как при полном дообучении. Лидер (pb2d — одна сеть, ns2d — две; режим действует на
# оба энкодера) учится дважды на режим (сиды обучения 1 и 2, тот же учитель и собранный буфер, что в волне 41),
# оценка на сидах 42-46 (страж, без сбросов, бюджет 7000, все 27 действий); на ns2d сиды 45-46 — отдельным
# кернелом по модели из HF. Контроль — лидер с MAE и полным дообучением на тех же сидах: pb2d t3m_ete_tbquart_g,
# t3m_w41one_m2_g, t3m_w41one_m3_g; ns2d t3m_ete_two_g, t3m_w41two_m2_g, t3m_w41two_m3_g. Решение: режим лучше,
# если медиана по двум обучениям не хуже контроля и разброс между обучениями меньше.
W52_MODES = (("bn", "только аффинные параметры BatchNorm энкодера"),
             ("lora", "адаптеры LoRA ранга 4 (alpha 8) и BatchNorm, веса энкодера стоят"),
             ("lastblock", "только последний блок ResNet и проекция"))
W52_SEED_H = 1.5 if NS2D else 0.75
if W28_K:
    _arch = W41_LEAD
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W52_SEED_H * len(sd.split(","))   # noqa: E731
    for _mode, _what in W52_MODES:
        _mt = f"w52{_mode}{_arch}"
        for _m in (1, 2):
            _scr, _a = w41_train(_arch, _m, _mt)
            add(f"w52-{_mode}-m{_m}", 52, "H06",
                [(_scr, f"{_a} --ft-mode {_mode}"),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[ENC_MAE, W41_TEACH[_arch]],
                note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w52-{_mode}-m{_m}-b", 52, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[1]}")

MODAL ="Adam:0.001:2500,LBFGS:0.5:1000,LBFGS:1:500"

# ---------------------------------------------------------------- волна 53
# Сколько карт других уравнений нужно MAE. ENC_MAE учился на ~180 тыс. карт 22 уравнений (сбор таких буферов —
# тысячи GPU-часов). Здесь MAE учится на тех же 22 уравнениях, но на малом числе карт каждого: 30, 100 и 300
# переходов на уравнение (--max-per-pde, цепочки целиком; около 660, 2 200 и 6 600 карт всего). Число шагов
# предобучения то же, что у ENC_MAE и волны 46 (1400). Предобучение не зависит от целевого уравнения, поэтому
# кернел предобучения один на объём (в очереди pb2d), а обучения обоих уравнений ждут энкодер в HF. Дальше как в
# волне 46: лидер дважды (сиды обучения 1, 2), оценка на сидах 42-46. Контроль — лидер с ENC_MAE.
W53_SIZES = ((30, "n30"), (100, "n100"), (300, "n300"))
if W28_K:
    _arch = W41_LEAD
    for _n, _sz in W53_SIZES:
        _cvtag = f"w53{_sz}"
        _enc = f"rl_arch/encoders/cvprobe_{_cvtag}_enc_mae_resnet.pt"
        if PREFIX == "pb2d":
            add(f"w53-pre-{_sz}", 53, "H06",
                [(CVPROBE, f"--tag {_cvtag} --subdirs all --max-per-pde {_n} --configs ssl-mae-resnet --pretrain-only "
                           f"--ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
                1.0, note=f"MAE на 22 уравнениях, по {_n} переходов на уравнение, {W46_SSL_STEPS} шагов, энкодер в HF")
        _mt = f"w53{_sz}{_arch}"
        for _m in (1, 2):
            _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
            _eh = lambda sd: (0.4 if NS2D else 0.3) + W46_SEED_H * len(sd.split(","))   # noqa: E731
            add(f"w53-{_sz}-m{_m}", 53, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"лидер ({_arch}) с MAE на {_n} переходах каждого из 22 уравнений, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w53-{_sz}-m{_m}-b", 53, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"лидер ({_arch}) с MAE на {_n} переходах каждого из 22 уравнений, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 54
# MAE на минимуме данных: мало уравнений и мало переходов в каждом, и ни одного из двух целевых (pb2d, ns2d) —
# чтобы ответ «кодировщику нужны буферы 22 уравнений» был проверен в самой строгой форме. Уравнения разных
# типов (без нерешаемых и без целей): 3 — burgers1d (нелинейное 1D), heat2d_complexgeometry (параболическое),
# poisson2d_classic (эллиптическое); 6 — плюс wave1d (гиперболическое), grayscott (реакция-диффузия),
# ns2d_backstep (течение). Три объёма: 3 x 100, 6 x 50 (тот же объём, больше разнообразия) и 6 x 150. Шагов
# предобучения столько же, сколько у ENC_MAE (1400). Один кернел предобучения на объём (в очереди pb2d), затем
# лидер дважды на каждом уравнении, оценка на сидах 42-46, как в волнах 46 и 53.
W54_P3 = ("burgers1d", "heat2d_complexgeometry", "poisson2d_classic")
W54_P6 = W54_P3 + ("wave1d", "grayscott", "ns2d_backstep")
W54_ARMS = (("p3n100", W54_P3, 100), ("p6n50", W54_P6, 50), ("p6n150", W54_P6, 150))
if W28_K:
    _arch = W41_LEAD
    for _sz, _subs, _n in W54_ARMS:
        _cvtag = f"w54{_sz}"
        _enc = f"rl_arch/encoders/cvprobe_{_cvtag}_enc_mae_resnet.pt"
        if PREFIX == "pb2d":
            add(f"w54-pre-{_sz}", 54, "H06",
                [(CVPROBE, f"--tag {_cvtag} --subdirs {','.join(_subs)} --max-per-pde {_n} --configs ssl-mae-resnet "
                           f"--pretrain-only --ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
                0.8, note=f"MAE на {len(_subs)} уравнениях без целевых, по {_n} переходов, {W46_SSL_STEPS} шагов")
        _mt = f"w54{_sz}{_arch}"
        for _m in (1, 2):
            _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
            _eh = lambda sd: (0.4 if NS2D else 0.3) + W46_SEED_H * len(sd.split(","))   # noqa: E731
            add(f"w54-{_sz}-m{_m}", 54, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"лидер ({_arch}) с MAE на {len(_subs)} уравнениях x {_n} переходов, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w54-{_sz}-m{_m}-b", 54, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"лидер ({_arch}) с MAE на {len(_subs)} уравнениях x {_n} переходов, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 56
# Сколько и каких данных нужно MAE, без обучения агентов (ответ рецензенту о цене сбора буферов). Кернел study
# (mae_data_study.py): A — распределения карт 22 уравнений (PCA, t-SNE/UMAP, MMD, Фреше, кластеры, покрытие
# k-NN, жадное ядро уравнений без целей, отбор карт k-center, избыточные пары, отдельно пара хаотических) в
# пространствах сырых карт, формы и признаков ENC_MAE; B — ~50 коротких прогонов MAE (тот же ResNet, цель и
# 1400 шагов, что у ENC_MAE): каждое уравнение отдельно, ядра 1-6 уравнений по 2000 переходов, лучшее ядро на
# 10-1000 переходах, повторы армов волн 46/53/54 для сверки прокси с агентом, наборы без хаотических/нерешаемых/
# избыточных, случайные наборы, второй сид; прокси — восстановление отложенных карт целей и невиденных
# уравнений, CKA с ENC_MAE и с полным набором, линейный зонд исхода. Кернелы cost: цена перехода с картами для
# каждого уравнения (цепочка Adam 1000 / L-BFGS 500 / PSO 200 той же средой, карты после 1-го и 3-го действия),
# 11 уравнений на кернел, poisson_boltzmann_2d в обоих — для приведения GPU разных кернелов. Предобучение и
# цена не зависят от цели, поэтому кернелы только в очереди pb2d. Всё — в HF rl_arch/mae_study/ (энкодеры армов
# — rl_arch/mae_study/encoders/, формат --encoder-init); отчёт — mae_study_report.py --tag w56 --cost-tags w56a,w56b.
MAESTUDY = "experiments/rl_arch/mae_data_study.py"
W56_COST = (("a", ("poisson_boltzmann_2d",) + ALL_SUBDIRS22[0::2]), ("b", ("poisson_boltzmann_2d",) + ALL_SUBDIRS22[1::2]))
if PREFIX == "pb2d":
    add("w56-study", 56, "H06",
        [(MAESTUDY, f"study --tag w56 --hours 8 --resume --out-dir {W33_CV}")],
        5.0, needs=[ENC_MAE], note="MAE и данные: распределения карт 22 уравнений, ядро, ~50 коротких MAE с прокси")
    for _k, _subs in W56_COST:
        _subs = tuple(dict.fromkeys(_subs))
        add(f"w56-cost-{_k}", 56, "H06",
            [(MAESTUDY, f"cost --tag w56{_k} --pdes {','.join(_subs)} --hours 8 --timeout-min 45 --resume "
                        f"--out-dir {W33_CV}")],
            round(0.3 * len(_subs), 1), note=f"цена перехода с картами: {len(_subs)} уравнений (вместе с pb2d для сверки GPU)")
# ---------------------------------------------------------------- волна 55
# Энкодер без буферов других уравнений: данные для самообучения даёт сам автокодировщик карт. Скрытая плоскость
# автокодировщика статьи ничем не закреплена, поэтому другой случайный старт на тех же копиях весов PINN даёт другой
# вид (поворот, отражение, масштаб, изгиб) того же ландшафта: V видов одного состояния стоят V автокодировщиков
# (сложены в один проход, map_views.train_view_aes) и V поверхностей, без обучения PINN. Тот же шум агент видит
# при развёртывании: вторая сборка карт меняла решение лидера ns2d на 0-60% шагов (волна 10, t3a_*_reb).
# Состояния — повторный прогон цепочек собранного буфера цели (W33_SRC, на нём учится лидер; render_views.py: до
# последней неплоской карты, виды только неплоских и ещё не встреченных состояний). Плоских карт (все копии весов
# совпали: застой L-BFGS, PSO) в tbuf_pb2d_w0 55%, в tbuf_ns2d_w0/w1 83% — у ns2d неплоские только первые две
# карты цепочки, поэтому у ns2d вторым кернелом ещё случайные цепочки цели со стопом после двух плоских карт.
# Для новой задачи повтор не нужен: collect_active.py --views строит виды во время направленного сбора.
# Армы (лидер дважды, сиды обучения 1 и 2, оценка на сидах 42-46, как в волнах 46/53/54):
#   mvmae — MAE на всех видах цели (одна задача, ~V+1 карт на неплоское состояние);
#   mvvic — MAE + согласованность видов одного состояния (VICReg, Bardes et al., ICLR 2022): признаки не
#           зависят от случайности автокодировщика;
#   proc  — MAE на процедурных картах (гауссовы поля, чаша или седло, рост к краю, плоские карты застоя; Baradad
#           et al., NeurIPS 2021): ноль уравнений, один кернел предобучения на оба уравнения (в очереди pb2d).
# Контроли уже есть: лидер с ENC_MAE (22 уравнения), волна 46 (MAE на 130/1000/всех картах случайного буфера цели),
# без предобучения t3m_ete_resnp_g. Кернел предобучения видов пишет ещё энкодер MAE на одной карте среды на
# состояние (--views-max 1, тег w55mv1<задача>): контроль «виды или сами состояния» — обучение по нему не в волне.
RENDER = "experiments/rl_arch/render_views.py"
W55_V = 12
W55_TAG = f"mv_{W28_SHORT}"
W55_SEED_H = 1.4 if NS2D else 0.7
if W28_K:
    _arch = W41_LEAD
    _vt = W55_TAG
    # кернелы видов: pb2d — два работника по сидам исходного буфера; ns2d — повтор целиком и случайные цепочки
    _ren = ((("0", f"--source {W33_SRC} --worker 0/1 --hours 5.0", 6.0,
              "повтор цепочек собранного буфера до второй карты (дальше плоские), виды неплоских состояний"),
             ("1", f"--random 5 --random-len 6 --flat-stop 2 {KEEPS} --worker 1/2 --hours 6.0", 8.0,
              "свежие случайные цепочки цели (все 27 действий) со стопом после двух плоских карт"))
            if NS2D else
            tuple((str(_k), f"--source {W33_SRC} --worker {_k}/2 --hours 4.0", 5.0,
                   f"повтор цепочек собранного буфера (половина по сидам, {_k} из 2) до последней неплоской карты")
                  for _k in range(2)))
    _done = []
    for _k, _a, _h, _what in _ren:
        add(f"w55-ren-{_k}", 55, "H06",
            [(RENDER, f"--pde {PDE} {_a} --views {W55_V} --tag {_vt} --resume --save-dir runs_rl_online")],
            _h, note=f"виды карт {SUBDIR} (V={W55_V}): {_what}")
        _done.append(f"rl_arch/online_env/{PDE}_{_vt}_w{_k}_views_done.json")
    _cvtag, _cv1 = f"w55mv{W28_SHORT}", f"w55mv1{W28_SHORT}"
    add("w55-pre", 55, "H06",
        [(CVPROBE, f"--tag {_cvtag} --subdirs {SUBDIR} --views rl_arch/views/{_vt} --configs ssl-mae-resnet,ssl-mv-resnet "
                   f"--pretrain-only --ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}"),
         (CVPROBE, f"--tag {_cv1} --subdirs {SUBDIR} --views rl_arch/views/{_vt} --views-max 1 --configs ssl-mae-resnet "
                   f"--pretrain-only --ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
        1.0, needs=_done, note=f"MAE и MAE+VICReg на видах карт {SUBDIR} ({W46_SSL_STEPS} шагов); контроль — MAE на "
                               "одной карте среды на состояние")
    if PREFIX == "pb2d":
        add("w55-pre-proc", 55, "H06",
            [(CVPROBE, f"--tag w55proc --procedural 5000 --configs ssl-mae-resnet --pretrain-only "
                       f"--ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}")],
            0.6, note=f"MAE на 5000 процедурных цепочках карт (40 тыс. карт, ни одного уравнения), {W46_SSL_STEPS} шагов")
    _arms = (("mvmae", f"rl_arch/encoders/cvprobe_{_cvtag}_enc_mae_resnet.pt", "MAE на видах карт цели"),
             ("mvvic", f"rl_arch/encoders/cvprobe_{_cvtag}_enc_mv_resnet.pt", "MAE + согласованность видов (VICReg)"),
             ("proc", "rl_arch/encoders/cvprobe_w55proc_enc_mae_resnet.pt", "MAE на процедурных картах (ноль уравнений)"))
    for _nm, _enc, _what in _arms:
        _mt = f"w55{_nm}{_arch}"
        for _m in (1, 2):
            _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
            _eh = lambda sd: (0.4 if NS2D else 0.3) + W55_SEED_H * len(sd.split(","))   # noqa: E731
            add(f"w55-{_nm}-m{_m}", 55, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"лидер ({_arch}), энкодер: {_what}, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w55-{_nm}-m{_m}-b", 55, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"лидер ({_arch}), энкодер: {_what}, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 57
# Лидер ns2d (две сети) без отката при росте лосса: агент статьи оценивался без отката, и если без него
# лучший метод не хуже, метод проще и сравнение симметрично. На pb2d это уже есть (t3m_ete_tbquart_free:
# медиана 0.0117 против 0.0118 с откатом, худший сид 0.0366 против 0.0181). Сиды 42-51, два кернела.
if NS2D:
    _m57 = "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_ete_two_seed1.pt"
    for _nm, _sd in (("a", "42,43,44,45,46"), ("b", "47,48,49,50,51")):
        add(f"w57-two-free-{_nm}", 57, "H06", [free_eval(_m57, "t3m_ete_two", _sd, guard=False)], 8.5,
            needs=[_m57], note=f"лидер ns2d (две сети) без отката, сиды {_sd}")

# ---------------------------------------------------------------- волна 58
# Разброс между обучениями лидера (одно из трёх обучений выходит плохим) — две правки обучения, без новых данных:
#   viewaug — обучение Q на видах карт (offline_rl.py --view-aug): у перехода собранного буфера, состояние которого
#             (сид + исполненное начало цепочки + шаг) есть в файлах видов волны 55 (rl_arch/views/mv_<задача>,
#             render_views.py: V = 12 автокодировщиков на тех же копиях весов), карты s и независимо s' с
#             вероятностью 0.5 заменяются случайным видом; канал delta — по тому же виду предыдущего шага. Вторая
#             сборка карт того же состояния меняет действие лидера ns2d на 22-63% шагов: агент учится на том же
#             шуме наблюдения, что видит при развёртывании (DrQ, Kostrikov et al. ICLR 2021: аугментация
#             наблюдений в офлайновом Q). Плоские карты видов не имеют (у ns2d виды только у первых двух карт
#             цепочки), модельные цели и цели учителя от карт не зависят. Ждёт меток конца рендера волны 55;
#   cos1k   — дольше и с затуханием шага: буфер (~130 переходов) меньше пачки (256), обучение — 200 шагов Adam
#             по всей выборке с постоянным шагом 1e-3, и развёртывается последняя шумная итерация; здесь 1000 эпох
#             с косинусным спадом шага до нуля (--epochs 1000 --lr-schedule cosine; Loshchilov, Hutter ICLR 2017).
# У каждой правки два обучения лидера (pb2d — одна сеть, ns2d — две; сиды обучения 1 и 2, тот же учитель и
# собранный буфер, что в волнах 41/50), оценка на сидах 42-46 (страж, без сбросов, бюджет 7000, все 27 действий);
# на ns2d сиды 45-46 — отдельным кернелом по модели из HF (кернел не больше 9 часов). Контроль — лидер с MAE волн
# 41/43: pb2d t3m_ete_tbquart_g, t3m_w41one_m2_g, t3m_w41one_m3_g; ns2d t3m_ete_two_g, t3m_w41two_m2_g,
# t3m_w41two_m3_g. Среднее весов (--wa ema) и обучение на всех переходах (--train-all) — в волне 50.
W58_REN_DONE = [f"rl_arch/online_env/{PDE}_{W55_TAG}_w{_k}_views_done.json" for _k in (0, 1)]
W58_ARMS = (("viewaug", f"--view-aug rl_arch/views/{W55_TAG} --view-aug-p 0.5", 1.0,
             "Q на видах карт волны 55 (замена с вероятностью 0.5)"),
            ("cos1k", "--epochs 1000 --lr-schedule cosine", 1.2, "1000 эпох, косинусный спад шага Adam до нуля"))
W58_SEED_H = 1.4 if NS2D else 0.7
if W28_K:
    _arch = W41_LEAD
    # метки конца рендера видов — те же, которых ждёт предобучение волны 55
    _pre55 = next(j for j in jobs if j["id"] == (f"{PREFIX}-w55-pre" if PREFIX else "w55-pre"))
    assert sorted(_pre55["needs"]) == sorted(W58_REN_DONE), (_pre55["needs"], W58_REN_DONE)
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W58_SEED_H * len(sd.split(","))   # noqa: E731
    for _nm, _fl, _th, _what in W58_ARMS:
        _mt = f"w58{_nm}{_arch}"
        _needs = [ENC_MAE, W41_TEACH[_arch]] + (W58_REN_DONE if _nm == "viewaug" else [])
        for _m in (1, 2):
            _scr, _a = w41_train(_arch, _m, _mt)
            assert "--epochs" not in _a and "--view-aug" not in _a, _a
            add(f"w58-{_nm}-m{_m}", 58, "H06",
                [(_scr, f"{_a} {_fl}"), free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(_th + _eh(_blocks[0]), 2), needs=_needs,
                note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w58-{_nm}-m{_m}-b", 58, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"{_what}: лидер ({_arch}), сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 59
# Подтверждение двух находок очереди 1 (18 октября). pb2d: ConvNeXt без MAE с обрезкой застрявшего действия дал
# 0.0089 / 0.0112 / 0.0115 на сидах 42-46 у трёх агентов — добор сидов 47-51 у всех трёх. ns2d: MAE на
# процедурных картах (ноль буферов) дал 0.0437 / 0.0558, на уровне лидера — третье обучение и добор сидов 47-51
# у всех трёх. Контроль обрезки на лидерах — задачи волны 51 (w51-lead*), ставятся вместе с этой волной.
if PREFIX == "pb2d":
    for _nm, _ck, _tg in (("cnx", "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ete_cnx_seed1.pt", "t3m_w51cnx"),
                          ("cnxone-m2", w41_hf_model("w51cnxone", 2), "t3m_w51cnxone_m2"),
                          ("cnxone-m3", w41_hf_model("w51cnxone", 3), "t3m_w51cnxone_m3")):
        add(f"w59-{_nm}-c", 59, "H06", [sg_eval(_ck, _tg, "47,48,49,50,51")], 4.1, needs=[_ck],
            note=f"ConvNeXt без MAE с обрезкой застрявшего действия, добор сидов 47-51 ({_tg})")
if NS2D:
    _arch = W41_LEAD
    _mt = f"w55proc{_arch}"
    _enc = "rl_arch/encoders/cvprobe_w55proc_enc_mae_resnet.pt"
    add("w59-proc-m3", 59, "H06",
        [w41_train(_arch, 3, _mt, init=_enc), free_eval(off_model("convnext_dqn", _mt, 3), f"t3m_{_mt}_m3", "42,43,44")],
        5.6, needs=[_enc, W41_TEACH[_arch]], note="процедурные карты, третье обучение, сиды 42-44")
    add("w59-proc-m3-b", 59, "H06", [free_eval(w41_hf_model(_mt, 3), f"t3m_{_mt}_m3", "45,46")], 3.2,
        needs=[w41_hf_model(_mt, 3)], note="процедурные карты, третье обучение, сиды 45-46")
    for _m in (1, 2, 3):
        add(f"w59-proc-m{_m}-c", 59, "H06", [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", "47,48,49,50,51")],
            8.0, needs=[w41_hf_model(_mt, _m)], note=f"процедурные карты, обучение {_m}, добор сидов 47-51")

# ---------------------------------------------------------------- волна 60
# pb2d только с двумя сетями (требование: число выходов растёт линейно при добавлении параметров). Лучшее на
# pb2d с одной сетью — ConvNeXt без MAE с обрезкой застрявшего действия. Здесь: (1) ConvNeXt без MAE с двумя
# сетями — суммой (как лидер ns2d) и последовательным выбором (лучший из вариантов двух сетей на pb2d), по два
# обучения, оценка с обрезкой на сидах 42-46; (2) обрезка на готовых агентах pb2d с двумя сетями и MAE (сумма:
# обучения волн 33 и 41; последовательный выбор: волна 48, обучение 2) — что даёт обрезка двум сетям.
if PREFIX == "pb2d":
    for _nm, _extra in (("cnxtwo", ""), ("cnxauto", " --two-nets-mode auto")):
        _mt = f"w60{_nm}"
        for _m in (1, 2):
            _scr, _a = w51_train("two", _m, _mt)
            add(f"w60-{_nm}-m{_m}", 60, "H06",
                [(_scr, _a + _extra), sg_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", S5)],
                5.1, needs=[W41_TEACH["two"]],
                note=f"pb2d, две сети ({'сумма' if not _extra else 'последовательный выбор'}), ConvNeXt без MAE, "
                     f"сид обучения {_m}, обрезка застрявшего действия")
    for _nm, _ck in (("two33", "rl_arch/models/convnext_dqn_cfix_rdelta_vb_ete_two_seed1.pt"),
                     ("two41m2", w41_hf_model("w41two", 2)), ("two41m3", w41_hf_model("w41two", 3)),
                     ("auto48m2", w41_hf_model("w48auto", 2))):
        add(f"w60-sg-{_nm}", 60, "H06", [sg_eval(_ck, f"t3m_w60{_nm}", S5)], 4.1, needs=[_ck],
            note=f"pb2d, две сети с MAE ({_nm}): обрезка застрявшего действия, сиды 42-46")

# ---------------------------------------------------------------- волна 61
# Энкодер на минимуме GPU-часов данных, агент всегда с двумя сетями (требование 18 октября; сумма --two-nets,
# учитель W41_TEACH["two"]) на pb2d и ns2d. Каждый арм — два обучения (сиды обучения 1 и 2) на собранном буфере
# лидера (W33_SRC), оценка на сидах 42-46 со стражем без сбросов (free_eval, тег _g); на ns2d сиды 45-46 — отдельным
# кернелом по модели из HF. Новые данные энкодера:
#   procreal — MAE на смеси: 5000 процедурных цепочек (как w55proc) и свои 130 переходов случайного буфера цели (как
#              w46 n130), половина каждой пачки — свои карты (cv_probe.py --mix-real --mix-real-frac 0.5);
#   rdir     — карты на 8 случайных плоскостях через веса PINN с нормировкой по фильтрам (Li et al., NeurIPS 2018) у
#              КАЖДОГО состояния буфера цели, застойные тоже (render_views.py --plane random): автокодировщик не
#              обучается, карта — 676 оценок лосса на сетке (pb2d ~6.9 с, ns2d ~12.6 с на карту по t_surface_s
#              оценок t3m_*), основная цена — повтор PINN (pb2d ~0.24 с, ns2d ~1.0 с на эпоху по повторам волны 55),
#              поэтому по одной цепочке на сид (--max-per-seed 1): pb2d 5 цепочек (~3.2 GPU-ч, из них сетка ~0.8; ~55
#              состояний, ~440 карт на плоскостях), ns2d 3 цепочки (--max-files 3; ~5.9 GPU-ч, сетка ~0.9; ~33
#              состояния, ~260 карт); у видов автокодировщика волны 55 карта стоила ~53 с. Масштабы плоскостей 1, 0.3,
#              0.1, 0.03 по кругу (от плоскости Li et al. до окрестности, как у карт среды); карта среды (вид 0) — из
#              файла буфера. MAE на этих видах (cv_probe --views) — в том же кернеле;
#   d4own    — MAE на своих 130 переходах со случайным поворотом/отражением каждой карты (--ssl-d4): плоскость
#              автокодировщика ориентации не имеет, D4 даёт данных в 8 раз больше бесплатно (агент — без D4);
#   maeout   — MAE + голова исхода (лучшая замена MAE волны 49) на своих 130 переходах (исходы в буфере цели есть).
# Готовые энкодеры дешёвых армов прежних волн, которые на pb2d шли только с одной сетью: proc (процедурные карты,
# w55proc), mvmae (виды автокодировщика pb2d, w55mvpb2d), p3n100 (3 уравнения x 100, w54p3n100) — два обучения с
# двумя сетями, только pb2d (на ns2d они уже с двумя сетями). procsg — обрезка застрявшего действия (sg_eval, тег _sg)
# на процедурном энкодере: pb2d — те же модели proc (пара с free_eval на одних обучениях), ns2d — готовые модели
# w55proctwo m1, m2 (без нового обучения). Контроли есть: лидеры, волна 46 n130 (pb2d), волны 55 и 59.
# Предобучения procreal, d4own и maeout одной задачи — один кернел (три шага cv_probe, энкодеры в HF).
W61_SEED_H = 1.4 if NS2D else 0.7
W61_SG_H = 1.5 if NS2D else 0.75
W61_PRE_H = 1.5 if NS2D else 1.2
W61_RD_TAG = f"rd_{W28_SHORT}"
W61_RD = (("--max-per-seed 1 --max-files 3 --hours 5.0", 7.6) if NS2D else ("--max-per-seed 1 --hours 4.0", 4.5))
if W28_K and (NS2D or PREFIX == "pb2d"):
    _arch = "two"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd, h=W61_SEED_H: (0.4 if NS2D else 0.3) + h * len(sd.split(","))   # noqa: E731
    _encs = {nm: f"rl_arch/encoders/cvprobe_w61{nm}{W28_SHORT}_enc_{kind}_resnet.pt"
             for nm, kind in (("procreal", "mae"), ("rdir", "mae"), ("d4own", "mae"), ("maeout", "maeout"))}
    _pre = f"--pretrain-only --ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}"
    add("w61-pre", 61, "H06",
        [(CVPROBE, f"--tag w61procreal{W28_SHORT} --procedural 5000 --mix-real --mix-real-frac 0.5 --subdirs {SUBDIR} "
                   f"--max-per-pde 130 --configs ssl-mae-resnet {_pre}"),
         (CVPROBE, f"--tag w61d4own{W28_SHORT} --subdirs {SUBDIR} --max-per-pde 130 --configs ssl-mae-resnet --ssl-d4 "
                   f"{_pre}"),
         (CVPROBE, f"--tag w61maeout{W28_SHORT} --subdirs {SUBDIR} --max-per-pde 130 --configs ssl-maeout-resnet {_pre}")],
        W61_PRE_H, note=f"предобучение procreal (5000 процедурных цепочек + 130 своих, половина пачки своя), d4own "
                        f"(MAE с D4 на 130 своих), maeout (MAE + исход на 130 своих), {W46_SSL_STEPS} шагов")
    _ra, _rh = W61_RD
    add("w61-rdir-pre", 61, "H06",
        [(RENDER, f"--pde {PDE} --source {W33_SRC} --plane random --planes 8 --plane-scales 1,0.3,0.1,0.03 {_ra} "
                  f"--worker 0/1 --tag {W61_RD_TAG} --resume --save-dir runs_rl_online"),
         (CVPROBE, f"--tag w61rdir{W28_SHORT} --subdirs {SUBDIR} --views runs_rl_online/views/{W61_RD_TAG} "
                   f"--configs ssl-mae-resnet {_pre}")],
        _rh, note=f"карты на 8 случайных плоскостях у всех состояний буфера цели (по цепочке на сид), затем MAE на них")
    _arms = [("procreal", _encs["procreal"], "MAE на процедурных + своих 130 (половина пачки)"),
             ("rdir", _encs["rdir"], "MAE на картах случайных плоскостей своих состояний"),
             ("d4own", _encs["d4own"], "MAE с D4 на своих 130"),
             ("maeout", _encs["maeout"], "MAE + исход на своих 130")]
    if PREFIX == "pb2d":
        _arms += [("proc", "rl_arch/encoders/cvprobe_w55proc_enc_mae_resnet.pt", "MAE на процедурных картах (w55proc)"),
                  ("mvmae", f"rl_arch/encoders/cvprobe_w55mv{W28_SHORT}_enc_mae_resnet.pt", "MAE на видах автокодировщика"),
                  ("p3n100", "rl_arch/encoders/cvprobe_w54p3n100_enc_mae_resnet.pt", "MAE на 3 уравнениях x 100")]
    for _nm, _enc, _what in _arms:
        _mt = f"w61{_nm}{_arch}"
        for _m in (1, 2):
            add(f"w61-{_nm}-m{_m}", 61, "H06",
                [w41_train(_arch, _m, _mt, init=_enc),
                 free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=[_enc, W41_TEACH[_arch]],
                note=f"две сети, энкодер: {_what}, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w61-{_nm}-m{_m}-b", 61, "H06",
                    [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"две сети, энкодер: {_what}, сид обучения {_m}, сиды {_blocks[1]}")
    # procsg: обрезка застрявшего действия на процедурном энкодере (pb2d — модели арма proc, ns2d — w55proctwo)
    _src = "w61proctwo" if PREFIX == "pb2d" else "w55proctwo"
    for _m in (1, 2):
        _ck = w41_hf_model(_src, _m)
        for _i, _sd in enumerate(_blocks):
            _sfx = "" if len(_blocks) == 1 else "-" + "ab"[_i]
            add(f"w61-procsg-m{_m}{_sfx}", 61, "H06", [sg_eval(_ck, f"t3m_w61procsg{_arch}_m{_m}", _sd)],
                round(_eh(_sd, W61_SG_H), 2), needs=[_ck],
                note=f"две сети, процедурный энкодер ({_src}, сид обучения {_m}), обрезка застрявшего действия, "
                     f"сиды {_sd}")

# ---------------------------------------------------------------- волна 62
# Один рецепт с двумя сетями на оба уравнения. Обрезка застрявшего действия (STALLG) спасает pb2d (две сети с MAE и
# ConvNeXt без MAE — 0.0085-0.0152 на сидах 42-46), но вредит ns2d: лидер 0.0616 против 0.0444 без неё, сид 44 —
# 0.378. Разбор сида 44: срезаны три L-BFGS подряд с лоссом ровно на месте (отношение за окно 1.0), остаток отдан
# запасному L-BFGS 1 x 500, который тоже стоит; без обрезки на этом месте L-BFGS 1 x 500 через 250 эпох застоя сам
# прорвался (лосс 0.023 -> 0.0008, l2re 0.378 -> 0.048). На ns2d 58 из 93 срезов — L-BFGS, на pb2d 65 из 259 (там
# режутся в основном Adam и PSO). Мягкая обрезка (STALLS, тег _ss): проверяются только действия Adam и PSO
# (--stall-opts Adam,PSO), окно и порог прежние. Оценки — по готовым моделям, без новых обучений:
#   ns2d: лидер (две сети, MAE) с мягкой обрезкой (пара к t3m_ete_two_g и t3m_w51lead_sg); ConvNeXt без MAE с двумя
#         сетями (обучения волны 51, m1 и m2) без обрезки (тег _g, пара к t3m_w51cnxtwo_m*_sg: 0.064 / 0.081) и с
#         мягкой обрезкой; сиды 42-44 и 45-46 отдельными кернелами;
#   pb2d: ConvNeXt без MAE с двумя сетями (волна 60, m1 и m2) и две сети с MAE (волна 41, m2: срыв 0.0876 без
#         обрезки, 0.0095 с полной) с мягкой обрезкой, сиды 42-46 — держит ли мягкая обрезка выигрыш pb2d.
# Решение: если мягкая обрезка на ns2d не хуже лидера без обрезки, а на pb2d не хуже полной — один рецепт на оба.
STALLS = f"{STALLG} --stall-opts Adam,PSO"


def ss_eval(model, tag, sd):
    """Оценка со стражем с откатом, без сбросов (safe) и с мягкой обрезкой (только Adam и PSO); тег _ss."""
    return (EVAL, f"--policy agent --model-file {model} {GUARD} {STALLS} {KEEPS} "
                  + t3ev(tag + "_ss", sd, hours=MAP_EH))


if W28_K and NS2D:
    _blocks = (seeds(42, 44), seeds(45, 46))
    _eh = lambda sd: 0.4 + 1.5 * len(sd.split(","))   # noqa: E731
    _lead = f"rl_arch/models/convnext_dqn_cfix_rdelta_vb{SUBSFX}_ete_two_seed1.pt"
    _jobs = [("lead-ss", ss_eval, _lead, "t3m_w62lead", "лидер (две сети, MAE), мягкая обрезка")]
    for _m in (1, 2):
        _ck = w41_hf_model("w51cnxtwo", _m)
        _jobs += [(f"cnx-m{_m}", free_eval, _ck, f"t3m_w51cnxtwo_m{_m}",
                   f"ConvNeXt без MAE, две сети, обучение {_m}, без обрезки"),
                  (f"cnxss-m{_m}", ss_eval, _ck, f"t3m_w51cnxtwo_m{_m}",
                   f"ConvNeXt без MAE, две сети, обучение {_m}, мягкая обрезка")]
    for _nm, _fn, _ck, _tg, _what in _jobs:
        for _i, _sd in enumerate(_blocks):
            add(f"w62-{_nm}-{'ab'[_i]}", 62, "H06", [_fn(_ck, _tg, _sd)], round(_eh(_sd), 2), needs=[_ck],
                note=f"ns2d: {_what}, сиды {_sd}")
if PREFIX == "pb2d":
    for _nm, _ck, _tg in (("cnxtwo-m1", w41_hf_model("w60cnxtwo", 1), "t3m_w60cnxtwo_m1"),
                          ("cnxtwo-m2", w41_hf_model("w60cnxtwo", 2), "t3m_w60cnxtwo_m2"),
                          ("two41m2", w41_hf_model("w41two", 2), "t3m_w60two41m2")):
        add(f"w62-{_nm}-ss", 62, "H06", [ss_eval(_ck, _tg, S5)], 4.1, needs=[_ck],
            note=f"pb2d, две сети ({_tg}): мягкая обрезка (только Adam и PSO), сиды 42-46")

# ---------------------------------------------------------------- волна 63
# Три вопроса после волны 62 (агент всегда с двумя сетями, сумма --two-nets, учитель W41_TEACH["two"], буфер W33_SRC).
# (1) VAE вместо MAE на pb2d при минимуме данных: β-VAE (ssl-vae: восстановление всей нормированной карты через
#     латент 32, β = 1) на 5000 процедурных цепочках (vaeproc, 0 GPU-ч данных, как w55proc) и на своих 130 переходах
#     случайного буфера цели (vaeown, как w61 maeout/d4own); предобучения — один кернел, 1400 шагов. По два
#     обучения, оценка на сидах 42-46 без обрезки (free_eval, тег _g; пара к w61) и с мягкой обрезкой (ss_eval, _ss).
# (2) ConvNeXt без MAE: (а) без обрезки на pb2d — готовые w60cnxtwo m1, m2 (с обрезкой 0.0122 / 0.0085);
#     (б) аугментация карт D4 при обучении Q (--aug d4: случайный поворот/отражение карт s и s' — одно на пачку, как
#     RAD, Laskin et al. NeurIPS 2020, и DrQ, Kostrikov et al. ICLR 2021, для RL по картинкам на малых данных).
#     Плоскость карты автокодировщика ориентации не имеет (проверка симметрии средней карты — у d4_apply), поэтому
#     D4 — точная симметрия состояния, а 130 переходов дают 8x наблюдений. ConvNeXt без предобучения проигрывает на
#     ns2d в 1.5-1.9 раза — похоже на нехватку данных для свёрток, D4 бьёт ровно туда. По два обучения на pb2d и
#     ns2d, оценка без обрезки и с мягкой (ns2d — сиды 42-44 и 45-46 отдельными кернелами).
# (3) ns2d, отбор обучения процедурного агента (w55proctwo: m1-m3 на 42-46 — 0.0558 / 0.0437 / 0.0947): ещё два
#     обучения (m4, m5) и у всех пяти — оценка на сиде отбора 52 (вне тестовых 42-51; эталон цели и так нужен
#     обучению — награды буфера считаются по нему). Выбранный по сиду 52 сравнивается с лидером на 42-51; m4 и m5
#     оцениваются и на 42-46 при любом исходе — видно, угадывает ли отбор. Добор 47-51 выбранного (если это m4
#     или m5) — после отбора.
W63_SEED_H = 1.5 if NS2D else 0.75
if W28_K and (NS2D or PREFIX == "pb2d"):
    _arch = "two"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W63_SEED_H * len(sd.split(","))   # noqa: E731
    _arms = []
    if PREFIX == "pb2d":
        _pre = f"--pretrain-only --ssl-steps {W46_SSL_STEPS} --save-encoder --out-dir {W33_CV}"
        add("w63-pre-vae", 63, "H06",
            [(CVPROBE, f"--tag w63vaeproc --procedural 5000 --configs ssl-vae-resnet {_pre}"),
             (CVPROBE, f"--tag w63vaeown{W28_SHORT} --subdirs {SUBDIR} --max-per-pde 130 --configs ssl-vae-resnet "
                       f"{_pre}")],
            1.2, note=f"предобучение β-VAE: 5000 процедурных цепочек (vaeproc) и свои 130 (vaeown), {W46_SSL_STEPS} шагов")
        for _nm, _enc, _what in (("vaeproc", "rl_arch/encoders/cvprobe_w63vaeproc_enc_vae_resnet.pt",
                                  "β-VAE на процедурных картах"),
                                 ("vaeown", f"rl_arch/encoders/cvprobe_w63vaeown{W28_SHORT}_enc_vae_resnet.pt",
                                  "β-VAE на своих 130")):
            _arms.append((_nm, w41_train, dict(init=_enc), [_enc], _what))
        for _m in (1, 2):
            _ck = w41_hf_model("w60cnxtwo", _m)
            add(f"w63-cnxfree-m{_m}", 63, "H06", [free_eval(_ck, f"t3m_w60cnxtwo_m{_m}", S5)], 4.1, needs=[_ck],
                note=f"pb2d, ConvNeXt без MAE (две сети, обучение {_m} волны 60) без обрезки, сиды 42-46")
    _arms.append(("cnxd4", w51_train, dict(extra=" --aug d4"), [], "ConvNeXt без MAE, аугментация D4"))
    for _nm, _fn, _kw, _needs, _what in _arms:
        _mt = f"w63{_nm}{_arch}"
        for _m in (1, 2):
            if _fn is w41_train:
                _tr = w41_train(_arch, _m, _mt, init=_kw["init"])
            else:
                _scr, _a = w51_train(_arch, _m, _mt)
                _tr = (_scr, _a + _kw["extra"])
            add(f"w63-{_nm}-m{_m}", 63, "H06",
                [_tr, free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=list(_needs) + [W41_TEACH[_arch]],
                note=f"две сети, {_what}, сид обучения {_m}, без обрезки, сиды {_blocks[0]}")
            _ck = w41_hf_model(_mt, _m)
            for _i, _sd in enumerate(_blocks):
                _sfx = "" if len(_blocks) == 1 else "-" + "ab"[_i]
                if _i:
                    add(f"w63-{_nm}-m{_m}{_sfx}", 63, "H06", [free_eval(_ck, f"t3m_{_mt}_m{_m}", _sd)],
                        round(_eh(_sd), 2), needs=[_ck], note=f"две сети, {_what}, сид обучения {_m}, без обрезки, сиды {_sd}")
                add(f"w63-{_nm}-m{_m}-ss{_sfx}", 63, "H06", [ss_eval(_ck, f"t3m_{_mt}_m{_m}", _sd)],
                    round(_eh(_sd), 2), needs=[_ck],
                    note=f"две сети, {_what}, сид обучения {_m}, мягкая обрезка, сиды {_sd}")
if NS2D:
    _arch = W41_LEAD
    _mt = f"w55proc{_arch}"
    _enc = "rl_arch/encoders/cvprobe_w55proc_enc_mae_resnet.pt"
    for _m in (4, 5):
        add(f"w63-proc-m{_m}", 63, "H06",
            [w41_train(_arch, _m, _mt, init=_enc),
             free_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", "52,42,43,44")],
            7.4, needs=[_enc, W41_TEACH[_arch]],
            note=f"процедурные карты, обучение {_m}: сид отбора 52, затем 42-44")
        add(f"w63-proc-m{_m}-b", 63, "H06", [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", "45,46")], 3.4,
            needs=[w41_hf_model(_mt, _m)], note=f"процедурные карты, обучение {_m}, сиды 45-46")
    for _m in (1, 2, 3):
        add(f"w63-proc-sel-m{_m}", 63, "H06", [free_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", "52")], 1.9,
            needs=[w41_hf_model(_mt, _m)], note=f"процедурные карты, обучение {_m}: сид отбора 52")

# ---------------------------------------------------------------- волна 64
# Метод с нуля: без данных других уравнений. Кроме MAE (его заменяет энкодер без чужих данных: pb2d — ConvNeXt без
# предобучения, ns2d — MAE на процедурных картах w55proc), буферы 21 другого уравнения нужны ещё в трёх местах:
# (1) направленный сбор 130 переходов — суррогат выбора цепочек учится на буферах остальных задач; (2) плотные
# модельные цели (--mb-aux) — тот же суррогат; (3) учитель MBPO на скалярах — агент в этом суррогате. Три арма,
# агент с двумя сетями (сумма), по два обучения (сиды 1, 2):
#   own     — собранный буфер лидера (W33_SRC), суррогат целей и учитель — только на нём (--mb-subdirs и --subdirs
#             с одной папкой цели: строк других задач в таблице ноль; исходный буфер цели тоже не берётся);
#   none    — тот же буфер, без модельных целей и без учителя: чистый DQN по 130 переходам;
#   scratch — новый направленный сбор, суррогат которого учится только на собранном здесь (collect_active.py
#             --subdirs <папка цели>: первая пачка случайная — холодный старт, дальше LCB на своём), те же
#             сборщики, сиды и 8 часов, что у W33_SRC; суррогат целей и учитель — только на нём. Это метод целиком
#             без чужих данных.
# Оценка: pb2d — с мягкой обрезкой (ss_eval; контроль — w60cnxtwo m1/m2 _ss 0.0122 / 0.0085), ns2d — без обрезки
# (free_eval, сиды 42-44 и 45-46 отдельно; контроль — w55proctwo m1/m2 _g 0.0558 / 0.0437 на 42-46).
W64_SEED_H = 1.5 if NS2D else 0.75
W64_SCR = [f"rl_arch/buffers_new/tbuf_{W28_SHORT}_own_w{_k}" for _k in range(max(1, W28_K // 4))] if W28_K else []
W64_TEACH = {nm: f"rl_arch/agents_online/{TEACH}_w64{nm}.pt" for nm in ("own", "scratch")}
W64_ENC_NS = "rl_arch/encoders/cvprobe_w55proc_enc_mae_resnet.pt"
if W28_K and (NS2D or PREFIX == "pb2d"):
    _arch = "two"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + W64_SEED_H * len(sd.split(","))   # noqa: E731
    _ev = free_eval if NS2D else ss_eval
    _scr_src = ",".join(W64_SCR)
    for _k, _src in enumerate(W64_SCR):
        add(f"w64-col-{_k}", 64, "H06",
            [(COLLECT, f"--pde {PDE} --budget {BUDGET} --rounds 99 --batch {_cbatch} --share-prefix --prefix-bonus 0.1 "
                       f"--truncate 3 {KEEPS} --acq lcb --lcb-k 1 --subdirs {SUBDIR} --save-buffer "
                       f"--worker {_k}/{W28_K} --hours 8.0 --kernel-hours 8.0 --resume --save-dir runs_rl_online "
                       f"--tag tbuf_{W28_SHORT}_own_w{_k}")],
            9.0, note=f"направленный сбор без чужих буферов (суррогат только на собранном), сборщик {_k} из {W28_K}")
    add("w64-teach-own", 64, "H06",
        [(COMBO, f"{TEACH_TR} --target-buffer {W33_SRC} --subdirs {SUBDIR} --tag {TEACH}_w64own")],
        1.0, note="учитель MBPO только на собранном буфере лидера (без других задач)")
    add("w64-teach-scratch", 64, "H06",
        [(COMBO, f"{TEACH_TR} --target-buffer {_scr_src} --subdirs {SUBDIR} --tag {TEACH}_w64scratch")],
        1.0, needs=[f"rl_arch/online_env/{PDE}_tbuf_{W28_SHORT}_own_w{_k}_collect.json" for _k in range(len(W64_SCR))],
        note="учитель MBPO только на новом сборе без чужих буферов (в план — после конца сбора: файл сбора "
             "пишется и по ходу)")
    _enc_what = "MAE на процедурных картах" if NS2D else "ConvNeXt без MAE"
    for _nm, _what in (("own", "суррогат и учитель только на своём буфере"),
                       ("none", "без модельных целей и учителя"),
                       ("scratch", "свой сбор без чужих буферов, суррогат и учитель только на нём")):
        _mt = f"w64{_nm}{_arch}"
        for _m in (1, 2):
            _scr, _a = (w41_train(_arch, _m, _mt, init=W64_ENC_NS) if NS2D else w51_train(_arch, _m, _mt))
            _tf = f"--teacher-from {W41_TEACH[_arch]} --teacher-w 0.5"
            assert _a.count(_tf) == 1 and _a.count(MB) == 1 and _a.count(f"--target-buffer {W33_SRC} ") == 1, _a
            if _nm == "none":
                _a = _a.replace(_tf, "").replace(MB, "")
                _needs = []
            else:
                _a = _a.replace(_tf, f"--teacher-from {W64_TEACH[_nm]} --teacher-w 0.5").replace(
                    MB, f"{MB} --mb-subdirs {SUBDIR}")
                _needs = [W64_TEACH[_nm]]
            if _nm == "scratch":
                _a = _a.replace(f"--target-buffer {W33_SRC} ", f"--target-buffer {_scr_src} ")
            _a = " ".join(_a.split())
            if NS2D:
                _needs = _needs + [W64_ENC_NS]
            add(f"w64-{_nm}-m{_m}", 64, "H06",
                [(_scr, _a), _ev(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
                round(1.0 + _eh(_blocks[0]), 2), needs=_needs,
                note=f"с нуля ({_enc_what}, две сети): {_what}, сид обучения {_m}, сиды {_blocks[0]}")
            if len(_blocks) > 1:
                add(f"w64-{_nm}-m{_m}-b", 64, "H06", [_ev(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                    round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                    note=f"с нуля ({_enc_what}, две сети): {_what}, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 65
# Сколько журналов других уравнений нужно модели среды (суррогат модельных целей и учитель). Волна 64 (обрывки):
# без модели среды и учителя pb2d 0.132, с ними только на своих 130 переходах ns2d 0.32 — журналы источников
# нужны; полные журналы 20 уравнений — тысячи GPU-часов. Журналы — скаляры без карт.
#   (1) surr_data_study.py — без агентов: модель среды для pb2d и ns2d (их собранные буферы лидера) на журналах
#       k уравнений x n цепочек; порядок уравнений жадный (по качеству прогноза целей) и по разнообразию
#       (без знания целей); кривая k = 1..6, n = 5..100, опорные точки own и all. Один кернел на обе цели.
#   (2) агент сразу в точке src3n20: 3 дешёвых уравнения разных типов — burgers1d (нелинейное 1D, ~2.7 GPU-мин
#       на переход с картой), poisson2d_classic (эллиптическое, ~2.9), ns2d_backstep (течение, ~5.6) — по 20
#       цепочек (~500 переходов): учитель (agent_combo --subdirs ... --src-chains 20) и суррогат целей ученика
#       (--mb-subdirs ... --mb-src-chains 20) видят одни и те же цепочки. Энкодер без чужих данных (pb2d ConvNeXt
#       без MAE, ns2d MAE на процедурных картах), две сети, два обучения; оценка pb2d с мягкой обрезкой, ns2d
#       без обрезки. Контроль — те же агенты с полными журналами: pb2d w60cnxtwo m1/m2 _ss 0.0122 / 0.0085, ns2d
#       w55proctwo m1/m2 _g 0.0558 / 0.0437 (сиды 42-46). Агент статьи в нашем протоколе: pb2d 0.0266, ns2d 0.0674.
W65_SRC = "burgers1d,poisson2d_classic,ns2d_backstep"
W65_N = 20
W65_TEACH = f"rl_arch/agents_online/{TEACH}_w65src3n20.pt"
W65_TB = {"poissonboltzmann2d": "rl_arch/buffers_new/tbuf_pb2d_w0",
          "ns2d_liddriven": "rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1"}
SURRSTUDY = "experiments/rl_arch/surr_data_study.py"
if PREFIX == "pb2d":
    add("w65-study", 65, "H06",
        [(SURRSTUDY, "--tag w65 " + " ".join(f"--target {k}={v}" for k, v in W65_TB.items())
                     + " --pool all --k-max 6 --sel-chains 20 --sizes 5,10,20,50,100 --seeds 0,1 --folds 5 "
                       "--hours 6.0 --resume --out-dir /kaggle/working")],
        7.0, note="модель среды для pb2d и ns2d на журналах k уравнений x n цепочек: жадный и разнообразный порядок")
if W28_K and (NS2D or PREFIX == "pb2d"):
    assert W33_SRC == W65_TB[PDE], (W33_SRC, W65_TB[PDE])
    _arch = "two"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + (1.5 if NS2D else 0.75) * len(sd.split(","))   # noqa: E731
    _ev = free_eval if NS2D else ss_eval
    _src = f"--src-chains {W65_N} --src-seed 0"
    add("w65-teach", 65, "H06",
        [(COMBO, f"{TEACH_TR} --target-buffer {W33_SRC} --subdirs {W65_SRC} {_src} --tag {TEACH}_w65src3n20")],
        1.0, note=f"учитель MBPO: журналы {W65_SRC} по {W65_N} цепочек и собранный буфер цели")
    _mt = f"w65src3n20{_arch}"
    for _m in (1, 2):
        _scr, _a = (w41_train(_arch, _m, _mt, init=W64_ENC_NS) if NS2D else w51_train(_arch, _m, _mt))
        _tf = f"--teacher-from {W41_TEACH[_arch]} --teacher-w 0.5"
        assert _a.count(_tf) == 1 and _a.count(MB) == 1, _a
        _a = _a.replace(_tf, f"--teacher-from {W65_TEACH} --teacher-w 0.5").replace(
            MB, f"{MB} --mb-subdirs {W65_SRC} --mb-src-chains {W65_N} --mb-src-seed 0")
        _needs = [W65_TEACH] + ([W64_ENC_NS] if NS2D else [])
        add(f"w65-src3n20-m{_m}", 65, "H06",
            [(_scr, _a), _ev(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
            round(1.0 + _eh(_blocks[0]), 2), needs=_needs,
            note=f"журналы 3 уравнений x {W65_N} цепочек, энкодер без чужих данных, сид обучения {_m}, сиды {_blocks[0]}")
        if len(_blocks) > 1:
            add(f"w65-src3n20-m{_m}-b", 65, "H06", [_ev(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                note=f"журналы 3 уравнений x {W65_N} цепочек, сид обучения {_m}, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 66
# Подтверждение метода без больших данных (волна 65: журналы 3 уравнений x 20 цепочек, энкодер без чужих данных)
# на сидах 47-51, чтобы сравнение с агентом статьи шло по тем же 10 сидам: pb2d — оба обучения (мягкая обрезка),
# ns2d — оба обучения без обрезки (сиды 47-49 и 50-51 отдельными кернелами).
if W28_K and (NS2D or PREFIX == "pb2d"):
    _mt = "w65src3n20two"
    _blocks = ((seeds(47, 49), seeds(50, 51)) if NS2D else (seeds(47, 51),))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + (1.5 if NS2D else 0.75) * len(sd.split(","))   # noqa: E731
    _ev = free_eval if NS2D else ss_eval
    for _m in (1, 2):
        _ck = w41_hf_model(_mt, _m)
        for _i, _sd in enumerate(_blocks):
            _sfx = "" if len(_blocks) == 1 else "-" + "ab"[_i]
            add(f"w66-src3n20-m{_m}{_sfx}", 66, "H06", [_ev(_ck, f"t3m_{_mt}_m{_m}", _sd)], round(_eh(_sd), 2),
                needs=[_ck], note=f"журналы 3 уравнений x 20 цепочек, обучение {_m}: добор сидов {_sd}")

# ---------------------------------------------------------------- волна 67
# Один метод на все уравнения: ResNet + MAE на процедурных картах (W64_ENC_NS, данных ноль), две сети, модель среды
# и учитель на журналах 3 уравнений x 20 цепочек (W65), развёртывание с откатом и мягкой обрезкой (ss_eval) на обоих.
#   pb2d — новые обучения с процедурным энкодером (сиды обучения 1, 2; учитель W65_TEACH уже в HF);
#   ns2d — готовые модели волны 65 (тот же рецепт) с мягкой обрезкой (сиды 42-44 и 45-46 отдельно).
# Волна 66 (ConvNeXt на pb2d, ns2d без обрезки) заменена этой: в ней не единый метод.
if W28_K and (NS2D or PREFIX == "pb2d"):
    _arch, _mt = "two", "w67proc3n20two"
    if PREFIX == "pb2d":
        for _m in (1, 2):
            _scr, _a = w41_train(_arch, _m, _mt, init=W64_ENC_NS)
            _tf = f"--teacher-from {W41_TEACH[_arch]} --teacher-w 0.5"
            assert _a.count(_tf) == 1 and _a.count(MB) == 1, _a
            _a = _a.replace(_tf, f"--teacher-from {W65_TEACH} --teacher-w 0.5").replace(
                MB, f"{MB} --mb-subdirs {W65_SRC} --mb-src-chains {W65_N} --mb-src-seed 0")
            add(f"w67-proc3n20-m{_m}", 67, "H06",
                [(_scr, _a), ss_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", S5)],
                5.05, needs=[W65_TEACH, W64_ENC_NS],
                note=f"единый метод на pb2d: процедурный MAE, журналы 3 x 20, мягкая обрезка, сид обучения {_m}")
    else:
        for _m in (1, 2):
            _ck = w41_hf_model("w65src3n20two", _m)
            for _i, _sd in enumerate((seeds(42, 44), seeds(45, 46))):
                add(f"w67-ss-m{_m}-{'ab'[_i]}", 67, "H06", [ss_eval(_ck, f"t3m_w65src3n20two_m{_m}", _sd)],
                    round(0.4 + 1.5 * len(_sd.split(",")), 2), needs=[_ck],
                    note=f"единый метод на ns2d (модель волны 65, обучение {_m}) с мягкой обрезкой, сиды {_sd}")
            # те же сиды 42-44 по одному: остатки квоты по аккаунтам меньше трёх сидов (тег тот же — строки общие)
            for _x in (42, 43, 44):
                add(f"w67-ss-m{_m}-s{_x}", 67, "H06", [ss_eval(_ck, f"t3m_w65src3n20two_m{_m}", str(_x))], 1.9,
                    needs=[_ck], note=f"единый метод на ns2d, обучение {_m}, мягкая обрезка, сид {_x} отдельно")

# ---------------------------------------------------------------- волна 68
# Итоговый метод (волна 67: процедурный MAE, две сети, журналы 3 x 20, откат и мягкая обрезка) + аугментация D4
# в духе DrQ (--aug drq --aug-k 2: своё преобразование у каждого перехода, s и s' независимо, цель TD — среднее по
# двум преобразованиям s'). D4 — точная симметрия карты (плоскость автокодировщика ориентации не имеет). По два
# обучения на pb2d и ns2d, оценка с мягкой обрезкой на сидах 42-46 (ns2d — блоками 42-44 и 45-46). Контроль — тот же
# метод без аугментации: pb2d w67proc3n20two m1/m2 _ss, ns2d w65src3n20two m1/m2 _ss (те же сиды обучения).
def w68_train(m, mt):
    """Итоговый метод (процедурный MAE, две сети, журналы 3 x 20) — строка обучения."""
    scr, a = w41_train("two", m, mt, init=W64_ENC_NS)
    tf = f"--teacher-from {W41_TEACH['two']} --teacher-w 0.5"
    assert a.count(tf) == 1 and a.count(MB) == 1, a
    return scr, a.replace(tf, f"--teacher-from {W65_TEACH} --teacher-w 0.5").replace(
        MB, f"{MB} --mb-subdirs {W65_SRC} --mb-src-chains {W65_N} --mb-src-seed 0")


if W28_K and (NS2D or PREFIX == "pb2d"):
    _mt = "w68drqtwo"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + (1.5 if NS2D else 0.75) * len(sd.split(","))   # noqa: E731
    for _m in (1, 2):
        _scr, _a = w68_train(_m, _mt)
        add(f"w68-drq-m{_m}", 68, "H06",
            [(_scr, _a + " --aug drq --aug-k 2"), ss_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
            round(1.0 + _eh(_blocks[0]), 2), needs=[W65_TEACH, W64_ENC_NS],
            note=f"итоговый метод + D4 (DrQ, K=2), сид обучения {_m}, мягкая обрезка, сиды {_blocks[0]}")
        if len(_blocks) > 1:
            add(f"w68-drq-m{_m}-b", 68, "H06", [ss_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                note=f"итоговый метод + D4 (DrQ, K=2), сид обучения {_m}, мягкая обрезка, сиды {_blocks[1]}")

# ---------------------------------------------------------------- волна 69
# Итоговый метод + прежняя аугментация D4 (--aug d4: одно преобразование на пачку, общее для s и s'). Волна 68
# (--aug drq) помогла pb2d (0.0106 / 0.0118 против 0.0120 / 0.0272), но ухудшила ns2d в 1.5-1.7 раза (сиды 42-44:
# 0.0868 / 0.0677 против 0.0503 / 0.0460); у ConvNeXt без MAE на ns2d --aug d4 помогала (0.079 -> 0.059 в одном
# обучении). Разница с drq — общее преобразование s и s' и цель без усреднения. Те же два обучения, оценка с мягкой
# обрезкой на сидах 42-46 (ns2d — блоками 42-44 и 45-46); контроль тот же, что у волны 68.
if W28_K and (NS2D or PREFIX == "pb2d"):
    _mt = "w69d4two"
    _blocks = ((seeds(42, 44), seeds(45, 46)) if NS2D else (S5,))
    _eh = lambda sd: (0.4 if NS2D else 0.3) + (1.5 if NS2D else 0.75) * len(sd.split(","))   # noqa: E731
    for _m in (1, 2):
        _scr, _a = w68_train(_m, _mt)
        add(f"w69-d4-m{_m}", 69, "H06",
            [(_scr, _a + " --aug d4"), ss_eval(off_model("convnext_dqn", _mt, _m), f"t3m_{_mt}_m{_m}", _blocks[0])],
            round(1.0 + _eh(_blocks[0]), 2), needs=[W65_TEACH, W64_ENC_NS],
            note=f"итоговый метод + D4 на пачку, сид обучения {_m}, мягкая обрезка, сиды {_blocks[0]}")
        if len(_blocks) > 1:
            add(f"w69-d4-m{_m}-b", 69, "H06", [ss_eval(w41_hf_model(_mt, _m), f"t3m_{_mt}_m{_m}", _blocks[1])],
                round(_eh(_blocks[1]), 2), needs=[w41_hf_model(_mt, _m)],
                note=f"итоговый метод + D4 на пачку, сид обучения {_m}, мягкая обрезка, сиды {_blocks[1]}")

MODAL = "Adam:0.001:2500,LBFGS:0.5:1000,LBFGS:1:500"
add("w23-modal", 23, "H06",
    [(EVAL, f"--policy script --script {MODAL} --script-tail repeat --no-state {KEEPS} "
            + t3ev("t3m_modal", S5, hours=MAP_EH))],
    1.0 + MAP_EH, note="контроль: самая частая цепочка лидера волны 22 на pb2d как статическая, без агента")

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
