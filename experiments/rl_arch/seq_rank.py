#!/usr/bin/env python
"""Последовательное сравнение агентов по сидам среды (без GPU, секунды): сколько сидов на самом деле
нужно, чтобы решение «A лучше B» или «вот лучшие k» было надёжным, вместо фиксированных 10.

    python experiments/rl_arch/seq_rank.py --rows <каталог строк> --pde ns2d --tags t3m_ete_two_g,t3m_w41two_m2_g
    python experiments/rl_arch/seq_rank.py --json res.json                  # {тег: {сид: l2re}}
    python experiments/rl_arch/seq_rank.py --replay --rows <каталог> [--fetch]   # волны 33, 36, 40, 41, 43
    python experiments/rl_arch/seq_rank.py --bound-stats --rows <каталог>  # откуда граница --bound

Строки — JSON online_eval_env.py (rl_arch/online_env/<pde>_<тег>_seed<NN>.json), как у map_results.py;
строки без stop_reason (промежуточная выгрузка, partial) пропускаются. --fetch скачивает строки
нужных тегов из датасета результатов HF в --rows (лёгкие JSON; токен чтения — hf_token_read из
experiments/chain_eval/accounts.json, не печатается).

1) Последовательный парный тест (как AnyRank, arXiv 2610.14603: попарные гипотезы с p-процессами,
   верными в любой момент остановки, и графическая поправка на множественность). Статистика пары
   (i, j) на общем сиде s: D = log10 l2re_i(s) - log10 l2re_j(s), обрезанная до [-B, B] (--bound).
   Гипотеза «i лучше j» отвергает нуль E[clip D] >= 0. Свидетельство — e-процесс ставок (betting,
   Waudby-Smith & Ramdas 2023): K_t = prod (1 + lam_t * Y_t), Y = -clip(D)/B в [-1, 1], lam_t
   предсказуемая (по прошлым сидам пары: lam = mean(Y) / mean(Y^2), обрезка [0, 0.8], на первом
   сиде 0.4). Под нулём K — неотрицательная супермартингала, поэтому p_t = 1 / max K верно при
   любой (в том числе зависящей от данных) остановке. В AnyRank парный тест — N-SCORE для
   бинарного успеха и без пар; у нас метрика непрерывная и сиды среды общие у всех агентов, поэтому
   берётся парная разность на общих сидах, а ограниченность даёт обрезка.
   Множественность: все упорядоченные пары (--family all) или каждый против контроля (--family
   control --control TAG) — графическая процедура с равными весами и равным перераспределением
   (это Холм на p-процессах): гипотеза отвергается при p <= alpha / (число ещё не отвергнутых).
   FWER всех направленных решений <= alpha (--alpha). Транзитивное замыкание (i<j, j<k => i<k,
   следствие 1 AnyRank) добавляет решения без расхода alpha (--no-closure — выключить).
   Остановка: агент перестаёт получать сиды, когда его решения больше не нужны цели (--goal):
     rank    — все пары с ним решены (или безнадёжны: даже 10 сидов в его пользу не дадут 1/alpha);
     topk    — он достоверно в лучших k (не хуже его доказано у m-k агентов) или вне их (лучше его
               доказано у k агентов); --k;
     control — решено его сравнение с контролем.
   Что значит B. Обрезка делает статистику ограниченной (иначе теста ставок нет) и робастной: сид,
   где один агент в 10 раз хуже другого (расходимость, неудачная инициализация), весит как сид,
   где он хуже в 10^B раз. По строкам волн 33-43 (--bound-stats): разброс log10 ошибки одного агента
   по сидам — медиана 0.10 (ns2d) и 0.21 (pb2d) порядка; |D| пар на общих сидах — 95-й процентиль
   0.48 на ns2d и 1.4 на pb2d, где |D| > 0.5 у 37% пар-сидов: это пары рабочего варианта со сломанным
   (медиана 0.1-0.3 против 0.012) и сорванные сиды (log10 ошибки до +0.4). B = 0.5 порядка (разница в
   3.2 раза) срезает 5% пар-сидов ns2d (только хвост), а на pb2d превращает «сломан против рабочего» в
   полный голос сида, не давая одному сорванному сиду перевесить остальные; при этом B в 2.5-5 раз
   выше разброса сидов, то есть обычные различия не искажает. Граница выбрана по распределению
   (--bound-stats), не по исходу воспроизведения; чувствительность — --replay-bounds 0.25,0.5,1.0.
   Предел возможного: свидетельство n сидов, все в одну сторону, не больше (1 + 0.4) * 1.8^(n - 1):
   4.5 за 3 сида, 14.7 за 5, 278 за 10, а решение требует K >= (число гипотез) / alpha. Поэтому при
   10 сидах одно семейство может содержать не больше ~27 направленных гипотез при alpha = 0.1, агент
   с 5 сидами не отделяется ни от кого даже в паре (нужно 20), а все пары волн сразу (380-1056
   гипотез) не разделяются в принципе.

2) Достоверный выбор лучших k (CAS, arXiv 2610.14996: «сертифицировать решение, а не каждую оценку»).
   Оценка агента — среднее log10 l2re по его ФИКСИРОВАННОМУ набору сидов (по умолчанию 42..51; при
   воспроизведении — сиды, по которым он измерен). Неизвестный сид лежит в [lo, hi] (детерминированные
   границы, --cas-lo/--cas-hi в log10; при воспроизведении — минимум и максимум всех строк этого
   уравнения), поэтому у оценки есть точные нижняя и верхняя границы L, U. Набор A — k агентов с
   наименьшей L; решение сертифицировано, когда max U по A < min L вне A (теорема 2 статьи: для
   независимых строк это необходимо и достаточно). Следующий сид — агенту из A с наибольшей U, если
   у него остались сиды, иначе агенту вне A с наименьшей L (правило CAS-PG; предиктора нет — сиды
   идут по порядку). Ответ ТОЧНО совпадает с лучшими k по полному среднему log10 на тех же сидах;
   цена — число сидов до сертификата против полного числа.

--replay: оценки волн 33, 36, 40, 41, 43 (теги t3m_* из очередей queue.json и queue_pb2d.json, их
пишет make_queue.py) на обоих уравнениях проигрываются так, будто сиды приходят по одному в порядке
42..51: каждый ещё нужный агент получает следующий сид, если строка для него есть. Четыре постановки:
(1) по волнам: каждый агент волны против лидера уравнения (LEADER — так и решали: «лучше ли новый
вариант лидера»), FWER alpha на волну, сиды лидера оплачены раньше и в счёт не идут; (2) каждая пара
с лидером отдельно (alpha на одно сравнение) — сколько сидов нужно одному решению; (3) все пары всех
волн одним семейством; (4) CAS, лучшие 1 и 3, границы по размаху строк уравнения и шире на 0.3
порядка. Отчёт: сколько сидов потрачено против имеющихся (и GPU-часов: сид pb2d около 0.85, ns2d около
1.6), сколько пар разделено и на скольких сидах, совпадает ли каждое решение с медианами по всем
имеющимся сидам (тем, по которым решали раньше).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_REPO = "danil-e/pinnacle-optuna-db"
ROW_DIR = "rl_arch/online_env"
SEEDS = tuple(range(42, 52))
ALIAS = {"ns2d": "ns2d_liddriven", "pb2d": "poissonboltzmann2d"}
QUEUES = {"ns2d_liddriven": os.path.join(HERE, "queue", "queue.json"),
          "poissonboltzmann2d": os.path.join(HERE, "queue", "queue_pb2d.json")}
GPU_H = {"ns2d_liddriven": 1.6, "poissonboltzmann2d": 0.85}     # GPU-часов на один сид оценки
REPLAY_WAVES = (33, 36, 40, 41, 43)
# лидер уравнения на 10 сидах — контроль, с которым сравнивались варианты волн
LEADER = {"poissonboltzmann2d": "t3m_ete_tbquart_g", "ns2d_liddriven": "t3m_ete_two_g"}
BOUND = 0.5
LAM_CAP, LAM_FIRST = 0.8, 0.4


# --------------------------------------------------------------------------------------------
# строки
# --------------------------------------------------------------------------------------------

def pde_name(p):
    return ALIAS.get(p, p)


def hf_token():
    """Токен чтения из experiments/chain_eval/accounts.json (без него HF отвечает 429). Не печатается."""
    try:
        with open(os.path.join(ROOT, "experiments", "chain_eval", "accounts.json")) as f:
            return json.load(f).get("hf_token_read") or None
    except Exception:       # noqa: BLE001
        return None


def fetch(patterns, dst, sub=ROW_DIR):
    """Скачать файлы датасета результатов по шаблонам имён (в папке sub) в dst. Возвращает число файлов."""
    import fnmatch
    import shutil
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    root = snapshot_download(OUT_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN") or hf_token(),
                             allow_patterns=[f"{sub}/{p}" for p in patterns])
    src = os.path.join(root, sub)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in (os.listdir(src) if os.path.isdir(src) else []):
        if any(fnmatch.fnmatch(f, p) for p in patterns):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
            n += 1
    return n


def load_rows(d, pde, tags=None):
    """{тег: {сид: l2re}} законченных сидов (строки без stop_reason — в работе, пропускаются)."""
    pde = pde_name(pde)
    out = {}
    rx = re.compile(re.escape(pde) + r"_(.+)_seed(\d+)\.json$")
    for f in sorted(glob.glob(os.path.join(d, f"{pde}_*_seed*.json"))):
        m = rx.match(os.path.basename(f))
        if not m or (tags is not None and m.group(1) not in tags):
            continue
        with open(f) as fh:
            x = json.load(fh)
        if x.get("stop_reason") is None or x.get("l2re") is None or not x["l2re"] > 0:
            continue
        out.setdefault(m.group(1), {})[int(m.group(2))] = float(x["l2re"])
    return out


def wave_tags(pde, waves=REPLAY_WAVES, queue=None):
    """Теги оценок t3m_* заданных волн из очереди этого уравнения (её пишет make_queue.py)."""
    with open(queue or QUEUES[pde_name(pde)]) as f:
        q = json.load(f)
    tags = set()
    for j in (q["jobs"] if isinstance(q, dict) else q):
        if j["wave"] in waves:
            for s in j["steps"]:
                m = re.search(r"--tag (t3m_\S+)", s["args"])
                if m:
                    tags.add(m.group(1))
    return sorted(tags)


def median(v):
    return float(np.median(list(v))) if len(v) else float("nan")


# --------------------------------------------------------------------------------------------
# последовательный парный тест
# --------------------------------------------------------------------------------------------

class Bet:
    """e-процесс ставок для одной односторонней гипотезы «среднее Y > 0» (нуль: E[Y] <= 0), Y в [-1, 1].
    lam_t предсказуема: считается до прихода Y_t по прошлым наблюдениям."""

    def __init__(self, cap=LAM_CAP, first=LAM_FIRST):
        self.cap, self.first = cap, first
        self.n, self.sy, self.sy2 = 0, 0.0, 0.0
        self.logk, self.maxlogk = 0.0, 0.0

    def lam(self):
        if self.n == 0:
            return self.first
        m, v = self.sy / self.n, self.sy2 / self.n
        return float(np.clip(m / v, 0.0, self.cap)) if v > 0 else self.cap

    def update(self, y):
        y = float(np.clip(y, -1.0, 1.0))
        self.logk += math.log1p(self.lam() * y)
        self.maxlogk = max(self.maxlogk, self.logk)
        self.n += 1
        self.sy += y
        self.sy2 += y * y

    @property
    def p(self):
        return min(1.0, math.exp(-self.maxlogk))

    def hopeless(self, left, alpha):
        """Даже left наблюдений Y = 1 при наибольшей ставке не поднимут K до 1/alpha."""
        return self.logk + left * math.log1p(self.cap) < math.log(1.0 / alpha) and self.maxlogk < math.log(1.0 / alpha)


class SeqRank:
    """Состояние последовательного сравнения. add(тег, сид, l2re) — пришёл сид; test() — графическая
    процедура; needs(тег) — нужен ли агенту ещё сид при цели goal."""

    def __init__(self, tags, bound=BOUND, alpha=0.1, family="all", control=None, closure=True,
                 goal="rank", k=1, max_seeds=len(SEEDS), cap=LAM_CAP, first=LAM_FIRST):
        if family == "control" and control not in tags:
            raise ValueError(f"--family control: контроль {control!r} не среди тегов")
        if goal == "control" and family != "control":
            raise ValueError("--goal control требует --family control")
        self.tags, self.bound, self.alpha = list(tags), float(bound), float(alpha)
        self.family, self.control, self.closure, self.goal = family, control, closure, goal
        self.k, self.max_seeds = int(k), int(max_seeds)
        self.x = {t: {} for t in self.tags}                       # log10 l2re по сидам
        self.hyp = {}                                             # (i, j): Bet — «i лучше j»
        for i in self.tags:
            for j in self.tags:
                if i != j and (family == "all" or control in (i, j)):
                    self.hyp[(i, j)] = Bet(cap, first)
        self.n_hyp = len(self.hyp)
        self.rejected = {}                                        # (i, j) -> общих сидов при решении
        self.closed = {}                                          # решения замыканием
        self.order = []                                           # порядок решений

    # -- данные
    def add(self, tag, seed, l2re):
        xi = math.log10(float(l2re))
        self.x[tag][int(seed)] = xi
        for j in self.tags:
            if j == tag or int(seed) not in self.x[j]:
                continue
            d = float(np.clip(xi - self.x[j][int(seed)], -self.bound, self.bound)) / self.bound
            for (a, b), y in (((tag, j), -d), ((j, tag), d)):
                h = self.hyp.get((a, b))
                if h is not None and (a, b) not in self.rejected:
                    h.update(y)

    def common(self, i, j):
        return len(set(self.x[i]) & set(self.x[j]))

    # -- решения
    def level(self):
        """Текущий уровень одной гипотезы: alpha / число не отвергнутых тестом (Холм)."""
        return self.alpha / max(1, self.n_hyp - len(self.rejected))

    def test(self):
        new = []
        while True:
            lv = self.level()
            cand = [(h.p, ij) for ij, h in self.hyp.items() if ij not in self.rejected and h.p <= lv]
            if not cand:
                break
            p, ij = min(cand)
            self.rejected[ij] = self.common(*ij)
            self.order.append(("test", ij, self.rejected[ij]))
            new.append(ij)
        if self.closure and new:
            self._close()
        return new

    def _close(self):
        better = self.better_pairs()
        changed = True
        while changed:
            changed = False
            for (i, j) in list(better):
                for (j2, k) in list(better):
                    if j2 == j and i != k and (i, k) not in better:
                        better.add((i, k))
                        self.closed[(i, k)] = max(self.rejected.get((i, j), self.closed.get((i, j), 0)),
                                                  self.rejected.get((j, k), self.closed.get((j, k), 0)))
                        self.order.append(("closure", (i, k), self.closed[(i, k)]))
                        changed = True

    def better_pairs(self):
        return set(self.rejected) | set(self.closed)

    def decided(self, i, j):
        b = self.better_pairs()
        return (i, j) in b or (j, i) in b

    def pair_open(self, i, j):
        """Пара ещё может быть решена: не решена, обе гипотезы в семействе, и хотя бы одна не безнадёжна."""
        if self.decided(i, j):
            return False
        hs = [self.hyp.get((i, j)), self.hyp.get((j, i))]
        if not any(hs):
            return False
        left = self.max_seeds - min(len(self.x[i]), len(self.x[j]))
        return any(h is not None and not h.hopeless(left, self.alpha) for h in hs)

    def n_better(self, t):
        return sum(1 for (i, j) in self.better_pairs() if i == t)

    def n_worse(self, t):
        return sum(1 for (i, j) in self.better_pairs() if j == t)

    def status(self, t):
        """В лучших k / вне / не решено (цель topk)."""
        m = len(self.tags)
        if self.n_better(t) >= m - self.k:
            return "in"
        if self.n_worse(t) >= self.k:
            return "out"
        return "?"

    def needs(self, t):
        if len(self.x[t]) >= self.max_seeds:
            return False
        if self.goal == "topk":
            if self.status(t) != "?":
                return False
            # нужен, пока хоть одна его пара может решиться (иначе сиды ему ничего не дадут)
            return any(self.pair_open(t, j) for j in self.tags if j != t)
        if self.goal == "control":
            if t == self.control:
                return any(self.needs(j) for j in self.tags if j != t)
            return self.pair_open(t, self.control)
        return any(self.pair_open(t, j) for j in self.tags if j != t)

    def done(self):
        return not any(self.needs(t) for t in self.tags)

    # -- итог
    def ranking(self):
        """Частичный порядок: по (доказано лучше - доказано хуже), при равенстве — по медиане."""
        sc = {t: self.n_better(t) - self.n_worse(t) for t in self.tags}
        return sorted(self.tags, key=lambda t: (-sc[t], median(self.x[t].values())))

    def summary(self):
        return dict(alpha=self.alpha, bound=self.bound, family=self.family, control=self.control, goal=self.goal,
                    k=self.k, n_hyp=self.n_hyp,
                    seeds={t: len(v) for t, v in self.x.items()},
                    better=sorted([list(ij) + [n, "test"] for ij, n in self.rejected.items()]
                                  + [list(ij) + [n, "closure"] for ij, n in self.closed.items()]),
                    ranking=self.ranking(),
                    status=({t: self.status(t) for t in self.tags} if self.goal == "topk" else None))


def run_sequential(res, seeds=SEEDS, **kw):
    """Сиды приходят по одному в порядке seeds: каждый ещё нужный агент получает следующий сид, если
    строка для него есть (res — {тег: {сид: l2re}}). Возвращает SeqRank после последнего сида."""
    sr = SeqRank(sorted(res), max_seeds=len(seeds), **kw)
    for s in seeds:
        active = [t for t in sr.tags if sr.needs(t)]
        if not active:
            break
        for t in active:
            if s in res[t]:
                sr.add(t, s, res[t][s])
        sr.test()
    return sr


# --------------------------------------------------------------------------------------------
# CAS: достоверные лучшие k с детерминированными границами
# --------------------------------------------------------------------------------------------

def cas_bounds(seen, scen, lo, hi):
    """Нижняя и верхняя граница среднего log10 по набору сидов scen при известных seen (сид -> log10)."""
    vals_l = [seen.get(s, lo) for s in scen]
    vals_u = [seen.get(s, hi) for s in scen]
    return float(np.mean(vals_l)), float(np.mean(vals_u))


def cas_select(res, k, lo, hi, scen=None, order=SEEDS, max_queries=None):
    """CAS-PG без предиктора: запросы по одному (агент, следующий его сид по order) до сертификата.
    res — {тег: {сид: l2re}}; scen — {тег: набор сидов оценки} (по умолчанию — его имеющиеся сиды).
    Сид, которого нет в res (ещё не измерен), запросить нельзя: тогда возврат без сертификата с полем
    next = [тег, сид] — что запустить следующим. Возвращает dict(top, queries, full, certified, seen, next)."""
    tags = sorted(res)
    scen = scen or {t: [s for s in order if s in res[t]] for t in tags}
    seen = {t: {} for t in tags}
    idx = {t: i for i, t in enumerate(tags)}
    q = 0
    full = sum(len(scen[t]) for t in tags)
    k = min(int(k), len(tags))
    while True:
        b = {t: cas_bounds(seen[t], scen[t], lo, hi) for t in tags}
        A = sorted(tags, key=lambda t: (b[t][0], idx[t]))[:k]
        rest = [t for t in tags if t not in A]
        i_star = max(A, key=lambda t: (b[t][1], idx[t]))
        if not rest:
            return dict(top=A, queries=q, full=full, certified=True, seen=seen, next=None)
        j_star = min(rest, key=lambda t: (b[t][0], idx[t]))
        if (b[i_star][1], idx[i_star]) < (b[j_star][0], idx[j_star]):
            return dict(top=sorted(A, key=lambda t: (b[t][0], idx[t])), queries=q, full=full, certified=True,
                        seen=seen, next=None)
        r = None
        for t in (i_star, j_star):
            left = [s for s in scen[t] if s not in seen[t]]
            if left:
                r = (t, left[0])
                break
        if r is None or (max_queries is not None and q >= max_queries) or r[1] not in res[r[0]]:
            # сид, который нужен сертификату, ещё не измерен: это и есть следующий запуск
            return dict(top=A, queries=q, full=full, certified=False, seen=seen,
                        next=(list(r) if r is not None and r[1] not in res[r[0]] else None))
        t, s = r
        seen[t][s] = math.log10(res[t][s])
        q += 1


# --------------------------------------------------------------------------------------------
# воспроизведение законченных волн
# --------------------------------------------------------------------------------------------

def full_reference(res):
    """Медианы по всем имеющимся сидам (так решали раньше) и среднее log10 — для сверки."""
    return ({t: median(v.values()) for t, v in res.items()},
            {t: float(np.mean(np.log10(list(v.values())))) for t, v in res.items()})


def max_evidence(n, cap=LAM_CAP, first=LAM_FIRST):
    """Наибольшее K после n сидов (все сиды в одну сторону, |D| >= B): предел того, что дают n сидов."""
    return (1.0 + first) * (1.0 + cap) ** max(0, n - 1) if n else 1.0


def replay_one(res, pde, goal, k=1, paid=(), **kw):
    """Один прогон по сидам 42..51 и сверка решений с полными данными. paid — теги, чьи сиды уже
    оплачены раньше (контроль): в счёт сидов не идут."""
    sr = run_sequential(res, goal=goal, k=k, **kw)
    med, mlog = full_reference(res)
    avail = {t: len([s for s in SEEDS if s in res[t]]) for t in res}
    used = {t: len(sr.x[t]) for t in res}
    cnt = [t for t in res if t not in paid]
    dec = []
    for kind, (i, j), n in sr.order:
        c = sorted(set(res[i]) & set(res[j]))
        paired = float(np.mean([math.log10(res[i][s]) - math.log10(res[j][s]) for s in c])) if c else float("nan")
        dec.append(dict(better=i, worse=j, kind=kind, seeds=n, common_full=len(c),
                        med_ok=bool(med[i] < med[j]), paired_ok=bool(paired < 0)))
    sa, su = sum(avail[t] for t in cnt), sum(used[t] for t in cnt)
    out = dict(pde=pde, goal=goal, k=k, tags=len(res), avail=sa, used=su, saved=sa - su,
               gpu_h_saved=round((sa - su) * GPU_H.get(pde, 1.0), 1),
               decisions=dec, n_test=sum(d["kind"] == "test" for d in dec),
               n_closure=sum(d["kind"] == "closure" for d in dec),
               mismatch_med=[d for d in dec if not d["med_ok"]],
               mismatch_paired=[d for d in dec if not d["paired_ok"]],
               per_tag={t: (used[t], avail[t]) for t in sorted(res)}, ranking=sr.ranking(), n_hyp=sr.n_hyp)
    if goal == "topk":
        st = {t: sr.status(t) for t in res}
        top_med = sorted(res, key=lambda t: med[t])[:k]
        out.update(cert_in=sorted(t for t in res if st[t] == "in"), cert_out=sum(st[t] == "out" for t in res),
                   unresolved=sorted(t for t in res if st[t] == "?"), top_by_median=top_med,
                   in_ok=all(t in top_med for t in res if st[t] == "in"),
                   out_ok=all(t not in top_med for t in res if st[t] == "out"))
    return out


def pair_alone(res, a, b, **kw):
    """Одно сравнение как отдельное семейство (две направленные гипотезы, FWER alpha на пару):
    сколько общих сидов ему понадобилось. Возвращает dict(better, seeds, common) или seeds=None."""
    sr = run_sequential({a: res[a], b: res[b]}, goal="rank", **kw)
    common = len(set(res[a]) & set(res[b]) & set(SEEDS))
    # used — сиды агента a (контроль b оплачен раньше)
    for kind, (i, j), n in sr.order:
        return dict(better=i, worse=j, seeds=n, common=common, used=len(sr.x[a]), avail=len(res[a]))
    return dict(better=None, worse=None, seeds=None, common=common, used=len(sr.x[a]), avail=len(res[a]))


def replay_cas(res, pde, k, margin=0.0):
    """margin — на сколько порядков шире границы, чем размах всех строк уравнения (без запаса — границы
    по самим данным; в живом применении их задают заранее, и они шире)."""
    allx = np.log10([v for r in res.values() for v in r.values()])
    lo, hi = float(allx.min()) - margin, float(allx.max()) + margin
    c = cas_select(res, k, lo, hi)
    med, mlog = full_reference(res)
    top_mean = sorted(res, key=lambda t: (mlog[t], sorted(res).index(t)))[:k]
    top_med = sorted(res, key=lambda t: med[t])[:k]
    return dict(pde=pde, k=k, margin=margin, lo=round(lo, 3), hi=round(hi, 3), queries=c["queries"], full=c["full"],
                saved=c["full"] - c["queries"], gpu_h_saved=round((c["full"] - c["queries"]) * GPU_H.get(pde, 1.0), 1),
                certified=c["certified"], top=c["top"], exact_ok=sorted(c["top"]) == sorted(top_mean),
                top_by_median=top_med, median_ok=sorted(c["top"]) == sorted(top_med),
                per_tag={t: (len(c["seen"][t]), len(res[t])) for t in sorted(res)})


def print_replay(r, log=print, head=""):
    log(f"  {head}[{r['pde']}] цель {r['goal']}" + (f" k={r['k']}" if r["goal"] == "topk" else "")
        + f": агентов {r['tags']}, гипотез {r['n_hyp']}, сидов потрачено {r['used']} из {r['avail']} "
        f"(сэкономлено {r['saved']}, ~{r['gpu_h_saved']} GPU-ч); решений тестом {r['n_test']}, замыканием "
        f"{r['n_closure']}; против медиан всех сидов: {len(r['mismatch_med'])}, против парного среднего log10: "
        f"{len(r['mismatch_paired'])}")
    for d in r["decisions"]:
        if d["kind"] == "test":
            log(f"      {d['better']} < {d['worse']}: на {d['seeds']} общих сидах из {d['common_full']}"
                + ("" if d["med_ok"] else "  [ПРОТИВ медиан]"))
    for d in r["mismatch_med"]:
        if d["kind"] != "test":
            log(f"      замыкание против медиан: {d['better']} < {d['worse']}")
    if r["goal"] == "topk":
        log(f"      в лучших {r['k']} достоверно: {r['cert_in']}; вне — {r['cert_out']}; не решено: "
            f"{len(r['unresolved'])}; лучшие по медиане: {r['top_by_median']}")


def replay(rows_dir, pdes=("poissonboltzmann2d", "ns2d_liddriven"), min_seeds=3, bounds=(BOUND,), alpha=0.1,
           do_fetch=False, log=print, waves=REPLAY_WAVES, cap=LAM_CAP, first=LAM_FIRST):
    """Воспроизведение: (1) по волнам, каждый агент волны против лидера уравнения (так и решали:
    «лучше ли новый вариант лидера»), FWER alpha на волну; (2) каждая пара с лидером отдельно
    (alpha на одно сравнение) — сколько сидов нужно одному решению; (3) все пары всех волн одним
    семейством; (4) CAS: лучшие 1 и 3 по среднему log10 с детерминированными границами."""
    out = dict(params=dict(min_seeds=min_seeds, alpha=alpha, bounds=list(bounds), waves=list(waves),
                           lam_cap=cap, lam_first=first,
                           max_evidence={n: round(max_evidence(n, cap, first), 1) for n in (3, 5, 10)}),
               waves=[], pairs=[], all_pairs=[], cas=[], totals=[])
    log(f"ставка: lam на первом сиде {first}, предел {cap}; предел свидетельства K после n сидов (все в одну "
        f"сторону): 3 — {max_evidence(3, cap, first):.1f}, 5 — {max_evidence(5, cap, first):.1f}, 10 — "
        f"{max_evidence(10, cap, first):.0f}; решение требует K >= (число гипотез) / alpha")
    for pde in pdes:
        tags = wave_tags(pde, waves)
        if do_fetch:
            n = fetch([f"{pde}_{t}_seed*.json" for t in tags], rows_dir)
            log(f"{pde}: скачано строк {n}")
        res = load_rows(rows_dir, pde, set(tags))
        res = {t: {s: v for s, v in r.items() if s in SEEDS} for t, r in res.items()}
        small = sorted(t for t, r in res.items() if len(r) < min_seeds)
        res = {t: r for t, r in res.items() if len(r) >= min_seeds}
        ctrl = LEADER[pde]
        log(f"{pde}: тегов волн {len(tags)}, со строками не меньше {min_seeds} сидов {len(res)} "
            f"(меньше: {small}; без строк: {sorted(set(tags) - set(res) - set(small))}); лидер-контроль {ctrl} "
            f"({len(res[ctrl])} сидов, медиана {median(res[ctrl].values()):.4f})")
        for B in bounds:
            kw = dict(bound=B, alpha=alpha, cap=cap, first=first)
            log(f"  граница B = {B}, alpha = {alpha}")
            tot = dict(pde=pde, bound=B, avail=0, used=0, decisions=0, mismatch=0)
            for w in waves:
                tw = [t for t in wave_tags(pde, (w,)) if t in res and t != ctrl]
                if not tw:
                    continue
                r = replay_one({t: res[t] for t in tw + [ctrl]}, pde, "control", family="control", control=ctrl,
                               paid=(ctrl,), **kw)
                r.update(wave=w, bound=B)
                print_replay(r, log, head=f"волна {w} ")
                out["waves"].append(r)
                tot["avail"] += r["avail"]
                tot["used"] += r["used"]
                tot["decisions"] += r["n_test"] + r["n_closure"]
                tot["mismatch"] += len(r["mismatch_med"])
            tot["saved"] = tot["avail"] - tot["used"]
            tot["gpu_h_saved"] = round(tot["saved"] * GPU_H.get(pde, 1.0), 1)
            log(f"  ИТОГ по волнам [{pde}] B={B}: сидов {tot['used']} из {tot['avail']} (сэкономлено {tot['saved']}, "
                f"~{tot['gpu_h_saved']} GPU-ч), решений {tot['decisions']}, против медиан {tot['mismatch']}")
            out["totals"].append(tot)
            # (2) каждая пара с лидером отдельно
            med = {t: median(v.values()) for t, v in res.items()}
            pr = []
            for t in sorted(res):
                if t == ctrl:
                    continue
                q = pair_alone(res, t, ctrl, **kw)
                q.update(tag=t, med_ok=(None if q["better"] is None
                                        else bool(med[q["better"]] < med[q["worse"]])), bound=B, pde=pde)
                pr.append(q)
            dec = [q for q in pr if q["seeds"] is not None]
            log(f"  пары с лидером по отдельности [{pde}] B={B}: решено {len(dec)} из {len(pr)}; сидов на решение "
                f"{sorted(q['seeds'] for q in dec)} (было общих {sorted(q['common'] for q in dec)}); против медиан "
                f"{sum(q['med_ok'] is False for q in dec)}; сидов агентов потрачено {sum(q['used'] for q in pr)} из "
                f"{sum(q['avail'] for q in pr)}")
            for q in dec:
                log(f"      {q['better']} < {q['worse']}: {q['seeds']} сидов из {q['common']}"
                    + ("" if q["med_ok"] else "  [ПРОТИВ медиан]"))
            out["pairs"] += pr
            # (3) все пары одним семейством
            r = replay_one(res, pde, "rank", **kw)
            r["bound"] = B
            log(f"  все пары одним семейством [{pde}]: гипотез {r['n_hyp']}, нужен K >= {r['n_hyp'] / alpha:.0f} при "
                f"пределе {max_evidence(10, cap, first):.0f} за 10 сидов; решений {r['n_test']}")
            out["all_pairs"].append({k: v for k, v in r.items() if k != "decisions"})
        for k, mg in ((1, 0.0), (3, 0.0), (1, 0.3), (3, 0.3)):
            c = replay_cas(res, pde, k, mg)
            log(f"  [{pde}] CAS k={k}, запас границ {mg}: сидов {c['queries']} из {c['full']} (сэкономлено {c['saved']}, ~{c['gpu_h_saved']} "
                f"GPU-ч), сертификат {c['certified']}, лучшие {c['top']}; совпадает с полным средним log10: "
                f"{c['exact_ok']}, с медианами: {c['median_ok']} (по медиане: {c['top_by_median']}); границы "
                f"log10 [{c['lo']}, {c['hi']}]")
            out["cas"].append(c)
    return out


def bound_stats(rows_dir, pdes=("poissonboltzmann2d", "ns2d_liddriven"), min_seeds=3, log=print):
    """Распределение парных разностей log10 на общих сидах и разброс сидов одного агента."""
    out = {}
    for pde in pdes:
        res = load_rows(rows_dir, pde, set(wave_tags(pde)))
        res = {t: r for t, r in res.items() if len(r) >= min_seeds}
        ts = sorted(res)
        D = []
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                c = sorted(set(res[ts[a]]) & set(res[ts[b]]))
                if len(c) >= min_seeds:
                    D += [abs(math.log10(res[ts[a]][s]) - math.log10(res[ts[b]][s])) for s in c]
        sp = [float(np.std(np.log10(list(r.values())))) for r in res.values() if len(r) >= 5]
        q = np.quantile(D, [0.5, 0.75, 0.9, 0.95, 0.99]) if D else []
        out[pde] = dict(pairs_obs=len(D), absD_q=[round(float(x), 3) for x in q],
                        frac_above_bound=round(float(np.mean(np.array(D) > BOUND)), 3) if D else None,
                        seed_sd_median=round(float(np.median(sp)), 3) if sp else None)
        log(f"{pde}: |D| квантили 50/75/90/95/99% {out[pde]['absD_q']}, доля |D| > {BOUND}: "
            f"{out[pde]['frac_above_bound']}, разброс log10 по сидам одного агента (медиана) {out[pde]['seed_sd_median']}")
    return out


# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="", help="каталог строк online_eval_env (по умолчанию — временный при --fetch)")
    ap.add_argument("--pde", default="ns2d", help="ns2d | pb2d | полное имя")
    ap.add_argument("--tags", default="", help="теги через запятую (по умолчанию — все в каталоге)")
    ap.add_argument("--json", default="", help="вместо строк: JSON {тег: {сид: l2re}}")
    ap.add_argument("--fetch", action="store_true", help="скачать строки тегов из HF в --rows")
    ap.add_argument("--alpha", type=float, default=0.1, help="FWER всех направленных решений")
    ap.add_argument("--bound", type=float, default=BOUND, help="обрезка парной разности log10 (порядков)")
    ap.add_argument("--family", choices=("all", "control"), default="all")
    ap.add_argument("--control", default="", help="контроль для --family control")
    ap.add_argument("--goal", choices=("rank", "topk", "control"), default="rank")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--lam-cap", type=float, default=LAM_CAP, help="предел ставки (меньше 1)")
    ap.add_argument("--lam-first", type=float, default=LAM_FIRST, help="ставка на первом сиде пары")
    ap.add_argument("--cas-lo", type=float, default=None, help="нижняя граница log10 l2re неизвестного сида (CAS)")
    ap.add_argument("--cas-hi", type=float, default=None, help="верхняя граница log10 l2re (CAS)")
    ap.add_argument("--replay", action="store_true", help="воспроизвести волны 33, 36, 40, 41, 43 на обоих уравнениях")
    ap.add_argument("--replay-bounds", default=str(BOUND), help="границы B для --replay через запятую")
    ap.add_argument("--min-seeds", type=int, default=3, help="--replay: теги с меньшим числом сидов не берутся")
    ap.add_argument("--bound-stats", action="store_true", help="распределение парных разностей (обоснование B)")
    ap.add_argument("--out", default="", help="JSON с итогом")
    a = ap.parse_args(argv)
    if not (a.json or a.rows or a.fetch):
        ap.error("нужен --rows, --json или --fetch")
    if not 0 < a.alpha < 1 or a.bound <= 0 or not 0 < a.lam_cap < 1 or not 0 <= a.lam_first <= a.lam_cap:
        ap.error("0 < --alpha < 1, --bound > 0, 0 < --lam-cap < 1, 0 <= --lam-first <= --lam-cap")
    if a.fetch and not a.rows:
        import tempfile
        a.rows = tempfile.mkdtemp(prefix="seq_rank_rows_")
    if a.bound_stats:
        res = bound_stats(a.rows)
    elif a.replay:
        res = replay(a.rows, bounds=[float(x) for x in a.replay_bounds.split(",")], alpha=a.alpha,
                     do_fetch=a.fetch, min_seeds=a.min_seeds, cap=a.lam_cap, first=a.lam_first)
        for t in res["totals"]:
            print(f"ИТОГ {t['pde']} B={t['bound']} по волнам против лидера: сэкономлено {t['saved']} сидов из "
                  f"{t['avail']} (~{t['gpu_h_saved']} GPU-ч), решений {t['decisions']}, против медиан {t['mismatch']}")
        for c in res["cas"]:
            print(f"ИТОГ {c['pde']} CAS k={c['k']} запас {c['margin']}: сэкономлено {c['saved']} сидов из {c['full']} (~{c['gpu_h_saved']} "
                  f"GPU-ч), совпадает со средним log10 {c['exact_ok']}, с медианами {c['median_ok']}")
    else:
        pde = pde_name(a.pde)
        tags = [t for t in a.tags.split(",") if t] or None
        if a.json:
            with open(a.json) as f:
                res = {t: {int(s): float(v) for s, v in r.items()} for t, r in json.load(f).items()}
            if tags:
                res = {t: res[t] for t in tags}
        else:
            if a.fetch:
                print(f"скачано строк: {fetch([f'{pde}_{t}_seed*.json' for t in (tags or ['t3m_*'])], a.rows)}")
            res = load_rows(a.rows, pde, set(tags) if tags else None)
        if len(res) < 2:
            ap.error(f"нужно хотя бы два тега со строками, есть {sorted(res)}")
        sr = run_sequential(res, bound=a.bound, alpha=a.alpha, family=a.family, control=a.control or None,
                            closure=not a.no_closure, goal=a.goal, k=a.k, cap=a.lam_cap, first=a.lam_first)
        sm = sr.summary()
        med = {t: median(r.values()) for t, r in res.items()}
        print(f"агентов {len(res)}, гипотез {sm['n_hyp']}, alpha {a.alpha}, B {a.bound}, цель {a.goal}")
        for t in sm["ranking"]:
            print(f"  {t:28s} сидов {len(sr.x[t])}/{len(res[t])}  медиана {med[t]:.4f}  лучше {sr.n_better(t)}, "
                  f"хуже {sr.n_worse(t)}" + (f"  [{sr.status(t)}]" if a.goal == "topk" else "")
                  + ("  ещё нужен сид" if sr.needs(t) else ""))
        for i, j, n, kind in sm["better"]:
            print(f"  {i} < {j}: {kind}, общих сидов {n}")
        allx = np.log10([v for r in res.values() for v in r.values()])
        lo = a.cas_lo if a.cas_lo is not None else float(allx.min()) - 0.3
        hi = a.cas_hi if a.cas_hi is not None else float(allx.max()) + 0.3
        c = cas_select(res, a.k, lo, hi, scen={t: list(SEEDS) for t in res})
        print(f"CAS k={a.k} на сидах 42..51, границы log10 [{lo:.2f}, {hi:.2f}]: "
              + (f"сертифицировано {c['top']}" if c["certified"] else f"не сертифицировано (кандидаты {c['top']})")
              + f", использовано сидов {c['queries']} из {c['full']}"
              + (f"; следующий нужный сид: {c['next'][0]} сид {c['next'][1]}" if c.get("next") else ""))
        res = dict(summary=sm, cas=dict(top=c["top"], certified=c["certified"], queries=c["queries"], lo=lo, hi=hi))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    return res


if __name__ == "__main__":
    main()
