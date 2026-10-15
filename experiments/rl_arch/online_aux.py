#!/usr/bin/env python
"""Лосс офлайнового рецепта в онлайновом дообучении (online_train_env.py --rlpd-target-buffer,
--mb-aux, --teacher-from / --teacher-w).

Лучший офлайновый агент на картах учится (offline_rl.py) на собранном буфере цели лоссом
    TD + W_mb * Huber(Q(s, .) - Q_mb(s, .)) + W_te * Huber(Q(s, .) - Q_учителя(s, .)),
обе добавки — по всем 27 действиям. Прежнее онлайновое дообучение оставляло от него один TD, а
офлайновую половину батча RLPD брало из исходного случайного буфера задачи, которого агент не видел.
Здесь собрано всё, что нужно, чтобы онлайновый лосс совпадал с офлайновым:

  * load_target — собранный буфер цели тем же загрузчиком и тем же преобразованием переходов, что
    offline_rl.py --target-buffer (target_buffer.load + episodes_to_arrays с chain_fix): те же
    состояния, награды, обрезка по бюджету;
  * Aux.offline — цели строк офлайновой половины: те же функции, что зовёт offline_rl.main
    (target_buffer.surrogate_table, mb_targets.fit_model, mb_targets.precompute,
    offline_rl.teacher_targets), с теми же единицами награды;
  * Aux.live — цели онлайнового перехода в момент его записи в буфер. Цепочка живого прогона
    (действия, фактические эпохи, измеренная ошибка после каждого) записывается строками набора
    тренера (live_data) с подставной последней строкой — состоянием перед очередным решением, и по
    ним считают ТЕ ЖЕ precompute и teacher_targets. Состояние суррогата поэтому собирает
    mb_targets.chain_states — тот же код, что у строк буфера и у щита оценки (live_state);
    tests/test_shield_teacher.py сверяет все пути бит в бит;
  * aux_loss — само слагаемое, то же выражение, что в offline_rl.train_variant.

Обрезка бюджетом: суррогат обрезает действие остатком бюджета (step_features) — как среда тренера
(epochs = min(номинал, бюджет - потрачено)); эпизод в модели кончается, когда бюджет выбран, — как
done в тренере при --episode-budget.

Правило остановки эпизода обучения (online_train_env.py --train-tol / --train-kmax, как у offline_rl.py):
tol и kmax передаются в precompute и teacher_targets, поэтому цели обеих половин за остановку не
смотрят (просмотр вперёд кончается на допуске и на K-й стадии; цель учителя — продолжение его политики
в суррогате до остановки, для неё суррогат строится и при --mb-aux 0). Офлайновая половина обрезается
тем же правилом до расчёта целей (offline_rl.stop_rule_cut в online_train_env.py).

Несколько горизонтов модельной цели (online_train_env.py --mb-hs, как offline_rl.py --mb-hs) передаются
в тот же precompute у обеих половин. Доверие к модельным целям (offline_rl.py --mb-trust) здесь не
поддержано: у онлайнового перехода в момент записи нет суррогата, который его не видел, а вес строк
офлайновой половины без веса онлайновых сделал бы половины несравнимыми — online_train_env.py отказывает.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "track3"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_ACTIONS = 27


def quiet(*a, **k):
    return None


def env_label(keep_opt, mode):
    """Метка среды, как у target_buffer.load: reset либо keep_<режим>."""
    return f"keep_{mode or 'all'}" if keep_opt else "reset"


def load_target(spec, subdir, reward_form, init_err, budget, err_scale=1.0, log=print):
    """Собранный буфер цели для офлайновой половины батча RLPD: загрузчик и преобразование переходов
    — те же вызовы, что в offline_rl.main при --target-buffer --chain-fix. Возвращает (файлы,
    массивы тренера, сводка источников)."""
    import offline_rl as O
    import target_buffer as TB
    episodes, info = TB.load(spec, subdir, log=log)
    data = O.episodes_to_arrays(episodes, fix_next_state=False, chain_fix=True, reward_form=reward_form,
                                init_err=init_err, budget=budget, err_scale=err_scale)
    info = dict(info, used_chains=int(len(np.unique(data["EP"]))), used_transitions=int(len(data["A"])))
    log(f"собранный буфер цели в офлайновой половине RLPD: цепочек {info['used_chains']}, переходов "
        f"{info['used_transitions']}; исходный буфер {subdir} не используется")
    return episodes, data, info


def live_data(acts, eps, errs):
    """Цепочка живого прогона строками набора тренера: исполненные действия (номер, фактические
    эпохи, ошибка после действия в единицах тренера) и подставная последняя строка — состояние перед
    очередным решением. Те же столбцы собирает mb_targets.live_state; ошибка последней строки ни в
    одно состояние не входит (состояние строки k берёт ошибку строки k - 1)."""
    k = len(acts)
    assert len(eps) == k and len(errs) == k, (k, len(eps), len(errs))
    ep = np.asarray(list(eps) + [0], dtype=np.int64)
    return dict(A=np.asarray(list(acts) + [0], dtype=np.int64), EPA=ep, EP=np.zeros(k + 1, np.int64),
                STEP=np.arange(k + 1, dtype=np.int64), SPENT=(np.cumsum(ep) - ep).astype(np.int64),
                ERR=np.asarray(list(errs) + [1.0], dtype=np.float32))


def aux_loss(net, s, batch, w_mb, w_te):
    """Плотные добавки на батче s: W * Huber(Q(s, .) - цель) по всем 27 действиям, среднее по строкам с
    целью (у строк без цели вес 0) — выражение offline_rl.train_variant (dense). batch — словарь
    stack_aux. Возвращает (слагаемое лосса или None, {имя: значение})."""
    import torch.nn.functional as F
    total, parts = None, {}
    qv = None
    for key, w in (("mb", w_mb), ("te", w_te)):
        if not w or batch is None or key not in batch:
            continue
        tgt, wt = batch[key]
        if float(wt.sum()) <= 0:
            continue
        if qv is None:
            qv = net.q_heads(s)                                   # (головы, B, 27)
        hub = F.smooth_l1_loss(qv, tgt[None].expand_as(qv), reduction="none").mean((0, 2))
        l = (hub * wt).sum() / wt.sum()
        total = w * l if total is None else total + w * l
        parts[key] = float(l.item())
    return total, parts


def stack_aux(rows, device):
    """Цели переходов батча (rows — то, что лежит в LocalBuffer.aux: None или словарь
    {mb: (27,), te: (27,)}) -> {mb: (Q (B, 27), вес (B,)), te: (...)}."""
    import torch
    out = {}
    for key in ("mb", "te"):
        q = np.zeros((len(rows), N_ACTIONS), dtype=np.float32)
        w = np.zeros(len(rows), dtype=np.float32)
        for i, r in enumerate(rows):
            if r is not None and r.get(key) is not None:
                q[i], w[i] = r[key], 1.0
        out[key] = (torch.as_tensor(q, device=device), torch.as_tensor(w, device=device))
    return out


def cat_aux(batches):
    """Склейка целей половин батча RLPD (в том же порядке, что состояния)."""
    import torch
    return {k: (torch.cat([b[k][0] for b in batches], 0), torch.cat([b[k][1] for b in batches], 0))
            for k in ("mb", "te")}


class Aux:
    """Цели плотных добавок для обеих половин батча RLPD.

    pde — задача; reward_form — форма награды обеих половин (delta | dlog); err_scale — на что тренер
    делит ошибку (1 без --err-norm init); init_err — ошибка необученной сети в единицах тренера
    (None: награда первого шага 0); budget — бюджет эпизода в эпохах (0 = нет); smdp_scale — как
    у mb_targets.precompute."""

    def __init__(self, pde, w_mb=0.0, w_te=0.0, reward_form="delta", err_scale=1.0, init_err=None, gamma=0.99,
                 budget=0, smdp_scale=0.0, mb_h=3, mb_lam=0.5, mb_subdirs="all", mb_table="", mb_members=5,
                 mb_work="agent_combo_work", quick=False, teacher_from="", device=None, log=print, tol=0.0, kmax=0,
                 mb_hs=None):
        self.pde, self.w_mb, self.w_te = pde, float(w_mb or 0.0), float(w_te or 0.0)
        self.form, self.gamma = reward_form, float(gamma)
        self.scale = np.array([float(err_scale)])
        self.init = None if init_err is None else np.array([float(init_err)])
        self.budget, self.smdp_scale = int(budget or 0), float(smdp_scale or 0.0)
        self.h, self.lam = int(mb_h), float(mb_lam)
        # несколько горизонтов (--mb-hs): глубина — наибольший; None — один горизонт mb_h (прежние цели)
        self.hs = sorted(set(int(x) for x in mb_hs)) if mb_hs else None
        if self.hs:
            self.h = self.hs[-1]
        self.subdirs, self.table, self.members, self.work = mb_subdirs, mb_table, int(mb_members), mb_work
        self.quick, self.device, self.log = bool(quick), device, log
        self.teacher = teacher_from
        self.model = None                   # (таблица суррогата, ансамбль)
        self.mb_info, self.te_info, self.te_mask = {}, {}, []
        self.live_n = dict(calls=0, mb=0, te=0, mb_err=0, te_err=0)
        # правило остановки эпизода обучения: 0 и 0 — выключено (прежние цели)
        self.tol, self.kmax = float(tol or 0.0), int(kmax or 0)
        self.stop_kw = dict(tol=self.tol, kmax=self.kmax) if (self.tol > 0 or self.kmax > 0) else {}
        self.last = {}                      # значения добавок на последнем обновлении

    # ---- модель среды ------------------------------------------------------------------------
    def build_model(self, episodes=None, model=None):
        """Таблица и ансамбль суррогата — как у offline_rl.py --mb-aux. episodes — файлы собранного
        буфера цели (--rlpd-target-buffer): строки цели в таблице только из них
        (target_buffer.surrogate_table, как при --target-buffer); без них — таблица исходных буферов
        (mb_targets.load_tab). model=(tab, ens) — готовые (тесты)."""
        import mb_targets as MB
        if model is not None:
            tab, ens = model
            info = dict(surr_rows=int(len(tab["A"])), surr_sig="given", surr_members=int(ens.members))
        else:
            if episodes is not None:
                import target_buffer as TB
                tab = TB.surrogate_table(episodes, self.pde, self.subdirs, self.table, self.work, log=self.log)
            else:
                tab = MB.load_tab([self.pde], self.subdirs, self.table, self.work, self.log)
            tab, ens, info = MB.fit_model(tab, self.members, self.work, self.quick, self.log)
        self.model, self.mb_info = (tab, ens), dict(info)
        return self.model

    def teacher_file(self):
        """Чекпоинт учителя: локальный путь (из HF скачивается один раз, дальше читается с диска)."""
        if self.teacher and not os.path.exists(self.teacher):
            import offline_rl as O
            from huggingface_hub import hf_hub_download
            self.teacher = hf_hub_download(O.OUT_REPO, self.teacher, repo_type="dataset")
        return self.teacher

    # ---- цели ------------------------------------------------------------------------------
    def _mb(self, data, mask, log):
        import mb_targets as MB
        return MB.precompute(data, mask, names=[self.pde], reward_form=self.form, err_scale=self.scale,
                             init_err=self.init, gamma=self.gamma, h=self.h, lam=self.lam, budget=self.budget,
                             smdp_scale=self.smdp_scale, model=self.model, log=log, **self.stop_kw,
                             **({"hs": self.hs} if self.hs else {}))

    def _te(self, data, log):
        import offline_rl as O
        kw = (dict(self.stop_kw, model=self.model, lam=self.lam, smdp_scale=self.smdp_scale)
              if self.stop_kw else {})
        return O.teacher_targets(self.teacher_file(), data, names=[self.pde], reward_form=self.form,
                                 err_scale=self.scale, init_err=self.init, gamma=self.gamma,
                                 budget=self.budget, device=self.device, log=log, **kw)

    def offline(self, data, train_mask, episodes=None, model=None):
        """Цели строк офлайновой половины (data — массивы тренера с chain_fix, train_mask — обучающие
        строки): те же вызовы, что в offline_rl.main. Возвращает {QMB, MBW, QTE, TEW} (нет добавки —
        нет ключей); у строк вне обучающей части и без цели вес 0."""
        out = {}
        train_mask = np.asarray(train_mask, dtype=bool)
        if self.w_mb:
            if self.model is None:
                self.build_model(episodes, model)
            out["QMB"], out["MBW"], info = self._mb(data, train_mask, self.log)
            # сведения о суррогате (отпечаток, файл ансамбля) — от fit_model, остальное — от расчёта целей
            self.mb_info.update({k: v for k, v in info.items() if not k.startswith("surr_")})
        if self.w_te:
            if self.stop_kw and self.model is None:
                self.build_model(episodes, model)       # цели учителя под правилом остановки идут через суррогат
            out["QTE"], tev, info = self._te(data, self.log)
            self.te_mask = list(info.pop("mask"))
            out["TEW"] = tev & train_mask
            self.te_info = dict(info, train_rows=int(out["TEW"].sum()))
        return out

    @staticmethod
    def row(off, i):
        """Цели строки i офлайнового набора в виде записи буфера (LocalBuffer.aux)."""
        r = {}
        if "QMB" in off and off["MBW"][i]:
            r["mb"] = off["QMB"][i]
        if "QTE" in off and off["TEW"][i]:
            r["te"] = off["QTE"][i]
        return r or None

    def live(self, acts, eps, errs):
        """Цели состояния живой цепочки ПЕРЕД очередным действием: acts, eps, errs — исполненные
        действия, их фактические эпохи и ошибка после каждого (в единицах тренера). Запись буфера
        {mb, te} или None, если ни одной цели нет (состояние вне бюджета учителя, нечисловое)."""
        data = live_data(acts, eps, errs)
        k = len(acts)
        self.live_n["calls"] += 1
        r = {}
        if self.w_mb:
            import agent_combo as AC
            try:
                last = np.zeros(k + 1, dtype=bool)
                last[k] = True
                # одна строка состояния: предсказания ансамбля в один поток (запуск потоков OpenMP на
                # десятках строк стоит дороже самой работы — agent_combo.one_thread)
                with AC.one_thread():
                    q, w, _ = self._mb(data, last, quiet)
                if w[k]:
                    r["mb"] = q[k]
                    self.live_n["mb"] += 1
            except (ValueError, FloatingPointError) as e:
                self._warn("mb_err", e)
        if self.w_te:
            try:
                q, v, _ = self._te(data, quiet)
                if v[k]:
                    r["te"] = q[k]
                    self.live_n["te"] += 1
            except ValueError as e:         # «ни у одной строки нет цели»: шаг вне бюджета учителя
                self._warn("te_err", e)
        return r or None

    def _warn(self, key, e):
        self.live_n[key] += 1
        if self.live_n[key] == 1:
            self.log(f"плотные цели онлайнового перехода ({key}): {type(e).__name__}: {str(e)[:160]} — у перехода "
                     "вес добавки 0 (дальше без повторов сообщения)")

    # ---- запись в meta и строку результата -----------------------------------------------------
    def meta(self):
        m = dict(mb_aux=self.w_mb, mb_h=self.h, mb_lam=self.lam, mb_subdirs=self.subdirs,
                 mb_members=self.members, mb_table=os.path.basename(self.table), teacher_w=self.w_te,
                 teacher_from=os.path.basename(self.teacher or ""))
        if self.stop_kw:
            m.update(aux_stop_tol=self.tol, aux_stop_kmax=self.kmax)
        if self.hs:
            m.update(mb_hs=",".join(str(x) for x in self.hs))
        if self.w_mb:
            m["mb"] = dict(self.mb_info)                # mb.surr_file читает щит оценки (--shield auto)
        if self.w_te:
            m["teacher"] = dict(self.te_info, mask=list(self.te_mask))   # маску учителя агент не наследует
        return m

    def row_fields(self):
        out = dict(mb_aux=self.w_mb, mb_h=self.h, mb_lam=self.lam, teacher_w=self.w_te,
                   teacher_from=os.path.basename(self.teacher or ""),
                   aux_online=dict(self.live_n), **{"aux_" + k + "_loss": v for k, v in self.last.items()})
        if self.hs:
            out.update(mb_hs=",".join(str(x) for x in self.hs))
        out.update({"mb_" + k: v for k, v in self.mb_info.items()})
        out.update({"teacher_" + k: v for k, v in self.te_info.items()})
        return out
