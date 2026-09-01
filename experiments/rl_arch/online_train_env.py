#!/usr/bin/env python
"""
ПОЛНЫЙ онлайн-цикл PINN-PELINE с обучением агента (как в rl_trainer.py авторов),
но с возможностью подменить архитектуру Q-сети (CNN / ConvNeXt, DQN / CQL /
дуэлинговая голова).

Отличие от online_eval_env.py: там политика заморожена и меряется одна цепочка;
здесь агент УЧИТСЯ по ходу — переходы копятся в ЛОКАЛЬНОМ буфере, после каждой
цепочки веса PINN переинициализируются, и так до исчерпания лимита времени.

Метрика: l2re последней ЗАВЕРШЁННОЙ цепочки. Оборванная по времени цепочка не
учитывается — её частичный результат хранится отдельно (last_partial), поэтому
срез сессии Kaggle никогда не портит итог.

Состояние строится их же кодом (автоэнкодер над траекторией весов -> сетка 26x26
в латенте -> лоссы -> sign*log1p -> канал delta), см. online_eval_env.

  python experiments/rl_arch/online_train_env.py --variant convnext_dqn \
      --pde poissonboltzmann2d --hours 11 --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("DDEBACKEND", "pytorch")

import torch  # noqa: E402
import deepxde as dde  # noqa: E402

from online_eval_env import (ACTION_TABLE, AE_MODEL_PARAMS, GRID_RANGE, LOSS_TYPES,  # noqa: E402
                             OUT_REPO, build_optimizer, build_state)
from offline_rl import QNet  # noqa: E402

EPS_START, EPS_END, EPS_DECAY = 0.5, 0.05, 50   # значения авторов (rl_algorithms.py)
GAMMA = 0.9                                      # их rl_agent_params


class LocalBuffer:
    """Локальный буфер переходов. На HF ничего не уходит — только итоги цепочек.

    gam хранится по переходу: для n-шаговых возвратов это gamma**n (или меньше,
    если эпизод кончился раньше), для обычных — gamma. prio нужен для PER."""

    def __init__(self, capacity=10000, per=False, per_alpha=0.6):
        self.cap = capacity
        self.per, self.alpha = per, per_alpha
        self.s, self.a, self.r, self.s2, self.d, self.gam, self.prio = ([] for _ in range(7))
        self.last_idx = None

    def push(self, s, a, r, s2, d, gam=None):
        if len(self.s) >= self.cap:
            for arr in (self.s, self.a, self.r, self.s2, self.d, self.gam, self.prio):
                arr.pop(0)
        self.s.append(s); self.a.append(a); self.r.append(r)
        self.s2.append(s2); self.d.append(d)
        self.gam.append(GAMMA if gam is None else gam)
        self.prio.append(max(self.prio) if self.prio else 1.0)   # новый переход — макс. приоритет

    def __len__(self):
        return len(self.s)

    def sample(self, n, device):
        k = min(n, len(self.s))
        if self.per and len(self.s) > 1:
            w = np.asarray(self.prio, dtype=np.float64) ** self.alpha
            idx = np.random.choice(len(self.s), size=k, p=w / w.sum())
        else:
            idx = np.random.randint(0, len(self.s), size=k)
        self.last_idx = idx
        t = lambda arr, dt: torch.as_tensor(np.array([arr[i] for i in idx]), dtype=dt, device=device)
        return (t(self.s, torch.float32), t(self.a, torch.long), t(self.r, torch.float32),
                t(self.s2, torch.float32), t(self.d, torch.float32), t(self.gam, torch.float32))

    def update_prio(self, td):
        if not self.per or self.last_idx is None:
            return
        for i, e in zip(self.last_idx, td):
            self.prio[int(i)] = float(abs(e)) + 1e-3


def agent_update(net, buf, opt, batch_size, iters, variant, cql_alpha=1.0,
                 munch=False, m_tau=0.03, m_alpha=0.9, l2_init=0.0, init_w=None):
    import torch.nn.functional as F
    dev = next(net.model.parameters()).device
    for _ in range(iters):
        s, a, r, s2, d, gm = buf.sample(batch_size, dev)
        q = net.q_scalar(s).gather(1, a[:, None]).squeeze(1)
        if munch:
            tgt = munchausen_target(net, s, a, r, s2, d, gm, m_tau, m_alpha)
        else:
            with torch.no_grad():
                a2 = net.q_scalar(s2).argmax(1)
                q2 = (net.q_target(s2).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                      else net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
                tgt = r + gm * (1 - d) * q2
        loss = F.mse_loss(q, tgt)
        if l2_init and init_w is not None:
            loss = loss + l2_init * sum(((p - p0) ** 2).sum()
                                        for p, p0 in zip(net.model.parameters(), init_w))
        if "cql" in variant:
            qs = net.q_scalar(s)
            loss = loss + cql_alpha * (torch.logsumexp(qs, 1) - qs.gather(1, a[:, None]).squeeze(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        net.soft_update()
    return float(loss.item())


def _dihedral_pair(s, s2):
    """Одинаковое случайное диэдральное преобразование на пару (s, s2) каждого
    примера: поворот на k*90 + отражение. Согласованность пары сохраняет
    отношение канала delta между состояниями."""
    B = s.shape[0]
    ks = torch.randint(0, 4, (B,))
    fs = torch.randint(0, 2, (B,))
    outs, outs2 = [], []
    for i in range(B):
        a, b = torch.rot90(s[i], int(ks[i]), (1, 2)), torch.rot90(s2[i], int(ks[i]), (1, 2))
        if fs[i]:
            a, b = torch.flip(a, (2,)), torch.flip(b, (2,))
        outs.append(a); outs2.append(b)
    return torch.stack(outs), torch.stack(outs2)


def agent_update_mixed(net, off_buf, on_buf, opt, half, variant, cql_alpha=1.0,
                       max_bellman=False, aug=False, munch=False, m_tau=0.03,
                       m_alpha=0.9, l2_init=0.0, init_w=None):
    """RLPD: симметричная выборка — половина батча офлайн, половина онлайн."""
    import torch.nn.functional as F
    dev = next(net.model.parameters()).device
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d, gm = [torch.cat([p[i] for p in parts], 0) for i in range(6)]
    if aug:
        s, s2 = _dihedral_pair(s, s2)
    q = net.q_scalar(s).gather(1, a[:, None]).squeeze(1)
    if munch:
        tgt = munchausen_target(net, s, a, r, s2, d, gm, m_tau, m_alpha)
    else:
        with torch.no_grad():
            a2 = net.q_scalar(s2).argmax(1)
            q2 = (net.q_target(s2).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                  else net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
            tgt = (torch.maximum(r, gm * (1 - d) * q2) if max_bellman
                   else r + gm * (1 - d) * q2)
    loss = F.mse_loss(q, tgt)
    if l2_init and init_w is not None:
        loss = loss + l2_init * sum(((p - p0) ** 2).sum()
                                    for p, p0 in zip(net.model.parameters(), init_w))
    if "cql" in variant:
        qs = net.q_scalar(s)
        loss = loss + cql_alpha * (torch.logsumexp(qs, 1) - qs.gather(1, a[:, None]).squeeze(1)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    net.soft_update()
    with torch.no_grad():
        td = (q - tgt).abs().cpu().numpy()
        half_n = len(td) // 2
        off_buf.update_prio(td[:half_n]); on_buf.update_prio(td[half_n:])
    return float(loss.item())


def behaviour_update(behaviour, opt, buf, batch_size, iters, device):
    """Модель поведения для маски BCQ: что вообще встречалось в данных."""
    import torch.nn.functional as F
    loss = 0.0
    for _ in range(iters):
        s, a, _, _, _, _ = buf.sample(batch_size, device)
        loss = F.cross_entropy(behaviour(s.flatten(1)), a)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def shrink_and_perturb(net, alpha, device):
    """SR-SPR/BBF: голова заново, энкодер = alpha*старые + (1-alpha)*свежие.
    Буфер и счётчик шагов сохраняются — сбрасывается только аппроксиматор."""
    import copy
    import torch.nn as nn

    def reinit(m):
        # переинициализация на месте: не зависит от сигнатуры конструктора сети
        # (у QNet это (variant, device), у BootQNet — (device, n_heads))
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    fresh = copy.deepcopy(net.model)
    fresh.apply(reinit)
    n = 0
    with torch.no_grad():
        for (name, p_old), (_, p_new) in zip(net.model.named_parameters(),
                                             fresh.named_parameters()):
            if name.startswith("1."):          # голова (ModuleList: 0=энкодер, 1=голова)
                p_old.copy_(p_new)
            else:                               # энкодер: сжатие с возмущением
                p_old.mul_(alpha).add_(p_new, alpha=1.0 - alpha)
            n += 1
    net.target.load_state_dict(net.model.state_dict())
    return n


SCALAR_CH = 5


EPOCHS_TABLE = [100, 1000, 2500, 100, 500, 1000, 100, 200, 300]


def offline_ctx(od, k_max=10, budget=31000):
    """Контекст для офлайновых переходов. Из пяти скаляров четыре восстановимы:
    номер шага — по позиции внутри эпизода, прошлое действие — по предыдущему
    переходу, доля бюджета — по сумме эпох уже выбранных действий (число эпох
    закодировано в самом действии). Уровень ошибки восстановить нельзя: награда
    даёт разности, а не абсолютные значения — этот канал остаётся нулевым."""
    EP, A = od["EP"], od["A"]
    n = len(A)
    out = np.zeros((n, SCALAR_CH), dtype=np.float32)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        spent = 0
        for j, i in enumerate(idx):
            prev = A[idx[j - 1]] if j > 0 else -1
            if prev < 0:
                opt_i = ep_i = -1.0
            else:
                opt_i = (prev // 9) / 2.0
                ep_i = (prev % 3) / 2.0
            out[i] = [j / max(1, k_max), min(1.0, spent / max(1, budget)),
                      opt_i, ep_i, 0.0]
            a = int(A[i])
            spent += EPOCHS_TABLE[(a // 9) * 3 + (a % 3)]
    return out


def attach_ctx(S, ctx):
    """Разворачивает скаляры в постоянные каналы и приклеивает к картам."""
    n, _, h, w = S.shape
    planes = np.broadcast_to(ctx[:, :, None, None], (n, SCALAR_CH, h, w))
    return np.concatenate([S, planes.astype(np.float32)], axis=1)


def pad_norm(mean, std, n_ch):
    """Статистики нормировки приходят из 4-канального буфера или чекпоинта тёплого
    старта. Контекстные каналы уже лежат в [-1,1], поэтому им нужны нулевое среднее
    и единичный разброс — дополняем, а не пересчитываем."""
    if mean is None or mean.shape[1] >= n_ch:
        return mean, std
    k = n_ch - mean.shape[1]
    mean = np.concatenate([mean, np.zeros((1, k, 1, 1), np.float32)], axis=1)
    std = np.concatenate([std, np.ones((1, k, 1, 1), np.float32)], axis=1)
    return mean, std


def add_scalar_ctx(state, step, k_max, spent, budget, last_action, err):
    """Контекст постоянными каналами: форма (4+5, 26, 26). Каналы-константы —
    приём из работ по добавлению координатных и временных признаков в свёрточные
    сети; сохраняет совместимость со всей машинерией буферов и нормировки."""
    h, w = state.shape[1], state.shape[2]
    if last_action is None or last_action < 0:
        opt_i = lr_i = ep_i = -1.0
    else:
        opt_i = (last_action // 9) / 2.0
        lr_i = ((last_action % 9) // 3) / 2.0
        ep_i = (last_action % 3) / 2.0
    vals = [
        step / max(1, k_max),                                   # доля пройденных шагов
        min(1.0, spent / max(1, budget)),                       # доля бюджета
        opt_i, ep_i,
        float(np.clip(np.log10(max(err, 1e-8)) / 3.0 + 1.0, -1, 1)),   # уровень ошибки
    ]
    planes = np.stack([np.full((h, w), v, dtype=np.float32) for v in vals])
    return np.concatenate([state, planes], axis=0)


def make_spr(dim, device, n_actions=27, hidden=256):
    """SPR (Schwarzer, ICLR 2021): латентная модель перехода + проектор и предиктор.
    Предсказывает представление следующего состояния по действию, цель — целевая
    копия энкодера. Наград не требует, поэтому учится и на сорванных цепочках.
    Собирается функцией, а не классом модульного уровня: torch импортируется
    внутри main()."""
    import torch.nn as nn

    class SprHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.act_emb = nn.Embedding(n_actions, dim)
            self.trans = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(),
                                       nn.Linear(hidden, dim))
            self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, dim))
            self.pred = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, dim))

        def forward(self, z, a):
            return self.trans(torch.cat([z, self.act_emb(a)], dim=-1))

    return SprHead().to(device)


def spr_loss(net, spr, s, a, s2):
    """Косинусная потеря между предсказанным и целевым латентом (BYOL-стиль:
    градиент идёт только через онлайн-ветвь)."""
    import torch.nn.functional as F
    z = net.model[0](s)
    z_hat = spr.pred(spr.proj(spr(z, a)))
    with torch.no_grad():
        z_tgt = spr.proj(net.target[0](s2))
    return -F.cosine_similarity(z_hat, z_tgt, dim=-1).mean()


def hlg_update(net, buf, opt, batch_size, iters, hlg, gamma_default=0.9,
               munch=False, m_tau=0.03, m_alpha=0.9):
    """HL-Gauss (Farebrother, ICML 2024): цель Q превращается в сглаженное
    категориальное распределение по бинам, обучение — кросс-энтропией вместо MSE.
    Работает поверх квантильной головы: её nq выходов трактуются как бины."""
    for _ in range(iters):
        s, a, r, s2, d, gm = buf.sample(batch_size, dev_of(net))
        logits = net.q_online(s)                                  # (B,A,bins)
        taken = logits.gather(1, a[:, None, None].expand(-1, 1, logits.shape[-1])).squeeze(1)
        with torch.no_grad():
            q2 = hlg.to_scalar(net.q_target(s2))                  # (B,A)
            a2 = hlg.to_scalar(net.q_online(s2)).argmax(1)
            y = r + gm * (1 - d) * q2.gather(1, a2[:, None]).squeeze(1)
        loss = hlg.loss(taken, y).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        net.soft_update()
    return float(loss.item())


def hlg_update_mixed(net, off_buf, on_buf, opt, half, hlg):
    """HL-Gauss на симметричном батче RLPD: половина офлайн, половина онлайн.
    Нужен отдельно, потому что ветка RLPD не проходит через hlg_update."""
    dev = dev_of(net)
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d, gm = [torch.cat([p[i] for p in parts], 0) for i in range(6)]
    logits = net.q_online(s)                                     # (B,A,bins)
    taken = logits.gather(1, a[:, None, None].expand(-1, 1, logits.shape[-1])).squeeze(1)
    with torch.no_grad():
        a2 = hlg.to_scalar(net.q_online(s2)).argmax(1)
        q2 = hlg.to_scalar(net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
        y = r + gm * (1 - d) * q2
    loss = hlg.loss(taken, y).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    net.soft_update()
    return float(loss.item())


def dev_of(net):
    return next(net.model.parameters()).device


def distill_loss(net, S, A, temp=1.0):
    """Дистилляция экспертных действий в Q-голову: кросс-энтропия softmax(Q/T)
    к действиям лучших цепочек. Выравнивает argmax политики с тем, что нашла
    разведка, — прямая атака на измеренный разрыв разведка→политика."""
    import torch.nn.functional as F
    return F.cross_entropy(net.q_scalar(S) / temp, A)


def anneal(step, total, lo, hi):
    """Линейный отжиг lo->hi по доле пройденного обучения (BBF)."""
    f = min(1.0, max(0.0, step / max(1, total)))
    return lo + (hi - lo) * f


def deep_search(net, wm, x, gamma=0.9, alpha=0.5, depth=2, beam=5):
    """Лучевой поиск глубины D по выученной модели мира. В статье QWM берёт D=1;
    здесь дерево разворачивается глубже, но узко — полное 27^D неподъёмно, а ошибка
    модели с глубиной копится, поэтому листья оцениваются целевой Q, а не
    продолжением развёртки."""
    from advanced_agents import N_ACTIONS
    with torch.no_grad():
        q_root = net.q_scalar(x)[0]                                  # (27,)
        eye = torch.eye(N_ACTIONS, device=x.device)
        s_flat = x.flatten(1).expand(N_ACTIONS, -1)
        d_hat, r_hat = wm(s_flat, eye)
        cur = (s_flat + d_hat).view(N_ACTIONS, *x.shape[1:])
        # ret[i] — накопленная награда ветви, root[i] — её корневое действие
        ret = r_hat.clone()
        root = torch.arange(N_ACTIONS, device=x.device)
        disc = gamma
        for _ in range(max(0, depth - 1)):
            v = net.q_target(cur).max(1).values
            k = min(beam, cur.shape[0])
            top = torch.topk(ret + disc * v, k=k).indices
            f = cur[top].flatten(1).repeat_interleave(N_ACTIONS, 0)
            e = eye.repeat(k, 1)
            d2, r2 = wm(f, e)
            cur = (f + d2).view(-1, *x.shape[1:])
            ret = ret[top].repeat_interleave(N_ACTIONS) + disc * r2
            root = root[top].repeat_interleave(N_ACTIONS)
            disc *= gamma
        score = ret + disc * net.q_target(cur).max(1).values
        # свёртка листьев к корневым действиям: максимум по ветвям каждого корня
        best = torch.full((N_ACTIONS,), -1e9, device=x.device)
        best = best.scatter_reduce(0, root, score, reduce="amax", include_self=True)
        seen = torch.zeros(N_ACTIONS, dtype=torch.bool, device=x.device)
        seen[root] = True
        best = torch.where(seen, best, q_root)      # неразвёрнутые корни судим по Q
        return int((alpha * q_root + (1 - alpha) * best).argmax().item())


def munchausen_target(net, s, a, r, s2, d, gm, tau=0.03, alpha=0.9, clip=-1.0):
    """Munchausen RL: к награде добавляется масштабированный log-policy текущего
    состояния. Одна строка по сути, но даёт неявную KL-регуляризацию к прошлой
    политике — против шумных целей. Мягкий максимум по целевой сети вместо жёсткого."""
    import torch.nn.functional as F
    with torch.no_grad():
        qt = net.q_target(s)
        if qt.dim() == 3:
            qt = qt.mean(-1)
        logp = F.log_softmax(qt / tau, dim=1)
        # клипуется произведение tau*log pi (как в статье), а не голый log pi:
        # при tau=0.03 log-softmax почти всегда ниже клипа, и штраф вырождался
        # в плоские -alpha за любое действие кроме фаворита — 96% переходов
        # на насыщении при медианной награде 0.05. Это и был двигатель коллапса.
        m = alpha * (tau * logp.gather(1, a[:, None]).squeeze(1)).clamp(min=clip)

        qt2 = net.q_target(s2)
        if qt2.dim() == 3:
            qt2 = qt2.mean(-1)
        p2 = F.softmax(qt2 / tau, dim=1)
        logp2 = F.log_softmax(qt2 / tau, dim=1)
        soft_v = (p2 * (qt2 - tau * logp2)).sum(1)
        return r + m + gm * (1 - d) * soft_v


def redo_recycle(net, buf, batch_size, dev, thresh=0.1):
    """ReDo (Sokar и др., ICML 2023). Спящий нейрон: нормированная активация ниже
    порога от средней по слою. Переработка по статье: входящие веса заново,
    ИСХОДЯЩИЕ — в ноль, чтобы переработка не возмущала выход сети.
    Исправлено после аудита: (1) для свёрток активация усреднялась по (B,C,H),
    оставляя ось ширины — спящесть считалась по столбцам карты, а не по каналам;
    (2) обнуления исходящих не было вовсе."""
    import torch.nn as nn
    acts, hooks = {}, []

    def mk(name):
        def h(_m, _i, o):
            t = o.detach().abs()
            # (B,C,H,W) -> по каналам; (B,F) -> по нейронам
            t = t.mean(dim=(0, 2, 3)) if t.dim() == 4 else t.mean(0)
            acts[name] = acts.get(name, 0) + t
        return h

    order = [(n, m) for n, m in net.model.named_modules()
             if isinstance(m, (nn.Linear, nn.Conv2d))]
    for name, m in order:
        hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        s, _, _, _, _, _ = buf.sample(min(batch_size, len(buf)), dev)
        net.q_scalar(s)
    for h in hooks:
        h.remove()

    n_reset = 0
    with torch.no_grad():
        for k, (name, m) in enumerate(order):
            a = acts.get(name)
            if a is None or a.numel() < 2 or a.numel() != m.weight.shape[0]:
                continue
            score = a / (a.mean() + 1e-9)
            dead = (score < thresh).nonzero(as_tuple=True)[0]
            if not len(dead) or len(dead) == a.numel():
                continue
            w = m.weight
            # следующий слой с совпадающей входной размерностью — для обнуления
            # исходящих; depthwise-свёртки и残 остатки пропускаются честно
            nxt = None
            for _, m2 in order[k + 1:k + 2]:
                if m2.weight.dim() >= 2 and m2.weight.shape[1] == w.shape[0]:
                    nxt = m2
                break
            for i in dead.tolist():
                nn.init.kaiming_uniform_(w[i:i + 1] if w.dim() > 1
                                         else w[i:i + 1].view(1, -1))
                if m.bias is not None:
                    m.bias[i] = 0.0
                if nxt is not None:
                    nxt.weight[:, i] = 0.0
                n_reset += 1
    if n_reset:
        net.target.load_state_dict(net.model.state_dict())
    return n_reset


def _hlg_meta(hlg):
    """Метаданные HL-Gauss для чекпоинта: без них оценка усреднит логиты бинов
    как квантили и выберет действие по бессмысленной величине."""
    return None if hlg is None else (float(hlg.edges[0]), float(hlg.edges[-1]), int(hlg.n_bins))


def save_agent(net, mean, std, variant, tag, n_chains, best, hl_gauss=None):
    """Чекпоинт агента: локально всегда, на HF — если есть токен. Формат тот же,
    что у офлайновых агентов, чтобы online_eval_env мог его загрузить."""
    payload = dict(variant=variant, hl_gauss=hl_gauss, state_dict={k: v.detach().cpu()
                                                for k, v in net.model.state_dict().items()},
                   mean=mean, std=std, n_chains=n_chains, l2re_best=best, tag=tag)
    local = f"agent_{tag}.pt"
    torch.save(payload, local)
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    from huggingface_hub import upload_file
    for attempt in range(4):
        try:
            upload_file(path_or_fileobj=local, path_in_repo=f"rl_arch/agents_online/{tag}.pt",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"online agent {tag} ({n_chains} цепочек)")
            return
        except Exception as e:
            print(f"agent upload retry {attempt}: {str(e)[:80]}", flush=True)
            time.sleep(min(300, 20 * 2 ** attempt))


def upload(row, name):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    import io
    from huggingface_hub import upload_file
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(json.dumps(row, indent=1).encode()),
                        path_in_repo=f"rl_arch/online_train/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch online_train {name}")
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(8 * (attempt + 1))


def main():
    global GAMMA        # BBF отжигает дисконт; без объявления GAMMA стала бы
                        # локальной для всей main() и падала бы на первом чтении
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="convnext_dqn")
    ap.add_argument("--pde", default="poissonboltzmann2d")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hours", type=float, default=11.0, help="лимит по времени (сессия Kaggle)")
    ap.add_argument("--max-chain-steps", type=int, default=12, help="K_max из статьи")
    ap.add_argument("--tolerance", type=float, default=0.0, help="досрочный конец цепочки по ошибке")
    ap.add_argument("--n-save-models", type=int, default=10)
    ap.add_argument("--ae-epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-buffer", type=int, default=32)
    ap.add_argument("--update-iters", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm-start", default=None, help="чекпоинт офлайн-агента (опционально)")
    ap.add_argument("--rlpd", action="store_true",
                    help="RLPD (arXiv 2302.02948): симметричная выборка 50/50 из офлайнового "
                         "и онлайнового буферов + высокое число обновлений на шаг среды")
    ap.add_argument("--rlpd-subdir", default="poisson3d_complexgeometry",
                    help="какой офлайн-буфер подмешивать")
    ap.add_argument("--rlpd-utd", type=int, default=8,
                    help="обновлений на шаг среды (update-to-data ratio)")
    ap.add_argument("--rlpd-full", action="store_true",
                    help="RLPD целиком: к симметричной выборке добавить LayerNorm в "
                         "критике и ансамбль из N критиков со случайным подмножеством "
                         "размера M для таргета (в статье N=10, M=2)")
    ap.add_argument("--rlpd-ensemble", type=int, default=10)
    ap.add_argument("--rlpd-subset", type=int, default=2)
    ap.add_argument("--qwm", action="store_true",
                    help="QWM (arXiv 2608.17163): мировая модель поверх Q-обучения — "
                         "одношаговый поиск по всем 27 действиям при выборе действия. "
                         "Модель предобучается на офлайновом буфере (--rlpd-subdir) и "
                         "дообучается онлайн; критик учится только на реальных переходах")
    ap.add_argument("--qwm-pretrain", type=int, default=4000,
                    help="шагов предобучения мировой модели на офлайновом буфере")
    ap.add_argument("--qwm-alpha", type=float, default=0.5,
                    help="вес критика в комбинированной оценке (Eq. 9: 0.5)")
    ap.add_argument("--plain-fnn", action="store_true",
                    help="обратные задачи: обычный FNN вместо PFNN — конвейер карт "
                         "авторов (extract_layers_from_dde_fnn) ветвящиеся сети не "
                         "разбирает; физике обратной задачи FNN с 2 выходами достаточен")
    ap.add_argument("--budget", type=int, default=31000,
                    help="бюджет эпох на цепочку — нужен скалярному контексту")
    ap.add_argument("--scalar-ctx", action="store_true",
                    help="добавить к картам постоянные каналы с контекстом: доля пройденных "
                         "шагов, доля израсходованного бюджета, прошлый оптимизатор и его "
                         "длительность, логарифм текущей ошибки. Без них политика решает "
                         "частично наблюдаемую задачу как полностью наблюдаемую")
    ap.add_argument("--smdp", action="store_true",
                    help="полу-марковская трактовка (Sutton, Precup, Singh 1999): дисконт "
                         "по ДЛИТЕЛЬНОСТИ действия gamma^(эпох/100), а не по числу шагов. "
                         "Наши действия длятся 100–2500 эпох — плоский дисконт объявляет их "
                         "равными по времени")
    ap.add_argument("--smdp-scale", type=float, default=100.0,
                    help="сколько эпох считать одной единицей времени SMDP")
    ap.add_argument("--search-depth", type=int, default=0,
                    help="поиск на этапе действия глубины D по модели мира (нужен --qwm): "
                         "разворачивает дерево 27^D с отсечением по лучшим ветвям")
    ap.add_argument("--search-beam", type=int, default=5,
                    help="ширина луча: сколько лучших действий разворачивать на каждом уровне")
    ap.add_argument("--pbrs", type=float, default=0.0,
                    help="потенциальное преобразование награды (Ng, Harada, Russell 1999): "
                         "F = gamma*Phi(s2) - Phi(s), Phi = -log10(ошибка). Сохраняет "
                         "множество оптимальных политик, но выравнивает плотный сигнал "
                         "с конечной метрикой")
    ap.add_argument("--greedy-probe", type=int, default=0,
                    help="каждая N-я траектория идёт жадно (eps=0) и служит пробой качества "
                         "политики; агент сохраняется по ЛУЧШЕЙ пробе, а не по последнему "
                         "чекпоинту — метрика отбора совпадает с метрикой деплоя")
    ap.add_argument("--distill", type=int, default=0,
                    help="дистилляция лучших цепочек: после каждой цепочки K обновлений "
                         "кросс-энтропией к действиям верхних цепочек прогона. Переносит "
                         "найденное разведкой в жадную политику")
    ap.add_argument("--distill-top", type=int, default=3,
                    help="сколько лучших цепочек считать экспертными")
    ap.add_argument("--distill-w", type=float, default=0.5,
                    help="вес дистилляционной потери")
    ap.add_argument("--hl-gauss", action="store_true",
                    help="HL-Gauss (Farebrother, ICML 2024): цель как категориальное "
                         "распределение по бинам вместо скалярной регрессии")
    ap.add_argument("--spr", type=int, default=0,
                    help="SPR (Schwarzer, ICLR 2021): предсказание будущего латента на K шагов "
                         "вперёд, косинусная потеря к EMA-цели. Не требует наград")
    ap.add_argument("--spr-w", type=float, default=1.0)
    ap.add_argument("--bbf", action="store_true",
                    help="полный BBF: SPR + отжиг горизонта n-step с 10 до 3 и дисконта "
                         "с 0.97 к 0.997 по ходу обучения")
    ap.add_argument("--munchausen", action="store_true",
                    help="Munchausen RL (Vieillard, NeurIPS 2020): неявная KL-регуляризация "
                         "к прошлой политике через log-policy в награде")
    ap.add_argument("--m-tau", type=float, default=0.03)
    ap.add_argument("--m-alpha", type=float, default=0.9)
    ap.add_argument("--redo-every", type=int, default=0,
                    help="ReDo (Sokar, ICML 2023): переработка спящих нейронов каждые N "
                         "обновлений — избирательная альтернатива полному сбросу")
    ap.add_argument("--redo-thresh", type=float, default=0.1)
    ap.add_argument("--l2-init", type=float, default=0.0,
                    help="регенеративная регуляризация (Kumar, ICLR 2024): штраф за отход "
                         "весов от инициализации вместо отхода от нуля")
    ap.add_argument("--reset-every", type=int, default=0,
                    help="сброс сети каждые N обновлений (SR-SPR/BBF): голова "
                         "инициализируется заново, энкодер сжимается и возмущается. "
                         "Лечит primacy bias — застревание на первых неудачных переходах")
    ap.add_argument("--reset-alpha", type=float, default=0.5,
                    help="доля старых весов энкодера при сбросе (shrink-and-perturb)")
    ap.add_argument("--self-prior", type=int, default=0,
                    help="RLPD без внешнего офлайн-буфера: первые N собранных переходов "
                         "замораживаются как опорная половина батча, дальше обычная "
                         "симметричная выборка 50/50 против растущего онлайн-буфера")
    ap.add_argument("--save-agent", action="store_true",
                    help="сохранять веса агента (локально и на HF) по ходу и в конце")
    ap.add_argument("--save-every", type=int, default=5,
                    help="как часто сохранять агента, в завершённых цепочках")
    ap.add_argument("--cvar-alpha", type=float, default=0.0,
                    help="классический CVaR: действовать по среднему НИЖНИХ alpha "
                         "квантилей (нужен вариант cnx_qrdqn); 0 = выкл")
    ap.add_argument("--n-step", type=int, default=1,
                    help="n-шаговые возвраты (Rainbow): при gamma=0.9 и цепочках "
                         "по 12 шагов кредит доходит до начала цепочки быстрее")
    ap.add_argument("--per", action="store_true",
                    help="приоритизированная выборка по TD-ошибке (была у авторов "
                         "в PrioritizedReplayBuffer, в нашем LocalBuffer потерялась)")
    ap.add_argument("--rlpd-max", type=int, default=0,
                    help="ограничить офлайновый буфер N переходами (целыми эпизодами, "
                         "детерминированный сабсэмпл); 0 = весь буфер")
    ap.add_argument("--aug", action="store_true",
                    help="DrQ-стиль: диэдральные аугментации карт в обучающих батчах. "
                         "Для наших карт это точные инварианты — оси латента AE "
                         "произвольны, ориентация не канонична")
    ap.add_argument("--wsrl-warmup", type=int, default=0,
                    help="WSRL (ICLR 2025): тёплый старт из офлайн-агента, первые N "
                         "шагов — сбор данных предобученной политикой без обновлений, "
                         "офлайн-буфер онлайн НЕ удерживается")
    ap.add_argument("--ssl-pretrain", type=int, default=0,
                    help="SGI-lite: N шагов reward-free предобучения энкодера "
                         "(маскированная реконструкция) на картах всех буферов")
    ap.add_argument("--ssl-subdirs",
                    default="poisson3d_complexgeometry,poisson_boltzmann_2d,ns2d_liddriven",
                    help="какие буферы дают карты для SSL (награды/действия не нужны)")
    ap.add_argument("--boot-heads", type=int, default=0,
                    help="bootstrapped DQN: K голов со случайными prior-сетями, одна "
                         "голова на цепочку вместо eps-дрожания (Osband 2016/2018)")
    ap.add_argument("--risk-tau", type=float, default=0.0,
                    help="risk-seeking: действовать по среднему квантилей выше tau "
                         "(нужен вариант cnx_qrdqn); 0 = выкл")
    ap.add_argument("--max-bellman", action="store_true",
                    help="max-reward Bellman (To the Max, ICML 2024): таргет "
                         "max(r, gamma*maxQ') вместо суммы")
    ap.add_argument("--sil", action="store_true",
                    help="self-imitation: доп. лосс к возвратам топ-5 цепочек")
    ap.add_argument("--go-explore", action="store_true",
                    help="Go-Explore: архив лучших срезов цепочек (веса PINN + карта), "
                         "продолжение с лучшего среза вместо старта с нуля")
    ap.add_argument("--preload", default=None,
                    help="схема авторов: предзаполнить ОБЩИЙ буфер прошлым опытом "
                         "(rl_trainer.collect_all_comet_transitions) и дальше "
                         "сэмплировать равномерно, а не половиной батча как RLPD")
    ap.add_argument("--preload-max", type=int, default=500,
                    help="сколько прошлых переходов подгружать (у авторов max_exps_last=500)")
    ap.add_argument("--bcq", action="store_true",
                    help="discrete BCQ: модель поведения по собранным переходам + "
                         "маска поддержки при выборе действия")
    ap.add_argument("--bcq-threshold", type=float, default=0.3,
                    help="порог маски: оставляем действия с p >= threshold * max p")
    ap.add_argument("--display-every", type=int, default=100)
    ap.add_argument("--save-dir", default="runs_rl_train")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.rlpd_full:
        args.rlpd = True        # полному RLPD нужен тот же офлайновый буфер
    if args.smoke:  # только как дефолты — явные флаги не перезаписываем
        given = set(x.split("=")[0] for x in sys.argv[1:] if x.startswith("--"))
        if "--hours" not in given: args.hours = 0.05
        if "--ae-epochs" not in given: args.ae_epochs = 40
        if "--n-save-models" not in given: args.n_save_models = 3
        if "--max-chain-steps" not in given: args.max_chain_steps = 2

    from experiments.chain_eval.pde_registry import build_get_model
    from src.utils.callbacks import TesterCallback, ModelSaverCallback
    from landscape_visualization._aux.visualization_model import VisualizationModel
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    from landscape_visualization._aux.early_stopping_plot import EarlyStopping

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    get_model = build_get_model(args.pde, args.hidden_layers, inverse_plain_fnn=args.plain_fnn)
    get_model_rec = build_get_model(args.pde, args.hidden_layers, inverse_plain_fnn=args.plain_fnn)

    if args.boot_heads:
        from advanced_agents import BootQNet
        net = BootQNet(dev, n_heads=args.boot_heads)
        print(f"bootstrapped DQN: {args.boot_heads} голов с prior-сетями, "
              f"eps-дрожание отключено", flush=True)
    elif args.rlpd_full:
        from advanced_agents import RlpdCritic
        net = RlpdCritic(dev, n_critics=args.rlpd_ensemble, subset=args.rlpd_subset)
        print(f"RLPD целиком: {args.rlpd_ensemble} критиков с LayerNorm, "
              f"таргет по минимуму из {args.rlpd_subset} случайных, "
              f"{net.n_params()/1e6:.2f} млн параметров", flush=True)
    else:
        net = QNet(args.variant, dev, in_ch=4 + (SCALAR_CH if args.scalar_ctx else 0))
    mean = std = None
    if args.warm_start and not os.path.exists(args.warm_start):
        # допускаем имя файла в HF-датасете (rl_arch/models/...)
        try:
            from huggingface_hub import hf_hub_download
            args.warm_start = hf_hub_download(OUT_REPO, f"rl_arch/models/{args.warm_start}",
                                              repo_type="dataset")
        except Exception as e:
            print(f"тёплый старт: чекпоинт не найден ({e}) — с нуля", flush=True)
    if args.warm_start and os.path.exists(args.warm_start):
        ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        # голова может не совпасть по форме (ансамбль голов, другое число квантилей) —
        # переносим то, что совпадает: энкодер полезен всегда, голова доучится
        cur = net.model.state_dict()
        ok = {k: v for k, v in ck["state_dict"].items()
              if k in cur and cur[k].shape == v.shape}
        skipped = len(ck["state_dict"]) - len(ok)
        net.model.load_state_dict(ok, strict=False)
        net.target.load_state_dict(net.model.state_dict())
        mean, std = ck["mean"], ck["std"]
        print(f"тёплый старт из {args.warm_start}: перенесено {len(ok)} тензоров"
              + (f", пропущено {skipped} (несовпадение формы)" if skipped else ""), flush=True)
    if args.scalar_ctx:
        mean, std = pad_norm(mean, std, 4 + SCALAR_CH)
    q_opt = torch.optim.Adam(net.params(), lr=args.lr)

    if args.ssl_pretrain:
        # SGI-lite: reward-free предобучение энкодера маскированной реконструкцией.
        # Карты есть в изобилии (награды и действия для этого не нужны) — дефицитен
        # только размеченный буфер, поэтому энкодер учим на всех буферах сразу
        from offline_rl import load_episodes, episodes_to_arrays
        parts = []
        for sub in args.ssl_subdirs.split(","):
            sub = sub.strip()
            try:
                od2 = episodes_to_arrays(load_episodes(None, sub), fix_next_state=False)
                parts.append(od2["S"])
                print(f"SSL: {sub}: {len(od2['S'])} карт", flush=True)
            except Exception as e:
                print(f"SSL: {sub} пропущен ({type(e).__name__})", flush=True)
        S_all = np.concatenate(parts, 0)
        m_ = S_all.mean(axis=(0, 2, 3), keepdims=True)
        s_ = S_all.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        dec = torch.nn.Linear(net.enc.out_dim, 4 * 26 * 26).to(dev)
        ssl_opt = torch.optim.Adam(list(net.enc.parameters()) + list(dec.parameters()), lr=1e-3)
        sl = 0.0
        for i in range(args.ssl_pretrain):
            idx = np.random.randint(0, len(S_all), size=256)
            x = torch.as_tensor((S_all[idx] - m_) / s_, dtype=torch.float32, device=dev)
            mask = (torch.rand(x.shape[0], 1, 26, 26, device=dev) > 0.5).float()
            rec = dec(net.enc(x * mask)).view(-1, 4, 26, 26)
            loss = (((rec - x) ** 2) * (1 - mask)).sum() / ((1 - mask).sum() * 4 + 1e-6)
            ssl_opt.zero_grad(); loss.backward(); ssl_opt.step()
            sl = float(loss.detach())
        del dec, S_all
        print(f"SSL: энкодер предобучен ({args.ssl_pretrain} шагов, лосс {sl:.4f}); "
              f"дальше обычное обучение с этой инициализацией", flush=True)

    # discrete BCQ: модель поведения по собранным переходам + маска поддержки на
    # выборе действия. Без неё вариант cnx_bcq в онлайне неотличим от обычного
    # DQN — вся суть метода именно в маске
    behaviour = beh_opt = None
    if args.bcq:
        import torch.nn as nn
        behaviour = nn.Sequential(nn.Linear(4 * 26 * 26, 256), nn.LayerNorm(256),
                                  nn.GELU(), nn.Linear(256, 27)).to(dev)
        beh_opt = torch.optim.Adam(behaviour.parameters(), lr=1e-3)

    vm = VisualizationModel(device=str(dev), path_to_plot_model=None,
                            path_to_trajectories=None, **AE_MODEL_PARAMS)
    buf = LocalBuffer(per=args.per)
    off_buf = None
    if (args.rlpd or args.qwm) and not args.self_prior:
        # при self-prior внешний буфер не нужен вовсе — опорную половину даст
        # собственный замороженный прайор; качать 300МБ незачем
        # офлайновая половина батча: тот же буфер, на котором учатся офлайн-агенты
        from offline_rl import load_episodes, episodes_to_arrays, split_by_episode
        od = episodes_to_arrays(load_episodes(None, args.rlpd_subdir),
                                fix_next_state=(args.rlpd_subdir == "poisson_boltzmann_2d"))
        if args.scalar_ctx:
            c = offline_ctx(od, args.max_chain_steps, args.budget)
            od["S"], od["S2"] = attach_ctx(od["S"], c), attach_ctx(od["S2"], c)
            print(f"офлайновым переходам приписан контекст: {od['S'].shape}", flush=True)
        otr, _ = split_by_episode(od)
        oidx = np.where(otr)[0]
        if args.rlpd_max and len(oidx) > args.rlpd_max:
            # целыми эпизодами и детерминированно — чтобы n-step/эпизодная структура
            # не рвалась и прогон был воспроизводим
            ep_ids = od["EP"][oidx]
            uniq = np.unique(ep_ids)
            np.random.default_rng(123).shuffle(uniq)
            keep, tot = [], 0
            for e in uniq:
                keep.append(e); tot += int((ep_ids == e).sum())
                if tot >= args.rlpd_max: break
            oidx = oidx[np.isin(ep_ids, keep)]
            print(f"офлайн-буфер ограничен: {len(oidx)} переходов "
                  f"({len(keep)} эпизодов) из запрошенных ~{args.rlpd_max}", flush=True)
        om = od["S"][oidx].mean(axis=(0, 2, 3), keepdims=True)
        os_ = od["S"][oidx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        off_buf = LocalBuffer(capacity=len(oidx) + 10, per=args.per)
        if args.n_step > 1:
            # n-шаговые возвраты внутри эпизода: сумма дисконтированных наград и
            # состояние через n шагов; дисконт таргета — gamma**k, где k фактическое
            pos = {int(i): j for j, i in enumerate(oidx)}
            EP, Rr, Dd = od["EP"], od["R"], od["D"]
            for i in oidx:
                G, k, j = 0.0, 0, int(i)
                last = j
                while k < args.n_step and (j in pos) and EP[j] == EP[int(i)]:
                    G += (GAMMA ** k) * float(Rr[j]); last = j; k += 1
                    if float(Dd[j]) > 0: break
                    j += 1
                off_buf.push(((od["S"][int(i)][None] - om) / os_)[0], int(od["A"][int(i)]), G,
                             ((od["S2"][last][None] - om) / os_)[0], float(Dd[last]),
                             gam=GAMMA ** k)
        else:
            for i in oidx:
                off_buf.push(((od["S"][i][None] - om) / os_)[0], int(od["A"][i]), float(od["R"][i]),
                             ((od["S2"][i][None] - om) / os_)[0], float(od["D"][i]))
        if mean is None:
            mean, std = om, os_
        if args.scalar_ctx:
            mean, std = pad_norm(mean, std, 4 + SCALAR_CH)
        print(f"RLPD: офлайновый буфер {len(off_buf)} переходов из {args.rlpd_subdir}, "
              f"UTD={args.rlpd_utd}", flush=True)
    if args.preload:
        # схема авторов: прошлые переходы кладутся в ТОТ ЖЕ буфер, из которого потом
        # идёт равномерная выборка — отличие от RLPD, где офлайн держится отдельно
        # и всегда занимает ровно половину батча
        from offline_rl import load_episodes, episodes_to_arrays, split_by_episode
        od = episodes_to_arrays(load_episodes(None, args.preload),
                                fix_next_state=(args.preload == "poisson_boltzmann_2d"))
        otr, _ = split_by_episode(od)
        oidx = np.where(otr)[0][-args.preload_max:]
        om = od["S"][oidx].mean(axis=(0, 2, 3), keepdims=True)
        os_ = od["S"][oidx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        for i in oidx:
            buf.push(((od["S"][i][None] - om) / os_)[0], int(od["A"][i]), float(od["R"][i]),
                     ((od["S2"][i][None] - om) / os_)[0], float(od["D"][i]))
        if mean is None:
            mean, std = om, os_
        if args.scalar_ctx:
            mean, std = pad_norm(mean, std, 4 + SCALAR_CH)
        print(f"предзагрузка по схеме авторов: {len(buf)} переходов из {args.preload}", flush=True)

    wm = wm_opt = None
    if args.qwm:
        from advanced_agents import QwmWorldModel, wm_update, qwm_select
        wm = QwmWorldModel().to(dev)
        wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        wl = 0.0
        for i in range(args.qwm_pretrain):
            wl = wm_update(wm, wm_opt, [off_buf], 128, dev)
        print(f"QWM: мировая модель предобучена ({args.qwm_pretrain} шагов, "
              f"итоговый лосс {wl:.4f}), поиск по Eq.9 с D=1", flush=True)

    rng = np.random.default_rng(args.seed)
    tag = args.tag or f"{args.variant}_{args.pde}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    prior_buf = None
    if args.self_prior:
        prior_buf = LocalBuffer(capacity=args.self_prior + 10)
        off_buf = prior_buf          # опорная половина батча — свои же ранние переходы
        print(f"RLPD на собственных данных: первые {args.self_prior} переходов станут "
              f"замороженным прайором, внешний буфер не используется", flush=True)

    sil_buf = None
    if args.sil:
        from advanced_agents import SilBuffer
        sil_buf = SilBuffer(top_k=5, gamma=GAMMA)
    archive = []          # Go-Explore: [(dict E, err, w, state, raw, spent, steps)]

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    steps_done = 0
    n_updates_total, last_reset, last_redo = 0, 0, 0
    # SPR: латентная модель поверх энкодера; обучается тем же оптимизатором
    spr = None
    hlg = None
    if args.hl_gauss:
        from advanced_agents import HLGauss
        # границы бинов по разумному диапазону возвратов нашей среды
        # число бинов = число выходов головы на действие, иначе формы не сойдутся
        hlg = HLGauss(v_min=-2.0, v_max=6.0, n_bins=net.nq).to(dev)
        print(f"HL-Gauss: {net.nq} бин(ов) на [-2, 6], обучение кросс-энтропией", flush=True)
    if args.spr or args.bbf:
        k_spr = args.spr or 1
        spr = make_spr(net.enc.out_dim, dev)
        spr_opt = torch.optim.Adam(spr.parameters(), lr=args.lr)
        print(f"SPR: предсказание латента на {k_spr} шаг(ов), вес {args.spr_w}", flush=True)
    # дистилляция: копим (состояния, действия) по цепочкам с их итоговой ошибкой
    chain_pool = []          # [(l2re, [s_norm...], [a...])]
    cur_states, cur_acts = [], []
    best_probe = float("inf")
    init_w = ([q.detach().clone() for q in net.model.parameters()]
              if args.l2_init else None)
    chains = []              # завершённые цепочки: (l2re, шагов, эпох)
    last_partial = None      # оборванная цепочка — отдельно, в итог не идёт

    traj = 0
    while time.time() < deadline:
        traj += 1
        model, loss_weights = get_model()

        def reinit(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.net.apply(reinit)

        state = np.zeros((4, 26, 26), dtype=np.float32)
        if args.scalar_ctx:
            state = add_scalar_ctx(state, 0, args.max_chain_steps, 0, 1, None, 1.0)
        prev_raw = None
        pending = []          # хвост цепочки для n-шаговых возвратов
        spent, chain, done = 0, [], 0
        truncated = False   # True только если ПРЕРВАЛИ цепочку посередине по лимиту
        l2re = float("inf")
        prev_err = None
        start_step, resumed_from = 0, None
        chain_trans = []    # (s_норм, a, r) текущей цепочки — для self-imitation

        if args.boot_heads:
            net.active = int(rng.integers(0, net.K))   # одна голова на всю цепочку

        if args.go_explore and archive and len(chains) >= 2 and rng.random() < 0.6:
            # ранг-взвешенный выбор: лучшие срезы чаще (Go-Explore 'return')
            w = np.array([1.0 / (i + 1) for i in range(len(archive))]); w /= w.sum()
            e = archive[int(rng.choice(len(archive), p=w))]
            model.net.load_state_dict({k: v.to(dev) for k, v in e["w"].items()})
            state = e["state"].copy()
            prev_raw = ({k: v.clone() for k, v in e["raw"].items()}
                        if e["raw"] is not None else None)
            spent, start_step, prev_err = e["spent"], e["steps"], e["err"]
            l2re, resumed_from = e["E"], e["E"]
            print(f"[траектория {traj}] Go-Explore: продолжаю срез l2re={e['E']:.4f} "
                  f"(шаг {e['steps']}, {e['spent']} эпох)", flush=True)

        for step in range(start_step, args.max_chain_steps):
            if time.time() >= deadline:
                truncated = True
                last_partial = dict(l2re=l2re, steps=len(chain), epochs=spent, trajectory=traj)
                print(f"[траектория {traj}] лимит времени — цепочка оборвана на шаге {len(chain)}", flush=True)
                break

            # eps-greedy как у авторов
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
            steps_done += 1
            if args.greedy_probe and (traj % args.greedy_probe == 0):
                eps = 0.0        # проба качества политики: метрика отбора = метрика деплоя
            in_warmup = args.wsrl_warmup and steps_done <= args.wsrl_warmup
            if rng.random() < eps and not args.boot_heads and not in_warmup:
                a = int(rng.integers(0, 27))
                how = "случайно"
            else:
                x = torch.as_tensor(((state[None] - mean) / std) if mean is not None else state[None],
                                    device=dev).float()
                if wm is not None and args.search_depth > 1:
                    a = deep_search(net, wm, x, gamma=GAMMA, alpha=args.qwm_alpha,
                                    depth=args.search_depth, beam=args.search_beam)
                    how = f"поиск D={args.search_depth}"
                elif wm is not None:
                    # QWM: одношаговый поиск вместо argmax Q (Eq. 9, D=1)
                    a = qwm_select(net, wm, x, gamma=GAMMA, alpha=args.qwm_alpha)
                    how = "поиск QWM"
                elif args.boot_heads:
                    with torch.no_grad():
                        a = int(net.q_head(x, net.active).argmax(1).item())
                    how = f"голова {net.active}"
                elif args.risk_tau > 0:
                    from advanced_agents import q_upper
                    with torch.no_grad():
                        a = int(q_upper(net, x, args.risk_tau).argmax(1).item())
                    how = f"квантиль>{args.risk_tau}"
                elif args.cvar_alpha > 0:
                    with torch.no_grad():
                        a = int(net.q_cvar(x, args.cvar_alpha).argmax(1).item())
                    how = f"CVaR@{args.cvar_alpha}"
                else:
                  with torch.no_grad():
                    # под HL-Gauss выходы головы — логиты по бинам; скаляр Q это
                    # взвешенная сумма центров, а не среднее (как у квантилей)
                    q = hlg.to_scalar(net.q_online(x)) if hlg is not None else net.q_scalar(x)
                    if behaviour is not None:
                        # BCQ: выбираем только среди действий, которые модель
                        # поведения считает правдоподобными для этих данных
                        p = behaviour(x.flatten(1)).softmax(-1)
                        keep = p >= args.bcq_threshold * p.max(-1, keepdim=True).values
                        q = q.masked_fill(~keep, -1e9)
                    a = int(q.argmax(1).item())
                  how = "по модели"
            opt_name, lr, epochs = ACTION_TABLE[a]

            optimizer = build_optimizer(opt_name, lr, model.net)
            model.compile(optimizer, loss_weights=loss_weights)
            tester = TesterCallback(log_every=args.display_every)
            saver = ModelSaverCallback(total_iterations=epochs, n_save_models=args.n_save_models)
            model.train(iterations=epochs, display_every=args.display_every,
                        callbacks=[tester, saver], model_save_path=save_dir, save_model=False)
            spent += epochs
            chain.append([opt_name, lr, epochs])

            rmse = float(getattr(tester, "rmse", float("inf")))
            brmse = float(getattr(tester, "brmse", float("inf")))
            l2re = math.hypot(float(getattr(tester, "l2re", float("inf"))),
                              float(getattr(tester, "bc_l2re", float("inf"))))
            if not (np.isfinite(rmse) or np.isfinite(brmse)):
                done = -1
                print(f"[траектория {traj}] метрики разошлись — обрыв", flush=True)
                break

            # награда авторов: absolute, E = rmse + brmse; r = E_t - E_{t+1}
            err = (rmse if np.isfinite(rmse) else 0.0) + (brmse if np.isfinite(brmse) else 0.0)
            reward = 0.0 if prev_err is None else (prev_err - err)
            err_before = prev_err          # для PBRS: Phi(s) считается ДО обновления
            prev_err = err
            done = 1 if (args.tolerance > 0 and err < args.tolerance) else 0

            # следующее состояние — их пайплайном
            ae = vm.train(5e-4, 1200, args.ae_epochs, 100, args.batch_size, True,
                          finetune_AE_model=False, callbacks=[EarlyStopping(patience=4000)],
                          solver_models=saver.saved_models)
            pls = PlotLossSurface(solver_models=saver.saved_models, AE_model=ae,
                                  dde_pde_model=get_model_rec, x_range=GRID_RANGE,
                                  batch_size=args.batch_size, loss_types=LOSS_TYPES,
                                  loss_name="loss_total", path_to_plot_model=None,
                                  path_to_trajectories=None, img_dir="")
            raw = pls.save_equation_loss_surface(log_key=True)
            next_state = build_state(raw, prev_raw)
            if args.scalar_ctx:
                next_state = add_scalar_ctx(next_state, len(chain), args.max_chain_steps,
                                            spent, args.budget if hasattr(args, "budget") else 31000,
                                            a, err)
            prev_raw = raw
            del pls, ae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if mean is None:      # нормировка по первому состоянию (агент с нуля)
                mean = next_state.mean(axis=(1, 2), keepdims=True)[None]
                std = next_state.std(axis=(1, 2), keepdims=True)[None] + 1e-6
                if args.scalar_ctx:
                    # контекстные каналы постоянны, их пространственный разброс равен
                    # нулю — деление на 1e-6 раздуло бы вход в миллион раз. Они уже
                    # лежат в [-1,1], поэтому нормировка им не нужна вовсе.
                    mean[:, 4:] = 0.0
                    std[:, 4:] = 1.0
            s_norm = ((state[None] - mean) / std)[0]
            s2_norm = ((next_state[None] - mean) / std)[0]
            if prior_buf is not None and len(prior_buf) < args.self_prior:
                prior_buf.push(s_norm, a, reward, s2_norm, float(done == 1))
            if args.n_step > 1:
                pending.append((s_norm, a, float(reward)))
                if len(pending) >= args.n_step:
                    s0, a0, _ = pending[0]
                    G = sum((GAMMA ** k) * tr[2] for k, tr in enumerate(pending))
                    buf.push(s0, a0, G, s2_norm, float(done == 1),
                             gam=GAMMA ** len(pending))
                    pending.pop(0)
            else:
                r_eff = reward
                if args.pbrs:
                    # F = gamma*Phi(s') - Phi(s), Phi = -log10(ошибка): политико-инвариантно
                    phi = lambda e: -math.log10(max(float(e), 1e-8))
                    if err_before is not None:
                        r_eff = reward + args.pbrs * (GAMMA * phi(err) - phi(err_before))
                g_eff = (GAMMA ** (epochs / args.smdp_scale)) if args.smdp else GAMMA
                buf.push(s_norm, a, r_eff, s2_norm, float(done == 1), gam=g_eff)
            if args.sil:
                chain_trans.append((s_norm.copy(), a, float(reward)))
            if args.distill:
                cur_states.append(s_norm.copy()); cur_acts.append(a)

            if args.bbf:
                # BBF: дисконт отжигается 0.97 -> 0.997, горизонт n-step 10 -> 3
                GAMMA = anneal(steps_done, 200, 0.97, 0.997)
            if spr is not None and len(buf) >= 8:
                # SPR: награда не нужна, поэтому учится и на сорванных цепочках
                ss, aa, _, ss2, _, _ = buf.sample(min(args.batch_size, len(buf)), dev)
                ls = args.spr_w * spr_loss(net, spr, ss, aa, ss2)
                q_opt.zero_grad(); spr_opt.zero_grad()
                ls.backward()
                q_opt.step(); spr_opt.step()

            if args.distill and chain_pool:
                # кросс-энтропия к действиям верхних цепочек прогона
                top = sorted(chain_pool, key=lambda t: t[0])[:args.distill_top]
                XS = np.concatenate([np.stack(t[1]) for t in top], 0)
                XA = np.concatenate([np.array(t[2]) for t in top], 0)
                if len(XS) >= 4:
                    k = min(args.batch_size, len(XS))
                    idx = np.random.choice(len(XS), k, replace=False)
                    Sx = torch.as_tensor(XS[idx], device=dev).float()
                    Ax = torch.as_tensor(XA[idx], device=dev).long()
                    ld = args.distill_w * distill_loss(net, Sx, Ax)
                    q_opt.zero_grad(); ld.backward(); q_opt.step()
            state = next_state

            if args.go_explore and len(chain) + start_step < args.max_chain_steps:
                # срез: веса PINN + карта + сырые лоссы — чтобы можно было вернуться
                entry = dict(E=l2re, err=prev_err, spent=spent,
                             steps=start_step + len(chain), state=next_state.copy(),
                             raw={k: v.detach().cpu().clone() for k, v in prev_raw.items()},
                             w={k: v.detach().cpu().clone()
                                for k, v in model.net.state_dict().items()})
                if len(archive) < 8 or l2re < archive[-1]["E"]:
                    archive.append(entry)
                    archive.sort(key=lambda x: x["E"])
                    del archive[8:]

            if in_warmup:
                # WSRL: фаза прогрева — предобученная политика собирает данные,
                # обновлений нет (рекалибровка Q начнётся на свежем буфере)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} "
                      f"(прогрев WSRL {steps_done}/{args.wsrl_warmup}) l2re={l2re:.4e}", flush=True)
                if done == 1:
                    break
                continue

            if behaviour is not None and len(buf) >= 8:
                behaviour_update(behaviour, beh_opt, buf, args.batch_size, 2, dev)

            if wm is not None:
                # дообучение мировой модели на свежих переходах (критик как учился
                # на реальных данных, так и учится — модель мира его не подменяет)
                wbufs = [off_buf] + ([buf] if len(buf) >= 8 else [])
                for _ in range(4):
                    wm_update(wm, wm_opt, wbufs, 64, dev)

            if prior_buf is not None and len(prior_buf) < args.self_prior:
                # прайор ещё набирается — обновлений нет, это фаза сбора
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} "
                      f"(набор прайора {len(prior_buf)}/{args.self_prior}) l2re={l2re:.4e}",
                      flush=True)
                if done == 1:
                    break
                continue

            if args.redo_every or args.reset_every:
                n_updates_total += (args.rlpd_utd if (args.rlpd and off_buf is not None)
                                    else args.update_iters)
            if args.redo_every:
                if n_updates_total - last_redo >= args.redo_every and len(buf) >= 8:
                    k = redo_recycle(net, buf, args.batch_size, dev, args.redo_thresh)
                    last_redo = n_updates_total
                    print(f"[ReDo] переработано спящих нейронов: {k}, "
                          f"обновлений {n_updates_total}", flush=True)

            if args.reset_every:
                if n_updates_total - last_reset >= args.reset_every:
                    k = shrink_and_perturb(net, args.reset_alpha, dev)
                    q_opt = torch.optim.Adam(net.params(), lr=args.lr)   # состояние Adam тоже сбрасываем
                    last_reset = n_updates_total
                    print(f"[сброс] SR-SPR: голова заново, энкодер сжат до "
                          f"{args.reset_alpha}, тензоров {k}, обновлений {n_updates_total}",
                          flush=True)

            if args.rlpd and off_buf is not None:
                # RLPD: половина батча из офлайна, половина из онлайна; много обновлений
                iters = args.rlpd_utd
                half = max(1, args.batch_size // 2)
                ql = 0.0
                for _ in range(iters):
                    src = buf if len(buf) >= 8 else off_buf
                    if args.boot_heads:
                        from advanced_agents import boot_rlpd_update
                        ql = boot_rlpd_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif args.risk_tau > 0 or args.cvar_alpha > 0:
                        from advanced_agents import rlpd_qr_update
                        ql = rlpd_qr_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif args.rlpd_full:
                        from advanced_agents import rlpd_update
                        ql = rlpd_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif hlg is not None:
                        src2 = buf if len(buf) >= 8 else off_buf
                        ql = hlg_update_mixed(net, off_buf, src2, q_opt, half, hlg)
                    elif len(buf) >= 8:
                        ql = agent_update_mixed(net, off_buf, buf, q_opt, half, args.variant,
                                                max_bellman=args.max_bellman, aug=args.aug,
                                                munch=args.munchausen, m_tau=args.m_tau,
                                                m_alpha=args.m_alpha, l2_init=args.l2_init,
                                                init_w=init_w)
                    else:
                        ql = agent_update(net, off_buf, q_opt, args.batch_size, 1, args.variant,
                                          munch=args.munchausen, m_tau=args.m_tau,
                                          m_alpha=args.m_alpha, l2_init=args.l2_init,
                                          init_w=init_w)
                if args.sil and sil_buf is not None and sil_buf.chains:
                    from advanced_agents import sil_update
                    sil_update(net, sil_buf, q_opt, args.batch_size, dev)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}, eps={eps:.2f}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)}+{len(off_buf)} q-loss={ql:.4f}", flush=True)
            elif hlg is not None and len(buf) >= args.min_buffer:
                ql = hlg_update(net, buf, q_opt, args.batch_size, args.update_iters, hlg,
                                munch=args.munchausen, m_tau=args.m_tau, m_alpha=args.m_alpha)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} "
                      f"({how}, eps={eps:.2f}) l2re={l2re:.4e} reward={reward:+.4f} "
                      f"буфер={len(buf)} hlg-loss={ql:.4f}", flush=True)
            elif len(buf) >= args.min_buffer:
                ql = agent_update(net, buf, q_opt, args.batch_size, args.update_iters, args.variant,
                                  munch=args.munchausen, m_tau=args.m_tau,
                                  m_alpha=args.m_alpha, l2_init=args.l2_init, init_w=init_w)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}, eps={eps:.2f}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)} q-loss={ql:.4f}", flush=True)
            else:
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)}", flush=True)
            if done == 1:
                break
        else:
            done = done or 0

        # цепочка засчитывается, если дошла до конца (K_max / допуск / расхождение),
        # даже если лимит времени истёк во время последнего шага; не засчитывается
        # только та, которую прервали ПОСЕРЕДИНЕ
        if args.n_step > 1 and pending:
            s2_last = ((state[None] - mean) / std)[0]
            while pending:
                s0, a0, _ = pending[0]
                G = sum((GAMMA ** k) * tr[2] for k, tr in enumerate(pending))
                buf.push(s0, a0, G, s2_last, 1.0, gam=GAMMA ** len(pending))
                pending.pop(0)

        if chain and not truncated:
            if args.sil and sil_buf is not None and np.isfinite(l2re):
                sil_buf.add_chain(l2re, chain_trans)
            rec = dict(l2re=l2re, steps=len(chain), epochs=spent, done=done, chain=chain)
            if resumed_from is not None:
                rec["resumed_from"] = resumed_from
            chains.append(rec)
            if args.distill and cur_states:
                # цепочка попадает в пул экспертных вместе со своей итоговой ошибкой
                chain_pool.append((float(rec["l2re"]), cur_states, cur_acts))
                chain_pool.sort(key=lambda t: t[0])
                del chain_pool[max(args.distill_top * 2, 6):]
            cur_states, cur_acts = [], []
            if args.greedy_probe and (traj % args.greedy_probe == 0):
                # отбор чекпоинта по метрике деплоя, а не по расписанию
                if float(rec["l2re"]) < best_probe:
                    best_probe = float(rec["l2re"])
                    if args.save_agent:
                        save_agent(net, mean, std, args.variant,
                                   args.tag + "_bestprobe", len(chains), best_probe,
                                   hl_gauss=_hlg_meta(hlg))
                    print(f"[проба] новая лучшая жадная цепочка {best_probe:.4f} — "
                          f"чекпоинт сохранён", flush=True)
            row = dict(variant=args.variant, pde=args.pde, seed=args.seed,
                       trajectories=len(chains), buffer=len(buf),
                       l2re_last_complete=chains[-1]["l2re"],
                       l2re_best=min(c["l2re"] for c in chains),
                       l2re_first=chains[0]["l2re"],
                       chains=[{k: v for k, v in c.items() if k != "chain"} for c in chains],
                       last_chain=chains[-1]["chain"], last_partial=last_partial,
                       elapsed_h=round((time.time() - t_start) / 3600, 2))
            print(json.dumps({k: v for k, v in row.items() if k not in ("chains", "last_chain")}), flush=True)
            if not args.smoke:
                upload(row, tag)
            if args.save_agent and (len(chains) % args.save_every == 0):
                save_agent(net, mean, std, args.variant, tag, len(chains),
                           min(c["l2re"] for c in chains), hl_gauss=_hlg_meta(hlg))

    if args.save_agent and chains:
        save_agent(net, mean, std, args.variant, tag, len(chains),
                   min(c["l2re"] for c in chains), hl_gauss=_hlg_meta(hlg))
        print(f"агент сохранён: agent_{tag}.pt и rl_arch/agents_online/{tag}.pt", flush=True)

    print(f"\nИТОГ: завершённых цепочек {len(chains)}, "
          f"l2re последней завершённой = {chains[-1]['l2re']:.4e}" if chains else "\nИТОГ: ни одной завершённой цепочки",
          flush=True)


if __name__ == "__main__":
    main()
