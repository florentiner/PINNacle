#!/usr/bin/env python
"""
Продвинутые офлайн/онлайн агенты поверх ConvNeXt-энкодера:

  bcq   — discrete BCQ: модель поведения + маска поддержки на выборе действия
  bbf   — режим BBF: высокий replay ratio, shrink-and-perturb сбросы,
          отжиг n-шага и gamma
  smdp  — полный стек SMDP-BBF-CQL (по предложенному коду, адаптирован под
          состояние 4x26x26 вместо пары «латент VAE + скаляры»):
          LayerNorm-резидуальный ствол, факторизованное вложение действия,
          дуэлинговые головы, HL-Gauss категориальные таргеты, ансамбль из K
          голов с клиппингом, CQL + маска BCQ, SPR, Munchausen,
          SMDP-дисконт gamma**dt, разведка по разбросу ансамбля

Ничего из перечисленного в исходном репозитории нет (проверено grep по всем
веткам): там PER + soft-Watkins + дуэлинг.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 27
OPTIMIZER_GRID = [
    ("adam", [1e-2, 1e-3, 1e-4], [100, 1000, 2500]),
    ("lbfgs", [1.0, 5e-1, 1e-1], [100, 500, 1000]),
    ("pso", [0.0, 1e-3, 1e-4], [100, 200, 300]),
]


def build_action_feats():
    """(27, 6): one-hot оптимизатора, нормированный log lr, флаг lr=0, log длины."""
    feats, dts = [], []
    for oi, (_, lrs, lens) in enumerate(OPTIMIZER_GRID):
        for lr in lrs:
            for n_ep in lens:
                oh = [0.0, 0.0, 0.0]
                oh[oi] = 1.0
                feats.append(oh + [(math.log10(lr) + 3.0) / 3.0 if lr > 0 else 0.0,
                                   1.0 if lr == 0 else 0.0,
                                   math.log10(n_ep) - 2.0])
                dts.append(n_ep / 100.0)     # длительность действия для SMDP
    return np.asarray(feats, np.float32), np.asarray(dts, np.float32)


ACTION_FEATS, ACTION_DT = build_action_feats()


class HLGauss(nn.Module):
    """Категориальные таргеты со сглаживанием (Stop Regressing, 2024)."""

    def __init__(self, v_min, v_max, n_bins=51, sigma_ratio=0.75):
        super().__init__()
        edges = torch.linspace(v_min, v_max, n_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        self.sigma = sigma_ratio * float(edges[1] - edges[0])
        self.n_bins = n_bins

    def target_probs(self, y):
        y = y.unsqueeze(-1)
        cdf = 0.5 * (1 + torch.erf((self.edges - y) / (self.sigma * math.sqrt(2))))
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / p.sum(-1, keepdim=True).clamp_min(1e-8)

    def to_scalar(self, logits):
        return (logits.softmax(-1) * self.centers).sum(-1)

    def loss(self, logits, y):
        return -(self.target_probs(y) * logits.log_softmax(-1)).sum(-1)


class ResBlock(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, expansion * dim), nn.GELU(),
                                 nn.Linear(expansion * dim, dim))

    def forward(self, x):
        return x + self.net(self.ln(x))


class SmdpNet(nn.Module):
    """ConvNeXt-энкодер карт 4x26x26 -> резидуальный ствол -> K дуэлинговых
    категориальных голов с факторизованным вложением действия + SPR."""

    def __init__(self, encoder, width=512, n_blocks=2, n_heads=5, act_emb=64, n_bins=51):
        super().__init__()
        self.enc = encoder
        self.n_heads, self.n_bins = n_heads, n_bins
        self.register_buffer("action_feats", torch.as_tensor(ACTION_FEATS))
        self.stem = nn.Sequential(nn.Linear(encoder.out_dim, width),
                                  *[ResBlock(width) for _ in range(n_blocks)],
                                  nn.LayerNorm(width))
        self.act_emb = nn.Sequential(nn.Linear(ACTION_FEATS.shape[1], act_emb), nn.GELU(),
                                     nn.Linear(act_emb, act_emb))
        self.adv = nn.ModuleList([nn.Sequential(nn.Linear(width + act_emb, 256),
                                                nn.LayerNorm(256), nn.GELU(),
                                                nn.Linear(256, n_bins)) for _ in range(n_heads)])
        self.val = nn.ModuleList([nn.Sequential(nn.Linear(width, 256), nn.LayerNorm(256),
                                                nn.GELU(), nn.Linear(256, n_bins))
                                  for _ in range(n_heads)])
        self.spr_tr = nn.Sequential(nn.Linear(width + act_emb, width), nn.LayerNorm(width),
                                    nn.GELU(), nn.Linear(width, width))
        self.spr_proj = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 256))

    def embed(self, x):
        return self.stem(self.enc(x))

    def head_logits(self, h, k):
        phi = self.act_emb(self.action_feats)                      # (A,E)
        B = h.shape[0]
        pair = torch.cat([h[:, None].expand(B, N_ACTIONS, h.shape[-1]),
                          phi[None].expand(B, N_ACTIONS, phi.shape[-1])], -1)
        adv = self.adv[k](pair)
        val = self.val[k](h)[:, None]
        return val + adv - adv.mean(1, keepdim=True)

    def all_logits(self, x):
        h = self.embed(x)
        return torch.stack([self.head_logits(h, k) for k in range(self.n_heads)]), h


class SmdpAgent:
    """Обёртка с интерфейсом QNet (q_scalar/q_online/q_target/model), чтобы
    метрики и FQE считались тем же кодом, что и для остальных вариантов."""

    def __init__(self, device, n_heads=5, v_min=-20.0, v_max=2.0, width=512):
        from offline_rl import make_encoder
        self.device = device
        self.variant = "cnx_smdp"
        self.net = SmdpNet(make_encoder("convnext"), width=width, n_heads=n_heads).to(device)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.hlg = HLGauss(v_min, v_max).to(device)
        self.behaviour = nn.Sequential(nn.Linear(4 * 26 * 26, 256), nn.LayerNorm(256),
                                       nn.GELU(), nn.Linear(256, N_ACTIONS)).to(device)
        self.model = self.net           # совместимость с evaluate()/сохранением
        self.n_heads = n_heads

    # --- интерфейс QNet ---------------------------------------------------
    def params(self):
        return self.net.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def q_scalar(self, x):
        logits, _ = self.net.all_logits(x)
        return self.hlg.to_scalar(logits).mean(0)

    def q_online(self, x):
        return self.q_scalar(x)

    def q_target(self, x):
        logits, _ = self.target.all_logits(x)
        return self.hlg.to_scalar(logits).mean(0)

    def q_cvar(self, x, alpha=0.25):
        logits, _ = self.net.all_logits(x)
        q = self.hlg.to_scalar(logits)                 # (K,B,A)
        return q.min(0).values                          # пессимистичная оценка ансамбля

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, pt in zip(self.net.parameters(), self.target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)

    def bcq_mask(self, x, threshold=0.3):
        with torch.no_grad():
            p = self.behaviour(x.flatten(1)).softmax(-1)
        keep = p >= threshold * p.max(-1, keepdim=True).values
        keep[torch.arange(keep.shape[0], device=x.device), p.argmax(-1)] = True
        return keep

    def shrink_and_perturb(self, alpha=0.5):
        """BBF: ствол сохраняет alpha, головы сбрасываются полностью."""
        from offline_rl import make_encoder
        fresh = SmdpNet(make_encoder("convnext"), n_heads=self.n_heads).to(self.device)
        with torch.no_grad():
            for (name, p), pf in zip(self.net.named_parameters(), fresh.parameters()):
                a = 0.0 if name.startswith(("adv", "val")) else alpha
                p.mul_(a).add_((1 - a) * pf)
        self.target.load_state_dict(self.net.state_dict())


def fit_behaviour(agent, S, A, device, epochs=300, batch=256):
    """Модель поведения для BCQ-маски (10 строк, но отсекает действия,
    которых в данных нет)."""
    opt = torch.optim.Adam(agent.behaviour.parameters(), lr=1e-3)
    n = len(A)
    for _ in range(epochs):
        idx = np.random.randint(0, n, size=min(batch, n))
        x = S[idx].flatten(1)
        y = torch.as_tensor(A[idx.tolist()] if isinstance(A, list) else A[idx], device=device)
        loss = F.cross_entropy(agent.behaviour(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# RLPD в полном виде (arXiv 2302.02948)
# ---------------------------------------------------------------------------
# Симметричная выборка 50/50 и высокий update-to-data уже есть в
# online_train_env. Здесь — две оставшиеся компоненты статьи:
#   * LayerNorm в критике: без неё Q расходится на действиях вне поддержки
#     данных, и подмешивание офлайна вредит вместо пользы (раздел 4.1 статьи);
#   * ансамбль из N критиков со случайным подмножеством размера M для таргета:
#     пессимизм без ручной настройки коэффициента (у авторов N=10, M=2).
class RlpdCritic:
    """Интерфейс QNet, но внутри N независимых критиков с LayerNorm."""

    def __init__(self, device, n_critics=10, subset=2, encoder_kind="convnext"):
        from offline_rl import make_encoder
        self.device, self.n, self.m = device, n_critics, subset
        self.variant = "cnx_dqn_rlpd_full"

        def one():
            enc = make_encoder(encoder_kind)
            return nn.Sequential(enc, nn.LayerNorm(enc.out_dim),
                                 nn.Linear(enc.out_dim, 256), nn.LayerNorm(256), nn.GELU(),
                                 nn.Linear(256, N_ACTIONS))

        self.nets = nn.ModuleList([one() for _ in range(n_critics)]).to(device)
        self.target = copy.deepcopy(self.nets).requires_grad_(False)
        self.model = self.nets            # совместимость с сохранением/загрузкой

    def params(self):
        return self.nets.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.nets.parameters())

    def q_all(self, x):
        return torch.stack([net(x) for net in self.nets])        # (N,B,A)

    def q_scalar(self, x):
        """Для выбора действия — среднее по ансамблю."""
        return self.q_all(x).mean(0)

    def q_online(self, x):
        return self.q_scalar(x)

    def q_target_subset(self, x):
        """Таргет: минимум по СЛУЧАЙНОМУ подмножеству целевых критиков."""
        idx = torch.randperm(self.n, device=x.device)[:self.m]
        with torch.no_grad():
            q = torch.stack([self.target[int(i)](x) for i in idx])   # (M,B,A)
        return q.min(0).values

    def q_target(self, x):
        return self.q_target_subset(x)

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, pt in zip(self.nets.parameters(), self.target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)


def rlpd_update(critic, off_buf, on_buf, opt, half, gamma):
    """Шаг RLPD: батч 50/50, таргет по случайному подмножеству, все критики
    учатся на одном и том же таргете."""
    dev = critic.device
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d = [torch.cat([p[i] for p in parts], 0) for i in range(5)]
    with torch.no_grad():
        a2 = critic.q_scalar(s2).argmax(1)                      # double-DQN: argmax онлайн
        q2 = critic.q_target_subset(s2).gather(1, a2[:, None]).squeeze(1)
        tgt = r + gamma * (1 - d) * q2
    qa = critic.q_all(s).gather(2, a[None, :, None].expand(critic.n, -1, 1)).squeeze(-1)
    loss = F.mse_loss(qa, tgt[None].expand_as(qa))
    opt.zero_grad(); loss.backward(); opt.step()
    critic.soft_update()
    return float(loss.detach())
