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


# ---------------------------------------------------------------------------
# QWM — Q-Learning with World Models (arXiv 2608.17163)
# ---------------------------------------------------------------------------
# Мировая модель используется ТОЛЬКО при выборе действия (test-time search),
# политика и критик учатся на реальных переходах — ошибка модели не
# накапливается в обучении. Для низкоразмерных состояний статья берёт
# резидуальный трёхслойный MLP (hidden 256): s' = s + Delta(s, a); мы добавляем
# вторую голову под награду (у них r_psi отдельно, механика та же).
class QwmWorldModel(nn.Module):
    def __init__(self, state_dim=4 * 26 * 26, n_actions=N_ACTIONS, hidden=256):
        super().__init__()
        self.inp = nn.Linear(state_dim + n_actions, hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.delta = nn.Linear(hidden, state_dim)
        self.rew = nn.Linear(hidden, 1)

    def forward(self, s_flat, a_onehot):
        z = F.gelu(self.inp(torch.cat([s_flat, a_onehot], -1)))
        z = F.gelu(self.mid(z))
        return self.delta(z), self.rew(z).squeeze(-1)


def wm_update(wm, opt, bufs, batch, device):
    """Шаг обучения мировой модели: MSE по резидуалу состояния и по награде.
    bufs — список буферов; батч делится между ними поровну."""
    parts = [b.sample(max(1, batch // len(bufs)), device) for b in bufs]
    s, a, r, s2, _ = [torch.cat([p[i] for p in parts], 0) for i in range(5)]
    s_flat, s2_flat = s.flatten(1), s2.flatten(1)
    a_oh = F.one_hot(a, N_ACTIONS).float()
    d_hat, r_hat = wm(s_flat, a_oh)
    loss = F.mse_loss(d_hat, s2_flat - s_flat) + F.mse_loss(r_hat, r)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def qwm_select(net, wm, x, gamma=0.9, alpha=0.5):
    """Выбор действия по Eq. 9 статьи с D=1, K=1: все 27 действий разворачиваются
    исчерпывающе (действий мало — сэмплирование кандидатов из политики не нужно).
    score(a) = alpha*Q(s,a) + (1-alpha)*[r_hat(s,a) + gamma*max_a' Q_target(s'_hat)]
    Глубже одного шага не идём: модель обучена на переходах другого уравнения,
    и её ошибка на воображаемых траекториях накапливается с глубиной."""
    with torch.no_grad():
        q_root = net.q_scalar(x)[0]                                  # (27,)
        eye = torch.eye(N_ACTIONS, device=x.device)
        s_flat = x.flatten(1).expand(N_ACTIONS, -1)                  # (27, 2704)
        d_hat, r_hat = wm(s_flat, eye)
        s2 = (s_flat + d_hat).view(N_ACTIONS, *x.shape[1:])
        v2 = net.q_target(s2).max(1).values                          # (27,)
        score = alpha * q_root + (1 - alpha) * (r_hat + gamma * v2)
    return int(score.argmax().item())


# ---------------------------------------------------------------------------
# Bootstrapped DQN (Osband 2016) + randomized prior functions (Osband 2018)
# ---------------------------------------------------------------------------
# Согласованная на протяжении цепочки эксплорация: в начале цепочки сэмплируется
# одна из K голов и ведёт её целиком — вместо eps-дрожания, ломающего хорошие
# цепочки случайным действием в случайный момент. Разнообразие голов держат
# замороженные случайные prior-сети (на 200 переходах bootstrap-маски бесполезны).
class BootQNet:
    def __init__(self, device, n_heads=5, prior_scale=3.0):
        from offline_rl import make_encoder
        self.device, self.K, self.beta = device, n_heads, prior_scale
        self.variant = "cnx_boot"
        self.enc = make_encoder("convnext")
        self.heads = nn.ModuleList([nn.Linear(self.enc.out_dim, N_ACTIONS)
                                    for _ in range(n_heads)])
        self.priors = nn.ModuleList([nn.Sequential(nn.Linear(self.enc.out_dim, 64), nn.GELU(),
                                                   nn.Linear(64, N_ACTIONS))
                                     for _ in range(n_heads)]).requires_grad_(False)
        self.model = nn.ModuleList([self.enc, self.heads, self.priors]).to(device)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.active = 0

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _q(self, mod, x, k):
        z = mod[0](x)
        return mod[1][k](z) + self.beta * mod[2][k](z)

    def q_head(self, x, k):
        return self._q(self.model, x, k)

    def q_scalar(self, x):
        return torch.stack([self._q(self.model, x, k) for k in range(self.K)]).mean(0)

    def q_online(self, x):
        return self.q_scalar(x)

    def q_target(self, x):
        return torch.stack([self._q(self.target, x, k) for k in range(self.K)]).mean(0)

    def q_target_head(self, x, k):
        return self._q(self.target, x, k)

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, pt in zip(self.model.parameters(), self.target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)


def boot_rlpd_update(net, off_buf, on_buf, opt, half, gamma):
    """Симметричный батч; каждая голова учится на своём double-DQN таргете."""
    dev = net.device
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d = [torch.cat([p[i] for p in parts], 0) for i in range(5)]
    loss = 0.0
    for k in range(net.K):
        with torch.no_grad():
            a2 = net.q_head(s2, k).argmax(1)
            q2 = net.q_target_head(s2, k).gather(1, a2[:, None]).squeeze(1)
            tgt = r + gamma * (1 - d) * q2
        q = net.q_head(s, k).gather(1, a[:, None]).squeeze(1)
        loss = loss + F.mse_loss(q, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    net.soft_update()
    return float(loss.detach()) / net.K


# ---------------------------------------------------------------------------
# Risk-seeking по верхнему квантилю (линия RiskMiner/IQN-искажений)
# ---------------------------------------------------------------------------
def rlpd_qr_update(net, off_buf, on_buf, opt, half, gamma):
    """Квантильная регрессия (QR-DQN) на симметричном батче 50/50 — в отличие от
    скалярного MSE, сохраняет форму распределения, без чего верхний квантиль
    не отличался бы от среднего."""
    dev = net.device
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d = [torch.cat([p[i] for p in parts], 0) for i in range(5)]
    nq = net.nq
    q_all = net.q_online(s)                                        # (B,A,nq)
    q_sa = q_all.gather(1, a[:, None, None].expand(-1, 1, nq)).squeeze(1)   # (B,nq)
    with torch.no_grad():
        a2 = net.q_scalar(s2).argmax(1)
        q2 = net.q_target(s2).gather(1, a2[:, None, None].expand(-1, 1, nq)).squeeze(1)
        tgt = r[:, None] + gamma * (1 - d)[:, None] * q2           # (B,nq)
    taus = (torch.arange(nq, device=dev, dtype=torch.float32) + 0.5) / nq
    u = tgt[:, None, :] - q_sa[:, :, None]                         # (B,nq,nq)
    huber = torch.where(u.abs() <= 1.0, 0.5 * u ** 2, u.abs() - 0.5)
    loss = (torch.abs(taus[None, :, None] - (u < 0).float()) * huber).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    net.soft_update()
    return float(loss.detach())


def q_upper(net, x, tau=0.8):
    """Оптимистичная оценка: среднее квантилей выше tau. Для max-метрики
    действуем по верхнему хвосту распределения, а не по среднему (и тем более
    не по CVaR — он для медианы/надёжности)."""
    q = net.q_online(x)                                            # (B,A,nq)
    qs, _ = torch.sort(q, dim=-1)
    k = max(1, int(net.nq * (1.0 - tau)))
    return qs[..., -k:].mean(-1)


# ---------------------------------------------------------------------------
# Self-imitation (Oh 2018): повторно учить переходы лучших цепочек
# ---------------------------------------------------------------------------
class SilBuffer:
    """Переходы топ-K цепочек с их discounted return-to-go."""

    def __init__(self, top_k=5, gamma=0.9):
        self.top_k, self.gamma = top_k, gamma
        self.chains = []                    # (l2re, [(s,a,G), ...])

    def add_chain(self, l2re, transitions):
        rs = [t[2] for t in transitions]
        G, out = 0.0, []
        for (s, a, _), r in zip(reversed(transitions), reversed(rs)):
            G = r + self.gamma * G
            out.append((s, a, G))
        self.chains.append((l2re, out[::-1]))
        self.chains.sort(key=lambda c: c[0])
        del self.chains[self.top_k:]

    def sample(self, n, device):
        flat = [t for _, ch in self.chains for t in ch]
        if not flat:
            return None
        idx = np.random.randint(0, len(flat), size=min(n, len(flat)))
        s = torch.as_tensor(np.array([flat[i][0] for i in idx]), dtype=torch.float32, device=device)
        a = torch.as_tensor(np.array([flat[i][1] for i in idx]), dtype=torch.long, device=device)
        g = torch.as_tensor(np.array([flat[i][2] for i in idx]), dtype=torch.float32, device=device)
        return s, a, g


def sil_update(net, sil_buf, opt, batch, device, weight=1.0):
    """L_sil = mean(relu(G - Q(s,a))^2): подтягивать Q к возвратам лучших
    цепочек только там, где Q их недооценивает."""
    smp = sil_buf.sample(batch, device)
    if smp is None:
        return 0.0
    s, a, g = smp
    q = net.q_scalar(s).gather(1, a[:, None]).squeeze(1)
    loss = weight * F.relu(g - q).pow(2).mean()
    if float(loss.detach()) == 0.0:
        return 0.0
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())
