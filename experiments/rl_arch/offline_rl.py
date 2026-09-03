#!/usr/bin/env python
"""
Offline-RL architecture benchmark on the rlpinn poisson_boltzmann_2d buffer.

Data: 80 episode files (danil-e/rlpinn-ablation-buffers), 5841 transitions;
state = 4x26x26 loss-landscape maps, action = 27 discrete (3 opt x 3 lr x 3 ep),
reward in [-3.47, 0). Behavior policy ~uniform (good offline coverage).

Variants (same data split, training protocol and FQE evaluator for all):
    cnn_dqn       baseline: small CNN encoder + double-DQN
    convnext_dqn  H1: ConvNeXt-style encoder + double-DQN
    cnn_cql       H2a: CNN + conservative Q (CQL penalty)
    cnn_iql       H2b: CNN + implicit Q-learning (expectile V + TD-to-V)
    cnn_qrdqn     H3: CNN + QR-DQN (32 quantiles); policies: mean and CVaR@0.25
    cnn_vqc       H4: CNN + variational quantum circuit Q-head (PennyLane)

Metrics per (variant, seed), holdout = 20% of episodes (fixed split):
    td_error          holdout double-DQN Bellman residual (calibration)
    spearman_q_rtg    rank corr of Q(s, a_logged) vs discounted return-to-go
    fqe_*             fitted Q evaluation of the greedy policy (same FQE arch
                      for every variant): mean value over holdout states /
                      initial states, and CVaR@0.25 over holdout states
    agree_behavior    greedy-policy agreement with logged actions
Results: JSON per run, uploaded to HF dataset rl_arch/{variant}_seed{n}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)   # RL.* / src.* imports when run as a script

import torch  # module-level: QNet/soft_update/q_cvar are used by online_eval too

GAMMA = 0.9   # их значение (rl_agent_params в poisson_boltzmann2d_chain.py)
N_ACTIONS = 27
# эпохи на действие: a -> (opt=a//9, lr=(a//3)%3, ep=a%3); таблица и индексация
# те же, что в online_train_env.py (иначе SMDP-дисконт разойдётся с онлайном)
EPOCHS_TABLE = [100, 1000, 2500, 100, 500, 1000, 100, 200, 300]
REPO = "danil-e/rlpinn-ablation-buffers"
SUBDIR = "poisson_boltzmann_2d"
OUT_REPO = "danil-e/pinnacle-optuna-db"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_episodes(data_dir: str | None, subdir: str = SUBDIR):
    import torch
    from huggingface_hub import list_repo_files, hf_hub_download

    if data_dir and os.path.isdir(data_dir) and any(f.endswith(".pt") for f in os.listdir(data_dir)):
        paths = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt"))
    else:
        names = [f for f in list_repo_files(REPO, repo_type="dataset")
                 if f.startswith(subdir + "/") and f.endswith(".pt")]
        paths = [hf_hub_download(REPO, f, repo_type="dataset") for f in sorted(names)]
    episodes = []
    for p in paths:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(d, list) or not d:
            continue
        episodes.append(d)
    return episodes


def episodes_to_arrays(episodes, fix_next_state: bool = False):
    """fix_next_state: the poisson_boltzmann_2d dump logs next_state as a COPY of
    state (100% of transitions; healthy dumps satisfy next_state[t]==state[t+1]
    in 92%). Reconstruct s' = state[t+1] within each episode, last step terminal."""
    base = ["loss_total", "loss_oper", "loss_bnd"]
    S, A, R, S2, D, EP = [], [], [], [], [], []
    for ei, ep in enumerate(episodes):
        prev_tot = prev_tot_ns = None
        for t in ep:
            # healthy dumps carry 3 channels; pb2d also stores `delta`. Keep the
            # 4-channel layout everywhere by deriving delta from consecutive maps
            # (same clip as the online env).
            cur = [np.asarray(t["state"][k], dtype=np.float32) for k in base]
            nxt = [np.asarray(t["next_state"][k], dtype=np.float32) for k in base]
            if "delta" in t["state"]:
                cur.append(np.asarray(t["state"]["delta"], dtype=np.float32))
                nxt.append(np.asarray(t["next_state"]["delta"], dtype=np.float32))
            else:
                cur.append(np.zeros_like(cur[0]) if prev_tot is None
                           else np.clip(cur[0] - prev_tot, -1, 1))
                nxt.append(np.clip(nxt[0] - cur[0], -1, 1))
                prev_tot = cur[0]
            S.append(np.stack(cur))
            S2.append(np.stack(nxt))
            a = t["action"]
            A.append(int(a[0]) * 9 + int(a[1]["lr"]) * 3 + int(a[1]["epochs"]))
            R.append(float(t["reward"]))
            D.append(float(t.get("done", 0)))
            EP.append(ei)
    if fix_next_state:
        by_ep = {}
        for i, e in enumerate(EP):
            by_ep.setdefault(e, []).append(i)
        for e, idxs in by_ep.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                S2[a] = S[b]
    S = np.stack(S); S2 = np.stack(S2)
    A = np.array(A, dtype=np.int64); R = np.array(R, dtype=np.float32)
    D = np.array(D, dtype=np.float32); EP = np.array(EP, dtype=np.int64)
    # force terminal at each episode's last transition (safety)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        D[idx[-1]] = 1.0
    # discounted return-to-go per episode
    RTG = np.zeros_like(R)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        run = 0.0
        for i in idx[::-1]:
            run = R[i] + GAMMA * run * (1.0 - D[i])
            RTG[i] = run
    FIRST = np.zeros_like(D)
    for ei in np.unique(EP):
        FIRST[np.where(EP == ei)[0][0]] = 1.0
    return dict(S=S, A=A, R=R, S2=S2, D=D, EP=EP, RTG=RTG, FIRST=FIRST)


def split_by_episode(data, test_frac=0.2, split_seed=0):
    rng = np.random.default_rng(split_seed)
    eps = np.unique(data["EP"])
    order = rng.permutation(eps)
    counts = {ei: int((data["EP"] == ei).sum()) for ei in eps}
    total = sum(counts.values())
    test_eps, acc = [], 0
    for ei in order:
        if acc < test_frac * total:
            test_eps.append(ei); acc += counts[ei]
    test_mask = np.isin(data["EP"], test_eps)
    return ~test_mask, test_mask


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_encoder(kind: str, in_ch: int = 4):
    import torch
    import torch.nn as nn

    if kind == "cnn":
        class CNN(nn.Module):
            out_dim = 256
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 13
                    nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 6
                    nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU(),
                )
            def forward(self, x):
                return self.net(x)
        return CNN()

    if kind == "their":
        # ИХ боевой энкодер (RL/rl_utils/DQN_classes.py, идентичен ветке
        # rlpinn_pde_tolerance): 3 свёртки -> GAP -> MLP(64). Возвращает
        # кортеж (flat, h) — оборачиваем, чтобы отдавать h.
        from RL.rl_utils.DQN_classes import ConvEncoder

        class TheirEncoder(nn.Module):
            out_dim = 64
            def __init__(self):
                super().__init__()
                self.enc = ConvEncoder()
            def forward(self, x):
                return self.enc(x)[1]
        return TheirEncoder()

    if kind == "convnext":
        class Block(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
                self.ln = nn.LayerNorm(c)
                self.p1 = nn.Linear(c, 4 * c)
                self.p2 = nn.Linear(4 * c, c)
                self.act = nn.GELU()
            def forward(self, x):
                y = self.dw(x).permute(0, 2, 3, 1)
                y = self.p2(self.act(self.p1(self.ln(y)))).permute(0, 3, 1, 2)
                return x + y
        class ConvNeXtTiny(nn.Module):
            out_dim = 256
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(in_ch, 48, 2, stride=2)          # 13
                self.s1 = nn.Sequential(Block(48), Block(48))
                self.down = nn.Conv2d(48, 96, 2, stride=2)         # 6
                self.s2 = nn.Sequential(Block(96), Block(96))
                self.head = nn.Sequential(nn.Linear(96, 256), nn.ReLU())
            def forward(self, x):
                x = self.s2(self.down(self.s1(self.stem(x))))
                x = x.mean(dim=(2, 3))
                return self.head(x)
        return ConvNeXtTiny()

    raise ValueError(kind)


def make_head(variant: str, in_dim: int, n_quantiles: int):
    import torch
    import torch.nn as nn

    if variant in ("their_dqn", "their_cql", "cnx_dueling"):
        # их дуэлинговая голова: Q = V + A - mean(A)
        from RL.rl_utils.DQN_classes import DuelingHead
        return DuelingHead(in_dim, N_ACTIONS)

    if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
        return nn.Linear(in_dim, N_ACTIONS * n_quantiles)
    if variant == "cnn_vqc":
        import pennylane as qml
        n_qubits, n_layers = 8, 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        qlayer = qml.qnn.TorchLayer(circuit, {"weights": wshape})

        class VQCHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.compress = nn.Linear(in_dim, n_qubits)
                self.q = qlayer
                self.out = nn.Linear(n_qubits, N_ACTIONS)
            def forward(self, z):
                z = torch.tanh(self.compress(z)) * math.pi / 2
                return self.out(self.q(z).float())
        return VQCHead()
    return nn.Linear(in_dim, N_ACTIONS)


class QNet:
    """Encoder + head with a target copy; variant-specific loss."""

    def __init__(self, variant, device, n_quantiles=32, in_ch=4):
        import torch
        import torch.nn as nn
        if variant in ("their_dqn", "their_cql"):
            enc_kind = "their"
        elif variant in ("convnext_dqn", "cnx_cql", "cnx_cql_qr", "cnx_dueling",
                         "cnx_bcq", "cnx_bbf", "cnx_qrdqn"):
            enc_kind = "convnext"
        else:
            enc_kind = "cnn"
        self.variant = variant
        self.nq = n_quantiles
        self.enc = make_encoder(enc_kind, in_ch)
        self.head = make_head(variant, self.enc.out_dim, n_quantiles)
        self.v_head = nn.Linear(self.enc.out_dim, 1) if variant == "cnn_iql" else None
        mods = [self.enc, self.head] + ([self.v_head] if self.v_head is not None else [])
        self.model = nn.ModuleList(mods).to(device)
        import copy
        self.target = copy.deepcopy(self.model).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.device = device

    def params(self):
        return self.model.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def _q(self, model, x):
        z = model[0](x)
        out = model[1](z)
        if self.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
            return out.view(-1, N_ACTIONS, self.nq)
        return out

    def q_online(self, x):
        return self._q(self.model, x)

    def q_target(self, x):
        return self._q(self.target, x)

    def v_online(self, x):
        return self.model[2](self.model[0](x)).squeeze(-1)

    def q_scalar(self, x):
        """(B, N_ACTIONS) scalar Q for metrics/policies (mean over quantiles)."""
        q = self.q_online(x)
        return q.mean(-1) if self.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn") else q

    def q_cvar(self, x, alpha=0.25):
        import torch  # module-level `torch` only exists after main(); keep importable
        q = self.q_online(x)          # (B, A, nq), quantiles unsorted -> sort
        qs, _ = torch.sort(q, dim=-1)
        k = max(1, int(self.nq * alpha))
        return qs[..., :k].mean(-1)

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(), self.target.parameters()):
                tp.mul_(1 - tau).add_(tau * p)


# --------------------------------------------------------------------------
# Targets: n-step returns, potential-based shaping, SMDP discount
# --------------------------------------------------------------------------

def action_epochs(a) -> int:
    """Сколько эпох тратит действие a. Кодировка a = opt*9 + lr*3 + ep."""
    return EPOCHS_TABLE[(int(a) // 9) * 3 + (int(a) % 3)]


def build_targets(data, args, gamma):
    """n-шаговые цели + PBRS + SMDP-дисконт.

    Возвращает (NR, NG, NIDX, ND):
        NR[i]   сумма (возможно преобразованных) наград на m шагах вперёд
        NG[i]   произведение дисконтов на этих m шагах — множитель бутстрэпа
        NIDX[i] индекс перехода, из next_state которого делается бутстрэп
        ND[i]   признак терминальности в точке бутстрэпа
    Цель обучения: tgt = NR + NG * (1 - ND) * Q(s'_{NIDX}).

    При n_step=1, pbrs=0 и выключенном smdp это в точности прежние
    (R, gamma, i, D), то есть поведение не меняется.

    Награда в буфере — «absolute» (RL/rl_environment.compute_reward):
    r_t = -(coeff_op*err_op + coeff_bnd*err_bnd) состояния s_{t+1}.
    Проверено на данных: все 26 858 наград строго отрицательны. Поэтому
    потенциал восстанавливается точно: Ф(s_{t+1}) = -log10(-r_t), а
    Ф(s_t) = -log10(-r_{t-1}). У первого перехода эпизода Ф(s_0) неизвестен
    и добавка обнуляется — так же, как в онлайновой реализации.
    """
    R, D, EP, A = data["R"], data["D"], data["EP"], data["A"]
    n = len(R)

    gam = np.full(n, float(gamma), dtype=np.float64)
    if getattr(args, "smdp", False):
        cost = np.array([action_epochs(a) for a in A], dtype=np.float64)
        gam = float(gamma) ** (cost / float(args.smdp_scale))

    R_eff = R.astype(np.float64).copy()
    w_pbrs = float(getattr(args, "pbrs", 0.0) or 0.0)
    if w_pbrs:
        def phi(err):
            return -math.log10(max(float(err), 1e-8))
        for ei in np.unique(EP):
            idx = np.where(EP == ei)[0]
            for pos in range(1, len(idx)):
                i, prev = idx[pos], idx[pos - 1]
                # ошибка s' — из награды самого перехода, ошибка s — из предыдущей
                R_eff[i] += w_pbrs * (gam[i] * phi(-R[i]) - phi(-R[prev]))

    nst = max(1, int(getattr(args, "n_step", 1) or 1))
    NR = np.zeros(n, dtype=np.float64)
    NG = np.zeros(n, dtype=np.float64)
    NIDX = np.arange(n, dtype=np.int64)
    ND = np.zeros(n, dtype=np.float64)
    for ei in np.unique(EP):
        idx = np.where(EP == ei)[0]
        L = len(idx)
        for pos in range(L):
            i = idx[pos]
            m = min(nst, L - pos)
            acc, g = 0.0, 1.0
            for k in range(m):
                j = idx[pos + k]
                acc += g * R_eff[j]
                g *= gam[j]
            last = idx[pos + m - 1]
            NR[i], NG[i], NIDX[i], ND[i] = acc, g, last, D[last]
    return (NR.astype(np.float32), NG.astype(np.float32), NIDX,
            ND.astype(np.float32))


# группа симметрий латентной сетки. Сетка строится на квадрате [-1.2, 1.2]^2
# с одинаковым шагом по обеим осям (plot_loss_surface.py: min_y, max_y =
# min_x, max_x; один step_size), 26 узлов симметричны относительно нуля,
# поэтому D4 переставляет узлы точно. Базис 2-мерного латента автоэнкодера
# обучается заново на каждом эпизоде и канонической ориентации не имеет.
# Проверено на буфере ns2d (3229 карт): центроид области минимума
# (12.47, 12.48) при симметричном центре 12.5, баланс масс верх/низ 0.970 и
# лево/право 0.979, расхождение средней карты с её отражениями 0.011-0.031
# при шуме половина-к-половине 0.022.
def d4_apply(x, g: int):
    """g в 0..7: g>=4 — отражение по последней оси, затем g%4 поворотов."""
    if g >= 4:
        x = torch.flip(x, dims=[-1])
    k = g % 4
    return torch.rot90(x, k, dims=[-2, -1]) if k else x


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_smdp(data, train_mask, args, seed):
    """Полный стек SMDP-BBF-CQL (см. advanced_agents.py)."""
    import torch.nn.functional as F
    from advanced_agents import SmdpAgent, fit_behaviour, ACTION_DT

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    agent = SmdpAgent(device)
    opt = torch.optim.AdamW(agent.params(), lr=1e-4, weight_decay=0.1)

    idx = np.where(train_mask)[0]
    mean = data["S"][idx].mean(axis=(0, 2, 3), keepdims=True)
    std = data["S"][idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    S = torch.as_tensor((data["S"] - mean) / std, device=device).float()
    S2 = torch.as_tensor((data["S2"] - mean) / std, device=device).float()
    A = torch.as_tensor(data["A"], device=device)
    R = torch.as_tensor(data["R"], device=device)
    D = torch.as_tensor(data["D"], device=device)
    DT = torch.as_tensor(ACTION_DT[data["A"]], device=device)      # длительность действия

    fit_behaviour(agent, S[idx], data["A"][idx], device)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    bs = args.batch_size
    total = args.epochs * max(1, len(idx) // bs)
    for step in range(total):
        b = torch.as_tensor(rng.integers(0, len(idx), size=bs), device=device).long()
        b = torch.as_tensor(idx, device=device)[b]
        gamma = 0.97 + min(1.0, step / 2000) * (0.997 - 0.97)      # отжиг gamma
        with torch.no_grad():
            q_next_online = agent.q_scalar(S2[b])
            mask = agent.bcq_mask(S2[b])                            # BCQ-поддержка
            a_star = q_next_online.masked_fill(~mask, -1e9).argmax(-1)
            tl, _ = agent.target.all_logits(S2[b])
            tq = agent.hlg.to_scalar(tl)                            # (K,B,A)
            k1, k2 = rng.choice(agent.n_heads, size=2, replace=False)
            q_next = torch.minimum(tq[k1], tq[k2]).gather(-1, a_star[:, None])[:, 0]
            y = (R[b] + (gamma ** DT[b]) * (1 - D[b]) * q_next).clamp(-20.0, 2.0)
        logits, h = agent.net.all_logits(S[b])
        ii = A[b][None, :, None, None].expand(agent.n_heads, -1, 1, agent.net.n_bins)
        taken = logits.gather(2, ii).squeeze(2)
        td = torch.stack([agent.hlg.loss(taken[k], y) for k in range(agent.n_heads)]).mean()
        q_all = agent.hlg.to_scalar(logits).mean(0)
        cql = (torch.logsumexp(q_all, -1) - q_all.gather(-1, A[b][:, None])[:, 0]).mean()
        alpha_cql = 0.5
        phi = agent.net.act_emb(agent.net.action_feats)[A[b]]
        h_hat = agent.net.spr_tr(torch.cat([h, phi], -1))
        with torch.no_grad():
            h_tgt = agent.target.spr_proj(agent.target.embed(S2[b]))
        spr = -F.cosine_similarity(agent.net.spr_proj(h_hat), h_tgt, dim=-1).mean()
        loss = td + alpha_cql * cql + 1.0 * spr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 10.0)
        opt.step(); agent.soft_update()
        if (step + 1) % 2000 == 0:                                  # BBF-сброс
            agent.shrink_and_perturb()
            opt = torch.optim.AdamW(agent.params(), lr=1e-4, weight_decay=0.1)
        if (step + 1) % max(1, total // 4) == 0:
            print(f"  [cnx_smdp s{seed}] шаг {step+1}/{total} td={td.item():.4f} "
                  f"cql={cql.item():.4f} spr={spr.item():.4f}", flush=True)
    return agent, dict(mean=mean, std=std), time.time() - t0


def train_variant(variant, data, train_mask, args, seed):
    if variant == "cnx_smdp":
        return train_smdp(data, train_mask, args, seed)
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = QNet(variant, device)
    opt = torch.optim.Adam(net.params(), lr=1e-3,
                           weight_decay=float(getattr(args, "l2", 0.0) or 0.0))

    idx = np.where(train_mask)[0]
    mean = data["S"][idx].mean(axis=(0, 2, 3), keepdims=True)
    std = data["S"][idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6

    def to_t(x):
        return torch.as_tensor(x, device=device)

    def norm(s):
        return (s - mean) / std

    S = to_t(norm(data["S"])).float(); S2 = to_t(norm(data["S2"])).float()
    A = to_t(data["A"]); R = to_t(data["R"]); D = to_t(data["D"])

    # n-шаговые цели / PBRS / SMDP: при значениях по умолчанию совпадают с (R, GAMMA, D)
    nr_np, ng_np, nidx_np, nd_np = build_targets(data, args, GAMMA)
    NR = to_t(nr_np); NG = to_t(ng_np); ND = to_t(nd_np)
    NIDX = to_t(nidx_np).long()
    aug_n = 8 if getattr(args, "aug", "none") == "d4" else 1
    w_margin = float(getattr(args, "dqfd", 0.0) or 0.0)
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    ckpts = []

    bs = args.batch_size
    n_epochs = args.epochs
    taus = None
    if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
        taus = (torch.arange(net.nq, device=device, dtype=torch.float32) + 0.5) / net.nq

    bcq_behaviour = None
    if variant == "cnx_bcq":
        from advanced_agents import fit_behaviour
        import torch.nn as nn
        bcq_behaviour = nn.Sequential(nn.Linear(4 * 26 * 26, 256), nn.LayerNorm(256),
                                      nn.GELU(), nn.Linear(256, N_ACTIONS)).to(device)
        class _Holder: pass
        _h = _Holder(); _h.behaviour = bcq_behaviour
        fit_behaviour(_h, S[np.where(train_mask)[0]], data["A"][train_mask], device)
        print(f"  [cnx_bcq s{seed}] модель поведения обучена", flush=True)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    best_td, stall = float("inf"), 0
    test_idx = np.where(~train_mask)[0]
    for epoch in range(n_epochs):
        order = rng.permutation(idx)
        for k in range(0, len(order), bs):
            b = to_t(order[k:k + bs]).long()
            # r/d/gm — n-шаговые: награда за m шагов, терминальность и дисконт
            # в точке бутстрэпа; s2 — next_state именно того перехода (NIDX)
            s, a, r, d, gm = S[b], A[b], NR[b], ND[b], NG[b]
            s2 = S2[NIDX[b]]
            s_raw = s                      # неаугментированное s: маска BCQ и margin
            if aug_n > 1:                  # одно преобразование на батч, общее для s и s'
                g = int(rng.integers(aug_n))
                s = d4_apply(s, g); s2 = d4_apply(s2, g)

            if variant == "cnn_iql":
                with torch.no_grad():
                    q_t = net.q_target(s)
                    q_data_t = q_t.gather(1, a[:, None]).squeeze(1)
                v = net.v_online(s)
                diff = q_data_t - v
                w = torch.where(diff > 0, torch.full_like(diff, 0.7), torch.full_like(diff, 0.3))
                v_loss = (w * diff ** 2).mean()
                with torch.no_grad():
                    v2 = net.v_online(s2)
                    tgt = r + gm * (1 - d) * v2
                q = net.q_online(s).gather(1, a[:, None]).squeeze(1)
                loss = v_loss + F.mse_loss(q, tgt)
            elif variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn"):
                q = net.q_online(s)                                   # (B,A,nq)
                q_data = q.gather(1, a[:, None, None].expand(-1, 1, net.nq)).squeeze(1)
                with torch.no_grad():
                    a2 = net.q_online(s2).mean(-1).argmax(1)
                    q2 = net.q_target(s2).gather(
                        1, a2[:, None, None].expand(-1, 1, net.nq)).squeeze(1)
                    tgt = r[:, None] + (gm * (1 - d))[:, None] * q2    # (B,nq)
                u = tgt[:, None, :] - q_data[:, :, None]               # (B,nq_pred,nq_tgt)
                huber = torch.where(u.abs() <= 1.0, 0.5 * u ** 2, u.abs() - 0.5)
                loss = (torch.abs(taus[None, :, None] - (u.detach() < 0).float()) * huber).mean()
                if variant == "cnx_cql_qr":
                    qm = q.mean(-1)
                    loss = loss + args.cql_alpha * (
                        torch.logsumexp(qm, dim=1) - qm.gather(1, a[:, None]).squeeze(1)
                    ).mean()
            else:
                q = net.q_online(s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = net.q_online(s2).argmax(1)
                    q2 = net.q_target(s2).gather(1, a2[:, None]).squeeze(1)
                    tgt = r + gm * (1 - d) * q2
                loss = F.mse_loss(q, tgt)
                if variant == "cnx_bcq":      # discrete BCQ: маска поддержки
                    # модель поведения обучена на неаугментированных состояниях,
                    # поэтому маска берётся с s_raw; сама маска считается
                    # инвариантной к D4 — это и есть предпосылка аугментации
                    with torch.no_grad():
                        pb = bcq_behaviour(s_raw.flatten(1)).softmax(-1)
                        keep = pb >= 0.3 * pb.max(-1, keepdim=True).values
                    qs_all = net.q_online(s)
                    loss = loss + 0.5 * (qs_all.masked_fill(keep, 0.0).clamp_min(0) ** 2).mean()
                if variant in ("cnn_cql", "cnx_cql", "their_cql"):
                    qs = net.q_online(s)
                    loss = loss + args.cql_alpha * (
                        torch.logsumexp(qs, dim=1) - qs.gather(1, a[:, None]).squeeze(1)
                    ).mean()

            if w_margin:
                # DQfD, большой отступ: J_E = max_a[Q(s,a)+L(a_E,a)] - Q(s,a_E).
                # Все переходы буфера считаются демонстрацией (чистый офлайн).
                qs_m = net.q_scalar(s)
                marg = torch.full_like(qs_m, float(args.dqfd_margin))
                marg.scatter_(1, a[:, None], 0.0)
                loss = loss + w_margin * (
                    (qs_m + marg).max(1).values
                    - qs_m.gather(1, a[:, None]).squeeze(1)).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            net.soft_update()
            if variant == "cnx_bbf" and (epoch * 1000 + k) % 4000 == 3999:
                # BBF: shrink-and-perturb — ствол сохраняет 50%, головы заново
                import copy as _copy
                fresh = QNet(variant, device)
                with torch.no_grad():
                    for (nm, p), pf in zip(net.model.named_parameters(),
                                           fresh.model.parameters()):
                        a_keep = 0.0 if nm.startswith("1") else 0.5
                        p.mul_(a_keep).add_((1 - a_keep) * pf)
                net.target.load_state_dict(net.model.state_dict())
                opt = torch.optim.Adam(net.params(), lr=1e-3)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  [{variant} s{seed}] epoch {epoch+1}/{n_epochs} loss={loss.item():.4f}", flush=True)
        if args.plateau_patience and (epoch + 1) % 25 == 0:
            with torch.no_grad():
                b = to_t(test_idx).long()
                qs = net.q_scalar(S[b]).gather(1, A[b][:, None]).squeeze(1)
                a2 = net.q_scalar(S2[b]).argmax(1)
                q2 = (net.q_target(S2[b]).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                      else net.q_target(S2[b])).gather(1, a2[:, None]).squeeze(1)
                td = float(((qs - (R[b] + GAMMA * (1 - D[b]) * q2)) ** 2).mean().sqrt())
            if td < best_td - 1e-3:
                best_td, stall = td, 0
            else:
                stall += 1
            print(f"  [{variant} s{seed}] plateau-check ep{epoch+1}: td={td:.3f} best={best_td:.3f} stall={stall}", flush=True)
            if stall >= args.plateau_patience:
                print(f"  [{variant} s{seed}] plateau reached at epoch {epoch+1}", flush=True)
                break
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            import copy as _c
            ckpts.append((epoch + 1, _c.deepcopy(net.model.state_dict())))

    train_time = time.time() - t0
    net.extra = {}
    if getattr(args, "fqe_select", False):
        # отбор чекпоинта по FQE на отложенных эпизодах: сравниваем ценность
        # жадной политики каждого снимка, берём лучший. Последняя эпоха всегда
        # в списке, поэтому отбор не может оказаться хуже обычного финала по
        # этому критерию (но по метрике в среде — может, критерий приблизителен)
        import copy as _c
        last_ep = epoch + 1
        pool = list(ckpts)
        if not pool or pool[-1][0] != last_ep:      # финальные веса всегда в пуле
            pool.append((last_ep, _c.deepcopy(net.model.state_dict())))
        stats_sel = dict(mean=mean, std=std)
        pol = "cvar" if (variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                         and getattr(args, "select_policy", "mean") == "cvar") else "mean"
        scored = []
        for pi, (ep_no, sd) in enumerate(pool):
            net.model.load_state_dict(sd)
            fq = fqe(net, stats_sel, data, train_mask, ~train_mask, pol, args, seed)
            v = fq["fqe_value_init"]
            if v is None:
                v = fq["fqe_value_all"]
            scored.append((float(v), pi))
            print(f"  [{variant} s{seed}] чекпоинт эпоха {ep_no}: FQE={v:.4f}", flush=True)
        best_v, best_pi = max(scored)
        best_ep = pool[best_pi][0]
        net.model.load_state_dict(pool[best_pi][1])
        net.target.load_state_dict(net.model.state_dict())
        net.extra = {"fqe_select_epoch": best_ep, "fqe_select_value": best_v,
                     "fqe_select_pool": [pool[pi][0] for _, pi in scored]}
        print(f"  [{variant} s{seed}] выбран чекпоинт эпохи {best_ep} (FQE={best_v:.4f})",
              flush=True)
    return net, dict(mean=mean, std=std), train_time


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(net, stats, data, test_mask, policy="mean"):
    import torch
    from scipy.stats import spearmanr

    device = net.device
    idx = np.where(test_mask)[0]
    S = torch.as_tensor((data["S"] - stats["mean"]) / stats["std"], device=device).float()
    S2 = torch.as_tensor((data["S2"] - stats["mean"]) / stats["std"], device=device).float()

    def batched(fn, X, bs=512):
        outs = []
        with torch.no_grad():
            for k in range(0, len(X), bs):
                outs.append(fn(X[k:k + bs]))
        return torch.cat(outs)

    qfun = (lambda x: net.q_cvar(x)) if policy == "cvar" else (lambda x: net.q_scalar(x))
    q_all = batched(qfun, S[idx])
    a_log = torch.as_tensor(data["A"][idx], device=device)
    q_data = q_all.gather(1, a_log[:, None]).squeeze(1).cpu().numpy()

    # holdout TD error (double-DQN residual on scalar Q)
    with torch.no_grad():
        q_scal = batched(lambda x: net.q_scalar(x), S[idx])
        a2 = batched(lambda x: net.q_scalar(x), S2[idx]).argmax(1)
        q2t = batched(lambda x: net.q_target(x).mean(-1)
                      if net.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn")
                      else net.q_target(x), S2[idx])
        q2 = q2t.gather(1, a2[:, None]).squeeze(1)
        r = torch.as_tensor(data["R"][idx], device=device)
        d = torch.as_tensor(data["D"][idx], device=device)
        td = (q_scal.gather(1, a_log[:, None]).squeeze(1)
              - (r + GAMMA * (1 - d) * q2)).cpu().numpy()

    rho = spearmanr(q_data, data["RTG"][idx]).statistic
    greedy = q_all.argmax(1).cpu().numpy()
    agree = float((greedy == data["A"][idx]).mean())
    ent = 0.0
    counts = np.bincount(greedy, minlength=N_ACTIONS).astype(float)
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return dict(td_error=float(np.sqrt((td ** 2).mean())),
                spearman_q_rtg=float(rho),
                agree_behavior=agree,
                policy_entropy=ent), greedy


def fqe(net, stats, data, train_mask, test_mask, policy, args, seed):
    """Fitted Q Evaluation of the variant's greedy policy with a FIXED
    CNN architecture (identical for every variant)."""
    import torch
    import torch.nn.functional as F

    device = net.device
    torch.manual_seed(seed + 10_000)
    enc = make_encoder("cnn").to(device)
    head = torch.nn.Linear(enc.out_dim, N_ACTIONS).to(device)
    model = torch.nn.ModuleList([enc, head])
    import copy
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    S = torch.as_tensor((data["S"] - stats["mean"]) / stats["std"], device=device).float()
    S2 = torch.as_tensor((data["S2"] - stats["mean"]) / stats["std"], device=device).float()
    A = torch.as_tensor(data["A"], device=device)
    R = torch.as_tensor(data["R"], device=device)
    D = torch.as_tensor(data["D"], device=device)

    # policy actions on next states (from the trained variant net)
    qfun = (lambda x: net.q_cvar(x)) if policy == "cvar" else (lambda x: net.q_scalar(x))
    with torch.no_grad():
        pi_s2 = []
        for k in range(0, len(S2), 512):
            pi_s2.append(qfun(S2[k:k + 512]).argmax(1))
        pi_s2 = torch.cat(pi_s2)

    idx = np.where(train_mask)[0]
    rng = np.random.default_rng(seed + 1)
    for epoch in range(args.fqe_epochs):
        order = rng.permutation(idx)
        for k in range(0, len(order), args.batch_size):
            b = torch.as_tensor(order[k:k + args.batch_size], device=device).long()
            q = head(enc(S[b])).gather(1, A[b][:, None]).squeeze(1)
            with torch.no_grad():
                q2 = target[1](target[0](S2[b])).gather(1, pi_s2[b][:, None]).squeeze(1)
                tgt = R[b] + GAMMA * (1 - D[b]) * q2
            loss = F.mse_loss(q, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.mul_(0.995).add_(0.005 * p)

    tidx = np.where(test_mask)[0]
    with torch.no_grad():
        vals = []
        for k in range(0, len(tidx), 512):
            b = torch.as_tensor(tidx[k:k + 512], device=device).long()
            qv = head(enc(S[b]))
            pa = []
            for kk in range(0, len(b), 512):
                pa.append(qfun(S[b][kk:kk + 512]).argmax(1))
            pa = torch.cat(pa)
            vals.append(qv.gather(1, pa[:, None]).squeeze(1))
        vals = torch.cat(vals).cpu().numpy()
    first = data["FIRST"][tidx].astype(bool)
    vs = np.sort(vals)
    k25 = max(1, int(0.25 * len(vs)))
    return dict(fqe_value_all=float(vals.mean()),
                fqe_value_init=float(vals[first].mean()) if first.any() else None,
                fqe_cvar25=float(vs[:k25].mean()))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def upload_result(row: dict, name: str):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF write token; result kept local only", flush=True)
        return
    from huggingface_hub import upload_file
    import io
    payload = json.dumps(row, indent=1).encode()
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(payload),
                        path_in_repo=f"rl_arch/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch: {name}")
            print(f"uploaded rl_arch/{name}.json", flush=True)
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(5)


def main():
    global GAMMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["cnn_dqn", "convnext_dqn", "cnn_cql", "cnn_iql",
                             "cnn_qrdqn", "cnn_vqc", "cnx_cql", "cnx_cql_qr",
                             "their_dqn", "their_cql", "cnx_dueling",
                             "cnx_bcq", "cnx_bbf", "cnx_smdp", "cnx_qrdqn"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--fqe-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--cql-alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--subdir", default=SUBDIR,
                    help="Buffer folder in the HF dataset (e.g. poisson3d_complexgeometry)")
    ap.add_argument("--fix-next-state", action="store_true",
                    help="Reconstruct s'=state[t+1] (poisson_boltzmann_2d dump has s'==s)")
    ap.add_argument("--save-model", action="store_true",
                    help="Save agent checkpoint and upload to HF rl_arch/models/")
    ap.add_argument("--plateau-patience", type=int, default=0,
                    help="Stop when holdout TD stops improving for N checks (every 25 epochs); 0=off")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # --- методы из статей (по умолчанию всё выключено: прежнее поведение) ---
    ap.add_argument("--aug", default="none", choices=["none", "d4"],
                    help="D4-аугментация карт: 8 симметрий квадратной латентной сетки")
    ap.add_argument("--n-step", type=int, default=1,
                    help="n-шаговые возвраты внутри эпизода (1 = как раньше)")
    ap.add_argument("--dqfd", type=float, default=0.0,
                    help="Вес отступного лосса DQfD (0 = выключен)")
    ap.add_argument("--dqfd-margin", type=float, default=0.8,
                    help="Отступ L(a_E,a) в лоссе DQfD")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="weight_decay в Adam (L2 из DQfD)")
    ap.add_argument("--pbrs", type=float, default=0.0,
                    help="Вес потенциального преобразования награды, Ф=-log10(ошибка)")
    ap.add_argument("--smdp", action="store_true",
                    help="Дисконт по длительности действия: gamma^(эпохи/scale)")
    ap.add_argument("--smdp-scale", type=float, default=100.0)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="Снимать чекпоинт каждые N эпох (для --fqe-select)")
    ap.add_argument("--fqe-select", action="store_true",
                    help="Выбрать чекпоинт с лучшей ценностью по FQE на отложенных эпизодах")
    ap.add_argument("--select-policy", default="mean", choices=["mean", "cvar"])
    ap.add_argument("--model-tag", default="",
                    help="Суффикс в именах результатов и чекпоинта: арм не перетирает базу")
    args = ap.parse_args()
    if args.fqe_select and not args.ckpt_every:
        ap.error("--fqe-select требует --ckpt-every N")
    if args.n_step < 1:
        ap.error("--n-step должен быть >= 1")
    if args.smoke:
        args.epochs, args.fqe_epochs = 2, 2

    GAMMA = args.gamma
    print(f"loading episodes... (gamma={GAMMA})", flush=True)
    episodes = load_episodes(args.data_dir, args.subdir)
    data = episodes_to_arrays(episodes, fix_next_state=args.fix_next_state)
    train_mask, test_mask = split_by_episode(data)
    print(f"episodes={len(episodes)} transitions={len(data['A'])} "
          f"train={int(train_mask.sum())} test={int(test_mask.sum())}", flush=True)

    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time()
        net, stats, train_time = train_variant(args.variant, data, train_mask, args, seed)
        policies = ["mean", "cvar"] if args.variant in ("cnn_qrdqn", "cnx_cql_qr", "cnx_qrdqn") else ["mean"]
        for pol in policies:
            m, _ = evaluate(net, stats, data, test_mask, policy=pol)
            fq = fqe(net, stats, data, train_mask, test_mask, pol, args, seed)
            row = dict(variant=args.variant, policy=pol, seed=seed, gamma=GAMMA,
                       fixed_ns=bool(args.fix_next_state), dataset=args.subdir,
                       n_params=net.n_params(), train_time_s=round(train_time, 1),
                       epochs=args.epochs, smoke=args.smoke,
                       aug=args.aug, n_step=args.n_step, dqfd=args.dqfd,
                       l2=args.l2, pbrs=args.pbrs, smdp=bool(args.smdp),
                       model_tag=args.model_tag, **getattr(net, "extra", {}), **m, **fq)
            print(json.dumps(row), flush=True)
            if not args.smoke:
                tag = ("_fixns" if args.fix_next_state else "") + \
                      ("" if args.subdir == SUBDIR else "_" + args.subdir.split("_")[0]) + \
                      (("_" + args.model_tag) if args.model_tag else "")
                upload_result(row, f"{args.variant}{tag}_{pol}_seed{seed}")
        if args.save_model and not args.smoke:
            import torch as _t
            ckpt = {"variant": args.variant, "seed": seed,
                    "state_dict": {k: v.cpu() for k, v in net.model.state_dict().items()},
                    "mean": stats["mean"], "std": stats["std"]}
            sfx = ("_fixns" if args.fix_next_state else "") + \
                  ("" if args.subdir == SUBDIR else "_" + args.subdir.split("_")[0]) + \
                  (("_" + args.model_tag) if args.model_tag else "")
            fn = f"/tmp/agent_{args.variant}{sfx}_seed{seed}.pt"
            _t.save(ckpt, fn)
            tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
            if tok:
                # выгрузка НЕ критична: локальный чекпоинт уже сохранён и его
                # достаточно для онлайн-оценки в этом же кернеле. При параллельной
                # записи из десятков кернелов HF отдаёт 429/конфликт коммита —
                # раньше это роняло весь кернел.
                from huggingface_hub import upload_file
                for attempt in range(4):
                    try:
                        upload_file(path_or_fileobj=fn,
                                    path_in_repo=f"rl_arch/models/{args.variant}{sfx}_seed{seed}.pt",
                                    repo_id=OUT_REPO, repo_type="dataset", token=tok,
                                    commit_message=f"rl_arch model {args.variant} seed {seed}")
                        print(f"model uploaded: rl_arch/models/{args.variant}{sfx}_seed{seed}.pt", flush=True)
                        break
                    except Exception as e:
                        wait = 10 * (attempt + 1) + (seed % 7)
                        print(f"WARNING: model upload attempt {attempt+1}/4 failed: {e}", flush=True)
                        if attempt < 3:
                            time.sleep(wait)
                else:
                    print(f"model kept locally only: {fn}", flush=True)
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
