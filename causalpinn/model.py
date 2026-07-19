"""Modified MLP with gated skip mixing + causal-paper input encodings.

Faithful PyTorch port of PredictiveIntelligenceLab/CausalPINNs:
  - KS   : modified_MLP   from KS/chaotic_KS.py  (1D Fourier x + multi-scale t)
  - GS   : modified_MLP_II from NS/NS.py          (2D tensor-product Fourier + multi-scale t)
  - plain: same gated MLP, raw (normalized) inputs + multi-scale t (ablation of encoding)
"""
import numpy as np
import torch
import torch.nn as nn


class KSEncoding(nn.Module):
    """[k_t * t, 1, cos(k w x), sin(k w x)];  k_t = 10^{-M_t//2 .. M_t//2-1}, k = 1..M_x."""

    def __init__(self, M_t=6, M_x=5, L=2 * np.pi):
        super().__init__()
        # NB: the reference uses jnp.power(10, arange(-M_t//2, M_t//2)) with
        # INTEGER dtype — negative exponents evaluate to 0 in JAX. The trained
        # networks therefore see k_t = [0,0,0,1,10,100]; replicate exactly.
        k_t = np.power(10, np.arange(-(M_t // 2), M_t // 2),
                       dtype=np.float64, where=np.arange(-(M_t // 2), M_t // 2) >= 0)
        k_t[np.arange(-(M_t // 2), M_t // 2) < 0] = 0.0
        self.register_buffer("k_t", torch.tensor(k_t, dtype=torch.float32))
        self.register_buffer("k_x", torch.arange(1, M_x + 1, dtype=torch.float32))
        self.w = 2 * np.pi / L
        self.dim = M_t + 1 + 2 * M_x

    def forward(self, t, x):
        # t, x: (N, 1)
        wx = self.k_x * (self.w * x)
        return torch.cat([self.k_t * t, torch.ones_like(t), torch.cos(wx), torch.sin(wx)], dim=1)


class GS2DEncoding(nn.Module):
    """modified_MLP_II encoding: [1, k_t*t, cos/sin singles, 4 cross blocks];
    k_t = 10^{0..M_t}."""

    def __init__(self, M_t=2, M_x=5, M_y=5, L_x=2.0, L_y=2.0):
        super().__init__()
        self.register_buffer("k_t", torch.tensor(10.0 ** np.arange(0, M_t + 1), dtype=torch.float32))
        self.register_buffer("k_x", torch.arange(1, M_x + 1, dtype=torch.float32))
        self.register_buffer("k_y", torch.arange(1, M_y + 1, dtype=torch.float32))
        kxx, kyy = np.meshgrid(np.arange(1, M_x + 1), np.arange(1, M_y + 1))
        self.register_buffer("k_xx", torch.tensor(kxx.flatten(), dtype=torch.float32))
        self.register_buffer("k_yy", torch.tensor(kyy.flatten(), dtype=torch.float32))
        self.w_x = 2 * np.pi / L_x
        self.w_y = 2 * np.pi / L_y
        self.dim = 1 + (M_t + 1) + 2 * M_x + 2 * M_y + 4 * M_x * M_y

    def forward(self, t, x, y):
        wx = self.k_x * (self.w_x * x)
        wy = self.k_y * (self.w_y * y)
        wxx = self.k_xx * (self.w_x * x)
        wyy = self.k_yy * (self.w_y * y)
        return torch.cat([
            torch.ones_like(t), self.k_t * t,
            torch.cos(wx), torch.cos(wy), torch.sin(wx), torch.sin(wy),
            torch.cos(wxx) * torch.cos(wyy), torch.cos(wxx) * torch.sin(wyy),
            torch.sin(wxx) * torch.cos(wyy), torch.sin(wxx) * torch.sin(wyy),
        ], dim=1)


class PlainEncoding(nn.Module):
    """Fallback (no Fourier): [1, k_t*t, x_1.., x_d] with coords scaled to [-1,1]."""

    def __init__(self, M_t, spatial_dim, bbox):
        super().__init__()
        self.register_buffer("k_t", torch.tensor(10.0 ** np.arange(0, M_t + 1), dtype=torch.float32))
        lo = torch.tensor([bbox[2 * i] for i in range(spatial_dim)], dtype=torch.float32)
        hi = torch.tensor([bbox[2 * i + 1] for i in range(spatial_dim)], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.dim = 1 + (M_t + 1) + spatial_dim

    def forward(self, t, *coords):
        xs = torch.cat(coords, dim=1)
        xs = 2 * (xs - self.lo) / (self.hi - self.lo) - 1
        return torch.cat([torch.ones_like(t), self.k_t * t, xs], dim=1)


class ModifiedMLP(nn.Module):
    """Gated MLP: U/V two-encoder mixing (Wang et al.), tanh activation."""

    def __init__(self, encoding, hidden, out_dim, seed=1234, gate_seeds=(12345, 54321)):
        super().__init__()
        self.encoding = encoding
        d0 = encoding.dim
        gen = torch.Generator().manual_seed(seed)

        def linear(d_in, d_out, g):
            # device="cpu" is load-bearing: deepxde's import sets the default
            # tensor type to CUDA when available, but the seeded generator is a
            # CPU generator — init on CPU, the caller moves the module later.
            lin = nn.Linear(d_in, d_out, device="cpu")
            # xavier_init of the reference: std = 1/sqrt((d_in+d_out)/2), zero bias
            std = 1.0 / np.sqrt((d_in + d_out) / 2.0)
            with torch.no_grad():
                lin.weight.normal_(0.0, std, generator=g)
                lin.bias.zero_()
            return lin

        # gate encoders use fixed seeds in the reference (PRNGKey 12345/54321)
        self.gate_u = linear(d0, hidden[0], torch.Generator().manual_seed(gate_seeds[0]))
        self.gate_v = linear(d0, hidden[0], torch.Generator().manual_seed(gate_seeds[1]))
        dims = [d0] + list(hidden)
        self.layers = nn.ModuleList([linear(dims[i], dims[i + 1], gen) for i in range(len(hidden))])
        self.out = linear(dims[-1], out_dim, gen)

    def forward(self, t, *coords):
        h = self.encoding(t, *coords)
        U = torch.tanh(self.gate_u(h))
        V = torch.tanh(self.gate_v(h))
        for lay in self.layers:
            z = torch.tanh(lay(h))
            h = z * U + (1 - z) * V
        return self.out(h)
