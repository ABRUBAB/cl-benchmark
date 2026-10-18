"""Backbones shared by all methods.

* ``MLP``        -- fully connected ReLU network (MNIST benchmarks).
* ``ConvNet``    -- 4 conv + 1 hidden FC layer, no batch-norm (CIFAR-100).
* ``SpikingMLP`` -- leaky integrate-and-fire MLP trained with surrogate
  gradients; same layer sizes as ``MLP`` so that every continual-learning
  method can be plugged in unchanged.
* ``MLPVAE`` / ``ConvVAE`` -- generators for deep generative replay.
* ``PNNMLP`` / ``PNNConvNet`` -- progressive networks (one column per task).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- feed-forward
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], n_out: int):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(dims[-1], n_out)

    def features(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


class ConvNet(nn.Module):
    """conv32-conv32-pool-conv64-conv64-pool-fc256-head (no batch-norm)."""

    def __init__(self, n_out: int, in_ch: int = 3, fc: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, fc)
        self.head = nn.Linear(fc, n_out)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


# ------------------------------------------------------------------- spiking
class _SpikeFn(torch.autograd.Function):
    """Heaviside spike with fast-sigmoid surrogate gradient (Zenke & Ganguli, 2018).

    Written with a separate ``setup_context`` so that it also works under
    ``torch.func`` transforms (used for per-sample Fisher information)."""

    slope = 25.0
    generate_vmap_rule = True

    @staticmethod
    def forward(v):
        return (v > 0).to(v.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + _SpikeFn.slope * v.abs()) ** 2


spike = _SpikeFn.apply


class SpikingMLP(nn.Module):
    """LIF network: direct (analogue) input current, soft reset, non-spiking readout.

    The logits are the time-average of the read-out currents over ``T`` steps.
    """

    def __init__(self, in_dim: int, hidden: list[int], n_out: int, T: int = 8, beta: float = 0.9, thr: float = 1.0):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(dims[-1], n_out)
        self.T, self.beta, self.thr = T, beta, thr

    def forward(self, x):
        mems = [torch.zeros(x.shape[0], l.out_features, device=x.device, dtype=x.dtype) for l in self.hidden]
        out = 0.0
        rate = 0.0
        for _ in range(self.T):
            h = x
            for k, layer in enumerate(self.hidden):
                mems[k] = self.beta * mems[k] + layer(h)
                s = spike(mems[k] - self.thr)
                mems[k] = mems[k] - s * self.thr
                h = s
                if not self.training:
                    rate = rate + s.mean()
            out = out + self.head(h)
        if not self.training:  # firing rate is only tracked at evaluation (avoids a sync per training step)
            self._rate = rate / (self.T * len(self.hidden))
        return out / self.T

    @property
    def last_rate(self):
        r = getattr(self, "_rate", None)
        return None if r is None else float(r)


# ---------------------------------------------------------------- generators
class MLPVAE(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 400, z_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar

    def loss(self, x):
        logits, mu, logvar = self(x)
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        return rec + kl

    @torch.no_grad()
    def sample(self, n: int, device):
        z = torch.randn(n, self.z_dim, device=device)
        return torch.sigmoid(self.dec(z))


class ConvVAE(nn.Module):
    def __init__(self, in_ch: int = 3, z_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.mu = nn.Linear(128 * 4 * 4, z_dim)
        self.logvar = nn.Linear(128 * 4 * 4, z_dim)
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1),
        )

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 128, 4, 4))

    def forward(self, x):
        h = self.enc(x).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    def loss(self, x):
        rec_x, mu, logvar = self(x)
        rec = F.mse_loss(rec_x, x, reduction="sum") / x.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        return rec + kl

    @torch.no_grad()
    def sample(self, n: int, device):
        return self.decode(torch.randn(n, self.z_dim, device=device))


# ----------------------------------------------------- progressive networks
class _MLPColumn(nn.Module):
    """One PNN column. Lateral connections feed layer j from layer j-1 of all
    earlier columns (Rusu et al., 2016, Eq. 1)."""

    def __init__(self, in_dim, hidden, n_out, n_prev):
        super().__init__()
        dims = [in_dim, *hidden]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(dims[-1], n_out)
        # laterals[j][k]: from column k's activation h_{j} (output of hidden layer j) into layer j+1 / head
        self.laterals = nn.ModuleList()
        outs = [*hidden[1:], n_out]
        for j in range(len(hidden)):
            self.laterals.append(nn.ModuleList(nn.Linear(hidden[j], outs[j], bias=False) for _ in range(n_prev)))

    def forward(self, x, prev_acts):
        acts = []
        h = F.relu(self.layers[0](x))
        acts.append(h)
        for j in range(1, len(self.layers) + 1):
            lin = self.layers[j] if j < len(self.layers) else self.head
            z = lin(h)
            for k, lat in enumerate(self.laterals[j - 1]):
                z = z + lat(prev_acts[k][j - 1])
            if j < len(self.layers):
                h = F.relu(z)
                acts.append(h)
            else:
                return z, acts


class PNNMLP(nn.Module):
    def __init__(self, in_dim, hidden, n_out):
        super().__init__()
        self.in_dim, self.hidden, self.n_out = in_dim, hidden, n_out
        self.columns = nn.ModuleList()

    def add_column(self):
        for c in self.columns:
            for p in c.parameters():
                p.requires_grad_(False)
        self.columns.append(_MLPColumn(self.in_dim, self.hidden, self.n_out, len(self.columns)))

    def forward_column(self, x, t):
        prev = []
        for k in range(t + 1):
            out, acts = self.columns[k](x, prev)
            prev.append(acts)
        return out


class _ConvColumn(nn.Module):
    def __init__(self, n_out, in_ch, fc, n_prev, red=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, fc)
        self.head = nn.Linear(fc, n_out)
        # lateral adapters (1x1 convs; the flattened conv features are first
        # reduced to `red` channels as in the adapters of Rusu et al.)
        self.l2 = nn.ModuleList(nn.Conv2d(32, 32, 1, bias=False) for _ in range(n_prev))
        self.l3 = nn.ModuleList(nn.Conv2d(32, 64, 1, bias=False) for _ in range(n_prev))
        self.l4 = nn.ModuleList(nn.Conv2d(64, 64, 1, bias=False) for _ in range(n_prev))
        self.lfc_red = nn.ModuleList(nn.Conv2d(64, red, 1, bias=False) for _ in range(n_prev))
        self.lfc = nn.ModuleList(nn.Linear(red * 8 * 8, fc, bias=False) for _ in range(n_prev))
        self.lhead = nn.ModuleList(nn.Linear(fc, n_out, bias=False) for _ in range(n_prev))

    def forward(self, x, prev):
        a1 = F.relu(self.conv1(x))
        z = self.conv2(a1) + sum((l(p[0]) for l, p in zip(self.l2, prev)), 0)
        a2 = F.max_pool2d(F.relu(z), 2)
        z = self.conv3(a2) + sum((l(p[1]) for l, p in zip(self.l3, prev)), 0)
        a3 = F.relu(z)
        z = self.conv4(a3) + sum((l(p[2]) for l, p in zip(self.l4, prev)), 0)
        a4 = F.max_pool2d(F.relu(z), 2)
        z = self.fc(a4.flatten(1)) + sum(
            (lf(F.relu(lr(p[3])).flatten(1)) for lr, lf, p in zip(self.lfc_red, self.lfc, prev)), 0
        )
        a5 = F.relu(z)
        out = self.head(a5) + sum((l(p[4]) for l, p in zip(self.lhead, prev)), 0)
        return out, [a1, a2, a3, a4, a5]


class PNNConvNet(nn.Module):
    def __init__(self, n_out, in_ch=3, fc=256):
        super().__init__()
        self.n_out, self.in_ch, self.fc = n_out, in_ch, fc
        self.columns = nn.ModuleList()

    def add_column(self):
        for c in self.columns:
            for p in c.parameters():
                p.requires_grad_(False)
        self.columns.append(_ConvColumn(self.n_out, self.in_ch, self.fc, len(self.columns)))

    def forward_column(self, x, t):
        prev = []
        for k in range(t + 1):
            out, acts = self.columns[k](x, prev)
            prev.append(acts)
        return out


# ------------------------------------------------------------------- factory
def build_backbone(arch: str, bench, cfg: dict) -> nn.Module:
    n_out = bench.n_outputs
    if arch == "mlp":
        return MLP(bench.input_shape[0], cfg["hidden"], n_out)
    if arch == "snn":
        return SpikingMLP(bench.input_shape[0], cfg["hidden"], n_out, T=cfg.get("snn_T", 8), beta=cfg.get("snn_beta", 0.9))
    if arch == "cnn":
        return ConvNet(n_out, in_ch=bench.input_shape[0], fc=cfg.get("fc", 256))
    raise ValueError(arch)


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
