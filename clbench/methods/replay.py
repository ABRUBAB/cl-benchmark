"""Replay-based methods: Experience Replay with a reservoir buffer and Deep
Generative Replay (Shin et al., 2017)."""

from __future__ import annotations

import copy

import torch

from ..models import ConvVAE, MLPVAE
from .base import Method, NEG, masked_ce, tensors_bytes


class Reservoir:
    """Fixed-size buffer filled by reservoir sampling (Vitter, 1985), so every
    sample seen so far has the same probability M / n_seen of being stored."""

    def __init__(self, size: int, input_shape, device, gen: torch.Generator | None = None):
        self.size = size
        # one extra "discard" slot so that rejected samples can be written without a host sync
        self.x = torch.zeros((size + 1, *input_shape), device=device)
        self.y = torch.zeros(size + 1, dtype=torch.long, device=device)
        self.n_seen = 0
        self.device = torch.device(device)

    def __len__(self):
        return min(self.n_seen, self.size)

    def add(self, x, y):
        B = x.shape[0]
        n = self.n_seen + torch.arange(B, device=self.device)  # index of each sample in the stream
        j = (torch.rand(B, device=self.device) * (n + 1).float()).long()  # uniform in [0, n]
        j = torch.where(n < self.size, n, j)
        j = torch.where(j < self.size, j, torch.full_like(j, self.size))  # rejected -> discard slot
        self.x[j] = x.detach()
        self.y[j] = y
        self.n_seen += B

    def sample(self, n: int):
        idx = torch.randint(0, len(self), (n,), device=self.device)
        return self.x[idx], self.y[idx]

    def nbytes(self):
        return self.size * (self.x[0].numel() * self.x.element_size() + self.y.element_size())


class ER(Method):
    """Experience Replay: L = L_new(B_t) + lambda_r * L_replay(B_M), with B_M
    drawn uniformly from a reservoir buffer of M samples (Chaudhry et al., 2019;
    Rolnick et al., 2019)."""

    name = "er"
    stores_data = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        self.buffer = Reservoir(cfg["buffer_size"], bench.input_shape, device, self.gen)

    def task_loss(self, x, y, t):
        loss = masked_ce(self.logits(x, t), self.bench.targets(y), self.train_mask(t))
        if len(self.buffer) > 0:
            xr, yr = self.buffer.sample(self.cfg["batch_size"])
            lr = masked_ce(self.logits(xr, t), self.bench.targets(yr), self.replay_mask(yr, t))
            loss = loss + self.cfg.get("replay_weight", 1.0) * lr
        return loss

    def after_step(self, x, y, t):
        self.buffer.add(x, y)

    def extra_state_bytes(self):
        return self.buffer.nbytes()


class DGR(Method):
    """Deep Generative Replay with a VAE generator.

    Solver loss at task t (Shin et al., 2017, Eq. 2), with r = 1/t:
        r * E_{(x,y)~D_t} CE(S(x), y) + (1-r) * E_{x'~G_{t-1}} CE(S(x'), argmax S_{t-1}(x'))
    The generator is trained on the same mixture of real (current-task) and
    generated (previous-task) inputs.
    """

    name = "dgr"
    needs_boundaries = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        if cfg["arch"] == "cnn":
            self.vae = ConvVAE(bench.input_shape[0], cfg.get("z_dim", 128)).to(device)
        else:
            self.vae = MLPVAE(bench.input_shape[0], cfg.get("vae_hidden", 400), cfg.get("z_dim", 32)).to(device)
        self.prev_solver = None
        self.prev_vae = None
        self.vae_opt = None

    def begin_task(self, t, task):
        if t > 0:
            self.prev_solver = copy.deepcopy(self.model).eval()
            self.prev_vae = copy.deepcopy(self.vae).eval()
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.cfg.get("vae_lr", 1e-3))

    def _replay_loss(self, xr, t):
        b = self.bench
        logits = self.logits(xr, t)
        with torch.no_grad():
            old = self.prev_solver(xr)
        if b.scenario == "task":
            loss = 0.0
            for s in range(t):
                m = self.task_masks[s]
                yr = old.masked_fill(~m.unsqueeze(0), NEG).argmax(1)
                loss = loss + masked_ce(logits, yr, m)
            return loss / t
        if b.scenario == "domain":
            return masked_ce(logits, old.argmax(1), self.train_mask(t))
        prev_units = b.unit_mask(b.active_units_train(t - 1), self.device)
        yr = old.masked_fill(~prev_units.unsqueeze(0), NEG).argmax(1)
        return masked_ce(logits, yr, self.train_mask(t))

    def observe(self, x, y, t):
        self.model.train()
        r = 1.0 / (t + 1)
        loss = masked_ce(self.logits(x, t), self.bench.targets(y), self.train_mask(t))
        xr = None
        if t > 0:
            xr = self.prev_vae.sample(x.shape[0], self.device)
            loss = r * loss + (1 - r) * self._replay_loss(xr, t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        # generator update on the same real/generated mixture
        self.vae.train()
        gl = self.vae.loss(x)
        if xr is not None:
            gl = r * gl + (1 - r) * self.vae.loss(xr)
        self.vae_opt.zero_grad(set_to_none=True)
        gl.backward()
        self.vae_opt.step()


    def extra_state_bytes(self):
        # generator + copies of the previous solver and generator
        return tensors_bytes(self.vae, self.prev_solver, self.prev_vae)
