"""Meta-Experience Replay (Riemer et al., ICLR 2019), mini-batch form.

MER approximates the objective
    min_theta E_{(x_i,y_i),(x_j,y_j) ~ M} [ L(x_i,y_i) + L(x_j,y_j)
                                           - alpha * dL(x_i,y_i)/dtheta . dL(x_j,y_j)/dtheta ]
(i.e. it rewards gradient alignment between samples) with Reptile-style
updates.  Each incoming mini-batch is combined with a replay mini-batch drawn
from a reservoir buffer; the union is split into k sub-batches on which plain
SGD steps (rate alpha) are taken sequentially, followed by the within-batch
Reptile interpolation theta <- theta_0 + beta (theta - theta_0).  With s > 1
batches an across-batch interpolation with gamma is applied as in Algorithm 1
of Riemer et al.  The buffer is then updated by reservoir sampling.
"""

from __future__ import annotations

import torch

from .base import Method, masked_ce
from .replay import Reservoir


class MER(Method):
    name = "mer"
    optimizer = "sgd"
    stores_data = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        self.buffer = Reservoir(cfg["buffer_size"], bench.input_shape, device, self.gen)

    def _params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @torch.no_grad()
    def _snapshot(self):
        return [p.detach().clone() for p in self._params()]

    @torch.no_grad()
    def _interpolate(self, start, coef):
        for p, s in zip(self._params(), start):
            p.copy_(s + coef * (p - s))

    def observe(self, x, y, t):
        self.model.train()
        k = self.cfg.get("mer_k", 5)
        S = self.cfg.get("mer_s", 1)
        beta, gamma = self.cfg["mer_beta"], self.cfg.get("mer_gamma", 1.0)
        theta_a0 = self._snapshot() if S > 1 else None
        for _ in range(S):
            if len(self.buffer) > 0:
                xr, yr = self.buffer.sample(self.cfg["batch_size"])
                xb, yb = torch.cat([x, xr]), torch.cat([y, yr])
            else:
                xb, yb = x, y
            perm = torch.randperm(xb.shape[0], device=self.device)
            xb, yb = xb[perm], yb[perm]
            mask = self.replay_mask(yb, t)
            theta_w0 = self._snapshot()
            for idx in torch.chunk(torch.arange(xb.shape[0], device=self.device), k):
                loss = masked_ce(self.logits(xb[idx], t), self.bench.targets(yb[idx]), mask[idx] if mask.dim() == 2 else mask)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.clip()
                self.opt.step()
            self._interpolate(theta_w0, beta)
        if S > 1:
            self._interpolate(theta_a0, gamma)
        self.buffer.add(x, y)

    def extra_state_bytes(self):
        return self.buffer.nbytes()
