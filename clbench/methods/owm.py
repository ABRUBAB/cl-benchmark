"""Orthogonal Weight Modification (Zeng, Chen, Cui & Yu, Nature Machine
Intelligence 1, 364-372, 2019).

For every layer l a projector P_l onto the orthogonal complement of the space
spanned by the layer's past *inputs* is maintained with the recursive
least-squares update of the original paper / code:

    k   = P x̄ / (alpha + x̄^T P x̄)
    P  <- P - k x̄^T P                       (x̄: mean input of the mini-batch)

and the back-propagated weight update is projected before the SGD step:

    dW_l <- dW_l P_l      (W_l stored as out x in, bias appended as an input of 1)

For convolutions, the update is applied with the batch-mean input patch at
every second spatial position, as in the original code (OWM_CNN/owm.py,
pro_weight); the sequence of rank-1 updates is computed as one equivalent block
update.  Plain SGD is used because adaptive optimisers would undo the
projection, and gradients are clipped to norm 10 as in the original code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, tensors_bytes


class OWM(Method):
    name = "owm"
    optimizer = "sgd"

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        self.layers = [m for m in self.model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        self.P = {}
        self._inputs = {}
        for m in self.layers:
            n_in = (m.in_features if isinstance(m, nn.Linear) else m.in_channels * m.kernel_size[0] * m.kernel_size[1]) + 1
            self.P[m] = torch.eye(n_in, device=device)
            m.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if module.training:
            self._inputs[module] = inp[0].detach()

    def _input_vectors(self, m):
        """Columns = input vectors used to update the projector of layer m.

        Linear layers: the mean input of the mini-batch (one vector).
        Convolutions: as in the original OWM code, the batch-mean input patch at
        every second spatial position (stride 2), one vector per position.
        A constant 1 is appended for the bias."""
        x = self._inputs[m]
        if isinstance(m, nn.Conv2d):
            xm = x.mean(0, keepdim=True)
            R = F.unfold(xm, m.kernel_size, padding=m.padding, stride=2)[0]  # (C k k) x L
        else:
            R = x.reshape(-1, x.shape[-1]).mean(0).unsqueeze(1)
        return torch.cat([R, R.new_ones(1, R.shape[1])], dim=0)  # (n_in+1) x L

    @torch.no_grad()
    def before_step(self, t):
        alpha = self.cfg["owm_alpha"]
        for m in self.layers:
            if m not in self._inputs or m.weight.grad is None:
                continue
            P = self.P[m]
            R = self._input_vectors(m)
            # Block form of the sequential rank-1 updates P <- P - k k^T / (alpha + r^T k), one per
            # column r of R: identical result (P^-1 <- P^-1 + R R^T / alpha), computed in one solve.
            K = P @ R
            S = alpha * torch.eye(R.shape[1], device=R.device) + R.T @ K
            S = 0.5 * (S + S.T) + 1e-6 * torch.eye(R.shape[1], device=R.device)
            try:
                solve_kt = torch.linalg.solve(S, K.T)
            except Exception:
                solve_kt = torch.linalg.lstsq(S, K.T).solution
            P.sub_(K @ solve_kt)
            P.copy_(0.5 * (P + P.T))
            g = m.weight.grad.reshape(m.weight.shape[0], -1)
            gb = m.bias.grad.unsqueeze(1) if m.bias is not None and m.bias.grad is not None else torch.zeros_like(g[:, :1])
            G = torch.cat([g, gb], dim=1) @ P  # P is symmetric
            m.weight.grad.copy_(G[:, :-1].reshape_as(m.weight))
            if m.bias is not None and m.bias.grad is not None:
                m.bias.grad.copy_(G[:, -1])
        self._inputs.clear()

    def extra_state_bytes(self):
        return tensors_bytes(list(self.P.values()))
