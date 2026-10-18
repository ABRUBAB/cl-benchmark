"""Regularisation-based methods: EWC (Kirkpatrick et al., 2017) and
Synaptic Intelligence (Zenke, Poole & Ganguli, 2017)."""

from __future__ import annotations

import torch

from .base import Method, NEG, tensors_bytes


class EWC(Method):
    """Elastic Weight Consolidation.

    After task k the diagonal Fisher information is estimated at the solution
    theta*_k with labels sampled from the model's own predictive distribution
    (the "true" Fisher of Kirkpatrick et al.).  While training task t the loss is

        L_t(theta) + sum_{k<t} lambda/2 * sum_i F_{k,i} (theta_i - theta*_{k,i})^2

    i.e. one quadratic penalty per previous task, as in the original paper.
    """

    name = "ewc"
    needs_boundaries = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        self.anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    def _params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def end_task(self, t, task):
        """Diagonal Fisher F_i = E_x E_{y~p(y|x)} [(d log p(y|x) / d theta_i)^2],
        computed from exact per-sample gradients (torch.func.vmap)."""
        from torch.func import functional_call, grad, vmap

        self.model.eval()
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        params = {n: p.detach() for n, p in self.model.named_parameters() if p.requires_grad}
        frozen = {n: p.detach() for n, p in self.model.named_parameters() if not p.requires_grad}
        buffers = dict(self.model.named_buffers())
        n = min(self.cfg.get("n_fisher", 1000), len(task.train))
        idx = torch.randperm(len(task.train), generator=self.gen)[:n].tolist()
        mask = self.train_mask(t).unsqueeze(0)
        x_all = task.train.x[idx]
        with torch.no_grad():  # labels sampled from the model's predictive distribution
            probs = torch.softmax(self.logits(x_all, t).masked_fill(~mask, NEG), dim=1)
            y_all = torch.multinomial(probs, 1).squeeze(1)

        def nll(p, x, y):
            out = functional_call(self.model, ({**p, **frozen}, buffers), (x.unsqueeze(0),))
            logp = torch.log_softmax(out.masked_fill(~mask, NEG), dim=1)
            return -logp[0].gather(0, y.unsqueeze(0)).squeeze(0)

        per_sample = vmap(grad(nll), in_dims=(None, 0, 0))
        fisher = {k: torch.zeros_like(v) for k, v in params.items()}
        chunk = self.cfg.get("fisher_chunk", 32)
        for s in range(0, n, chunk):
            g = per_sample(params, x_all[s : s + chunk], y_all[s : s + chunk])
            for k in fisher:
                fisher[k] += (g[k] ** 2).sum(0)
        fisher = [fisher[k] / n for k in names]
        star = [params[k].clone() for k in names]
        self.anchors.append((fisher, star))
        self.model.zero_grad(set_to_none=True)

    def penalty(self, t):
        if not self.anchors:
            return 0.0
        lam = self.cfg["ewc_lambda"]
        loss = 0.0
        params = self._params()
        for fisher, star in self.anchors:
            for f, s, p in zip(fisher, star, params):
                loss = loss + (f * (p - s) ** 2).sum()
        return 0.5 * lam * loss

    def extra_state_bytes(self):
        return tensors_bytes(self.anchors)


class SI(Method):
    """Synaptic Intelligence.

    Per-parameter path integral of the loss decrease (Zenke et al., Eq. 3-5):
        omega_i  <-  omega_i - g_i * delta_theta_i          (every step)
        Omega_i  <-  Omega_i + omega_i / (Delta_i^2 + xi)  (end of task)
    with Delta_i the total change of theta_i over the task.  Surrogate loss:
        L_t(theta) + c * sum_i Omega_i (theta_i - theta*_i)^2.
    """

    name = "si"
    needs_boundaries = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        ps = self._params()
        self.Omega = [torch.zeros_like(p) for p in ps]
        self.star = [p.detach().clone() for p in ps]
        self.omega = None
        self.start = None
        self._prev = None
        self._grad = None

    def _params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def begin_task(self, t, task):
        ps = self._params()
        self.omega = [torch.zeros_like(p) for p in ps]
        self.start = [p.detach().clone() for p in ps]

    def before_step(self, t):
        ps = self._params()
        self._prev = [p.detach().clone() for p in ps]
        self._grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ps]

    def after_step(self, x, y, t):
        for w, g, p, q in zip(self.omega, self._grad, self._params(), self._prev):
            w.add_(-g * (p.detach() - q))

    def end_task(self, t, task):
        xi = self.cfg.get("si_xi", 0.1)
        for O, w, p, s in zip(self.Omega, self.omega, self._params(), self.start):
            delta = p.detach() - s
            O.add_(w / (delta**2 + xi))
        self.star = [p.detach().clone() for p in self._params()]

    def penalty(self, t):
        if t == 0:
            return 0.0
        c = self.cfg["si_c"]
        loss = 0.0
        for O, s, p in zip(self.Omega, self.star, self._params()):
            loss = loss + (O * (p - s) ** 2).sum()
        return c * loss

    def extra_state_bytes(self):
        # Omega, theta*, running omega and theta at task start
        return tensors_bytes(self.Omega, self.star, self.omega, self.start)
