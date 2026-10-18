"""Learning without Forgetting (Li & Hoiem, 2017)."""

from __future__ import annotations

import copy

import torch

from .base import Method, kd_loss, masked_ce, tensors_bytes


class LwF(Method):
    """At the start of task t a frozen copy of the network is kept.  While
    learning task t, its responses on the *new* inputs serve as soft targets
    for the previously learned outputs (temperature T = 2, weight lambda_o):

        L = CE(new task) + lambda_o * sum_{old heads} T^2 KL(p_old^T || p_new^T)

    Old heads are the previous task heads (Task-IL), the shared head
    (Domain-IL) or the units of previously seen classes (Class-IL).
    """

    name = "lwf"
    needs_boundaries = True

    def __init__(self, bench, cfg, device):
        super().__init__(bench, cfg, device)
        self.old = None

    def begin_task(self, t, task):
        if t > 0:
            self.old = copy.deepcopy(self.model).eval()
            for p in self.old.parameters():
                p.requires_grad_(False)
        self._groups = [torch.as_tensor(u, device=self.device) for u in self._old_unit_groups(t)]

    def _old_unit_groups(self, t):
        b = self.bench
        if b.scenario == "task":
            return [b.task_units(s) for s in range(t)]
        if b.scenario == "domain":
            return [list(range(b.classes_per_task))]
        units = []
        for s in range(t):
            units += b.tasks[s].classes
        return [units]

    def task_loss(self, x, y, t):
        logits = self.logits(x, t)
        loss = masked_ce(logits, self.bench.targets(y), self.train_mask(t))
        if t > 0:
            with torch.no_grad():
                old_logits = self.old(x)
            T = self.cfg.get("lwf_T", 2.0)
            kd = 0.0
            for u in self._groups:
                kd = kd + kd_loss(logits[:, u], old_logits[:, u], T)
            loss = loss + self.cfg.get("lwf_lambda", 1.0) * kd
        return loss

    def extra_state_bytes(self):
        return tensors_bytes(self.old)
