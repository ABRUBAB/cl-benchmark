"""Joint (non-sequential) training on all tasks: the usual upper bound.

The same total number of optimisation steps as the sequential methods is used
(T x iters).  In Task-IL every sample keeps its own task head; in Domain-IL and
Class-IL all output units are active.
"""

from __future__ import annotations

from ..data import ConcatSampler
from .base import Method, masked_ce


class Joint(Method):
    name = "joint"

    def train_all(self):
        b = self.bench
        sampler = ConcatSampler([t.train for t in b.tasks], self.cfg["batch_size"], self.gen)
        self.opt = self.make_optimizer()
        full = self.train_mask(b.n_tasks - 1)
        for _ in range(self.cfg["iters"] * b.n_tasks):
            xb, yb = sampler.next()
            mask = self.task_masks[self.task_of_class[yb]] if b.scenario == "task" else full
            loss = masked_ce(self.logits(xb), b.targets(yb), mask)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
