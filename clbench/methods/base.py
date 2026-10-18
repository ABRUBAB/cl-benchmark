"""Common training / evaluation logic shared by all continual-learning methods.

A method sees the tasks one after another. For each task it runs a fixed
number of optimisation steps (identical for all methods of a benchmark) and,
where the method requires it, performs extra work at the task boundary
(``end_task``). Task identity is used at test time only in the Task-IL
scenario, exactly as defined by van de Ven et al. (2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Benchmark, BatchSampler, Task
from ..models import build_backbone

NEG = -1e9


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy restricted to the output units selected by ``mask``.

    ``mask`` is either (n_out,) for a whole batch or (B, n_out) per sample.
    """
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return F.cross_entropy(logits.masked_fill(~mask, NEG), targets)


def kd_loss(new_logits: torch.Tensor, old_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Knowledge-distillation loss T^2 * KL(p_old^T || p_new^T) (Hinton et al., 2015)."""
    p_old = F.softmax(old_logits / T, dim=1)
    return F.kl_div(F.log_softmax(new_logits / T, dim=1), p_old, reduction="batchmean") * T * T


class Method:
    name = "finetune"
    optimizer = "adam"  # "adam" or "sgd"
    needs_task_id = False  # True if the method can only be used in Task-IL
    needs_boundaries = False  # True if the method acts at task boundaries
    stores_data = False  # True if raw past samples are stored

    def __init__(self, bench: Benchmark, cfg: dict, device: torch.device):
        self.bench, self.cfg, self.device = bench, cfg, device
        self.model = self.build_model().to(device)
        self.gen = torch.Generator().manual_seed(cfg["seed"])
        n_out = bench.n_outputs
        T = bench.n_tasks
        # lookup tables for per-sample Task-IL masks
        self.task_masks = torch.zeros(T, n_out, dtype=torch.bool, device=device)
        for t in range(T):
            self.task_masks[t, torch.as_tensor(bench.task_units(t), device=device)] = True
        self.task_of_class = torch.zeros(bench.n_global_classes, dtype=torch.long, device=device)
        for t, task in enumerate(bench.tasks):
            self.task_of_class[torch.as_tensor(task.classes, device=device)] = t
        self.opt = None

    # ------------------------------------------------------------------ hooks
    def build_model(self) -> nn.Module:
        return build_backbone(self.cfg["arch"], self.bench, self.cfg)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def make_optimizer(self):
        params = self.trainable_parameters()
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.cfg["lr"])
        return torch.optim.Adam(params, lr=self.cfg["lr"], betas=(0.9, 0.999))

    def begin_task(self, t: int, task: Task) -> None:
        pass

    def end_task(self, t: int, task: Task) -> None:
        pass

    def penalty(self, t: int) -> torch.Tensor | float:
        return 0.0

    def before_step(self, t: int) -> None:
        pass

    def after_step(self, x, y, t: int) -> None:
        pass

    def extra_state_bytes(self) -> int:
        """Bytes of persistent state kept by the method in addition to the model."""
        return 0

    # ---------------------------------------------------------------- helpers
    def logits(self, x: torch.Tensor, t: int | None = None) -> torch.Tensor:
        return self.model(x)

    def train_mask(self, t: int) -> torch.Tensor:
        return self.bench.unit_mask(self.bench.active_units_train(t), self.device)

    def replay_mask(self, y_global: torch.Tensor, t: int) -> torch.Tensor:
        """Output mask for replayed samples (per sample in Task-IL)."""
        if self.bench.scenario == "task":
            return self.task_masks[self.task_of_class[y_global]]
        return self.train_mask(t)

    def task_loss(self, x, y, t: int) -> torch.Tensor:
        return masked_ce(self.logits(x, t), self.bench.targets(y), self.train_mask(t))

    # --------------------------------------------------------------- training
    def observe(self, x, y, t: int) -> None:
        # no .item()/float() here: a host-device sync per step would dominate run time
        self.model.train()
        loss = self.task_loss(x, y, t) + self.penalty(t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.before_step(t)
        self.clip()
        self.opt.step()
        self.after_step(x, y, t)

    def clip(self) -> None:
        """Gradient-norm clipping, used by the plain-SGD methods (cfg clip_grad)."""
        c = self.cfg.get("clip_grad")
        if c:
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), c)

    def train_task(self, t: int, task: Task) -> None:
        self.begin_task(t, task)
        self.opt = self.make_optimizer()
        sampler = BatchSampler(task.train, self.cfg["batch_size"], self.gen)
        for _ in range(self.cfg["iters"]):
            x, y = sampler.next()
            self.observe(x, y, t)
        self.end_task(t, task)

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, split, i: int, t: int, bs: int | None = None) -> float:
        """Accuracy on task ``i`` after training up to task ``t``."""
        bs = bs or (250 if self.cfg["arch"] == "cnn" else 1000)
        self.model.eval()
        mask = self.bench.unit_mask(self.bench.eval_units(i, t), self.device)
        correct = 0
        for s in range(0, len(split), bs):
            x = split.x[s : s + bs]
            y = self.bench.targets(split.y[s : s + bs])
            out = self.logits(x, i).masked_fill(~mask.unsqueeze(0), NEG)
            correct += (out.argmax(1) == y).sum().item()
        return correct / len(split)


def n_params_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def tensors_bytes(*ts) -> int:
    n = 0
    for t in ts:
        if t is None:
            continue
        if isinstance(t, dict):
            n += tensors_bytes(*t.values())
        elif isinstance(t, (list, tuple)):
            n += tensors_bytes(*t)
        elif isinstance(t, nn.Module):
            n += sum(p.numel() * p.element_size() for p in t.parameters())
            n += sum(b.numel() * b.element_size() for b in t.buffers())
        elif torch.is_tensor(t):
            n += t.numel() * t.element_size()
    return n
