"""Benchmarks and continual-learning scenarios.

Three benchmarks are provided:

* ``splitmnist``    -- MNIST split into 5 tasks of 2 digits each (0/1, 2/3, ...).
* ``permmnist``     -- 10 tasks, each a fixed random pixel permutation of MNIST.
* ``splitcifar100`` -- CIFAR-100 split into 10 tasks of 10 classes each.

and three scenarios (van de Ven, Tuytelaars & Tolias, 2022):

* ``task``   (Task-IL)   -- task identity is given at test time (multi-head).
* ``domain`` (Domain-IL) -- no task identity; the output space is shared
  (e.g. "first or second digit of the pair").
* ``class``  (Class-IL)  -- no task identity; the model must discriminate all
  classes seen so far.

Images are kept on the target device as uint8 and converted (and, for
Permuted MNIST, permuted) on the fly, which removes data-loader overhead and
keeps the memory footprint small enough to run several experiments in parallel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SCENARIOS = ("task", "domain", "class")


class ImageView:
    """Lazily converts stored uint8 images into normalised float inputs.

    ``view[idx]`` returns ``((base[idx] / 255) - mean) / std`` (flattened for
    MLPs), optionally followed by a fixed pixel permutation.
    """

    def __init__(self, base: torch.Tensor, flatten: bool, mean=None, std=None, perm=None):
        self.base, self.flatten = base, flatten
        self.mean, self.std, self.perm = mean, std, perm

    def __len__(self):
        return self.base.shape[0]

    def __getitem__(self, idx):
        x = self.base[idx].float().div_(255.0)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        if self.perm is not None:
            x = x[:, self.perm]
        return x

    def to(self, device):
        return ImageView(
            self.base.to(device),
            self.flatten,
            None if self.mean is None else self.mean.to(device),
            None if self.std is None else self.std.to(device),
            None if self.perm is None else self.perm.to(device),
        )

    def subset(self, idx):
        return ImageView(self.base[idx], self.flatten, self.mean, self.std, self.perm)


@dataclass
class Split:
    x: ImageView  # indexable -> float inputs
    y: torch.Tensor  # global class labels (long)

    def __len__(self) -> int:
        return self.y.shape[0]


@dataclass
class Task:
    index: int
    classes: list[int]  # global class ids belonging to this task
    train: Split
    val: Split
    test: Split


@dataclass
class Benchmark:
    name: str
    scenario: str
    tasks: list[Task]
    input_shape: tuple[int, ...]
    classes_per_task: int
    n_global_classes: int
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ outputs
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_outputs(self) -> int:
        """Size of the output layer of a single (non-progressive) network."""
        if self.scenario == "domain":
            return self.classes_per_task
        return self.n_global_classes

    def targets(self, y_global: torch.Tensor) -> torch.Tensor:
        """Map global labels to output-unit indices for this scenario."""
        if self.scenario == "domain":
            return y_global % self.classes_per_task
        return y_global

    def task_units(self, t: int) -> list[int]:
        """Output units that belong to task ``t`` (Task-IL heads)."""
        if self.scenario == "domain":
            return list(range(self.classes_per_task))
        return list(self.tasks[t].classes)

    def active_units_train(self, t: int) -> list[int]:
        """Output units that are active while training on task ``t``."""
        if self.scenario == "task":
            return self.task_units(t)
        if self.scenario == "domain":
            return list(range(self.classes_per_task))
        units: list[int] = []
        for s in range(t + 1):
            units += self.tasks[s].classes
        return units

    def eval_units(self, i: int, t: int) -> list[int]:
        """Output units used to evaluate task ``i`` after training up to task ``t``."""
        if self.scenario == "task":
            return self.task_units(i)
        if self.scenario == "domain":
            return list(range(self.classes_per_task))
        return self.active_units_train(max(i, t))

    def unit_mask(self, units: list[int], device) -> torch.Tensor:
        key = (tuple(units), str(device))
        cache = self.meta.setdefault("_mask_cache", {})
        if key not in cache:  # cached: building it every step would force a host-device sync
            m = torch.zeros(self.n_outputs, dtype=torch.bool, device=device)
            m[torch.as_tensor(units, device=device)] = True
            cache[key] = m
        return cache[key]


# ---------------------------------------------------------------------- loading
def _torchvision():
    import torchvision  # imported lazily so that CPU-only tools can import this module

    return torchvision


def load_mnist(root: str | Path):
    tv = _torchvision()
    tr = tv.datasets.MNIST(root, train=True, download=True)
    te = tv.datasets.MNIST(root, train=False, download=True)
    return (tr.data.clone(), tr.targets.clone()), (te.data.clone(), te.targets.clone())


def load_cifar100(root: str | Path):
    tv = _torchvision()
    tr = tv.datasets.CIFAR100(root, train=True, download=True)
    te = tv.datasets.CIFAR100(root, train=False, download=True)
    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
    xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
    return (xtr, torch.as_tensor(tr.targets)), (xte, torch.as_tensor(te.targets))


def _train_val_split(n: int, val_frac: float, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    n_val = int(round(val_frac * n))
    return perm[n_val:], perm[:n_val]


def _split(view: ImageView, y: torch.Tensor, idx, device) -> Split:
    return Split(view.subset(idx).to(device), y[idx].to(device))


# ------------------------------------------------------------------- benchmarks
def make_benchmark(
    name: str,
    scenario: str,
    seed: int,
    root: str | Path = "data",
    device: str | torch.device = "cpu",
    val_frac: float = 0.1,
    n_tasks: int | None = None,
) -> Benchmark:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    gen = torch.Generator().manual_seed(10_000 + seed)
    device = torch.device(device)

    if name == "splitmnist":
        (xtr, ytr), (xte, yte) = load_mnist(root)
        T, cpt = n_tasks or 5, 2
        tasks = []
        for t in range(T):
            cls = [2 * t, 2 * t + 1]
            mtr = torch.nonzero((ytr == cls[0]) | (ytr == cls[1])).squeeze(1)
            mte = torch.nonzero((yte == cls[0]) | (yte == cls[1])).squeeze(1)
            tr_idx, va_idx = _train_val_split(len(mtr), val_frac, gen)
            vtr, vte = ImageView(xtr, True), ImageView(xte, True)
            tasks.append(
                Task(t, cls, _split(vtr, ytr, mtr[tr_idx], device), _split(vtr, ytr, mtr[va_idx], device),
                     _split(vte, yte, mte, device))
            )
        return Benchmark(name, scenario, tasks, (784,), cpt, cpt * T)

    if name == "permmnist":
        (xtr, ytr), (xte, yte) = load_mnist(root)
        T, cpt = n_tasks or 10, 10
        tr_idx, va_idx = _train_val_split(len(ytr), val_frac, gen)
        base_tr = ImageView(xtr[tr_idx], True).to(device)
        base_va = ImageView(xtr[va_idx], True).to(device)
        base_te = ImageView(xte, True).to(device)
        ytr_tr, ytr_va, yte_d = ytr[tr_idx].to(device), ytr[va_idx].to(device), yte.to(device)
        tasks = []
        for t in range(T):
            perm = torch.randperm(784, generator=gen).to(device)
            cls = list(range(10 * t, 10 * t + 10))
            views = [ImageView(b.base, True, perm=perm) for b in (base_tr, base_va, base_te)]
            tasks.append(
                Task(t, cls, Split(views[0], ytr_tr + 10 * t), Split(views[1], ytr_va + 10 * t),
                     Split(views[2], yte_d + 10 * t))
            )
        return Benchmark(name, scenario, tasks, (784,), cpt, cpt * T)

    if name == "splitcifar100":
        (xtr, ytr), (xte, yte) = load_cifar100(root)
        mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
        std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
        T = n_tasks or 10
        cpt = 100 // T
        vtr, vte = ImageView(xtr, False, mean, std), ImageView(xte, False, mean, std)
        tasks = []
        for t in range(T):
            cls = list(range(cpt * t, cpt * (t + 1)))
            mtr = torch.nonzero((ytr >= cls[0]) & (ytr <= cls[-1])).squeeze(1)
            mte = torch.nonzero((yte >= cls[0]) & (yte <= cls[-1])).squeeze(1)
            tr_idx, va_idx = _train_val_split(len(mtr), val_frac, gen)
            tasks.append(
                Task(t, cls, _split(vtr, ytr, mtr[tr_idx], device), _split(vtr, ytr, mtr[va_idx], device),
                     _split(vte, yte, mte, device))
            )
        return Benchmark(name, scenario, tasks, (3, 32, 32), cpt, cpt * T)

    raise ValueError(f"unknown benchmark {name!r}")


class BatchSampler:
    """Infinite sampler over a split that reshuffles every epoch."""

    def __init__(self, split: Split, batch_size: int, gen: torch.Generator):
        self.split = split
        self.bs = batch_size
        self.gen = gen
        self._perm = None
        self._pos = 0

    def next(self):
        n = len(self.split)
        if self._perm is None or self._pos + self.bs > n:
            self._perm = torch.randperm(n, generator=self.gen).to(self.split.y.device)
            self._pos = 0
        idx = self._perm[self._pos : self._pos + self.bs]
        self._pos += self.bs
        return self.split.x[idx], self.split.y[idx]


class ConcatSampler:
    """Uniform mini-batches over the union of several splits (joint training)."""

    def __init__(self, splits: list[Split], batch_size: int, gen: torch.Generator):
        self.splits, self.bs, self.gen = splits, batch_size, gen
        self.sizes = torch.tensor([len(s) for s in splits])
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), self.sizes.cumsum(0)])
        self.total = int(self.sizes.sum())

    def next(self):
        idx = torch.randint(0, self.total, (self.bs,), generator=self.gen)
        seg = torch.bucketize(idx, self.offsets[1:], right=True)
        xs, ys = [], []
        for k in seg.unique().tolist():
            local = (idx[seg == k] - self.offsets[k]).to(self.splits[k].y.device)
            xs.append(self.splits[k].x[local])
            ys.append(self.splits[k].y[local])
        return torch.cat(xs), torch.cat(ys)


def steps_per_epoch(n: int, bs: int) -> int:
    return max(1, math.ceil(n / bs))


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
