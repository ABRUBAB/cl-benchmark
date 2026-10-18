"""Architecture-based method: Progressive Neural Networks (Rusu et al., 2016)."""

from __future__ import annotations

from ..models import PNNConvNet, PNNMLP
from .base import Method, n_params_bytes


class PNN(Method):
    """A new column is added for every task; earlier columns are frozen and
    feed the new column through lateral connections (Rusu et al., Eq. 1).
    The column of task i is used to predict task i, so task identity is
    required at test time (Task-IL only)."""

    name = "pnn"
    needs_task_id = True
    needs_boundaries = True

    def build_model(self):
        if self.bench.scenario != "task":
            raise ValueError("PNN needs task identity at test time (Task-IL only)")
        if self.cfg["arch"] == "cnn":
            return PNNConvNet(self.bench.n_outputs, self.bench.input_shape[0], self.cfg.get("fc", 256))
        return PNNMLP(self.bench.input_shape[0], self.cfg["hidden"], self.bench.n_outputs)

    def begin_task(self, t, task):
        self.model.add_column()
        self.model.to(self.device)

    def logits(self, x, t=None):
        t = len(self.model.columns) - 1 if t is None else min(t, len(self.model.columns) - 1)
        return self.model.forward_column(x, t)

    def extra_state_bytes(self):
        # parameters beyond a single column (frozen columns + laterals)
        total = n_params_bytes(self.model)
        first = n_params_bytes(self.model.columns[0]) if len(self.model.columns) else 0
        return total - first
