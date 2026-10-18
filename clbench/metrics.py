"""Continual-learning metrics computed from the accuracy matrix.

``R[t, i]`` is the accuracy on task i after training on tasks 0..t
(Lopez-Paz & Ranzato, 2017); ``b[i]`` is the accuracy of the randomly
initialised network on task i.

ACC  = 1/T sum_i R[T-1, i]                                   (average accuracy)
AIA  = 1/T sum_t 1/(t+1) sum_{i<=t} R[t, i]                  (average incremental accuracy, Rebuffi et al. 2017)
BWT  = 1/(T-1) sum_{i<T-1} (R[T-1, i] - R[i, i])             (backward transfer)
FWT  = 1/(T-1) sum_{i>=1} (R[i-1, i] - b[i])                 (forward transfer)
FM   = 1/(T-1) sum_{i<T-1} (max_{i<=t<T-1} R[t, i] - R[T-1, i])   (forgetting, Chaudhry et al. 2018)
"""

from __future__ import annotations

import numpy as np


def compute_metrics(R: np.ndarray, b: np.ndarray | None = None) -> dict:
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    out = {"ACC": float(R[-1].mean())}
    out["AIA"] = float(np.mean([R[t, : t + 1].mean() for t in range(T)]))
    if T > 1:
        out["BWT"] = float(np.mean([R[-1, i] - R[i, i] for i in range(T - 1)]))
        out["FM"] = float(np.mean([R[i : T - 1, i].max() - R[-1, i] for i in range(T - 1)]))
        if b is not None:
            b = np.asarray(b, dtype=float)
            out["FWT"] = float(np.mean([R[i - 1, i] - b[i] for i in range(1, T)]))
    return out
