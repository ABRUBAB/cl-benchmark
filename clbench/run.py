"""Run one continual-learning experiment and write a JSON result file.

Example
-------
    python -m clbench.run --benchmark splitmnist --scenario class --method ewc --seed 1 \
        --set ewc_lambda=1e5 --out results/test
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .configs import make_config
from .data import make_benchmark, seed_everything
from .methods import METHODS, Method
from .metrics import compute_metrics
from .models import n_params

_BENCH_CACHE: dict = {}


def get_benchmark(cfg, root, device):
    key = (cfg["benchmark"], cfg["scenario"], cfg["seed"])
    if key not in _BENCH_CACHE:
        _BENCH_CACHE.clear()
        _BENCH_CACHE[key] = make_benchmark(cfg["benchmark"], cfg["scenario"], cfg["seed"], root=root, device=device)
    return _BENCH_CACHE[key]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(cfg: dict, root: str = "data", device: str | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bench = get_benchmark(cfg, root, device)
    split_name = cfg.get("eval_split", "test")
    T = bench.n_tasks

    # accuracy of the randomly initialised backbone (for forward transfer)
    seed_everything(cfg["seed"])
    base = Method(bench, {**cfg, "arch": cfg["arch"]}, device)
    b = [base.evaluate(getattr(bench.tasks[i], split_name), i, i) for i in range(T)]
    del base

    seed_everything(cfg["seed"])
    method = METHODS[cfg["method"]](bench, cfg, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    R = np.full((T, T), np.nan)
    if cfg["method"] == "joint":
        _sync(device)
        t0 = time.perf_counter()
        method.train_all()
        _sync(device)
        times.append(time.perf_counter() - t0)
        final = [method.evaluate(getattr(bench.tasks[i], split_name), i, T - 1) for i in range(T)]
        metrics = {"ACC": float(np.mean(final))}
        R[-1] = final
    else:
        for t, task in enumerate(bench.tasks):
            _sync(device)
            t0 = time.perf_counter()
            method.train_task(t, task)
            _sync(device)
            times.append(time.perf_counter() - t0)
            for i in range(T):
                R[t, i] = method.evaluate(getattr(bench.tasks[i], split_name), i, t)
        metrics = compute_metrics(R, np.array(b))

    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    result = {
        "config": cfg,
        "metrics": metrics,
        "R": R.tolist(),
        "b": b,
        "time_per_task_s": times,
        "train_time_s": float(sum(times)),
        "peak_gpu_mem_mb": peak,
        "extra_state_mb": method.extra_state_bytes() / 2**20,
        "n_params": n_params(method.model),
        "flags": {
            "needs_task_id": bool(getattr(method, "needs_task_id", False)),
            "needs_boundaries": bool(getattr(method, "needs_boundaries", False)),
            "stores_data": bool(getattr(method, "stores_data", False)),
        },
        "env": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "python": platform.python_version(),
        },
    }
    if hasattr(method.model, "last_rate") and method.model.last_rate is not None:
        result["snn_mean_firing_rate"] = method.model.last_rate
    return result


def result_path(out_dir: str | Path, cfg: dict, tag: str = "") -> Path:
    arch = cfg["arch"]
    name = cfg["method"] if arch != "snn" else f"snn-{cfg['method']}"
    return Path(out_dir) / cfg["benchmark"] / cfg["scenario"] / name / (tag or "default") / f"seed{cfg['seed']}.json"


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=["splitmnist", "permmnist", "splitcifar100"])
    ap.add_argument("--scenario", required=True, choices=["task", "domain", "class"])
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--arch", default=None, choices=[None, "mlp", "cnn", "snn"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-split", default="test", choices=["test", "val"])
    ap.add_argument("--set", nargs="*", default=[], help="override config entries, e.g. ewc_lambda=1e5")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/manual")
    args = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: _parse_value(v) for k, v in overrides.items()}
    cfg = make_config(args.benchmark, args.scenario, args.method, args.seed, args.arch, args.eval_split, overrides)
    res = run_experiment(cfg, root=args.data)
    path = result_path(args.out, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({"metrics": res["metrics"], "train_time_s": res["train_time_s"], "peak_gpu_mem_mb": res["peak_gpu_mem_mb"]}))


if __name__ == "__main__":
    main()
