"""Experiment grid: hyper-parameter search, final runs and timing runs.

Phases
------
hp      seed 0, evaluated on the validation split, every combination of the
        grids in ``configs.HP_GRIDS``.
final   seeds 1-5 (``--seeds``), evaluated on the test split, with the
        hyper-parameters selected in the ``hp`` phase.
timing  seed 1, selected hyper-parameters, run with a single worker so that
        wall-clock times are not affected by other jobs on the same GPU.

Usage
-----
    python -m clbench.grid launch --phase hp --workers 4 --benchmarks splitmnist permmnist
    python -m clbench.grid select
    python -m clbench.grid launch --phase final --workers 4 --benchmarks splitmnist permmnist
    python -m clbench.grid launch --phase timing --workers 1 --benchmarks splitmnist permmnist
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .configs import HP_GRIDS, hp_grid, make_config, methods_for

SCENARIOS_FOR = {"splitmnist": ["task", "domain", "class"], "permmnist": ["domain"], "splitcifar100": ["task", "class"]}
ARCHS_FOR = {"splitmnist": ["mlp", "snn"], "permmnist": ["mlp", "snn"], "splitcifar100": ["cnn"]}


def hp_tag(over: dict) -> str:
    return "_".join(f"{k}={over[k]:g}" if isinstance(over[k], float) else f"{k}={over[k]}" for k in sorted(over)) or "default"


def run_name(method: str, arch: str) -> str:
    return f"snn-{method}" if arch == "snn" else method


def build_jobs(phase: str, benchmarks: list[str], seeds: list[int], out: Path) -> list[dict]:
    jobs = []
    selected = {}
    if phase in ("final", "timing"):
        sel_path = out / "selected_hparams.json"
        selected = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    for bench in benchmarks:
        for scenario in SCENARIOS_FOR[bench]:
            for arch in ARCHS_FOR[bench]:
                only = {m for m in os.environ.get("CLBENCH_METHODS", "").split(",") if m}  # optional filter
                for method in methods_for(scenario, arch):
                    if only and method not in only:
                        continue
                    name = run_name(method, arch)
                    if phase == "hp":
                        grid = hp_grid(method, arch)
                        if not grid:
                            continue
                        keys = sorted(grid)
                        for vals in itertools.product(*(grid[k] for k in keys)):
                            over = dict(zip(keys, vals))
                            jobs.append(dict(phase=phase, cfg=make_config(bench, scenario, method, 0, arch, "val", over),
                                             tag=hp_tag(over)))
                    else:
                        over = selected.get(f"{bench}/{scenario}/{name}", {})
                        for seed in seeds:
                            jobs.append(dict(phase=phase, cfg=make_config(bench, scenario, method, seed, arch, "test", over),
                                             tag=phase))
    # longest benchmarks first so that the tail of the schedule is short
    order = {"splitcifar100": 0, "permmnist": 1, "splitmnist": 2}
    jobs.sort(key=lambda j: (order[j["cfg"]["benchmark"]], j["cfg"]["arch"] != "snn"))
    return jobs


def job_id(job) -> str:
    c = job["cfg"]
    return f"{job['phase']}__{c['benchmark']}__{c['scenario']}__{run_name(c['method'], c['arch'])}__{job['tag']}__seed{c['seed']}"


def out_path(out: Path, job) -> Path:
    c = job["cfg"]
    return out / job["phase"] / c["benchmark"] / c["scenario"] / run_name(c["method"], c["arch"]) / job["tag"] / f"seed{c['seed']}.json"


def worker(phase, benchmarks, seeds, out: Path, data: str, wid: int, mem_frac: float = 0.0, device: str | None = None):
    import torch

    from .run import run_experiment  # heavy imports only in workers

    if mem_frac > 0 and torch.cuda.is_available():
        # fail with a clean OOM instead of silently paging GPU memory to host RAM (Windows WDDM)
        torch.cuda.set_per_process_memory_fraction(mem_frac)
    jobs = build_jobs(phase, benchmarks, seeds, out)
    locks = out / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    failed = out / "failed"
    failed.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("CLBENCH_DEADLINE", "inf"))  # unix time after which no new job starts
    for job in jobs:
        if time.time() > deadline:
            print(f"[w{wid}] deadline reached, stopping", flush=True)
            break
        path = out_path(out, job)
        if path.exists() or (failed / (job_id(job) + ".txt")).exists():
            continue
        lock = locks / (job_id(job) + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(wid).encode())
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            res = run_experiment(job["cfg"], root=data, device=device)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(path)
            print(f"[w{wid}] done {job_id(job)} ACC={res['metrics']['ACC']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            # recorded once; other workers skip it during this launch (relaunch to retry)
            try:
                (failed / (job_id(job) + ".txt")).write_text(f"worker {wid}\n" + traceback.format_exc())
            except OSError:
                pass
            print(f"[w{wid}] FAILED {job_id(job)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            lock.unlink(missing_ok=True)


def launch(args):
    out = Path(args.out)
    for d in ("locks", "failed"):  # a new launch retries failed jobs
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in (out / d).iterdir():
            f.unlink()
    (out / "logs").mkdir(exist_ok=True)
    jobs = build_jobs(args.phase, args.benchmarks, args.seeds, out)
    todo = [j for j in jobs if not out_path(out, j).exists()]
    print(f"{len(jobs)} jobs in phase {args.phase}, {len(todo)} to run, {args.workers} workers", flush=True)
    procs = []
    for w in range(args.workers):
        cmd = [sys.executable, "-m", "clbench.grid", "worker", "--phase", args.phase, "--out", str(out), "--data", args.data,
               "--wid", str(w), "--benchmarks", *args.benchmarks, "--seeds", *map(str, args.seeds),
               "--mem-frac", str(args.mem_frac)]
        if args.gpus > 1:  # spread workers over several GPUs (e.g. Kaggle T4 x2)
            cmd += ["--device", f"cuda:{w % args.gpus}"]
        log = open(out / "logs" / f"{args.phase}_w{w}.log", "a")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)
    for p in procs:
        p.wait()
    done = sum(out_path(out, j).exists() for j in jobs)
    print(f"finished: {done}/{len(jobs)} results present", flush=True)


def select(args):
    """Pick, for every (benchmark, scenario, method), the grid point with the
    best final validation accuracy (ACC)."""
    out = Path(args.out)
    best: dict[str, tuple[float, dict]] = {}
    for f in (out / "hp").rglob("seed0.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        key = f"{c['benchmark']}/{c['scenario']}/{run_name(c['method'], c['arch'])}"
        grid = hp_grid(c["method"], c["arch"])
        over = {k: c[k] for k in grid}
        acc = r["metrics"]["ACC"]
        if key not in best or acc > best[key][0]:
            best[key] = (acc, over)
    sel = {k: v[1] for k, v in sorted(best.items())}
    (out / "selected_hparams.json").write_text(json.dumps(sel, indent=1))
    for k, (acc, over) in sorted(best.items()):
        print(f"{k:45s} val ACC={acc:.4f}  {over}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["launch", "worker", "select", "list"])
    ap.add_argument("--phase", default="hp", choices=["hp", "final", "timing"])
    ap.add_argument("--benchmarks", nargs="+", default=["splitmnist", "permmnist"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.0, help="per-worker cap on GPU memory (fraction of total)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.phase == "timing" and args.cmd == "launch":
        args.seeds = [1]
    if args.cmd == "launch":
        launch(args)
    elif args.cmd == "worker":
        seeds = [1] if args.phase == "timing" else args.seeds
        if args.device and args.device.startswith("cuda"):
            import torch

            torch.cuda.set_device(args.device)
        worker(args.phase, args.benchmarks, seeds, Path(args.out), args.data, args.wid, args.mem_frac, args.device)
    elif args.cmd == "select":
        select(args)
    else:
        for j in build_jobs(args.phase, args.benchmarks, args.seeds, Path(args.out)):
            print(job_id(j))


if __name__ == "__main__":
    main()
