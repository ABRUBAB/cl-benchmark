"""Bundle clbench into a single Kaggle script kernel.

Kaggle script kernels consist of one file, so the package is embedded as a
base64-encoded zip.  The kernel runs the requested grid phases on the Kaggle
GPU(s) and leaves all JSON results in /kaggle/working/results.

    python kaggle/make_kernel.py --slug clbench-cifar100 --phases hp select final timing \
        --benchmarks splitcifar100 --workers 4 --gpus 2
    kaggle kernels push -p kaggle/build/clbench-cifar100
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNNER = r'''
import base64, io, json, os, subprocess, sys, time, zipfile, shutil
T0 = time.time()
PAYLOAD = "{payload}"
PLAN = {plan}
CODE = "/kaggle/working/code"
OUT = "/kaggle/working/results"
DATA = "/tmp/clbench-data"
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(CODE)
os.makedirs(OUT, exist_ok=True)
# use the attached copy of the original CIFAR-100 python files (same MD5 as the official archive)
for root, dirs, files in os.walk("/kaggle/input"):
    if {"train", "test", "meta"} <= set(files):
        dst = os.path.join(DATA, "cifar-100-python")
        os.makedirs(dst, exist_ok=True)
        for fn in ("train", "test", "meta"):
            if not os.path.exists(os.path.join(dst, fn)):
                os.symlink(os.path.join(root, fn), os.path.join(dst, fn))
        print("CIFAR-100 from", root, flush=True)
        break
seed_dir = "/kaggle/input"
# reuse results of earlier kernels attached as inputs (e.g. selected hyper-parameters)
for root, dirs, files in os.walk(seed_dir):
    if "selected_hparams.json" in files and not os.path.exists(os.path.join(OUT, "selected_hparams.json")):
        shutil.copy(os.path.join(root, "selected_hparams.json"), OUT)
subprocess.run(["nvidia-smi"])
env = dict(os.environ, PYTHONPATH=CODE, CLBENCH_DEADLINE=str(T0 + PLAN["hours"] * 3600),
           CLBENCH_METHODS=",".join(PLAN.get("methods", [])))
py = sys.executable
subprocess.run([py, "-c", "import torch;print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"])
for argv in PLAN.get("pre", []):  # e.g. quick smoke tests
    print("PRE", argv, flush=True)
    subprocess.run([py, "-m", *argv, "--data", DATA, "--out", OUT + "/smoke"], cwd=CODE, env=env)
for phase in PLAN["phases"]:
    if phase == "select":
        subprocess.run([py, "-m", "clbench.grid", "select", "--out", OUT], cwd=CODE, env=env)
        continue
    workers = 1 if phase == "timing" else PLAN["workers"]
    gpus = 1 if phase == "timing" else PLAN["gpus"]
    cmd = [py, "-m", "clbench.grid", "launch", "--phase", phase, "--workers", str(workers), "--gpus", str(gpus),
           "--benchmarks", *PLAN["benchmarks"], "--out", OUT, "--data", DATA, "--seeds", *map(str, PLAN["seeds"])]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, env=env)
    for f in sorted(os.listdir(os.path.join(OUT, "logs"))):
        print("==", f); print(open(os.path.join(OUT, "logs", f)).read()[-3000:])
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print("elapsed hours", (time.time() - T0) / 3600)
'''


def bundle() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "clbench").rglob("*.py")):
            z.write(f, f.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--phases", nargs="+", default=["hp", "select", "final", "timing"])
    ap.add_argument("--benchmarks", nargs="+", default=["splitcifar100"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--hours", type=float, default=10.5, help="no new job is started after this many hours")
    ap.add_argument("--sources", nargs="*", default=[], help="kernel outputs to attach, e.g. user/clbench-cifar100-hp")
    ap.add_argument("--methods", nargs="*", default=[], help="only run these methods (default: all)")
    ap.add_argument("--smoke", action="store_true", help="only run two tiny experiments (checks GPU/internet/packaging)")
    args = ap.parse_args()
    plan = dict(phases=args.phases, benchmarks=args.benchmarks, seeds=args.seeds, workers=args.workers, gpus=args.gpus,
                hours=args.hours, methods=args.methods)
    if args.smoke:
        plan["phases"] = []
        plan["pre"] = [
            ["clbench.run", "--benchmark", "splitmnist", "--scenario", "class", "--method", "ewc", "--set", "iters=200"],
            ["clbench.run", "--benchmark", "splitcifar100", "--scenario", "task", "--method", "pnn", "--set", "iters=50"],
            ["clbench.run", "--benchmark", "splitmnist", "--scenario", "domain", "--method", "ewc", "--arch", "snn",
             "--set", "iters=100"],
        ]
    out = ROOT / "kaggle" / "build" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_kaggle.py").write_text(RUNNER.replace("{payload}", bundle()).replace("{plan}", json.dumps(plan)))
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": "run_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["fedesoriano/cifar100"],
        "competition_sources": [],
        "kernel_sources": args.sources,
        "machine_shape": "NvidiaTeslaT4" if args.gpus > 1 else "NvidiaTeslaP100",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
