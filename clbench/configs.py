"""Default settings and hyper-parameter grids.

Training budgets and backbones follow common practice for these benchmarks
(van de Ven, Tuytelaars & Tolias, Nat. Mach. Intell. 2022): Adam, batch 128,
a fixed number of iterations per task.  Method hyper-parameters that are not
fixed by the original papers are selected on a held-out validation split
(10 % of each task's training data, seed 0) from the grids below; the test
set is never used for selection.
"""

BENCH_DEFAULTS = {
    "splitmnist": dict(arch="mlp", hidden=[400, 400], iters=2000, batch_size=128, lr=1e-3,
                       buffer_size=500, n_fisher=1000, z_dim=32, vae_hidden=400),
    "permmnist": dict(arch="mlp", hidden=[1000, 1000], iters=2000, batch_size=128, lr=1e-3,
                      buffer_size=1000, n_fisher=500, z_dim=100, vae_hidden=1000),
    "splitcifar100": dict(arch="cnn", fc=256, iters=2000, batch_size=128, lr=1e-3,
                          buffer_size=2000, n_fisher=500, z_dim=128),
}

# values used when a method is run without a grid search
METHOD_DEFAULTS = {
    "ewc": dict(ewc_lambda=1e4),
    "si": dict(si_c=1.0, si_xi=0.1),
    "lwf": dict(lwf_lambda=1.0, lwf_T=2.0),
    "er": dict(replay_weight=1.0),
    # plain-SGD methods: gradients clipped to norm 10, as in the original OWM code
    "owm": dict(lr=0.2, owm_alpha=1.0, clip_grad=10.0),
    "mer": dict(lr=0.1, mer_beta=0.1, mer_gamma=1.0, mer_k=5, mer_s=1, clip_grad=10.0),
}

HP_GRIDS = {
    "ewc": dict(ewc_lambda=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]),
    "si": dict(si_c=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
    "owm": dict(lr=[0.05, 0.2], owm_alpha=[0.01, 0.1, 1.0]),
    "mer": dict(lr=[0.03, 0.1], mer_beta=[0.03, 0.1, 0.3]),
}

# the spiking network's Fisher information has a different scale, so EWC gets its own grid
SNN_HP_GRIDS = {
    "ewc": dict(ewc_lambda=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6]),
}


def hp_grid(method: str, arch: str) -> dict:
    return (SNN_HP_GRIDS if arch == "snn" else HP_GRIDS).get(method, {})


# methods evaluated per scenario (PNN needs task identity -> Task-IL only)
def methods_for(scenario: str, arch: str) -> list[str]:
    if arch == "snn":  # representative spiking model: lower bound, one regulariser, one replay method
        return ["finetune", "ewc", "er"]
    base = ["finetune", "joint", "ewc", "si", "lwf", "er", "dgr", "owm", "mer"]
    if scenario == "task":
        base.append("pnn")
    return base


def make_config(benchmark: str, scenario: str, method: str, seed: int, arch: str | None = None,
                eval_split: str = "test", overrides: dict | None = None) -> dict:
    cfg = dict(BENCH_DEFAULTS[benchmark])
    cfg.update(METHOD_DEFAULTS.get(method, {}))
    if arch is not None:
        cfg["arch"] = arch
    cfg.update(overrides or {})
    cfg.update(benchmark=benchmark, scenario=scenario, method=method, seed=seed, eval_split=eval_split)
    return cfg
