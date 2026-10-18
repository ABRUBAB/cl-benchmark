# clbench: a unified continual-learning benchmark

Companion code for the review *"Overcoming Catastrophic Interference Strategies for Continual Learning in Neural Networks"* (Discover Artificial Intelligence, revised version).

It re-implements representative methods from every family discussed in the review in **one framework**, so that they share the same data pipeline, backbone, training budget, hyper-parameter selection protocol and metrics:

| Family | Methods |
|---|---|
| Baselines | sequential fine-tuning (lower bound), joint training (upper bound) |
| Regularization | EWC (Kirkpatrick et al., 2017), SI (Zenke et al., 2017) |
| Orthogonal projection | OWM (Zeng et al., 2019) |
| Distillation | LwF (Li & Hoiem, 2017) |
| Rehearsal | ER with reservoir buffer, Deep Generative Replay with a VAE (Shin et al., 2017) |
| Architecture | Progressive Neural Networks (Rusu et al., 2016) |
| Meta-learning | MER, mini-batch form (Riemer et al., 2019) |
| Spiking | LIF spiking MLP trained with surrogate gradients, combined with fine-tuning, EWC and ER |

Benchmarks and scenarios (van de Ven, Tuytelaars & Tolias, 2022):

| Benchmark | Tasks | Scenarios |
|---|---|---|
| Split MNIST | 5 × 2 digits | Task-IL, Domain-IL, Class-IL |
| Permuted MNIST | 10 permutations | Domain-IL |
| Split CIFAR-100 | 10 × 10 classes | Task-IL, Class-IL |

The pseudocode in Appendix B of the paper corresponds line by line to the files in `clbench/methods/`.

## Installation

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python on Linux/macOS
```

PyTorch with CUDA is recommended (any recent version ≥ 2.1). MNIST and CIFAR-100 are downloaded automatically into `data/`.

## Running

One experiment (writes a JSON file with the accuracy matrix, metrics, run time and memory):

```bash
python -m clbench.run --benchmark splitmnist --scenario class --method ewc --seed 1 --set ewc_lambda=1e5
python -m clbench.run --benchmark permmnist --scenario domain --method er --arch snn --seed 1
```

The full protocol used in the paper:

```bash
# 1. hyper-parameter search (seed 0, validation split)
python -m clbench.grid launch --phase hp --workers 4 --benchmarks splitmnist permmnist splitcifar100
python -m clbench.grid select                      # -> results/selected_hparams.json
# 2. final runs (seeds 1-5, test split)
python -m clbench.grid launch --phase final --workers 4 --benchmarks splitmnist permmnist splitcifar100
# 3. timing runs (one job at a time, so that run times are not affected by other jobs)
python -m clbench.grid launch --phase timing --workers 1 --benchmarks splitmnist permmnist splitcifar100
# 4. tables and figure of the paper
python scripts/make_tables.py --results results --tables tables --figures figures --summary summary
```

The paper's results were produced on Kaggle (2 × NVIDIA T4) with `kaggle/make_kernel.py`, which packs this repository into a single script kernel; the raw result files are in `results/`. On Windows with a small GPU, use at most 2–3 workers (`--mem-frac 0.3`), otherwise the driver pages GPU memory to system RAM and everything slows down.

## Protocol

* 10 % of each task's training data is held out for validation; hyper-parameters are selected on it (seed 0) from the grids in `clbench/configs.py`; the test set is used only for the final runs.
* Adam (lr 1e-3, batch 128), 2000 iterations per task; OWM and MER use plain SGD as required by their update rules.
* Metrics (`clbench/metrics.py`): average accuracy (ACC), average incremental accuracy (AIA), backward transfer (BWT), forgetting (FM) and forward transfer (FWT); plus training time per task, peak GPU memory and size of the persistent state kept by each method.

## Repository layout

```
clbench/            library: data.py (benchmarks/scenarios), models.py, metrics.py, methods/, run.py, grid.py
scripts/            make_tables.py (paper tables and figure)
kaggle/             make_kernel.py (Kaggle runner)
literature_search/  search script and log of the updated literature search (Section 3 of the paper)
results/            raw JSON results of all runs reported in the paper
```

## Citation

If you use this benchmark or codebase in your research, please cite our review paper:

```bibtex
@article{dabool2026overcoming,
  title={Overcoming Catastrophic Interference Strategies for Continual Learning in Neural Networks},
  author={Dabool, Hamzah and Alashwal, Hany and van Zyl, Terence and Khanday, Shifan and van der Haar, Dustin and Chakravarthy, Srinivasa and Deldari, Shohreh and Lerner, Itamar and Hassan, Mubashir and Rubab, Abdullah and Hossain, Md. Ali and Moustafa, Ahmed A.},
  journal={Discover Artificial Intelligence},
  year={2026}
}
```

## Repository

- **GitHub**: [https://github.com/ABRUBAB/cl-benchmark](https://github.com/ABRUBAB/cl-benchmark)
- **Author**: Abdullah Rubab (`rubab2712@gmail.com`)

## License

MIT, see `LICENSE`.
