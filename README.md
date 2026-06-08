# Comparing Noise-Robust Training Strategies for Dermatoscopic Image Classification under Instance-Dependent Label Noise

Code for a DTU Bachelor Thesis, as part of the BSc in Artificial Intelligence and Data, benchmarking noise-robust training methods on
dermatoscopic images. Three noise-robust strategies — **SCE** (robust loss), **ELR**
(early-learning regularization), and **AsyCo** (sample selection + DivideMix MixMatch) —
are compared against a plain cross-entropy **baseline** on **HAM10000**, under realistic
*feature-driven instance-dependent label noise*: flip targets are drawn from a held-out
classifier's confusion, so corruptions land on visually similar classes rather than at
random. All methods share one backbone (ResNet-34), identical 10-fold splits, and the
same training budget, and are evaluated across six noise rates (τ = 0.0–0.5) and four
optimizer × initialization protocols — including a clean-data (τ = 0) reference and a
class-imbalanced setting representative of clinical data.

## Layout

```
configs/    YAML configs (base, data, method, model, optim, noise) + final Optuna search spaces
src/
  data/       HAM10000 dataset, fold splits, transforms, two-view wrapper, samplers
  models/     ResNet-18/34 builder
  noise/      instance-dependent label noise (Xia + feature-driven) and characterization
  methods/    training methods: baseline, sce, elr, asyco_divmix (+ base class, factory)
  training/   runner (trains one fold end-to-end), optimizers, metrics, samplers
  analysis/   results aggregation, plots, statistical tests
  utils/      config/IO, run manifests, seeding
scripts/    pipeline stages (stage0-stage4) and thesis figure/table scripts
hpc/        LSF submit scripts for the DTU cluster
notebooks/  exploratory notebooks
```

## Pipeline

Scripts run as modules from the repo root (`python -m scripts.<name>`):

1. **stage0** - download and prepare HAM10000 (deduplicate to one image per lesion).
2. **stage1a** - create stratified 10-fold assignments.
3. **stage1b** - train out-of-fold ResNet-18 and collect/merge its softmax probabilities (the basis for feature-driven noise).
4. **stage1c** - inject instance-dependent label noise into the training folds (per τ, per fold).
5. **stage2** - Optuna hyperparameter search for SCE/ELR/AsyCo, then analyze to pick per-protocol settings.
6. **stage3** - main experiment: one invocation trains one `(method, dataset, init, optim, τ, fold)`; `stage3_status` reports grid completeness.
7. **stage4** - aggregate runs into figures, tables, and paired statistical tests; `stage4_train_diagnostics` plots per-epoch / per-class noise diagnostics.

`scripts/results_part*.py` and `thesis_paired_stats.py` build the thesis analysis, figures and tables from the aggregated results.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Global settings (seed, folds, image size, noise rates, paths) live in `configs/base.yaml`;
each run composes its choices from `configs/{data,method,model,optim,noise}/`. Outputs are
written under `results/`.