# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-05-13

### Added

- `FairModel` — PyTorch `nn.Module` with differentiable fairness layer (cvxpylayers) enforcing hard per-batch constraints during training
- `FairModel.wrap()` — wrap any existing `nn.Module` with fairness constraints; prediction bounds inferred automatically via dry-run forward pass
- `FairTrainer` — high-level training loop with early stopping, LR scheduling, checkpoint save/load, and per-epoch fairness gap tracking
- Three built-in fairness metrics: `MeanPredictionParity` (`'mean_pred'`), `MeanResidualFairness` (`'mean_residual'`), `EqualizedOdds` (`'equalized_odds'`)
- `FairnessMetric` abstract base class for custom affine fairness constraints
- `validate_metric()` — pre-training DPP compliance check for custom metrics
- `create_stratified_dataloaders()` — stratified batching that maintains constant group ratios per batch, accepting both numpy arrays and torch tensors
- Online primal-dual inference algorithm for small-batch / streaming deployment with sample-weighted average violation guarantee (converges to ≤ ε as T → ∞)
- `fairness_training.viz` module: `plot_training_history`, `plot_group_distributions`, `plot_fairness_tradeoff`
- Quickstart Colab notebook (`notebooks/quickstart.ipynb`)
- Full MkDocs documentation site
- CI workflow (pytest on Python 3.10/3.11) and PyPI trusted-publishing workflow

### Notes

- Protected attributes must be binary (0/1) columns in the feature matrix X
- Supports up to 2 protected attributes (marginal fairness, enforced independently)
- Training always uses hard per-batch constraints regardless of `b_tau`; `b_tau` controls the inference regime only (default: 64)
- `weighted_avg_fairness_gap` (the theorem-bounded quantity) ≠ `pooled_fairness_gap`; they coincide only when group ratios are constant across batches

[Unreleased]: https://github.com/dtroxell19/fairness_training/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dtroxell19/fairness_training/releases/tag/v0.1.0
