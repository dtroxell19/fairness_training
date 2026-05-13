# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
# Training + optimization (typical development setup)
pip install -e .[train]
pip install -e .[verify]

# Full installation (all features)
pip install -e .[full]

# Development tools
pip install -e .[dev]
```

The `train` extra adds PyTorch/NumPy/Transformers; the `verify` extra adds CVXPY and its solvers (required for the differentiable fairness layer).

If you hit `AttributeError: module 'onnxscript.values' has no attribute 'ParamSchema'` on first run, uninstall the conflicting package — it's only needed for ONNX export, which this library doesn't use:

```bash
pip uninstall onnxscript -y
```

## Running the example

From the project root (after `pip install -e .[train,verify]`):

```bash
python -m fairness_training.example_usage
```

## Architecture

The package is a PyTorch library for training neural networks with **guaranteed fairness constraints** enforced through a differentiable convex optimization layer (cvxpylayers).

### Core data flow

```
Input X → FFNN (ffnn) → raw predictions y_hat → CvxpyLayer → fair predictions ỹ
```

The cvxpy layer projects y_hat onto the set of predictions satisfying fairness constraints, minimizing squared distortion. The layer is differentiable, so gradients flow back through the projection into the network weights.

### Module responsibilities

**`fair_model.py` — `FairModel`**  
The main model. Wraps an arbitrary PyTorch network (`ffnn`) with a dynamically-created `CvxpyLayer`. Two inference regimes based on batch size vs `b_tau` (default 64):
- `batch_size >= b_tau` → hard per-batch constraints (cvxpy layer solves a constrained QP)
- `batch_size < b_tau` → online primal-dual algorithm (fairness penalty in objective, dual variable `lambda_dual` updated after each batch)

**Training always uses hard constraints regardless of batch size** — `b_tau` only affects inference. `FairModel.wrap(..., exclude_protected_from_backbone=True)` strips protected attribute columns before the backbone, preventing the network from directly learning on them.

**`fair_trainer.py` — `FairTrainer`**  
High-level training loop. Calls `model.reset_inference_state()` before every validation/test pass to reset the primal-dual state. Tracks aggregate fairness gap (not per-batch) in training history. Saves/loads checkpoints via `torch.save`.

**`fairness_metrics.py` — `FairnessMetric` + built-ins**  
Abstract base class defining the metric interface. Built-in implementations:
- `MeanPredictionParity` — `|E[Ŷ|A=0] - E[Ŷ|A=1]| ≤ ε` (demographic parity; no targets needed)
- `MeanResidualFairness` — `|E[Y-Ŷ|A=a]| ≤ ε` for each group (regression fairness; needs targets)
- `EqualizedOdds` — equalized predictions conditioned on outcome class (needs binary targets)

**`utils.py`**  
`create_dataloaders` and `create_stratified_dataloaders`. Stratified loaders maintain fixed group proportions per batch (important because the cvxpy layer requires both groups present in every batch). Drops incomplete batches with a warning.

### Key constraints and invariants

- **Protected attributes must be binary (0/1).** The model checks for both groups in every batch and skips batches that are missing a group.
- **Training always uses hard per-batch constraints** regardless of batch size. `b_tau` (default 64) controls only inference: batches ≥ b_tau use hard constraints; smaller batches fall back to primal-dual. Use `create_stratified_dataloaders` to ensure both groups appear in every batch.
- **Custom fairness metrics must be DPP-compliant** for cvxpylayers: constraints must be affine, use selection matrices (constant numpy 0/1 arrays) for group masking, and avoid multiplying two cvxpy Parameters together.
- **`FairModel` supports at most 2 protected attributes.**
- The cvxpy layer is recreated on every forward pass (it depends on batch-specific selection matrices). This is intentional but expensive — large batches amortize the cost.

### Extending with custom metrics

Subclass `FairnessMetric` and implement:
1. `create_selection_matrices()` — build numpy 0/1 arrays selecting group members
2. `create_constraints()` — return DPP-compliant cvxpy constraints
3. `compute_gap()` — return a scalar monitoring value (not used in optimization)
4. `create_primal_dual_penalty()` — penalty for small-batch inference
5. Set `requires_targets = True` if ground-truth labels are needed; set `requires_y_in_constraints = True` if `y` must appear as a cvxpy `Parameter` in the problem.

String aliases for built-in metrics (passed as `fairness_metric=` to `FairModel`): `'mean_pred'`, `'mean_residual'`, `'equalized_odds'`.
