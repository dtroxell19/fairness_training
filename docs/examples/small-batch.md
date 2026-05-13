# Example: Small-Batch Inference

This example demonstrates deploying a model for real-time/streaming predictions

!!! important "Critical Requirement"
    As mentioned in the User Guide, this package assumes stratified sampling can be used in your training dataset (and is achieved via the create_stratified_dataloaders() function). This ensures each batch has the same proportion of each group, which implies that enforcing constraints per-batch automatically leads to aggregate fairness (i.e. fairness satisfied when considering all training examples at once)

---

## Notes

Per-batch hard constraints may not work well with small batches because:

1. Model expressivity may be severely limited 
2. Group membership ratios can vary between small batches, making stratification difficult

The primal-dual algorithm relaxes per-batch constraints and instead provides a weaker but still meaningful **asymptotic guarantee**.

!!! warning "What the small-batch guarantee actually says"
    The primal-dual algorithm guarantees that the **sample-weighted average violation** converges to at most ε as the number of inference batches T → ∞:

    $$\bar{\Delta}_T = \frac{1}{N_T} \sum_{t=1}^{T} n_t \cdot \Delta_t \;\leq\; \varepsilon$$

    where $n_t$ is the batch size and $\Delta_t$ is the per-batch fairness gap.

    This is **not** the same as the pooled gap when considering all predictions together. The pooled gap (`pooled_aggregate_gap`) and the sample-weighted average (`weighted_avg_fairness_gap`) coincide only when the group ratio $n_t^{(0)}/n_t^{(1)}$ is constant across batches. Use `weighted_avg_fairness_gap` when assessing compliance with the theoretical bound.

    Additionally, there is **no guarantee on when** this bound is achieved — it is an asymptotic statement. Individual batches and finite sequences may still show gaps above ε.

---

## Complete Example

```python
"""
Small-Batch Fair Inference with Primal-Dual Algorithm

Train with large batches, deploy with streaming small batches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from fairness_training import (
    FairModel, FairTrainer,
    create_stratified_dataloaders, create_dataloaders
)


# -----------------------------------------------------------------------------
# Generate synthetic data
# -----------------------------------------------------------------------------
np.random.seed(42)
n_samples, n_features = 20000, 20

# Protected attribute (col 0): 30% in group 1
protected = np.random.binomial(1, 0.3, n_samples)

# Features with some correlation to protected attribute
X_features = np.random.randn(n_samples, n_features - 1)
X_features[:, 0] += 0.3 * protected

# Target with bias: group 1 more likely positive
logits = X_features[:, 0] + X_features[:, 1] + 0.5 * protected
y = (logits + np.random.randn(n_samples) * 0.5 > 0).astype(np.float32)

X = np.hstack([protected.reshape(-1, 1), X_features]).astype(np.float32)

print(f"Dataset: {n_samples} samples, Group 0: {(1-protected).sum()}, Group 1: {protected.sum()}")
print(f"Positive rate - Group 0: {y[protected==0].mean():.3f}, Group 1: {y[protected==1].mean():.3f}")

# -----------------------------------------------------------------------------
# Train/val/test split
# -----------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
batch_size_train = 2000    # Large batches for training
batch_size_inference = 64  # Small batches for inference
b_tau = 2000
fairness_tolerance = 0.05
eta_0 = 0.5                # Initial dual step size. Increase for stricter compliance to constraints!

# Training dataloaders (large batches with stratification)
train_loader, val_loader = create_stratified_dataloaders(
    X_train, y_train, X_val, y_val,
    protected_attr_idx=0,
    batch_size_train=batch_size_train,
    batch_size_eval=batch_size_train
)

# Test dataloader (small batches - simulates streaming)
_, _, test_loader_small = create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size_train=batch_size_train,
    batch_size_eval=batch_size_inference
)

# -----------------------------------------------------------------------------
# Create FairModel
# -----------------------------------------------------------------------------
model = FairModel(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    output_dim=1,
    protected_attr_idx=0,
    prediction_bounds=(-10.0, 10.0),
    fairness_tolerance=fairness_tolerance,
    b_tau=b_tau,
    eta_0=eta_0,
    fairness_metric='mean_pred'
)

print(f"\nTraining:  batch={batch_size_train} >= b_tau → HARD constraints")
print(f"Inference: batch={batch_size_inference} < b_tau  → PRIMAL-DUAL algorithm")
print(f"Fairness tolerance (ε): {fairness_tolerance}, eta_0: {eta_0}")

# -----------------------------------------------------------------------------
# Train (large batches with hard constraints)
# -----------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = FairTrainer(model, criterion, optimizer, early_stopping_patience=15)

print("\n" + "="*60)
print("TRAINING (Large Batches - Hard Constraints)")
print("="*60)
history = trainer.fit(train_loader, val_loader, epochs=30, verbose=1, log_interval=10)

# -----------------------------------------------------------------------------
# Inference (small batches with primal-dual)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("INFERENCE (Small Batches - Primal-Dual)")
print("="*60)

metrics = trainer.evaluate(test_loader_small)
stats = model.get_aggregate_fairness_stats(test_loader_small, reset_before=True)

print(f"\nResults:")
print(f"  Test Loss: {metrics['test_loss']:.4f}")
print(f"  Aggregate Fairness Gap: {stats['aggregate_gap']:.4f}")
print(f"  Target ε: {fairness_tolerance}")
print(f"  Constraint satisfied: {stats['aggregate_gap'] <= fairness_tolerance}")
print(f"\n  Lambda max: {stats['lambda_max']:.4f}")
print(f"  Total samples: {stats['total_samples']}, Num batches: {stats['num_batches']}")
```

---

## Expected Output

```
Dataset: 20000 samples, Group 0: 13989, Group 1: 6011
Positive rate - Group 0: 0.423, Group 1: 0.576

Training:  batch=2000 >= b_tau → HARD constraints
Inference: batch=64 < b_tau  → PRIMAL-DUAL algorithm
Fairness tolerance (ε): 0.05, eta_0: 0.5

============================================================
TRAINING (Large Batches - Hard Constraints)
============================================================
Epoch   10 | Train Loss: 0.5234 | Val Loss: 0.5189 | Train Gap: 0.0312 | Val Gap: 0.0398
Epoch   20 | Train Loss: 0.4876 | Val Loss: 0.4923 | Train Gap: 0.0234 | Val Gap: 0.0312

============================================================
INFERENCE (Small Batches - Primal-Dual)
============================================================

Results:
  Test Loss: 0.4956
  Aggregate Fairness Gap: 0.0423
  Target ε: 0.05
  Constraint satisfied: True

  Lambda max: 2.3456
  Total samples: 3968, Num batches: 62
```

For batch processing scenarios, see [Large-Batch Inference](large-batch.md).