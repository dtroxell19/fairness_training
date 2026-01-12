# Example: Large-Batch Inference

This example demonstrates training and inference with large batches, where hard per-batch fairness constraints are enforced

!!! important "Critical Requirement"
    As mentioned in the User Guide, this package assumes stratified sampling can be used in your training dataset (and is achieved via the create_stratified_dataloaders() function). This ensures each batch has the same proportion of each group, which implies that enforcing constraints per-batch automatically leads to aggregate fairness (i.e. fairness satisfied when considering all training examples at once)

---

## Complete Example

```python
"""
Large-Batch Fair Classification with Synthetic Data

Demonstrates hard per-batch fairness constraints on 2 attributes.
"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from fairness_training import FairModel, FairTrainer, create_stratified_dataloaders


# -----------------------------------------------------------------------------
# Generate synthetic data
# -----------------------------------------------------------------------------
np.random.seed(42)
n_samples, n_features = 20000, 50

# Protected attributes: attr1 (col 0), attr2 (col 1)
attr1 = np.random.binomial(1, 0.6, n_samples)  # 60% male
attr2 = np.random.binomial(1, 0.7, n_samples)    # 70% white

# Features correlated with protected attributes (to create bias)
X_features = np.random.randn(n_samples, n_features - 2)
X_features[:, 0] += 0.5 * attr1  # Feature correlated with attr1
X_features[:, 1] += 0.3 * attr2    # Feature correlated with attr2

# Stack: protected attributes as first two columns
X = np.hstack([attr1.reshape(-1, 1), attr2.reshape(-1, 1), X_features]).astype(np.float32)

# Target with built-in bias (to test fairness correction)
y = (0.5 * X[:, 2] - 0.3 * X[:, 3] + 0.8 * attr1 + 0.4 * attr2 + 
     np.random.randn(n_samples) * 0.5 > 0.5).astype(np.float32)

print(f"Dataset: {n_samples} samples, {X.shape[1]} features")
print(f"attr1: {100*attr1.mean():.1f}% Male, attr2: {100*attr2.mean():.1f}% White")
print(f"Positive class: {100*y.mean():.1f}%")

# -----------------------------------------------------------------------------
# Train/val/test split
# -----------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
batch_size = 2000
b_tau = 2000
fairness_tolerance = 0.05

# Create stratified dataloaders (maintains group proportions per batch)
train_loader, val_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    protected_attr_idx=[0, 1],
    batch_size_train=batch_size,
    batch_size_eval=batch_size
)

# -----------------------------------------------------------------------------
# Create FairModel
# -----------------------------------------------------------------------------
model = FairModel(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    output_dim=1,
    protected_attr_idx=[0, 1],       # 2 attributes
    prediction_bounds=(-1000.0, 1000.0),
    fairness_tolerance=fairness_tolerance,
    b_tau=b_tau,
    fairness_metric='mean_pred'      # Mean Prediction Parity
)

print(f"\nBatch size {batch_size} >= b_tau {b_tau} → HARD constraints")
print(f"Fairness tolerance (ε): {fairness_tolerance}")

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = FairTrainer(model, criterion, optimizer, early_stopping_patience=15)
history = trainer.fit(train_loader, val_loader, epochs=50, verbose=1, log_interval=10)

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
metrics = trainer.evaluate(test_loader)

print(f"\nResults:")
print(f"  Test Loss: {metrics['test_loss']:.4f}")
print(f"  Fairness Gap (attr1): {metrics['fairness_gap_attr_0']:.4f}")
print(f"  Fairness Gap (attr2): {metrics['fairness_gap_attr_1']:.4f}")
print(f"  Max Gap: {metrics['fairness_gap']:.4f} (target ≤ {fairness_tolerance})")
print(f"  Constraint satisfied: {metrics['fairness_gap'] <= fairness_tolerance}")
```

---

## Expected Output

```
Dataset: 20000 samples, 50 features
attr1: 60.2% Male, attr2: 70.1% White
Positive class: 62.3%

=== Creating stratified training batches ===
  Per-batch allocation: group 00=119, group 01=280, group 10=481, group 11=1120
  Created 6 batches of size 2000

Batch size 2000 >= b_tau 2000 → HARD constraints
Fairness tolerance (ε): 0.05

Epoch   10 | Train Loss: 0.5823 | Val Loss: 0.5756 | Train Gap: 0.0312 | Val Gap: 0.0389
Epoch   20 | Train Loss: 0.5512 | Val Loss: 0.5523 | Train Gap: 0.0287 | Val Gap: 0.0356
...

Results:
  Test Loss: 0.5301
  Fairness Gap (attr1): 0.0398
  Fairness Gap (attr2): 0.0287
  Max Gap: 0.0398 (target ≤ 0.05)
  Constraint satisfied: True
```

---

## Key Points

1. **Stratified batching** ensures consistent group membership ratios per batch

2. **Hard constraints** guarantee `|E[Ŷ|A=0] - E[Ŷ|A=1]| ≤ ε` for every batch

3. **Marginal fairness** is enforced independently for each protected attribute

For streaming/real-time scenarios, see [Small-Batch Inference](small-batch.md).
