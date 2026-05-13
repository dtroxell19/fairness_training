# Quickstart

Train a fairness-constrained neural network in five steps. This example uses a synthetic classification dataset, but the same pattern works for any tabular task.

---

## The Setup

We have a binary classification problem where we need to ensure the model's predicted scores are similar across two groups — a requirement imposed by AI regulations or internal policy.

---

## Step 1: Prepare Your Data

Protected attributes must be **binary (0/1) columns** in your feature matrix `X`:

```python
import numpy as np
import torch

# X[:, 0] is the protected attribute (binary 0/1)
X = np.random.randn(10000, 20).astype(np.float32)
X[:, 0] = np.random.binomial(1, 0.3, 10000)  # 30% in group 1

y = (X[:, 1] + X[:, 2] + 0.5 * X[:, 0] > 0).astype(np.float32)

# Split into train / val / test
n = len(X)
X_train, y_train = X[:7000], y[:7000]
X_val,   y_val   = X[7000:8500], y[7000:8500]
X_test,  y_test  = X[8500:], y[8500:]
```

---

## Step 2: Create Stratified DataLoaders

Stratified batching keeps the group ratio constant across batches. This is important: when batch composition is constant, per-batch fairness constraints automatically imply aggregate fairness over the full dataset.

```python
from fairness_training import create_stratified_dataloaders

train_loader, val_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    protected_attr_idx=0,
    batch_size_train=256,
    batch_size_eval=256,
)
```

---

## Step 3: Build a Fair Model

The easiest path is `FairModel.wrap()`, which takes any `nn.Module` and adds the fairness layer. Prediction bounds are inferred automatically.

```python
import torch.nn as nn
from fairness_training import FairModel

backbone = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),  nn.Sigmoid(),
)

model = FairModel.wrap(
    backbone,
    protected_attr_idx=0,
    fairness_tolerance=0.05,    # ε: max allowed group mean prediction gap
    fairness_metric='mean_pred',
)
```

You can also construct `FairModel` directly to customise the built-in FFNN:

```python
model = FairModel(
    input_dim=20,
    hidden_dims=[64, 32],
    protected_attr_idx=0,
    fairness_tolerance=0.05,
    fairness_metric='mean_pred',
    prediction_bounds=(0.0, 1.0),
)
```

---

## Step 4: Train with FairTrainer

```python
import torch.optim as optim
from fairness_training import FairTrainer

criterion  = nn.BCELoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3)

trainer = FairTrainer(model, criterion, optimizer, early_stopping_patience=10)

history = trainer.fit(
    train_loader,
    val_loader=val_loader,
    epochs=50,
    verbose=1,
    log_interval=5,
)
```

You'll see output like:

```
Epoch   5 | Train Loss: 0.4523 | Val Loss: 0.4612 | Train Gap: 0.0491 | Val Gap: 0.0498
Epoch  10 | Train Loss: 0.4156 | Val Loss: 0.4298 | Train Gap: 0.0478 | Val Gap: 0.0482
...
```

The `Train Gap` and `Val Gap` are the fairness gaps — they stay at or below the 0.05 tolerance.

---

## Step 5: Evaluate

```python
metrics = trainer.evaluate(test_loader)

print(f"Test Loss:                 {metrics['test_loss']:.4f}")
print(f"Weighted avg fairness gap: {metrics['weighted_avg_fairness_gap']:.4f}  (target ≤ 0.05)")
print(f"Pooled fairness gap:       {metrics['pooled_fairness_gap']:.4f}")
```

Expected output:

```
Test Loss:                 0.4312
Weighted avg fairness gap: 0.0412  (target ≤ 0.05)
Pooled fairness gap:       0.0418
```

---

## Complete Example

```python
import numpy as np
import torch.nn as nn
import torch.optim as optim

from fairness_training import FairModel, FairTrainer, create_stratified_dataloaders

# 1. Prepare data
X = np.random.randn(10000, 20).astype(np.float32)
X[:, 0] = np.random.binomial(1, 0.3, 10000)
y = (X[:, 1] + X[:, 2] + 0.5 * X[:, 0] > 0).astype(np.float32)

n = len(X)
X_train, y_train = X[:7000], y[:7000]
X_val,   y_val   = X[7000:8500], y[7000:8500]
X_test,  y_test  = X[8500:], y[8500:]

# 2. Stratified dataloaders
train_loader, val_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    protected_attr_idx=0,
    batch_size_train=256,
    batch_size_eval=256,
)

# 3. Wrap any backbone with fairness constraints
backbone = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),  nn.Sigmoid(),
)
model = FairModel.wrap(backbone, protected_attr_idx=0, fairness_tolerance=0.05)

# 4. Train
trainer = FairTrainer(
    model,
    nn.BCELoss(),
    optim.Adam(model.parameters(), lr=1e-3),
    early_stopping_patience=10,
)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=50, verbose=1)

# 5. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Weighted avg fairness gap: {metrics['weighted_avg_fairness_gap']:.4f}  (target ≤ 0.05)")
```

---

## What's Next?

- **[Core Concepts](../user-guide/concepts.md)**: Understand the theory — why hard constraints work, and how the primal-dual algorithm provides aggregate guarantees
- **[Fairness Metrics](../user-guide/fairness-metrics.md)**: Mean prediction parity, mean residual fairness, equalized odds
- **[Small-Batch Inference](../examples/small-batch.md)**: Deploy with streaming / real-time predictions
- **[Custom Metrics](../user-guide/custom-metrics.md)**: Define your own affine fairness constraints
