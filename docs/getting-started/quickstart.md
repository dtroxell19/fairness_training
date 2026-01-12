# Quickstart

This guide walks you through training a simple classifier with fairness constraints.

---

## The Problem

You have a binary classification task where your model must comply with new AI regulations. In order to meet regulations, the model must score men and women roughly the same on some task

---

## Step 1: Prepare Your Data

Your data should have protected attributes as columns in your feature matrix:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Example: X has protected attribute in column 0
# X[:, 0] should be binary (0 or 1)
X = np.random.randn(10000, 20).astype(np.float32)
X[:, 0] = np.random.binomial(1, 0.3, 10000)  # Protected attribute

y = (X[:, 1] + X[:, 2] + 0.5 * X[:, 0] > 0).astype(np.float32)  # labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Step 2: Create DataLoaders

Use stratified batching to maintain consistent group proportions:

```python
from fairness_training import create_stratified_dataloaders

train_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_test=X_test, y_test=y_test,
    protected_attr_idx=0,
    batch_size_train=500,  # Should be >= b_tau for hard constraints
    batch_size_eval=500
)
```

---

## Step 3: Create the Fair Model

```python
from fairness_training import FairModel

model = FairModel(
    input_dim=20,
    b_tau = 500,
    hidden_dims=[64, 32],
    output_dim=1,
    protected_attr_idx=0,
    fairness_tolerance=0.05,
    fairness_metric='mean_pred',
    prediction_bounds= (-100.0, 100.0) #bounds are for logits, not predicted pred
)
```

---

## Step 4: Train with FairTrainer

```python
import torch.nn as nn
import torch.optim as optim
from fairness_training import FairTrainer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = FairTrainer(
    model, criterion, optimizer,
    early_stopping_patience=15
)

history = trainer.fit(
    train_loader,
    val_loader=test_loader,
    epochs=20,
    verbose=1,
    log_interval=1
)
```

You'll see output like this. Note that the fairness gaps in training and validation are under the .05 tolerance:

```
Epoch   1 | Train Loss: 0.4523 | Val Loss: 0.4612 | Train Gap: 0.0500 | Val Gap: 0.0498
Epoch   2 | Train Loss: 0.4156 | Val Loss: 0.4298 | Train Gap: 0.0492 | Val Gap: 0.0500
...
```

---

## Step 5: Evaluate

```python
metrics = trainer.evaluate(test_loader)

print(f"Test Loss: {metrics['test_loss']:.4f}")
print(f"Fairness Gap: {metrics['fairness_gap']:.4f}")
print(f"Target Tolerance: {model.fairness_tolerance}")
```

Output:
```
Test Loss: 0.4312
Fairness Gap: 0.0412
Target Tolerance: 0.05
```

The fairness gap is guaranteed to be ≤ 0.05! ✓

---

## Complete Code

```python
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from fair_model import FairModel
from fair_trainer import FairTrainer
from utils import create_stratified_dataloaders

# 1. Prepare data
X = np.random.randn(10000, 20).astype(np.float32)
X[:, 0] = np.random.binomial(1, 0.3, 10000)
y = (X[:, 1] + X[:, 2] + 0.5 * X[:, 0] > 0).astype(np.float32)

print(X[:, 0])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Create dataloaders
train_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_test=X_test, y_test=y_test,
    protected_attr_idx=0,
    batch_size_train=500,
    batch_size_eval=500
)

# 3. Create model
model = FairModel(
    input_dim=20,
    b_tau = 500,
    hidden_dims=[64, 32],
    output_dim=1,
    protected_attr_idx=0,
    fairness_tolerance=0.05,
    fairness_metric='mean_pred',
    prediction_bounds= (-100.0, 100.0) #bounds are for logits, not predicted pred
)

# 4. Train
trainer = FairTrainer(
    model,
    nn.BCEWithLogitsLoss(),
    optim.Adam(model.parameters(), lr=0.001),
    early_stopping_patience=15
)

history = trainer.fit(train_loader, test_loader, epochs=20, log_interval=1)

# 5. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Fairness Gap: {metrics['fairness_gap']:.4f} (target: <= 0.05)")
```

---

## What's Next?

- **[Core Concepts](../user-guide/concepts.md)**: Understand the theory behind fairness_training
- **[Fairness Metrics](../user-guide/fairness-metrics.md)**: Learn about different fairness criteria
- **[Small-Batch Inference](../examples/small-batch.md)**: Deploy with real-time/streaming predictions
- **[Custom Metrics](../user-guide/custom-metrics.md)**: Define your own fairness constraints
