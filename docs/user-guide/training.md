# Training Models

This guide covers best practices for training neural networks with a fairness layer via the fairness_training package.

!!! important "Critical Requirement"
    This package assumes stratified sampling can be used in your training dataset. This ensures each batch has the same proportion of each group, which implies that enforcing constraints per-batch automatically leads to aggregate fairness (i.e. fairness satisfied when considering all training examples at once).

---

## Recommended Training Setup

### 1. Use Stratified Data Loaders

Stratified batching maintains consistent group proportions across batches:

```python
from fairness_training import create_stratified_dataloaders

train_loader, val_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_val, y_val,
    protected_attr_idx=[0, 1],  # Can be int or list
    batch_size_train=1000,
    batch_size_eval=1000
)
```

**What stratified batching does**:

- Ensures each batch has samples from all groups
- Maintains the same group proportions as the full dataset
- Drops samples that don't fit into complete batches (with a warning)

### 2. Configure the Model

```python
from fairness_training import FairModel

model = FairModel(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    output_dim=1,
    protected_attr_idx=[0, 1],
    prediction_bounds=(-100.0, 100.0),
    fairness_tolerance=0.05,
    b_tau=500, #if batch_size_eval is larger than b_tau, then constraints are also enforced per-batch in inference
    fairness_metric='mean_pred'
)
```

**Key parameters**:

| Parameter | Recommendation | Notes |
|-----------|---------------|-------|
| `hidden_dims` | [128, 64, 32] | Adjust to your problem complexity |
| `prediction_bounds` | (-1000, 1000) for logits | Prevents unbounded predictions and aids stability|
| `fairness_tolerance` | 0.05 | Start here, adjust as needed |
| `b_tau` | 1000 | If batch_size_eval is larger than b_tau, then constraints are also enforced per-batch in inference |

### 3. Set Up Training

```python
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fairness_training import FairTrainer

criterion = nn.BCEWithLogitsLoss()  # or nn.MSELoss() for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

trainer = FairTrainer(
    model, 
    criterion, 
    optimizer,
    scheduler=scheduler,
    early_stopping_patience=15,
    early_stopping_delta=1e-5
)
```

### 4. Train

```python
history = trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
    verbose=1,
    log_interval=10
)
```

---

## Handling Training Issues

### Solver Failures

If the cvxpy solver fails, you'll see debug output like:

```
================================================================================
CVXPY SOLVER FAILED
================================================================================
Error: ...
Batch size: 2000
Fairness tolerance: 0.05
...
```

**Common causes and solutions**:

| Issue | Potential Solution |
|-------|----------|
| Empty groups in batch | Use stratified dataloaders |
| Unstable/Inaccurate Solution | Reduce the batch size |
| Tolerance too tight | Increase `fairness_tolerance` |
| Numerical issues | Try `pip install ecos` for a different solver |

### Custom Logging

For TensorBoard or W&B integration, wrap the training loop:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Manual training loop
for epoch in range(epochs):
    train_loss, train_gap = trainer._train_epoch(train_loader)
    val_loss, val_gap = trainer._validate(val_loader)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Fairness/train', train_gap, epoch)
    writer.add_scalar('Fairness/val', val_gap, epoch)
```

---

## Checkpointing

Save and load model checkpoints:

```python
# Save
trainer.save_checkpoint('model_checkpoint.pt', include_history=True)

# Load
trainer.load_checkpoint('model_checkpoint.pt')
```

---

## Multiple Protected Attributes

Training with 2 protected attributes:

```python
model = FairModel(
    ...,
    protected_attr_idx=[0, 1], #2 attributes defined the groups
    fairness_tolerance=0.05
)

# Uses stratified batching for BOTH attributes
train_loader, val_loader = create_stratified_dataloaders(
    X_train, y_train, X_val, y_val,
    protected_attr_idx=[0, 1],  # Maintains 4 intersectional group proportions
    batch_size_train=1000
)
```

!!! note "Marginal Fairness"
    Constraints are enforced independently per attribute, not intersectionally. In other words, a model can ensure:
    
    - |E[Ŷ|attribute1=0] - E[Ŷ|attribute1=1]| ≤ ε
    - |E[Ŷ|attribute2=0] - E[Ŷ|attribute2=1]| ≤ ε
    
    But does NOT directly constrain |E[Ŷ|attribute1=0,attribute2=0] - E[Ŷ|attribute1=1,attribute2=1]|

---

## Hyperparameter Recommendations

Hyperparameter values largely depend on your dataset and fairness setting. The following table merely serves as a recommendation to get started:

| Parameter | Value |
|-----------|---------------|
| Learning rate | 0.001 |
| Batch size | 1000+ |
| fairness_tolerance | 0.05 |
| prediction_bounds | (-1000, 1000) |
| Early stopping patience | 15 |

---

## Next Steps

- **[Inference](inference.md)**: Deploy your trained model
- **[Examples](../examples/large-batch.md)**: Complete working examples
