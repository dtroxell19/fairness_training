# Utilities API

Data loading and helper utilities for fairness_training.

---

## create_stratified_dataloaders

```python
def create_stratified_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    protected_attr_idx: Union[int, List[int]] = 0,
    batch_size_train: int = 2000,
    batch_size_eval: int = 2000
) -> Tuple[DataLoader, ...]
```

Create DataLoaders with stratified sampling to maintain constant group ratios.

**Required for training** if user desires strict fairness guarantees.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `np.ndarray` | required | Training features |
| `y_train` | `np.ndarray` | required | Training targets |
| `X_val` | `np.ndarray` | `None` | Validation features |
| `y_val` | `np.ndarray` | `None` | Validation targets |
| `X_test` | `np.ndarray` | `None` | Test features |
| `y_test` | `np.ndarray` | `None` | Test targets |
| `protected_attr_idx` | `int` or `List[int]` | `0` | Protected attribute column(s) |
| `batch_size_train` | `int` | `2000` | Training batch size |
| `batch_size_eval` | `int` | `2000` | Evaluation batch size |

### Returns

Tuple of stratified DataLoaders for each provided dataset.

### Stratification Behavior

**Single attribute:** Maintains 2-group proportions (A=0, A=1)

**Two attributes:** Maintains 4 intersectional group proportions (00, 01, 10, 11)

### Example

```python
from fairness_training import create_stratified_dataloaders

# Single protected attribute
train_loader, val_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_val, y_val,
    protected_attr_idx=0,
    batch_size_train=2000
)

# Two protected attributes
train_loader, val_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    protected_attr_idx=[0, 1],
    batch_size_train=2000,
    batch_size_eval=2000
)
```

### Output Example

```
=== Creating stratified training batches ===
  Original samples: 19600
  Per-batch allocation: group 0=1400, group 1=600
  Created 9 batches of size 2000
  Total samples used: 18000
Warning: Dropping 1600 samples (1200 from group 0, 400 from group 1) to ensure complete batches
```

### Notes

- Protected attributes must be binary (0/1 values)
- Samples that don't fit into complete batches are dropped with a warning
- Batches are pre-arranged (loader uses `shuffle=False`)
- Maximum 2 protected attributes supported

### Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: not binary` | Protected column has non-0/1 values | Binarize your protected attributes |
| `ValueError: must have samples in both groups` | One group is empty | Check your data has both groups |
| `ValueError: No complete batches` | Too few samples or batch too large | Reduce batch size or add data |

---

## get_fairness_metric

```python
def get_fairness_metric(
    metric: Union[str, FairnessMetric],
    num_protected_attrs: int = 1
) -> FairnessMetric
```

Factory function to get a fairness metric instance.

See [Fairness Metrics API](fairness-metrics.md#get_fairness_metric) for details.

---

## Data Preparation Tips

### Protected Attribute Format

Protected attributes must be:
- **Binary** (0 or 1 values)
- **In specific columns** of your feature matrix

```python
import numpy as np

# Ensure binary
gender = (df['gender'] == 'Male').astype(np.float32)
race = (df['race'] == 'White').astype(np.float32)

# Place in first columns
X = np.column_stack([gender, race, other_features])
# Now protected_attr_idx=[0, 1]
```

### Handling Imbalanced Groups

If one group is much smaller:

1. **Use stratified batching** (always recommended)
2. **Increase batch size** to ensure both groups present
3. **Check minimum samples** per group:

```python
for attr_idx in protected_attr_idx:
    n_0 = (X[:, attr_idx] == 0).sum()
    n_1 = (X[:, attr_idx] == 1).sum()
    print(f"Attribute {attr_idx}: group 0 = {n_0}, group 1 = {n_1}")
    
    # Minimum per batch
    ratio = n_1 / (n_0 + n_1)
    min_per_batch = max(int(batch_size * ratio), 1)
    print(f"  Min group 1 per batch of {batch_size}: {min_per_batch}")
```

### Train/Val/Test Split

Use stratified splitting to maintain group proportions:

```python
from sklearn.model_selection import train_test_split

# Stratify by both target and protected attribute
stratify_col = y.astype(str) + '_' + X[:, 0].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=stratify_col, random_state=42
)
```

## See Also

- [Training Guide](../user-guide/training.md) - Using dataloaders for training
- [FairTrainer](fair-trainer.md) - Training utilities
