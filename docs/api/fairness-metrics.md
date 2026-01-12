# Fairness Metrics API

This page documents the fairness metric classes and the base class for creating custom metrics.

---

## FairnessMetric (Base Class)

```python
class FairnessMetric(ABC)
```

Abstract base class for fairness metrics. Subclass this to create custom metrics.

### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `requires_targets` | `bool` | `False` | Whether metric needs ground truth labels |
| `requires_y_in_constraints` | `bool` | `False` | Whether y appears in cvxpy constraints |

### Constructor

```python
def __init__(self, num_protected_attrs: int = 1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_protected_attrs` | `int` | `1` | Number of protected attributes |

### Abstract Methods

#### create_constraints

```python
@abstractmethod
def create_constraints(
    self,
    yhat: cp.Variable,
    selection_matrices: List[np.ndarray],
    slack: cp.Parameter,
    y: Optional[cp.Parameter] = None
) -> List
```

Create cvxpy constraints for the fairness metric.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `yhat` | `cp.Variable` | Predictions to constrain, shape `(batch_size,)` |
| `selection_matrices` | `List[np.ndarray]` | Group selection matrices with 0/1 entries |
| `slack` | `cp.Parameter` | Tolerance parameter (ε) |
| `y` | `cp.Parameter` | Targets (if `requires_y_in_constraints=True`) |

**Returns:** List of cvxpy constraints

---

#### compute_gap

```python
@abstractmethod
def compute_gap(
    self,
    predictions: np.ndarray,
    targets: Optional[np.ndarray],
    protected_indicators: Dict[int, np.ndarray]
) -> float
```

Compute fairness gap for monitoring.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `np.ndarray` | Predictions, shape `(n,)` or `(n, 1)` |
| `targets` | `np.ndarray` | Targets (may be None) |
| `protected_indicators` | `Dict[int, np.ndarray]` | Map of attr_idx → indicator array |

**Returns:** Fairness gap value (float)

### Optional Methods

#### create_selection_matrices

```python
def create_selection_matrices(
    self,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    protected_attr_idx: List[int]
) -> List[np.ndarray]
```

Create selection matrices for constraints. Override for custom structures.

**Default behavior:** Creates 2 selectors per attribute `[A=0, A=1]`

---

#### create_primal_dual_penalty

```python
def create_primal_dual_penalty(
    self,
    yhat: cp.Variable,
    selection_matrices: List[np.ndarray],
    y: Optional[cp.Parameter] = None
) -> tuple
```

Create penalty for primal-dual optimization.

**Returns:** Tuple of `(max_gap_variable, constraints)`

---

## MeanPredictionParity

```python
class MeanPredictionParity(FairnessMetric)
```

**Also known as:** Demographic Parity (for continuous predictions)

**Constraint:**

\[
\left| E[\hat{Y} | A=0] - E[\hat{Y} | A=1] \right| \leq \epsilon
\]

### Attributes

| Attribute | Value |
|-----------|-------|
| `requires_targets` | `False` |
| `requires_y_in_constraints` | `False` |

### Usage

```python
from fairness_training import MeanPredictionParity, FairModel

# As string
model = FairModel(..., fairness_metric='mean_pred')

# As class
metric = MeanPredictionParity(num_protected_attrs=2)
model = FairModel(..., fairness_metric=metric)
```

---

## MeanResidualFairness

```python
class MeanResidualFairness(FairnessMetric)
```

**Also known as:** Equalized Residuals

**Constraint:**

\[
\left| E[Y - \hat{Y} | A=a] \right| \leq \epsilon \quad \text{for all groups } a
\]

### Attributes

| Attribute | Value |
|-----------|-------|
| `requires_targets` | `True` |
| `requires_y_in_constraints` | `True` |

### Usage

```python
from fairness_training import MeanResidualFairness, FairModel

# As string
model = FairModel(..., fairness_metric='mean_residual')

# As class
metric = MeanResidualFairness(num_protected_attrs=1)
model = FairModel(..., fairness_metric=metric)

# Must pass y during forward
predictions = model(X, y=y, inference=True)
```

---

## EqualizedOdds

```python
class EqualizedOdds(FairnessMetric)
```

**Constraint:**

\[
\left| E[\hat{Y} | Y=y, A=0] - E[\hat{Y} | Y=y, A=1] \right| \leq \epsilon \quad \text{for each outcome } y
\]

### Attributes

| Attribute | Value |
|-----------|-------|
| `requires_targets` | `True` |
| `requires_y_in_constraints` | `False` |

### Selection Matrix Structure

Creates 4 selectors per attribute: `[Y=0 & A=0, Y=0 & A=1, Y=1 & A=0, Y=1 & A=1]`

### Usage

```python
from fairness_training import EqualizedOdds, FairModel

# As string
model = FairModel(..., fairness_metric='equalized_odds')

# As class
metric = EqualizedOdds(num_protected_attrs=1)
model = FairModel(..., fairness_metric=metric)

# Must pass y during forward (for selection matrix computation)
predictions = model(X, y=y, inference=True)
```

---

## get_fairness_metric

```python
def get_fairness_metric(
    metric: Union[str, FairnessMetric],
    num_protected_attrs: int = 1
) -> FairnessMetric
```

Factory function to get a metric instance from string or return existing instance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | `str` or `FairnessMetric` | Metric name or instance |
| `num_protected_attrs` | `int` | Number of protected attributes |

**String options:**
- `'mean_pred'`, `'mean_prediction_parity'`, `'demographic_parity'`
- `'mean_residual'`, `'mean_residual_fairness'`, `'equalized_residuals'`
- `'equalized_odds'`

**Returns:** `FairnessMetric` instance

**Example:**

```python
from fairness_training import get_fairness_metric

metric = get_fairness_metric('mean_pred', num_protected_attrs=2)
# Returns MeanPredictionParity(num_protected_attrs=2)

existing_metric = EqualizedOdds(num_protected_attrs=1)
same_metric = get_fairness_metric(existing_metric)
# Returns the same instance
```

---

## Creating Custom Metrics

See the [Custom Metrics Guide](../user-guide/custom-metrics.md) for detailed instructions.

### Quick Template

```python
from fairness_training import FairnessMetric
import cvxpy as cp
import numpy as np

class MyMetric(FairnessMetric):
    requires_targets = False  # Set True if needed
    requires_y_in_constraints = False
    
    def create_constraints(self, yhat, selection_matrices, slack, y=None):
        constraints = []
        # Your constraint logic
        return constraints
    
    def compute_gap(self, predictions, targets, protected_indicators):
        # Calculate and return the gap
        return gap_value
    
    def create_primal_dual_penalty(self, yhat, selection_matrices, y=None):
        max_gap = cp.Variable(1, nonneg=True)
        constraints = []
        # Define max_gap constraints
        return max_gap, constraints
```

---

## See Also

- [Custom Metrics Guide](../user-guide/custom-metrics.md) - Detailed custom metric tutorial
- [Fairness Metrics Guide](../user-guide/fairness-metrics.md) - When to use each metric
- [FairModel](fair-model.md) - Using metrics with FairModel
