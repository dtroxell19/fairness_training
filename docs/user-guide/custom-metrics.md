# Custom Fairness Metrics

`fairness_training` supports custom fairness metrics through the `FairnessMetric` base class. This lets you encode any affine fairness constraint — group-conditional means, quantiles approximated linearly, industry-specific fairness criteria, and more.

---

## Requirements

Your custom metric must:

1. **Express constraints as affine functions** of the predictions `yhat` and (optionally) targets `y`
2. **Be DPP-compliant** for cvxpylayers (see rules below)
3. **Subclass `FairnessMetric`** and implement the four required methods

---

## DPP (Disciplined Parameterized Programming) Rules

cvxpylayers requires that constraints are **affine in variables and parameters separately** — no variable × parameter products.

| Role | cvxpy type | Notes |
|------|-----------|-------|
| `yhat` | `cp.Variable` | Predictions being optimized |
| `y` | `cp.Parameter` | Ground-truth targets (if needed) |
| Selection matrices | `np.ndarray` (constant) | Group membership indicators — must be numpy, not torch |
| `slack` | `cp.Parameter` | Tolerance ε |

**Allowed:**
```python
cp.multiply(yhat, selector)             # Variable × constant ✓
cp.sum(cp.multiply(y, selector)) / n    # Parameter × constant / constant ✓
mean_group_0 - mean_group_1 <= slack    # Expression <= Parameter ✓
```

**Not allowed:**
```python
cp.multiply(yhat, y)       # Variable × Parameter ✗
cp.multiply(slack, gap)    # Parameter × Variable ✗
```

---

## Complete Example: Weighted Mean Parity

Here is a complete custom metric that enforces weighted mean prediction parity. Use it as a template.

```python
import numpy as np
import cvxpy as cp
import torch
from typing import List, Optional, Tuple

from fairness_training.fairness_metrics import FairnessMetric


class WeightedMeanParity(FairnessMetric):
    """
    Weighted mean prediction parity:
        |w0 * E[ŷ|A=0] - w1 * E[ŷ|A=1]| ≤ ε
    
    Useful when you want to up-weight the minority group's fairness signal.
    """

    requires_targets = False            # set True if y is needed
    requires_y_in_constraints = False   # set True if y appears in cvxpy constraints

    def __init__(self, num_protected_attrs: int = 1, weights: Tuple[float, float] = (1.0, 1.0)):
        super().__init__(num_protected_attrs=num_protected_attrs)
        self.w0, self.w1 = weights

    # ------------------------------------------------------------------
    # 1. Selection matrices
    # ------------------------------------------------------------------
    def create_selection_matrices(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        protected_attr_idx: List[int],
    ) -> List[np.ndarray]:
        """Return one [n]-shaped 0/1 array per group per attribute."""
        matrices = []
        for attr_idx in protected_attr_idx:
            a = x[:, attr_idx].cpu().numpy()
            matrices.append((a == 0).astype(np.float32))   # group 0 indicator
            matrices.append((a == 1).astype(np.float32))   # group 1 indicator
        return matrices

    # ------------------------------------------------------------------
    # 2. cvxpy constraints (must be DPP-compliant)
    # ------------------------------------------------------------------
    def create_constraints(
        self,
        yhat: cp.Variable,
        y: Optional[cp.Parameter],
        selection_matrices: List[np.ndarray],
        slack: cp.Parameter,
        attr_idx: int,
    ) -> List[cp.Constraint]:
        """Build constraints for attribute `attr_idx`."""
        base = 2 * attr_idx           # each attribute occupies 2 slots in the list
        sel0 = selection_matrices[base]       # group 0 indicator, shape (n,)
        sel1 = selection_matrices[base + 1]   # group 1 indicator, shape (n,)

        n0 = max(sel0.sum(), 1)
        n1 = max(sel1.sum(), 1)

        # Mean predictions per group — affine in yhat
        mean0 = cp.sum(cp.multiply(yhat, sel0)) / n0
        mean1 = cp.sum(cp.multiply(yhat, sel1)) / n1

        # Weighted gap
        gap = self.w0 * mean0 - self.w1 * mean1

        return [gap <= slack, -gap <= slack]

    # ------------------------------------------------------------------
    # 3. Monitoring gap (numpy, not used in optimisation)
    # ------------------------------------------------------------------
    def compute_gap(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray],
        x: np.ndarray,
        protected_attr_idx: List[int],
    ) -> float:
        gaps = []
        for attr_idx in protected_attr_idx:
            a = x[:, attr_idx]
            group0 = predictions[a == 0]
            group1 = predictions[a == 1]
            if len(group0) == 0 or len(group1) == 0:
                continue
            gaps.append(abs(self.w0 * group0.mean() - self.w1 * group1.mean()))
        return max(gaps) if gaps else 0.0

    # ------------------------------------------------------------------
    # 4. Primal-dual penalty (used for small-batch inference)
    # ------------------------------------------------------------------
    def create_primal_dual_penalty(
        self,
        yhat: torch.Tensor,
        y: Optional[torch.Tensor],
        x: torch.Tensor,
        lambda_dual: torch.Tensor,
        protected_attr_idx: List[int],
    ) -> torch.Tensor:
        total = torch.tensor(0.0)
        for i, attr_idx in enumerate(protected_attr_idx):
            a = x[:, attr_idx]
            g0 = yhat[a == 0]
            g1 = yhat[a == 1]
            if len(g0) == 0 or len(g1) == 0:
                continue
            gap = self.w0 * g0.mean() - self.w1 * g1.mean()
            total = total + lambda_dual[i] * gap
        return total
```

### Using the custom metric

```python
from fairness_training import FairModel, FairTrainer, validate_metric

metric = WeightedMeanParity(num_protected_attrs=1, weights=(1.0, 2.0))

# Validate DPP compliance before training (raises FairnessMetricError if broken)
validate_metric(metric, input_dim=20, batch_size=64)

model = FairModel(
    input_dim=20,
    hidden_dims=[64, 32],
    protected_attr_idx=0,
    fairness_tolerance=0.05,
    fairness_metric=metric,          # pass instance directly
    prediction_bounds=(0.0, 1.0),
)
```

---

## Validate Before Training

`validate_metric` runs a dry pass through the metric's `create_constraints()` to catch DPP violations immediately, before any real training starts:

```python
from fairness_training import validate_metric

validate_metric(metric, input_dim=20, batch_size=64)
# Passes silently if valid.
# Raises FairnessMetricError with the offending constraint index if not.
```

---

## Troubleshooting

### "Problem is not DPP"

Your constraints violate DPP rules. Common causes:

- Multiplying two `cp.Parameter` objects (e.g. `slack * y_param`)
- Multiplying a `cp.Variable` by another `cp.Variable`
- Selection matrices stored as torch tensors instead of numpy arrays

### "Problem is infeasible"

The constraints can't be satisfied for this batch. Common causes:

- Tolerance `ε` is too tight relative to the natural group gap
- A group has zero members in the batch — use stratified dataloaders to prevent this

### Gradients are NaN

- Check for division by zero when computing group means (guard with `max(n, 1)`)
- Ensure selection matrices sum to at least 1 before creating constraints

---

## Next Steps

- **[API Reference: FairnessMetric](../api/fairness-metrics.md)**: Full base class documentation
- **[Examples](../examples/large-batch.md)**: See built-in metrics in action
