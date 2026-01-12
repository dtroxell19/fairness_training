# Custom Fairness Metrics

The fairness_training package supports custom fairness metrics through the `FairnessMetric` base class. This guide shows how to create your own.

---

## Requirements for Custom Metrics

Your custom metric must:

1. **Be affine** in predictions and/or targets
2. **Be DPP-compliant** for cvxpylayers (see rules below)
3. **Subclass `FairnessMetric`** and implement required methods

### DPP (Disciplined Parameterized Programming) Rules

For cvxpylayers to work, constraints must follow these rules:

- `yhat` is a `cp.Variable` (the predictions being optimized)
- `y` is a `cp.Parameter` (targets, if needed)
- Selection matrices must be **constant numpy arrays** (0/1 values)
- `slack` is a `cp.Parameter` (the tolerance ε)
- Products are only allowed when one operand is constant

**Allowed**:
```python
cp.multiply(yhat, selector)  # Variable × constant
cp.sum(cp.multiply(y, selector)) / n  # Parameter × constant / constant
mean_0 - mean_1 <= slack  # Expression <= Parameter
```

**Not allowed**:
```python
cp.multiply(yhat, y)  # Variable × Parameter (both varying)
cp.multiply(slack, gap_var)  # Parameter × Variable
```

---

## Custom Selection Matrices

By default, selection matrices are structured as `[A=0, A=1]` for each attribute. Override `create_selection_matrices` for custom structures:

```python
def create_selection_matrices(
    self,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    protected_attr_idx: List[int]
) -> List[np.ndarray]:
    """
    Override for custom selector structure.
    
    Example: Create selectors conditioned on outcome (like EqualizedOdds)
    """
    if y is None:
        raise ValueError("This metric requires targets")
    
    y_np = y.squeeze().cpu().numpy()
    selection_matrices = []
    
    for attr_idx in protected_attr_idx:
        protected_vals = x[:, attr_idx].cpu().numpy()
        
        # 4 selectors: (Y=0, A=0), (Y=0, A=1), (Y=1, A=0), (Y=1, A=1)
        for y_val in [0, 1]:
            for a_val in [0, 1]:
                selector = ((y_np == y_val) & (protected_vals == a_val)).astype(np.float32)
                selection_matrices.append(selector)
    
    return selection_matrices
```

---

## Troubleshooting

### "Problem is not DPP"

Your constraints violate DPP rules. Check:
- Are you multiplying two parameters or two variables?
- Is your selection matrix a constant numpy array (not a torch tensor)?

### "Problem is infeasible"

Your constraints can't be satisfied for this batch. Check:
- Are bounds too tight?
- Is the batch missing samples from a group?

### Gradients are NaN

- Check for division by zero (when a group has no samples)
- Ensure selection matrices have proper sums

---

## Next Steps

- **[API Reference: FairnessMetric](../api/fairness-metrics.md)**: Full base class documentation
- **[Examples](../examples/large-batch.md)**: See built-in metrics in action
