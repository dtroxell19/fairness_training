# Fairness Metrics

The fairness_training package provides several built-in fairness metrics and supports custom metrics. This guide explains each metric and when to use it.

---

## Built-in Metrics

### Mean Prediction Parity (Demographic Parity)

\(  \lvert E[\hat{y} \mid x_j=0] - E[\hat{y} \mid x_j=1]  \rvert \leq \epsilon \)

Ensures average predictions are similar across groups.

**Example**:
```python
from fairness_training import FairModel

model = FairModel(
    input_dim=20,
    hidden_dims=[64, 32],
    protected_attr_idx=0,
    fairness_metric='mean_pred',  # or 'demographic_parity'
    fairness_tolerance=0.05
)
```

!!! note "Classification vs Regression"
    For binary classification, this applies to the predicted probabilities (pre-sigmoid or post-sigmoid depending on your setup), not hard 0/1 predictions.

### Mean Residual Fairness

\( \lvert E[y - \hat{y} \mid x_j=a] \rvert \leq \epsilon \ \forall a \)

Ensures the model doesn't systematically over/under-predict for any group compared to another

**Example**:
```python
model = FairModel(
    input_dim=20,
    hidden_dims=[64, 32],
    protected_attr_idx=0,
    fairness_metric='mean_residual',  # or 'equalized_residuals'
    fairness_tolerance=0.1
)
```

!!! warning "Requires Targets"
    This metric requires ground truth labels `y` during both training and inference. Therefore, instead of predictive modeling, this fairness metric is used in "model auditing" or "data modeling" tasks.
    
    ```python
    # Must pass y during forward
    predictions = model(X, y=y, inference=True)
    ```

### Equalized Odds

\( \lvert E[\hat{y} \mid x_j=0, y=a] - E[\hat{y} \mid x_j=1, y = a] \rvert \leq \epsilon\ \forall a \in \{0,1\} \)

Ensures predictions are similar across groups, conditioned on the true outcome

**Example**:
```python
from fairness_training import FairModel, EqualizedOdds

# Using string shorthand
model = FairModel(
    input_dim=20,
    hidden_dims=[64, 32],
    protected_attr_idx=0,
    fairness_metric='equalized_odds',
    fairness_tolerance=0.05
)

# Or using the class directly
metric = EqualizedOdds(num_protected_attrs=1)
model = FairModel(
    ...,
    fairness_metric=metric
)
```

!!! warning "Requires Targets"
    Like mean residual fairness, equalized odds requires ground truth labels.

---

## Using Multiple Metrics

Currently, fairness_training supports one metric at a time. If you need multiple fairness criteria, you can:

1. **Use the most restrictive metric** for your use case
2. **Create a custom metric** that combines constraints (see [Custom Metrics](custom-metrics.md))

---

## Metric-Specific Parameters

### fairness_tolerance (epsilon)

The maximum allowed fairness gap. Smaller = stricter fairness. This number is chosen by domain experts.

```python
# Strict fairness
model = FairModel(..., fairness_tolerance=0.01)

# More permissive
model = FairModel(..., fairness_tolerance=0.1)
```

### num_protected_attrs

Specify how many protected attributes are in the dataset. The fairness_training package currently supports 1 or 2 protected attributes.

```python
from fairness_training import MeanPredictionParity

# Single attribute
metric = MeanPredictionParity(num_protected_attrs=1)

# Two attributes (marginal fairness)
metric = MeanPredictionParity(num_protected_attrs=2)
```

---

## Next Steps

- **[Custom Metrics](custom-metrics.md)**: Create your own fairness constraints
- **[Training Models](training.md)**: Best practices for training with these metrics
