# Visualization

`fairness_training` includes a lightweight plotting module, `fairness_training.viz`, with three functions for the most common fairness paper figures. All functions require **matplotlib**, which is an optional dependency.

---

## Setup

```bash
# Install with visualization support
pip install -e .[train,verify,viz]

# Or add matplotlib to an existing environment
pip install matplotlib
```

```python
from fairness_training.viz import (
    plot_training_history,
    plot_group_distributions,
    plot_fairness_tradeoff,
)
```

---

## `plot_training_history`

Plot loss and fairness gap curves from the dict returned by `trainer.fit()`.

```python
history = trainer.fit(train_loader, val_loader=val_loader, epochs=50)

fig = plot_training_history(
    history,
    title="FairModel — mean_pred metric, ε=0.05",
)
```

**What it shows:** Two side-by-side panels — train/val loss on the left, train/val fairness gap on the right. The fairness gap panel makes it easy to see whether the model is staying within ε throughout training.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `history` | required | Dict from `trainer.fit()` |
| `title` | `"Training History"` | Figure title |
| `figsize` | `(12, 4)` | `(width, height)` in inches |
| `save_path` | `None` | Save to file instead of displaying |

**Expected keys in `history`:** `train_loss`, `val_loss` (optional), `train_fairness_gap`, `val_fairness_gap` (optional).

---

## `plot_group_distributions`

Histogram of model predictions for each protected group. Useful for visually confirming that the fairness layer is working — the two group distributions should have similar means.

```python
# Pass return_predictions=True to get predictions and group indicators
metrics = trainer.evaluate(test_loader, return_predictions=True)

fig = plot_group_distributions(
    metrics['predictions'],
    metrics['protected'],
    attr_names={0: 'Gender', 1: 'Race'},   # optional human-readable names
    title="Test-set prediction distributions after fairness training",
)
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `predictions` | required | Shape `(n,)` or `(n, 1)` array of model outputs |
| `protected_indicators` | required | Dict `{attr_idx: binary 0/1 array}` from `metrics['protected']` |
| `attr_names` | `None` | Dict `{attr_idx: "Name"}` for axis labels |
| `title` | `"Prediction Distributions by Group"` | Figure title |
| `bins` | `30` | Number of histogram bins |
| `figsize` | auto | `(6 * n_attrs, 4)` by default |
| `save_path` | `None` | Save to file instead of displaying |

---

## `plot_fairness_tradeoff`

Scatter plot of loss vs fairness gap across multiple model configurations. This is the standard "fairness–accuracy tradeoff" figure from fairness papers.

```python
import torch.nn as nn
import torch.optim as optim

epsilons = [0.01, 0.05, 0.10, 0.20]
results = {}

for eps in epsilons:
    backbone = nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 32),        nn.ReLU(),
        nn.Linear(32, 1),         nn.Sigmoid(),
    )
    m = FairModel.wrap(backbone, protected_attr_idx=0, fairness_tolerance=eps)
    t = FairTrainer(m, criterion, optim.Adam(m.parameters()))
    t.fit(train_loader, epochs=30, verbose=False)
    results[f"ε={eps}"] = t.evaluate(test_loader)

fig = plot_fairness_tradeoff(
    results,
    loss_key='test_loss',
    gap_key='weighted_avg_fairness_gap',
    title='Fairness–Accuracy Tradeoff (mean_pred)',
)
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `results` | required | Dict `{label: metrics_dict}` |
| `loss_key` | `"test_loss"` | Key to use as x-axis |
| `gap_key` | `"weighted_avg_fairness_gap"` | Key to use as y-axis |
| `title` | `"Fairness–Accuracy Tradeoff"` | Figure title |
| `figsize` | `(7, 5)` | `(width, height)` in inches |
| `save_path` | `None` | Save to file instead of displaying |

!!! tip "Which gap metric to use on the y-axis"
    Use `weighted_avg_fairness_gap` for the y-axis — it's the quantity the theorem bounds, and it's what the model is actually optimising against. `pooled_fairness_gap` may look better or worse depending on batch group ratios, and doesn't directly correspond to the training objective.

---

## Saving Figures

All three functions accept a `save_path` argument:

```python
fig = plot_training_history(history, save_path="training_history.pdf")
fig = plot_group_distributions(preds, protected, save_path="group_dist.png")
fig = plot_fairness_tradeoff(results, save_path="tradeoff.pdf")
```

Supported formats: anything matplotlib supports — `.png`, `.pdf`, `.svg`, `.eps`.

---

## Quickstart Notebook

The [Quickstart Colab notebook](https://colab.research.google.com/github/dtroxell19/fairness_training/blob/main/notebooks/quickstart.ipynb) shows all three functions with live output on a synthetic dataset.
