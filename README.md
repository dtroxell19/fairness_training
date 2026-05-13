# fairness_training

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dtroxell19/fairness_training/blob/main/notebooks/quickstart.ipynb)

**Guaranteed fairness constraints for PyTorch neural networks — in three lines of code.**

```python
from fairness_training import FairModel, FairTrainer, create_stratified_dataloaders

model = FairModel.wrap(my_network, protected_attr_idx=0, fairness_tolerance=0.05)
trainer = FairTrainer(model, criterion, optimizer)
trainer.fit(train_loader, val_loader, epochs=50)
```

---

## Why fairness_training?

Most fairness methods either *encourage* fairness (soft penalties) or correct predictions *after* training (post-hoc). Both leave the door open for violations.

`fairness_training` uses a **differentiable convex optimization layer** appended to any neural network. This layer projects predictions onto the set satisfying your fairness constraints — making violations mathematically impossible on every batch.

| Approach | Guarantee | Differentiable | Works with any architecture |
|----------|-----------|----------------|-----------------------------|
| Penalty / regularization | No | Yes | Yes |
| Post-hoc calibration | No | No | Yes |
| **fairness_training** | **Yes** | **Yes** | **Yes** |

---

## Features

- **Hard constraints** — fairness gaps are bounded by your chosen ε on every training batch
- **Drop-in wrapper** — `FairModel.wrap(your_model, ...)` works with any `nn.Module`
- **Online inference** — primal-dual algorithm provides aggregate guarantees for streaming / small-batch deployment
- **Three built-in metrics** — demographic parity, mean residual fairness, equalized odds
- **Extensible** — subclass `FairnessMetric` to define custom affine fairness constraints

---

## Installation

```bash
# Standard install (training + fairness layer)
pip install fairness_training[train,verify]

# Full install (adds datasets, vision, visualization, TensorBoard)
pip install fairness_training[full]

# Visualization only (for plotting utilities)
pip install fairness_training[viz]
```

---

## Quick Example

```python
import torch.nn as nn
import torch.optim as optim
from fairness_training import FairModel, FairTrainer, create_stratified_dataloaders

# Any standard PyTorch backbone
backbone = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1), nn.Sigmoid(),
)

# Wrap with fairness constraints — bounds inferred automatically
model = FairModel.wrap(
    backbone,
    protected_attr_idx=0,       # column index of protected attribute in X
    fairness_tolerance=0.05,    # ε: max allowed group mean prediction gap
    fairness_metric='mean_pred',
)

# Stratified batching keeps group ratios constant → per-batch constraints = aggregate fairness
train_loader, val_loader, test_loader = create_stratified_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    protected_attr_idx=0, batch_size_train=256,
)

trainer = FairTrainer(model, nn.BCELoss(), optim.Adam(model.parameters()))
history = trainer.fit(train_loader, val_loader, epochs=50)

metrics = trainer.evaluate(test_loader)
print(f"Test loss:    {metrics['test_loss']:.4f}")
print(f"Fairness gap: {metrics['weighted_avg_fairness_gap']:.4f}  (target ≤ 0.05)")
```

---

## Supported Fairness Metrics

| Metric | Constraint | String alias |
|--------|-----------|--------------|
| Mean Prediction Parity | `|E[ŷ|A=0] − E[ŷ|A=1]| ≤ ε` | `'mean_pred'` |
| Mean Residual Fairness | `|E[y−ŷ|A=a]| ≤ ε ∀a` | `'mean_residual'` |
| Equalized Odds | `|E[ŷ|A=0,y=c] − E[ŷ|A=1,y=c]| ≤ ε ∀c` | `'equalized_odds'` |
| Custom | Subclass `FairnessMetric` | — |

---

## Documentation

Full documentation — concepts, API reference, and end-to-end examples — is available at the project's GitHub Pages site.

---

## Citation

If you use `fairness_training` in your research, please cite:

```bibtex
@inproceedings{author2025fairness,
  title={Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning},
  author={Anonymous},
  year={2025}
}
```

This library builds on the excellent [cvxpylayers](https://locuslab.github.io/2019-10-28-cvxpylayers/) package:

```bibtex
@inproceedings{agrawal2019differentiable,
  title={Differentiable Convex Optimization Layers},
  author={Agrawal, Akshay and Amos, Brandon and Barratt, Shane and Boyd, Stephen and Diamond, Steven and Kolter, Zico},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

---

## License

MIT
