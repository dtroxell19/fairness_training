# FairModel

`FairModel` is the core class that implements a neural network with differentiable fairness constraints. It wraps a standard feedforward network with a cvxpylayers optimization layer that projects predictions onto the fairness constraint set.

---

::: fairness_training.FairModel
    options:
      show_root_heading: true
      show_source: false

---

## See Also

- [FairTrainer](fair-trainer.md) - Training utilities
- [FairnessMetric](fairness-metrics.md) - Fairness metric base class
- [Core Concepts](../user-guide/concepts.md) - Theory and background
