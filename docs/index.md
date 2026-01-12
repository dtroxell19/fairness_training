# Fairness Layers in Neural Networks

**Guaranteed fairness constraints in deep learning through differentiable optimization layers**

[![PyPI version](https://badge.fury.io/py/fairness_training.svg)](https://badge.fury.io/py/fairness_training)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is the fairness_training package?

fairness_training is a Python package that enables training neural networks with **hard fairness constraints**. Unlike penalty-based methods that only encourage constraints, fairness_training guarantees that your model's predictions satisfy specified criteria. 

Through [cvxpylayers](https://locuslab.github.io/2019-10-28-cvxpylayers/), this package uses a **differentiable fairness layer**: a convex optimization layer appended to the neural network that projects predictions onto the feasible set defined by your constraints while remaining fully differentiable for end-to-end training.

---

## Key Features

<div class="grid cards" markdown>

-   **Verified Fairness**

    ---

    Hard constraints guarantee that the specified constraints are satisfied

-   **End-to-End Learning and Constraint-Aware**

    ---

    The fairness layer is fully differentiable, enabling the model to learn how to satisfy constraints during training as opposed to relying on post-hoc corrections after model training

-   **Flexible Architecture**

    ---

    Works with any classification or regression architecture. Just append the fairness layer to the end of your model

-   **Online Inference**

    ---

    Novel primal-dual algorithm provides aggregate fairness guarantees over time even with small batch sizes during real-time inference

</div>

---

## How It Works

```mermaid
flowchart LR
    A[Input X] --> B["Neural Network f(·)"]
    B --> C["Raw Predictions ẑ = f(X)"]
    C --> D["Fairness Layer g(·)"]
    D --> E["Fair Predictions ŷ = g(ẑ)"]
    
    style D fill:#e1f5fe
```

The fairness layer solves a convex optimization problem:

\[
g(z) = \arg\min_{\tilde{y}} \|\tilde{y} - z\|_2^2 \quad \text{s.t.} \quad \text{constraints satisfied}
\]

This projection is differentiable via implicit differentiation through the KKT conditions, enabling standard backpropagation.

---

## Supported Fairness Criteria

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Mean Prediction Parity** | \(  \lvert E[\hat{y} \mid x_j=0] - E[\hat{y} \mid x_j=1]  \rvert \leq \epsilon \) | Regression or Classification Scores |
| **Mean Residual Fairness** | \( \lvert E[y - \hat{y} \mid x_j=a] \rvert \leq \epsilon \ \forall a \) | Regression or Classification Scores |
| **Equalized Odds** | \( \lvert E[\hat{y} \mid x_j=0, y=a] - E[\hat{y} \mid x_j=1, y = a] \rvert \leq \epsilon\ \forall a \in \{0,1\} \)| Binary Classification |
| **Custom Metrics** | Extend `FairnessMetric` base class | Affine fairness constraints |

---

## Citation

If you use fairness_training in your research, please cite:

```bibtex
@inproceedings{author2025fairness,
  title={Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning},
  author={removed during review period for anonymity},
  year={2025}
}
```
This package relies heavily on the wonderful [cvxpylayers](https://locuslab.github.io/2019-10-28-cvxpylayers/) package. We encourage you to also cite their work:

```bibtex
@inproceedings{agrawal2019differentiable,
  title     = {Differentiable Convex Optimization Layers},
  author    = {Agrawal, Akshay and Amos, Brandon and Barratt, Shane and Boyd, Stephen and Diamond, Steven and Kolter, Zico},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019}
}
```

---

## Next Steps

<div class="grid cards" markdown>

-   [:material-rocket-launch: **Getting Started**](getting-started/installation.md)

    ---

    Install the fairness_training package

-   [:material-book-open-variant: **User Guide**](user-guide/concepts.md)

    ---

    Learn the core concepts, assumptions, and how to use fairness_training effectively

-   [:material-code-tags: **API Reference**](api/fair-model.md)

    ---

    Complete documentation of all classes and functions

-   [:material-flask: **Examples**](examples/large-batch.md)

    ---

    End-to-end examples on real datasets

</div>
