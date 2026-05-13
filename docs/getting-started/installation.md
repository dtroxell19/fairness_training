# Installation

## Requirements

- **Python 3.9 or newer**
- `pip` (or another PEP 517–compatible installer)

---

## Install via pip

### Recommended setup

For training and the differentiable fairness layer (the most common setup):

```bash
pip install fairness_training[train,verify]
```

### Other install options

| Extra | What it adds | When to use |
|-------|-------------|-------------|
| `train` | PyTorch, NumPy, SciPy, Transformers | Model training |
| `verify` | CVXPY + solvers (ECOS) | Fairness layer (required for `FairModel`) |
| `full` | Everything above + datasets, vision, TensorBoard | All features |
| `cpu` | CPU-only PyTorch | CPU-only environments |
| `viz` | Matplotlib | Plotting utilities |
| `dev` | pytest, black, pipdeptree | Development / testing |

```bash
# Full installation
pip install fairness_training[full]

# Visualization support only
pip install fairness_training[viz]

# Development tools
pip install fairness_training[dev]
```

---

## Install from Source

```bash
git clone https://github.com/dtroxell19/fairness_training.git
cd fairness_training
pip install -e .[train,verify]
```

---

## Verify Installation

```python
import fairness_training
print(fairness_training.__version__)  # 0.1.0
```

---

## Troubleshooting

### `AttributeError: module 'onnxscript.values' has no attribute 'ParamSchema'`

This is a conflict with `onnxscript`, which this library doesn't use. Remove it:

```bash
pip uninstall onnxscript -y
```

### Solver issues

The `verify` extra installs [ECOS](https://github.com/embotech/ecos), which handles most problems. If you hit numerical issues, try installing additional CVXPY-compatible solvers:

```bash
pip install clarabel   # recommended alternative
pip install scs        # another option
```

---

## Next Steps

Head to the **[Quickstart](quickstart.md)** to train your first fairness-aware model.
