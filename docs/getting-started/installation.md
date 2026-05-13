# Installation

## Requirements

- **Python 3.9 or newer**
- `pip` (or another PEP 517–compatible installer)

---

## Install via pip

```bash
pip install fairness_training[train,verify]
```

This installs the core package with PyTorch and the CVXPY fairness layer — the typical setup for training.

### Install options

| Extra | What it adds | When to use |
|-------|-------------|-------------|
| `train` | PyTorch, torchvision, NumPy, SciPy, Transformers | Model training |
| `verify` | CVXPY, cvxpylayers, ECOS, diffcp | Fairness layer — **required for `FairModel`** |
| `full` | Everything above + datasets, vision tools, TensorBoard | All features |
| `cpu` | CPU-only PyTorch + NumPy | CPU-only environments (pair with `verify`: `[cpu,verify]`) |
| `viz` | Matplotlib | Plotting utilities (`fairness_training.viz`) |
| `dev` | pytest, pytest-cov, black | Running the test suite |

```bash
# Training + fairness + visualization
pip install fairness_training[train,verify,viz]

# CPU-only (no CUDA) + fairness layer
pip install fairness_training[cpu,verify]

# Full installation
pip install fairness_training[full]

# Development
pip install fairness_training[dev]
```

---

## Install from Source

For the latest development version:

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

A conflict with `onnxscript`, which this library doesn't use:

```bash
pip uninstall onnxscript -y
```

### Solver issues

The `verify` extra installs [ECOS](https://github.com/embotech/ecos), which handles most problems. If you hit numerical issues, try:

```bash
pip install clarabel   # recommended alternative
pip install scs        # another option
```

---

## Next Steps

Head to the **[Quickstart](quickstart.md)** to train your first fairness-aware model.
