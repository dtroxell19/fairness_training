## Requirements

- **Python 3.9 or newer**
- `pip` (or another PEP 517–compatible installer)

---

## Install via pip (recommended)

### Lightweight base installation
Installs core utilities and shared infrastructure.
This is sufficient for configuration, data handling, and inspection tools.

```bash
pip install fairness_training
```

---

## Optional Feature Sets

fairness_training uses **optional dependency groups** to keep installations lightweight and explicit.
Install only what you need.

### Training support
Includes PyTorch, NumPy, SciPy, and Transformers.
```bash
pip install fairness_training[train]
```

### Verification / optimization support
Includes CVXPY and associated solvers for convex verification.
```bash
pip install fairness_training[verify]
```

### Full installation
Includes all supported functionality: training, verification, datasets, vision tools, logging, and utilities.
```bash
pip install fairness_training[full]
```

### CPU-only environment
Explicitly targets CPU-only workflows.
```bash
pip install fairness_training[cpu]
```

### Development tools
Formatting, dependency inspection, and packaging utilities.
```bash
pip install fairness_training[dev]
```

---

## Install from Source

For the latest development version:
```bash
git clone https://github.com/fairness_training/fairness_training.git
cd fairness_training
pip install -e .
```
You may combine this with optional dependencies, for example:
```bash
pip install -e .[full]
```

---

## Dependency Management Philosophy

fairness_training follows these principles:

- **Exact version pinning** for major numerical and ML libraries
- **Optional dependency groups** for modular installs
- **No implicit heavy dependencies** in the base install

All dependencies and extras are defined in `project.toml`, which is the single source of truth for installation behavior

---

## Verify Installation
```python
import fairness_training
print(fairness_training.__version__)
```

Expected output: 0.1.0

If this succeeds, the installation is complete.

---

## Troubleshooting

### Solver issues (verification)

Some optimization problems may require additional solvers.
You can install them manually if needed:
```bash
pip install ecos
```

Other solvers supported by CVXPY may also be used.

---

## Next Steps

Once installed, head to the **Quickstart** to train or verify your first fairness-aware model