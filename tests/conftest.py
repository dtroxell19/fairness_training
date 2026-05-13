"""Shared fixtures for fairness_training tests."""

import sys
import os
import numpy as np
import pytest
import torch
import torch.nn as nn

# Allow the package to be imported from the source tree without installing it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fairness_training import FairModel
from fairness_training.fairness_metrics import MeanPredictionParity, MeanResidualFairness, EqualizedOdds
from fairness_training.utils import create_dataloaders, create_stratified_dataloaders


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """200-sample binary classification dataset with 2 protected attributes."""
    np.random.seed(42)
    n = 200
    # Column 0: protected attr A (binary), Column 1: protected attr B (binary)
    A = np.random.binomial(1, 0.4, n).astype(np.float32)
    B = np.random.binomial(1, 0.6, n).astype(np.float32)
    # 8 other features
    X_other = np.random.randn(n, 8).astype(np.float32)
    X = np.concatenate([A[:, None], B[:, None], X_other], axis=1)
    y = (A + 0.5 * B + X_other[:, 0] > 0.5).astype(np.float32)
    return X, y


@pytest.fixture
def regression_data():
    """200-sample regression dataset with 1 protected attribute."""
    np.random.seed(0)
    n = 200
    A = np.random.binomial(1, 0.4, n).astype(np.float32)
    X_other = np.random.randn(n, 9).astype(np.float32)
    X = np.concatenate([A[:, None], X_other], axis=1)
    y = (A * 2 + X_other[:, 0]).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Minimal FairModel instances (small b_tau so tests run on tiny batches)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_model():
    """FairModel with b_tau=50 so a 100-sample batch uses hard constraints."""
    return FairModel(
        input_dim=10,
        hidden_dims=[16, 8],
        protected_attr_idx=0,
        fairness_tolerance=0.05,
        b_tau=50,
    )


@pytest.fixture
def dual_attr_model():
    """FairModel with 2 protected attributes."""
    return FairModel(
        input_dim=10,
        hidden_dims=[16, 8],
        protected_attr_idx=[0, 1],
        fairness_tolerance=0.05,
        b_tau=50,
    )


@pytest.fixture
def residual_model():
    """FairModel using MeanResidualFairness."""
    return FairModel(
        input_dim=10,
        hidden_dims=[16, 8],
        protected_attr_idx=0,
        fairness_tolerance=0.10,
        b_tau=50,
        fairness_metric="mean_residual",
    )
